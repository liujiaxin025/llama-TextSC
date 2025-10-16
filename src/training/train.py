import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, Any, List
import os
import json
import sys

# Add the parent directory to the path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import Config
from src.models.mlp_codec import ChannelCodec
from src.channel.awgn import AWGNChannel
from src.core.semantic_condenser import SemanticCondenser
from src.core.semantic_vectorizer import SemanticVectorizer

class TextDataset(Dataset):
    # ✅ 接收预先初始化好的 condenser 和 vectorizer 作为参数
    def __init__(self, texts: List[str], config: Config, condenser: SemanticCondenser, vectorizer: SemanticVectorizer):
        self.texts = texts
        self.config = config

        # ✅ 直接使用传入的模型实例，而不是重新创建
        self.condenser = condenser
        self.vectorizer = vectorizer

        # Pre-process all texts to get semantic vectors
        print("Pre-processing dataset...")
        self.semantic_vectors = []
        for text in tqdm(texts, desc="Extracting semantic vectors"):
            semantic_json = self.condenser.condense(text)
            vector = self.vectorizer.vectorize(semantic_json)
            self.semantic_vectors.append(vector.squeeze(0))

        self.semantic_vectors = torch.stack(self.semantic_vectors)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.semantic_vectors[idx]

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.get('training.device')

        # Initialize model
        encoder_config = config.get('mlp.encoder')
        decoder_config = config.get('mlp.decoder')

        self.codec = ChannelCodec(encoder_config, decoder_config)
        self.codec.to(self.device)

        # Initialize channel
        self.channel = AWGNChannel(self.device)

        # ✅ 在这里一次性创建需要共享的语义模型
        print("Initializing shared semantic models...")
        self.condenser = SemanticCondenser(
            config.get('model.llama3.model_name'),
            config.get('model.llama3.device'),
            config.get('model.llama3.torch_dtype')
        )
        self.vectorizer = SemanticVectorizer(
            config.get('model.sentence_bert.model_name'),
            config.get('model.sentence_bert.device')
        )
        print("✅ Shared semantic models initialized successfully!")

        # Training parameters
        self.learning_rate = config.get('training.learning_rate')
        self.batch_size = config.get('training.batch_size')
        self.epochs = config.get('training.epochs')
        self.snr_range = config.get('training.snr_range')
        self.gradient_accumulation_steps = config.get('training.gradient_accumulation_steps', 1)
        self.mixed_precision = config.get('training.mixed_precision', False)
        self.max_grad_norm = config.get('training.max_grad_norm', 1.0)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.codec.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Mixed precision scaler - GradScaler会自动从张量中判断设备
        self.scaler = GradScaler() if self.mixed_precision else None

        # Learning rate scheduler
        scheduler_type = config.get('training.scheduler.type', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        else:
            self.scheduler = None

        # Initialize wandb
        self.use_wandb = config.get('training.use_wandb', False)
        if self.use_wandb:
            wandb.init(project="llama-textsc")
            
    def create_dataset(self, texts: List[str]) -> DataLoader:
        # ✅ 将共享的模型实例传递给 TextDataset
        dataset = TextDataset(texts, self.config, self.condenser, self.vectorizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        total_loss = 0.0
        num_batches = 0

        self.codec.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch_idx, vectors in enumerate(pbar):
            vectors = vectors.to(self.device)

            # Random SNR for each batch
            snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])

            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.autocast('cuda'):
                    encoded = self.codec.encode(vectors)
                    noisy = self.channel.add_noise(encoded, snr_db)
                    decoded = self.codec.decode(noisy)
                    loss = self.criterion(decoded, vectors) / self.gradient_accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.codec.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                encoded = self.codec.encode(vectors)
                noisy = self.channel.add_noise(encoded, snr_db)
                decoded = self.codec.decode(noisy)
                loss = self.criterion(decoded, vectors) / self.gradient_accumulation_steps

                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.codec.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            effective_loss = loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({
                'Loss': f'{effective_loss:.4f}',
                'SNR': f'{snr_db:.1f}dB',
                'Avg Loss': f'{total_loss/max(1, num_batches):.4f}'
            })

            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': effective_loss,
                    'snr_db': snr_db,
                    'epoch': epoch,
                    'batch': batch_idx
                })

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss
    
    def validate(self, dataloader: DataLoader, snr_db: float = 10.0) -> Dict[str, float]:
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        
        self.codec.eval()
        
        with torch.no_grad():
            for vectors in dataloader:
                vectors = vectors.to(self.device)
                
                # Forward pass
                encoded = self.codec.encode(vectors)
                noisy = self.channel.add_noise(encoded, snr_db)
                decoded = self.codec.decode(noisy)
                
                # Calculate metrics
                loss = self.criterion(decoded, vectors)
                mse = self.channel.calculate_mse(vectors, decoded)
                
                total_loss += loss.item()
                total_mse += mse
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        
        return {
            'loss': avg_loss,
            'mse': avg_mse,
            'snr_db': snr_db
        }
    
    def train(self, train_texts: List[str], val_texts: List[str] = None):
        # Create datasets
        train_loader = self.create_dataset(train_texts)
        val_loader = self.create_dataset(val_texts) if val_texts else None
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, Val MSE = {val_metrics['mse']:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('best_model.pth')
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss
                }
                if val_loader:
                    log_dict.update(val_metrics)
                wandb.log(log_dict)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
    
    def save_model(self, filename: str):
        save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        torch.save({
            'encoder_state_dict': self.codec.encoder.state_dict(),
            'decoder_state_dict': self.codec.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.config
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.codec.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.codec.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath}")

def load_cnn_dailymail_subset(num_samples: int = 1000, split: str = "train") -> List[str]:
    """
    Load CNN/Daily Mail dataset subset for training
    Uses first 5000 samples to speed up download and processing
    """
    try:
        from datasets import load_dataset

        print(f"Loading CNN/Daily Mail {split} dataset (first 5000 samples)...")

        # Load only the first 5000 samples to speed up download
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"{split}[:5000]")

        texts = []

        for item in dataset:
            # Extract the main article (use first few sentences for manageable length)
            article = item['article']
            if isinstance(article, str):
                # Split into sentences and take first 3-4
                sentences = [s.strip() for s in article.split('.') if s.strip()]
                if len(sentences) >= 2:
                    article_text = '. '.join(sentences[:3]) + '.'
                    if 50 < len(article_text) < 500:  # Reasonable length
                        texts.append(article_text)

            # Also use highlights as separate training examples
            highlights = item['highlights']
            if isinstance(highlights, str) and 30 < len(highlights) < 300:
                texts.append(highlights)

            # Stop if we have enough samples
            if len(texts) >= num_samples:
                break

        print(f"Successfully loaded {len(texts)} text samples from CNN/Daily Mail")
        return texts[:num_samples]

    except ImportError:
        print("datasets library not found. Installing...")
        os.system("pip install datasets")
        return load_cnn_dailymail_subset(num_samples, split)

    except Exception as e:
        print(f"Error loading CNN/Daily Mail dataset: {e}")
        print("Using fallback news-style texts...")
        return load_news_fallback(num_samples)

def load_news_fallback(num_samples: int = 1000) -> List[str]:
    """Fallback dataset with diverse news-style texts"""
    news_texts = [
        # Technology News
        "Apple announced its latest iPhone with advanced AI capabilities and improved battery life.",
        "Google's quantum computer achieved a major breakthrough in computational power.",
        "Microsoft invested billions in artificial intelligence research and development.",
        "Tesla's new electric vehicle model promises extended range and autonomous driving features.",
        "Amazon expanded its cloud computing services with new machine learning tools.",

        # Science & Health
        "Scientists discovered a new species of deep-sea creatures with unique bioluminescent properties.",
        "Medical researchers developed a breakthrough treatment for Alzheimer's disease.",
        "Climate scientists warned about accelerating ice melt in polar regions.",
        "NASA's Mars rover discovered evidence of ancient riverbeds on the red planet.",
        "Pharmaceutical companies announced progress on universal flu vaccine development.",

        # Business & Economy
        "Stock markets reached record highs amid strong corporate earnings reports.",
        "Federal Reserve signaled potential interest rate changes in coming months.",
        "Global supply chains showed signs of recovery after recent disruptions.",
        "Cryptocurrency markets experienced volatility following regulatory announcements.",
        "Unemployment rates dropped to pre-pandemic levels in many countries.",

        # World News
        "International climate agreement reached at summit in Geneva.",
        "Peace talks resumed between conflicting nations in Middle East region.",
        "European Union proposed new regulations for artificial intelligence development.",
        "Asian markets reported strong economic growth in quarterly reports.",
        "Trade negotiations between major economies showed promising progress.",

        # Social & Culture
        "Social media platforms faced increased scrutiny over data privacy practices.",
        "Streaming services competed for original content in entertainment industry.",
        "Educational institutions adopted hybrid learning models permanently.",
        "Mental health awareness campaigns gained momentum among young adults.",
        "Cultural festivals returned to in-person celebrations after pandemic restrictions."
    ]

    # Generate variations to reach desired number
    expanded_texts = news_texts.copy()

    # Create variations with different details
    companies = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta", "Netflix"]
    topics = ["artificial intelligence", "cloud computing", "electric vehicles",
              "social media", "streaming services", "autonomous driving", "quantum computing"]

    while len(expanded_texts) < num_samples:
        company = np.random.choice(companies)
        topic = np.random.choice(topics)

        templates = [
            f"{company} announced new developments in {topic}.",
            f"Latest reports show {company} leading innovation in {topic}.",
            f"Industry experts praise {company}'s approach to {topic}.",
            f"{company} invested heavily in {topic} research this quarter.",
            f"Competitors struggle to match {company}'s {topic} capabilities."
        ]

        expanded_texts.append(np.random.choice(templates))

    np.random.shuffle(expanded_texts)
    return expanded_texts[:num_samples]

if __name__ == "__main__":
    config = Config()

    # Show training configuration
    print("=== Training Configuration ===")
    print(f"Model: {config.get('model.llama3.model_name')}")
    print(f"Device: {config.get('training.device')}")
    print(f"Mixed Precision: {config.get('training.mixed_precision')}")
    print(f"Epochs: {config.get('training.epochs')}")
    print(f"Batch Size: {config.get('training.batch_size')}")
    print(f"Gradient Accumulation Steps: {config.get('training.gradient_accumulation_steps')}")
    print(f"Effective Batch Size: {config.get('training.batch_size') * config.get('training.gradient_accumulation_steps', 1)}")
    print(f"Learning Rate: {config.get('training.learning_rate')}")
    print(f"SNR Range: {config.get('training.snr_range')}")
    print("=" * 30)

    # Load training data using CNN/Daily Mail dataset
    print("Loading training data...")
    train_texts = load_cnn_dailymail_subset(1000, "train")  # 1000 training samples
    val_texts = load_cnn_dailymail_subset(200, "validation")  # 200 validation samples

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Initialize trainer
    trainer = Trainer(config)

    # Train model
    print("Starting training...")
    trainer.train(train_texts, val_texts)

    print("Training completed! Model saved to checkpoints/best_model.pth")