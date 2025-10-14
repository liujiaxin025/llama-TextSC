import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, Any, List, Tuple
import os
import json

from ..utils.config import Config
from ..models.mlp_codec import ChannelCodec
from ..channel.awgn import AWGNChannel
from ..core.semantic_condenser import SemanticCondenser
from ..core.semantic_vectorizer import SemanticVectorizer

class TextDataset(Dataset):
    def __init__(self, texts: List[str], config: Config):
        self.texts = texts
        self.config = config
        
        # Initialize semantic components
        self.condenser = SemanticCondenser(
            config.get('model.llama3.model_name'),
            config.get('model.llama3.device'),
            config.get('model.llama3.torch_dtype')
        )
        self.vectorizer = SemanticVectorizer(
            config.get('model.sentence_bert.model_name'),
            config.get('model.sentence_bert.device')
        )
        
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
        
        # Training parameters
        self.learning_rate = config.get('training.learning_rate')
        self.batch_size = config.get('training.batch_size')
        self.epochs = config.get('training.epochs')
        self.snr_range = config.get('training.snr_range')
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.codec.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
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
        dataset = TextDataset(texts, self.config)
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
            
            # Forward pass
            encoded = self.codec.encode(vectors)
            
            # Add channel noise
            noisy = self.channel.add_noise(encoded, snr_db)
            
            # Decode
            decoded = self.codec.decode(noisy)
            
            # Calculate loss
            loss = self.criterion(decoded, vectors)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'SNR': f'{snr_db:.1f}dB',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'snr_db': snr_db,
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / num_batches
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

def load_sample_texts(num_samples: int = 1000) -> List[str]:
    """Load sample texts for training"""
    # This is a placeholder - you would load your actual dataset here
    sample_texts = [
        "The weather is beautiful today with clear blue skies and warm sunshine.",
        "Machine learning models require large amounts of training data to perform well.",
        "Climate change is one of the most pressing challenges of our time.",
        "The stock market experienced significant volatility during the past quarter.",
        "New advances in medical technology are improving patient outcomes worldwide.",
        "Renewable energy sources are becoming more cost-effective and efficient.",
        "Social media platforms have transformed how people communicate and share information.",
        "Artificial intelligence is revolutionizing many industries and professions.",
        "Urban planning must consider sustainability and quality of life factors.",
        "Space exploration continues to reveal fascinating discoveries about our universe."
    ]
    
    # Repeat and shuffle to get desired number of samples
    texts = []
    for i in range(num_samples):
        texts.append(sample_texts[i % len(sample_texts)])
    
    np.random.shuffle(texts)
    return texts

if __name__ == "__main__":
    config = Config()
    
    # Load training data
    train_texts = load_sample_texts(800)
    val_texts = load_sample_texts(200)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train(train_texts, val_texts)