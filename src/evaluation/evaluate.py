import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from bert_score import score as bert_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import nltk

from ..utils.config import Config
from ..models.mlp_codec import ChannelCodec
from ..channel.awgn import AWGNChannel
from ..core.semantic_condenser import SemanticCondenser
from ..core.semantic_vectorizer import SemanticVectorizer
from ..core.semantic_generator import SemanticGenerator, SemanticDeVectorizer

class SemanticCommEvaluator:
    def __init__(self, config: Config, model_path: str = None):
        self.config = config
        self.device = config.get('training.device')
        
        # Load trained model
        encoder_config = config.get('mlp.encoder')
        decoder_config = config.get('mlp.decoder')
        
        self.codec = ChannelCodec(encoder_config, decoder_config)
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.codec.to(self.device)
        self.codec.eval()
        
        # Initialize components
        self.channel = AWGNChannel(self.device)
        self.condenser = SemanticCondenser(
            config.get('model.llama3.model_name'),
            config.get('model.llama3.device'),
            config.get('model.llama3.torch_dtype')
        )
        self.vectorizer = SemanticVectorizer(
            config.get('model.sentence_bert.model_name'),
            config.get('model.sentence_bert.device')
        )
        self.devectorizer = SemanticDeVectorizer(
            vector_dim=config.get('mlp.decoder.output_dim'),
            hidden_dims=[512, 256],
            device=self.device
        )
        self.generator = SemanticGenerator(
            config.get('model.llama3.model_name'),
            config.get('model.llama3.device'),
            config.get('model.llama3.torch_dtype')
        )
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.codec.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.codec.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Model loaded from {filepath}")
    
    def transmit_text(self, text: str, snr_db: float) -> Tuple[str, Dict[str, Any]]:
        """
        Transmit text through the semantic communication system
        """
        with torch.no_grad():
            # 1. Semantic Condensation
            semantic_json = self.condenser.condense(text)
            
            # 2. Semantic Vectorization
            semantic_vector = self.vectorizer.vectorize(semantic_json)
            
            # 3. Channel Encoding
            encoded = self.codec.encode(semantic_vector)
            
            # 4. Channel Transmission (with noise)
            noisy_signal = self.channel.add_noise(encoded, snr_db)
            
            # 5. Channel Decoding
            decoded_vector = self.codec.decode(noisy_signal)
            
            # 6. Semantic De-Vectorization
            recovered_json = self.devectorizer.decode_to_json(decoded_vector)[0]
            
            # 7. Semantic Generation
            recovered_text = self.generator.generate(recovered_json)
            
            # Calculate metrics
            mse = self.channel.calculate_mse(semantic_vector, decoded_vector)
            
            metrics = {
                'mse': mse,
                'original_semantic': semantic_json,
                'recovered_semantic': recovered_json,
                'snr_db': snr_db
            }
            
        return recovered_text, metrics
    
    def evaluate_semantic_similarity(self, original_text: str, recovered_text: str) -> Dict[str, float]:
        """
        Evaluate semantic similarity between original and recovered texts
        """
        # BERTScore
        P, R, F1 = bert_score([recovered_text], [original_text], lang='en')
        
        # ROUGE scores
        rouge = Rouge()
        try:
            rouge_scores = rouge.get_scores(recovered_text, original_text)[0]
            rouge_f1 = rouge_scores['rouge-l']['f']
        except:
            rouge_f1 = 0.0
        
        # BLEU score
        try:
            original_tokens = nltk.word_tokenize(original_text.lower())
            recovered_tokens = nltk.word_tokenize(recovered_text.lower())
            bleu_score = sentence_bleu([original_tokens], recovered_tokens)
        except:
            bleu_score = 0.0
        
        # Cosine similarity of semantic embeddings
        original_embedding = self.vectorizer.vectorize(original_text)
        recovered_embedding = self.vectorizer.vectorize(recovered_text)
        cosine_sim = torch.cosine_similarity(original_embedding, recovered_embedding).item()
        
        return {
            'bert_score_f1': F1.mean().item(),
            'rouge_f1': rouge_f1,
            'bleu_score': bleu_score,
            'cosine_similarity': cosine_sim
        }
    
    def evaluate_dataset(self, test_texts: List[str], snr_test_points: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate system on a dataset across different SNR points
        """
        if snr_test_points is None:
            snr_test_points = self.config.get('evaluation.snr_test_points', [0, 5, 10, 15, 20])
        
        results = {
            'snr_points': snr_test_points,
            'metrics_by_snr': {},
            'overall_results': []
        }
        
        for snr_db in snr_test_points:
            print(f"\nEvaluating at SNR = {snr_db} dB")
            snr_results = []
            
            for text in tqdm(test_texts, desc=f"SNR {snr_db}dB"):
                # Transmit through system
                recovered_text, channel_metrics = self.transmit_text(text, snr_db)
                
                # Evaluate semantic similarity
                semantic_metrics = self.evaluate_semantic_similarity(text, recovered_text)
                
                # Combine all metrics
                combined_metrics = {
                    'original_text': text,
                    'recovered_text': recovered_text,
                    'channel_mse': channel_metrics['mse'],
                    'snr_db': snr_db
                }
                combined_metrics.update(semantic_metrics)
                
                snr_results.append(combined_metrics)
            
            # Calculate averages for this SNR
            avg_metrics = self._calculate_average_metrics(snr_results)
            results['metrics_by_snr'][snr_db] = avg_metrics
            results['overall_results'].extend(snr_results)
            
            print(f"Average metrics at SNR {snr_db}dB:")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        return results
    
    def _calculate_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics from a list of results"""
        metric_keys = ['channel_mse', 'bert_score_f1', 'rouge_f1', 'bleu_score', 'cosine_similarity']
        avg_metrics = {}
        
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """
        Plot evaluation results
        """
        snr_points = results['snr_points']
        
        # Extract metrics for plotting
        metrics = {
            'Channel MSE': [],
            'BERT Score F1': [],
            'ROUGE F1': [],
            'BLEU Score': [],
            'Cosine Similarity': []
        }
        
        for snr in snr_points:
            snr_results = results['metrics_by_snr'][snr]
            metrics['Channel MSE'].append(snr_results['channel_mse'])
            metrics['BERT Score F1'].append(snr_results['bert_score_f1'])
            metrics['ROUGE F1'].append(snr_results['rouge_f1'])
            metrics['BLEU Score'].append(snr_results['bleu_score'])
            metrics['Cosine Similarity'].append(snr_results['cosine_similarity'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Semantic Communication System Performance', fontsize=16)
        
        # Plot each metric
        for idx, (metric_name, values) in enumerate(metrics.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            ax.plot(snr_points, values, 'bo-', linewidth=2, markersize=6)
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs SNR')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = str(value)  # Convert to string for non-JSON objects
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filepath}")

def load_test_texts(num_samples: int = 100) -> List[str]:
    """Load test texts for evaluation"""
    test_texts = [
        "The rapid advancement of artificial intelligence has transformed many industries in recent years.",
        "Climate change poses significant challenges to global ecosystems and human societies.",
        "Renewable energy technologies are becoming increasingly efficient and cost-effective.",
        "The integration of IoT devices in smart cities improves urban infrastructure management.",
        "Medical breakthroughs in genetics and personalized medicine offer new treatment possibilities.",
        "Sustainable agriculture practices are essential for food security and environmental protection.",
        "Quantum computing has the potential to revolutionize computational capabilities.",
        "Space exploration continues to reveal fascinating discoveries about our universe.",
        "Digital transformation is reshaping business models and customer experiences.",
        "Educational technology platforms are making learning more accessible and personalized."
    ]
    
    # Generate more samples by variation
    texts = []
    for i in range(num_samples):
        base_text = test_texts[i % len(test_texts)]
        # Add small variations
        texts.append(base_text)
    
    return texts

if __name__ == "__main__":
    config = Config()
    
    # Initialize evaluator
    model_path = "checkpoints/best_model.pth"
    evaluator = SemanticCommEvaluator(config, model_path)
    
    # Load test data
    test_texts = load_test_texts(50)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_texts)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    evaluator.plot_results(results, "performance_curves.png")