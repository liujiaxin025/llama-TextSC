import torch
import numpy as np
from typing import Dict, Any, Optional

from ..utils.config import Config
from ..models.mlp_codec import ChannelCodec
from ..channel.awgn import AWGNChannel
from ..core.semantic_condenser import SemanticCondenser
from ..core.semantic_vectorizer import SemanticVectorizer
from ..core.semantic_generator import SemanticGenerator, SemanticDeVectorizer

class SemanticCommSystem:
    """
    Complete Semantic Communication System integrating all components
    """
    def __init__(self, config: Config, model_path: Optional[str] = None):
        self.config = config
        self.device = config.get('training.device')
        
        # Initialize MLP codec
        encoder_config = config.get('mlp.encoder')
        decoder_config = config.get('mlp.decoder')
        self.codec = ChannelCodec(encoder_config, decoder_config)
        
        if model_path:
            self._load_model(model_path)
        
        self.codec.to(self.device)
        self.codec.eval()
        
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
        
        # Initialize channel
        self.channel = AWGNChannel(self.device)
        
        print("Semantic Communication System initialized successfully!")
    
    def _load_model(self, model_path: str):
        """Load pre-trained channel codec"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.codec.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.codec.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using randomly initialized weights")
    
    def transmit(self, text: str, snr_db: float = 10.0) -> str:
        """
        Transmit text through the semantic communication system
        
        Args:
            text: Input text to transmit
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Recovered text after transmission through noisy channel
        """
        with torch.no_grad():
            # 1. Semantic Condensation (Llama 3)
            print(f"Original text: {text}")
            semantic_json = self.condenser.condense(text)
            print(f"Compressed semantic: {semantic_json}")
            
            # 2. Semantic Vectorization (Sentence-BERT)
            semantic_vector = self.vectorizer.vectorize(semantic_json)
            print(f"Semantic vector shape: {semantic_vector.shape}")
            
            # 3. Channel Encoding (MLP)
            encoded_signal = self.codec.encode(semantic_vector)
            print(f"Encoded signal shape: {encoded_signal.shape}")
            
            # 4. Channel Transmission (AWGN)
            noisy_signal = self.channel.add_noise(encoded_signal, snr_db)
            print(f"Added noise at {snr_db} dB SNR")
            
            # 5. Channel Decoding (MLP)
            decoded_vector = self.codec.decode(noisy_signal)
            print(f"Decoded vector shape: {decoded_vector.shape}")
            
            # 6. Semantic De-Vectorization
            recovered_json = self.devectorizer.decode_to_json(decoded_vector)[0]
            print(f"Recovered semantic: {recovered_json}")
            
            # 7. Semantic Generation (Llama 3)
            recovered_text = self.generator.generate(recovered_json)
            print(f"Recovered text: {recovered_text}")
            
            # Calculate transmission metrics
            mse = self.channel.calculate_mse(semantic_vector, decoded_vector)
            cosine_sim = torch.cosine_similarity(semantic_vector, decoded_vector).item()
            
            print(f"Channel MSE: {mse:.4f}")
            print(f"Cosine Similarity: {cosine_sim:.4f}")
            
            return recovered_text
    
    def batch_transmit(self, texts: list, snr_db: float = 10.0) -> list:
        """
        Transmit multiple texts through the system
        """
        recovered_texts = []
        
        for text in texts:
            recovered_text = self.transmit(text, snr_db)
            recovered_texts.append(recovered_text)
            
        return recovered_texts
    
    def evaluate_transmission(self, original_text: str, recovered_text: str) -> Dict[str, float]:
        """
        Evaluate transmission quality between original and recovered text
        """
        # Vectorize both texts for similarity comparison
        original_vector = self.vectorizer.vectorize(original_text)
        recovered_vector = self.vectorizer.vectorize(recovered_text)
        
        # Calculate similarity metrics
        cosine_sim = torch.cosine_similarity(original_vector, recovered_vector).item()
        mse = torch.mean((original_vector - recovered_vector) ** 2).item()
        
        return {
            'cosine_similarity': cosine_sim,
            'mse': mse
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system configuration information"""
        return {
            'device': self.device,
            'llama3_model': self.config.get('model.llama3.model_name'),
            'sentence_bert_model': self.config.get('model.sentence_bert.model_name'),
            'encoder_dims': {
                'input': self.config.get('mlp.encoder.input_dim'),
                'hidden': self.config.get('mlp.encoder.hidden_dims'),
                'output': self.config.get('mlp.encoder.output_dim')
            },
            'decoder_dims': {
                'input': self.config.get('mlp.decoder.input_dim'),
                'hidden': self.config.get('mlp.decoder.hidden_dims'),
                'output': self.config.get('mlp.decoder.output_dim')
            }
        }

# Convenience function for quick usage
def quick_transmit(text: str, snr_db: float = 10.0, model_path: str = None) -> str:
    """
    Quick transmission function for testing
    """
    config = Config()
    system = SemanticCommSystem(config, model_path)
    return system.transmit(text, snr_db)

if __name__ == "__main__":
    # Example usage
    config = Config()
    system = SemanticCommSystem(config)
    
    # Test transmission
    test_text = "Machine learning models require large amounts of training data to perform well."
    recovered_text = system.transmit(test_text, snr_db=10.0)
    
    # Print system info
    print("\nSystem Information:")
    info = system.get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")