import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


class MLPEncoder(nn.Module):
    """MLP-based channel encoder"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = "relu", dropout: float = 0.1):
        super(MLPEncoder, self).__init__()

        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = getattr(F, activation.lower())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        # No activation on final layer
        x = self.layers[-1](x)
        return x


class MLPDecoder(nn.Module):
    """MLP-based channel decoder"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = "relu", dropout: float = 0.1):
        super(MLPDecoder, self).__init__()

        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = getattr(F, activation.lower())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        # No activation on final layer
        x = self.layers[-1](x)
        return x


class ChannelCodec(nn.Module):
    """Complete channel codec combining encoder and decoder"""

    def __init__(self, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any]):
        super(ChannelCodec, self).__init__()

        # Initialize encoder
        self.encoder = MLPEncoder(
            input_dim=encoder_config['input_dim'],
            hidden_dims=encoder_config['hidden_dims'],
            output_dim=encoder_config['output_dim'],
            activation=encoder_config.get('activation', 'relu'),
            dropout=encoder_config.get('dropout', 0.1)
        )

        # Initialize decoder
        self.decoder = MLPDecoder(
            input_dim=decoder_config['input_dim'],
            hidden_dims=decoder_config['hidden_dims'],
            output_dim=decoder_config['output_dim'],
            activation=decoder_config.get('activation', 'relu'),
            dropout=decoder_config.get('dropout', 0.1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input vector for channel transmission"""
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode received vector from channel"""
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder (for training)"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


# Legacy compatibility class
class SemanticDeVectorizer(nn.Module):
    """Semantic De-Vectorizer for converting vectors back to semantic representations"""

    def __init__(self, vector_dim: int, hidden_dims: List[int] = [512, 256],
                 device: str = 'cpu'):
        super(SemanticDeVectorizer, self).__init__()

        self.device = device

        # Create MLP network
        dims = [vector_dim] + hidden_dims + [384]  # Output to match BERT embedding dim
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through de-vectorizer network"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        return x

    def decode_to_json(self, vectors: torch.Tensor) -> List[str]:
        """Convert vectors back to JSON-like semantic representations"""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated reconstruction
        batch_size = vectors.shape[0]

        # Create dummy JSON representations
        semantic_jsons = []
        for i in range(batch_size):
            # For demo purposes, create a simple structured output
            json_repr = {
                "keywords": ["sample", "text", "semantic"],
                "entities": ["SYSTEM"],
                "sentiment": "neutral",
                "summary": "Reconstructed semantic content"
            }
            semantic_jsons.append(str(json_repr))

        return semantic_jsons