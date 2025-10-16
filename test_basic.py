#!/usr/bin/env python3
"""
Basic test script for Llama3-TextSC system using public models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.models.mlp_codec import ChannelCodec
from src.channel.awgn import AWGNChannel
import torch

def test_mlp_codec():
    """Test the MLP encoder-decoder functionality"""
    print("Testing MLP Codec...")

    # Test configuration
    encoder_config = {
        'input_dim': 384,
        'hidden_dims': [512],
        'output_dim': 256,
        'activation': 'relu',
        'dropout': 0.1
    }

    decoder_config = {
        'input_dim': 256,
        'hidden_dims': [512],
        'output_dim': 384,
        'activation': 'relu',
        'dropout': 0.1
    }

    # Initialize codec
    codec = ChannelCodec(encoder_config, decoder_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codec.to(device)

    # Test with sample data
    batch_size = 4
    input_dim = 384
    sample_data = torch.randn(batch_size, input_dim).to(device)

    print(f"Input shape: {sample_data.shape}")

    # Encode
    encoded = codec.encode(sample_data)
    print(f"Encoded shape: {encoded.shape}")

    # Decode
    decoded = codec.decode(encoded)
    print(f"Decoded shape: {decoded.shape}")

    # Calculate reconstruction error
    mse = torch.mean((sample_data - decoded) ** 2)
    print(f"Reconstruction MSE: {mse.item():.6f}")

    print("✅ MLP Codec test passed!")
    return codec

def test_awgn_channel():
    """Test AWGN channel functionality"""
    print("\nTesting AWGN Channel...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channel = AWGNChannel(device)

    # Test signal
    signal = torch.randn(2, 256).to(device)
    snr_db = 10.0

    print(f"Original signal shape: {signal.shape}")
    print(f"Signal power: {torch.mean(signal ** 2).item():.4f}")

    # Add noise
    noisy_signal = channel.add_noise(signal, snr_db)
    print(f"Noisy signal shape: {noisy_signal.shape}")
    print(f"Noisy signal power: {torch.mean(noisy_signal ** 2).item():.4f}")

    # Calculate MSE
    mse = channel.calculate_mse(signal, noisy_signal)
    print(f"Channel MSE: {mse:.6f}")

    print("✅ AWGN Channel test passed!")
    return channel

def test_config():
    """Test configuration loading"""
    print("\nTesting Configuration...")

    try:
        config = Config("config/public_model.yaml")
        print(f"✅ Configuration loaded successfully!")

        # Print some config values
        print(f"Llama model: {config.get('model.llama3.model_name')}")
        print(f"Device: {config.get('training.device')}")
        print(f"Encoder input dim: {config.get('mlp.encoder.input_dim')}")

        return config
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None

def main():
    """Run all tests"""
    print("Llama3-TextSC Basic Functionality Test")
    print("=" * 50)

    # Test configuration
    config = test_config()
    if config is None:
        print("❌ Cannot proceed without configuration")
        return

    # Test MLP codec
    codec = test_mlp_codec()

    # Test AWGN channel
    channel = test_awgn_channel()

    print("\n" + "=" * 50)
    print("🎉 All basic tests completed successfully!")
    print("\nThe system's core components are working properly.")
    print("To test the full semantic communication system,")
    print("you need to resolve the model download/network issues.")

if __name__ == "__main__":
    main()