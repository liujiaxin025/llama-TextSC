#!/usr/bin/env python3
"""
Training script for Llama3.2-3B model on 4070ti GPU
Optimized for VRAM constraints with mixed precision and gradient accumulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train import Trainer, load_cnn_dailymail_subset
from src.utils.config import Config

def main():
    # Load the Llama3.2-3B optimized configuration
    config = Config("config/llama3_2_3b.yaml")

    # Show training configuration
    print("=== Llama3.2-3B Training Configuration ===")
    print(f"Model: {config.get('model.llama3.model_name')}")
    print(f"Device: {config.get('training.device')}")
    print(f"Mixed Precision: {config.get('training.mixed_precision')}")
    print(f"Epochs: {config.get('training.epochs')}")
    print(f"Batch Size: {config.get('training.batch_size')}")
    print(f"Gradient Accumulation Steps: {config.get('training.gradient_accumulation_steps')}")
    print(f"Effective Batch Size: {config.get('training.batch_size') * config.get('training.gradient_accumulation_steps', 1)}")
    print(f"Learning Rate: {config.get('training.learning_rate')}")
    print(f"SNR Range: {config.get('training.snr_range')}")
    print("=" * 50)

    # Load training data
    print("Loading training data...")
    try:
        train_texts = load_cnn_dailymail_subset(1000, "train")  # 1000 training samples
        val_texts = load_cnn_dailymail_subset(200, "validation")  # 200 validation samples
    except Exception as e:
        print(f"Error loading CNN/DailyMail dataset: {e}")
        print("Using fallback dataset...")
        from src.training.train import load_news_fallback
        train_texts = load_news_fallback(1000)
        val_texts = load_news_fallback(200)

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Initialize trainer
    trainer = Trainer(config)

    # Train model
    print("Starting training...")
    trainer.train(train_texts, val_texts)

    print("Training completed! Model saved to checkpoints/best_model.pth")
    print("You can now use the trained model for semantic communication.")

if __name__ == "__main__":
    main()