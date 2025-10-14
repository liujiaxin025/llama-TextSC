#!/usr/bin/env python3
"""
Demo script for Llama3-TextSC system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.semantic_comm_system import SemanticCommSystem, quick_transmit
from src.utils.config import Config
from src.evaluation.evaluate import SemanticCommEvaluator

def demo_basic_transmission():
    """Demonstrate basic text transmission through the system"""
    print("=" * 60)
    print("Llama3-TextSC Basic Transmission Demo")
    print("=" * 60)
    
    # Sample texts for demonstration
    test_texts = [
        "The weather is beautiful today with clear blue skies and warm sunshine.",
        "Machine learning models require large amounts of training data to perform well.",
        "Climate change is one of the most pressing challenges of our time.",
        "The stock market experienced significant volatility during the past quarter.",
        "New advances in medical technology are improving patient outcomes worldwide."
    ]
    
    # Initialize system
    config = Config()
    system = SemanticCommSystem(config)
    
    print(f"\nSystem Configuration:")
    info = system.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")
    print("Text Transmission Examples")
    print(f"{'='*60}")
    
    # Test different SNR levels
    snr_levels = [5.0, 10.0, 15.0, 20.0]
    
    for i, text in enumerate(test_texts[:2]):  # Test first 2 texts
        print(f"\n--- Example {i+1} ---")
        print(f"Original: {text}")
        
        for snr_db in snr_levels:
            print(f"\nSNR: {snr_db} dB")
            recovered = system.transmit(text, snr_db)
            print(f"Recovered: {recovered}")
            
            # Evaluate quality
            metrics = system.evaluate_transmission(text, recovered)
            print(f"Quality - Cosine Similarity: {metrics['cosine_similarity']:.3f}, "
                  f"MSE: {metrics['mse']:.4f}")
        
        print("-" * 40)

def demo_quick_transmit():
    """Demonstrate quick transmission function"""
    print("\n" + "=" * 60)
    print("Quick Transmission Demo")
    print("=" * 60)
    
    text = "Artificial intelligence is revolutionizing healthcare diagnostics."
    print(f"Original: {text}")
    
    for snr_db in [5.0, 10.0, 15.0]:
        print(f"\nTransmitting at {snr_db} dB SNR...")
        recovered = quick_transmit(text, snr_db)
        print(f"Recovered: {recovered}")

def demo_evaluation():
    """Demonstrate system evaluation"""
    print("\n" + "=" * 60)
    print("System Evaluation Demo")
    print("=" * 60)
    
    config = Config()
    
    # Small test dataset for demo
    test_texts = [
        "Renewable energy sources are becoming more cost-effective and efficient.",
        "Social media platforms have transformed how people communicate and share information.",
        "Urban planning must consider sustainability and quality of life factors.",
        "Space exploration continues to reveal fascinating discoveries about our universe."
    ]
    
    # Initialize evaluator (without pre-trained model for demo)
    evaluator = SemanticCommEvaluator(config)
    
    print("Running evaluation on small test dataset...")
    results = evaluator.evaluate_dataset(test_texts, snr_test_points=[5.0, 10.0, 15.0])
    
    print("\nEvaluation Results Summary:")
    for snr_db, metrics in results['metrics_by_snr'].items():
        print(f"\nSNR {snr_db} dB:")
        for metric, value in metrics.items():
            if not metric.endswith('_std'):
                print(f"  {metric}: {value:.4f}")

def demo_training():
    """Demonstrate training setup (without actually training)"""
    print("\n" + "=" * 60)
    print("Training Setup Demo")
    print("=" * 60)
    
    print("To train the semantic communication system, run:")
    print("python src/training/train.py")
    print("\nTraining configuration:")
    
    config = Config()
    training_params = {
        'Batch size': config.get('training.batch_size'),
        'Learning rate': config.get('training.learning_rate'),
        'Epochs': config.get('training.epochs'),
        'SNR training range': config.get('training.snr_range'),
        'Device': config.get('training.device')
    }
    
    for param, value in training_params.items():
        print(f"  {param}: {value}")
    
    print("\nNote: This would require a dataset of texts and significant computational resources.")
    print("The MLP encoder-decoder would be trained end-to-end using MSE loss.")

if __name__ == "__main__":
    print("Llama3-TextSC Demo Suite")
    print("This demo showcases the semantic communication system capabilities.")
    
    try:
        # Run demos
        demo_basic_transmission()
        demo_quick_transmit()
        demo_evaluation()
        demo_training()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train the system: python src/training/train.py")
        print("3. Run full evaluation: python src/evaluation/evaluate.py")
        print("4. Use the system in your own projects!")
        
    except Exception as e:
        print(f"\nDemo encountered an error: {e}")
        print("\nMake sure to install the required dependencies:")
        print("pip install -r requirements.txt")
        print("\nNote: Some components may require GPU access and model downloads.")