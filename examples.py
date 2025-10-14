#!/usr/bin/env python3
"""
Simple usage examples for Llama3-TextSC
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.semantic_comm_system import quick_transmit
from src.utils.config import Config

def simple_example():
    """Most basic usage example"""
    print("Simple Text Transmission Example:")
    print("-" * 40)
    
    # Your text to transmit
    original_text = "The weather is beautiful today with clear blue skies."
    
    # Transmit through noisy channel (10 dB SNR)
    recovered_text = quick_transmit(original_text, snr_db=10.0)
    
    print(f"Original:  {original_text}")
    print(f"Recovered: {recovered_text}")

def batch_example():
    """Example with multiple texts"""
    print("\nBatch Transmission Example:")
    print("-" * 40)
    
    texts = [
        "Machine learning models require training data.",
        "Climate change affects global ecosystems.",
        "Renewable energy is becoming more efficient."
    ]
    
    print("Transmitting multiple texts at different SNR levels:")
    
    for snr in [5, 10, 15]:
        print(f"\n--- SNR: {snr} dB ---")
        for i, text in enumerate(texts):
            recovered = quick_transmit(text, snr_db=snr)
            print(f"Text {i+1}: {recovered}")

def quality_example():
    """Example showing transmission quality metrics"""
    print("\nQuality Analysis Example:")
    print("-" * 40)
    
    from src.core.semantic_comm_system import SemanticCommSystem
    
    # Initialize system with more control
    config = Config()
    system = SemanticCommSystem(config)
    
    text = "Artificial intelligence is transforming healthcare diagnostics."
    
    print("Analyzing transmission quality at different SNR levels:")
    
    for snr in [5, 10, 15, 20]:
        recovered = system.transmit(text, snr_db=snr)
        metrics = system.evaluate_transmission(text, recovered)
        
        print(f"\nSNR {snr} dB:")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.3f}")
        print(f"  MSE: {metrics['mse']:.4f}")

if __name__ == "__main__":
    print("Llama3-TextSC Simple Usage Examples")
    print("===================================")
    
    # Run examples
    simple_example()
    batch_example() 
    quality_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo use in your own code:")
    print("from src.core.semantic_comm_system import quick_transmit")
    print("recovered = quick_transmit('your text here', snr_db=10.0)")