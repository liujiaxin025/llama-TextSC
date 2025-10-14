import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class AWGNChannel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def add_noise(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add AWGN noise to signal
        
        Args:
            signal: Input signal tensor [batch_size, signal_length]
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy signal tensor
        """
        # Calculate signal power
        signal_power = torch.mean(signal ** 2, dim=1, keepdim=True)
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate noise power
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        
        # Add noise to signal
        noisy_signal = signal + noise
        
        return noisy_signal
    
    def calculate_ber(self, original_signal: torch.Tensor, 
                     recovered_signal: torch.Tensor, 
                     threshold: float = 0.0) -> float:
        """
        Calculate Bit Error Rate (BER) for binary signals
        
        Args:
            original_signal: Original binary signal
            recovered_signal: Recovered signal after demodulation
            threshold: Decision threshold for binary detection
            
        Returns:
            BER value
        """
        # Binarize signals
        original_binary = (original_signal > threshold).float()
        recovered_binary = (recovered_signal > threshold).float()
        
        # Calculate errors
        errors = torch.sum(original_binary != recovered_binary).item()
        total_bits = original_binary.numel()
        
        return errors / total_bits if total_bits > 0 else 0.0
    
    def calculate_mse(self, original_signal: torch.Tensor, 
                     recovered_signal: torch.Tensor) -> float:
        """
        Calculate Mean Squared Error
        
        Args:
            original_signal: Original signal
            recovered_signal: Recovered signal
            
        Returns:
            MSE value
        """
        mse = torch.mean((original_signal - recovered_signal) ** 2).item()
        return mse
    
    def batch_add_noise(self, signals: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add AWGN noise to a batch of signals
        
        Args:
            signals: Batch of input signals [batch_size, signal_length]
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Batch of noisy signals
        """
        return self.add_noise(signals, snr_db)