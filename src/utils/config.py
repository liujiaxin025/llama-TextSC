import yaml
import torch
from typing import Dict, Any
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config['model']['llama3']['device'] = device
        self.config['model']['sentence_bert']['device'] = device
        self.config['training']['device'] = device
        
        return config
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def update(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value