#!/usr/bin/env python3
"""
Base Configuration for LDPC Steganography System
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field
import torch


@dataclass
class BaseConfig:
    """Base configuration class for LDPC steganography system"""
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Basic model settings
    image_size: int = 256
    channels: int = 3
    batch_size: int = 8
    
    # Message settings
    message_length: int = 1024
    
    # Data paths
    data_train_folder: str = "data/train"
    data_val_folder: str = "data/val"
    output_dir: str = "results"
    log_dir: str = "logs"
    
    # Experiment settings
    experiment_name: str = "ldpc_steganography"
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set random seeds
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration"""
    
    # UNet settings
    unet_base_channels: int = 64
    unet_depth: int = 5
    attention_layers: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # CVAE settings
    latent_dim: int = 256
    
    # Activation functions
    activation: str = "gelu"
    output_activation: str = "tanh"


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration"""
    
    # Training settings
    num_epochs: int = 300
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    clip_grad_norm: float = 1.0
    
    # Scheduler settings
    scheduler_type: str = "cosine"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'message': 12.0,
        'mse': 2.0,
        'lpips': 1.5,
        'ssim': 1.0,
        'adversarial': 0.5,
        'recovery_mse': 1.0,
        'recovery_kl': 0.1,
    })
    
    # Validation settings
    validation_frequency: int = 1
    save_frequency: int = 10
    
    # Early stopping
    patience: int = 30
    min_delta: float = 1e-4
    
    # Checkpoint settings
    save_best_only: bool = False
    save_last: bool = True
    
    def get_scheduler_config(self):
        """Get scheduler configuration"""
        if self.scheduler_type == "cosine":
            return {
                "T_max": self.num_epochs,
                "eta_min": self.learning_rate * 0.1,
                **self.scheduler_params
            }
        elif self.scheduler_type == "step":
            return {
                "step_size": 100,
                "gamma": 0.1,
                **self.scheduler_params
            }
        else:
            return self.scheduler_params