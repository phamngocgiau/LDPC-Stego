#!/usr/bin/env python3
from .base_config import BaseConfig
from dataclasses import dataclass, field
from typing import Dict, List
import torch

@dataclass
class LDPCConfig(BaseConfig):
    ldpc_min_redundancy: float = 0.1
    ldpc_max_redundancy: float = 0.5
    ldpc_use_neural_decoder: bool = True
    ldpc_parallel_encoder: bool = False
    
    unet_base_channels: int = 64
    unet_depth: int = 5
    attention_layers: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    latent_dim: int = 256
    
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'message': 12.0, 'mse': 2.0, 'lpips': 1.5, 'ssim': 1.0,
        'adversarial': 0.5, 'recovery_mse': 1.0, 'recovery_kl': 0.1,
    })
    
    learning_rate: float = 2e-4
    num_epochs: int = 300
    validation_frequency: int = 1
    save_frequency: int = 10
    patience: int = 30
    clip_grad_norm: float = 1.0
    
    data_train_folder: str = "data/train"
    data_val_folder: str = "data/val"
    data_test_folder: str = "data/test"

def setup_debug_config():
    config = LDPCConfig()
    config.device = 'cpu'
    config.batch_size = 1
    config.image_size = 64
    config.message_length = 128
    config.unet_base_channels = 16
    config.unet_depth = 3
    return config
