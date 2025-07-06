#!/usr/bin/env python3
"""
LDPC-Aware Dual UNet Encoder
Encoder network designed to work with LDPC error correction codes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

from ..blocks.conv_block import ConvBlock
from ..blocks.residual_block import ResidualBlock
from ..blocks.unet_layers import UNet23Layers
from ..attention.self_attention import SelfAttention
from ...utils.helpers import calculate_output_padding


class LDPCAwareDualUNetEncoder(nn.Module):
    """Dual UNet encoder with LDPC awareness"""
    
    def __init__(self, config):
        """
        Initialize LDPC-aware encoder
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        # Configuration
        self.config = config
        self.image_size = config.image_size
        self.channels = config.channels
        self.message_length = config.message_length
        
        # Calculate maximum encoded message length with LDPC
        self.max_encoded_length = int(self.message_length / (1 - config.ldpc_max_redundancy))
        
        # Message preprocessing
        self.message_preprocessor = MessagePreprocessor(
            message_length=self.max_encoded_length,
            embed_dim=256,
            output_dim=512
        )
        
        # Feature injection preparation
        self.feature_injector = FeatureInjector(
            message_dim=512,
            feature_channels=config.unet_base_channels
        )
        
        # Primary UNet for feature extraction
        self.primary_unet = UNet23Layers(
            in_channels=self.channels,
            out_channels=config.unet_base_channels,
            base_channels=config.unet_base_channels,
            attention_layers=config.attention_layers,
            attention_type='self',
            norm_type="group"
        )
        
        # Secondary UNet for steganography
        self.secondary_unet = UNet23Layers(
            in_channels=config.unet_base_channels * 2,  # Concatenated features
            out_channels=self.channels,
            base_channels=config.unet_base_channels,
            attention_layers=config.attention_layers,
            attention_type='self',
            norm_type="group"
        )
        
        # Adaptive strength control
        self.strength_controller = AdaptiveStrengthController(
            feature_dim=config.unet_base_channels,
            min_strength=0.1,
            max_strength=0.5
        )
        
        # Final refinement
        self.refinement_network = RefinementNetwork(
            channels=self.channels,
            hidden_channels=config.unet_base_channels
        )
        
        logging.info(f"LDPC-Aware Encoder initialized: max_encoded_length={self.max_encoded_length}")
    
    def forward(self, cover_images: torch.Tensor, ldpc_encoded_messages: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            cover_images: Cover images [B, C, H, W]
            ldpc_encoded_messages: LDPC encoded messages [B, encoded_length]
            
        Returns:
            Stego images [B, C, H, W]
        """
        batch_size = cover_images.size(0)
        
        # Pad messages to max length if needed
        if ldpc_encoded_messages.size(1) < self.max_encoded_length:
            padding = self.max_encoded_length - ldpc_encoded_messages.size(1)
            ldpc_encoded_messages = F.pad(ldpc_encoded_messages, (0, padding))
        elif ldpc_encoded_messages.size(1) > self.max_encoded_length:
            ldpc_encoded_messages = ldpc_encoded_messages[:, :self.max_encoded_length]
        
        # Process messages
        message_features = self.message_preprocessor(ldpc_encoded_messages)
        
        # Extract cover features using primary UNet
        cover_features = self.primary_unet(cover_images)
        
        # Calculate adaptive embedding strength
        embedding_strength = self.strength_controller(cover_features, message_features)
        
        # Inject message features
        injected_features = self.feature_injector(
            cover_features, message_features, embedding_strength
        )
        
        # Generate stego features using secondary UNet
        stego_features = self.secondary_unet(
            torch.cat([cover_features, injected_features], dim=1)
        )
        
        # Refine to get final stego images
        stego_images = self.refinement_network(cover_images, stego_features)
        
        return stego_images


class MessagePreprocessor(nn.Module):
    """Preprocess binary messages for embedding"""
    
    def __init__(self, message_length: int, embed_dim: int = 256, output_dim: int = 512):
        super().__init__()
        
        self.message_length = message_length
        
        # Message embedding
        self.embedding = nn.Sequential(
            nn.Linear(message_length, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, output_dim)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(output_dim)
        
        # Self-attention for message understanding
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=3
        )
        
    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """Process messages"""
        # Embed messages
        embedded = self.embedding(messages)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded.unsqueeze(1))
        
        # Apply self-attention
        attended = self.self_attention(embedded).squeeze(1)
        
        return attended


class FeatureInjector(nn.Module):
    """Inject message features into cover features"""
    
    def __init__(self, message_dim: int, feature_channels: int):
        super().__init__()
        
        # Message to spatial features
        self.message_to_spatial = nn.Sequential(
            nn.Linear(message_dim, feature_channels * 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels * 16, feature_channels * 8 * 8),
            nn.ReLU(inplace=True)
        )
        
        # Feature modulation
        self.modulation = nn.Sequential(
            ConvBlock(feature_channels * 2, feature_channels),
            ResidualBlock(feature_channels, use_attention=True),
            ConvBlock(feature_channels, feature_channels)
        )
        
    def forward(self, cover_features: torch.Tensor, message_features: torch.Tensor,
                strength: torch.Tensor) -> torch.Tensor:
        """Inject message into cover features"""
        B, C, H, W = cover_features.shape
        
        # Convert message to spatial features
        spatial_message = self.message_to_spatial(message_features)
        spatial_message = spatial_message.view(B, C, 8, 8)
        
        # Upsample to match cover features
        spatial_message = F.interpolate(spatial_message, size=(H, W), 
                                      mode='bilinear', align_corners=False)
        
        # Modulate features
        combined = torch.cat([cover_features, spatial_message], dim=1)
        modulated = self.modulation(combined)
        
        # Apply adaptive strength
        injected = cover_features + strength * modulated
        
        return injected


class AdaptiveStrengthController(nn.Module):
    """Control embedding strength based on cover and message"""
    
    def __init__(self, feature_dim: int, min_strength: float = 0.1, max_strength: float = 0.5):
        super().__init__()
        
        self.min_strength = min_strength
        self.max_strength = max_strength
        
        # Cover complexity analyzer
        self.cover_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        # Message importance analyzer
        self.message_analyzer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        # Strength predictor
        self.strength_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, cover_features: torch.Tensor, message_features: torch.Tensor) -> torch.Tensor:
        """Calculate adaptive embedding strength"""
        # Analyze cover complexity
        cover_analysis = self.cover_analyzer(cover_features)
        
        # Analyze message importance
        message_analysis = self.message_analyzer(message_features)
        
        # Combine analyses
        combined = torch.cat([cover_analysis, message_analysis], dim=1)
        
        # Predict strength
        strength = self.strength_predictor(combined)
        
        # Scale to desired range
        strength = self.min_strength + (self.max_strength - self.min_strength) * strength
        
        return strength.unsqueeze(-1).unsqueeze(-1)


class RefinementNetwork(nn.Module):
    """Refine stego images for better quality"""
    
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        
        # Multi-scale refinement
        self.refinement_blocks = nn.ModuleList([
            ResidualBlock(channels, use_attention=False),
            ResidualBlock(channels, use_attention=True),
            ResidualBlock(channels, use_attention=False)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            ConvBlock(channels * 2, hidden_channels),
            ConvBlock(hidden_channels, channels),
            nn.Conv2d(channels, channels, 1)
        )
        
        # Final activation
        self.final_activation = nn.Tanh()
        
    def forward(self, cover_images: torch.Tensor, stego_features: torch.Tensor) -> torch.Tensor:
        """Refine stego images"""
        # Multi-scale refinement
        refined = stego_features
        for block in self.refinement_blocks:
            refined = block(refined)
        
        # Fuse with cover
        fused = self.fusion(torch.cat([cover_images, refined], dim=1))
        
        # Ensure output is in valid range
        stego_images = self.final_activation(fused)
        
        return stego_images


class PositionalEncoding(nn.Module):
    """Positional encoding for messages"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding"""
        return x + self.pe[:, :x.size(1), :]