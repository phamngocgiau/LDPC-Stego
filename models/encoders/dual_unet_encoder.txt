#!/usr/bin/env python3
"""
Dual UNet Encoder
Dual UNet architecture for image encoding without LDPC awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import logging

from ..blocks.conv_block import ConvBlock
from ..blocks.residual_block import ResidualBlock
from ..blocks.unet_layers import UNetEncoder, UNetDecoder
from ..attention.self_attention import SelfAttention


class DualUNetEncoder(nn.Module):
    """Dual UNet encoder for steganography"""
    
    def __init__(self, config):
        """
        Initialize dual UNet encoder
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.channels = config.channels
        self.message_length = config.message_length
        self.base_channels = config.unet_base_channels
        
        # Message processing
        self.message_processor = MessageProcessor(
            message_length=self.message_length,
            feature_dim=self.base_channels * 4
        )
        
        # First UNet - Feature extraction
        self.feature_unet = FeatureExtractionUNet(
            in_channels=self.channels,
            base_channels=self.base_channels,
            depth=config.unet_depth
        )
        
        # Second UNet - Steganography
        self.stego_unet = SteganographyUNet(
            in_channels=self.channels + self.base_channels,
            out_channels=self.channels,
            base_channels=self.base_channels,
            depth=config.unet_depth
        )
        
        # Message injection network
        self.message_injector = MessageInjector(
            feature_channels=self.base_channels,
            message_channels=self.base_channels * 4
        )
        
        logging.info("Dual UNet Encoder initialized")
    
    def forward(self, cover_images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            cover_images: Cover images [B, C, H, W]
            messages: Binary messages [B, message_length]
            
        Returns:
            Stego images [B, C, H, W]
        """
        # Extract features from cover images
        cover_features = self.feature_unet(cover_images)
        
        # Process messages
        message_features = self.message_processor(messages)
        
        # Inject messages into features
        modified_features = self.message_injector(cover_features, message_features)
        
        # Generate stego images
        combined_input = torch.cat([cover_images, modified_features], dim=1)
        stego_images = self.stego_unet(combined_input)
        
        return stego_images


class FeatureExtractionUNet(nn.Module):
    """UNet for feature extraction"""
    
    def __init__(self, in_channels: int, base_channels: int, depth: int = 5):
        super().__init__()
        
        self.encoder = UNetEncoder(in_channels, base_channels, depth)
        self.decoder = UNetDecoder(base_channels, depth, base_channels)
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** (depth - 1))
        self.bottleneck = nn.Sequential(
            ResidualBlock(bottleneck_channels, use_attention=True),
            ConvBlock(bottleneck_channels, bottleneck_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        # Encode
        encoder_features = self.encoder(x)
        
        # Bottleneck
        bottleneck_feat = self.bottleneck(encoder_features[-1])
        encoder_features[-1] = bottleneck_feat
        
        # Decode
        features = self.decoder(encoder_features)
        
        return features


class SteganographyUNet(nn.Module):
    """UNet for steganography"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 base_channels: int, depth: int = 5):
        super().__init__()
        
        # Encoder path
        self.encoders = nn.ModuleList()
        channels = in_channels
        
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(
                nn.Sequential(
                    ConvBlock(channels, out_ch),
                    ResidualBlock(out_ch, use_attention=(i >= depth // 2))
                )
            )
            channels = out_ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(channels, channels * 2),
            ResidualBlock(channels * 2, use_attention=True),
            ConvBlock(channels * 2, channels)
        )
        
        # Decoder path
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            in_ch = channels
            out_ch = base_channels * (2 ** i)
            
            self.upsamplers.append(
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
            )
            
            self.decoders.append(
                nn.Sequential(
                    ConvBlock(out_ch * 2, out_ch),  # Skip connection
                    ResidualBlock(out_ch, use_attention=(i >= depth // 2))
                )
            )
            channels = out_ch
        
        # Output layer
        self.output = nn.Sequential(
            ConvBlock(base_channels, base_channels // 2),
            nn.Conv2d(base_channels // 2, out_channels, 1),
            nn.Tanh()
        )
        
        # Downsampling
        self.downsample = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate stego image"""
        # Encoder path
        encoder_features = []
        
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
            x = self.downsample(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, (decoder, upsampler) in enumerate(zip(self.decoders, self.upsamplers)):
            x = upsampler(x)
            
            # Skip connection
            skip_feat = encoder_features[-(i + 2)]  # -2 because we skip the last encoder
            x = torch.cat([x, skip_feat], dim=1)
            
            x = decoder(x)
        
        # Output
        output = self.output(x)
        
        return output


class MessageProcessor(nn.Module):
    """Process binary messages into features"""
    
    def __init__(self, message_length: int, feature_dim: int):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(message_length, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Attention for message
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """Process messages"""
        # MLP processing
        features = self.mlp(messages)
        
        # Self-attention
        features = features.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.attention(features, features, features)
        features = attended.squeeze(1)
        
        return features


class MessageInjector(nn.Module):
    """Inject message features into image features"""
    
    def __init__(self, feature_channels: int, message_channels: int):
        super().__init__()
        
        # Spatial expansion of message
        self.spatial_expand = nn.Sequential(
            nn.Linear(message_channels, feature_channels * 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels * 16, feature_channels * 64)
        )
        
        # Feature modulation
        self.modulator = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 3, 1, 1),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        )
        
        # Adaptive gating
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image_features: torch.Tensor, 
                message_features: torch.Tensor) -> torch.Tensor:
        """Inject messages into features"""
        B, C, H, W = image_features.shape
        
        # Expand message to spatial dimensions
        message_spatial = self.spatial_expand(message_features)
        message_spatial = message_spatial.view(B, C, 8, 8)
        message_spatial = F.interpolate(message_spatial, size=(H, W), 
                                      mode='bilinear', align_corners=False)
        
        # Combine features
        combined = torch.cat([image_features, message_spatial], dim=1)
        modulated = self.modulator(combined)
        
        # Adaptive gating
        gate = self.gate(image_features)
        
        # Apply gated injection
        output = image_features + gate * modulated
        
        return output