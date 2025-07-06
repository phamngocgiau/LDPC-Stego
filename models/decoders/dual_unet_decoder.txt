#!/usr/bin/env python3
"""
Dual UNet Decoder
Dual UNet architecture for message extraction without LDPC awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import logging

from ..blocks.conv_block import ConvBlock
from ..blocks.residual_block import ResidualBlock
from ..blocks.unet_layers import UNetEncoder, UNetDecoder
from ..attention.self_attention import SelfAttention
from ..attention.cross_attention import CrossAttention


class DualUNetDecoder(nn.Module):
    """Dual UNet decoder for message extraction"""
    
    def __init__(self, config):
        """
        Initialize dual UNet decoder
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.channels = config.channels
        self.message_length = config.message_length
        self.base_channels = config.unet_base_channels
        
        # First UNet - Feature extraction
        self.feature_extractor = FeatureExtractorUNet(
            in_channels=self.channels,
            base_channels=self.base_channels,
            depth=config.unet_depth
        )
        
        # Second UNet - Message extraction
        self.message_extractor = MessageExtractorUNet(
            in_channels=self.base_channels * 2,
            base_channels=self.base_channels,
            depth=config.unet_depth - 1
        )
        
        # Message decoder
        self.message_decoder = MessageDecoder(
            feature_channels=self.base_channels,
            message_length=self.message_length
        )
        
        # Attention refinement
        self.attention_refiner = AttentionRefiner(
            channels=self.base_channels
        )
        
        logging.info("Dual UNet Decoder initialized")
    
    def forward(self, stego_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            stego_images: Stego images [B, C, H, W]
            
        Returns:
            Extracted messages [B, message_length]
        """
        # Extract features from stego images
        stego_features, multi_scale_features = self.feature_extractor(stego_images)
        
        # Refine features with attention
        refined_features = self.attention_refiner(stego_features, multi_scale_features)
        
        # Extract message features
        message_features = self.message_extractor(refined_features)
        
        # Decode messages
        messages = self.message_decoder(message_features)
        
        return messages


class FeatureExtractorUNet(nn.Module):
    """UNet for feature extraction from stego images"""
    
    def __init__(self, in_channels: int, base_channels: int, depth: int = 5):
        super().__init__()
        
        # Encoder
        self.encoders = nn.ModuleList()
        channels = in_channels
        
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(
                nn.Sequential(
                    ConvBlock(channels, out_ch),
                    ResidualBlock(out_ch, use_attention=(i >= 2)),
                    ConvBlock(out_ch, out_ch)
                )
            )
            channels = out_ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels, use_attention=True),
            ConvBlock(channels, channels)
        )
        
        # Decoder
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
                    ConvBlock(out_ch * 2, out_ch),
                    ResidualBlock(out_ch, use_attention=(i >= 2)),
                    ConvBlock(out_ch, out_ch)
                )
            )
            channels = out_ch
        
        # Output projection
        self.output = nn.Conv2d(base_channels, base_channels * 2, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract features"""
        # Encoder path
        encoder_features = []
        
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        multi_scale_features = []
        
        for i, (decoder, upsampler) in enumerate(zip(self.decoders, self.upsamplers)):
            x = upsampler(x)
            
            # Skip connection
            skip_feat = encoder_features[-(i + 2)]
            x = torch.cat([x, skip_feat], dim=1)
            
            x = decoder(x)
            multi_scale_features.append(x)
        
        # Output
        output = self.output(x)
        
        return output, multi_scale_features


class MessageExtractorUNet(nn.Module):
    """UNet for message extraction"""
    
    def __init__(self, in_channels: int, base_channels: int, depth: int = 4):
        super().__init__()
        
        # Encoder path
        self.encoders = nn.ModuleList()
        channels = in_channels
        
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(
                nn.Sequential(
                    ConvBlock(channels, out_ch),
                    ResidualBlock(out_ch),
                    nn.MaxPool2d(2)
                )
            )
            channels = out_ch
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            ConvBlock(channels, channels),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 2, base_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract message features"""
        # Encode through multiple scales
        for encoder in self.encoders:
            x = encoder(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Project to feature space
        features = self.feature_proj(x)
        
        return features


class MessageDecoder(nn.Module):
    """Decode messages from features"""
    
    def __init__(self, feature_channels: int, message_length: int):
        super().__init__()
        
        # Main decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_channels, feature_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_channels * 4, feature_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels * 2, message_length)
        )
        
        # Residual path
        self.residual = nn.Linear(feature_channels, message_length)
        
        # Output activation
        self.output_activation = nn.Tanh()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode messages"""
        # Main decoding
        main_output = self.decoder(features)
        
        # Residual connection
        residual_output = self.residual(features)
        
        # Combine
        output = main_output + 0.1 * residual_output
        
        # Apply activation
        output = self.output_activation(output)
        
        return output


class AttentionRefiner(nn.Module):
    """Refine features using attention mechanisms"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Self-attention
        self.self_attention = SelfAttention(channels * 2)
        
        # Cross-attention for multi-scale features
        self.cross_attentions = nn.ModuleList([
            CrossAttention(channels * 2, channels * (2 ** i))
            for i in range(4)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2 * 5, channels * 4, 1),
            nn.GroupNorm(8, channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels * 2, 1)
        )
    
    def forward(self, main_features: torch.Tensor, 
                multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """Refine features"""
        # Self-attention on main features
        attended_main = self.self_attention(main_features)
        
        # Cross-attention with multi-scale features
        cross_attended = [attended_main]
        
        for i, (cross_attn, scale_feat) in enumerate(
            zip(self.cross_attentions, multi_scale_features[:4])
        ):
            # Resize if needed
            if scale_feat.shape[2:] != main_features.shape[2:]:
                scale_feat = F.interpolate(
                    scale_feat, size=main_features.shape[2:],
                    mode='bilinear', align_corners=False
                )
            
            attended = cross_attn(main_features, scale_feat)
            cross_attended.append(attended)
        
        # Fuse all attended features
        fused = torch.cat(cross_attended, dim=1)
        refined = self.fusion(fused)
        
        return refined