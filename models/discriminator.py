#!/usr/bin/env python3
"""
Discriminator Network for Adversarial Training
Multi-scale discriminator for steganography system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

from .blocks.conv_block import ConvBlock
from .attention.self_attention import SelfAttention, SpatialAttention


class Discriminator(nn.Module):
    """Multi-scale discriminator for adversarial training"""
    
    def __init__(self, config):
        """
        Initialize discriminator
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.channels = config.channels
        self.base_channels = config.unet_base_channels
        
        # Multi-scale discriminators
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(self.channels, self.base_channels),
            PatchDiscriminator(self.channels, self.base_channels // 2),
            PatchDiscriminator(self.channels, self.base_channels // 4)
        ])
        
        # Global discriminator
        self.global_discriminator = GlobalDiscriminator(
            self.channels, self.base_channels
        )
        
        # Feature matching layers
        self.feature_matching = FeatureMatching(self.base_channels)
        
        logging.info("Multi-scale discriminator initialized")
    
    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Tuple of (outputs, features) for each scale
        """
        outputs = []
        features = []
        
        # Multi-scale processing
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                # Downsample for different scales
                images = F.avg_pool2d(images, 2)
            
            output, feat = disc(images)
            outputs.append(output)
            features.append(feat)
        
        # Global discrimination
        global_output, global_feat = self.global_discriminator(images)
        outputs.append(global_output)
        features.append(global_feat)
        
        return outputs, features


class PatchDiscriminator(nn.Module):
    """Patch-based discriminator"""
    
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        
        # Discriminator layers
        self.layers = nn.ModuleList([
            # Layer 1
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            
            # Layer 2
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            
            # Layer 3
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            
            # Layer 4
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            
            # Output layer
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)
        ])
        
        # Attention layers
        self.attention = SelfAttention(base_channels * 4)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning output and features"""
        features = []
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            features.append(x)
            
            # Apply attention at specific layer
            if i == 2:
                x = self.attention(x)
        
        # Final output
        output = self.layers[-1](x)
        features.append(output)
        
        return output, features


class GlobalDiscriminator(nn.Module):
    """Global image discriminator"""
    
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        
        # Encoder path
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 4, base_channels * 8),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 8, base_channels * 16),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 16, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass"""
        features = []
        
        # Extract features
        encoded = self.encoder(x)
        features.append(encoded)
        
        # Classify
        output = self.classifier(encoded)
        
        return output, features


class SpectralNormDiscriminator(nn.Module):
    """Discriminator with spectral normalization"""
    
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        
        # Apply spectral norm to all layers
        self.layers = nn.ModuleList([
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, base_channels, 4, 2, 1)
            ),
            nn.utils.spectral_norm(
                nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
            ),
            nn.utils.spectral_norm(
                nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
            ),
            nn.utils.spectral_norm(
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)
            ),
            nn.utils.spectral_norm(
                nn.Conv2d(base_channels * 8, 1, 4, 1, 0)
            )
        ])
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x


class FeatureMatching(nn.Module):
    """Feature matching for improved training stability"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Feature processors for different scales
        self.processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim * (2 ** i), 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            for i in range(4)
        ])
        
    def forward(self, real_features: List[torch.Tensor], 
                fake_features: List[torch.Tensor]) -> torch.Tensor:
        """Calculate feature matching loss"""
        loss = 0.0
        
        for i, (real_feat, fake_feat) in enumerate(zip(real_features, fake_features)):
            if i < len(self.processors):
                # Process features
                real_processed = self.processors[i](real_feat)
                fake_processed = self.processors[i](fake_feat)
                
                # L1 loss
                loss += F.l1_loss(fake_processed, real_processed.detach())
        
        return loss / len(real_features)


class ConditionalDiscriminator(nn.Module):
    """Conditional discriminator for guided training"""
    
    def __init__(self, in_channels: int, condition_dim: int, base_channels: int):
        super().__init__()
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 4, base_channels * 16 * 16)
        )
        
        # Main discriminator
        self.discriminator = nn.Sequential(
            ConvBlock(in_channels + base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 4, base_channels * 8),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 8, base_channels * 16),
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            nn.Linear(base_channels * 16, 1)
        )
        
    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning"""
        B, C, H, W = images.shape
        
        # Embed condition
        cond_embed = self.condition_embed(condition)
        cond_embed = cond_embed.view(B, -1, 16, 16)
        
        # Resize to match image
        cond_embed = F.interpolate(cond_embed, size=(H, W), 
                                  mode='bilinear', align_corners=False)
        
        # Concatenate and discriminate
        x = torch.cat([images, cond_embed], dim=1)
        output = self.discriminator(x)
        
        return output


class ProgressiveDiscriminator(nn.Module):
    """Progressive discriminator that grows during training"""
    
    def __init__(self, in_channels: int, base_channels: int, max_layers: int = 5):
        super().__init__()
        
        self.max_layers = max_layers
        self.current_layer = 1
        
        # Build layers progressively
        self.from_rgb_layers = nn.ModuleList([
            nn.Conv2d(in_channels, base_channels * (2 ** i), 1)
            for i in range(max_layers)
        ])
        
        self.progression_layers = nn.ModuleList()
        for i in range(max_layers):
            if i == 0:
                layer = nn.Sequential(
                    ConvBlock(base_channels, base_channels),
                    nn.Conv2d(base_channels, 1, 4, 1, 0)
                )
            else:
                layer = nn.Sequential(
                    ConvBlock(base_channels * (2 ** i), base_channels * (2 ** i)),
                    ConvBlock(base_channels * (2 ** i), base_channels * (2 ** (i-1))),
                    nn.AvgPool2d(2)
                )
            self.progression_layers.append(layer)
        
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward pass with progressive growing"""
        # Start from current layer
        x = self.from_rgb_layers[self.current_layer - 1](x)
        
        # Apply progression layers
        for i in range(self.current_layer - 1, -1, -1):
            x = self.progression_layers[i](x)
        
        return x
    
    def grow(self):
        """Grow discriminator by one layer"""
        if self.current_layer < self.max_layers:
            self.current_layer += 1
            logging.info(f"Discriminator grown to {self.current_layer} layers")