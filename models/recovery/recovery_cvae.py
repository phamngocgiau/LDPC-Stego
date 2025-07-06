#!/usr/bin/env python3
"""
Recovery CVAE (Conditional Variational Autoencoder)
Network for recovering original images from attacked stego images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

from ..blocks.conv_block import ConvBlock
from ..blocks.residual_block import ResidualBlock
from ..attention.self_attention import SelfAttention, CBAM


class RecoveryCVAE(nn.Module):
    """Conditional VAE for image recovery"""
    
    def __init__(self, config):
        """
        Initialize Recovery CVAE
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.image_size = config.image_size
        self.channels = config.channels
        self.latent_dim = config.latent_dim
        self.base_channels = config.unet_base_channels
        
        # Encoder
        self.encoder = CVAEEncoder(
            in_channels=self.channels * 2,  # Concatenated stego + cover
            latent_dim=self.latent_dim,
            base_channels=self.base_channels
        )
        
        # Decoder
        self.decoder = CVAEDecoder(
            latent_dim=self.latent_dim,
            out_channels=self.channels,
            base_channels=self.base_channels,
            image_size=self.image_size
        )
        
        # Condition processor
        self.condition_processor = ConditionProcessor(
            channels=self.channels,
            feature_dim=self.base_channels
        )
        
        # Quality enhancement network
        self.quality_enhancer = QualityEnhancementNetwork(
            channels=self.channels,
            base_channels=self.base_channels
        )
        
        logging.info(f"Recovery CVAE initialized: latent_dim={self.latent_dim}")
    
    def forward(self, stego_images: torch.Tensor, 
                cover_images: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            stego_images: Attacked stego images [B, C, H, W]
            cover_images: Original cover images for conditioning [B, C, H, W]
            
        Returns:
            Tuple of (recovered_images, mu, logvar)
        """
        # Use stego as condition if cover not provided
        if cover_images is None:
            cover_images = stego_images
        
        # Process condition
        condition_features = self.condition_processor(cover_images)
        
        # Encode to latent space
        mu, logvar = self.encoder(stego_images, cover_images)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode with conditioning
        reconstructed = self.decoder(z, condition_features)
        
        # Enhance quality
        recovered = self.quality_enhancer(reconstructed, stego_images)
        
        return recovered, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def sample(self, num_samples: int, condition: torch.Tensor, 
               device: torch.device) -> torch.Tensor:
        """Sample from the model"""
        # Sample from prior
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Process condition
        condition_features = self.condition_processor(condition)
        
        # Decode
        samples = self.decoder(z, condition_features)
        
        return samples


class CVAEEncoder(nn.Module):
    """CVAE Encoder network"""
    
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            ConvBlock(in_channels, base_channels),
            ResidualBlock(base_channels, use_attention=False),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels, base_channels * 2),
            ResidualBlock(base_channels * 2, use_attention=True),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 2, base_channels * 4),
            ResidualBlock(base_channels * 4, use_attention=False),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 4, base_channels * 8),
            ResidualBlock(base_channels * 8, use_attention=True),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels * 8, base_channels * 16),
            ResidualBlock(base_channels * 16, use_attention=False),
        ])
        
        # Calculate bottleneck size
        self.bottleneck_size = base_channels * 16 * (256 // 16) * (256 // 16)
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.bottleneck_size, latent_dim)
        self.fc_logvar = nn.Linear(self.bottleneck_size, latent_dim)
        
        # Attention for feature selection
        self.attention = CBAM(base_channels * 16)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space"""
        # Concatenate input and condition
        x = torch.cat([x, condition], dim=1)
        
        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Project to latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class CVAEDecoder(nn.Module):
    """CVAE Decoder network"""
    
    def __init__(self, latent_dim: int, out_channels: int, base_channels: int = 64, 
                 image_size: int = 256):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.initial_size = image_size // 16
        
        # Initial projection
        self.fc = nn.Linear(latent_dim, base_channels * 16 * self.initial_size * self.initial_size)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                ResidualBlock(base_channels * 16, use_attention=True),
                ConvBlock(base_channels * 16, base_channels * 8),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            
            # Block 2
            nn.Sequential(
                ResidualBlock(base_channels * 8, use_attention=False),
                ConvBlock(base_channels * 8, base_channels * 4),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            
            # Block 3
            nn.Sequential(
                ResidualBlock(base_channels * 4, use_attention=True),
                ConvBlock(base_channels * 4, base_channels * 2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            
            # Block 4
            nn.Sequential(
                ResidualBlock(base_channels * 2, use_attention=False),
                ConvBlock(base_channels * 2, base_channels),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            
            # Final block
            nn.Sequential(
                ResidualBlock(base_channels, use_attention=True),
                ConvBlock(base_channels, out_channels),
                nn.Tanh()
            )
        ])
        
        # Condition integration layers
        self.condition_integrators = nn.ModuleList([
            nn.Conv2d(base_channels * 16 + base_channels, base_channels * 16, 1),
            nn.Conv2d(base_channels * 8 + base_channels, base_channels * 8, 1),
            nn.Conv2d(base_channels * 4 + base_channels, base_channels * 4, 1),
            nn.Conv2d(base_channels * 2 + base_channels, base_channels * 2, 1),
        ])
        
    def forward(self, z: torch.Tensor, condition_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode from latent space"""
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 16, self.initial_size, self.initial_size)
        
        # Decode with condition integration
        for i, (block, integrator) in enumerate(zip(self.decoder_blocks[:-1], self.condition_integrators)):
            # Get appropriate condition feature
            cond_key = f'level_{i}'
            if cond_key in condition_features:
                cond_feat = condition_features[cond_key]
                # Resize condition feature if needed
                if cond_feat.shape[2:] != x.shape[2:]:
                    cond_feat = F.interpolate(cond_feat, size=x.shape[2:], 
                                            mode='bilinear', align_corners=False)
                # Integrate condition
                x = torch.cat([x, cond_feat], dim=1)
                x = integrator(x)
            
            # Apply decoder block
            x = block(x)
        
        # Final block
        x = self.decoder_blocks[-1](x)
        
        return x


class ConditionProcessor(nn.Module):
    """Process conditioning information"""
    
    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        
        # Multi-scale feature extractors
        self.feature_extractors = nn.ModuleDict({
            'level_0': nn.Sequential(
                ConvBlock(channels, feature_dim),
                ResidualBlock(feature_dim),
                nn.MaxPool2d(16)
            ),
            'level_1': nn.Sequential(
                ConvBlock(channels, feature_dim),
                ResidualBlock(feature_dim),
                nn.MaxPool2d(8)
            ),
            'level_2': nn.Sequential(
                ConvBlock(channels, feature_dim),
                ResidualBlock(feature_dim),
                nn.MaxPool2d(4)
            ),
            'level_3': nn.Sequential(
                ConvBlock(channels, feature_dim),
                ResidualBlock(feature_dim),
                nn.MaxPool2d(2)
            ),
        })
        
    def forward(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale condition features"""
        features = {}
        for name, extractor in self.feature_extractors.items():
            features[name] = extractor(condition)
        return features


class QualityEnhancementNetwork(nn.Module):
    """Enhance quality of recovered images"""
    
    def __init__(self, channels: int, base_channels: int):
        super().__init__()
        
        # Enhancement blocks
        self.enhancement_blocks = nn.ModuleList([
            # Detail enhancement
            nn.Sequential(
                ConvBlock(channels * 2, base_channels),
                ResidualBlock(base_channels, use_attention=True),
                ConvBlock(base_channels, base_channels // 2)
            ),
            
            # Structure preservation
            nn.Sequential(
                ConvBlock(base_channels // 2, base_channels // 2),
                ResidualBlock(base_channels // 2),
                ConvBlock(base_channels // 2, base_channels // 4)
            ),
            
            # Final refinement
            nn.Sequential(
                ConvBlock(base_channels // 4, base_channels // 4),
                ResidualBlock(base_channels // 4, use_attention=True),
                ConvBlock(base_channels // 4, channels)
            )
        ])
        
        # Skip connections
        self.skip_conv = nn.Conv2d(channels * 2, channels, 1)
        
        # Final activation
        self.final_activation = nn.Tanh()
        
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Enhance reconstructed image quality"""
        # Concatenate for processing
        x = torch.cat([reconstructed, original], dim=1)
        
        # Skip connection
        skip = self.skip_conv(x)
        
        # Enhancement pipeline
        for block in self.enhancement_blocks:
            x = block(x)
        
        # Combine with skip
        enhanced = x + skip
        
        # Final activation
        enhanced = self.final_activation(enhanced)
        
        return enhanced


class AdaptiveRecoveryCVAE(RecoveryCVAE):
    """Adaptive recovery that adjusts to attack severity"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Attack severity estimator
        self.severity_estimator = nn.Sequential(
            ConvBlock(self.channels, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # 3 severity levels
            nn.Softmax(dim=1)
        )
        
        # Severity-specific decoders
        self.light_decoder = self.decoder
        self.medium_decoder = CVAEDecoder(
            self.latent_dim, self.channels, self.base_channels, self.image_size
        )
        self.heavy_decoder = CVAEDecoder(
            self.latent_dim, self.channels, self.base_channels * 2, self.image_size
        )
        
    def forward(self, stego_images: torch.Tensor, 
                cover_images: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adaptive forward pass"""
        # Estimate attack severity
        severity_probs = self.severity_estimator(stego_images)
        
        # Process condition
        if cover_images is None:
            cover_images = stego_images
        condition_features = self.condition_processor(cover_images)
        
        # Encode
        mu, logvar = self.encoder(stego_images, cover_images)
        z = self.reparameterize(mu, logvar)
        
        # Decode with appropriate decoder
        light_recovery = self.light_decoder(z, condition_features)
        medium_recovery = self.medium_decoder(z, condition_features)
        heavy_recovery = self.heavy_decoder(z, condition_features)
        
        # Weighted combination based on severity
        severity_probs = severity_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        recovered = (severity_probs[:, 0] * light_recovery +
                    severity_probs[:, 1] * medium_recovery +
                    severity_probs[:, 2] * heavy_recovery)
        
        # Enhance quality
        recovered = self.quality_enhancer(recovered, stego_images)
        
        return recovered, mu, logvar