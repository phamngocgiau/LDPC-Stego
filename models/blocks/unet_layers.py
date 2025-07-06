#!/usr/bin/env python3
"""
UNet Layers for Steganography System
Complete UNet architecture with 23 layers and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .conv_block import ConvBlock
from .residual_block import ResidualBlock


class UNet23Layers(nn.Module):
    """23-layer UNet with attention mechanisms"""
    
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64,
                 attention_layers: Optional[List[int]] = None, attention_type: str = 'self',
                 context_dim: Optional[int] = None, norm_type: str = "group"):
        """
        Initialize 23-layer UNet
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            base_channels: Base number of channels
            attention_layers: List of layer indices to add attention
            attention_type: Type of attention ('self', 'cross')
            context_dim: Context dimension for cross attention
            norm_type: Normalization type
        """
        super().__init__()
        
        self.attention_layers = attention_layers or []
        self.attention_type = attention_type
        self.context_dim = context_dim
        
        # Encoder path (11 layers down)
        self.enc1 = self._make_encoder_block(in_channels, base_channels, 1)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2, 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4, 3)
        self.enc4 = self._make_encoder_block(base_channels * 4, base_channels * 8, 4)
        self.enc5 = self._make_encoder_block(base_channels * 8, base_channels * 16, 5)
        
        # Bottleneck (1 layer)
        self.bottleneck = ResidualBlock(
            base_channels * 16,
            use_attention=(6 in self.attention_layers),
            attention_type=attention_type,
            context_dim=context_dim,
            norm_type=norm_type
        )
        
        # Decoder path (11 layers up)
        self.dec5 = self._make_decoder_block(base_channels * 16, base_channels * 8, 7)
        self.dec4 = self._make_decoder_block(base_channels * 16, base_channels * 4, 8)  # Skip connection
        self.dec3 = self._make_decoder_block(base_channels * 8, base_channels * 2, 9)
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels, 10)
        self.dec1 = self._make_decoder_block(base_channels * 2, base_channels, 11)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
        
    def _make_encoder_block(self, in_channels: int, out_channels: int, layer_idx: int):
        """Create encoder block"""
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ResidualBlock(
                out_channels,
                use_attention=(layer_idx in self.attention_layers),
                attention_type=self.attention_type,
                context_dim=self.context_dim
            ),
            nn.MaxPool2d(2)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int, layer_idx: int):
        """Create decoder block"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(in_channels, out_channels),
            ResidualBlock(
                out_channels,
                use_attention=(layer_idx in self.attention_layers),
                attention_type=self.attention_type,
                context_dim=self.context_dim
            )
        )
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            context: Context tensor for cross attention
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        # Encoder path with skip connections
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        
        # Bottleneck
        x = self.bottleneck(x5, context)
        
        # Decoder path with skip connections
        x = self.dec5(x)
        x = self.dec4(torch.cat([x, x4], dim=1))  # Skip connection
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.dec1(torch.cat([x, x1], dim=1))
        
        # Final output
        x = self.final_conv(x)
        return x


class UNetEncoder(nn.Module):
    """UNet encoder path"""
    
    def __init__(self, in_channels: int, base_channels: int = 64, depth: int = 5):
        """
        Initialize UNet encoder
        
        Args:
            in_channels: Input channels
            base_channels: Base number of channels
            depth: Encoder depth
        """
        super().__init__()
        
        self.depth = depth
        self.encoders = nn.ModuleList()
        
        # Build encoder blocks
        channels = in_channels
        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            self.encoders.append(
                nn.Sequential(
                    ConvBlock(channels, out_channels),
                    ConvBlock(out_channels, out_channels),
                    nn.MaxPool2d(2) if i < depth - 1 else nn.Identity()
                )
            )
            channels = out_channels
    
    def forward(self, x: torch.Tensor):
        """Forward pass returning all intermediate features"""
        features = []
        
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        
        return features


class UNetDecoder(nn.Module):
    """UNet decoder path"""
    
    def __init__(self, base_channels: int = 64, depth: int = 5, out_channels: int = 3):
        """
        Initialize UNet decoder
        
        Args:
            base_channels: Base number of channels
            depth: Decoder depth
            out_channels: Output channels
        """
        super().__init__()
        
        self.depth = depth
        self.decoders = nn.ModuleList()
        
        # Build decoder blocks
        for i in range(depth - 1, 0, -1):
            in_channels = base_channels * (2 ** i)
            out_channels_dec = base_channels * (2 ** (i - 1))
            
            self.decoders.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ConvBlock(in_channels + out_channels_dec, out_channels_dec),  # +skip connection
                    ConvBlock(out_channels_dec, out_channels_dec)
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, encoder_features: List[torch.Tensor]):
        """Forward pass with skip connections"""
        x = encoder_features[-1]  # Start from deepest feature
        
        # Decode with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(encoder_features) - 2 - i
            skip_feature = encoder_features[skip_idx]
            
            # Upsample
            x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip_feature], dim=1)
            
            # Process
            x = decoder[1:](x)  # Skip the upsample layer in decoder
        
        # Final output
        return self.final_conv(x)


class NestedUNet(nn.Module):
    """Nested UNet (UNet++) architecture"""
    
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32):
        """
        Initialize Nested UNet
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            base_channels: Base number of channels
        """
        super().__init__()
        
        # Encoder
        self.conv0_0 = self._conv_block(in_channels, base_channels)
        self.conv1_0 = self._conv_block(base_channels, base_channels * 2)
        self.conv2_0 = self._conv_block(base_channels * 2, base_channels * 4)
        self.conv3_0 = self._conv_block(base_channels * 4, base_channels * 8)
        self.conv4_0 = self._conv_block(base_channels * 8, base_channels * 16)
        
        # Nested connections
        self.conv0_1 = self._conv_block(base_channels + base_channels * 2, base_channels)
        self.conv1_1 = self._conv_block(base_channels * 2 + base_channels * 4, base_channels * 2)
        self.conv2_1 = self._conv_block(base_channels * 4 + base_channels * 8, base_channels * 4)
        self.conv3_1 = self._conv_block(base_channels * 8 + base_channels * 16, base_channels * 8)
        
        self.conv0_2 = self._conv_block(base_channels * 2 + base_channels * 2, base_channels)
        self.conv1_2 = self._conv_block(base_channels * 4 + base_channels * 4, base_channels * 2)
        self.conv2_2 = self._conv_block(base_channels * 8 + base_channels * 8, base_channels * 4)
        
        self.conv0_3 = self._conv_block(base_channels * 3 + base_channels * 2, base_channels)
        self.conv1_3 = self._conv_block(base_channels * 6 + base_channels * 4, base_channels * 2)
        
        self.conv0_4 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels)
        
        # Output
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        """Create convolution block"""
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor):
        """Forward pass through nested UNet"""
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output


class AttentionUNet(nn.Module):
    """UNet with attention gates"""
    
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
        """
        Initialize Attention UNet
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            base_channels: Base number of channels
        """
        super().__init__()
        
        # Encoder
        self.encoder1 = self._conv_block(in_channels, base_channels)
        self.encoder2 = self._conv_block(base_channels, base_channels * 2)
        self.encoder3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)
        
        # Attention gates
        self.att4 = AttentionGate(base_channels * 8, base_channels * 16, base_channels * 4)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 8, base_channels * 2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 4, base_channels)
        self.att1 = AttentionGate(base_channels, base_channels * 2, base_channels // 2)
        
        # Decoder
        self.decoder4 = self._conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.decoder3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.decoder2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = self._conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Output
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        """Create convolution block"""
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor):
        """Forward pass with attention"""
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention
        a4 = self.att4(e4, b)
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.decoder4(torch.cat([d4, a4], dim=1))
        
        a3 = self.att3(e3, d4)
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d3, a3], dim=1))
        
        a2 = self.att2(e2, d3)
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.decoder2(torch.cat([d2, a2], dim=1))
        
        a1 = self.att1(e1, d2)
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(torch.cat([d1, a1], dim=1))
        
        return self.final(d1)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Initialize attention gate
        
        Args:
            F_g: Gating signal channels
            F_l: Feature map channels
            F_int: Intermediate channels
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor):
        """Forward pass"""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class MultiScaleUNet(nn.Module):
    """Multi-scale UNet for capturing features at different scales"""
    
    def __init__(self, in_channels: int, out_channels: int, scales: List[int] = [1, 2, 4]):
        """
        Initialize Multi-scale UNet
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            scales: List of scales for multi-scale processing
        """
        super().__init__()
        
        self.scales = scales
        
        # Multiple UNets for different scales
        self.unets = nn.ModuleList([
            UNet23Layers(in_channels, out_channels, base_channels=64 // scale)
            for scale in scales
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            ConvBlock(out_channels * len(scales), out_channels * 2),
            ConvBlock(out_channels * 2, out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
    def forward(self, x: torch.Tensor):
        """Multi-scale forward pass"""
        B, C, H, W = x.shape
        outputs = []
        
        for scale, unet in zip(self.scales, self.unets):
            if scale == 1:
                # Original scale
                out = unet(x)
            else:
                # Downsample, process, upsample
                x_down = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
                out_down = unet(x_down)
                out = F.interpolate(out_down, size=(H, W), mode='bilinear', align_corners=False)
            
            outputs.append(out)
        
        # Concatenate and fuse
        fused = torch.cat(outputs, dim=1)
        final_output = self.fusion(fused)
        
        return final_output