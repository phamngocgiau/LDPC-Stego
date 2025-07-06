#!/usr/bin/env python3
"""
Convolution Blocks for UNet Architecture
Basic building blocks with normalization and activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class ConvBlock(nn.Module):
    """Basic convolution block with normalization and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = False,
                 norm_type: str = "group", activation: str = "gelu", dropout: float = 0.0):
        """
        Initialize convolution block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            bias: Whether to use bias in convolution
            norm_type: Normalization type ("batch", "group", "layer", "instance")
            activation: Activation function ("relu", "gelu", "silu", "leaky_relu")
            dropout: Dropout rate
        """
        super().__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # Normalization layer
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "group":
            num_groups = min(32, out_channels // 4) if out_channels >= 4 else 1
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm_type == "layer":
            self.norm = nn.GroupNorm(1, out_channels)  # Layer norm equivalent for 2D
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        # Activation function
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.act = nn.SiLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, out_channels, H', W']
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DepthwiseConvBlock(nn.Module):
    """Depthwise separable convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, expansion: int = 1,
                 norm_type: str = "group", activation: str = "gelu"):
        """
        Initialize depthwise separable convolution block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            expansion: Expansion ratio for intermediate channels
            norm_type: Normalization type
            activation: Activation function
        """
        super().__init__()
        
        hidden_channels = in_channels * expansion
        
        # Pointwise expansion
        self.expand_conv = ConvBlock(
            in_channels, hidden_channels, 1, 1, 0, 
            norm_type=norm_type, activation=activation
        ) if expansion > 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, stride, padding,
            groups=hidden_channels, bias=False
        )
        self.dw_norm = self._get_norm_layer(norm_type, hidden_channels)
        self.dw_act = self._get_activation(activation)
        
        # Pointwise projection
        self.project_conv = ConvBlock(
            hidden_channels, out_channels, 1, 1, 0,
            norm_type=norm_type, activation="none"
        )
        
    def _get_norm_layer(self, norm_type: str, channels: int):
        """Get normalization layer"""
        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        elif norm_type == "group":
            num_groups = min(32, channels // 4) if channels >= 4 else 1
            return nn.GroupNorm(num_groups, channels)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(channels)
        else:
            return nn.Identity()
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Expansion
        if not isinstance(self.expand_conv, nn.Identity):
            x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.dw_norm(x)
        x = self.dw_act(x)
        
        # Projection
        x = self.project_conv(x)
        
        return x


class ConvBlockWithAttention(nn.Module):
    """Convolution block with built-in attention mechanism"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, attention_type: str = "channel"):
        """
        Initialize convolution block with attention
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            attention_type: Type of attention ("channel", "spatial", "cbam")
        """
        super().__init__()
        
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        
        # Attention mechanism
        if attention_type == "channel":
            from ..attention.self_attention import ChannelAttention
            self.attention = ChannelAttention(out_channels)
        elif attention_type == "spatial":
            from ..attention.self_attention import SpatialAttention
            self.attention = SpatialAttention()
        elif attention_type == "cbam":
            from ..attention.self_attention import CBAM
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention"""
        x = self.conv_block(x)
        
        if hasattr(self.attention, '__class__') and self.attention.__class__.__name__ != 'Identity':
            if 'Channel' in self.attention.__class__.__name__:
                x = x * self.attention(x)
            elif 'Spatial' in self.attention.__class__.__name__:
                x = x * self.attention(x)
            else:
                x = self.attention(x)
        
        return x


class DilatedConvBlock(nn.Module):
    """Dilated convolution block for enlarged receptive field"""
    
    def __init__(self, in_channels: int, out_channels: int, dilations: list = [1, 2, 4, 8],
                 norm_type: str = "group", activation: str = "gelu"):
        """
        Initialize dilated convolution block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            dilations: List of dilation rates
            norm_type: Normalization type
            activation: Activation function
        """
        super().__init__()
        
        # Multiple dilated convolutions
        self.dilated_convs = nn.ModuleList([
            ConvBlock(
                in_channels, out_channels // len(dilations), 3, 1, dilation,
                norm_type=norm_type, activation=activation
            )
            for dilation in dilations
        ])
        
        # Feature fusion
        self.fusion = ConvBlock(
            out_channels, out_channels, 1, 1, 0,
            norm_type=norm_type, activation=activation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multiple dilations"""
        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))
        
        # Concatenate and fuse
        fused = torch.cat(features, dim=1)
        output = self.fusion(fused)
        
        return output


class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize SE block
        
        Args:
            channels: Number of channels
            reduction: Reduction ratio
        """
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, H, W = x.shape
        
        # Global average pooling
        y = self.global_pool(x).view(B, C)
        
        # FC layers
        y = self.fc(y).view(B, C, 1, 1)
        
        # Scale input
        return x * y


class FireBlock(nn.Module):
    """Fire block from SqueezeNet architecture"""
    
    def __init__(self, in_channels: int, squeeze_channels: int, 
                 expand1x1_channels: int, expand3x3_channels: int):
        """
        Initialize Fire block
        
        Args:
            in_channels: Input channels
            squeeze_channels: Squeeze layer channels
            expand1x1_channels: 1x1 expand channels
            expand3x3_channels: 3x3 expand channels
        """
        super().__init__()
        
        # Squeeze layer
        self.squeeze = ConvBlock(in_channels, squeeze_channels, 1, 1, 0)
        
        # Expand layers
        self.expand1x1 = ConvBlock(squeeze_channels, expand1x1_channels, 1, 1, 0)
        self.expand3x3 = ConvBlock(squeeze_channels, expand3x3_channels, 3, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.squeeze(x)
        
        # Parallel expand paths
        expand1x1 = self.expand1x1(x)
        expand3x3 = self.expand3x3(x)
        
        # Concatenate expand outputs
        return torch.cat([expand1x1, expand3x3], dim=1)


class InvertedResidualBlock(nn.Module):
    """Inverted residual block from MobileNetV2"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expansion: int = 6, norm_type: str = "group"):
        """
        Initialize inverted residual block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for depthwise conv
            expansion: Expansion ratio
            norm_type: Normalization type
        """
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expansion))
        
        layers = []
        
        # Expansion phase
        if expansion != 1:
            layers.append(ConvBlock(in_channels, hidden_dim, 1, 1, 0, norm_type=norm_type))
        
        # Depthwise phase
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            self._get_norm_layer(norm_type, hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Projection phase
        layers.append(ConvBlock(hidden_dim, out_channels, 1, 1, 0, norm_type=norm_type, activation="none"))
        
        self.conv = nn.Sequential(*layers)
        
    def _get_norm_layer(self, norm_type: str, channels: int):
        """Get normalization layer"""
        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        elif norm_type == "group":
            num_groups = min(32, channels // 4) if channels >= 4 else 1
            return nn.GroupNorm(num_groups, channels)
        else:
            return nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GhostConvBlock(nn.Module):
    """Ghost convolution block for efficient computation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, ratio: int = 2, dw_size: int = 3):
        """
        Initialize Ghost convolution block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Primary convolution kernel size
            stride: Convolution stride
            ratio: Ratio for ghost features
            dw_size: Depthwise convolution size
        """
        super().__init__()
        
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        
        # Primary convolution
        self.primary_conv = ConvBlock(
            in_channels, init_channels, kernel_size, stride, kernel_size//2
        )
        
        # Cheap operation (depthwise conv)
        self.cheap_operation = ConvBlock(
            init_channels, new_channels, dw_size, 1, dw_size//2
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        
        return torch.cat([x1, x2], dim=1)