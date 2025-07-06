#!/usr/bin/env python3
"""
Residual Blocks for UNet Architecture
Various residual block implementations with attention support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .conv_block import ConvBlock


class ResidualBlock(nn.Module):
    """Basic residual block with optional attention"""
    
    def __init__(self, channels: int, use_attention: bool = False, 
                 attention_type: str = 'self', context_dim: Optional[int] = None,
                 norm_type: str = "group", activation: str = "gelu", dropout: float = 0.0):
        """
        Initialize residual block
        
        Args:
            channels: Number of channels
            use_attention: Whether to use attention
            attention_type: Type of attention ('self', 'cross')
            context_dim: Context dimension for cross attention
            norm_type: Normalization type
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        
        # Two convolution layers
        self.conv1 = ConvBlock(channels, channels, norm_type=norm_type, 
                              activation=activation, dropout=dropout)
        self.conv2 = ConvBlock(channels, channels, norm_type=norm_type, 
                              activation="none", dropout=dropout)
        
        # Final activation
        self.final_activation = self._get_activation(activation)
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            if attention_type == 'self':
                from ..attention.self_attention import SelfAttention
                self.attention = SelfAttention(channels)
            elif attention_type == 'cross':
                from ..attention.cross_attention import CrossAttention
                self.attention = CrossAttention(channels, context_dim or channels)
            else:
                self.attention = None
                self.use_attention = False
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            context: Context tensor for cross attention
            
        Returns:
            Output tensor [B, C, H, W]
        """
        residual = x
        
        # Two convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Apply attention if enabled
        if self.use_attention and self.attention is not None:
            if hasattr(self.attention, 'to_q'):  # Cross attention
                x = self.attention(x, context)
            else:  # Self attention
                x = self.attention(x)
        
        # Residual connection and final activation
        x = x + residual
        x = self.final_activation(x)
        
        return x


class BottleneckBlock(nn.Module):
    """Bottleneck residual block for efficiency"""
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int,
                 stride: int = 1, expansion: int = 4, norm_type: str = "group"):
        """
        Initialize bottleneck block
        
        Args:
            in_channels: Input channels
            bottleneck_channels: Bottleneck channels
            out_channels: Output channels
            stride: Convolution stride
            expansion: Expansion ratio
            norm_type: Normalization type
        """
        super().__init__()
        
        # 1x1 compress
        self.conv1 = ConvBlock(in_channels, bottleneck_channels, 1, 1, 0, norm_type=norm_type)
        
        # 3x3 process
        self.conv2 = ConvBlock(bottleneck_channels, bottleneck_channels, 3, stride, 1, norm_type=norm_type)
        
        # 1x1 expand
        self.conv3 = ConvBlock(bottleneck_channels, out_channels, 1, 1, 0, norm_type=norm_type, activation="none")
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, 1, stride, 0, norm_type=norm_type, activation="none")
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x += residual
        x = self.relu(x)
        
        return x


class PreActivationBlock(nn.Module):
    """Pre-activation residual block"""
    
    def __init__(self, channels: int, norm_type: str = "group", activation: str = "gelu"):
        """
        Initialize pre-activation block
        
        Args:
            channels: Number of channels
            norm_type: Normalization type
            activation: Activation function
        """
        super().__init__()
        
        # Pre-activation structure: Norm -> Act -> Conv
        self.norm1 = self._get_norm_layer(norm_type, channels)
        self.act1 = self._get_activation(activation)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        
        self.norm2 = self._get_norm_layer(norm_type, channels)
        self.act2 = self._get_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        
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
        residual = x
        
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        
        return x + residual


class DenseBlock(nn.Module):
    """Dense block with concatenated features"""
    
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int,
                 norm_type: str = "group", activation: str = "gelu"):
        """
        Initialize dense block
        
        Args:
            in_channels: Input channels
            growth_rate: Growth rate (channels added per layer)
            num_layers: Number of dense layers
            norm_type: Normalization type
            activation: Activation function
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                ConvBlock(layer_in_channels, growth_rate, 3, 1, 1,
                         norm_type=norm_type, activation=activation)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with feature concatenation"""
        features = [x]
        
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        
        return torch.cat(features, dim=1)


class SEResidualBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation"""
    
    def __init__(self, channels: int, reduction: int = 16, norm_type: str = "group"):
        """
        Initialize SE residual block
        
        Args:
            channels: Number of channels
            reduction: SE reduction ratio
            norm_type: Normalization type
        """
        super().__init__()
        
        # Convolution layers
        self.conv1 = ConvBlock(channels, channels, norm_type=norm_type)
        self.conv2 = ConvBlock(channels, channels, norm_type=norm_type, activation="none")
        
        # Squeeze-and-Excitation
        from .conv_block import SqueezeExciteBlock
        self.se = SqueezeExciteBlock(channels, reduction)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        residual = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        
        x += residual
        x = self.relu(x)
        
        return x


class PyramidBlock(nn.Module):
    """Pyramid residual block with multi-scale features"""
    
    def __init__(self, channels: int, scales: list = [1, 2, 4], norm_type: str = "group"):
        """
        Initialize pyramid block
        
        Args:
            channels: Number of channels
            scales: List of pooling scales
            norm_type: Normalization type
        """
        super().__init__()
        
        self.scales = scales
        self.channels_per_scale = channels // len(scales)
        
        # Multi-scale branches
        self.branches = nn.ModuleList([
            ConvBlock(channels, self.channels_per_scale, 3, 1, 1, norm_type=norm_type)
            for _ in scales
        ])
        
        # Feature fusion
        self.fusion = ConvBlock(channels, channels, 1, 1, 0, norm_type=norm_type, activation="none")
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, H, W = x.shape
        residual = x
        
        # Multi-scale processing
        features = []
        for scale, branch in zip(self.scales, self.branches):
            if scale == 1:
                feat = branch(x)
            else:
                # Downsample -> process -> upsample
                x_down = F.avg_pool2d(x, scale)
                feat_down = branch(x_down)
                feat = F.interpolate(feat_down, size=(H, W), mode='bilinear', align_corners=False)
            
            features.append(feat)
        
        # Concatenate and fuse
        fused = torch.cat(features, dim=1)
        output = self.fusion(fused)
        
        # Residual connection
        output += residual
        output = self.relu(output)
        
        return output


class NonLocalBlock(nn.Module):
    """Non-local neural network block for long-range dependencies"""
    
    def __init__(self, channels: int, reduction: int = 2):
        """
        Initialize non-local block
        
        Args:
            channels: Number of channels
            reduction: Channel reduction ratio
        """
        super().__init__()
        
        self.channels = channels
        self.inter_channels = channels // reduction
        
        # Query, Key, Value projections
        self.g = nn.Conv2d(channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(channels, self.inter_channels, 1)
        
        # Output projection
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        # Initialize W to zero for residual learning
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, H, W = x.shape
        
        # Generate g, theta, phi
        g_x = self.g(x).view(B, self.inter_channels, -1)  # [B, C//r, HW]
        g_x = g_x.permute(0, 2, 1)  # [B, HW, C//r]
        
        theta_x = self.theta(x).view(B, self.inter_channels, -1)  # [B, C//r, HW]
        theta_x = theta_x.permute(0, 2, 1)  # [B, HW, C//r]
        
        phi_x = self.phi(x).view(B, self.inter_channels, -1)  # [B, C//r, HW]
        
        # Compute attention
        f = torch.matmul(theta_x, phi_x)  # [B, HW, HW]
        f_div_C = F.softmax(f, dim=-1)
        
        # Apply attention
        y = torch.matmul(f_div_C, g_x)  # [B, HW, C//r]
        y = y.permute(0, 2, 1).contiguous()  # [B, C//r, HW]
        y = y.view(B, self.inter_channels, H, W)  # [B, C//r, H, W]
        
        # Output projection
        W_y = self.W(y)
        
        # Residual connection
        return W_y + x


class ChannelShuffleBlock(nn.Module):
    """Channel shuffle block for efficient feature mixing"""
    
    def __init__(self, channels: int, groups: int = 4):
        """
        Initialize channel shuffle block
        
        Args:
            channels: Number of channels
            groups: Number of groups for shuffling
        """
        super().__init__()
        
        self.channels = channels
        self.groups = groups
        assert channels % groups == 0
        
        channels_per_group = channels // groups
        
        # Grouped convolution
        self.conv1 = ConvBlock(channels, channels, 1, 1, 0)
        
        # Depthwise convolution
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Pointwise convolution
        self.conv3 = ConvBlock(channels, channels, 1, 1, 0, activation="none")
        
        self.relu = nn.ReLU(inplace=True)
        
    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Channel shuffle operation"""
        B, C, H, W = x.shape
        channels_per_group = C // self.groups
        
        # Reshape and permute
        x = x.view(B, self.groups, channels_per_group, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        residual = x
        
        # 1x1 conv
        x = self.conv1(x)
        
        # Channel shuffle
        x = self.channel_shuffle(x)
        
        # Depthwise conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 1x1 conv
        x = self.conv3(x)
        
        # Residual connection
        x += residual
        x = self.relu(x)
        
        return x