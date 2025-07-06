#!/usr/bin/env python3
"""
Self-Attention Mechanism for UNet Layers
Multi-head self-attention with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelfAttention(nn.Module):
    """Self-attention mechanism for UNet layers"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize self-attention module
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence format: [B, C, H*W] -> [B, H*W, C]
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        
        # Generate Q, K, V
        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        
        # Output projection
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        # Residual connection with normalization
        x_out = self.norm(x_attn + x_flat)
        
        # Reshape back to spatial format: [B, H*W, C] -> [B, C, H, W]
        return x_out.transpose(1, 2).view(B, C, H, W)


class MultiScaleSelfAttention(nn.Module):
    """Multi-scale self-attention for better feature extraction"""
    
    def __init__(self, dim: int, scales: list = [1, 2, 4], num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-scale self-attention
        
        Args:
            dim: Input dimension
            scales: List of downsampling scales
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            SelfAttention(dim, num_heads, dropout) for _ in scales
        ])
        
        # Feature fusion
        self.fusion = nn.Conv2d(dim * len(scales), dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale attention forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        features = []
        
        for scale, attention in zip(self.scales, self.attentions):
            if scale == 1:
                # Original scale
                feat = attention(x)
            else:
                # Downsample, apply attention, upsample
                x_down = F.avg_pool2d(x, scale)
                feat_down = attention(x_down)
                feat = F.interpolate(feat_down, size=(H, W), mode='bilinear', align_corners=False)
            
            features.append(feat)
        
        # Concatenate and fuse features
        fused = torch.cat(features, dim=1)
        output = self.fusion(fused)
        output = self.norm(output)
        
        # Residual connection
        return output + x


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention
        
        Args:
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial attention forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attention weights [B, 1, H, W]
        """
        # Compute spatial statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and compute attention
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(spatial_input)
        attention = self.sigmoid(attention)
        
        return attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, dim: int, reduction: int = 16):
        """
        Initialize channel attention
        
        Args:
            dim: Input dimension
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Channel attention forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attention weights [B, C, 1, 1]
        """
        # Average and max pooling
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        return attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    
    def __init__(self, dim: int, reduction: int = 16, kernel_size: int = 7):
        """
        Initialize CBAM
        
        Args:
            dim: Input dimension
            reduction: Channel attention reduction ratio
            kernel_size: Spatial attention kernel size
        """
        super().__init__()
        self.channel_attention = ChannelAttention(dim, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CBAM forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Channel attention
        x = x * self.channel_attention(x)
        
        # Spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class EfficientSelfAttention(nn.Module):
    """Memory-efficient self-attention for large feature maps"""
    
    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1, dropout: float = 0.1):
        """
        Initialize efficient self-attention
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            sr_ratio: Spatial reduction ratio for K, V
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        # Query projection
        self.q = nn.Linear(dim, dim, bias=False)
        
        # Key, Value projections with spatial reduction
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient attention forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Query
        q = self.q(x_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Key, Value with spatial reduction
        if self.sr_ratio > 1:
            x_sr = self.sr(x).reshape(B, C, -1).transpose(1, 2)  # [B, H*W/sr^2, C]
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_flat).reshape(B, H * W, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv.unbind(0)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        return x_attn.transpose(1, 2).view(B, C, H, W)