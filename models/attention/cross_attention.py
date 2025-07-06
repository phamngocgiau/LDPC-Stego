#!/usr/bin/env python3
"""
Cross-Attention Mechanism for Decoder
Multi-head cross-attention between query and context features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttention(nn.Module):
    """Cross-attention mechanism for decoder"""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize cross-attention module
        
        Args:
            query_dim: Query dimension
            context_dim: Context dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert query_dim % num_heads == 0, f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        
        # Query, Key, Value projections
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(dropout)
        
        # Output projection
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention forward pass
        
        Args:
            x: Query tensor [B, C, H, W]
            context: Context tensor [B, seq_len, context_dim] or [B, context_dim, H_c, W_c]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape query to sequence format
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Handle context tensor format
        if len(context.shape) == 4:  # [B, context_dim, H_c, W_c]
            context = context.view(B, context.shape[1], -1).transpose(1, 2)  # [B, H_c*W_c, context_dim]
        elif len(context.shape) == 3:  # [B, seq_len, context_dim]
            pass  # Already in correct format
        else:
            raise ValueError(f"Unsupported context shape: {context.shape}")
        
        # Generate Q, K, V
        q = self.to_q(x_flat).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
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
        
        # Reshape back to spatial format
        return x_out.transpose(1, 2).view(B, C, H, W)


class AdaptiveCrossAttention(nn.Module):
    """Adaptive cross-attention with learnable temperature"""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, temperature_init: float = 1.0):
        """
        Initialize adaptive cross-attention
        
        Args:
            query_dim: Query dimension
            context_dim: Context dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature_init: Initial temperature value
        """
        super().__init__()
        self.cross_attn = CrossAttention(query_dim, context_dim, num_heads, dropout)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * temperature_init)
        
        # Context gating
        self.context_gate = nn.Sequential(
            nn.Linear(context_dim, context_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Adaptive cross-attention forward pass
        
        Args:
            x: Query tensor [B, C, H, W]
            context: Context tensor [B, seq_len, context_dim] or [B, context_dim, H_c, W_c]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Apply temperature scaling to attention
        original_scale = self.cross_attn.scale
        self.cross_attn.scale = original_scale / self.temperature
        
        # Compute context gate
        if len(context.shape) == 4:
            context_pooled = F.adaptive_avg_pool2d(context, 1).squeeze(-1).squeeze(-1)
        else:
            context_pooled = context.mean(dim=1)
        
        gate = self.context_gate(context_pooled).unsqueeze(-1).unsqueeze(-1)
        
        # Apply cross-attention
        output = self.cross_attn(x, context)
        
        # Apply gating
        output = gate * output + (1 - gate) * x
        
        # Restore original scale
        self.cross_attn.scale = original_scale
        
        return output


class MultiLevelCrossAttention(nn.Module):
    """Multi-level cross-attention for hierarchical features"""
    
    def __init__(self, query_dim: int, context_dims: list, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-level cross-attention
        
        Args:
            query_dim: Query dimension
            context_dims: List of context dimensions for each level
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.cross_attentions = nn.ModuleList([
            CrossAttention(query_dim, context_dim, num_heads, dropout)
            for context_dim in context_dims
        ])
        
        # Feature fusion
        self.fusion = nn.Conv2d(query_dim * (len(context_dims) + 1), query_dim, 1)
        self.norm = nn.BatchNorm2d(query_dim)
        
    def forward(self, x: torch.Tensor, contexts: list) -> torch.Tensor:
        """
        Multi-level cross-attention forward pass
        
        Args:
            x: Query tensor [B, C, H, W]
            contexts: List of context tensors
            
        Returns:
            Output tensor [B, C, H, W]
        """
        features = [x]  # Include original query
        
        for cross_attn, context in zip(self.cross_attentions, contexts):
            feat = cross_attn(x, context)
            features.append(feat)
        
        # Concatenate and fuse features
        fused = torch.cat(features, dim=1)
        output = self.fusion(fused)
        output = self.norm(output)
        
        return output


class DeformableCrossAttention(nn.Module):
    """Deformable cross-attention with learnable sampling offsets"""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, 
                 num_points: int = 4, dropout: float = 0.1):
        """
        Initialize deformable cross-attention
        
        Args:
            query_dim: Query dimension
            context_dim: Context dimension
            num_heads: Number of attention heads
            num_points: Number of sampling points per head
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = query_dim // num_heads
        
        # Offset and weight prediction
        self.offset_net = nn.Conv2d(query_dim, num_heads * num_points * 2, 3, padding=1)
        self.weight_net = nn.Conv2d(query_dim, num_heads * num_points, 3, padding=1)
        
        # Attention computation
        self.cross_attn = CrossAttention(query_dim, context_dim, num_heads, dropout)
        
        # Initialize offsets to zero
        nn.init.zeros_(self.offset_net.weight)
        nn.init.zeros_(self.offset_net.bias)
        nn.init.zeros_(self.weight_net.weight)
        nn.init.zeros_(self.weight_net.bias)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Deformable cross-attention forward pass
        
        Args:
            x: Query tensor [B, C, H, W]
            context: Context tensor [B, context_dim, H_c, W_c]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Predict sampling offsets and weights
        offsets = self.offset_net(x)  # [B, num_heads*num_points*2, H, W]
        weights = self.weight_net(x)  # [B, num_heads*num_points, H, W]
        weights = torch.softmax(weights, dim=1)
        
        # Reshape offsets and weights
        offsets = offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        weights = weights.view(B, self.num_heads, self.num_points, H, W)
        
        # Sample features from context using offsets
        # (This is a simplified version - full implementation would require custom CUDA ops)
        sampled_context = self._sample_features(context, offsets, weights)
        
        # Apply cross-attention
        output = self.cross_attn(x, sampled_context)
        
        return output
    
    def _sample_features(self, context: torch.Tensor, offsets: torch.Tensor, 
                        weights: torch.Tensor) -> torch.Tensor:
        """
        Sample features from context using deformable offsets
        
        Args:
            context: Context tensor [B, context_dim, H_c, W_c]
            offsets: Sampling offsets [B, num_heads, num_points, 2, H, W]
            weights: Sampling weights [B, num_heads, num_points, H, W]
            
        Returns:
            Sampled context features
        """
        # Simplified implementation - in practice would use bilinear sampling
        # For now, just return average pooled context
        B, context_dim, H_c, W_c = context.shape
        pooled_context = F.adaptive_avg_pool2d(context, (offsets.size(-2), offsets.size(-1)))
        
        # Expand to match the expected format
        pooled_context = pooled_context.unsqueeze(2).expand(-1, -1, self.num_points, -1, -1)
        
        # Apply weights and sum over points
        weighted_context = (pooled_context * weights.unsqueeze(1)).sum(dim=2)
        
        return weighted_context


class ContextualCrossAttention(nn.Module):
    """Contextual cross-attention with position encoding"""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, use_pos_encoding: bool = True):
        """
        Initialize contextual cross-attention
        
        Args:
            query_dim: Query dimension
            context_dim: Context dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_pos_encoding: Whether to use positional encoding
        """
        super().__init__()
        self.cross_attn = CrossAttention(query_dim, context_dim, num_heads, dropout)
        self.use_pos_encoding = use_pos_encoding
        
        if use_pos_encoding:
            # Learnable positional embeddings
            self.pos_embedding_q = nn.Parameter(torch.randn(1, query_dim, 1, 1))
            self.pos_embedding_c = nn.Parameter(torch.randn(1, context_dim, 1, 1))
        
        # Context normalization
        self.context_norm = nn.LayerNorm(context_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Contextual cross-attention forward pass
        
        Args:
            x: Query tensor [B, C, H, W]
            context: Context tensor [B, context_dim, H_c, W_c]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Add positional encoding
        if self.use_pos_encoding:
            x = x + self.pos_embedding_q
            context = context + self.pos_embedding_c
        
        # Normalize context
        if len(context.shape) == 4:
            B, C_ctx, H_ctx, W_ctx = context.shape
            context_flat = context.view(B, C_ctx, -1).transpose(1, 2)
            context_flat = self.context_norm(context_flat)
            context = context_flat.transpose(1, 2).view(B, C_ctx, H_ctx, W_ctx)
        
        # Apply cross-attention
        output = self.cross_attn(x, context)
        
        return output