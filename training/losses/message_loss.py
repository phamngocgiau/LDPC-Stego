#!/usr/bin/env python3
"""
Message Loss Functions
Loss functions for message extraction accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MessageLoss(nn.Module):
    """Basic message loss using BCE"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, pred_messages: torch.Tensor, true_messages: torch.Tensor) -> torch.Tensor:
        """
        Calculate message loss
        
        Args:
            pred_messages: Predicted messages (logits or probabilities)
            true_messages: True binary messages
            
        Returns:
            Message loss
        """
        # Ensure messages are in correct range
        if true_messages.max() > 1 or true_messages.min() < 0:
            true_messages = (true_messages > 0.5).float()
        
        # If predictions are already probabilities, convert to logits
        if pred_messages.min() >= 0 and pred_messages.max() <= 1:
            # Avoid log(0) by clamping
            pred_messages = torch.clamp(pred_messages, 1e-7, 1 - 1e-7)
            pred_messages = torch.log(pred_messages / (1 - pred_messages))
        
        return self.bce_loss(pred_messages, true_messages)


class LDPCMessageLoss(nn.Module):
    """Message loss aware of LDPC structure"""
    
    def __init__(self, systematic: bool = True, reduction: str = 'mean'):
        """
        Initialize LDPC message loss
        
        Args:
            systematic: Whether LDPC code is systematic
            reduction: Loss reduction method
        """
        super().__init__()
        self.systematic = systematic
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred_messages: torch.Tensor, true_messages: torch.Tensor,
                message_length: Optional[int] = None) -> torch.Tensor:
        """
        Calculate LDPC-aware message loss
        
        Args:
            pred_messages: Predicted encoded messages
            true_messages: True messages (may be shorter than predictions)
            message_length: Original message length
            
        Returns:
            LDPC-aware message loss
        """
        # Get dimensions
        batch_size = pred_messages.size(0)
        pred_length = pred_messages.size(1)
        true_length = true_messages.size(1)
        
        if message_length is None:
            message_length = true_length
        
        # Ensure binary targets
        if true_messages.max() > 1 or true_messages.min() < 0:
            true_messages = (true_messages > 0.5).float()
        
        # Convert predictions to logits if needed
        if pred_messages.min() >= 0 and pred_messages.max() <= 1:
            pred_messages = torch.clamp(pred_messages, 1e-7, 1 - 1e-7)
            pred_messages = torch.log(pred_messages / (1 - pred_messages))
        
        # Calculate base loss
        if pred_length == true_length:
            # Direct comparison
            loss = self.bce_loss(pred_messages, true_messages)
        else:
            # LDPC encoded case - compare information bits
            if self.systematic and pred_length > true_length:
                # For systematic codes, information bits are first
                pred_info = pred_messages[:, :true_length]
                loss = self.bce_loss(pred_info, true_messages)
            else:
                # Non-systematic or truncated case
                min_length = min(pred_length, true_length)
                loss = self.bce_loss(
                    pred_messages[:, :min_length],
                    true_messages[:, :min_length]
                )
        
        # Weight information bits more heavily
        if self.systematic and pred_length > message_length:
            # Create weight mask
            weights = torch.ones_like(loss)
            weights[:, :message_length] = 2.0  # Double weight for info bits
            loss = loss * weights
        
        # Reduce loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SoftMessageLoss(nn.Module):
    """Soft message loss for continuous predictions"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_messages: torch.Tensor, true_messages: torch.Tensor) -> torch.Tensor:
        """Calculate soft message loss"""
        # Apply temperature scaling
        pred_soft = torch.sigmoid(pred_messages / self.temperature)
        true_soft = true_messages.float()
        
        return self.mse_loss(pred_soft, true_soft)


class HammingLoss(nn.Module):
    """Hamming distance based loss"""
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, pred_messages: torch.Tensor, true_messages: torch.Tensor) -> torch.Tensor:
        """Calculate Hamming loss"""
        # Convert to binary
        pred_binary = (pred_messages > 0).float()
        true_binary = (true_messages > 0.5).float()
        
        # Calculate Hamming distance
        hamming_dist = torch.abs(pred_binary - true_binary).sum(dim=1)
        
        if self.normalize:
            hamming_dist = hamming_dist / true_messages.size(1)
        
        return hamming_dist.mean()


class CorrelationLoss(nn.Module):
    """Correlation-based message loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_messages: torch.Tensor, true_messages: torch.Tensor) -> torch.Tensor:
        """Calculate correlation loss"""
        # Normalize messages
        pred_norm = pred_messages - pred_messages.mean(dim=1, keepdim=True)
        true_norm = true_messages - true_messages.mean(dim=1, keepdim=True)
        
        # Calculate correlation
        correlation = (pred_norm * true_norm).sum(dim=1) / (
            torch.sqrt((pred_norm ** 2).sum(dim=1)) * 
            torch.sqrt((true_norm ** 2).sum(dim=1)) + 1e-8
        )
        
        # Loss is 1 - correlation
        return 1 - correlation.mean()


class SequentialMessageLoss(nn.Module):
    """Loss that considers message structure/dependencies"""
    
    def __init__(self, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        self.conv_loss = nn.L1Loss()
    
    def forward(self, pred_messages: torch.Tensor, true_messages: torch.Tensor) -> torch.Tensor:
        """Calculate sequential message loss"""
        batch_size, message_length = pred_messages.shape
        
        # Basic BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_messages, true_messages)
        
        # Sequential consistency loss using 1D convolution
        if message_length >= self.window_size:
            # Create sliding window kernel
            kernel = torch.ones(1, 1, self.window_size) / self.window_size
            kernel = kernel.to(pred_messages.device)
            
            # Apply convolution
            pred_conv = F.conv1d(
                pred_messages.unsqueeze(1), kernel, padding=self.window_size//2
            )
            true_conv = F.conv1d(
                true_messages.unsqueeze(1), kernel, padding=self.window_size//2
            )
            
            # Consistency loss
            consistency_loss = self.conv_loss(pred_conv, true_conv)
            
            return bce_loss + 0.1 * consistency_loss
        else:
            return bce_loss


class AdaptiveMessageLoss(nn.Module):
    """Adaptive message loss based on prediction confidence"""
    
    def __init__(self, confidence_threshold: float = 0.9):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred_messages: torch.Tensor, true_messages: torch.Tensor) -> torch.Tensor:
        """Calculate adaptive message loss"""
        # Base loss
        loss = self.base_loss(pred_messages, true_messages)
        
        # Calculate prediction confidence
        pred_probs = torch.sigmoid(pred_messages)
        confidence = torch.max(pred_probs, 1 - pred_probs)
        
        # Weight by confidence (focus on uncertain predictions)
        weights = torch.where(
            confidence < self.confidence_threshold,
            torch.ones_like(confidence) * 2.0,  # Higher weight for uncertain
            torch.ones_like(confidence)
        )
        
        weighted_loss = loss * weights
        
        return weighted_loss.mean()