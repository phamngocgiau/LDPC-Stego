#!/usr/bin/env python3
"""
Combined Loss Functions
Combined loss functions for comprehensive training objectives
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging

from .base_loss import (
    BaseLoss, MessageLoss, ImageQualityLoss, VAELoss,
    DiscriminatorLoss, GeneratorLoss, RobustnessLoss,
    CapacityLoss, GradientPenalty
)
from .ldpc_loss import LDPCConsistencyLoss, SyndromeLoss


class CombinedSteganographyLoss(nn.Module):
    """Combined loss for steganography without LDPC"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.weights = config.loss_weights.copy()
        
        # Initialize individual losses
        self.message_loss = MessageLoss()
        self.image_quality_loss = ImageQualityLoss(
            mse_weight=1.0,
            lpips_weight=1.0,
            ssim_weight=1.0
        )
        self.generator_loss = GeneratorLoss(loss_type='hinge')
        self.capacity_loss = CapacityLoss(target_capacity=0.3)
        
        # Optional losses
        self.vae_loss = VAELoss() if hasattr(config, 'use_vae') and config.use_vae else None
        self.robustness_loss = RobustnessLoss() if hasattr(config, 'use_robustness') and config.use_robustness else None
        
        logging.info(f"Combined loss initialized with weights: {self.weights}")
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Total loss and individual components
        """
        losses = {}
        
        # Message recovery loss
        if 'decoded_messages' in outputs and 'messages' in targets:
            losses['message'] = self.message_loss(
                outputs['decoded_messages'],
                targets['messages']
            )
        
        # Image quality loss
        if 'stego_images' in outputs and 'cover_images' in targets:
            losses['image_quality'] = self.image_quality_loss(
                outputs['stego_images'],
                targets['cover_images']
            )
            
            # Capacity loss
            losses['capacity'] = self.capacity_loss(
                outputs['stego_images'],
                targets['cover_images']
            )
        
        # Adversarial loss
        if 'discriminator_pred' in outputs:
            losses['adversarial'] = self.generator_loss(
                outputs['discriminator_pred']
            )
        
        # VAE loss
        if self.vae_loss and all(k in outputs for k in ['recovered_images', 'mu', 'logvar']):
            losses['vae'] = self.vae_loss(
                outputs['recovered_images'],
                targets['cover_images'],
                outputs['mu'],
                outputs['logvar']
            )
        
        # Robustness loss
        if self.robustness_loss and 'attacked_decoded' in outputs:
            losses['robustness'] = self.robustness_loss(
                outputs['decoded_messages'],
                outputs['attacked_decoded'],
                targets['messages']
            )
        
        # Calculate total loss
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
        for name, loss in losses.items():
            weight = self.weights.get(name, 1.0)
            total_loss = total_loss + weight * loss
        
        # Convert to float for logging
        loss_dict = {name: float(loss.item()) for name, loss in losses.items()}
        loss_dict['total'] = float(total_loss.item())
        
        return total_loss, loss_dict


class MultiTaskLoss(nn.Module):
    """Multi-task learning loss with uncertainty weighting"""
    
    def __init__(self, task_names: List[str], initial_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.task_names = task_names
        
        # Learnable uncertainty parameters (log variance)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in task_names
        })
        
        # Initial weights (optional)
        self.initial_weights = initial_weights or {task: 1.0 for task in task_names}
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate multi-task loss with uncertainty weighting
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            Total loss and weighted components
        """
        weighted_losses = {}
        total_loss = 0.0
        
        for task in self.task_names:
            if task in losses:
                # Uncertainty weighting: L = 1/(2*sigma^2) * L_task + log(sigma)
                precision = torch.exp(-self.log_vars[task])
                weighted_loss = precision * losses[task] + self.log_vars[task]
                
                # Apply initial weight
                weighted_loss = self.initial_weights[task] * weighted_loss
                
                weighted_losses[task] = weighted_loss
                total_loss = total_loss + weighted_loss
        
        # Convert to float for logging
        loss_dict = {
            f"{task}_weighted": float(loss.item())
            for task, loss in weighted_losses.items()
        }
        
        # Add uncertainty values
        for task in self.task_names:
            if task in self.log_vars:
                loss_dict[f"{task}_uncertainty"] = float(torch.exp(self.log_vars[task]).item())
        
        loss_dict['total'] = float(total_loss.item())
        
        return total_loss, loss_dict


class ProgressiveLoss(nn.Module):
    """Progressive loss that changes during training"""
    
    def __init__(self, loss_schedule: Dict[int, Dict[str, float]]):
        """
        Initialize progressive loss
        
        Args:
            loss_schedule: Dictionary mapping epochs to weight configurations
        """
        super().__init__()
        
        self.loss_schedule = loss_schedule
        self.current_weights = {}
        
        # Get all unique loss names
        self.loss_names = set()
        for weights in loss_schedule.values():
            self.loss_names.update(weights.keys())
        
        # Initialize losses
        self.losses = nn.ModuleDict({
            'message': MessageLoss(),
            'image_quality': ImageQualityLoss(),
            'adversarial': GeneratorLoss(),
            'capacity': CapacityLoss(),
            'robustness': RobustnessLoss()
        })
        
    def update_weights(self, epoch: int):
        """Update weights based on current epoch"""
        # Find the appropriate weight configuration
        epochs = sorted(self.loss_schedule.keys())
        current_config_epoch = 0
        
        for e in epochs:
            if epoch >= e:
                current_config_epoch = e
            else:
                break
        
        self.current_weights = self.loss_schedule[current_config_epoch].copy()
        logging.info(f"Updated loss weights for epoch {epoch}: {self.current_weights}")
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate progressive loss"""
        # Update weights if needed
        self.update_weights(epoch)
        
        losses = {}
        
        # Calculate individual losses
        if 'message' in self.current_weights and self.current_weights['message'] > 0:
            losses['message'] = self.losses['message'](
                outputs.get('decoded_messages'),
                targets.get('messages')
            )
        
        if 'image_quality' in self.current_weights and self.current_weights['image_quality'] > 0:
            losses['image_quality'] = self.losses['image_quality'](
                outputs.get('stego_images'),
                targets.get('cover_images')
            )
        
        # Calculate weighted total
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
        for name, loss in losses.items():
            weight = self.current_weights.get(name, 0.0)
            if weight > 0:
                total_loss = total_loss + weight * loss
        
        # Convert to float
        loss_dict = {name: float(loss.item()) for name, loss in losses.items()}
        loss_dict['total'] = float(total_loss.item())
        
        return total_loss, loss_dict


class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights based on gradient magnitudes"""
    
    def __init__(self, loss_names: List[str], alpha: float = 0.9):
        super().__init__()
        
        self.loss_names = loss_names
        self.alpha = alpha  # Exponential moving average factor
        
        # Initialize gradient magnitudes
        self.grad_magnitudes = {name: 1.0 for name in loss_names}
        self.loss_weights = {name: 1.0 for name in loss_names}
        
        # Individual losses
        self.losses = nn.ModuleDict({
            'message': MessageLoss(),
            'image_quality': ImageQualityLoss(),
            'adversarial': GeneratorLoss(),
            'capacity': CapacityLoss()
        })
    
    def update_weights(self, gradients: Dict[str, float]):
        """Update weights based on gradient magnitudes"""
        # Update exponential moving average of gradients
        for name in self.loss_names:
            if name in gradients:
                self.grad_magnitudes[name] = (
                    self.alpha * self.grad_magnitudes[name] +
                    (1 - self.alpha) * gradients[name]
                )
        
        # Calculate inverse gradient weights
        total_inv_grad = sum(1.0 / (g + 1e-8) for g in self.grad_magnitudes.values())
        
        for name in self.loss_names:
            self.loss_weights[name] = (1.0 / (self.grad_magnitudes[name] + 1e-8)) / total_inv_grad
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                return_individual: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Calculate adaptive loss"""
        individual_losses = {}
        
        # Calculate individual losses
        for name in self.loss_names:
            if name in self.losses and name in self.loss_weights:
                if name == 'message':
                    loss = self.losses[name](
                        outputs.get('decoded_messages'),
                        targets.get('messages')
                    )
                elif name == 'image_quality':
                    loss = self.losses[name](
                        outputs.get('stego_images'),
                        targets.get('cover_images')
                    )
                # Add other losses as needed
                
                if return_individual:
                    loss.retain_grad()  # Keep gradients for weight update
                
                individual_losses[name] = loss
        
        # Calculate weighted total
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
        for name, loss in individual_losses.items():
            weight = self.loss_weights.get(name, 1.0)
            total_loss = total_loss + weight * loss
        
        # Prepare output
        loss_dict = {
            f"{name}_loss": float(loss.item())
            for name, loss in individual_losses.items()
        }
        loss_dict.update({
            f"{name}_weight": self.loss_weights[name]
            for name in self.loss_names
        })
        loss_dict['total'] = float(total_loss.item())
        
        if return_individual:
            loss_dict['individual_losses'] = individual_losses
        
        return total_loss, loss_dict