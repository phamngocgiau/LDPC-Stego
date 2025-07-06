#!/usr/bin/env python3
"""
Adversarial Loss Functions
Loss functions for adversarial training (GANs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdversarialLoss(nn.Module):
    """General adversarial loss supporting multiple GAN types"""
    
    def __init__(self, loss_type: str = 'lsgan'):
        """
        Initialize adversarial loss
        
        Args:
            loss_type: Type of GAN loss ('gan', 'lsgan', 'wgan', 'hinge')
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'gan':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None  # Wasserstein loss doesn't need criterion
        elif loss_type == 'hinge':
            self.criterion = None  # Hinge loss doesn't need criterion
        else:
            raise ValueError(f"Unknown adversarial loss type: {loss_type}")
    
    def forward(self, predictions: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Calculate adversarial loss
        
        Args:
            predictions: Discriminator predictions
            is_real: Whether to calculate loss for real or fake samples
            
        Returns:
            Adversarial loss
        """
        if self.loss_type == 'gan':
            target = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
            return self.criterion(predictions, target)
        
        elif self.loss_type == 'lsgan':
            target = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
            return self.criterion(predictions, target)
        
        elif self.loss_type == 'wgan':
            return -predictions.mean() if is_real else predictions.mean()
        
        elif self.loss_type == 'hinge':
            if is_real:
                return F.relu(1.0 - predictions).mean()
            else:
                return F.relu(1.0 + predictions).mean()
    
    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate discriminator loss"""
        real_loss = self.forward(real_pred, True)
        fake_loss = self.forward(fake_pred, False)
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate generator loss"""
        if self.loss_type == 'wgan':
            return -fake_pred.mean()
        elif self.loss_type == 'hinge':
            return -fake_pred.mean()
        else:
            return self.forward(fake_pred, True)


class WassersteinLoss(nn.Module):
    """Wasserstein GAN loss with gradient penalty"""
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate Wasserstein loss"""
        return fake_pred.mean() - real_pred.mean()
    
    def gradient_penalty(self, discriminator, real_data: torch.Tensor,
                        fake_data: torch.Tensor) -> torch.Tensor:
        """Calculate gradient penalty"""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty


class RelativisticLoss(nn.Module):
    """Relativistic average GAN loss"""
    
    def __init__(self, loss_type: str = 'lsgan'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'gan':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type for RaGAN: {loss_type}")
    
    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate relativistic discriminator loss"""
        real_avg = real_pred.mean()
        fake_avg = fake_pred.mean()
        
        real_loss = self.criterion(real_pred - fake_avg, torch.ones_like(real_pred))
        fake_loss = self.criterion(fake_pred - real_avg, torch.zeros_like(fake_pred))
        
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate relativistic generator loss"""
        real_avg = real_pred.mean()
        fake_avg = fake_pred.mean()
        
        real_loss = self.criterion(real_pred - fake_avg, torch.zeros_like(real_pred))
        fake_loss = self.criterion(fake_pred - real_avg, torch.ones_like(fake_pred))
        
        return (real_loss + fake_loss) * 0.5


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for stable GAN training"""
    
    def __init__(self, layers: Optional[list] = None):
        super().__init__()
        self.layers = layers or [-2, -1]  # Use last two layers by default
        self.criterion = nn.L1Loss()
    
    def forward(self, real_features: list, fake_features: list) -> torch.Tensor:
        """Calculate feature matching loss"""
        total_loss = 0
        
        for idx in self.layers:
            if abs(idx) <= len(real_features):
                real_feat = real_features[idx]
                fake_feat = fake_features[idx]
                total_loss += self.criterion(fake_feat, real_feat.detach())
        
        return total_loss / len(self.layers)


class SpectralNormLoss(nn.Module):
    """Loss with spectral normalization awareness"""
    
    def __init__(self, base_loss: str = 'hinge'):
        super().__init__()
        self.base_loss = AdversarialLoss(base_loss)
    
    def forward(self, predictions: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Calculate loss (spectral norm is applied in discriminator)"""
        return self.base_loss(predictions, is_real)


class ConsistencyLoss(nn.Module):
    """Consistency regularization for discriminator"""
    
    def __init__(self, augment_fn=None):
        super().__init__()
        self.augment_fn = augment_fn or self._default_augment
        self.criterion = nn.MSELoss()
    
    def _default_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Default augmentation: add small noise"""
        noise = torch.randn_like(x) * 0.05
        return x + noise
    
    def forward(self, discriminator, real_data: torch.Tensor) -> torch.Tensor:
        """Calculate consistency loss"""
        # Get predictions for original data
        with torch.no_grad():
            pred_orig = discriminator(real_data)
        
        # Get predictions for augmented data
        augmented_data = self.augment_fn(real_data)
        pred_aug = discriminator(augmented_data)
        
        return self.criterion(pred_aug, pred_orig)


class PerceptualAdversarialLoss(nn.Module):
    """Combine perceptual and adversarial losses"""
    
    def __init__(self, perceptual_weight: float = 1.0, adversarial_weight: float = 0.1):
        super().__init__()
        from .perceptual_loss import PerceptualLoss
        
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss('lsgan')
        
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                discriminator_pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate combined loss"""
        # Perceptual loss
        p_loss = self.perceptual_loss(pred, target) * self.perceptual_weight
        
        # Adversarial loss (if discriminator predictions provided)
        if discriminator_pred is not None:
            a_loss = self.adversarial_loss.generator_loss(discriminator_pred) * self.adversarial_weight
            return p_loss + a_loss
        else:
            return p_loss