#!/usr/bin/env python3
"""
Base Loss Functions
Basic loss functions for steganography training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import lpips


class BaseLoss(nn.Module):
    """Base class for all loss functions"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction"""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class MessageLoss(BaseLoss):
    """Loss for message recovery"""
    
    def __init__(self, reduction: str = 'mean', weight_decay: float = 0.0):
        super().__init__(reduction)
        self.weight_decay = weight_decay
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate message loss"""
        # Binary cross entropy
        loss = F.binary_cross_entropy_with_logits(
            predicted, target, reduction='none'
        )
        
        # Optional weight decay for bit balance
        if self.weight_decay > 0:
            bit_balance = torch.abs(predicted.mean() - 0.5)
            loss = loss + self.weight_decay * bit_balance
        
        return self._reduce(loss)


class ImageQualityLoss(BaseLoss):
    """Combined image quality loss"""
    
    def __init__(self, mse_weight: float = 1.0, lpips_weight: float = 1.0,
                 ssim_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.ssim_weight = ssim_weight
        
        # LPIPS loss
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        
        # SSIM loss
        self.ssim_fn = SSIMLoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined image quality loss"""
        loss = 0.0
        
        # MSE loss
        if self.mse_weight > 0:
            mse = F.mse_loss(predicted, target, reduction='none')
            loss = loss + self.mse_weight * self._reduce(mse)
        
        # LPIPS loss
        if self.lpips_weight > 0:
            lpips_loss = self.lpips_fn(predicted, target)
            loss = loss + self.lpips_weight * lpips_loss.mean()
        
        # SSIM loss
        if self.ssim_weight > 0:
            ssim_loss = self.ssim_fn(predicted, target)
            loss = loss + self.ssim_weight * ssim_loss
        
        return loss


class SSIMLoss(BaseLoss):
    """Structural Similarity Index loss"""
    
    def __init__(self, window_size: int = 11, channel: int = 3, reduction: str = 'mean'):
        super().__init__(reduction)
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window"""
        def gaussian(window_size, sigma):
            gauss = torch.exp(-(torch.arange(window_size).float() - window_size//2)**2 / (2*sigma**2))
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Calculate SSIM loss"""
        channel = img1.size(1)
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        return self._ssim(img1, img2, window, self.window_size, channel)
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor,
              window_size: int, channel: int) -> torch.Tensor:
        """Calculate SSIM"""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        loss = 1 - ssim_map
        return self._reduce(loss)


class VAELoss(BaseLoss):
    """Variational Autoencoder loss"""
    
    def __init__(self, kl_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.kl_weight = kl_weight
    
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate VAE loss"""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = self._reduce(recon_loss.sum(dim=[1, 2, 3]))
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = self._reduce(kl_loss)
        
        return recon_loss + self.kl_weight * kl_loss


class DiscriminatorLoss(BaseLoss):
    """Discriminator loss for adversarial training"""
    
    def __init__(self, loss_type: str = 'vanilla', reduction: str = 'mean'):
        super().__init__(reduction)
        self.loss_type = loss_type
    
    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate discriminator loss"""
        if self.loss_type == 'vanilla':
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
            return real_loss + fake_loss
        
        elif self.loss_type == 'lsgan':
            real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
            fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            return real_loss + fake_loss
        
        elif self.loss_type == 'hinge':
            real_loss = F.relu(1.0 - real_pred).mean()
            fake_loss = F.relu(1.0 + fake_pred).mean()
            return real_loss + fake_loss
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class GeneratorLoss(BaseLoss):
    """Generator loss for adversarial training"""
    
    def __init__(self, loss_type: str = 'vanilla', reduction: str = 'mean'):
        super().__init__(reduction)
        self.loss_type = loss_type
    
    def forward(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate generator loss"""
        if self.loss_type == 'vanilla':
            return F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )
        
        elif self.loss_type == 'lsgan':
            return F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        
        elif self.loss_type == 'hinge':
            return -fake_pred.mean()
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class RobustnessLoss(BaseLoss):
    """Loss for improving robustness against attacks"""
    
    def __init__(self, consistency_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.consistency_weight = consistency_weight
    
    def forward(self, clean_output: torch.Tensor, attacked_output: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Calculate robustness loss"""
        # Message recovery loss for attacked output
        attack_loss = F.binary_cross_entropy_with_logits(
            attacked_output, target, reduction='none'
        )
        
        # Consistency between clean and attacked outputs
        consistency_loss = F.mse_loss(
            torch.sigmoid(attacked_output),
            torch.sigmoid(clean_output).detach(),
            reduction='none'
        )
        
        total_loss = attack_loss + self.consistency_weight * consistency_loss
        
        return self._reduce(total_loss)


class CapacityLoss(BaseLoss):
    """Loss for controlling embedding capacity"""
    
    def __init__(self, target_capacity: float = 0.5, reduction: str = 'mean'):
        super().__init__(reduction)
        self.target_capacity = target_capacity
    
    def forward(self, stego_images: torch.Tensor, cover_images: torch.Tensor) -> torch.Tensor:
        """Calculate capacity loss"""
        # Calculate embedding strength
        diff = torch.abs(stego_images - cover_images)
        embedding_strength = diff.mean(dim=[1, 2, 3])
        
        # Penalize deviation from target capacity
        capacity_loss = (embedding_strength - self.target_capacity).pow(2)
        
        return self._reduce(capacity_loss)


class GradientPenalty(BaseLoss):
    """Gradient penalty for WGAN-GP"""
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator: nn.Module, real_samples: torch.Tensor,
                fake_samples: torch.Tensor) -> torch.Tensor:
        """Calculate gradient penalty"""
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random weight term for interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        # Get discriminator output
        d_interpolates = discriminator(interpolates)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty