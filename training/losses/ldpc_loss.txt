#!/usr/bin/env python3
"""
LDPC Loss Functions
Comprehensive loss functions for LDPC steganography training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import lpips
import logging


class LDPCSteganographyLoss(nn.Module):
    """Complete loss function for LDPC steganography"""
    
    def __init__(self, config):
        """
        Initialize LDPC steganography loss
        
        Args:
            config: Configuration object with loss weights
        """
        super().__init__()
        
        self.config = config
        self.weights = config.loss_weights
        
        # Message recovery loss
        self.message_loss = MessageRecoveryLoss(
            reduction='mean',
            ldpc_aware=True
        )
        
        # Image quality losses
        self.mse_loss = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='alex', verbose=False)
        self.ssim_loss = SSIMLoss()
        
        # Adversarial loss
        self.adversarial_loss = AdversarialLoss(loss_type='hinge')
        
        # Recovery losses
        self.recovery_mse_loss = nn.MSELoss()
        self.kl_loss = KLDivergenceLoss()
        
        # LDPC-specific losses
        self.ldpc_consistency_loss = LDPCConsistencyLoss()
        self.syndrome_loss = SyndromeLoss()
        
        # Attack robustness loss
        self.robustness_loss = AttackRobustnessLoss()
        
        logging.info("LDPC Steganography Loss initialized with weights:")
        for name, weight in self.weights.items():
            logging.info(f"  {name}: {weight}")
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                phase: str = 'generator') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate total loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            phase: Training phase ('generator' or 'discriminator')
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        losses = {}
        
        if phase == 'generator':
            # Message recovery loss
            losses['message'] = self.message_loss(
                outputs['decoded_messages'],
                targets['messages'],
                outputs.get('ldpc_encoded_messages')
            )
            
            # Image quality losses
            losses['mse'] = self.mse_loss(
                outputs['stego_images'],
                targets['cover_images']
            )
            
            losses['lpips'] = self.lpips_loss(
                outputs['stego_images'],
                targets['cover_images']
            ).mean()
            
            losses['ssim'] = self.ssim_loss(
                outputs['stego_images'],
                targets['cover_images']
            )
            
            # Adversarial loss
            if 'discriminator_pred' in outputs:
                losses['adversarial'] = self.adversarial_loss(
                    outputs['discriminator_pred'],
                    is_real=True
                )
            
            # Recovery losses
            if 'recovered_images' in outputs:
                losses['recovery_mse'] = self.recovery_mse_loss(
                    outputs['recovered_images'],
                    targets['cover_images']
                )
                
                if 'mu' in outputs and 'logvar' in outputs:
                    losses['recovery_kl'] = self.kl_loss(
                        outputs['mu'],
                        outputs['logvar']
                    )
            
            # LDPC-specific losses
            if 'decoded_ldpc_soft' in outputs:
                losses['ldpc_consistency'] = self.ldpc_consistency_loss(
                    outputs['decoded_ldpc_soft'],
                    outputs.get('ldpc_encoded_messages')
                )
                
                losses['syndrome'] = self.syndrome_loss(
                    outputs['decoded_ldpc_soft'],
                    targets.get('ldpc_H_matrix')
                )
            
            # Attack robustness loss
            if 'attacked_images' in outputs:
                losses['robustness'] = self.robustness_loss(
                    outputs['decoded_messages'],
                    targets['messages'],
                    outputs['attacked_images'],
                    outputs['stego_images']
                )
            
            # Calculate total loss
            total_loss = sum(
                self.weights.get(name, 1.0) * loss 
                for name, loss in losses.items()
            )
            
        else:  # discriminator phase
            # Real/fake discrimination loss
            real_pred = outputs.get('real_pred', [])
            fake_pred = outputs.get('fake_pred', [])
            
            losses['d_real'] = self.adversarial_loss(real_pred, is_real=True)
            losses['d_fake'] = self.adversarial_loss(fake_pred, is_real=False)
            
            # Gradient penalty
            if 'gradient_penalty' in outputs:
                losses['gradient_penalty'] = outputs['gradient_penalty']
            
            total_loss = sum(losses.values())
        
        # Convert losses to float for logging
        loss_dict = {name: float(loss.item()) for name, loss in losses.items()}
        
        return total_loss, loss_dict


class MessageRecoveryLoss(nn.Module):
    """Loss for message recovery with LDPC awareness"""
    
    def __init__(self, reduction: str = 'mean', ldpc_aware: bool = True):
        super().__init__()
        self.reduction = reduction
        self.ldpc_aware = ldpc_aware
        
    def forward(self, decoded: torch.Tensor, target: torch.Tensor, 
                ldpc_encoded: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate message recovery loss
        
        Args:
            decoded: Decoded messages
            target: Target messages
            ldpc_encoded: LDPC encoded messages (for LDPC-aware loss)
        """
        # Basic BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            decoded, target, reduction=self.reduction
        )
        
        if self.ldpc_aware and ldpc_encoded is not None:
            # Additional loss for LDPC codeword recovery
            ldpc_loss = F.mse_loss(
                torch.sigmoid(decoded[:, :ldpc_encoded.size(1)]),
                ldpc_encoded,
                reduction=self.reduction
            )
            return bce_loss + 0.5 * ldpc_loss
        
        return bce_loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss"""
    
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
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
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training"""
    
    def __init__(self, loss_type: str = 'hinge'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, predictions: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Calculate adversarial loss"""
        if isinstance(predictions, list):
            # Multi-scale discriminator
            losses = []
            for pred in predictions:
                losses.append(self._single_loss(pred, is_real))
            return sum(losses) / len(losses)
        else:
            return self._single_loss(predictions, is_real)
    
    def _single_loss(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Calculate loss for single prediction"""
        if self.loss_type == 'hinge':
            if is_real:
                return F.relu(1 - pred).mean()
            else:
                return F.relu(1 + pred).mean()
        elif self.loss_type == 'vanilla':
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return F.binary_cross_entropy_with_logits(pred, target)
        elif self.loss_type == 'lsgan':
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class KLDivergenceLoss(nn.Module):
    """KL divergence loss for VAE"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence"""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return self.beta * kl.mean()


class LDPCConsistencyLoss(nn.Module):
    """Loss for LDPC codeword consistency"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, soft_codeword: torch.Tensor, 
                target_codeword: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate LDPC consistency loss
        
        Args:
            soft_codeword: Soft LDPC codeword predictions
            target_codeword: Target LDPC codeword (if available)
        """
        # Encourage binary values
        binary_loss = torch.mean(
            torch.min(soft_codeword.abs(), (1 - soft_codeword.abs()).abs())
        )
        
        if target_codeword is not None:
            # Direct supervision
            consistency_loss = F.mse_loss(
                torch.sigmoid(soft_codeword),
                target_codeword
            )
            return binary_loss + consistency_loss
        
        return binary_loss


class SyndromeLoss(nn.Module):
    """Loss based on LDPC syndrome"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, soft_codeword: torch.Tensor, 
                H_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate syndrome-based loss
        
        Args:
            soft_codeword: Soft codeword predictions
            H_matrix: LDPC parity check matrix
        """
        if H_matrix is None:
            return torch.tensor(0.0, device=soft_codeword.device)
        
        # Convert to hard decisions
        hard_codeword = (torch.sigmoid(soft_codeword) > 0.5).float()
        
        # Calculate syndrome
        syndrome = torch.matmul(hard_codeword, H_matrix.T) % 2
        
        # Syndrome should be zero for valid codewords
        syndrome_loss = syndrome.abs().mean()
        
        return syndrome_loss


class AttackRobustnessLoss(nn.Module):
    """Loss for improving robustness against attacks"""
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, decoded_clean: torch.Tensor, decoded_attacked: torch.Tensor,
                clean_images: torch.Tensor, attacked_images: torch.Tensor) -> torch.Tensor:
        """
        Calculate robustness loss
        
        Args:
            decoded_clean: Messages decoded from clean images
            decoded_attacked: Messages decoded from attacked images
            clean_images: Clean stego images
            attacked_images: Attacked stego images
        """
        # Message consistency under attack
        message_consistency = F.mse_loss(
            torch.sigmoid(decoded_attacked),
            torch.sigmoid(decoded_clean).detach()
        )
        
        # Feature consistency
        feature_consistency = F.mse_loss(
            attacked_images,
            clean_images.detach()
        )
        
        return self.alpha * message_consistency + (1 - self.alpha) * feature_consistency


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, layer_weights: Dict[str, float] = None):
        super().__init__()
        
        if layer_weights is None:
            layer_weights = {
                'conv1_2': 0.1,
                'conv2_2': 0.1,
                'conv3_3': 1.0,
                'conv4_3': 1.0,
                'conv5_3': 1.0
            }
        
        self.layer_weights = layer_weights
        
        # Load pretrained VGG
        from torchvision import models
        vgg = models.vgg19(pretrained=True).features
        
        # Extract specific layers
        self.slices = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:18], # conv3_3
            vgg[18:27], # conv4_3
            vgg[27:36]  # conv5_3
        ])
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss"""
        loss = 0.0
        
        x = pred
        y = target
        
        for i, (slice_model, (layer_name, weight)) in enumerate(
            zip(self.slices, self.layer_weights.items())
        ):
            x = slice_model(x)
            y = slice_model(y).detach()
            
            loss += weight * F.mse_loss(x, y)
            
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss"""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss with dynamic weighting"""
    
    def __init__(self, loss_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize combined loss
        
        Args:
            loss_configs: Dictionary of loss configurations
        """
        super().__init__()
        
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.dynamic_weights = {}
        
        for name, config in loss_configs.items():
            loss_class = config['class']
            loss_params = config.get('params', {})
            weight = config.get('weight', 1.0)
            dynamic = config.get('dynamic', False)
            
            self.losses[name] = loss_class(**loss_params)
            self.weights[name] = weight
            self.dynamic_weights[name] = dynamic
            
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate combined loss with dynamic weighting"""
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            # Calculate individual loss
            loss_value = loss_fn(predictions, targets)
            
            # Apply dynamic weighting if enabled
            if self.dynamic_weights[name]:
                weight = self._calculate_dynamic_weight(name, epoch)
            else:
                weight = self.weights[name]
                
            loss_dict[name] = weight * loss_value
            
        total_loss = sum(loss_dict.values())
        
        # Convert to float for logging
        loss_dict = {k: float(v.item()) for k, v in loss_dict.items()}
        
        return total_loss, loss_dict
        
    def _calculate_dynamic_weight(self, loss_name: str, epoch: int) -> float:
        """Calculate dynamic weight based on training progress"""
        base_weight = self.weights[loss_name]
        
        # Example: exponential decay for certain losses
        if loss_name in ['adversarial', 'perceptual']:
            return base_weight * (0.9 ** (epoch // 10))
        
        # Example: increase for message loss over time
        elif loss_name == 'message':
            return base_weight * (1.1 ** (epoch // 20))
            
        return base_weight