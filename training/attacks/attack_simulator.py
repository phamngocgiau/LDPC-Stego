#!/usr/bin/env python3
"""
Attack Simulator
Simulates various attacks on stego images for robustness training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Union, Optional, Dict, Any
import numpy as np
import logging


class AttackSimulator:
    """Simulates various attacks on stego images"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize attack simulator
        
        Args:
            device: PyTorch device
        """
        self.device = device
        
        # Initialize different attack methods
        self.jpeg_compression = JPEGCompression()
        self.gaussian_noise = GaussianNoise()
        self.gaussian_blur = GaussianBlur()
        self.median_filter = MedianFilter()
        self.resize_attack = ResizeAttack()
        self.crop_attack = CropAttack()
        self.dropout_attack = DropoutAttack()
        self.combined_attack = CombinedAttack()
        
        # Attack mapping
        self.attacks = {
            'none': self._no_attack,
            'jpeg': self._jpeg_attack,
            'noise': self._noise_attack,
            'blur': self._blur_attack,
            'median': self._median_attack,
            'resize': self._resize_attack,
            'crop': self._crop_attack,
            'dropout': self._dropout_attack,
            'combined': self._combined_attack,
            'adaptive': self._adaptive_attack
        }
        
        logging.info(f"Attack Simulator initialized with {len(self.attacks)} attack types")
    
    def apply_attack(self, images: torch.Tensor, attack_type: str = 'none', 
                    strength: float = 0.5) -> torch.Tensor:
        """
        Apply specified attack to images
        
        Args:
            images: Input images [B, C, H, W]
            attack_type: Type of attack to apply
            strength: Attack strength (0-1)
            
        Returns:
            Attacked images
        """
        if attack_type not in self.attacks:
            logging.warning(f"Unknown attack type: {attack_type}, using 'none'")
            attack_type = 'none'
        
        return self.attacks[attack_type](images, strength)
    
    def _no_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """No attack - return original images"""
        return images
    
    def _jpeg_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """JPEG compression attack"""
        quality = int(90 - strength * 70)  # Quality from 90 to 20
        return self.jpeg_compression(images, quality)
    
    def _noise_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Gaussian noise attack"""
        std = strength * 0.1  # Standard deviation up to 0.1
        return self.gaussian_noise(images, std)
    
    def _blur_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Gaussian blur attack"""
        kernel_size = int(3 + strength * 8)  # Kernel size from 3 to 11
        sigma = 0.5 + strength * 2.5  # Sigma from 0.5 to 3.0
        return self.gaussian_blur(images, kernel_size, sigma)
    
    def _median_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Median filter attack"""
        kernel_size = int(3 + strength * 4)  # Kernel size from 3 to 7
        return self.median_filter(images, kernel_size)
    
    def _resize_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Resize attack"""
        scale = 1.0 - strength * 0.5  # Scale from 1.0 to 0.5
        return self.resize_attack(images, scale)
    
    def _crop_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Random crop and resize attack"""
        crop_ratio = 1.0 - strength * 0.3  # Crop ratio from 1.0 to 0.7
        return self.crop_attack(images, crop_ratio)
    
    def _dropout_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Dropout attack"""
        dropout_rate = strength * 0.3  # Dropout rate up to 0.3
        return self.dropout_attack(images, dropout_rate)
    
    def _combined_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Combined multiple attacks"""
        return self.combined_attack(images, strength)
    
    def _adaptive_attack(self, images: torch.Tensor, strength: float) -> torch.Tensor:
        """Adaptive attack based on image content"""
        # Analyze image characteristics
        image_stats = self._analyze_images(images)
        
        # Choose attack based on image properties
        if image_stats['high_frequency'] > 0.5:
            # High frequency content - use blur
            return self._blur_attack(images, strength)
        elif image_stats['low_variance'] > 0.5:
            # Low variance - use noise
            return self._noise_attack(images, strength)
        else:
            # Default to JPEG
            return self._jpeg_attack(images, strength)
    
    def _analyze_images(self, images: torch.Tensor) -> Dict[str, float]:
        """Analyze image characteristics"""
        # Calculate frequency content
        fft = torch.fft.fft2(images)
        high_freq = torch.abs(fft[:, :, images.size(2)//4:, images.size(3)//4:]).mean()
        total_freq = torch.abs(fft).mean()
        
        # Calculate variance
        variance = torch.var(images, dim=(2, 3)).mean()
        
        return {
            'high_frequency': float(high_freq / total_freq),
            'low_variance': float(1.0 / (1.0 + variance))
        }


class JPEGCompression(nn.Module):
    """JPEG compression simulation"""
    
    def __init__(self):
        super().__init__()
        
        # DCT basis functions
        self.dct_basis = self._create_dct_basis()
        
    def _create_dct_basis(self, block_size: int = 8) -> torch.Tensor:
        """Create DCT basis functions"""
        basis = torch.zeros(block_size, block_size, block_size, block_size)
        
        for u in range(block_size):
            for v in range(block_size):
                for x in range(block_size):
                    for y in range(block_size):
                        cu = 1.0 if u == 0 else np.sqrt(2)
                        cv = 1.0 if v == 0 else np.sqrt(2)
                        
                        basis[u, v, x, y] = (cu * cv / 4.0) * \
                            np.cos((2*x + 1) * u * np.pi / (2*block_size)) * \
                            np.cos((2*y + 1) * v * np.pi / (2*block_size))
        
        return basis
    
    def forward(self, images: torch.Tensor, quality: int = 50) -> torch.Tensor:
        """Apply JPEG compression"""
        # Simplified JPEG simulation using frequency domain filtering
        B, C, H, W = images.shape
        
        # Convert to frequency domain
        freq = torch.fft.fft2(images)
        
        # Create quality-based mask
        mask = self._create_frequency_mask(H, W, quality).to(images.device)
        
        # Apply mask
        freq_filtered = freq * mask
        
        # Convert back to spatial domain
        compressed = torch.fft.ifft2(freq_filtered).real
        
        # Clamp to valid range
        compressed = torch.clamp(compressed, -1, 1)
        
        return compressed
    
    def _create_frequency_mask(self, H: int, W: int, quality: int) -> torch.Tensor:
        """Create frequency mask based on quality"""
        # Higher quality = more frequencies preserved
        cutoff = quality / 100.0
        
        # Create radial mask
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
        center_y, center_x = H // 2, W // 2
        
        # Calculate normalized distance from center
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = torch.sqrt(center_x**2 + center_y**2)
        norm_dist = dist / max_dist
        
        # Create smooth mask
        mask = torch.exp(-5 * (norm_dist / cutoff)**2)
        mask = torch.fft.ifftshift(mask)
        
        return mask


class GaussianNoise(nn.Module):
    """Additive Gaussian noise"""
    
    def forward(self, images: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = torch.randn_like(images) * std
        noisy_images = images + noise
        return torch.clamp(noisy_images, -1, 1)


class GaussianBlur(nn.Module):
    """Gaussian blur filter"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, images: torch.Tensor, kernel_size: int = 5, 
                sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian blur"""
        # Ensure kernel size is odd
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma).to(images.device)
        
        # Apply convolution
        channels = images.size(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        
        blurred = F.conv2d(images, kernel, padding=kernel_size//2, groups=channels)
        
        return blurred
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-((i-center)**2 + (j-center)**2) / (2*sigma**2))
        
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)


class MedianFilter(nn.Module):
    """Median filter"""
    
    def forward(self, images: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Apply median filter"""
        B, C, H, W = images.shape
        
        # Unfold patches
        patches = F.unfold(images, kernel_size, padding=kernel_size//2)
        patches = patches.view(B, C, kernel_size*kernel_size, H*W)
        
        # Get median
        median_vals, _ = torch.median(patches, dim=2)
        
        # Reshape back
        filtered = median_vals.view(B, C, H, W)
        
        return filtered


class ResizeAttack(nn.Module):
    """Resize attack - downscale and upscale"""
    
    def forward(self, images: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        """Apply resize attack"""
        B, C, H, W = images.shape
        
        # Downscale
        new_size = (int(H * scale), int(W * scale))
        downscaled = F.interpolate(images, size=new_size, mode='bilinear', 
                                  align_corners=False)
        
        # Upscale back
        upscaled = F.interpolate(downscaled, size=(H, W), mode='bilinear', 
                                align_corners=False)
        
        return upscaled


class CropAttack(nn.Module):
    """Random crop and resize attack"""
    
    def forward(self, images: torch.Tensor, crop_ratio: float = 0.8) -> torch.Tensor:
        """Apply crop attack"""
        B, C, H, W = images.shape
        
        # Calculate crop size
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)
        
        # Random crop position
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        
        # Crop
        cropped = images[:, :, top:top+crop_h, left:left+crop_w]
        
        # Resize back
        resized = F.interpolate(cropped, size=(H, W), mode='bilinear', 
                               align_corners=False)
        
        return resized


class DropoutAttack(nn.Module):
    """Dropout attack - randomly drop pixels"""
    
    def forward(self, images: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
        """Apply dropout attack"""
        if dropout_rate == 0:
            return images
        
        # Create dropout mask
        mask = torch.bernoulli(torch.ones_like(images) * (1 - dropout_rate))
        
        # Apply dropout
        dropped = images * mask
        
        return dropped


class CombinedAttack(nn.Module):
    """Combine multiple attacks"""
    
    def __init__(self):
        super().__init__()
        self.noise = GaussianNoise()
        self.blur = GaussianBlur()
        self.jpeg = JPEGCompression()
        
    def forward(self, images: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
        """Apply combined attacks"""
        # Progressive attacks with scaled strength
        attacked = images
        
        # Light noise
        attacked = self.noise(attacked, std=strength * 0.03)
        
        # Light blur
        if strength > 0.3:
            attacked = self.blur(attacked, kernel_size=3, sigma=strength)
        
        # JPEG compression
        if strength > 0.5:
            quality = int(90 - strength * 40)
            attacked = self.jpeg(attacked, quality)
        
        return attacked


class AdversarialAttack(nn.Module):
    """Adversarial attack using gradients"""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        
    def forward(self, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply FGSM attack"""
        images.requires_grad = True
        
        # Forward pass
        output = self.model(images)
        loss = F.mse_loss(output, target)
        
        # Calculate gradients
        loss.backward()
        
        # Create adversarial examples
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        
        # Perturb images
        perturbed = images + self.epsilon * sign_data_grad
        perturbed = torch.clamp(perturbed, -1, 1)
        
        return perturbed.detach()