#!/usr/bin/env python3
"""
Perceptual Loss Functions
Loss functions based on perceptual similarity
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional
import logging

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS not available. Install with: pip install lpips")


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, layers: List[str] = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'],
                 weights: Optional[List[float]] = None, normalize: bool = True):
        """
        Initialize perceptual loss
        
        Args:
            layers: VGG layers to use for feature extraction
            weights: Weights for each layer
            normalize: Whether to normalize inputs
        """
        super().__init__()
        
        self.layers = layers
        self.weights = weights or [1.0] * len(layers)
        self.normalize = normalize
        
        # Load pretrained VGG
        vgg = models.vgg16(pretrained=True)
        
        # Create feature extractor
        self.features = self._create_feature_extractor(vgg)
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Loss function
        self.criterion = nn.L1Loss()
    
    def _create_feature_extractor(self, vgg) -> nn.ModuleDict:
        """Create feature extraction layers"""
        layers = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14,
            'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21,
            'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28
        }
        
        features = nn.ModuleDict()
        for name in self.layers:
            if name in layers:
                layer_idx = layers[name]
                features[name] = nn.Sequential(*list(vgg.features.children())[:layer_idx+1])
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss"""
        # Normalize if needed
        if self.normalize:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        
        # Denormalize from [-1, 1] to [0, 1] if needed
        if pred.min() < 0:
            pred = (pred + 1) / 2
            target = (target + 1) / 2
        
        total_loss = 0
        
        # Extract features and calculate loss
        for layer_name, weight in zip(self.layers, self.weights):
            if layer_name in self.features:
                pred_features = self.features[layer_name](pred)
                target_features = self.features[layer_name](target)
                
                layer_loss = self.criterion(pred_features, target_features)
                total_loss += weight * layer_loss
        
        return total_loss


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss"""
    
    def __init__(self, net: str = 'alex', use_gpu: bool = True):
        """
        Initialize LPIPS loss
        
        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
            use_gpu: Whether to use GPU
        """
        super().__init__()
        
        if not LPIPS_AVAILABLE:
            raise ImportError("LPIPS is not installed. Install with: pip install lpips")
        
        self.loss_fn = lpips.LPIPS(net=net, verbose=False)
        
        if use_gpu and torch.cuda.is_available():
            self.loss_fn = self.loss_fn.cuda()
        
        # Freeze LPIPS network
        for param in self.loss_fn.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate LPIPS loss"""
        # LPIPS expects inputs in [-1, 1]
        if pred.min() >= 0:
            pred = pred * 2 - 1
            target = target * 2 - 1
        
        return self.loss_fn(pred, target).mean()


class StyleLoss(nn.Module):
    """Style loss using Gram matrices"""
    
    def __init__(self, layers: List[str] = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']):
        super().__init__()
        
        self.layers = layers
        
        # Load VGG for feature extraction
        vgg = models.vgg19(pretrained=True)
        self.features = self._create_feature_extractor(vgg)
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.criterion = nn.MSELoss()
    
    def _create_feature_extractor(self, vgg) -> nn.ModuleDict:
        """Create feature extraction layers"""
        layers = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34
        }
        
        features = nn.ModuleDict()
        for name in self.layers:
            if name in layers:
                layer_idx = layers[name]
                features[name] = nn.Sequential(*list(vgg.features.children())[:layer_idx+1])
        
        return features
    
    def _gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate style loss"""
        total_loss = 0
        
        for layer_name in self.layers:
            if layer_name in self.features:
                pred_features = self.features[layer_name](pred)
                target_features = self.features[layer_name](target)
                
                pred_gram = self._gram_matrix(pred_features)
                target_gram = self._gram_matrix(target_features)
                
                layer_loss = self.criterion(pred_gram, target_gram)
                total_loss += layer_loss
        
        return total_loss


class TextureLoss(nn.Module):
    """Texture similarity loss"""
    
    def __init__(self, patch_size: int = 11, stride: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        
        # Create Gabor filters for texture analysis
        self.gabor_filters = self._create_gabor_filters()
    
    def _create_gabor_filters(self) -> nn.Module:
        """Create Gabor filter bank"""
        filters = []
        ksize = 31
        
        for theta in [0, 45, 90, 135]:
            for frequency in [0.05, 0.1, 0.2]:
                # Create Gabor kernel (simplified)
                kernel = torch.zeros(1, 1, ksize, ksize)
                # Initialize with simple pattern (full implementation would use cv2.getGaborKernel)
                filters.append(kernel)
        
        filters = torch.cat(filters, dim=0)
        
        conv = nn.Conv2d(3, len(filters), ksize, padding=ksize//2, bias=False)
        conv.weight.data = filters.repeat(1, 3, 1, 1)
        conv.requires_grad = False
        
        return conv
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate texture loss"""
        # Extract texture features
        pred_texture = self.gabor_filters(pred)
        target_texture = self.gabor_filters(target)
        
        # Calculate statistics
        pred_mean = pred_texture.mean(dim=[2, 3])
        pred_std = pred_texture.std(dim=[2, 3])
        
        target_mean = target_texture.mean(dim=[2, 3])
        target_std = target_texture.std(dim=[2, 3])
        
        # Loss is difference in statistics
        mean_loss = F.l1_loss(pred_mean, target_mean)
        std_loss = F.l1_loss(pred_std, target_std)
        
        return mean_loss + std_loss


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1)"""
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Charbonnier loss"""
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()


class EdgeLoss(nn.Module):
    """Edge-aware loss"""
    
    def __init__(self):
        super().__init__()
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate edge loss"""
        # Convert to grayscale if needed
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
            target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
            pred_gray = pred_gray.unsqueeze(1)
            target_gray = target_gray.unsqueeze(1)
        else:
            pred_gray = pred
            target_gray = target
        
        # Calculate edges
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        
        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return F.l1_loss(pred_edges, target_edges)