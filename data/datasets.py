#!/usr/bin/env python3
"""
Steganography Dataset
Dataset classes for loading images and generating messages
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from torchvision import transforms
import random


class SteganographyDataset(Dataset):
    """Dataset for steganography training"""
    
    def __init__(self, 
                 image_dir: Union[str, Path],
                 image_size: int = 256,
                 message_length: int = 1024,
                 transform: Optional[transforms.Compose] = None,
                 augment: bool = True,
                 cache_images: bool = False):
        """
        Initialize dataset
        
        Args:
            image_dir: Directory containing images
            image_size: Size to resize images to
            message_length: Length of binary messages
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
            cache_images: Whether to cache images in memory
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.message_length = message_length
        self.augment = augment
        self.cache_images = cache_images
        
        # Get image paths
        self.image_paths = self._get_image_paths()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # Cache for images
        self.image_cache = {} if cache_images else None
        
        logging.info(f"Initialized dataset with {len(self.image_paths)} images")
    
    def _get_image_paths(self) -> List[Path]:
        """Get all image paths from directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in valid_extensions:
            image_paths.extend(self.image_dir.glob(f'*{ext}'))
            image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_paths)
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transform"""
        transform_list = []
        
        # Resize and crop
        transform_list.extend([
            transforms.Resize(int(self.image_size * 1.1)),
            transforms.RandomCrop(self.image_size) if self.augment else transforms.CenterCrop(self.image_size),
        ])
        
        # Data augmentation
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        
        # Convert to tensor and normalize to [-1, 1]
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.cache_images:
                self.image_cache[idx] = image
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Generate random binary message
        message = torch.randint(0, 2, (self.message_length,), dtype=torch.float32)
        
        return {
            'cover': image_tensor,
            'message': message,
            'index': idx
        }


class PairedSteganographyDataset(Dataset):
    """Dataset with paired cover and stego images for testing"""
    
    def __init__(self,
                 cover_dir: Union[str, Path],
                 stego_dir: Union[str, Path],
                 message_file: Optional[Union[str, Path]] = None,
                 image_size: int = 256,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize paired dataset
        
        Args:
            cover_dir: Directory containing cover images
            stego_dir: Directory containing stego images
            message_file: Optional file containing messages
            image_size: Size to resize images to
            transform: Optional transform to apply
        """
        self.cover_dir = Path(cover_dir)
        self.stego_dir = Path(stego_dir)
        self.image_size = image_size
        
        # Get paired image paths
        self.image_pairs = self._get_paired_images()
        
        # Load messages if provided
        self.messages = None
        if message_file:
            self.messages = self._load_messages(message_file)
        
        # Setup transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        logging.info(f"Initialized paired dataset with {len(self.image_pairs)} pairs")
    
    def _get_paired_images(self) -> List[Tuple[Path, Path]]:
        """Get paired cover and stego images"""
        cover_images = {img.name: img for img in self.cover_dir.glob('*') 
                       if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}}
        
        pairs = []
        for stego_path in self.stego_dir.glob('*'):
            if stego_path.name in cover_images:
                pairs.append((cover_images[stego_path.name], stego_path))
        
        return sorted(pairs)
    
    def _load_messages(self, message_file: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """Load messages from file"""
        # Implementation depends on message file format
        # For now, return empty dict
        return {}
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cover_path, stego_path = self.image_pairs[idx]
        
        # Load images
        cover_image = Image.open(cover_path).convert('RGB')
        stego_image = Image.open(stego_path).convert('RGB')
        
        # Apply transforms
        cover_tensor = self.transform(cover_image)
        stego_tensor = self.transform(stego_image)
        
        result = {
            'cover': cover_tensor,
            'stego': stego_tensor,
            'index': idx
        }
        
        # Add message if available
        if self.messages and cover_path.name in self.messages:
            result['message'] = self.messages[cover_path.name]
        
        return result


class SyntheticSteganographyDataset(Dataset):
    """Synthetic dataset for testing without real images"""
    
    def __init__(self,
                 num_samples: int = 1000,
                 image_size: int = 256,
                 channels: int = 3,
                 message_length: int = 1024):
        """
        Initialize synthetic dataset
        
        Args:
            num_samples: Number of synthetic samples
            image_size: Size of synthetic images
            channels: Number of image channels
            message_length: Length of binary messages
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels
        self.message_length = message_length
        
        # Pre-generate synthetic patterns
        self.patterns = self._generate_patterns()
    
    def _generate_patterns(self) -> List[str]:
        """Generate different synthetic patterns"""
        return ['gradient', 'checkerboard', 'circles', 'noise', 'stripes']
    
    def _generate_synthetic_image(self, pattern: str, idx: int) -> torch.Tensor:
        """Generate synthetic image based on pattern"""
        H, W = self.image_size, self.image_size
        
        if pattern == 'gradient':
            # Gradient pattern
            x = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
            y = torch.linspace(-1, 1, H).unsqueeze(1).expand(H, W)
            r = x
            g = y
            b = (x + y) / 2
            
        elif pattern == 'checkerboard':
            # Checkerboard pattern
            size = 32
            x, y = torch.meshgrid(torch.arange(H), torch.arange(W))
            checkerboard = ((x // size + y // size) % 2).float()
            r = g = b = checkerboard * 2 - 1
            
        elif pattern == 'circles':
            # Concentric circles
            x, y = torch.meshgrid(torch.arange(H), torch.arange(W))
            cx, cy = H // 2, W // 2
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            circles = torch.sin(dist * 0.1)
            r = g = b = circles
            
        elif pattern == 'stripes':
            # Stripe pattern
            x = torch.arange(W).float()
            stripes = torch.sin(x * 0.1)
            r = g = b = stripes.unsqueeze(0).expand(H, W)
            
        else:  # noise
            # Random noise
            r = torch.randn(H, W)
            g = torch.randn(H, W)
            b = torch.randn(H, W)
        
        # Stack channels
        image = torch.stack([r, g, b], dim=0)
        
        # Add some variation based on index
        variation = torch.randn_like(image) * 0.1
        image = torch.clamp(image + variation, -1, 1)
        
        return image
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select pattern
        pattern = self.patterns[idx % len(self.patterns)]
        
        # Generate synthetic image
        image = self._generate_synthetic_image(pattern, idx)
        
        # Generate random message
        message = torch.randint(0, 2, (self.message_length,), dtype=torch.float32)
        
        return {
            'cover': image,
            'message': message,
            'index': idx
        }


class DataAugmentation:
    """Advanced data augmentation for steganography"""
    
    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        
        # Geometric augmentations
        self.geometric_aug = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        ])
        
        # Color augmentations
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
        ])
        
        # Blur augmentations
        self.blur_aug = transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Lambda(lambda x: x),  # No blur
        ])
        
        # Noise augmentation
        self.noise_aug = AddGaussianNoise(std=0.02)
    
    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Apply augmentations"""
        if isinstance(image, torch.Tensor):
            # Convert to PIL for some augmentations
            image = transforms.ToPILImage()(image)
        
        # Apply augmentations with probability
        if random.random() > 0.5:
            image = self.geometric_aug(image)
        
        if random.random() > 0.5:
            image = self.color_aug(image)
        
        if random.random() > 0.3:
            image = self.blur_aug(image)
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        # Apply noise
        if random.random() > 0.5:
            image = self.noise_aug(image)
        
        return image


class AddGaussianNoise:
    """Add Gaussian noise to tensor"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, -1, 1)


class MixedDataset(Dataset):
    """Dataset that mixes multiple datasets"""
    
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        """
        Initialize mixed dataset
        
        Args:
            datasets: List of datasets to mix
            weights: Optional weights for each dataset
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Calculate total length
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)
        
        # Create mapping from global index to (dataset_idx, local_idx)
        self._create_index_mapping()
    
    def _create_index_mapping(self):
        """Create mapping from global index to dataset and local index"""
        self.index_mapping = []
        
        for dataset_idx, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            num_samples = int(self.total_length * weight)
            for i in range(num_samples):
                local_idx = i % len(dataset)
                self.index_mapping.append((dataset_idx, local_idx))
        
        # Shuffle mapping
        random.shuffle(self.index_mapping)
    
    def __len__(self) -> int:
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_idx, local_idx = self.index_mapping[idx]
        return self.datasets[dataset_idx][local_idx]