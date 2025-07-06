#!/usr/bin/env python3
"""
Data Transforms
Image transformation utilities for data preprocessing
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
import numpy as np
from typing import Union, Tuple, List


class RandomJPEGCompression:
    """Random JPEG compression augmentation"""
    
    def __init__(self, quality_range: Tuple[int, int] = (30, 95)):
        self.quality_range = quality_range
    
    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """Apply random JPEG compression"""
        quality = random.randint(*self.quality_range)
        
        if isinstance(img, torch.Tensor):
            # Convert to PIL
            img_pil = TF.to_pil_image(img)
        else:
            img_pil = img
        
        # Apply JPEG compression by saving and loading
        import io
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)
        
        # Convert back to original format
        if isinstance(img, torch.Tensor):
            return TF.to_tensor(img_compressed)
        else:
            return img_compressed


class RandomGaussianNoise:
    """Add random Gaussian noise"""
    
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.05)):
        self.std_range = std_range
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor"""
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)


class RandomMedianFilter:
    """Random median filter"""
    
    def __init__(self, kernel_sizes: List[int] = [3, 5]):
        self.kernel_sizes = kernel_sizes
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply median filter"""
        from scipy.ndimage import median_filter
        
        kernel_size = random.choice(self.kernel_sizes)
        
        # Convert to numpy
        img_np = img.numpy()
        
        # Apply median filter to each channel
        filtered = np.zeros_like(img_np)
        for c in range(img_np.shape[0]):
            filtered[c] = median_filter(img_np[c], size=kernel_size)
        
        return torch.from_numpy(filtered)


class RandomDownsample:
    """Random downsampling and upsampling"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 0.9)):
        self.scale_range = scale_range
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply downsampling"""
        scale = random.uniform(*self.scale_range)
        
        # Get original size
        _, h, w = img.shape
        
        # Downsample
        new_h, new_w = int(h * scale), int(w * scale)
        img_down = TF.resize(img, [new_h, new_w], interpolation=Image.BILINEAR)
        
        # Upsample back
        img_up = TF.resize(img_down, [h, w], interpolation=Image.BILINEAR)
        
        return img_up


class ComposedTransform:
    """Compose multiple transforms with probability"""
    
    def __init__(self, transforms_list: List[Tuple[object, float]]):
        """
        Args:
            transforms_list: List of (transform, probability) tuples
        """
        self.transforms_list = transforms_list
    
    def __call__(self, img):
        """Apply transforms based on probability"""
        for transform, prob in self.transforms_list:
            if random.random() < prob:
                img = transform(img)
        return img


def get_train_transforms(image_size: int = 256) -> transforms.Compose:
    """Get training data transforms"""
    
    transform_list = [
        # Basic transforms
        transforms.Resize(int(image_size * 1.1)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Color augmentation
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        ),
        
        # Random grayscale
        transforms.RandomGrayscale(p=0.1),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Custom augmentations
        ComposedTransform([
            (RandomGaussianNoise((0.0, 0.02)), 0.3),
            (RandomJPEGCompression((70, 95)), 0.2),
            (RandomDownsample((0.7, 0.95)), 0.2),
        ]),
        
        # Normalize to [-1, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int = 256) -> transforms.Compose:
    """Get validation/test data transforms"""
    
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    return transforms.Compose(transform_list)


def get_inference_transforms(image_size: int = 256) -> transforms.Compose:
    """Get inference transforms"""
    
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    return transforms.Compose(transform_list)


class AttackTransform:
    """Transform that simulates attacks during training"""
    
    def __init__(self, attack_types: List[str] = ['jpeg', 'noise', 'blur', 'none'],
                 attack_probs: List[float] = [0.25, 0.25, 0.25, 0.25]):
        self.attack_types = attack_types
        self.attack_probs = attack_probs
        
        # Initialize attack transforms
        self.attacks = {
            'jpeg': RandomJPEGCompression((30, 90)),
            'noise': RandomGaussianNoise((0.01, 0.05)),
            'blur': transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            'median': RandomMedianFilter([3, 5]),
            'downsample': RandomDownsample((0.5, 0.9)),
            'none': lambda x: x
        }
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random attack"""
        # Select attack type
        attack_type = np.random.choice(self.attack_types, p=self.attack_probs)
        
        # Apply attack
        if attack_type in self.attacks:
            return self.attacks[attack_type](img)
        else:
            return img


class PairedTransform:
    """Transform that applies same random transform to paired images"""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.params = None
    
    def __call__(self, img1: Image.Image, img2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply same transform to both images"""
        # Set random seed for consistency
        seed = random.randint(0, 2**32 - 1)
        
        # Transform first image
        random.seed(seed)
        torch.manual_seed(seed)
        img1_transformed = self.base_transform(img1)
        
        # Transform second image with same parameters
        random.seed(seed)
        torch.manual_seed(seed)
        img2_transformed = self.base_transform(img2)
        
        return img1_transformed, img2_transformed


class NormalizeInverse:
    """Inverse normalization transform"""
    
    def __init__(self, mean: List[float] = [0.5, 0.5, 0.5], 
                 std: List[float] = [0.5, 0.5, 0.5]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor"""
        return tensor * self.std + self.mean


def denormalize(tensor: torch.Tensor, mean: List[float] = [0.5, 0.5, 0.5],
                std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """Denormalize a tensor"""
    mean_tensor = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    
    return tensor * std_tensor + mean_tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL image"""
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = denormalize(tensor)
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    return TF.to_pil_image(tensor)


def apply_test_time_augmentation(model, image: torch.Tensor, num_augmentations: int = 5) -> torch.Tensor:
    """Apply test-time augmentation for more robust predictions"""
    
    predictions = []
    
    # Original prediction
    with torch.no_grad():
        pred = model(image)
        predictions.append(pred)
    
    # Augmented predictions
    for _ in range(num_augmentations - 1):
        # Random horizontal flip
        if random.random() > 0.5:
            aug_image = TF.hflip(image)
        else:
            aug_image = image.clone()
        
        # Random slight rotation
        angle = random.uniform(-5, 5)
        aug_image = TF.rotate(aug_image, angle)
        
        # Random slight scale
        scale = random.uniform(0.95, 1.05)
        size = aug_image.shape[-2:]
        new_size = [int(s * scale) for s in size]
        aug_image = TF.resize(aug_image, new_size)
        aug_image = TF.center_crop(aug_image, size)
        
        with torch.no_grad():
            pred = model(aug_image)
            predictions.append(pred)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)