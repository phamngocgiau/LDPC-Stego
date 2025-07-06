#!/usr/bin/env python3
"""
Utility Helper Functions
Common helper functions for LDPC steganography system
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
import yaml
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_directories(base_dir: Union[str, Path]) -> Dict[str, Path]:
    """Setup project directories"""
    base_path = Path(base_dir)
    
    directories = {
        'checkpoints': base_path / 'checkpoints',
        'logs': base_path / 'logs',
        'figures': base_path / 'figures',
        'results': base_path / 'results',
        'tensorboard': base_path / 'tensorboard',
        'samples': base_path / 'samples'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return directories


def calculate_output_padding(input_size: int, kernel_size: int, 
                           stride: int, padding: int) -> int:
    """Calculate output padding for transposed convolution"""
    return input_size - ((input_size - kernel_size + 2 * padding) // stride + 1)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image"""
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Move to CPU and convert to numpy
    image = tensor.detach().cpu().numpy()
    
    # Convert from CHW to HWC
    if image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    
    # Convert from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    
    # Remove single channel dimension
    if image.shape[2] == 1:
        image = image.squeeze(2)
    
    return image


def image_to_tensor(image: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Convert numpy image to tensor"""
    # Ensure float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Ensure 3D
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    
    # Convert from HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    
    # Convert from [0, 1] to [-1, 1]
    image = image * 2 - 1
    
    # Convert to tensor
    tensor = torch.from_numpy(image).float()
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor.to(device)


def save_config(config: Any, filepath: Union[str, Path]):
    """Save configuration to file"""
    filepath = Path(filepath)
    
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = vars(config)
    
    if filepath.suffix == '.yaml':
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.yaml':
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return config_dict


def create_timestamp() -> str:
    """Create timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: float, precision: int = 3) -> str:
    """Format number for display"""
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def compute_gradient_penalty(discriminator: torch.nn.Module,
                           real_samples: torch.Tensor,
                           fake_samples: torch.Tensor,
                           device: str = 'cuda') -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    
    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Get discriminator output for interpolated samples
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
    
    return gradient_penalty


def generate_binary_message(batch_size: int, message_length: int,
                          device: str = 'cpu') -> torch.Tensor:
    """Generate random binary messages"""
    return torch.randint(0, 2, (batch_size, message_length), 
                        dtype=torch.float32, device=device)


def calculate_ber(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate bit error rate"""
    # Ensure binary values
    pred_binary = (predicted > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # Calculate errors
    errors = (pred_binary != target_binary).float()
    ber = errors.mean().item()
    
    return ber


def calculate_message_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate message-level accuracy (all bits correct)"""
    batch_size = target.size(0)
    
    # Ensure binary values
    pred_binary = (predicted > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # Check if all bits in each message are correct
    correct = (pred_binary == target_binary).view(batch_size, -1).all(dim=1)
    accuracy = correct.float().mean().item()
    
    return accuracy


def plot_metrics(metrics_dict: Dict[str, List[float]], save_path: Optional[Path] = None):
    """Plot training metrics"""
    num_metrics = len(metrics_dict)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        ax.plot(values)
        ax.set_title(metric_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_comparison_grid(images_dict: Dict[str, torch.Tensor], 
                          titles: Optional[List[str]] = None,
                          save_path: Optional[Path] = None):
    """Create grid of image comparisons"""
    num_images = len(images_dict)
    batch_size = next(iter(images_dict.values())).size(0)
    
    # Limit to first 4 images in batch
    batch_size = min(batch_size, 4)
    
    fig, axes = plt.subplots(batch_size, num_images, 
                            figsize=(4 * num_images, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for col_idx, (name, images) in enumerate(images_dict.items()):
        for row_idx in range(batch_size):
            ax = axes[row_idx, col_idx]
            
            # Convert tensor to image
            image = tensor_to_image(images[row_idx])
            
            # Display image
            if image.ndim == 2:
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image)
            
            # Set title for first row
            if row_idx == 0:
                title = titles[col_idx] if titles else name
                ax.set_title(title)
            
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_confusion_matrix(predicted: torch.Tensor, target: torch.Tensor,
                           num_classes: int = 2) -> np.ndarray:
    """Compute confusion matrix for binary classification"""
    # Ensure binary values
    pred_binary = (predicted > 0.5).long()
    target_binary = (target > 0.5).long()
    
    # Flatten
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))
    
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = ['0', '1'],
                         save_path: Optional[Path] = None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()