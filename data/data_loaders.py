#!/usr/bin/env python3
"""
Data Loaders
Data loader utilities for LDPC steganography system
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import numpy as np

from .datasets import SteganographyDataset, SyntheticSteganographyDataset, DataAugmentation
from .transforms import get_train_transforms, get_val_transforms


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    message_length: int = 1024,
    pin_memory: bool = True,
    drop_last: bool = True,
    augment_train: bool = True,
    cache_images: bool = False,
    val_split: float = 0.1,
    debug: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory (optional)
        image_size: Size to resize images to
        batch_size: Batch size
        num_workers: Number of data loading workers
        message_length: Length of binary messages
        pin_memory: Whether to pin memory for GPU
        drop_last: Whether to drop last incomplete batch
        augment_train: Whether to augment training data
        cache_images: Whether to cache images in memory
        val_split: Validation split ratio if val_dir not provided
        debug: Debug mode with reduced dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Get transforms
    train_transform = get_train_transforms(image_size) if augment_train else get_val_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Create datasets
    if debug or not Path(train_dir).exists():
        logging.warning("Using synthetic dataset for debugging")
        train_dataset = SyntheticSteganographyDataset(
            num_samples=100 if debug else 1000,
            image_size=image_size,
            message_length=message_length
        )
        val_dataset = SyntheticSteganographyDataset(
            num_samples=20 if debug else 200,
            image_size=image_size,
            message_length=message_length
        )
    else:
        # Training dataset
        train_dataset = SteganographyDataset(
            image_dir=train_dir,
            image_size=image_size,
            message_length=message_length,
            transform=train_transform,
            augment=augment_train,
            cache_images=cache_images
        )
        
        # Validation dataset
        if Path(val_dir).exists() and len(list(Path(val_dir).glob('*'))) > 0:
            val_dataset = SteganographyDataset(
                image_dir=val_dir,
                image_size=image_size,
                message_length=message_length,
                transform=val_transform,
                augment=False,
                cache_images=cache_images
            )
        else:
            # Split training data for validation
            logging.info(f"Splitting training data for validation with ratio {val_split}")
            val_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Limit dataset size in debug mode
    if debug:
        train_dataset = Subset(train_dataset, range(min(100, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(20, len(val_dataset))))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    # Test loader (optional)
    test_loader = None
    if test_dir and Path(test_dir).exists():
        test_dataset = SteganographyDataset(
            image_dir=test_dir,
            image_size=image_size,
            message_length=message_length,
            transform=val_transform,
            augment=False,
            cache_images=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    logging.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader


def create_inference_loader(
    image_dir: str,
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 0
) -> DataLoader:
    """
    Create data loader for inference
    
    Args:
        image_dir: Directory containing images
        image_size: Size to resize images to
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Inference data loader
    """
    
    transform = get_val_transforms(image_size)
    
    dataset = SteganographyDataset(
        image_dir=image_dir,
        image_size=image_size,
        message_length=1,  # Dummy value for inference
        transform=transform,
        augment=False,
        cache_images=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    
    return loader


class InfiniteDataLoader:
    """Data loader that loops infinitely"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter_loader = iter(self.data_loader)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            batch = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.data_loader)
            batch = next(self.iter_loader)
        return batch
    
    def __len__(self):
        return len(self.data_loader)


def collate_fn_with_metadata(batch: list) -> Dict[str, Any]:
    """Custom collate function that preserves metadata"""
    
    # Separate tensors and metadata
    covers = torch.stack([item['cover'] for item in batch])
    messages = torch.stack([item['message'] for item in batch])
    indices = torch.tensor([item['index'] for item in batch])
    
    collated = {
        'cover': covers,
        'message': messages,
        'index': indices
    }
    
    # Add any additional metadata
    if 'metadata' in batch[0]:
        metadata = [item['metadata'] for item in batch]
        collated['metadata'] = metadata
    
    return collated


class BalancedBatchSampler:
    """Sampler that ensures balanced batches"""
    
    def __init__(self, dataset, batch_size: int, num_classes: int = 2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Group indices by class
        self.class_indices = self._group_by_class()
        
    def _group_by_class(self) -> Dict[int, list]:
        """Group dataset indices by class"""
        # This is a placeholder - implement based on your dataset structure
        # For now, randomly assign classes
        import random
        class_indices = {i: [] for i in range(self.num_classes)}
        
        for idx in range(len(self.dataset)):
            class_idx = random.randint(0, self.num_classes - 1)
            class_indices[class_idx].append(idx)
            
        return class_indices
    
    def __iter__(self):
        # Calculate samples per class in each batch
        samples_per_class = self.batch_size // self.num_classes
        
        # Create batches
        while True:
            batch = []
            
            for class_idx in range(self.num_classes):
                # Sample from class
                class_samples = np.random.choice(
                    self.class_indices[class_idx],
                    size=samples_per_class,
                    replace=True
                )
                batch.extend(class_samples)
            
            # Shuffle batch
            np.random.shuffle(batch)
            
            yield batch
            
    def __len__(self):
        return len(self.dataset) // self.batch_size