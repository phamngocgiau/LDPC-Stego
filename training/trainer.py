#!/usr/bin/env python3
"""
Trainer Module
Main training logic for LDPC steganography system
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple, Any, List
import logging
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
from collections import defaultdict
import wandb

from .losses import CombinedLoss
from .metrics import ImageMetrics, MessageMetrics, RobustnessMetrics
from .attacks import AttackSimulator
from ..utils.logging_utils import MetricLogger, setup_logger
from ..utils.helpers import save_checkpoint, load_checkpoint, EarlyStopping


class Trainer:
    """Base trainer class for steganography models"""
    
    def __init__(self, config, model, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer=None, scheduler=None, device='cuda'):
        """
        Initialize trainer
        
        Args:
            config: Configuration object
            model: Steganography model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (will be created if None)
            scheduler: Learning rate scheduler
            device: Training device
        """
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup optimizer
        if optimizer is None:
            from .optimizer import get_optimizer
            self.optimizer = get_optimizer(model, config)
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler is None:
            from .scheduler import get_scheduler
            self.scheduler = get_scheduler(self.optimizer, config)
        else:
            self.scheduler = scheduler
        
        # Setup loss
        self.criterion = CombinedLoss(config)
        
        # Setup metrics
        self.image_metrics = ImageMetrics()
        self.message_metrics = MessageMetrics()
        self.robustness_metrics = RobustnessMetrics()
        
        # Attack simulator
        self.attack_simulator = AttackSimulator(device)
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Logging
        self.logger = setup_logger(__name__)
        self.metric_logger = MetricLogger(config.log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # Wandb
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project="ldpc-steganography", config=config.__dict__)
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation epoch
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint()
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.logger.info("Training completed")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        metrics = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            cover_images = batch['cover'].to(self.device)
            messages = batch['message'].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(cover_images, messages)
                
                # Calculate losses
                losses = self.criterion(outputs, cover_images, messages)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                if self.config.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                
                # Gradient clipping
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            for key, value in losses.items():
                metrics[key].append(value.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.global_step += 1
            
            # Log to wandb
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({
                    'train/loss': losses['total'].item(),
                    'train/step': self.global_step
                })
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                cover_images = batch['cover'].to(self.device)
                messages = batch['message'].to(self.device)
                
                # Forward pass
                outputs = self.model(cover_images, messages)
                
                # Calculate losses
                losses = self.criterion(outputs, cover_images, messages)
                
                # Calculate metrics
                stego_images = outputs['stego_images']
                extracted_messages = outputs['extracted_messages']
                
                # Image quality metrics
                image_quality = self.image_metrics.calculate(stego_images, cover_images)
                
                # Message accuracy
                message_accuracy = self.message_metrics.calculate(
                    extracted_messages, messages
                )
                
                # Update metrics
                for key, value in losses.items():
                    metrics[f'loss_{key}'].append(value.item())
                
                metrics['psnr'].append(image_quality['psnr'])
                metrics['ssim'].append(image_quality['ssim'])
                metrics['message_acc'].append(message_accuracy['accuracy'])
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        avg_metrics['loss'] = avg_metrics['loss_total']
        
        return avg_metrics
    
    def test_robustness(self, num_samples: int = 100) -> Dict[str, float]:
        """Test model robustness against attacks"""
        self.model.eval()
        robustness_results = defaultdict(list)
        
        # Sample batches
        for i, batch in enumerate(self.val_loader):
            if i >= num_samples // self.config.batch_size:
                break
            
            cover_images = batch['cover'].to(self.device)
            messages = batch['message'].to(self.device)
            
            with torch.no_grad():
                # Generate stego images
                outputs = self.model(cover_images, messages)
                stego_images = outputs['stego_images']
                
                # Test different attacks
                for attack_type in ['jpeg', 'noise', 'blur', 'crop']:
                    for strength in [0.1, 0.3, 0.5]:
                        # Apply attack
                        attacked_images = self.attack_simulator.apply_attack(
                            stego_images, attack_type, strength
                        )
                        
                        # Extract messages
                        extracted = self.model.extract_message(attacked_images)
                        
                        # Calculate accuracy
                        acc = self.message_metrics.calculate(extracted, messages)
                        
                        key = f'{attack_type}_{strength}'
                        robustness_results[key].append(acc['accuracy'])
        
        # Average results
        avg_robustness = {key: np.mean(values) for key, values in robustness_results.items()}
        
        return avg_robustness
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.output_dir) / f'checkpoint_epoch_{self.epoch}.pth'
        save_checkpoint(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.output_dir) / 'best_model.pth'
            save_checkpoint(checkpoint, best_path)
            self.logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def _log_metrics(self, train_metrics: Dict[str, float], 
                     val_metrics: Dict[str, float]):
        """Log training metrics"""
        # Console logging
        self.logger.info(
            f"Epoch {self.epoch} - "
            f"Train Loss: {train_metrics['total']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val PSNR: {val_metrics['psnr']:.2f}, "
            f"Val Message Acc: {val_metrics['message_acc']:.4f}"
        )
        
        # File logging
        self.metric_logger.log({
            'epoch': self.epoch,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Wandb logging
        if self.use_wandb:
            wandb.log({
                'epoch': self.epoch,
                'train/loss': train_metrics['total'],
                'val/loss': val_metrics['loss'],
                'val/psnr': val_metrics['psnr'],
                'val/ssim': val_metrics['ssim'],
                'val/message_acc': val_metrics['message_acc']
            })
        
        # Update best model
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(is_best=True)


class LDPCTrainer(Trainer):
    """Specialized trainer for LDPC steganography"""
    
    def __init__(self, config, model, ldpc_system, train_loader, val_loader, 
                 optimizer=None, scheduler=None, device='cuda'):
        """
        Initialize LDPC trainer
        
        Args:
            config: Configuration object
            model: LDPC-aware steganography model
            ldpc_system: Adaptive LDPC system
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
        """
        super().__init__(config, model, train_loader, val_loader, 
                        optimizer, scheduler, device)
        
        self.ldpc_system = ldpc_system
        
        # LDPC-specific metrics
        self.ldpc_metrics = defaultdict(list)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with LDPC encoding"""
        self.model.train()
        metrics = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            cover_images = batch['cover'].to(self.device)
            messages = batch['message'].to(self.device)
            
            # Sample attack strength for adaptive LDPC
            attack_strength = np.random.uniform(0.0, 1.0)
            
            # LDPC encode messages
            encoded_messages = self.ldpc_system.encode(messages, attack_strength)
            encoded_messages = torch.tensor(encoded_messages, device=self.device, dtype=torch.float32)
            
            # Forward pass
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(cover_images, encoded_messages)
                
                # Simulate attack
                if np.random.random() < 0.5:  # 50% chance of attack
                    attack_type = np.random.choice(['jpeg', 'noise', 'blur'])
                    stego_images = self.attack_simulator.apply_attack(
                        outputs['stego_images'], attack_type, attack_strength
                    )
                else:
                    stego_images = outputs['stego_images']
                
                # Extract and decode
                extracted_encoded = self.model.extract_message(stego_images)
                
                # LDPC decode
                decoded_messages = self.ldpc_system.decode(
                    extracted_encoded.cpu().numpy(), attack_strength
                )
                decoded_messages = torch.tensor(decoded_messages, device=self.device, dtype=torch.float32)
                
                # Calculate losses
                outputs['decoded_messages'] = decoded_messages
                losses = self.criterion(outputs, cover_images, messages)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(losses['total']).backward()
                
                if self.config.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            for key, value in losses.items():
                metrics[key].append(value.item())
            
            # LDPC metrics
            ldpc_info = self.ldpc_system.get_code_info(attack_strength)
            metrics['ldpc_redundancy'].append(ldpc_info['redundancy'])
            metrics['ldpc_rate'].append(ldpc_info['rate'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'redundancy': ldpc_info['redundancy'],
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.global_step += 1
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        return avg_metrics
    
    def test_ldpc_performance(self) -> Dict[str, Any]:
        """Test LDPC system performance"""
        self.logger.info("Testing LDPC performance...")
        
        # Benchmark LDPC performance
        ldpc_benchmark = self.ldpc_system.benchmark_performance(
            num_tests=1000,
            noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Compare with Reed-Solomon
        from ..core.adaptive_ldpc import LDPCPerformanceAnalyzer
        analyzer = LDPCPerformanceAnalyzer(self.ldpc_system)
        comparison = analyzer.compare_with_reed_solomon(num_tests=500)
        
        # Capacity analysis
        capacity_analysis = analyzer.capacity_analysis()
        
        results = {
            'ldpc_benchmark': ldpc_benchmark,
            'reed_solomon_comparison': comparison,
            'capacity_analysis': capacity_analysis
        }
        
        # Log results
        self.logger.info(f"LDPC vs Reed-Solomon improvements: {comparison['improvements']}")
        
        return results