#!/usr/bin/env python3
"""
Learning Rate Scheduler Module
Various learning rate schedulers for training
"""

import torch
from torch.optim import lr_scheduler
from typing import Any, Optional
import math
import logging


def get_scheduler(optimizer, config) -> Optional[Any]:
    """
    Get learning rate scheduler based on configuration
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object with scheduler settings
        
    Returns:
        Learning rate scheduler or None
    """
    scheduler_type = config.scheduler_type
    scheduler_params = config.get_scheduler_config()
    
    if scheduler_type == "none":
        return None
    
    elif scheduler_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', config.num_epochs),
            eta_min=scheduler_params.get('eta_min', 0)
        )
    
    elif scheduler_type == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    
    elif scheduler_type == "multistep":
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_params.get('milestones', [30, 60, 90]),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    
    elif scheduler_type == "exponential":
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_params.get('gamma', 0.95)
        )
    
    elif scheduler_type == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 10),
            min_lr=scheduler_params.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == "cyclic":
        return lr_scheduler.CyclicLR(
            optimizer,
            base_lr=scheduler_params.get('base_lr', 1e-5),
            max_lr=scheduler_params.get('max_lr', config.learning_rate),
            step_size_up=scheduler_params.get('step_size_up', 2000),
            mode=scheduler_params.get('mode', 'triangular2')
        )
    
    elif scheduler_type == "onecycle":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=scheduler_params.get('total_steps', config.num_epochs * 1000),
            pct_start=scheduler_params.get('pct_start', 0.3)
        )
    
    elif scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=scheduler_params.get('warmup_epochs', 5),
            max_epochs=config.num_epochs,
            eta_min=scheduler_params.get('eta_min', 0)
        )
    
    elif scheduler_type == "warmup_linear":
        return WarmupLinearScheduler(
            optimizer,
            warmup_epochs=scheduler_params.get('warmup_epochs', 5),
            max_epochs=config.num_epochs,
            end_lr=scheduler_params.get('end_lr', 0)
        )
    
    else:
        logging.warning(f"Unknown scheduler type: {scheduler_type}")
        return None


class WarmupCosineScheduler(lr_scheduler._LRScheduler):
    """Cosine scheduler with linear warmup"""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 eta_min: float = 0, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * progress)) / 2
                   for base_lr in self.base_lrs]


class WarmupLinearScheduler(lr_scheduler._LRScheduler):
    """Linear scheduler with warmup"""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int,
                 end_lr: float = 0, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [base_lr + (self.end_lr - base_lr) * progress
                   for base_lr in self.base_lrs]


class PolynomialLRScheduler(lr_scheduler._LRScheduler):
    """Polynomial learning rate scheduler"""
    
    def __init__(self, optimizer, max_epochs: int, power: float = 0.9,
                 min_lr: float = 0, last_epoch: int = -1):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        progress = self.last_epoch / self.max_epochs
        factor = (1 - progress) ** self.power
        return [self.min_lr + (base_lr - self.min_lr) * factor
               for base_lr in self.base_lrs]


class RestartCosineScheduler(lr_scheduler._LRScheduler):
    """Cosine scheduler with restarts (SGDR)"""
    
    def __init__(self, optimizer, T_0: int, T_mult: int = 1, 
                 eta_min: float = 0, last_epoch: int = -1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.epoch_at_restart = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        self.T_cur = self.last_epoch - self.epoch_at_restart
        
        if self.T_cur >= self.T_i:
            self.epoch_at_restart = self.last_epoch
            self.T_i = self.T_i * self.T_mult
            self.T_cur = 0
        
        return [self.eta_min + (base_lr - self.eta_min) * 
               (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
               for base_lr in self.base_lrs]


class AdaptiveScheduler:
    """Adaptive scheduler that adjusts based on metrics"""
    
    def __init__(self, optimizer, initial_lr: float, patience: int = 5,
                 factor: float = 0.5, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_metric = float('inf')
        self.bad_epochs = 0
        
        self._set_lr(self.lr)
    
    def step(self, metric: float):
        """Update learning rate based on metric"""
        if metric < self.best_metric:
            self.best_metric = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        
        if self.bad_epochs >= self.patience:
            # Reduce learning rate
            self.lr = max(self.lr * self.factor, self.min_lr)
            self._set_lr(self.lr)
            self.bad_epochs = 0
            
            logging.info(f"Reduced learning rate to {self.lr}")
    
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        """Get scheduler state"""
        return {
            'lr': self.lr,
            'best_metric': self.best_metric,
            'bad_epochs': self.bad_epochs
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.lr = state_dict['lr']
        self.best_metric = state_dict['best_metric']
        self.bad_epochs = state_dict['bad_epochs']
        self._set_lr(self.lr)