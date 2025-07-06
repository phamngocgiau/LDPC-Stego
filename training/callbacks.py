#!/usr/bin/env python3
"""
Training Callbacks
Callback functions for training monitoring and control
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


class Callback:
    """Base callback class"""
    
    def on_train_begin(self, trainer):
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, trainer):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the beginning of an epoch"""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Called at the end of an epoch"""
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int):
        """Called at the beginning of a batch"""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Called at the end of a batch"""
        pass
    
    def on_validation_begin(self, trainer):
        """Called at the beginning of validation"""
        pass
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Called at the end of validation"""
        pass


class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 mode: str = 'min', min_delta: float = 1e-4):
        """
        Initialize early stopping
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Check if should stop"""
        if self.monitor not in metrics:
            logging.warning(f"Early stopping: metric '{self.monitor}' not found")
            return
        
        current_value = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                trainer.stop_training = True
                logging.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_train_end(self, trainer):
        """Log early stopping info"""
        if self.stopped_epoch > 0:
            logging.info(f"Training stopped early at epoch {self.stopped_epoch}")


class ModelCheckpoint(Callback):
    """Model checkpoint callback"""
    
    def __init__(self, save_dir: Path, monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True,
                 save_freq: int = 1):
        """
        Initialize model checkpoint
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            save_freq: Save frequency (epochs)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if needed"""
        # Check if should save regular checkpoint
        if not self.save_best_only and epoch % self.save_freq == 0:
            filepath = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            self._save_checkpoint(trainer, filepath, epoch, metrics)
        
        # Check if best model
        if self.monitor in metrics:
            current_value = metrics[self.monitor]
            
            if self.mode == 'min':
                is_best = current_value < self.best_value
            else:
                is_best = current_value > self.best_value
            
            if is_best:
                self.best_value = current_value
                filepath = self.save_dir / 'best_model.pth'
                self._save_checkpoint(trainer, filepath, epoch, metrics)
                logging.info(f"Saved best model: {self.monitor}={current_value:.4f}")
    
    def _save_checkpoint(self, trainer, filepath: Path, epoch: int, 
                        metrics: Dict[str, float]):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_G_state_dict': trainer.optimizer_G.state_dict(),
            'optimizer_D_state_dict': trainer.optimizer_D.state_dict(),
            'scheduler_G_state_dict': trainer.scheduler_G.state_dict(),
            'scheduler_D_state_dict': trainer.scheduler_D.state_dict(),
            'metrics': metrics,
            'best_value': self.best_value,
            'config': trainer.config.to_dict() if hasattr(trainer.config, 'to_dict') else {}
        }
        
        torch.save(checkpoint, filepath)


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(self, scheduler_G, scheduler_D=None, scheduler_LDPC=None):
        """
        Initialize LR scheduler callback
        
        Args:
            scheduler_G: Generator scheduler
            scheduler_D: Discriminator scheduler
            scheduler_LDPC: LDPC scheduler
        """
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D
        self.scheduler_LDPC = scheduler_LDPC
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Step schedulers"""
        if hasattr(self.scheduler_G, 'step'):
            if isinstance(self.scheduler_G, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_value = metrics.get('val_loss', 0)
                self.scheduler_G.step(metric_value)
            else:
                self.scheduler_G.step()
        
        if self.scheduler_D and hasattr(self.scheduler_D, 'step'):
            if isinstance(self.scheduler_D, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_value = metrics.get('val_loss', 0)
                self.scheduler_D.step(metric_value)
            else:
                self.scheduler_D.step()
        
        if self.scheduler_LDPC and hasattr(self.scheduler_LDPC, 'step'):
            self.scheduler_LDPC.step()
        
        # Log learning rates
        lr_G = trainer.optimizer_G.param_groups[0]['lr']
        lr_D = trainer.optimizer_D.param_groups[0]['lr']
        logging.info(f"Learning rates - G: {lr_G:.6f}, D: {lr_D:.6f}")


class MetricLogger(Callback):
    """Metric logging callback"""
    
    def __init__(self, log_dir: Path, log_freq: int = 10):
        """
        Initialize metric logger
        
        Args:
            log_dir: Directory for logs
            log_freq: Logging frequency (batches)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = log_freq
        
        self.train_metrics = []
        self.val_metrics = []
        self.batch_losses = []
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Log batch loss"""
        self.batch_losses.append(loss)
        
        if batch_idx % self.log_freq == 0:
            avg_loss = np.mean(self.batch_losses[-self.log_freq:])
            logging.info(f"Batch {batch_idx}: loss={avg_loss:.4f}")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics"""
        self.train_metrics.append({
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
        
        # Save metrics
        import json
        metrics_file = self.log_dir / 'train_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.train_metrics, f, indent=2)
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Log validation metrics"""
        self.val_metrics.append({
            'epoch': trainer.current_epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
        
        # Save metrics
        import json
        metrics_file = self.log_dir / 'val_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.val_metrics, f, indent=2)


class WandbLogger(Callback):
    """Weights & Biases logging callback"""
    
    def __init__(self, project: str, name: str, config: Dict[str, Any]):
        """
        Initialize W&B logger
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
        """
        self.run = wandb.init(project=project, name=name, config=config)
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Log batch metrics"""
        wandb.log({
            'batch_loss': loss,
            'global_step': trainer.global_step
        })
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics"""
        wandb.log({
            'epoch': epoch,
            **{f'train/{k}': v for k, v in metrics.items()}
        })
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        """Log validation metrics"""
        wandb.log({
            **{f'val/{k}': v for k, v in metrics.items()}
        })
    
    def on_train_end(self, trainer):
        """Finish W&B run"""
        wandb.finish()


class ProgressBar(Callback):
    """Progress bar callback"""
    
    def __init__(self):
        self.pbar = None
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Create progress bar"""
        total = len(trainer.train_loader)
        self.pbar = tqdm(total=total, desc=f"Epoch {epoch}")
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Update progress bar"""
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Close progress bar"""
        if self.pbar:
            self.pbar.close()


class Visualizer(Callback):
    """Visualization callback"""
    
    def __init__(self, save_dir: Path, num_samples: int = 4, freq: int = 5):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
            freq: Visualization frequency (epochs)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.freq = freq
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Create visualizations"""
        if epoch % self.freq != 0:
            return
        
        trainer.model.eval()
        
        with torch.no_grad():
            # Get a batch of data
            batch = next(iter(trainer.val_loader))
            cover_images = batch['cover'][:self.num_samples].to(trainer.device)
            messages = batch['message'][:self.num_samples].to(trainer.device)
            
            # Generate outputs
            outputs = trainer.model(cover_images, messages, training=False)
            
            # Create visualization
            self._create_visualization(
                cover_images,
                outputs['stego_images'],
                outputs.get('recovered_images'),
                epoch
            )
        
        trainer.model.train()
    
    def _create_visualization(self, cover: torch.Tensor, stego: torch.Tensor,
                            recovered: Optional[torch.Tensor], epoch: int):
        """Create and save visualization"""
        from ..utils.visualization import visualize_results
        
        save_path = self.save_dir / f'epoch_{epoch}.png'
        visualize_results(
            cover_images=cover,
            stego_images=stego,
            recovered_images=recovered,
            save_path=save_path,
            num_samples=self.num_samples
        )


class CallbackList:
    """Container for callbacks"""
    
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
    
    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_begin(self, trainer, batch_idx: int):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx)
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss)
    
    def on_validation_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_validation_begin(trainer)
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_validation_end(trainer, metrics)