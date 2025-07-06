#!/usr/bin/env python3
"""
Checkpoint Manager
Utilities for saving and loading model checkpoints
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import shutil
from datetime import datetime
import json


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, save_dir: Path, max_checkpoints: int = 5, 
                 keep_best: bool = True):
        """
        Initialize checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best: Whether to keep best checkpoints regardless of limit
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        # Track saved checkpoints
        self.checkpoint_history = []
        self.best_checkpoints = set()
        
        # Load existing checkpoint info
        self._load_checkpoint_info()
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, state_dict: Dict[str, Any], filename: str, 
                       is_best: bool = False, metrics: Optional[Dict[str, float]] = None):
        """
        Save checkpoint
        
        Args:
            state_dict: State dictionary to save
            filename: Checkpoint filename
            is_best: Whether this is the best checkpoint
            metrics: Optional metrics to save with checkpoint
        """
        filepath = self.save_dir / filename
        
        # Add metadata
        checkpoint = {
            'state_dict': state_dict,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'is_best': is_best
        }
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint: {filepath}")
        
        # Update history
        self.checkpoint_history.append({
            'filename': filename,
            'filepath': str(filepath),
            'timestamp': checkpoint['timestamp'],
            'metrics': metrics or {},
            'is_best': is_best
        })
        
        if is_best:
            self.best_checkpoints.add(filename)
            # Also save as 'best.pth'
            best_path = self.save_dir / 'best.pth'
            shutil.copy2(filepath, best_path)
            self.logger.info(f"Updated best checkpoint: {best_path}")
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save checkpoint info
        self._save_checkpoint_info()
    
    def load_checkpoint(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            filename: Checkpoint filename (if None, loads latest/best)
            
        Returns:
            Checkpoint dictionary
        """
        if filename is None:
            # Try to load best checkpoint first
            best_path = self.save_dir / 'best.pth'
            if best_path.exists():
                filepath = best_path
                self.logger.info("Loading best checkpoint")
            else:
                # Load latest checkpoint
                filepath = self._get_latest_checkpoint()
                if filepath is None:
                    raise FileNotFoundError("No checkpoints found")
        else:
            filepath = self.save_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        self.logger.info(f"Loaded checkpoint: {filepath}")
        
        return checkpoint
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        checkpoints = list(self.save_dir.glob('*.pth'))
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding the limit"""
        if self.max_checkpoints <= 0:
            return
        
        # Get non-best checkpoints
        regular_checkpoints = [
            cp for cp in self.checkpoint_history 
            if cp['filename'] not in self.best_checkpoints
        ]
        
        # Sort by timestamp (newest first)
        regular_checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Remove old checkpoints
        to_remove = regular_checkpoints[self.max_checkpoints:]
        for cp_info in to_remove:
            filepath = Path(cp_info['filepath'])
            if filepath.exists():
                filepath.unlink()
                self.logger.info(f"Removed old checkpoint: {filepath}")
            
            # Remove from history
            self.checkpoint_history.remove(cp_info)
    
    def _save_checkpoint_info(self):
        """Save checkpoint information to JSON"""
        info_path = self.save_dir / 'checkpoint_info.json'
        
        info = {
            'history': self.checkpoint_history,
            'best_checkpoints': list(self.best_checkpoints)
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_checkpoint_info(self):
        """Load checkpoint information from JSON"""
        info_path = self.save_dir / 'checkpoint_info.json'
        
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            self.checkpoint_history = info.get('history', [])
            self.best_checkpoints = set(info.get('best_checkpoints', []))
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all saved checkpoints"""
        return self.checkpoint_history
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about best checkpoint"""
        best_checkpoints = [
            cp for cp in self.checkpoint_history 
            if cp['is_best']
        ]
        
        if best_checkpoints:
            # Return most recent best checkpoint
            return max(best_checkpoints, key=lambda x: x['timestamp'])
        
        return None


class ModelCheckpoint:
    """Callback for saving model checkpoints during training"""
    
    def __init__(self, checkpoint_manager: CheckpointManager,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_freq: int = 1):
        """
        Initialize model checkpoint callback
        
        Args:
            checkpoint_manager: CheckpointManager instance
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_freq: Save frequency (epochs)
        """
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.epochs_since_improvement = 0
    
    def __call__(self, epoch: int, model: torch.nn.Module, 
                 metrics: Dict[str, float], optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """Save checkpoint if conditions are met"""
        
        # Check if should save regular checkpoint
        if epoch % self.save_freq == 0:
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            
            if scheduler is not None:
                state_dict['scheduler_state_dict'] = scheduler.state_dict()
            
            filename = f'checkpoint_epoch_{epoch}.pth'
            self.checkpoint_manager.save_checkpoint(state_dict, filename, 
                                                  is_best=False, metrics=metrics)
        
        # Check if best checkpoint
        if self.monitor in metrics:
            current_value = metrics[self.monitor]
            
            is_best = False
            if self.mode == 'min' and current_value < self.best_value:
                is_best = True
                self.best_value = current_value
                self.epochs_since_improvement = 0
            elif self.mode == 'max' and current_value > self.best_value:
                is_best = True
                self.best_value = current_value
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
            
            if is_best:
                state_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    f'best_{self.monitor}': self.best_value
                }
                
                if scheduler is not None:
                    state_dict['scheduler_state_dict'] = scheduler.state_dict()
                
                filename = f'best_{self.monitor}.pth'
                self.checkpoint_manager.save_checkpoint(state_dict, filename,
                                                      is_best=True, metrics=metrics)


def save_model_for_inference(model: torch.nn.Module, save_path: Path,
                           config: Optional[Dict[str, Any]] = None):
    """Save model for inference only (without training state)"""
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': config or {},
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(save_dict, save_path)
    logging.info(f"Saved inference model to {save_path}")


def load_model_for_inference(model_class, load_path: Path, 
                           device: str = 'cpu', **kwargs):
    """Load model for inference"""
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # Create model instance
    config = checkpoint.get('model_config', {})
    config.update(kwargs)
    
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logging.info(f"Loaded inference model from {load_path}")
    
    return model