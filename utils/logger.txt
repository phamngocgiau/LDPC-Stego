#!/usr/bin/env python3
"""
Logger Utilities
Logging configuration and utilities
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import yaml
from logging.handlers import RotatingFileHandler
import colorlog


def setup_logger(name: str, log_dir: Optional[str] = None, 
                 level: str = 'INFO', console: bool = True,
                 file: bool = True, max_bytes: int = 10485760,
                 backup_count: int = 5) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Enable console output
        file: Enable file output
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    if console and sys.stdout.isatty():
        # Colored console formatter
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        # Plain console formatter
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Also create a latest.log symlink
        latest_log = log_dir / f'{name}_latest.log'
        if latest_log.exists():
            latest_log.unlink()
        latest_log.symlink_to(log_file.name)
    
    return logger


class ExperimentLogger:
    """Logger for experiment tracking"""
    
    def __init__(self, experiment_dir: Path, experiment_name: str):
        """
        Initialize experiment logger
        
        Args:
            experiment_dir: Directory for experiment logs
            experiment_name: Name of experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        
        # Create directories
        self.log_dir = self.experiment_dir / 'logs'
        self.metrics_dir = self.experiment_dir / 'metrics'
        self.config_dir = self.experiment_dir / 'configs'
        
        for dir_path in [self.log_dir, self.metrics_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            experiment_name,
            log_dir=str(self.log_dir),
            level='INFO'
        )
        
        # Metrics storage
        self.metrics = {
            'train': [],
            'val': [],
            'test': {}
        }
        
        # Start time
        self.start_time = datetime.now()
        
    def log_config(self, config: Dict[str, Any], filename: str = 'config.yaml'):
        """Log configuration"""
        config_path = self.config_dir / filename
        
        if filename.endswith('.yaml'):
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif filename.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def log_metrics(self, metrics: Dict[str, float], phase: str = 'train', 
                   epoch: Optional[int] = None, step: Optional[int] = None):
        """Log metrics"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            **metrics
        }
        
        if phase in ['train', 'val']:
            self.metrics[phase].append(entry)
            
            # Save to file
            metrics_file = self.metrics_dir / f'{phase}_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics[phase], f, indent=2)
        else:
            self.metrics['test'][phase] = entry
            
            # Save test results
            test_file = self.metrics_dir / 'test_results.json'
            with open(test_file, 'w') as f:
                json.dump(self.metrics['test'], f, indent=2)
        
        # Log to console
        self.logger.info(f"{phase.capitalize()} metrics: {metrics}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information"""
        info_path = self.experiment_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Model info saved to {info_path}")
    
    def log_summary(self):
        """Log experiment summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': str(duration),
            'total_epochs': len(self.metrics['train']),
            'best_metrics': self._get_best_metrics()
        }
        
        summary_path = self.experiment_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiment summary saved to {summary_path}")
        self.logger.info(f"Experiment completed in {duration}")
    
    def _get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics from training"""
        best_metrics = {}
        
        if self.metrics['val']:
            # Find best validation metrics
            best_val_loss = float('inf')
            best_val_acc = 0.0
            
            for entry in self.metrics['val']:
                if 'total_loss' in entry and entry['total_loss'] < best_val_loss:
                    best_val_loss = entry['total_loss']
                    best_metrics['best_val_loss'] = {
                        'value': best_val_loss,
                        'epoch': entry['epoch']
                    }
                
                if 'message_acc' in entry and entry['message_acc'] > best_val_acc:
                    best_val_acc = entry['message_acc']
                    best_metrics['best_val_acc'] = {
                        'value': best_val_acc,
                        'epoch': entry['epoch']
                    }
        
        return best_metrics


class TensorBoardLogger:
    """TensorBoard logger wrapper"""
    
    def __init__(self, log_dir: Path):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        self.step = 0
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log scalar value"""
        if step is None:
            step = self.step
        self.writer.add_scalar(name, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], 
                   step: Optional[int] = None):
        """Log multiple scalars"""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: Optional[int] = None):
        """Log images"""
        if step is None:
            step = self.step
        self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram"""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple):
        """Log model graph"""
        dummy_input = torch.randn(1, *input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MetricTracker:
    """Track and aggregate metrics"""
    
    def __init__(self, metrics_to_track: list):
        """
        Initialize metric tracker
        
        Args:
            metrics_to_track: List of metric names to track
        """
        self.metrics_to_track = metrics_to_track
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {metric: [] for metric in self.metrics_to_track}
        self.best_values = {
            metric: {'value': None, 'epoch': None}
            for metric in self.metrics_to_track
        }
    
    def update(self, metric_dict: Dict[str, float], epoch: int):
        """Update metrics"""
        for metric, value in metric_dict.items():
            if metric in self.metrics:
                self.metrics[metric].append(value)
                
                # Update best value
                if self.best_values[metric]['value'] is None:
                    self.best_values[metric] = {'value': value, 'epoch': epoch}
                else:
                    # Assume lower is better for loss metrics
                    if 'loss' in metric:
                        if value < self.best_values[metric]['value']:
                            self.best_values[metric] = {'value': value, 'epoch': epoch}
                    else:
                        if value > self.best_values[metric]['value']:
                            self.best_values[metric] = {'value': value, 'epoch': epoch}
    
    def get_last(self, metric: str) -> Optional[float]:
        """Get last value of metric"""
        if metric in self.metrics and self.metrics[metric]:
            return self.metrics[metric][-1]
        return None
    
    def get_best(self, metric: str) -> Optional[Dict[str, Any]]:
        """Get best value of metric"""
        if metric in self.best_values:
            return self.best_values[metric]
        return None
    
    def get_average(self, metric: str, last_n: Optional[int] = None) -> float:
        """Get average of metric"""
        if metric not in self.metrics or not self.metrics[metric]:
            return 0.0
        
        values = self.metrics[metric]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def get_history(self, metric: str) -> list:
        """Get history of metric"""
        return self.metrics.get(metric, [])
    
    def save(self, filepath: Path):
        """Save metrics to file"""
        data = {
            'metrics': self.metrics,
            'best_values': self.best_values
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path):
        """Load metrics from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics = data['metrics']
        self.best_values = data['best_values']


def create_experiment_logger(base_dir: Path, experiment_name: str, 
                           config: Dict[str, Any]) -> ExperimentLogger:
    """
    Create experiment logger with full setup
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of experiment
        config: Experiment configuration
        
    Returns:
        Configured experiment logger
    """
    # Create unique experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = base_dir / f"{experiment_name}_{timestamp}"
    
    # Create logger
    exp_logger = ExperimentLogger(experiment_dir, experiment_name)
    
    # Log configuration
    exp_logger.log_config(config)
    
    # Log git info if available
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        git_info = {
            'commit': repo.head.object.hexsha,
            'branch': repo.active_branch.name,
            'dirty': repo.is_dirty()
        }
        exp_logger.log_config(git_info, 'git_info.json')
    except:
        pass
    
    return exp_logger