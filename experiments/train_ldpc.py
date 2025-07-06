#!/usr/bin/env python3
"""
Main Training Script for LDPC Steganography
Complete training pipeline with LDPC error correction
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.ldpc_config import LDPCTrainingConfig
from core.adaptive_ldpc import AdaptiveLDPC, create_ldpc_system
from models.steganography_model import AdvancedSteganographyModelWithLDPC
from data.data_loaders import create_data_loaders
from training.trainer import LDPCTrainer
from training.optimizer import get_optimizer
from training.scheduler import get_scheduler
from utils.logging_utils import setup_logger
from utils.helpers import set_random_seed, count_parameters


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LDPC Steganography Model')
    
    # Data arguments
    parser.add_argument('--data-train', type=str, default='data/train',
                      help='Path to training data')
    parser.add_argument('--data-val', type=str, default='data/val',
                      help='Path to validation data')
    
    # Model arguments
    parser.add_argument('--image-size', type=int, default=256,
                      help='Image size')
    parser.add_argument('--message-length', type=int, default=1024,
                      help='Message length in bits')
    parser.add_argument('--base-channels', type=int, default=64,
                      help='Base channels for UNet')
    
    # LDPC arguments
    parser.add_argument('--ldpc-min-redundancy', type=float, default=0.1,
                      help='Minimum LDPC redundancy')
    parser.add_argument('--ldpc-max-redundancy', type=float, default=0.5,
                      help='Maximum LDPC redundancy')
    parser.add_argument('--ldpc-neural-decoder', action='store_true',
                      help='Use neural LDPC decoder')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay')
    
    # Loss weights
    parser.add_argument('--loss-message', type=float, default=12.0,
                      help='Message loss weight')
    parser.add_argument('--loss-mse', type=float, default=2.0,
                      help='MSE loss weight')
    parser.add_argument('--loss-lpips', type=float, default=1.5,
                      help='LPIPS loss weight')
    parser.add_argument('--loss-ssim', type=float, default=1.0,
                      help='SSIM loss weight')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data workers')
    parser.add_argument('--experiment-name', type=str, default='ldpc_steganography',
                      help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                      help='Debug mode')
    parser.add_argument('--use-wandb', action='store_true',
                      help='Use Weights & Biases logging')
    
    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment configuration"""
    # Create configuration
    config = LDPCTrainingConfig(
        # Data settings
        data_train_folder=args.data_train,
        data_val_folder=args.data_val,
        image_size=args.image_size,
        message_length=args.message_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        
        # Model settings
        unet_base_channels=args.base_channels,
        
        # LDPC settings
        ldpc_min_redundancy=args.ldpc_min_redundancy,
        ldpc_max_redundancy=args.ldpc_max_redundancy,
        ldpc_use_neural_decoder=args.ldpc_neural_decoder and args.device == 'cuda',
        
        # Training settings
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        
        # Loss weights
        loss_weights={
            'message': args.loss_message,
            'mse': args.loss_mse,
            'lpips': args.loss_lpips,
            'ssim': args.loss_ssim,
            'adversarial': 0.5,
            'recovery_mse': 1.0,
            'recovery_kl': 0.1,
            'decoded_message': 10.0
        },
        
        # Other settings
        device=args.device,
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb
    )
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = Path(config.output_dir) / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    config.output_dir = str(experiment_dir)
    config.log_dir = str(experiment_dir / 'logs')
    
    # Save configuration
    config.save(experiment_dir / 'config.json')
    
    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup experiment
    config = setup_experiment(args)
    
    # Setup logging
    logger = setup_logger(__name__, log_file=Path(config.log_dir) / 'training.log')
    logger.info(f"Starting LDPC Steganography Training")
    logger.info(f"Configuration: {config.__dict__}")
    
    # Set random seed
    set_random_seed(config.seed)
    
    # Initialize wandb if requested
    if config.use_wandb:
        wandb.init(
            project="ldpc-steganography",
            name=f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config.__dict__
        )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, _ = create_data_loaders(
        train_dir=config.data_train_folder,
        val_dir=config.data_val_folder,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        message_length=config.message_length,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        augment_train=True,
        debug=args.debug
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create LDPC system
    logger.info("Initializing LDPC system...")
    ldpc_system = create_ldpc_system(config)
    
    # Test LDPC system
    logger.info("Testing LDPC system...")
    test_message = torch.randint(0, 2, (1, config.message_length), dtype=torch.float32)
    test_encoded = ldpc_system.encode(test_message.numpy(), attack_strength=0.3)
    test_decoded = ldpc_system.decode(test_encoded, attack_strength=0.3)
    
    logger.info(f"LDPC test - Original shape: {test_message.shape}, "
                f"Encoded shape: {test_encoded.shape}, "
                f"Decoded shape: {test_decoded.shape}")
    
    # Create model
    logger.info("Creating model...")
    model = AdvancedSteganographyModelWithLDPC(config, ldpc_system)
    
    # Count parameters
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = LDPCTrainer(
        config=config,
        model=model,
        ldpc_system=ldpc_system,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Training loop
    try:
        logger.info("Starting training...")
        trainer.train()
        
        # Test LDPC performance
        logger.info("Testing LDPC performance...")
        ldpc_results = trainer.test_ldpc_performance()
        
        # Save LDPC results
        import json
        with open(Path(config.output_dir) / 'ldpc_performance.json', 'w') as f:
            json.dump(ldpc_results, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Save final checkpoint
        trainer.save_checkpoint()
        
        # Close wandb
        if config.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    main()