#!/usr/bin/env python3
"""
Main Training Script for LDPC Steganography
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.ldpc_config import LDPCTrainingConfig
from models.steganography_model import AdvancedSteganographyModelWithLDPC
from data.datasets import SteganographyDataset
from data.data_loaders import create_data_loaders
from training.trainer import LDPCSteganographyTrainer
from utils.helpers import set_random_seeds, setup_directories


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LDPC Steganography Model')
    
    # Basic arguments
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results/runs',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default='ldpc_exp',
                       help='Experiment name')
    
    # Model arguments
    parser.add_argument('--message_length', type=int, default=1024,
                       help='Message length in bits')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--ldpc_min_redundancy', type=float, default=0.1,
                       help='Minimum LDPC redundancy')
    parser.add_argument('--ldpc_max_redundancy', type=float, default=0.5,
                       help='Maximum LDPC redundancy')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Advanced arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced dataset')
    
    # GPU arguments
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU id to use')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use multiple GPUs')
    
    return parser.parse_args()


def setup_logging(output_dir: Path, experiment_name: str):
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_name}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('train_ldpc')


def create_config(args) -> LDPCTrainingConfig:
    """Create configuration from arguments"""
    
    # Load base config from file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    
    # Override with command line arguments
    config_dict.update({
        'data_train_folder': os.path.join(args.data_dir, 'train'),
        'data_val_folder': os.path.join(args.data_dir, 'val'),
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name,
        'message_length': args.message_length,
        'image_size': args.image_size,
        'ldpc_min_redundancy': args.ldpc_min_redundancy,
        'ldpc_max_redundancy': args.ldpc_max_redundancy,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'num_workers': args.num_workers,
        'mixed_precision': args.mixed_precision,
        'use_wandb': args.use_wandb,
        'device': f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    })
    
    # Create config object
    config = LDPCTrainingConfig(**config_dict)
    
    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.experiment_name)
    logger.info("Starting LDPC Steganography Training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create configuration
    config = create_config(args)
    config.output_dir = str(output_dir)
    config.log_dir = str(output_dir / 'logs')
    
    # Save configuration
    config_path = output_dir / 'config.yaml'
    config.save(str(config_path))
    logger.info(f"Configuration saved to {config_path}")
    
    # Set random seeds
    set_random_seeds(config.seed)
    
    # Setup device
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        device = torch.device('cuda')
    else:
        device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dir=config.data_train_folder,
        val_dir=config.data_val_folder,
        test_dir=config.data_train_folder if args.debug else None,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        message_length=config.message_length,
        debug=args.debug
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = AdvancedSteganographyModelWithLDPC(config)
    
    # Load pretrained weights if provided
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = LDPCSteganographyTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Validate model integrity
    logger.info("Validating model integrity...")
    validation_results = model.validate_model_integrity() if not args.multi_gpu else \
                        model.module.validate_model_integrity()
    
    for component, is_valid in validation_results.items():
        status = "✓" if is_valid else "✗"
        logger.info(f"  {status} {component}")
    
    if not all(validation_results.values()):
        logger.error("Model validation failed! Please check the implementation.")
        return
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint('interrupted.pth')
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    # Final evaluation
    logger.info("Running final evaluation...")
    test_results = trainer.test()
    
    # Save results
    results_path = output_dir / 'final_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(test_results, f, default_flow_style=False)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training script completed!")


if __name__ == "__main__":
    main()