#!/usr/bin/env python3
"""
Testing Script for LDPC Steganography
Comprehensive testing and evaluation of trained models
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.ldpc_config import LDPCConfig
from core.adaptive_ldpc import AdaptiveLDPC, create_ldpc_system
from models.steganography_model import AdvancedSteganographyModelWithLDPC
from data.data_loaders import create_data_loaders
from evaluation.evaluator import LDPCEvaluator
from evaluation.analysis import ResultAnalyzer
from utils.logging_utils import setup_logger
from utils.helpers import load_checkpoint, set_random_seed
from utils.visualization import save_sample_images


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test LDPC Steganography Model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to config file (uses checkpoint config if not specified)')
    
    # Data arguments
    parser.add_argument('--data-test', type=str, default='data/test',
                      help='Path to test data')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for testing')
    
    # Evaluation arguments
    parser.add_argument('--num-samples', type=int, default=100,
                      help='Number of samples for robustness testing')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for results')
    
    # Testing options
    parser.add_argument('--test-attacks', action='store_true',
                      help='Test robustness against attacks')
    parser.add_argument('--test-capacity', action='store_true',
                      help='Test capacity limits')
    parser.add_argument('--test-ldpc', action='store_true',
                      help='Test LDPC performance')
    parser.add_argument('--save-images', action='store_true',
                      help='Save sample images')
    parser.add_argument('--generate-report', action='store_true',
                      help='Generate analysis report')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()


def load_model_and_config(args):
    """Load model and configuration"""
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    
    # Load config
    if args.config:
        config = LDPCConfig.load(args.config)
    else:
        # Use config from checkpoint
        config_dict = checkpoint.get('config', {})
        config = LDPCConfig.from_dict(config_dict)
    
    # Update device
    config.device = args.device
    
    # Create LDPC system
    ldpc_system = create_ldpc_system(config)
    
    # Create model
    model = AdvancedSteganographyModelWithLDPC(config, ldpc_system)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    return model, config, ldpc_system


def test_basic_performance(model, test_loader, config, logger):
    """Test basic performance metrics"""
    logger.info("Testing basic performance...")
    
    from evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(model, test_loader, config, device=config.device)
    
    basic_results = evaluator.evaluate_basic()
    
    logger.info(f"Basic Performance Results:")
    logger.info(f"  PSNR: {basic_results['psnr']:.2f} dB")
    logger.info(f"  SSIM: {basic_results['ssim']:.4f}")
    logger.info(f"  Message Accuracy: {basic_results['message_accuracy']:.4f}")
    logger.info(f"  Bit Error Rate: {basic_results['bit_error_rate']:.6f}")
    
    return basic_results


def test_robustness(model, test_loader, config, logger):
    """Test robustness against various attacks"""
    logger.info("Testing robustness...")
    
    from evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(model, test_loader, config, device=config.device)
    
    robustness_results = evaluator.evaluate_robustness()
    
    logger.info("Robustness Results:")
    for attack_type, attack_results in robustness_results.items():
        logger.info(f"\n  {attack_type.upper()}:")
        for strength, accuracy in attack_results.items():
            logger.info(f"    Strength {strength}: {accuracy:.4f}")
    
    return robustness_results


def test_capacity(model, test_loader, config, logger):
    """Test model capacity at different message lengths"""
    logger.info("Testing capacity...")
    
    from evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(model, test_loader, config, device=config.device)
    
    capacity_results = evaluator.evaluate_capacity()
    
    logger.info("Capacity Results:")
    for msg_len, results in capacity_results.items():
        logger.info(f"  Message length {msg_len}: "
                   f"Accuracy={results['accuracy']:.4f}, "
                   f"PSNR={results['psnr']:.2f}, "
                   f"BPP={results['bpp']:.4f}")
    
    return capacity_results


def test_ldpc_performance(model, ldpc_system, test_loader, config, logger):
    """Test LDPC-specific performance"""
    logger.info("Testing LDPC performance...")
    
    evaluator = LDPCEvaluator(model, ldpc_system, test_loader, config, device=config.device)
    
    ldpc_results = evaluator.evaluate_ldpc_performance()
    
    logger.info("LDPC Performance Results:")
    for redundancy, results in ldpc_results.items():
        logger.info(f"  Redundancy {redundancy:.1f}: "
                   f"Accuracy={results['accuracy']:.4f}, "
                   f"Code Rate={results['code_rate']:.3f}")
    
    # Compare with Reed-Solomon simulation
    logger.info("\nComparing with Reed-Solomon...")
    from evaluation.benchmarks import ReedSolomonSimulator
    
    rs_simulator = ReedSolomonSimulator(n=255, k=int(255 * (1 - redundancy)))
    
    # Test on subset
    rs_accuracies = []
    ldpc_accuracies = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:
                break
            
            messages = batch['message'].numpy()
            
            # Simulate RS
            rs_encoded = rs_simulator.encode(messages)
            rs_decoded = rs_simulator.decode(rs_encoded, error_rate=0.1)
            rs_acc = (rs_decoded == (messages > 0.5)).mean()
            rs_accuracies.append(rs_acc)
            
            # LDPC accuracy
            ldpc_encoded = ldpc_system.encode(messages, attack_strength=0.3)
            ldpc_decoded = ldpc_system.decode(ldpc_encoded, attack_strength=0.3)
            ldpc_acc = (ldpc_decoded > 0.5).astype(int) == (messages > 0.5).astype(int)
            ldpc_acc = ldpc_acc.mean()
            ldpc_accuracies.append(ldpc_acc)
    
    rs_avg = np.mean(rs_accuracies)
    ldpc_avg = np.mean(ldpc_accuracies)
    improvement = ((ldpc_avg - rs_avg) / rs_avg * 100) if rs_avg > 0 else 0
    
    logger.info(f"\nLDPC vs Reed-Solomon:")
    logger.info(f"  LDPC Average Accuracy: {ldpc_avg:.4f}")
    logger.info(f"  RS Average Accuracy: {rs_avg:.4f}")
    logger.info(f"  Improvement: {improvement:.1f}%")
    
    return ldpc_results


def save_test_images(model, test_loader, config, output_dir, logger):
    """Save sample test images"""
    logger.info("Saving sample images...")
    
    output_dir = Path(output_dir) / 'sample_images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5:  # Save 5 batches
                break
            
            cover_images = batch['cover'].to(config.device)
            messages = batch['message'].to(config.device)
            
            # Generate stego images
            outputs = model(cover_images, messages)
            stego_images = outputs['stego_images']
            
            # Extract messages
            extracted_messages = outputs['extracted_messages']
            
            # Calculate metrics
            from utils.helpers import calculate_psnr, calculate_ssim
            psnr = calculate_psnr(stego_images, cover_images)
            ssim = calculate_ssim(stego_images, cover_images)
            
            # Message accuracy
            acc = (extracted_messages > 0) == (messages > 0.5)
            acc = acc.float().mean(dim=1)
            
            # Save images
            for j in range(cover_images.size(0)):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Cover image
                cover_np = cover_images[j].cpu().numpy().transpose(1, 2, 0)
                cover_np = (cover_np + 1) / 2  # Denormalize
                axes[0].imshow(np.clip(cover_np, 0, 1))
                axes[0].set_title('Cover Image')
                axes[0].axis('off')
                
                # Stego image
                stego_np = stego_images[j].cpu().numpy().transpose(1, 2, 0)
                stego_np = (stego_np + 1) / 2  # Denormalize
                axes[1].imshow(np.clip(stego_np, 0, 1))
                axes[1].set_title(f'Stego Image\nPSNR: {psnr[j]:.2f} dB')
                axes[1].axis('off')
                
                # Difference (amplified)
                diff = np.abs(cover_np - stego_np)
                diff_amplified = np.clip(diff * 10, 0, 1)
                axes[2].imshow(diff_amplified)
                axes[2].set_title(f'Difference (10x)\nMsg Acc: {acc[j]:.2%}')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'sample_batch{i}_image{j}.png', dpi=150)
                plt.close()
    
    logger.info(f"Sample images saved to {output_dir}")


def generate_test_report(results, output_path, logger):
    """Generate comprehensive test report"""
    logger.info("Generating test report...")
    
    report = []
    report.append("# LDPC Steganography Test Report\n")
    report.append(f"Generated on: {Path(output_path).parent.name}\n")
    
    # Basic Performance
    if 'basic' in results:
        report.append("## Basic Performance\n")
        report.append(f"- **PSNR**: {results['basic']['psnr']:.2f} dB")
        report.append(f"- **SSIM**: {results['basic']['ssim']:.4f}")
        report.append(f"- **Message Accuracy**: {results['basic']['message_accuracy']:.4f}")
        report.append(f"- **Bit Error Rate**: {results['basic']['bit_error_rate']:.6f}\n")
    
    # Robustness
    if 'robustness' in results:
        report.append("## Robustness Analysis\n")
        
        for attack_type, attack_results in results['robustness'].items():
            report.append(f"\n### {attack_type.upper()} Attack")
            report.append("| Strength | Accuracy |")
            report.append("|----------|----------|")
            
            for strength, accuracy in attack_results.items():
                report.append(f"| {strength} | {accuracy:.4f} |")
    
    # Capacity
    if 'capacity' in results:
        report.append("\n## Capacity Analysis\n")
        report.append("| Message Length | Accuracy | PSNR (dB) | BPP |")
        report.append("|----------------|----------|-----------|-----|")
        
        for msg_len, res in results['capacity'].items():
            report.append(f"| {msg_len} | {res['accuracy']:.4f} | "
                         f"{res['psnr']:.2f} | {res['bpp']:.4f} |")
    
    # LDPC Performance
    if 'ldpc' in results:
        report.append("\n## LDPC Performance\n")
        report.append("| Redundancy | Accuracy | Code Rate |")
        report.append("|------------|----------|-----------|")
        
        for redundancy, res in results['ldpc'].items():
            report.append(f"| {redundancy:.1f} | {res['accuracy']:.4f} | "
                         f"{res['code_rate']:.3f} |")
    
    # Save report
    report_text = '\n'.join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Test report saved to {output_path}")


def main():
    """Main testing function"""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        checkpoint_name = Path(args.checkpoint).stem
        args.output_dir = f'results/test_{checkpoint_name}'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(__name__, log_file=output_dir / 'testing.log')
    logger.info(f"Starting LDPC Steganography Testing")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load model and config
    logger.info("Loading model and configuration...")
    model, config, ldpc_system = load_model_and_config(args)
    
    # Create test data loader
    logger.info("Creating test data loader...")
    _, _, test_loader = create_data_loaders(
        train_dir=config.data_train_folder,
        val_dir=config.data_val_folder,
        test_dir=args.data_test,
        image_size=config.image_size,
        batch_size=args.batch_size,
        num_workers=config.num_workers,
        message_length=config.message_length,
        augment_train=False
    )
    
    if test_loader is None:
        logger.warning("Test data not found, using validation data")
        _, test_loader, _ = create_data_loaders(
            train_dir=config.data_train_folder,
            val_dir=config.data_val_folder,
            image_size=config.image_size,
            batch_size=args.batch_size,
            num_workers=config.num_workers,
            message_length=config.message_length,
            augment_train=False
        )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Run tests
    results = {}
    
    # Basic performance
    basic_results = test_basic_performance(model, test_loader, config, logger)
    results['basic'] = basic_results
    
    # Robustness testing
    if args.test_attacks:
        robustness_results = test_robustness(model, test_loader, config, logger)
        results['robustness'] = robustness_results
    
    # Capacity testing
    if args.test_capacity:
        capacity_results = test_capacity(model, test_loader, config, logger)
        results['capacity'] = capacity_results
    
    # LDPC performance
    if args.test_ldpc:
        ldpc_results = test_ldpc_performance(model, ldpc_system, test_loader, config, logger)
        results['ldpc'] = ldpc_results
    
    # Save results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save sample images
    if args.save_images:
        save_test_images(model, test_loader, config, output_dir, logger)
    
    # Generate report
    if args.generate_report:
        generate_test_report(results, output_dir / 'test_report.md', logger)
    
    logger.info("Testing completed successfully!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()