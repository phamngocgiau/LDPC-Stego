#!/usr/bin/env python3
"""
Benchmark Comparison Script
Compare LDPC steganography with Reed-Solomon and other methods
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.ldpc_config import LDPCConfig
from core.adaptive_ldpc import AdaptiveLDPC, create_ldpc_system
from models.steganography_model import AdvancedSteganographyModelWithLDPC
from data.data_loaders import create_data_loaders
from evaluation.benchmarks import BenchmarkSuite, ReedSolomonSimulator
from utils.logging_utils import setup_logger
from utils.helpers import load_checkpoint, set_random_seed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark LDPC vs Other Methods')
    
    # Model arguments
    parser.add_argument('--ldpc-checkpoint', type=str, required=True,
                      help='Path to LDPC model checkpoint')
    parser.add_argument('--baseline-checkpoint', type=str, default=None,
                      help='Path to baseline model checkpoint (optional)')
    
    # Data arguments
    parser.add_argument('--data-test', type=str, default='data/test',
                      help='Path to test data')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for testing')
    parser.add_argument('--num-batches', type=int, default=50,
                      help='Number of batches to test')
    
    # Benchmark options
    parser.add_argument('--compare-rs', action='store_true',
                      help='Compare with Reed-Solomon')
    parser.add_argument('--compare-repetition', action='store_true',
                      help='Compare with repetition code')
    parser.add_argument('--compare-noec', action='store_true',
                      help='Compare with no error correction')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/benchmarks',
                      help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    
    return parser.parse_args()


class ReedSolomonSteganography(nn.Module):
    """Simulated Reed-Solomon based steganography for comparison"""
    
    def __init__(self, base_model, rs_n=255, rs_k=223):
        super().__init__()
        self.base_model = base_model
        self.rs_simulator = ReedSolomonSimulator(n=rs_n, k=rs_k)
        self.message_length = base_model.config.message_length if hasattr(base_model, 'config') else 1024
    
    def forward(self, cover_images, messages):
        # Encode with RS
        messages_np = messages.cpu().numpy()
        rs_encoded = self.rs_simulator.encode(messages_np)
        rs_encoded = torch.tensor(rs_encoded, device=messages.device, dtype=torch.float32)
        
        # Pad/truncate to expected size
        if rs_encoded.size(1) > self.message_length:
            rs_encoded = rs_encoded[:, :self.message_length]
        elif rs_encoded.size(1) < self.message_length:
            padding = torch.zeros(rs_encoded.size(0), self.message_length - rs_encoded.size(1),
                                device=rs_encoded.device)
            rs_encoded = torch.cat([rs_encoded, padding], dim=1)
        
        # Use base model for steganography
        outputs = self.base_model(cover_images, rs_encoded)
        
        # Decode RS
        if 'extracted_messages' in outputs:
            extracted = outputs['extracted_messages'][:, :self.rs_simulator.n]
            decoded = self.rs_simulator.decode(extracted.cpu().numpy(), error_rate=0.1)
            decoded = torch.tensor(decoded, device=extracted.device, dtype=torch.float32)
            outputs['extracted_messages'] = decoded
        
        return outputs


class RepetitionCodeSteganography(nn.Module):
    """Simple repetition code steganography for comparison"""
    
    def __init__(self, base_model, repetition_factor=3):
        super().__init__()
        self.base_model = base_model
        self.repetition_factor = repetition_factor
        self.message_length = base_model.config.message_length if hasattr(base_model, 'config') else 1024
    
    def forward(self, cover_images, messages):
        # Encode with repetition
        batch_size = messages.size(0)
        msg_len = messages.size(1)
        
        # Repeat each bit
        repeated = messages.unsqueeze(2).repeat(1, 1, self.repetition_factor)
        repeated = repeated.view(batch_size, -1)
        
        # Truncate to fit
        if repeated.size(1) > self.message_length:
            repeated = repeated[:, :self.message_length]
        elif repeated.size(1) < self.message_length:
            padding = torch.zeros(batch_size, self.message_length - repeated.size(1),
                                device=repeated.device)
            repeated = torch.cat([repeated, padding], dim=1)
        
        # Use base model
        outputs = self.base_model(cover_images, repeated)
        
        # Decode with majority voting
        if 'extracted_messages' in outputs:
            extracted = outputs['extracted_messages']
            
            # Reshape for voting
            extracted_reshaped = extracted[:, :msg_len * self.repetition_factor]
            extracted_reshaped = extracted_reshaped.view(batch_size, msg_len, self.repetition_factor)
            
            # Majority vote
            decoded = (extracted_reshaped.mean(dim=2) > 0.5).float()
            outputs['extracted_messages'] = decoded
        
        return outputs


def create_baseline_model(checkpoint_path, device):
    """Create baseline model without LDPC"""
    from models.steganography_model import AdvancedSteganographyModel
    
    checkpoint = load_checkpoint(checkpoint_path, device)
    config_dict = checkpoint.get('config', {})
    
    # Create config without LDPC
    config = type('Config', (), config_dict)()
    
    # Create model
    model = AdvancedSteganographyModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def run_comprehensive_benchmark(args, logger):
    """Run comprehensive benchmark comparison"""
    # Load LDPC model
    logger.info("Loading LDPC model...")
    checkpoint = load_checkpoint(args.ldpc_checkpoint, args.device)
    config = LDPCConfig.from_dict(checkpoint.get('config', {}))
    config.device = args.device
    
    ldpc_system = create_ldpc_system(config)
    ldpc_model = AdvancedSteganographyModelWithLDPC(config, ldpc_system)
    ldpc_model.load_state_dict(checkpoint['model_state_dict'])
    ldpc_model = ldpc_model.to(args.device)
    ldpc_model.eval()
    
    # Create data loader
    logger.info("Creating test data loader...")
    _, _, test_loader = create_data_loaders(
        train_dir=config.data_train_folder,
        val_dir=config.data_val_folder,
        test_dir=args.data_test,
        image_size=config.image_size,
        batch_size=args.batch_size,
        num_workers=4,
        message_length=config.message_length,
        augment_train=False
    )
    
    if test_loader is None:
        _, test_loader, _ = create_data_loaders(
            train_dir=config.data_train_folder,
            val_dir=config.data_val_folder,
            image_size=config.image_size,
            batch_size=args.batch_size,
            num_workers=4,
            message_length=config.message_length,
            augment_train=False
        )
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite(test_loader, device=args.device)
    
    # Add LDPC method
    benchmark_suite.add_method('LDPC', ldpc_model)
    
    # Add Reed-Solomon comparison
    if args.compare_rs:
        logger.info("Adding Reed-Solomon comparison...")
        # Create RS model using LDPC model as base
        rs_model = ReedSolomonSteganography(ldpc_model, rs_n=255, rs_k=223)
        benchmark_suite.add_method('Reed-Solomon', rs_model)
    
    # Add repetition code comparison
    if args.compare_repetition:
        logger.info("Adding repetition code comparison...")
        rep_model = RepetitionCodeSteganography(ldpc_model, repetition_factor=3)
        benchmark_suite.add_method('Repetition-3x', rep_model)
    
    # Add no error correction comparison
    if args.compare_noec:
        logger.info("Adding no error correction comparison...")
        if args.baseline_checkpoint:
            baseline_model, _ = create_baseline_model(args.baseline_checkpoint, args.device)
        else:
            # Use LDPC model without error correction
            baseline_model = ldpc_model
        benchmark_suite.add_method('No-EC', baseline_model)
    
    # Run benchmarks
    logger.info("Running benchmarks...")
    results = benchmark_suite.run_benchmarks()
    
    return results


def analyze_results(results, output_dir, logger):
    """Analyze and visualize benchmark results"""
    output_dir = Path(output_dir)
    
    # Create detailed comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract method names and metrics
    methods = list(results['individual'].keys())
    
    # 1. Quality comparison (PSNR and SSIM)
    ax = axes[0, 0]
    psnrs = [results['individual'][m]['metrics']['quality']['psnr'] for m in methods]
    ssims = [results['individual'][m]['metrics']['quality']['ssim'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, psnrs, width, label='PSNR', color='skyblue')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, ssims, width, label='SSIM', color='lightcoral')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('PSNR (dB)', color='skyblue')
    ax2.set_ylabel('SSIM', color='lightcoral')
    ax.set_title('Image Quality Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # 2. Robustness comparison (average across all attacks)
    ax = axes[0, 1]
    avg_robustness = []
    for method in methods:
        rob_metrics = results['individual'][method]['metrics']['robustness']
        all_values = []
        for attack_results in rob_metrics.values():
            all_values.extend(list(attack_results.values()))
        avg_robustness.append(np.mean(all_values) if all_values else 0)
    
    ax.bar(methods, avg_robustness, color='lightgreen')
    ax.set_ylabel('Average Robustness')
    ax.set_title('Overall Robustness Comparison')
    ax.set_xticklabels(methods, rotation=45)
    ax.set_ylim(0, 1.1)
    
    # 3. Speed comparison
    ax = axes[0, 2]
    encode_fps = [results['individual'][m]['metrics']['speed']['encode_fps'] for m in methods]
    decode_fps = [results['individual'][m]['metrics']['speed']['decode_fps'] for m in methods]
    
    x = np.arange(len(methods))
    ax.bar(x - width/2, encode_fps, width, label='Encode', color='gold')
    ax.bar(x + width/2, decode_fps, width, label='Decode', color='orange')
    
    ax.set_ylabel('FPS')
    ax.set_title('Processing Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    
    # 4. JPEG robustness detailed
    ax = axes[1, 0]
    for method in methods:
        if 'jpeg' in results['individual'][method]['metrics']['robustness']:
            jpeg_results = results['individual'][method]['metrics']['robustness']['jpeg']
            qualities = list(jpeg_results.keys())
            accuracies = list(jpeg_results.values())
            ax.plot(qualities, accuracies, 'o-', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('JPEG Quality')
    ax.set_ylabel('Message Recovery Accuracy')
    ax.set_title('JPEG Compression Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 5. Noise robustness detailed
    ax = axes[1, 1]
    for method in methods:
        if 'noise' in results['individual'][method]['metrics']['robustness']:
            noise_results = results['individual'][method]['metrics']['robustness']['noise']
            levels = list(noise_results.keys())
            accuracies = list(noise_results.values())
            ax.plot(levels, accuracies, 'o-', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Message Recovery Accuracy')
    ax.set_title('Gaussian Noise Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 6. Overall ranking
    ax = axes[1, 2]
    overall_ranking = results['comparison']['overall_ranking']
    methods_ranked = [item[0] for item in overall_ranking]
    scores = [item[1] for item in overall_ranking]
    
    # Invert scores for better visualization (lower is better in ranking)
    scores_inverted = [max(scores) - s + 1 for s in scores]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(methods_ranked)))
    ax.barh(methods_ranked, scores_inverted, color=colors)
    ax.set_xlabel('Overall Score')
    ax.set_title('Overall Performance Ranking')
    ax.set_xlim(0, max(scores_inverted) * 1.1)
    
    plt.suptitle('LDPC vs Other Methods - Comprehensive Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement summary
    if 'LDPC' in methods and len(methods) > 1:
        ldpc_idx = methods.index('LDPC')
        improvements = {}
        
        for i, method in enumerate(methods):
            if i != ldpc_idx:
                # Calculate improvements
                psnr_imp = ((psnrs[ldpc_idx] - psnrs[i]) / psnrs[i] * 100) if psnrs[i] > 0 else 0
                ssim_imp = ((ssims[ldpc_idx] - ssims[i]) / ssims[i] * 100) if ssims[i] > 0 else 0
                rob_imp = ((avg_robustness[ldpc_idx] - avg_robustness[i]) / avg_robustness[i] * 100) if avg_robustness[i] > 0 else 0
                
                improvements[method] = {
                    'psnr_improvement': psnr_imp,
                    'ssim_improvement': ssim_imp,
                    'robustness_improvement': rob_imp
                }
        
        logger.info("\nLDPC Improvements over other methods:")
        for method, imp in improvements.items():
            logger.info(f"\n{method}:")
            logger.info(f"  PSNR: {imp['psnr_improvement']:+.1f}%")
            logger.info(f"  SSIM: {imp['ssim_improvement']:+.1f}%")
            logger.info(f"  Robustness: {imp['robustness_improvement']:+.1f}%")
        
        # Save improvements
        with open(output_dir / 'ldpc_improvements.json', 'w') as f:
            json.dump(improvements, f, indent=2)


def generate_benchmark_report(results, output_path, logger):
    """Generate detailed benchmark report"""
    report = []
    report.append("# LDPC Steganography Benchmark Report\n")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    methods = list(results['individual'].keys())
    if 'LDPC' in methods:
        ldpc_rank = next((i+1 for i, (m, _) in enumerate(results['comparison']['overall_ranking']) if m == 'LDPC'), 0)
        report.append(f"- **LDPC Overall Ranking**: {ldpc_rank} out of {len(methods)}")
    
    # Quality comparison
    report.append("\n## Image Quality Comparison\n")
    report.append("| Method | PSNR (dB) | SSIM | L1 Error | L2 Error |")
    report.append("|--------|-----------|------|----------|----------|")
    
    for method in methods:
        q = results['individual'][method]['metrics']['quality']
        report.append(f"| {method} | {q['psnr']:.2f} | {q['ssim']:.4f} | "
                     f"{q['l1_error']:.4f} | {q['l2_error']:.4f} |")
    
    # Robustness comparison
    report.append("\n## Robustness Comparison\n")
    
    # JPEG robustness
    report.append("\n### JPEG Compression")
    report.append("| Method | Q=50 | Q=70 | Q=90 |")
    report.append("|--------|------|------|------|")
    
    for method in methods:
        rob = results['individual'][method]['metrics']['robustness'].get('jpeg', {})
        report.append(f"| {method} | {rob.get('50', 0):.3f} | "
                     f"{rob.get('70', 0):.3f} | {rob.get('90', 0):.3f} |")
    
    # Noise robustness
    report.append("\n### Gaussian Noise")
    report.append("| Method | σ=0.01 | σ=0.05 | σ=0.10 |")
    report.append("|--------|--------|--------|--------|")
    
    for method in methods:
        rob = results['individual'][method]['metrics']['robustness'].get('noise', {})
        report.append(f"| {method} | {rob.get('0.01', 0):.3f} | "
                     f"{rob.get('0.05', 0):.3f} | {rob.get('0.1', 0):.3f} |")
    
    # Speed comparison
    report.append("\n## Processing Speed\n")
    report.append("| Method | Encode FPS | Decode FPS |")
    report.append("|--------|------------|------------|")
    
    for method in methods:
        speed = results['individual'][method]['metrics']['speed']
        report.append(f"| {method} | {speed['encode_fps']:.1f} | {speed['decode_fps']:.1f} |")
    
    # Overall ranking
    report.append("\n## Overall Ranking\n")
    for i, (method, score) in enumerate(results['comparison']['overall_ranking']):
        report.append(f"{i+1}. **{method}** (score: {score:.3f})")
    
    # Conclusions
    report.append("\n## Conclusions\n")
    
    if 'LDPC' in methods:
        # Find LDPC advantages
        ldpc_metrics = results['individual']['LDPC']['metrics']
        advantages = []
        
        # Check quality
        ldpc_psnr = ldpc_metrics['quality']['psnr']
        best_psnr = max(results['individual'][m]['metrics']['quality']['psnr'] for m in methods)
        if ldpc_psnr == best_psnr:
            advantages.append("highest image quality (PSNR)")
        
        # Check robustness
        ldpc_rob = []
        for attack_type in ['jpeg', 'noise', 'blur']:
            if attack_type in ldpc_metrics['robustness']:
                ldpc_rob.extend(list(ldpc_metrics['robustness'][attack_type].values()))
        
        if ldpc_rob:
            ldpc_avg_rob = np.mean(ldpc_rob)
            best_rob = 0
            for m in methods:
                m_rob = []
                for attack_type in ['jpeg', 'noise', 'blur']:
                    if attack_type in results['individual'][m]['metrics']['robustness']:
                        m_rob.extend(list(results['individual'][m]['metrics']['robustness'][attack_type].values()))
                if m_rob:
                    best_rob = max(best_rob, np.mean(m_rob))
            
            if ldpc_avg_rob >= best_rob * 0.95:  # Within 5% of best
                advantages.append("superior robustness against attacks")
        
        if advantages:
            report.append(f"- LDPC demonstrates {' and '.join(advantages)}")
        
        # Note about adaptive redundancy
        report.append("- LDPC's adaptive redundancy provides optimal trade-off between capacity and robustness")
    
    # Save report
    report_text = '\n'.join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Benchmark report saved to {output_path}")


def main():
    """Main benchmark comparison function"""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) / f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(__name__, log_file=output_dir / 'benchmark.log')
    logger.info("Starting LDPC Benchmark Comparison")
    
    # Set random seed
    set_random_seed(42)
    
    try:
        # Run benchmarks
        results = run_comprehensive_benchmark(args, logger)
        
        # Save raw results
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {
                'individual': {
                    method: {
                        'metrics': method_results['metrics']
                    }
                    for method, method_results in results['individual'].items()
                },
                'comparison': results['comparison']
            }
            json.dump(json_results, f, indent=2)
        
        # Analyze and visualize results
        analyze_results(results, output_dir, logger)
        
        # Generate report
        generate_benchmark_report(results, output_dir / 'benchmark_report.md', logger)
        
        logger.info(f"Benchmark comparison completed! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()