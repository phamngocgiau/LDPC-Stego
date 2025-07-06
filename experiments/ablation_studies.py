#!/usr/bin/env python3
"""
Ablation Studies for LDPC Steganography
Analyze the contribution of different components
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
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.ldpc_config import LDPCTrainingConfig
from core.adaptive_ldpc import AdaptiveLDPC, create_ldpc_system
from models.steganography_model import AdvancedSteganographyModelWithLDPC
from data.data_loaders import create_data_loaders
from training.trainer import LDPCTrainer
from evaluation.evaluator import ModelEvaluator
from utils.logging_utils import setup_logger
from utils.helpers import set_random_seed, count_parameters


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LDPC Steganography Ablation Studies')
    
    # Base configuration
    parser.add_argument('--base-config', type=str, default=None,
                      help='Base configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Base checkpoint for fine-tuning studies')
    
    # Data arguments
    parser.add_argument('--data-train', type=str, default='data/train',
                      help='Path to training data')
    parser.add_argument('--data-val', type=str, default='data/val',
                      help='Path to validation data')
    
    # Ablation studies to run
    parser.add_argument('--ablate-ldpc', action='store_true',
                      help='Ablate LDPC component')
    parser.add_argument('--ablate-attention', action='store_true',
                      help='Ablate attention mechanisms')
    parser.add_argument('--ablate-cvae', action='store_true',
                      help='Ablate recovery CVAE')
    parser.add_argument('--ablate-losses', action='store_true',
                      help='Ablate different loss components')
    parser.add_argument('--ablate-architecture', action='store_true',
                      help='Ablate architectural components')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs for each ablation')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='results/ablations',
                      help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()


class AblationStudy:
    """Run ablation studies"""
    
    def __init__(self, args, base_config):
        self.args = args
        self.base_config = base_config
        self.results = {}
        
        # Setup output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(args.output_dir) / f'ablation_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(__name__, log_file=self.output_dir / 'ablation.log')
        self.logger.info("Starting ablation studies")
        
        # Create data loaders
        self.train_loader, self.val_loader, _ = create_data_loaders(
            train_dir=args.data_train,
            val_dir=args.data_val,
            image_size=base_config.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            message_length=base_config.message_length,
            augment_train=True
        )
    
    def run_all_ablations(self):
        """Run all requested ablation studies"""
        # Baseline model
        self.logger.info("Training baseline model...")
        baseline_results = self.train_model(self.base_config, "baseline")
        self.results['baseline'] = baseline_results
        
        # LDPC ablation
        if self.args.ablate_ldpc:
            self.logger.info("Running LDPC ablation...")
            ldpc_results = self.ablate_ldpc()
            self.results.update(ldpc_results)
        
        # Attention ablation
        if self.args.ablate_attention:
            self.logger.info("Running attention ablation...")
            attention_results = self.ablate_attention()
            self.results.update(attention_results)
        
        # CVAE ablation
        if self.args.ablate_cvae:
            self.logger.info("Running CVAE ablation...")
            cvae_results = self.ablate_cvae()
            self.results.update(cvae_results)
        
        # Loss ablation
        if self.args.ablate_losses:
            self.logger.info("Running loss ablation...")
            loss_results = self.ablate_losses()
            self.results.update(loss_results)
        
        # Architecture ablation
        if self.args.ablate_architecture:
            self.logger.info("Running architecture ablation...")
            arch_results = self.ablate_architecture()
            self.results.update(arch_results)
        
        # Analyze results
        self.analyze_results()
        
        # Generate report
        self.generate_report()
    
    def train_model(self, config, name: str) -> Dict[str, Any]:
        """Train a model with given configuration"""
        self.logger.info(f"Training model: {name}")
        
        # Create model
        if hasattr(config, 'use_ldpc') and config.use_ldpc:
            ldpc_system = create_ldpc_system(config)
            model = AdvancedSteganographyModelWithLDPC(config, ldpc_system)
            trainer_class = LDPCTrainer
            trainer_kwargs = {'ldpc_system': ldpc_system}
        else:
            from models.steganography_model import AdvancedSteganographyModel
            model = AdvancedSteganographyModel(config)
            from training.trainer import Trainer
            trainer_class = Trainer
            trainer_kwargs = {}
        
        # Count parameters
        num_params = count_parameters(model)
        self.logger.info(f"Model {name} parameters: {num_params:,}")
        
        # Create trainer
        trainer = trainer_class(
            config=config,
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.args.device,
            **trainer_kwargs
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        evaluator = ModelEvaluator(model, self.val_loader, config, device=self.args.device)
        eval_results = evaluator.evaluate_basic()
        
        # Test robustness
        robustness_results = evaluator.test_robustness(num_samples=20)
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_{name}.pth'
        trainer.save_checkpoint()
        
        return {
            'name': name,
            'num_params': num_params,
            'eval_results': eval_results,
            'robustness_results': robustness_results,
            'checkpoint': str(checkpoint_path)
        }
    
    def ablate_ldpc(self) -> Dict[str, Any]:
        """Ablate LDPC component"""
        results = {}
        
        # 1. No error correction
        config_no_ec = self.create_modified_config(self.base_config)
        config_no_ec.use_ldpc = False
        config_no_ec.loss_weights['decoded_message'] = 0
        results['no_error_correction'] = self.train_model(config_no_ec, 'no_ec')
        
        # 2. Fixed redundancy LDPC
        config_fixed = self.create_modified_config(self.base_config)
        config_fixed.ldpc_min_redundancy = 0.3
        config_fixed.ldpc_max_redundancy = 0.3  # Fixed at 30%
        results['ldpc_fixed_redundancy'] = self.train_model(config_fixed, 'ldpc_fixed')
        
        # 3. Without neural decoder
        config_no_neural = self.create_modified_config(self.base_config)
        config_no_neural.ldpc_use_neural_decoder = False
        results['ldpc_no_neural'] = self.train_model(config_no_neural, 'ldpc_no_neural')
        
        return results
    
    def ablate_attention(self) -> Dict[str, Any]:
        """Ablate attention mechanisms"""
        results = {}
        
        # 1. No attention
        config_no_attn = self.create_modified_config(self.base_config)
        config_no_attn.attention_layers = []
        results['no_attention'] = self.train_model(config_no_attn, 'no_attention')
        
        # 2. Self-attention only
        config_self_only = self.create_modified_config(self.base_config)
        config_self_only.use_cross_attention = False
        results['self_attention_only'] = self.train_model(config_self_only, 'self_attn_only')
        
        # 3. Different attention positions
        config_early_attn = self.create_modified_config(self.base_config)
        config_early_attn.attention_layers = [2, 4, 6, 8]  # Earlier layers
        results['early_attention'] = self.train_model(config_early_attn, 'early_attn')
        
        return results
    
    def ablate_cvae(self) -> Dict[str, Any]:
        """Ablate recovery CVAE"""
        results = {}
        
        # 1. No recovery network
        config_no_recovery = self.create_modified_config(self.base_config)
        config_no_recovery.use_recovery_network = False
        config_no_recovery.loss_weights['recovery_mse'] = 0
        config_no_recovery.loss_weights['recovery_kl'] = 0
        results['no_recovery'] = self.train_model(config_no_recovery, 'no_recovery')
        
        # 2. Simple recovery (no VAE)
        config_simple_recovery = self.create_modified_config(self.base_config)
        config_simple_recovery.use_vae_recovery = False
        config_simple_recovery.loss_weights['recovery_kl'] = 0
        results['simple_recovery'] = self.train_model(config_simple_recovery, 'simple_recovery')
        
        return results
    
    def ablate_losses(self) -> Dict[str, Any]:
        """Ablate different loss components"""
        results = {}
        
        # Loss components to ablate
        loss_ablations = [
            ('no_perceptual', {'perceptual': 0}),
            ('no_lpips', {'lpips': 0}),
            ('no_ssim', {'ssim': 0}),
            ('no_adversarial', {'adversarial': 0}),
            ('high_message', {'message': 20.0}),  # Double message weight
            ('low_message', {'message': 6.0}),     # Half message weight
        ]
        
        for name, loss_mods in loss_ablations:
            config = self.create_modified_config(self.base_config)
            for loss_name, weight in loss_mods.items():
                config.loss_weights[loss_name] = weight
            results[f'loss_{name}'] = self.train_model(config, f'loss_{name}')
        
        return results
    
    def ablate_architecture(self) -> Dict[str, Any]:
        """Ablate architectural components"""
        results = {}
        
        # 1. Shallower UNet
        config_shallow = self.create_modified_config(self.base_config)
        config_shallow.unet_depth = 3  # vs default 5
        results['shallow_unet'] = self.train_model(config_shallow, 'shallow_unet')
        
        # 2. Fewer channels
        config_narrow = self.create_modified_config(self.base_config)
        config_narrow.unet_base_channels = 32  # vs default 64
        results['narrow_unet'] = self.train_model(config_narrow, 'narrow_unet')
        
        # 3. Single UNet (no dual architecture)
        config_single = self.create_modified_config(self.base_config)
        config_single.use_dual_unet = False
        results['single_unet'] = self.train_model(config_single, 'single_unet')
        
        # 4. Different activation
        config_relu = self.create_modified_config(self.base_config)
        config_relu.activation = 'relu'  # vs default gelu
        results['relu_activation'] = self.train_model(config_relu, 'relu_activation')
        
        return results
    
    def create_modified_config(self, base_config):
        """Create a modified copy of the base configuration"""
        import copy
        config = copy.deepcopy(base_config)
        
        # Reduce epochs for ablation studies
        config.num_epochs = self.args.epochs
        config.learning_rate = self.args.lr
        
        # Ensure all necessary attributes exist
        if not hasattr(config, 'use_ldpc'):
            config.use_ldpc = True
        if not hasattr(config, 'use_cross_attention'):
            config.use_cross_attention = True
        if not hasattr(config, 'use_recovery_network'):
            config.use_recovery_network = True
        if not hasattr(config, 'use_vae_recovery'):
            config.use_vae_recovery = True
        if not hasattr(config, 'use_dual_unet'):
            config.use_dual_unet = True
        
        return config
    
    def analyze_results(self):
        """Analyze ablation results"""
        # Create comparison plots
        self.plot_performance_comparison()
        self.plot_robustness_comparison()
        self.plot_parameter_efficiency()
        
        # Statistical analysis
        self.statistical_analysis()
    
    def plot_performance_comparison(self):
        """Plot performance metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics
        models = list(self.results.keys())
        psnrs = [self.results[m]['eval_results']['psnr'] for m in models]
        ssims = [self.results[m]['eval_results']['ssim'] for m in models]
        accuracies = [self.results[m]['eval_results']['message_accuracy'] for m in models]
        bers = [self.results[m]['eval_results']['bit_error_rate'] for m in models]
        
        # PSNR comparison
        ax = axes[0, 0]
        bars = ax.bar(models, psnrs, color='skyblue')
        ax.axhline(y=psnrs[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        # SSIM comparison
        ax = axes[0, 1]
        bars = ax.bar(models, ssims, color='lightcoral')
        ax.axhline(y=ssims[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_ylabel('SSIM')
        ax.set_title('SSIM Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        # Message accuracy
        ax = axes[1, 0]
        bars = ax.bar(models, accuracies, color='lightgreen')
        ax.axhline(y=accuracies[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_ylabel('Message Accuracy')
        ax.set_title('Message Accuracy Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        # BER comparison
        ax = axes[1, 1]
        bars = ax.bar(models, bers, color='gold')
        ax.axhline(y=bers[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_ylabel('Bit Error Rate')
        ax.set_title('BER Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')
        ax.legend()
        
        plt.suptitle('Ablation Study - Performance Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_performance.png', dpi=300)
        plt.close()
    
    def plot_robustness_comparison(self):
        """Plot robustness comparison"""
        # Average robustness across all attacks
        models = list(self.results.keys())
        avg_robustness = []
        
        for model in models:
            rob_values = list(self.results[model]['robustness_results'].values())
            avg_robustness.append(np.mean(rob_values) if rob_values else 0)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, avg_robustness, color='purple', alpha=0.7)
        plt.axhline(y=avg_robustness[0], color='r', linestyle='--', alpha=0.5, label='Baseline')
        plt.ylabel('Average Robustness')
        plt.title('Robustness Comparison Across All Attacks')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_robustness.png', dpi=300)
        plt.close()
    
    def plot_parameter_efficiency(self):
        """Plot parameter efficiency analysis"""
        models = list(self.results.keys())
        params = [self.results[m]['num_params'] for m in models]
        accuracies = [self.results[m]['eval_results']['message_accuracy'] for m in models]
        
        # Efficiency score: accuracy per million parameters
        efficiency = [acc / (p / 1e6) for acc, p in zip(accuracies, params)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parameters vs Accuracy
        ax1.scatter(params, accuracies, s=100)
        for i, model in enumerate(models):
            ax1.annotate(model, (params[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('Message Accuracy')
        ax1.set_title('Model Size vs Performance')
        ax1.grid(True, alpha=0.3)
        
        # Efficiency comparison
        ax2.bar(models, efficiency, color='green', alpha=0.7)
        ax2.set_ylabel('Efficiency (Accuracy per Million Params)')
        ax2.set_title('Parameter Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_efficiency.png', dpi=300)
        plt.close()
    
    def statistical_analysis(self):
        """Perform statistical analysis of results"""
        import scipy.stats as stats
        
        analysis = {}
        baseline_acc = self.results['baseline']['eval_results']['message_accuracy']
        
        # Compare each ablation to baseline
        for model_name, model_results in self.results.items():
            if model_name == 'baseline':
                continue
            
            model_acc = model_results['eval_results']['message_accuracy']
            
            # Calculate relative change
            relative_change = (model_acc - baseline_acc) / baseline_acc * 100
            
            # Statistical significance (would need multiple runs for proper test)
            # Here we use a simple threshold
            significant = abs(relative_change) > 5  # 5% threshold
            
            analysis[model_name] = {
                'relative_change': relative_change,
                'significant': significant,
                'better_than_baseline': model_acc > baseline_acc
            }
        
        # Save analysis
        with open(self.output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Log significant findings
        self.logger.info("\nStatistical Analysis Results:")
        for model, stats in analysis.items():
            if stats['significant']:
                direction = "better" if stats['better_than_baseline'] else "worse"
                self.logger.info(f"{model}: {stats['relative_change']:+.1f}% ({direction} than baseline)")
    
    def generate_report(self):
        """Generate comprehensive ablation study report"""
        report = []
        report.append("# LDPC Steganography Ablation Study Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        
        # Find most important components
        baseline_acc = self.results['baseline']['eval_results']['message_accuracy']
        
        largest_drops = []
        for model_name, results in self.results.items():
            if model_name == 'baseline':
                continue
            acc = results['eval_results']['message_accuracy']
            drop = baseline_acc - acc
            if drop > 0:
                largest_drops.append((model_name, drop))
        
        largest_drops.sort(key=lambda x: x[1], reverse=True)
        
        if largest_drops:
            report.append("### Most Important Components (by accuracy drop when removed):\n")
            for i, (component, drop) in enumerate(largest_drops[:5]):
                report.append(f"{i+1}. **{component}**: -{drop:.3f} accuracy drop")
        
        # Detailed Results
        report.append("\n## Detailed Results\n")
        
        # Create results table
        report.append("| Model | PSNR | SSIM | Accuracy | BER | Parameters |")
        report.append("|-------|------|------|----------|-----|------------|")
        
        for model_name, results in self.results.items():
            eval_res = results['eval_results']
            report.append(f"| {model_name} | {eval_res['psnr']:.2f} | "
                         f"{eval_res['ssim']:.4f} | {eval_res['message_accuracy']:.4f} | "
                         f"{eval_res['bit_error_rate']:.6f} | {results['num_params']:,} |")
        
        # Component Analysis
        report.append("\n## Component Analysis\n")
        
        # LDPC analysis
        if 'no_error_correction' in self.results:
            report.append("\n### LDPC Error Correction")
            no_ec_acc = self.results['no_error_correction']['eval_results']['message_accuracy']
            improvement = (baseline_acc - no_ec_acc) / no_ec_acc * 100
            report.append(f"- LDPC provides **{improvement:.1f}%** improvement in message accuracy")
            
            if 'ldpc_no_neural' in self.results:
                neural_acc = baseline_acc
                no_neural_acc = self.results['ldpc_no_neural']['eval_results']['message_accuracy']
                neural_improvement = (neural_acc - no_neural_acc) / no_neural_acc * 100
                report.append(f"- Neural decoder adds **{neural_improvement:.1f}%** additional improvement")
        
        # Attention analysis
        if 'no_attention' in self.results:
            report.append("\n### Attention Mechanisms")
            no_attn_psnr = self.results['no_attention']['eval_results']['psnr']
            psnr_drop = self.results['baseline']['eval_results']['psnr'] - no_attn_psnr
            report.append(f"- Removing attention decreases PSNR by **{psnr_drop:.2f} dB**")
        
        # Recovery network analysis
        if 'no_recovery' in self.results:
            report.append("\n### Recovery Network")
            baseline_rob = np.mean(list(self.results['baseline']['robustness_results'].values()))
            no_recovery_rob = np.mean(list(self.results['no_recovery']['robustness_results'].values()))
            rob_drop = (baseline_rob - no_recovery_rob) / baseline_rob * 100
            report.append(f"- Recovery network improves robustness by **{rob_drop:.1f}%**")
        
        # Conclusions
        report.append("\n## Conclusions\n")
        
        # Identify critical components
        critical_components = []
        for model_name, results in self.results.items():
            if model_name == 'baseline':
                continue
            acc_drop = baseline_acc - results['eval_results']['message_accuracy']
            if acc_drop > 0.05:  # More than 5% drop
                critical_components.append(model_name.replace('_', ' ').title())
        
        if critical_components:
            report.append(f"- Critical components: {', '.join(critical_components)}")
        
        # Architecture insights
        if 'shallow_unet' in self.results and 'narrow_unet' in self.results:
            shallow_params = self.results['shallow_unet']['num_params']
            narrow_params = self.results['narrow_unet']['num_params']
            baseline_params = self.results['baseline']['num_params']
            
            report.append(f"\n- Shallow UNet reduces parameters by {(1 - shallow_params/baseline_params)*100:.1f}%")
            report.append(f"- Narrow UNet reduces parameters by {(1 - narrow_params/baseline_params)*100:.1f}%")
        
        # Save report
        report_text = '\n'.join(report)
        with open(self.output_dir / 'ablation_report.md', 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Ablation report saved to {self.output_dir / 'ablation_report.md'}")
        
        # Save all results
        results_to_save = {}
        for model_name, results in self.results.items():
            results_copy = results.copy()
            results_copy.pop('checkpoint', None)  # Remove checkpoint path
            results_to_save[model_name] = results_copy
        
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)


def main():
    """Main ablation study function"""
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load base configuration
    if args.base_config:
        base_config = LDPCTrainingConfig.load(args.base_config)
    else:
        # Create default configuration
        base_config = LDPCTrainingConfig(
            data_train_folder=args.data_train,
            data_val_folder=args.data_val,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )
    
    # Update configuration
    base_config.num_epochs = args.epochs
    base_config.device = args.device
    
    # Create ablation study
    ablation_study = AblationStudy(args, base_config)
    
    try:
        # Run ablations
        ablation_study.run_all_ablations()
        
        print(f"\nAblation studies completed! Results saved to {ablation_study.output_dir}")
        
    except Exception as e:
        logging.error(f"Ablation study failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()