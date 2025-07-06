#!/usr/bin/env python3
"""
Model Evaluator
Comprehensive evaluation of LDPC steganography models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..training.metrics import ImageMetrics, MessageMetrics, RobustnessMetrics
from ..training.attacks import AttackSimulator
from ..utils.visualization import plot_results, save_sample_images
from ..utils.helpers import calculate_psnr, calculate_ssim, calculate_lpips


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, test_loader: DataLoader, config, device='cuda'):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            test_loader: Test data loader
            config: Configuration object
            device: Evaluation device
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Metrics
        self.image_metrics = ImageMetrics()
        self.message_metrics = MessageMetrics()
        self.robustness_metrics = RobustnessMetrics()
        
        # Attack simulator
        self.attack_simulator = AttackSimulator(device)
        
        # Results storage
        self.results = defaultdict(list)
        
        logging.info("Model evaluator initialized")
    
    def evaluate(self) -> Dict[str, Any]:
        """Run complete evaluation"""
        logging.info("Starting model evaluation...")
        
        # Basic evaluation
        basic_results = self.evaluate_basic()
        
        # Robustness evaluation
        robustness_results = self.evaluate_robustness()
        
        # Capacity analysis
        capacity_results = self.evaluate_capacity()
        
        # Visual quality analysis
        visual_results = self.evaluate_visual_quality()
        
        # Combine results
        results = {
            'basic': basic_results,
            'robustness': robustness_results,
            'capacity': capacity_results,
            'visual': visual_results
        }
        
        # Save results
        self.save_results(results)
        
        return results
    
    def evaluate_basic(self) -> Dict[str, float]:
        """Basic evaluation metrics"""
        self.model.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Basic evaluation'):
                cover_images = batch['cover'].to(self.device)
                messages = batch['message'].to(self.device)
                
                # Generate stego images
                outputs = self.model(cover_images, messages)
                stego_images = outputs['stego_images']
                extracted_messages = outputs['extracted_messages']
                
                # Image quality metrics
                psnr = calculate_psnr(stego_images, cover_images)
                ssim = calculate_ssim(stego_images, cover_images)
                
                # Message accuracy
                message_acc = self.message_metrics.calculate(
                    extracted_messages, messages
                )
                
                # Store metrics
                metrics['psnr'].append(psnr.item())
                metrics['ssim'].append(ssim.item())
                metrics['message_accuracy'].append(message_acc['accuracy'])
                metrics['bit_error_rate'].append(message_acc['ber'])
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        logging.info(f"Basic evaluation - PSNR: {avg_metrics['psnr']:.2f}, "
                    f"SSIM: {avg_metrics['ssim']:.4f}, "
                    f"Message Acc: {avg_metrics['message_accuracy']:.4f}")
        
        return avg_metrics
    
    def evaluate_robustness(self) -> Dict[str, Dict[str, float]]:
        """Evaluate robustness against various attacks"""
        self.model.eval()
        
        attack_configs = [
            ('jpeg', [30, 50, 70, 90]),
            ('noise', [0.01, 0.02, 0.05, 0.1]),
            ('blur', [1, 3, 5, 7]),
            ('crop', [0.9, 0.8, 0.7, 0.6]),
            ('rotation', [5, 10, 15, 20]),
            ('scaling', [0.5, 0.75, 1.25, 1.5])
        ]
        
        results = {}
        
        for attack_type, strengths in attack_configs:
            attack_results = {}
            
            for strength in strengths:
                accuracies = []
                
                with torch.no_grad():
                    for batch in self.test_loader:
                        cover_images = batch['cover'].to(self.device)
                        messages = batch['message'].to(self.device)
                        
                        # Generate stego images
                        outputs = self.model(cover_images, messages)
                        stego_images = outputs['stego_images']
                        
                        # Apply attack
                        if attack_type == 'jpeg':
                            attacked = self.attack_simulator.jpeg_compression(stego_images, int(strength))
                        elif attack_type == 'noise':
                            attacked = self.attack_simulator.gaussian_noise(stego_images, strength)
                        elif attack_type == 'blur':
                            attacked = self.attack_simulator.gaussian_blur(stego_images, int(strength))
                        elif attack_type == 'crop':
                            attacked = self.attack_simulator.random_crop(stego_images, strength)
                        elif attack_type == 'rotation':
                            attacked = self.attack_simulator.rotation(stego_images, strength)
                        elif attack_type == 'scaling':
                            attacked = self.attack_simulator.scaling(stego_images, strength)
                        else:
                            attacked = stego_images
                        
                        # Extract messages from attacked images
                        extracted = self.model.extract_message(attacked)
                        
                        # Calculate accuracy
                        acc = self.message_metrics.calculate(extracted, messages)
                        accuracies.append(acc['accuracy'])
                
                attack_results[str(strength)] = np.mean(accuracies)
            
            results[attack_type] = attack_results
            
            logging.info(f"Robustness {attack_type}: {attack_results}")
        
        return results
    
    def evaluate_capacity(self) -> Dict[str, Any]:
        """Evaluate model capacity"""
        self.model.eval()
        
        # Test different message lengths
        message_lengths = [256, 512, 1024, 2048, 4096]
        capacity_results = {}
        
        for msg_len in message_lengths:
            if msg_len > self.config.message_length:
                continue
            
            accuracies = []
            psnrs = []
            
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    if i >= 10:  # Test on subset
                        break
                    
                    cover_images = batch['cover'].to(self.device)
                    batch_size = cover_images.size(0)
                    
                    # Generate messages of specific length
                    messages = torch.randint(0, 2, (batch_size, msg_len), 
                                           dtype=torch.float32, device=self.device)
                    
                    # Pad to model's expected length
                    if msg_len < self.config.message_length:
                        padding = torch.zeros(batch_size, self.config.message_length - msg_len,
                                            device=self.device)
                        padded_messages = torch.cat([messages, padding], dim=1)
                    else:
                        padded_messages = messages
                    
                    # Encode and decode
                    outputs = self.model(cover_images, padded_messages)
                    stego_images = outputs['stego_images']
                    extracted = outputs['extracted_messages'][:, :msg_len]
                    
                    # Metrics
                    acc = (extracted > 0) == (messages > 0.5)
                    acc = acc.float().mean().item()
                    accuracies.append(acc)
                    
                    psnr = calculate_psnr(stego_images, cover_images)
                    psnrs.append(psnr.item())
            
            capacity_results[msg_len] = {
                'accuracy': np.mean(accuracies),
                'psnr': np.mean(psnrs),
                'bpp': msg_len / (self.config.image_size ** 2)  # Bits per pixel
            }
        
        logging.info(f"Capacity evaluation: {capacity_results}")
        
        return capacity_results
    
    def evaluate_visual_quality(self) -> Dict[str, Any]:
        """Detailed visual quality evaluation"""
        self.model.eval()
        
        # Sample a few batches for detailed analysis
        sample_results = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= 5:  # Analyze 5 batches
                    break
                
                cover_images = batch['cover'].to(self.device)
                messages = batch['message'].to(self.device)
                
                # Generate stego images
                outputs = self.model(cover_images, messages)
                stego_images = outputs['stego_images']
                
                # Calculate detailed metrics
                batch_results = {
                    'psnr': calculate_psnr(stego_images, cover_images).cpu().numpy(),
                    'ssim': calculate_ssim(stego_images, cover_images).cpu().numpy(),
                    'lpips': calculate_lpips(stego_images, cover_images).cpu().numpy(),
                    'l1_error': F.l1_loss(stego_images, cover_images, reduction='none').mean([1,2,3]).cpu().numpy(),
                    'l2_error': F.mse_loss(stego_images, cover_images, reduction='none').mean([1,2,3]).cpu().numpy()
                }
                
                # Frequency analysis
                cover_fft = torch.fft.fft2(cover_images.mean(dim=1))
                stego_fft = torch.fft.fft2(stego_images.mean(dim=1))
                freq_error = torch.abs(cover_fft - stego_fft).mean().item()
                batch_results['frequency_error'] = freq_error
                
                sample_results.append(batch_results)
        
        # Aggregate results
        visual_results = {}
        for key in sample_results[0].keys():
            values = [r[key] for r in sample_results]
            if isinstance(values[0], np.ndarray):
                values = np.concatenate(values)
            visual_results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return visual_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        output_dir = Path(self.config.output_dir) / 'evaluation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self.visualize_results(results, output_dir)
        
        logging.info(f"Results saved to {output_dir}")
    
    def visualize_results(self, results: Dict[str, Any], output_dir: Path):
        """Create result visualizations"""
        # Robustness plots
        if 'robustness' in results:
            self._plot_robustness(results['robustness'], output_dir)
        
        # Capacity plot
        if 'capacity' in results:
            self._plot_capacity(results['capacity'], output_dir)
        
        # Sample images
        self._save_sample_images(output_dir)
    
    def _plot_robustness(self, robustness_results: Dict, output_dir: Path):
        """Plot robustness results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (attack_type, attack_results) in enumerate(robustness_results.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            strengths = list(attack_results.keys())
            accuracies = list(attack_results.values())
            
            ax.plot(strengths, accuracies, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(f'{attack_type.capitalize()} Strength')
            ax.set_ylabel('Message Recovery Accuracy')
            ax.set_title(f'Robustness to {attack_type.capitalize()}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_evaluation.png', dpi=300)
        plt.close()
    
    def _plot_capacity(self, capacity_results: Dict, output_dir: Path):
        """Plot capacity results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        msg_lengths = list(capacity_results.keys())
        accuracies = [v['accuracy'] for v in capacity_results.values()]
        psnrs = [v['psnr'] for v in capacity_results.values()]
        bpps = [v['bpp'] for v in capacity_results.values()]
        
        # Accuracy vs capacity
        ax1.plot(bpps, accuracies, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Capacity (bits per pixel)')
        ax1.set_ylabel('Message Recovery Accuracy')
        ax1.set_title('Accuracy vs Capacity')
        ax1.grid(True, alpha=0.3)
        
        # PSNR vs capacity
        ax2.plot(bpps, psnrs, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Capacity (bits per pixel)')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Image Quality vs Capacity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'capacity_evaluation.png', dpi=300)
        plt.close()
    
    def _save_sample_images(self, output_dir: Path):
        """Save sample images"""
        self.model.eval()
        
        with torch.no_grad():
            batch = next(iter(self.test_loader))
            cover_images = batch['cover'][:4].to(self.device)
            messages = batch['message'][:4].to(self.device)
            
            outputs = self.model(cover_images, messages)
            stego_images = outputs['stego_images']
            
            # Apply some attacks for visualization
            jpeg_attacked = self.attack_simulator.jpeg_compression(stego_images, 50)
            noise_attacked = self.attack_simulator.gaussian_noise(stego_images, 0.05)
            
            # Save comparison
            save_sample_images(
                cover_images, stego_images, 
                [jpeg_attacked, noise_attacked],
                ['Cover', 'Stego', 'JPEG-50', 'Noise-0.05'],
                output_dir / 'sample_images.png'
            )


class LDPCEvaluator(ModelEvaluator):
    """Specialized evaluator for LDPC models"""
    
    def __init__(self, model, ldpc_system, test_loader, config, device='cuda'):
        super().__init__(model, test_loader, config, device)
        self.ldpc_system = ldpc_system
    
    def evaluate_ldpc_performance(self) -> Dict[str, Any]:
        """Evaluate LDPC-specific performance"""
        self.model.eval()
        
        redundancy_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        ldpc_results = {}
        
        for redundancy in redundancy_levels:
            if redundancy < self.ldpc_system.min_redundancy or redundancy > self.ldpc_system.max_redundancy:
                continue
            
            # Get corresponding attack strength
            attack_strength = (redundancy - self.ldpc_system.min_redundancy) / \
                            (self.ldpc_system.max_redundancy - self.ldpc_system.min_redundancy)
            
            accuracies = []
            
            with torch.no_grad():
                for batch in self.test_loader:
                    cover_images = batch['cover'].to(self.device)
                    messages = batch['message'].to(self.device)
                    
                    # LDPC encode
                    encoded = self.ldpc_system.encode(messages.cpu().numpy(), attack_strength)
                    encoded = torch.tensor(encoded, device=self.device, dtype=torch.float32)
                    
                    # Generate stego
                    outputs = self.model(cover_images, encoded)
                    stego_images = outputs['stego_images']
                    
                    # Simulate attack
                    attack_type = np.random.choice(['jpeg', 'noise', 'blur'])
                    attacked = self.attack_simulator.apply_attack(
                        stego_images, attack_type, attack_strength
                    )
                    
                    # Extract and decode
                    extracted_encoded = self.model.extract_message(attacked)
                    decoded = self.ldpc_system.decode(
                        extracted_encoded.cpu().numpy(), attack_strength
                    )
                    decoded = torch.tensor(decoded, device=self.device)
                    
                    # Calculate accuracy
                    acc = (decoded > 0.5) == (messages > 0.5)
                    acc = acc.float().mean().item()
                    accuracies.append(acc)
            
            ldpc_results[redundancy] = {
                'accuracy': np.mean(accuracies),
                'code_rate': 1 - redundancy,
                'attack_strength': attack_strength
            }
        
        return ldpc_results