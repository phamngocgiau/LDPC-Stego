#!/usr/bin/env python3
"""
Benchmark Comparisons
Compare LDPC steganography with other methods
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..training.metrics import ImageMetrics, MessageMetrics
from ..training.attacks import AttackSimulator
from ..utils.helpers import calculate_psnr, calculate_ssim


class BenchmarkSuite:
    """Comprehensive benchmark suite for steganography methods"""
    
    def __init__(self, test_loader: DataLoader, device='cuda'):
        """
        Initialize benchmark suite
        
        Args:
            test_loader: Test data loader
            device: Evaluation device
        """
        self.test_loader = test_loader
        self.device = device
        
        # Metrics
        self.image_metrics = ImageMetrics()
        self.message_metrics = MessageMetrics()
        
        # Attack simulator
        self.attack_simulator = AttackSimulator(device)
        
        # Results storage
        self.results = {}
        
        logging.info("Benchmark suite initialized")
    
    def add_method(self, name: str, model: nn.Module, 
                   preprocessing_fn=None, postprocessing_fn=None):
        """
        Add a method to benchmark
        
        Args:
            name: Method name
            model: Model to benchmark
            preprocessing_fn: Optional preprocessing function
            postprocessing_fn: Optional postprocessing function
        """
        self.results[name] = {
            'model': model.to(self.device),
            'preprocessing': preprocessing_fn,
            'postprocessing': postprocessing_fn,
            'metrics': {}
        }
        logging.info(f"Added method: {name}")
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks"""
        logging.info("Starting benchmark suite...")
        
        for method_name in self.results.keys():
            logging.info(f"\nBenchmarking {method_name}...")
            
            # Image quality benchmark
            quality_results = self._benchmark_image_quality(method_name)
            self.results[method_name]['metrics']['quality'] = quality_results
            
            # Capacity benchmark
            capacity_results = self._benchmark_capacity(method_name)
            self.results[method_name]['metrics']['capacity'] = capacity_results
            
            # Robustness benchmark
            robustness_results = self._benchmark_robustness(method_name)
            self.results[method_name]['metrics']['robustness'] = robustness_results
            
            # Speed benchmark
            speed_results = self._benchmark_speed(method_name)
            self.results[method_name]['metrics']['speed'] = speed_results
        
        # Comparative analysis
        comparison = self._compare_methods()
        
        # Save results
        self._save_results()
        
        return {
            'individual': self.results,
            'comparison': comparison
        }
    
    def _benchmark_image_quality(self, method_name: str) -> Dict[str, float]:
        """Benchmark image quality metrics"""
        model = self.results[method_name]['model']
        model.eval()
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'l1_error': [],
            'l2_error': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f'{method_name} - Quality'):
                cover_images = batch['cover'].to(self.device)
                messages = batch['message'].to(self.device)
                
                # Preprocessing
                if self.results[method_name]['preprocessing']:
                    cover_images, messages = self.results[method_name]['preprocessing'](
                        cover_images, messages
                    )
                
                # Generate stego images
                if hasattr(model, 'encode'):
                    stego_images = model.encode(cover_images, messages)
                else:
                    outputs = model(cover_images, messages)
                    stego_images = outputs['stego_images'] if isinstance(outputs, dict) else outputs
                
                # Postprocessing
                if self.results[method_name]['postprocessing']:
                    stego_images = self.results[method_name]['postprocessing'](stego_images)
                
                # Calculate metrics
                metrics['psnr'].append(calculate_psnr(stego_images, cover_images).item())
                metrics['ssim'].append(calculate_ssim(stego_images, cover_images).item())
                metrics['l1_error'].append(F.l1_loss(stego_images, cover_images).item())
                metrics['l2_error'].append(F.mse_loss(stego_images, cover_images).item())
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        return avg_metrics
    
    def _benchmark_capacity(self, method_name: str) -> Dict[str, Any]:
        """Benchmark capacity at different message lengths"""
        model = self.results[method_name]['model']
        model.eval()
        
        message_lengths = [256, 512, 1024, 2048]
        capacity_results = {}
        
        for msg_len in message_lengths:
            accuracies = []
            
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    if i >= 10:  # Test on subset
                        break
                    
                    cover_images = batch['cover'].to(self.device)
                    batch_size = cover_images.size(0)
                    
                    # Generate messages
                    messages = torch.randint(0, 2, (batch_size, msg_len), 
                                           dtype=torch.float32, device=self.device)
                    
                    try:
                        # Encode
                        if hasattr(model, 'encode'):
                            stego_images = model.encode(cover_images, messages)
                            extracted = model.decode(stego_images)
                        else:
                            outputs = model(cover_images, messages)
                            stego_images = outputs['stego_images']
                            extracted = outputs['extracted_messages']
                        
                        # Calculate accuracy
                        if extracted.size(1) >= msg_len:
                            extracted = extracted[:, :msg_len]
                            acc = (extracted > 0) == (messages > 0.5)
                            acc = acc.float().mean().item()
                            accuracies.append(acc)
                    except:
                        # Method doesn't support this message length
                        accuracies.append(0.0)
            
            capacity_results[msg_len] = {
                'accuracy': np.mean(accuracies) if accuracies else 0.0,
                'supported': len(accuracies) > 0
            }
        
        return capacity_results
    
    def _benchmark_robustness(self, method_name: str) -> Dict[str, Dict[str, float]]:
        """Benchmark robustness against attacks"""
        model = self.results[method_name]['model']
        model.eval()
        
        attacks = {
            'jpeg': [50, 70, 90],
            'noise': [0.01, 0.05, 0.1],
            'blur': [1, 3, 5]
        }
        
        robustness_results = {}
        
        for attack_type, strengths in attacks.items():
            attack_results = {}
            
            for strength in strengths:
                accuracies = []
                
                with torch.no_grad():
                    for i, batch in enumerate(self.test_loader):
                        if i >= 5:  # Test on subset
                            break
                        
                        cover_images = batch['cover'].to(self.device)
                        messages = batch['message'].to(self.device)
                        
                        try:
                            # Generate stego
                            if hasattr(model, 'encode'):
                                stego_images = model.encode(cover_images, messages)
                            else:
                                outputs = model(cover_images, messages)
                                stego_images = outputs['stego_images']
                            
                            # Apply attack
                            if attack_type == 'jpeg':
                                attacked = self.attack_simulator.jpeg_compression(
                                    stego_images, int(strength)
                                )
                            elif attack_type == 'noise':
                                attacked = self.attack_simulator.gaussian_noise(
                                    stego_images, strength
                                )
                            elif attack_type == 'blur':
                                attacked = self.attack_simulator.gaussian_blur(
                                    stego_images, int(strength)
                                )
                            
                            # Extract messages
                            if hasattr(model, 'decode'):
                                extracted = model.decode(attacked)
                            elif hasattr(model, 'extract_message'):
                                extracted = model.extract_message(attacked)
                            else:
                                outputs = model(attacked, messages)
                                extracted = outputs['extracted_messages']
                            
                            # Calculate accuracy
                            acc = self.message_metrics.calculate(extracted, messages)
                            accuracies.append(acc['accuracy'])
                        except:
                            accuracies.append(0.0)
                
                attack_results[str(strength)] = np.mean(accuracies) if accuracies else 0.0
            
            robustness_results[attack_type] = attack_results
        
        return robustness_results
    
    def _benchmark_speed(self, method_name: str) -> Dict[str, float]:
        """Benchmark encoding/decoding speed"""
        model = self.results[method_name]['model']
        model.eval()
        
        encode_times = []
        decode_times = []
        batch_sizes = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= 10:  # Test on subset
                    break
                
                cover_images = batch['cover'].to(self.device)
                messages = batch['message'].to(self.device)
                batch_size = cover_images.size(0)
                batch_sizes.append(batch_size)
                
                # Warm up
                if i == 0:
                    for _ in range(3):
                        if hasattr(model, 'encode'):
                            _ = model.encode(cover_images, messages)
                        else:
                            _ = model(cover_images, messages)
                
                # Time encoding
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                if hasattr(model, 'encode'):
                    stego_images = model.encode(cover_images, messages)
                else:
                    outputs = model(cover_images, messages)
                    stego_images = outputs['stego_images']
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                encode_time = time.time() - start_time
                encode_times.append(encode_time)
                
                # Time decoding
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                if hasattr(model, 'decode'):
                    _ = model.decode(stego_images)
                elif hasattr(model, 'extract_message'):
                    _ = model.extract_message(stego_images)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                decode_time = time.time() - start_time
                decode_times.append(decode_time)
        
        # Calculate throughput
        total_images = sum(batch_sizes)
        total_encode_time = sum(encode_times)
        total_decode_time = sum(decode_times)
        
        return {
            'encode_fps': total_images / total_encode_time if total_encode_time > 0 else 0,
            'decode_fps': total_images / total_decode_time if total_decode_time > 0 else 0,
            'avg_encode_time': np.mean(encode_times) if encode_times else 0,
            'avg_decode_time': np.mean(decode_times) if decode_times else 0
        }
    
    def _compare_methods(self) -> Dict[str, Any]:
        """Compare all methods"""
        comparison = {
            'quality_ranking': self._rank_by_quality(),
            'robustness_ranking': self._rank_by_robustness(),
            'speed_ranking': self._rank_by_speed(),
            'overall_ranking': self._overall_ranking()
        }
        
        return comparison
    
    def _rank_by_quality(self) -> List[Tuple[str, float]]:
        """Rank methods by image quality"""
        quality_scores = []
        
        for method_name, results in self.results.items():
            if 'quality' in results['metrics']:
                # Combined quality score (higher PSNR and SSIM is better)
                psnr = results['metrics']['quality']['psnr']
                ssim = results['metrics']['quality']['ssim']
                score = 0.5 * (psnr / 50.0) + 0.5 * ssim  # Normalize PSNR to [0,1]
                quality_scores.append((method_name, score))
        
        return sorted(quality_scores, key=lambda x: x[1], reverse=True)
    
    def _rank_by_robustness(self) -> List[Tuple[str, float]]:
        """Rank methods by robustness"""
        robustness_scores = []
        
        for method_name, results in self.results.items():
            if 'robustness' in results['metrics']:
                # Average accuracy across all attacks
                all_accuracies = []
                for attack_type, attack_results in results['metrics']['robustness'].items():
                    all_accuracies.extend(list(attack_results.values()))
                score = np.mean(all_accuracies) if all_accuracies else 0
                robustness_scores.append((method_name, score))
        
        return sorted(robustness_scores, key=lambda x: x[1], reverse=True)
    
    def _rank_by_speed(self) -> List[Tuple[str, float]]:
        """Rank methods by speed"""
        speed_scores = []
        
        for method_name, results in self.results.items():
            if 'speed' in results['metrics']:
                # Combined speed score
                encode_fps = results['metrics']['speed']['encode_fps']
                decode_fps = results['metrics']['speed']['decode_fps']
                score = (encode_fps + decode_fps) / 2
                speed_scores.append((method_name, score))
        
        return sorted(speed_scores, key=lambda x: x[1], reverse=True)
    
    def _overall_ranking(self) -> List[Tuple[str, float]]:
        """Overall ranking combining all metrics"""
        overall_scores = {}
        
        # Get rankings
        quality_ranking = {name: rank for rank, (name, _) in enumerate(self._rank_by_quality())}
        robustness_ranking = {name: rank for rank, (name, _) in enumerate(self._rank_by_robustness())}
        speed_ranking = {name: rank for rank, (name, _) in enumerate(self._rank_by_speed())}
        
        # Combine rankings (lower is better)
        for method_name in self.results.keys():
            quality_rank = quality_ranking.get(method_name, len(self.results))
            robustness_rank = robustness_ranking.get(method_name, len(self.results))
            speed_rank = speed_ranking.get(method_name, len(self.results))
            
            # Weighted combination (quality: 40%, robustness: 40%, speed: 20%)
            overall_rank = 0.4 * quality_rank + 0.4 * robustness_rank + 0.2 * speed_rank
            overall_scores[method_name] = overall_rank
        
        # Sort by score (lower is better)
        return sorted(overall_scores.items(), key=lambda x: x[1])
    
    def _save_results(self):
        """Save benchmark results"""
        output_dir = Path('results/benchmarks')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON
        json_results = {}
        for method_name, method_results in self.results.items():
            json_results[method_name] = {
                'metrics': method_results['metrics']
            }
        
        # Save JSON
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(output_dir)
        
        logging.info(f"Benchmark results saved to {output_dir}")
    
    def _create_visualizations(self, output_dir: Path):
        """Create benchmark visualizations"""
        # Quality comparison
        self._plot_quality_comparison(output_dir)
        
        # Robustness comparison
        self._plot_robustness_comparison(output_dir)
        
        # Speed comparison
        self._plot_speed_comparison(output_dir)
        
        # Overall comparison
        self._plot_overall_comparison(output_dir)
    
    def _plot_quality_comparison(self, output_dir: Path):
        """Plot quality metrics comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = []
        psnrs = []
        ssims = []
        
        for method_name, results in self.results.items():
            if 'quality' in results['metrics']:
                methods.append(method_name)
                psnrs.append(results['metrics']['quality']['psnr'])
                ssims.append(results['metrics']['quality']['ssim'])
        
        # PSNR comparison
        ax1.bar(methods, psnrs)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('PSNR Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # SSIM comparison
        ax2.bar(methods, ssims)
        ax2.set_ylabel('SSIM')
        ax2.set_title('SSIM Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_comparison.png', dpi=300)
        plt.close()
    
    def _plot_robustness_comparison(self, output_dir: Path):
        """Plot robustness comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        attack_types = ['jpeg', 'noise', 'blur']
        
        for idx, attack_type in enumerate(attack_types):
            ax = axes[idx]
            
            for method_name, results in self.results.items():
                if 'robustness' in results['metrics'] and attack_type in results['metrics']['robustness']:
                    attack_results = results['metrics']['robustness'][attack_type]
                    strengths = list(attack_results.keys())
                    accuracies = list(attack_results.values())
                    
                    ax.plot(strengths, accuracies, 'o-', label=method_name)
            
            ax.set_xlabel(f'{attack_type.capitalize()} Strength')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Robustness to {attack_type.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_comparison.png', dpi=300)
        plt.close()
    
    def _plot_speed_comparison(self, output_dir: Path):
        """Plot speed comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = []
        encode_fps = []
        decode_fps = []
        
        for method_name, results in self.results.items():
            if 'speed' in results['metrics']:
                methods.append(method_name)
                encode_fps.append(results['metrics']['speed']['encode_fps'])
                decode_fps.append(results['metrics']['speed']['decode_fps'])
        
        # Encoding speed
        ax1.bar(methods, encode_fps)
        ax1.set_ylabel('FPS')
        ax1.set_title('Encoding Speed')
        ax1.tick_params(axis='x', rotation=45)
        
        # Decoding speed
        ax2.bar(methods, decode_fps)
        ax2.set_ylabel('FPS')
        ax2.set_title('Decoding Speed')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'speed_comparison.png', dpi=300)
        plt.close()
    
    def _plot_overall_comparison(self, output_dir: Path):
        """Plot overall comparison radar chart"""
        from math import pi
        
        # Prepare data
        categories = ['Quality', 'Robustness', 'Speed']
        num_vars = len(categories)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Compute angles
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Plot for each method
        for method_name in self.results.keys():
            values = []
            
            # Get normalized scores (0-1)
            # Quality score
            if 'quality' in self.results[method_name]['metrics']:
                psnr = self.results[method_name]['metrics']['quality']['psnr']
                ssim = self.results[method_name]['metrics']['quality']['ssim']
                quality_score = 0.5 * min(psnr / 50.0, 1.0) + 0.5 * ssim
            else:
                quality_score = 0
            values.append(quality_score)
            
            # Robustness score
            if 'robustness' in self.results[method_name]['metrics']:
                all_acc = []
                for attack_results in self.results[method_name]['metrics']['robustness'].values():
                    all_acc.extend(list(attack_results.values()))
                robustness_score = np.mean(all_acc) if all_acc else 0
            else:
                robustness_score = 0
            values.append(robustness_score)
            
            # Speed score (normalize to 0-1)
            if 'speed' in self.results[method_name]['metrics']:
                fps = self.results[method_name]['metrics']['speed']['encode_fps']
                speed_score = min(fps / 100.0, 1.0)  # Normalize by 100 FPS
            else:
                speed_score = 0
            values.append(speed_score)
            
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title('Overall Performance Comparison')
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_comparison.png', dpi=300)
        plt.close()


class ReedSolomonSimulator:
    """Simulate Reed-Solomon error correction for comparison"""
    
    def __init__(self, n: int = 255, k: int = 223):
        """
        Initialize RS simulator
        
        Args:
            n: Total symbols
            k: Data symbols
        """
        self.n = n
        self.k = k
        self.redundancy = 1 - k/n
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Simulate RS encoding"""
        # Simple simulation: repeat data with redundancy
        redundancy_factor = self.n / self.k
        return np.repeat(data, int(redundancy_factor), axis=-1)[:, :self.n]
    
    def decode(self, received: np.ndarray, error_rate: float = 0.1) -> np.ndarray:
        """Simulate RS decoding"""
        # Simple simulation: majority voting
        reshaped = received.reshape(received.shape[0], self.k, -1)
        decoded = np.mean(reshaped, axis=2) > 0.5
        
        # Simulate error correction capability
        max_errors = (self.n - self.k) // 2
        if error_rate * self.n > max_errors:
            # Too many errors, decoding fails
            success_rate = max(0, 1 - (error_rate * self.n - max_errors) / max_errors)
            mask = np.random.random(decoded.shape) < success_rate
            decoded = decoded * mask
        
        return decoded