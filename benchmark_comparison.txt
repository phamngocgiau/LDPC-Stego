#!/usr/bin/env python3
"""
Benchmark Comparison Script
Compare LDPC vs Reed-Solomon error correction performance
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.adaptive_ldpc import AdaptiveLDPC, LDPCPerformanceAnalyzer
from utils.logger import setup_logger
from utils.helpers import set_random_seeds


class ReedSolomonSimulator:
    """Simulated Reed-Solomon error correction for comparison"""
    
    def __init__(self, n: int, k: int):
        """
        Initialize RS simulator
        
        Args:
            n: Total symbols (codeword length)
            k: Data symbols (message length)
        """
        self.n = n
        self.k = k
        self.redundancy = (n - k) / n
        
        # Error correction capability
        self.t = (n - k) // 2  # Can correct t errors
    
    def encode(self, message: np.ndarray) -> np.ndarray:
        """Simulate RS encoding"""
        # Simple simulation: add redundancy
        redundancy_bits = int(len(message) * self.redundancy / (1 - self.redundancy))
        
        # Generate parity (simplified)
        parity = np.random.randint(0, 2, redundancy_bits)
        
        return np.concatenate([message, parity])
    
    def decode(self, received: np.ndarray, error_rate: float) -> np.ndarray:
        """Simulate RS decoding"""
        # Extract message part
        message_length = int(len(received) * self.k / self.n)
        message = received[:message_length]
        
        # Simulate error correction based on theoretical performance
        # RS can correct up to t symbol errors
        num_errors = int(error_rate * len(message))
        
        if num_errors <= self.t:
            # Can correct all errors
            return message
        else:
            # Cannot correct all errors - partial correction
            correction_rate = self.t / num_errors if num_errors > 0 else 1.0
            corrected = message.copy()
            
            # Simulate partial correction
            error_positions = np.random.choice(len(message), num_errors, replace=False)
            num_corrected = int(len(error_positions) * correction_rate)
            corrected_positions = np.random.choice(error_positions, num_corrected, replace=False)
            
            # Flip back corrected bits (simplified)
            corrected[corrected_positions] = 1 - corrected[corrected_positions]
            
            return corrected


def benchmark_error_correction(message_length: int, redundancy_levels: list,
                              noise_levels: list, num_trials: int = 100):
    """Benchmark LDPC vs Reed-Solomon performance"""
    
    results = {
        'ldpc': {},
        'reed_solomon': {},
        'parameters': {
            'message_length': message_length,
            'redundancy_levels': redundancy_levels,
            'noise_levels': noise_levels,
            'num_trials': num_trials
        }
    }
    
    for redundancy in redundancy_levels:
        ldpc_results = []
        rs_results = []
        
        # Create LDPC system
        ldpc = AdaptiveLDPC(
            message_length=message_length,
            min_redundancy=redundancy,
            max_redundancy=redundancy,
            device='cpu'
        )
        
        # Create RS simulator
        n = int(message_length / (1 - redundancy))
        rs = ReedSolomonSimulator(n=n, k=message_length)
        
        for noise_level in tqdm(noise_levels, desc=f"Redundancy {redundancy}"):
            ldpc_ber = []
            rs_ber = []
            
            for _ in range(num_trials):
                # Generate random message
                message = torch.randint(0, 2, (1, message_length), dtype=torch.float32)
                
                # LDPC encoding/decoding
                ldpc_encoded = ldpc.encode(message, attack_strength=0)
                
                # Add noise
                noise = torch.randn_like(ldpc_encoded) * noise_level
                ldpc_noisy = ldpc_encoded + noise
                
                # Decode
                ldpc_decoded = ldpc.decode(ldpc_noisy, attack_strength=0)
                
                # Calculate BER
                ldpc_ber_val = calculate_ber(message, ldpc_decoded)
                ldpc_ber.append(ldpc_ber_val)
                
                # RS encoding/decoding
                message_np = message.numpy().flatten()
                rs_encoded = rs.encode(message_np)
                
                # Add noise
                rs_noise = np.random.randn(len(rs_encoded)) * noise_level
                rs_noisy = rs_encoded + rs_noise
                rs_noisy_binary = (rs_noisy > 0.5).astype(np.float32)
                
                # Decode
                rs_decoded = rs.decode(rs_noisy_binary, noise_level)
                
                # Calculate BER
                rs_ber_val = calculate_ber_numpy(message_np, rs_decoded)
                rs_ber.append(rs_ber_val)
            
            ldpc_results.append({
                'noise_level': noise_level,
                'ber_mean': np.mean(ldpc_ber),
                'ber_std': np.std(ldpc_ber)
            })
            
            rs_results.append({
                'noise_level': noise_level,
                'ber_mean': np.mean(rs_ber),
                'ber_std': np.std(rs_ber)
            })
        
        results['ldpc'][redundancy] = ldpc_results
        results['reed_solomon'][redundancy] = rs_results
    
    return results


def benchmark_computational_performance(message_lengths: list, redundancy: float = 0.3,
                                      num_trials: int = 100):
    """Benchmark computational performance"""
    
    results = {
        'ldpc': {'encode_time': [], 'decode_time': []},
        'parameters': {
            'message_lengths': message_lengths,
            'redundancy': redundancy,
            'num_trials': num_trials
        }
    }
    
    for msg_len in tqdm(message_lengths, desc="Message lengths"):
        # LDPC system
        ldpc = AdaptiveLDPC(
            message_length=msg_len,
            min_redundancy=redundancy,
            max_redundancy=redundancy,
            device='cpu'
        )
        
        # Timing LDPC
        ldpc_encode_times = []
        ldpc_decode_times = []
        
        for _ in range(num_trials):
            message = torch.randint(0, 2, (1, msg_len), dtype=torch.float32)
            
            # Encode timing
            start = time.time()
            encoded = ldpc.encode(message, attack_strength=0)
            ldpc_encode_times.append(time.time() - start)
            
            # Add noise
            noisy = encoded + torch.randn_like(encoded) * 0.1
            
            # Decode timing
            start = time.time()
            decoded = ldpc.decode(noisy, attack_strength=0)
            ldpc_decode_times.append(time.time() - start)
        
        results['ldpc']['encode_time'].append({
            'message_length': msg_len,
            'mean': np.mean(ldpc_encode_times),
            'std': np.std(ldpc_encode_times)
        })
        
        results['ldpc']['decode_time'].append({
            'message_length': msg_len,
            'mean': np.mean(ldpc_decode_times),
            'std': np.std(ldpc_decode_times)
        })
    
    return results


def plot_ber_comparison(results: dict, save_path: Path):
    """Plot BER comparison"""
    fig, axes = plt.subplots(1, len(results['parameters']['redundancy_levels']), 
                            figsize=(5 * len(results['parameters']['redundancy_levels']), 5))
    
    if len(results['parameters']['redundancy_levels']) == 1:
        axes = [axes]
    
    for idx, redundancy in enumerate(results['parameters']['redundancy_levels']):
        ax = axes[idx]
        
        # LDPC results
        ldpc_data = results['ldpc'][redundancy]
        noise_levels = [d['noise_level'] for d in ldpc_data]
        ldpc_ber = [d['ber_mean'] for d in ldpc_data]
        ldpc_std = [d['ber_std'] for d in ldpc_data]
        
        # RS results
        rs_data = results['reed_solomon'][redundancy]
        rs_ber = [d['ber_mean'] for d in rs_data]
        rs_std = [d['ber_std'] for d in rs_data]
        
        # Plot with error bars
        ax.errorbar(noise_levels, ldpc_ber, yerr=ldpc_std, 
                   label='LDPC', marker='o', capsize=5)
        ax.errorbar(noise_levels, rs_ber, yerr=rs_std,
                   label='Reed-Solomon', marker='s', capsize=5)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Bit Error Rate')
        ax.set_title(f'Redundancy = {redundancy:.1%}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_comparison(results: dict, save_path: Path):
    """Plot computational performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Encode time
    msg_lengths = [d['message_length'] for d in results['ldpc']['encode_time']]
    encode_times = [d['mean'] * 1000 for d in results['ldpc']['encode_time']]  # Convert to ms
    
    ax1.plot(msg_lengths, encode_times, 'o-', label='LDPC')
    ax1.set_xlabel('Message Length (bits)')
    ax1.set_ylabel('Encoding Time (ms)')
    ax1.set_title('Encoding Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Decode time
    decode_times = [d['mean'] * 1000 for d in results['ldpc']['decode_time']]
    
    ax2.plot(msg_lengths, decode_times, 'o-', label='LDPC')
    ax2.set_xlabel('Message Length (bits)')
    ax2.set_ylabel('Decoding Time (ms)')
    ax2.set_title('Decoding Performance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_ber(original: torch.Tensor, decoded: torch.Tensor) -> float:
    """Calculate bit error rate for torch tensors"""
    original_binary = (original > 0.5).float()
    decoded_binary = (torch.sigmoid(decoded) > 0.5).float()
    
    errors = (original_binary != decoded_binary).float()
    return errors.mean().item()


def calculate_ber_numpy(original: np.ndarray, decoded: np.ndarray) -> float:
    """Calculate bit error rate for numpy arrays"""
    original_binary = (original > 0.5).astype(float)
    decoded_binary = (decoded > 0.5).astype(float)
    
    errors = (original_binary != decoded_binary).astype(float)
    return errors.mean()


def create_comparison_report(results: dict, save_path: Path):
    """Create detailed comparison report"""
    report = []
    report.append("=" * 80)
    report.append("LDPC vs REED-SOLOMON COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Error correction performance
    report.append("ERROR CORRECTION PERFORMANCE")
    report.append("-" * 40)
    
    for redundancy in results['ldpc'].keys():
        report.append(f"\nRedundancy Level: {redundancy:.1%}")
        report.append("Noise Level | LDPC BER    | RS BER      | LDPC Advantage")
        report.append("-" * 60)
        
        ldpc_data = results['ldpc'][redundancy]
        rs_data = results['reed_solomon'][redundancy]
        
        for ldpc, rs in zip(ldpc_data, rs_data):
            noise = ldpc['noise_level']
            ldpc_ber = ldpc['ber_mean']
            rs_ber = rs['ber_mean']
            
            if rs_ber > 0:
                advantage = (rs_ber - ldpc_ber) / rs_ber * 100
            else:
                advantage = 0
            
            report.append(f"{noise:11.2f} | {ldpc_ber:11.6f} | {rs_ber:11.6f} | {advantage:+13.1f}%")
    
    report.append("\n" + "=" * 80)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description='Benchmark LDPC vs Reed-Solomon')
    
    parser.add_argument('--output_dir', type=str, default='results/benchmark',
                       help='Output directory')
    parser.add_argument('--message_length', type=int, default=1024,
                       help='Message length in bits')
    parser.add_argument('--redundancy_levels', nargs='+', type=float,
                       default=[0.1, 0.2, 0.3, 0.4, 0.5],
                       help='Redundancy levels to test')
    parser.add_argument('--noise_levels', nargs='+', type=float,
                       default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                       help='Noise levels to test')
    parser.add_argument('--num_trials', type=int, default=100,
                       help='Number of trials per configuration')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('benchmark', str(output_dir / 'logs'))
    logger.info("Starting LDPC vs Reed-Solomon benchmark")
    logger.info(f"Arguments: {vars(args)}")
    
    set_random_seeds(args.seed)
    
    # Run error correction benchmark
    logger.info("Running error correction benchmark...")
    ber_results = benchmark_error_correction(
        args.message_length,
        args.redundancy_levels,
        args.noise_levels,
        args.num_trials
    )
    
    # Run computational performance benchmark
    logger.info("Running computational performance benchmark...")
    message_lengths = [256, 512, 1024, 2048, 4096]
    perf_results = benchmark_computational_performance(
        message_lengths,
        redundancy=0.3,
        num_trials=50
    )
    
    # Save results
    results = {
        'error_correction': ber_results,
        'computational_performance': perf_results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    logger.info("Creating visualizations...")
    plot_ber_comparison(ber_results, output_dir / 'ber_comparison.png')
    plot_performance_comparison(perf_results, output_dir / 'performance_comparison.png')
    
    # Create report
    create_comparison_report(ber_results, output_dir / 'comparison_report.txt')
    
    logger.info("Benchmark completed successfully!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()