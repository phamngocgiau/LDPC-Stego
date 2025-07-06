#!/usr/bin/env python3
"""
Result Analysis Module
Analyze and visualize evaluation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class ResultAnalyzer:
    """Analyze evaluation results"""
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing results
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        self.df = None
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_results(self, experiment_names: Optional[List[str]] = None):
        """Load results from experiments"""
        if experiment_names is None:
            # Load all available results
            result_files = list(self.results_dir.glob('*/evaluation_results.json'))
        else:
            result_files = [self.results_dir / name / 'evaluation_results.json' 
                          for name in experiment_names]
        
        for result_file in result_files:
            if result_file.exists():
                with open(result_file, 'r') as f:
                    experiment_name = result_file.parent.name
                    self.results[experiment_name] = json.load(f)
        
        # Convert to DataFrame for easier analysis
        self._create_dataframe()
        
        print(f"Loaded {len(self.results)} experiment results")
    
    def _create_dataframe(self):
        """Create pandas DataFrame from results"""
        data = []
        
        for exp_name, exp_results in self.results.items():
            # Basic metrics
            if 'basic' in exp_results:
                row = {
                    'experiment': exp_name,
                    'psnr': exp_results['basic'].get('psnr', 0),
                    'ssim': exp_results['basic'].get('ssim', 0),
                    'message_accuracy': exp_results['basic'].get('message_accuracy', 0),
                    'ber': exp_results['basic'].get('bit_error_rate', 0)
                }
                
                # Add robustness metrics
                if 'robustness' in exp_results:
                    for attack_type, attack_results in exp_results['robustness'].items():
                        for strength, accuracy in attack_results.items():
                            row[f'{attack_type}_{strength}'] = accuracy
                
                data.append(row)
        
        self.df = pd.DataFrame(data)
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis"""
        stats_results = {}
        
        # Basic statistics
        stats_results['basic_stats'] = self.df.describe().to_dict()
        
        # Correlation analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        stats_results['correlations'] = correlation_matrix.to_dict()
        
        # Best performing experiments
        stats_results['best_psnr'] = self.df.nlargest(5, 'psnr')[['experiment', 'psnr']].to_dict('records')
        stats_results['best_accuracy'] = self.df.nlargest(5, 'message_accuracy')[['experiment', 'message_accuracy']].to_dict('records')
        
        # ANOVA test for significant differences
        if len(self.df['experiment'].unique()) > 2:
            experiments = [group['psnr'].values for name, group in self.df.groupby('experiment')]
            f_stat, p_value = stats.f_oneway(*experiments)
            stats_results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return stats_results
    
    def visualize_results(self, output_dir: Optional[str] = None):
        """Create comprehensive visualizations"""
        if output_dir is None:
            output_dir = self.results_dir / 'analysis'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics comparison
        self._plot_quality_metrics(output_dir)
        
        # Robustness heatmap
        self._plot_robustness_heatmap(output_dir)
        
        # Trade-off analysis
        self._plot_tradeoff_analysis(output_dir)
        
        # Correlation heatmap
        self._plot_correlation_heatmap(output_dir)
        
        # Performance distribution
        self._plot_performance_distribution(output_dir)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _plot_quality_metrics(self, output_dir: Path):
        """Plot quality metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR comparison
        ax = axes[0, 0]
        self.df.plot(x='experiment', y='psnr', kind='bar', ax=ax, legend=False)
        ax.set_title('PSNR Comparison')
        ax.set_ylabel('PSNR (dB)')
        ax.tick_params(axis='x', rotation=45)
        
        # SSIM comparison
        ax = axes[0, 1]
        self.df.plot(x='experiment', y='ssim', kind='bar', ax=ax, legend=False, color='orange')
        ax.set_title('SSIM Comparison')
        ax.set_ylabel('SSIM')
        ax.tick_params(axis='x', rotation=45)
        
        # Message accuracy
        ax = axes[1, 0]
        self.df.plot(x='experiment', y='message_accuracy', kind='bar', ax=ax, legend=False, color='green')
        ax.set_title('Message Accuracy')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
        
        # BER comparison
        ax = axes[1, 1]
        self.df.plot(x='experiment', y='ber', kind='bar', ax=ax, legend=False, color='red')
        ax.set_title('Bit Error Rate')
        ax.set_ylabel('BER')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_metrics.png', dpi=300)
        plt.close()
    
    def _plot_robustness_heatmap(self, output_dir: Path):
        """Plot robustness heatmap"""
        # Extract robustness columns
        robustness_cols = [col for col in self.df.columns if '_' in col and 
                          any(attack in col for attack in ['jpeg', 'noise', 'blur', 'crop'])]
        
        if not robustness_cols:
            return
        
        # Create heatmap data
        heatmap_data = self.df.set_index('experiment')[robustness_cols].T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy'})
        plt.title('Robustness Heatmap')
        plt.xlabel('Experiment')
        plt.ylabel('Attack Type & Strength')
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_heatmap.png', dpi=300)
        plt.close()
    
    def _plot_tradeoff_analysis(self, output_dir: Path):
        """Plot trade-off between quality and robustness"""
        # Calculate average robustness
        robustness_cols = [col for col in self.df.columns if '_' in col and 
                          any(attack in col for attack in ['jpeg', 'noise', 'blur'])]
        
        if robustness_cols:
            self.df['avg_robustness'] = self.df[robustness_cols].mean(axis=1)
        else:
            self.df['avg_robustness'] = self.df['message_accuracy']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # PSNR vs Robustness
        ax1.scatter(self.df['psnr'], self.df['avg_robustness'], s=100)
        for idx, row in self.df.iterrows():
            ax1.annotate(row['experiment'], (row['psnr'], row['avg_robustness']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('PSNR (dB)')
        ax1.set_ylabel('Average Robustness')
        ax1.set_title('Quality vs Robustness Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # SSIM vs Message Accuracy
        ax2.scatter(self.df['ssim'], self.df['message_accuracy'], s=100, color='orange')
        for idx, row in self.df.iterrows():
            ax2.annotate(row['experiment'], (row['ssim'], row['message_accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('SSIM')
        ax2.set_ylabel('Message Accuracy')
        ax2.set_title('Perceptual Quality vs Accuracy Trade-off')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tradeoff_analysis.png', dpi=300)
        plt.close()
    
    def _plot_correlation_heatmap(self, output_dir: Path):
        """Plot correlation heatmap"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   cbar_kws={'label': 'Correlation'})
        plt.title('Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=300)
        plt.close()
    
    def _plot_performance_distribution(self, output_dir: Path):
        """Plot performance distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR distribution
        ax = axes[0, 0]
        self.df['psnr'].hist(bins=20, ax=ax, color='blue', alpha=0.7)
        ax.axvline(self.df['psnr'].mean(), color='red', linestyle='dashed', linewidth=2)
        ax.set_xlabel('PSNR (dB)')
        ax.set_ylabel('Count')
        ax.set_title(f'PSNR Distribution (mean={self.df["psnr"].mean():.2f})')
        
        # SSIM distribution
        ax = axes[0, 1]
        self.df['ssim'].hist(bins=20, ax=ax, color='orange', alpha=0.7)
        ax.axvline(self.df['ssim'].mean(), color='red', linestyle='dashed', linewidth=2)
        ax.set_xlabel('SSIM')
        ax.set_ylabel('Count')
        ax.set_title(f'SSIM Distribution (mean={self.df["ssim"].mean():.4f})')
        
        # Message accuracy distribution
        ax = axes[1, 0]
        self.df['message_accuracy'].hist(bins=20, ax=ax, color='green', alpha=0.7)
        ax.axvline(self.df['message_accuracy'].mean(), color='red', linestyle='dashed', linewidth=2)
        ax.set_xlabel('Message Accuracy')
        ax.set_ylabel('Count')
        ax.set_title(f'Accuracy Distribution (mean={self.df["message_accuracy"].mean():.4f})')
        
        # Box plot comparison
        ax = axes[1, 1]
        metrics_to_plot = ['psnr', 'ssim', 'message_accuracy']
        normalized_data = []
        for metric in metrics_to_plot:
            # Normalize to 0-1 range for comparison
            data = self.df[metric].values
            normalized = (data - data.min()) / (data.max() - data.min())
            normalized_data.append(normalized)
        
        ax.boxplot(normalized_data, labels=['PSNR', 'SSIM', 'Accuracy'])
        ax.set_ylabel('Normalized Value')
        ax.set_title('Metric Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_distribution.png', dpi=300)
        plt.close()
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive analysis report"""
        if output_path is None:
            output_path = self.results_dir / 'analysis_report.md'
        else:
            output_path = Path(output_path)
        
        # Perform statistical analysis
        stats = self.statistical_analysis()
        
        report = []
        report.append("# LDPC Steganography Analysis Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(f"- Total experiments analyzed: {len(self.results)}")
        report.append(f"- Best PSNR: {self.df['psnr'].max():.2f} dB ({self.df.loc[self.df['psnr'].idxmax(), 'experiment']})")
        report.append(f"- Best SSIM: {self.df['ssim'].max():.4f} ({self.df.loc[self.df['ssim'].idxmax(), 'experiment']})")
        report.append(f"- Best Message Accuracy: {self.df['message_accuracy'].max():.4f} ({self.df.loc[self.df['message_accuracy'].idxmax(), 'experiment']})\n")
        
        # Detailed Results
        report.append("## Detailed Results\n")
        report.append("### Quality Metrics Summary")
        report.append("```")
        report.append(self.df[['experiment', 'psnr', 'ssim', 'message_accuracy', 'ber']].to_string(index=False))
        report.append("```\n")
        
        # Statistical Analysis
        report.append("### Statistical Analysis\n")
        if 'anova' in stats:
            report.append(f"- ANOVA F-statistic: {stats['anova']['f_statistic']:.4f}")
            report.append(f"- ANOVA p-value: {stats['anova']['p_value']:.4f}")
            report.append(f"- Significant difference: {'Yes' if stats['anova']['significant'] else 'No'}\n")
        
        # Robustness Analysis
        report.append("### Robustness Analysis\n")
        robustness_cols = [col for col in self.df.columns if '_' in col and 
                          any(attack in col for attack in ['jpeg', 'noise', 'blur'])]
        
        if robustness_cols:
            for attack in ['jpeg', 'noise', 'blur']:
                attack_cols = [col for col in robustness_cols if attack in col]
                if attack_cols:
                    avg_robustness = self.df[attack_cols].mean().mean()
                    report.append(f"- Average {attack.upper()} robustness: {avg_robustness:.4f}")
        
        report.append("\n## Recommendations\n")
        
        # Find best overall performer
        self.df['overall_score'] = (
            0.3 * (self.df['psnr'] / self.df['psnr'].max()) +
            0.3 * self.df['ssim'] +
            0.4 * self.df['message_accuracy']
        )
        best_overall = self.df.loc[self.df['overall_score'].idxmax(), 'experiment']
        report.append(f"- Best overall performer: **{best_overall}**")
        
        # Quality vs Robustness recommendation
        if 'avg_robustness' in self.df.columns:
            high_quality = self.df.nlargest(3, 'psnr')['experiment'].tolist()
            high_robustness = self.df.nlargest(3, 'avg_robustness')['experiment'].tolist()
            balanced = list(set(high_quality) & set(high_robustness))
            
            if balanced:
                report.append(f"- Balanced quality/robustness: {', '.join(balanced)}")
        
        # Save report
        report_text = '\n'.join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"Analysis report saved to {output_path}")
        
        return report_text
    
    def compare_experiments(self, exp1: str, exp2: str) -> Dict[str, Any]:
        """Detailed comparison between two experiments"""
        if exp1 not in self.results or exp2 not in self.results:
            raise ValueError("Experiment not found in results")
        
        comparison = {
            'experiment_1': exp1,
            'experiment_2': exp2,
            'metrics': {}
        }
        
        # Compare basic metrics
        metrics_to_compare = ['psnr', 'ssim', 'message_accuracy', 'ber']
        for metric in metrics_to_compare:
            val1 = self.df[self.df['experiment'] == exp1][metric].values[0]
            val2 = self.df[self.df['experiment'] == exp2][metric].values[0]
            
            comparison['metrics'][metric] = {
                exp1: val1,
                exp2: val2,
                'difference': val1 - val2,
                'percentage_change': ((val1 - val2) / val2 * 100) if val2 != 0 else 0
            }
        
        # Compare robustness
        robustness_comparison = {}
        for col in self.df.columns:
            if '_' in col and any(attack in col for attack in ['jpeg', 'noise', 'blur']):
                val1 = self.df[self.df['experiment'] == exp1][col].values[0]
                val2 = self.df[self.df['experiment'] == exp2][col].values[0]
                robustness_comparison[col] = {
                    exp1: val1,
                    exp2: val2,
                    'difference': val1 - val2
                }
        
        comparison['robustness'] = robustness_comparison
        
        # Overall recommendation
        score1 = sum(1 for m in comparison['metrics'].values() 
                    if m['difference'] > 0 and m != 'ber')
        score2 = len(comparison['metrics']) - score1
        
        if score1 > score2:
            comparison['recommendation'] = f"{exp1} performs better overall"
        elif score2 > score1:
            comparison['recommendation'] = f"{exp2} performs better overall"
        else:
            comparison['recommendation'] = "Both experiments show similar performance"
        
        return comparison


class FeatureAnalyzer:
    """Analyze learned features and representations"""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize feature analyzer
        
        Args:
            model: Trained model
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.features = {}
    
    def extract_features(self, data_loader, layer_names: List[str]):
        """Extract features from specified layers"""
        self.model.eval()
        
        # Register hooks
        handles = []
        for name, module in self.model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(
                    lambda m, i, o, n=name: self._save_features(n, o)
                )
                handles.append(handle)
        
        # Extract features
        with torch.no_grad():
            for batch in data_loader:
                images = batch['cover'].to(self.device)
                _ = self.model(images, batch['message'].to(self.device))
                break  # Just one batch for analysis
        
        # Remove hooks
        for handle in handles:
            handle.remove()
    
    def _save_features(self, name: str, output: torch.Tensor):
        """Hook function to save features"""
        self.features[name] = output.detach().cpu()
    
    def visualize_features(self, output_dir: Path):
        """Visualize extracted features"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_name, features in self.features.items():
            # Visualize first few channels
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(min(16, features.size(1))):
                ax = axes[i]
                feature_map = features[0, i].numpy()
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
            
            plt.suptitle(f'Feature Maps: {layer_name}')
            plt.tight_layout()
            plt.savefig(output_dir / f'features_{layer_name.replace("/", "_")}.png')
            plt.close()
    
    def analyze_feature_statistics(self) -> Dict[str, Any]:
        """Analyze statistical properties of features"""
        stats = {}
        
        for layer_name, features in self.features.items():
            features_flat = features.view(features.size(0), -1)
            
            stats[layer_name] = {
                'mean': features_flat.mean(dim=1).numpy(),
                'std': features_flat.std(dim=1).numpy(),
                'sparsity': (features_flat == 0).float().mean().item(),
                'activation_rate': (features_flat > 0).float().mean().item()
            }
        
        return stats