#!/usr/bin/env python3
"""
Visualization Utilities
Tools for visualizing steganography results and metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from PIL import Image
import io
import base64


def visualize_results(cover_images: torch.Tensor,
                     stego_images: torch.Tensor,
                     recovered_images: Optional[torch.Tensor] = None,
                     messages: Optional[torch.Tensor] = None,
                     decoded_messages: Optional[torch.Tensor] = None,
                     save_path: Optional[Path] = None,
                     num_samples: int = 4):
    """
    Visualize steganography results
    
    Args:
        cover_images: Original cover images
        stego_images: Stego images with embedded messages
        recovered_images: Recovered images from attack
        messages: Original messages
        decoded_messages: Decoded messages
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    
    num_samples = min(num_samples, cover_images.size(0))
    
    # Determine number of columns
    num_cols = 3 if recovered_images is None else 4
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Cover image
        ax = axes[i, 0]
        cover_img = tensor_to_numpy(cover_images[i])
        ax.imshow(cover_img)
        ax.set_title('Cover Image')
        ax.axis('off')
        
        # Stego image
        ax = axes[i, 1]
        stego_img = tensor_to_numpy(stego_images[i])
        ax.imshow(stego_img)
        ax.set_title('Stego Image')
        ax.axis('off')
        
        # Difference image
        ax = axes[i, 2]
        diff = np.abs(cover_img - stego_img)
        diff_enhanced = diff * 10  # Enhance visibility
        ax.imshow(diff_enhanced)
        ax.set_title('Difference (10x)')
        ax.axis('off')
        
        # Recovered image (if available)
        if recovered_images is not None and num_cols > 3:
            ax = axes[i, 3]
            recovered_img = tensor_to_numpy(recovered_images[i])
            ax.imshow(recovered_img)
            ax.set_title('Recovered')
            ax.axis('off')
    
    # Add message accuracy if available
    if messages is not None and decoded_messages is not None:
        # Calculate accuracy
        if messages.dim() > 1:
            messages_binary = (messages > 0.5).float()
            decoded_binary = (torch.sigmoid(decoded_messages) > 0.5).float()
            accuracy = (messages_binary == decoded_binary).float().mean(dim=1)
            
            # Add text
            fig.suptitle(f'Message Accuracy: {accuracy.mean().item():.3f}', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[Path] = None):
    """Plot training history"""
    
    num_metrics = len(history)
    fig, axes = plt.subplots((num_metrics + 1) // 2, 2, figsize=(12, 4 * ((num_metrics + 1) // 2)))
    axes = axes.flatten() if num_metrics > 1 else [axes]
    
    for idx, (metric_name, values) in enumerate(history.items()):
        if idx < len(axes):
            ax = axes[idx]
            ax.plot(values)
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(history), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_attack_robustness(attack_results: Dict[str, Dict[str, float]],
                          save_path: Optional[Path] = None):
    """Plot attack robustness results"""
    
    # Convert to DataFrame
    df_data = []
    for attack_name, metrics in attack_results.items():
        for metric_name, value in metrics.items():
            df_data.append({
                'Attack': attack_name,
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(df_data)
    
    # Create subplots for each metric
    metrics = df['Metric'].unique()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_df = df[df['Metric'] == metric]
        
        # Create bar plot
        sns.barplot(data=metric_df, x='Attack', y='Value', ax=ax)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Value')
        
        # Rotate x labels
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_ldpc_analysis(ldpc_analysis: Dict[str, Any],
                      save_path: Optional[Path] = None):
    """Plot LDPC system analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Redundancy levels
    ax = axes[0, 0]
    redundancy_levels = ldpc_analysis.get('redundancy_levels', [])
    code_rates = ldpc_analysis.get('code_rates', [])
    
    ax.plot(redundancy_levels, code_rates, 'bo-')
    ax.set_xlabel('Redundancy')
    ax.set_ylabel('Code Rate')
    ax.set_title('LDPC Code Rate vs Redundancy')
    ax.grid(True, alpha=0.3)
    
    # Expansion factors
    ax = axes[0, 1]
    expansion_factors = ldpc_analysis.get('expansion_factors', [])
    
    ax.bar(range(len(redundancy_levels)), expansion_factors)
    ax.set_xlabel('Redundancy Level Index')
    ax.set_ylabel('Expansion Factor')
    ax.set_title('LDPC Expansion Factors')
    ax.set_xticks(range(len(redundancy_levels)))
    ax.set_xticklabels([f'{r:.2f}' for r in redundancy_levels])
    
    # Performance comparison (if available)
    if 'ldpc_vs_reed_solomon' in ldpc_analysis:
        ax = axes[1, 0]
        comparison = ldpc_analysis['ldpc_vs_reed_solomon']
        
        noise_levels = comparison.get('noise_levels', [])
        ldpc_ber = comparison.get('ldpc_ber', [])
        rs_ber = comparison.get('reed_solomon_ber', [])
        
        ax.semilogy(noise_levels, ldpc_ber, 'b-', label='LDPC')
        ax.semilogy(noise_levels, rs_ber, 'r--', label='Reed-Solomon')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Bit Error Rate')
        ax.set_title('LDPC vs Reed-Solomon Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_metric_dashboard(metrics_history: Dict[str, List[float]],
                          save_path: Optional[Path] = None):
    """Create comprehensive metrics dashboard"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Loss plots
    ax1 = fig.add_subplot(gs[0, :2])
    if 'total_loss' in metrics_history:
        ax1.plot(metrics_history['total_loss'], label='Total Loss')
    if 'message_loss' in metrics_history:
        ax1.plot(metrics_history['message_loss'], label='Message Loss')
    if 'image_loss' in metrics_history:
        ax1.plot(metrics_history['image_loss'], label='Image Loss')
    ax1.set_title('Training Losses')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = fig.add_subplot(gs[0, 2])
    if 'message_acc' in metrics_history:
        ax2.plot(metrics_history['message_acc'])
        ax2.set_title('Message Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
    
    # Image quality metrics
    ax3 = fig.add_subplot(gs[1, 0])
    if 'psnr' in metrics_history:
        ax3.plot(metrics_history['psnr'])
        ax3.set_title('PSNR')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('dB')
        ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    if 'ssim' in metrics_history:
        ax4.plot(metrics_history['ssim'])
        ax4.set_title('SSIM')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 2])
    if 'lpips' in metrics_history:
        ax5.plot(metrics_history['lpips'])
        ax5.set_title('LPIPS')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3)
    
    # BER plot
    ax6 = fig.add_subplot(gs[2, :])
    if 'bit_error_rate' in metrics_history:
        ax6.semilogy(metrics_history['bit_error_rate'])
        ax6.set_title('Bit Error Rate')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('BER')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics Dashboard', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization"""
    # Move to CPU and detach
    tensor = tensor.detach().cpu()
    
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.size(0) == 3:
        # RGB image
        img = tensor.permute(1, 2, 0).numpy()
    elif tensor.size(0) == 1:
        # Grayscale image
        img = tensor.squeeze(0).numpy()
    else:
        img = tensor.numpy()
    
    return img


def save_image_grid(images: torch.Tensor, save_path: Path,
                   nrow: int = 8, padding: int = 2,
                   normalize: bool = True):
    """Save images as grid"""
    from torchvision.utils import save_image
    
    save_image(images, save_path, nrow=nrow, padding=padding,
               normalize=normalize, value_range=(-1, 1))


def create_html_report(results: Dict[str, Any], save_path: Path):
    """Create HTML report with results"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LDPC Steganography Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; }
            .metric { background-color: #f0f0f0; padding: 10px; margin: 10px 0; }
            .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .image-item { text-align: center; }
            img { max-width: 100%; height: auto; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>LDPC Steganography Results Report</h1>
    """
    
    # Add metrics section
    if 'metrics' in results:
        html_content += "<h2>Performance Metrics</h2>"
        html_content += "<table>"
        html_content += "<tr><th>Metric</th><th>Value</th></tr>"
        
        for metric, value in results['metrics'].items():
            html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        
        html_content += "</table>"
    
    # Add images section
    if 'images' in results:
        html_content += "<h2>Visual Results</h2>"
        html_content += '<div class="image-grid">'
        
        for img_name, img_path in results['images'].items():
            # Convert image to base64
            with open(img_path, 'rb') as f:
                img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode()
            
            html_content += f'''
            <div class="image-item">
                <h3>{img_name}</h3>
                <img src="data:image/png;base64,{img_base64}">
            </div>
            '''
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)


def plot_bit_distribution(messages: torch.Tensor, 
                         decoded_messages: torch.Tensor,
                         save_path: Optional[Path] = None):
    """Plot bit distribution comparison"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original message distribution
    ax = axes[0]
    original_bits = messages.flatten().cpu().numpy()
    ax.hist(original_bits, bins=50, alpha=0.7, color='blue')
    ax.set_title('Original Message Bit Distribution')
    ax.set_xlabel('Bit Value')
    ax.set_ylabel('Frequency')
    
    # Decoded message distribution
    ax = axes[1]
    decoded_bits = torch.sigmoid(decoded_messages).flatten().cpu().numpy()
    ax.hist(decoded_bits, bins=50, alpha=0.7, color='green')
    ax.set_title('Decoded Message Bit Distribution')
    ax.set_xlabel('Bit Value')
    ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_error_heatmap(messages: torch.Tensor,
                        decoded_messages: torch.Tensor,
                        save_path: Optional[Path] = None):
    """Create heatmap of bit errors"""
    
    # Calculate errors
    messages_binary = (messages > 0.5).float()
    decoded_binary = (torch.sigmoid(decoded_messages) > 0.5).float()
    errors = (messages_binary != decoded_binary).float()
    
    # Reshape for visualization
    batch_size, msg_len = errors.shape
    grid_size = int(np.ceil(np.sqrt(msg_len)))
    
    # Pad to square
    padding = grid_size * grid_size - msg_len
    if padding > 0:
        errors_padded = F.pad(errors, (0, padding), value=-1)
    else:
        errors_padded = errors
    
    # Reshape to grid
    error_grid = errors_padded.reshape(batch_size, grid_size, grid_size)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Average across batch
    avg_errors = error_grid.mean(dim=0).cpu().numpy()
    
    # Create custom colormap
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color='gray')
    
    # Mask padding
    masked_errors = np.ma.masked_where(avg_errors < 0, avg_errors)
    
    im = ax.imshow(masked_errors, cmap=cmap, vmin=0, vmax=1)
    ax.set_title('Average Bit Error Heatmap')
    ax.set_xlabel('Bit Position (x)')
    ax.set_ylabel('Bit Position (y)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Error Rate')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()