#!/usr/bin/env python3
"""
LDPC Metrics Calculator
Calculates comprehensive metrics for LDPC steganography evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import logging


class LDPCMetricsCalculator:
    """Calculate metrics for LDPC steganography system"""
    
    def __init__(self, config):
        """
        Initialize metrics calculator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.device
        
        # LPIPS perceptual metric
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)
        
        # Metric history for tracking
        self.metric_history = {
            'message_acc': [],
            'bit_error_rate': [],
            'psnr': [],
            'ssim': [],
            'lpips': []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_batch_metrics(self, outputs: Dict[str, torch.Tensor],
                               targets: Dict[str, torch.Tensor],
                               loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate metrics for a batch
        
        Args:
            outputs: Model outputs
            targets: Target values
            loss_components: Individual loss components
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Message recovery metrics
        if 'decoded_messages' in outputs and 'messages' in targets:
            message_metrics = self.calculate_message_metrics(
                outputs['decoded_messages'],
                targets['messages']
            )
            metrics.update(message_metrics)
        
        # Image quality metrics
        if 'stego_images' in outputs and 'cover_images' in targets:
            image_metrics = self.calculate_image_metrics(
                outputs['stego_images'],
                targets['cover_images']
            )
            metrics.update(image_metrics)
        
        # LDPC-specific metrics
        if 'decoded_ldpc_soft' in outputs:
            ldpc_metrics = self.calculate_ldpc_metrics(
                outputs['decoded_ldpc_soft'],
                outputs.get('ldpc_encoded_messages'),
                targets.get('messages')
            )
            metrics.update(ldpc_metrics)
        
        # Recovery metrics
        if 'recovered_images' in outputs and 'cover_images' in targets:
            recovery_metrics = self.calculate_recovery_metrics(
                outputs['recovered_images'],
                targets['cover_images']
            )
            metrics.update(recovery_metrics)
        
        # Attack robustness metrics
        if 'attacked_images' in outputs:
            robustness_metrics = self.calculate_robustness_metrics(
                outputs, targets
            )
            metrics.update(robustness_metrics)
        
        # Add loss components
        metrics.update({f'loss_{k}': v for k, v in loss_components.items()})
        
        # Update history
        self._update_history(metrics)
        
        return metrics
    
    def calculate_message_metrics(self, decoded: torch.Tensor, 
                                 target: torch.Tensor) -> Dict[str, float]:
        """Calculate message recovery metrics"""
        with torch.no_grad():
            # Convert to binary predictions
            if decoded.dim() > target.dim():
                decoded = decoded.squeeze()
            
            pred_binary = (torch.sigmoid(decoded) > 0.5).float()
            target_binary = target.float()
            
            # Accuracy
            correct = (pred_binary == target_binary).float()
            accuracy = correct.mean().item()
            
            # Bit error rate
            errors = (pred_binary != target_binary).float()
            ber = errors.mean().item()
            
            # Per-message accuracy (all bits correct)
            batch_size = target.size(0)
            message_correct = correct.view(batch_size, -1).all(dim=1).float()
            message_acc = message_correct.mean().item()
            
            # Hamming distance
            hamming_dist = errors.view(batch_size, -1).sum(dim=1).mean().item()
            
        return {
            'bit_accuracy': accuracy,
            'bit_error_rate': ber,
            'message_acc': message_acc,
            'hamming_distance': hamming_dist
        }
    
    def calculate_image_metrics(self, pred_images: torch.Tensor,
                               target_images: torch.Tensor) -> Dict[str, float]:
        """Calculate image quality metrics"""
        metrics = {}
        
        with torch.no_grad():
            # Ensure images are in [-1, 1] range
            pred_images = torch.clamp(pred_images, -1, 1)
            target_images = torch.clamp(target_images, -1, 1)
            
            # Convert to [0, 1] for metrics
            pred_01 = (pred_images + 1) / 2
            target_01 = (target_images + 1) / 2
            
            # MSE
            mse = F.mse_loss(pred_images, target_images).item()
            metrics['mse'] = mse
            
            # PSNR (batch average)
            psnr_values = []
            for i in range(pred_images.size(0)):
                pred_np = pred_01[i].cpu().numpy().transpose(1, 2, 0)
                target_np = target_01[i].cpu().numpy().transpose(1, 2, 0)
                psnr_val = psnr(target_np, pred_np, data_range=1.0)
                psnr_values.append(psnr_val)
            metrics['psnr'] = np.mean(psnr_values)
            
            # SSIM (batch average)
            ssim_values = []
            for i in range(pred_images.size(0)):
                pred_np = pred_01[i].cpu().numpy().transpose(1, 2, 0)
                target_np = target_01[i].cpu().numpy().transpose(1, 2, 0)
                ssim_val = ssim(target_np, pred_np, data_range=1.0, multichannel=True)
                ssim_values.append(ssim_val)
            metrics['ssim'] = np.mean(ssim_values)
            
            # LPIPS
            lpips_val = self.lpips_fn(pred_images, target_images).mean().item()
            metrics['lpips'] = lpips_val
            
            # L1 loss
            l1_loss = F.l1_loss(pred_images, target_images).item()
            metrics['l1_loss'] = l1_loss
        
        return metrics
    
    def calculate_ldpc_metrics(self, soft_decoded: torch.Tensor,
                              ldpc_encoded: Optional[torch.Tensor],
                              original_messages: Optional[torch.Tensor]) -> Dict[str, float]:
        """Calculate LDPC-specific metrics"""
        metrics = {}
        
        with torch.no_grad():
            # Soft decision statistics
            soft_values = torch.sigmoid(soft_decoded)
            
            # Confidence (how close to 0 or 1)
            confidence = torch.min(soft_values, 1 - soft_values).mean().item()
            metrics['ldpc_confidence'] = 1 - 2 * confidence  # Higher is better
            
            # Entropy of soft decisions
            entropy = -(soft_values * torch.log(soft_values + 1e-8) + 
                       (1 - soft_values) * torch.log(1 - soft_values + 1e-8))
            metrics['ldpc_entropy'] = entropy.mean().item()
            
            if ldpc_encoded is not None:
                # LDPC codeword recovery rate
                pred_codeword = (soft_values > 0.5).float()
                ldpc_accuracy = (pred_codeword == ldpc_encoded).float().mean().item()
                metrics['ldpc_codeword_acc'] = ldpc_accuracy
            
            # Syndrome check (if H matrix available)
            if hasattr(self, 'H_matrix') and self.H_matrix is not None:
                syndrome = self._check_syndrome(soft_decoded)
                metrics['syndrome_rate'] = syndrome
        
        return metrics
    
    def calculate_recovery_metrics(self, recovered: torch.Tensor,
                                  original: torch.Tensor) -> Dict[str, float]:
        """Calculate recovery network metrics"""
        metrics = {}
        
        with torch.no_grad():
            # Basic quality metrics
            recovery_mse = F.mse_loss(recovered, original).item()
            metrics['recovery_mse'] = recovery_mse
            
            # PSNR for recovery
            recovered_01 = (torch.clamp(recovered, -1, 1) + 1) / 2
            original_01 = (torch.clamp(original, -1, 1) + 1) / 2
            
            psnr_values = []
            for i in range(recovered.size(0)):
                rec_np = recovered_01[i].cpu().numpy().transpose(1, 2, 0)
                orig_np = original_01[i].cpu().numpy().transpose(1, 2, 0)
                psnr_val = psnr(orig_np, rec_np, data_range=1.0)
                psnr_values.append(psnr_val)
            
            metrics['recovery_psnr'] = np.mean(psnr_values)
            
            # Perceptual quality
            recovery_lpips = self.lpips_fn(recovered, original).mean().item()
            metrics['recovery_lpips'] = recovery_lpips
        
        return metrics
    
    def calculate_robustness_metrics(self, outputs: Dict[str, torch.Tensor],
                                    targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate attack robustness metrics"""
        metrics = {}
        
        if 'attacked_images' not in outputs or 'stego_images' not in outputs:
            return metrics
        
        with torch.no_grad():
            # Image degradation from attack
            attack_mse = F.mse_loss(outputs['attacked_images'], 
                                   outputs['stego_images']).item()
            metrics['attack_degradation'] = attack_mse
            
            # Message survival rate under attack
            if 'decoded_messages' in outputs and 'messages' in targets:
                decoded_binary = (torch.sigmoid(outputs['decoded_messages']) > 0.5).float()
                survival_rate = (decoded_binary == targets['messages'].float()).float().mean().item()
                metrics['message_survival_rate'] = survival_rate
        
        return metrics
    
    def _check_syndrome(self, codeword: torch.Tensor) -> float:
        """Check syndrome for LDPC codewords"""
        hard_codeword = (torch.sigmoid(codeword) > 0.5).float()
        syndrome = torch.matmul(hard_codeword, self.H_matrix.T) % 2
        syndrome_rate = (syndrome.sum(dim=1) == 0).float().mean().item()
        return syndrome_rate
    
    def _update_history(self, metrics: Dict[str, float]):
        """Update metric history"""
        for key in self.metric_history:
            if key in metrics:
                self.metric_history[key].append(metrics[key])
                
                # Keep only last 100 values
                if len(self.metric_history[key]) > 100:
                    self.metric_history[key] = self.metric_history[key][-100:]
    
    def get_metric_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of metric history"""
        summary = {}
        
        for metric_name, values in self.metric_history.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1]
                }
        
        return summary
    
    def calculate_comprehensive_evaluation(self, model, test_loader,
                                         attack_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive evaluation across multiple scenarios"""
        results = {
            'overall': {},
            'per_attack': {},
            'capacity_analysis': {},
            'ldpc_analysis': {}
        }
        
        # Test each attack configuration
        for attack_config in attack_configs:
            attack_type = attack_config['type']
            attack_strength = attack_config['strength']
            
            attack_metrics = self._evaluate_single_attack(
                model, test_loader, attack_type, attack_strength
            )
            
            results['per_attack'][f'{attack_type}_{attack_strength}'] = attack_metrics
        
        # Overall metrics
        all_metrics = []
        for attack_results in results['per_attack'].values():
            all_metrics.append(attack_results)
        
        # Average across all attacks
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            results['overall'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Capacity analysis
        results['capacity_analysis'] = self._analyze_capacity(model)
        
        # LDPC performance analysis
        results['ldpc_analysis'] = self._analyze_ldpc_performance(model)
        
        return results
    
    def _evaluate_single_attack(self, model, test_loader, 
                               attack_type: str, attack_strength: float) -> Dict[str, float]:
        """Evaluate model under single attack configuration"""
        model.eval()
        metrics_accumulator = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                cover_images = batch['cover'].to(self.device)
                messages = batch['message'].to(self.device)
                
                outputs = model(
                    cover_images, messages,
                    attack_type=attack_type,
                    attack_strength=attack_strength,
                    training=False
                )
                
                targets = {
                    'cover_images': cover_images,
                    'messages': messages
                }
                
                batch_metrics = self.calculate_batch_metrics(outputs, targets, {})
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key not in metrics_accumulator:
                        metrics_accumulator[key] = 0.0
                    metrics_accumulator[key] += value
                
                num_batches += 1
        
        # Average metrics
        for key in metrics_accumulator:
            metrics_accumulator[key] /= num_batches
        
        return metrics_accumulator
    
    def _analyze_capacity(self, model) -> Dict[str, float]:
        """Analyze model capacity"""
        return model.calculate_model_capacity()
    
    def _analyze_ldpc_performance(self, model) -> Dict[str, Any]:
        """Analyze LDPC system performance"""
        ldpc_info = model.get_ldpc_info()
        
        analysis = {
            'redundancy_levels': model.ldpc_system.redundancy_levels,
            'code_rates': [],
            'expansion_factors': []
        }
        
        for redundancy in model.ldpc_system.redundancy_levels:
            info = model.ldpc_system.get_code_info(redundancy)
            analysis['code_rates'].append(info['rate'])
            analysis['expansion_factors'].append(info['expansion_factor'])
        
        return analysis