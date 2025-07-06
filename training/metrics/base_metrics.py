#!/usr/bin/env python3
"""
Base Metrics
Basic metrics for steganography evaluation
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips


class BaseMetric:
    """Base class for all metrics"""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset metric state"""
        self.total = 0.0
        self.count = 0
    
    def update(self, value: float, count: int = 1):
        """Update metric with new value"""
        self.total += value * count
        self.count += count
    
    def compute(self) -> float:
        """Compute metric value"""
        if self.count == 0:
            return 0.0
        return self.total / self.count
    
    def __call__(self, *args, **kwargs) -> float:
        """Calculate metric for batch"""
        raise NotImplementedError


class MessageAccuracy(BaseMetric):
    """Message accuracy metric"""
    
    def __init__(self):
        super().__init__('message_accuracy')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate message accuracy"""
        with torch.no_grad():
            # Convert to binary
            pred_binary = (torch.sigmoid(predicted) > 0.5).float()
            target_binary = (target > 0.5).float()
            
            # Calculate accuracy
            correct = (pred_binary == target_binary).float()
            accuracy = correct.mean().item()
            
            return accuracy


class BitErrorRate(BaseMetric):
    """Bit error rate metric"""
    
    def __init__(self):
        super().__init__('bit_error_rate')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate bit error rate"""
        with torch.no_grad():
            # Convert to binary
            pred_binary = (torch.sigmoid(predicted) > 0.5).float()
            target_binary = (target > 0.5).float()
            
            # Calculate errors
            errors = (pred_binary != target_binary).float()
            ber = errors.mean().item()
            
            return ber


class MessageSuccessRate(BaseMetric):
    """Message-level success rate (all bits correct)"""
    
    def __init__(self):
        super().__init__('message_success_rate')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate message success rate"""
        with torch.no_grad():
            batch_size = target.size(0)
            
            # Convert to binary
            pred_binary = (torch.sigmoid(predicted) > 0.5).float()
            target_binary = (target > 0.5).float()
            
            # Check if all bits in each message are correct
            correct = (pred_binary == target_binary).view(batch_size, -1).all(dim=1)
            success_rate = correct.float().mean().item()
            
            return success_rate


class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio metric"""
    
    def __init__(self, data_range: float = 2.0):
        super().__init__('psnr')
        self.data_range = data_range
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate PSNR"""
        with torch.no_grad():
            # Convert to numpy
            pred_np = predicted.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # Calculate PSNR for each image in batch
            psnr_values = []
            for i in range(pred_np.shape[0]):
                # Convert from [-1, 1] to [0, 1]
                pred_img = (pred_np[i] + 1) / 2
                target_img = (target_np[i] + 1) / 2
                
                # Transpose to HWC format
                pred_img = np.transpose(pred_img, (1, 2, 0))
                target_img = np.transpose(target_img, (1, 2, 0))
                
                # Calculate PSNR
                psnr_val = psnr(target_img, pred_img, data_range=1.0)
                psnr_values.append(psnr_val)
            
            return np.mean(psnr_values)


class SSIM(BaseMetric):
    """Structural Similarity Index metric"""
    
    def __init__(self):
        super().__init__('ssim')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate SSIM"""
        with torch.no_grad():
            # Convert to numpy
            pred_np = predicted.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # Calculate SSIM for each image in batch
            ssim_values = []
            for i in range(pred_np.shape[0]):
                # Convert from [-1, 1] to [0, 1]
                pred_img = (pred_np[i] + 1) / 2
                target_img = (target_np[i] + 1) / 2
                
                # Transpose to HWC format
                pred_img = np.transpose(pred_img, (1, 2, 0))
                target_img = np.transpose(target_img, (1, 2, 0))
                
                # Calculate SSIM
                ssim_val = ssim(target_img, pred_img, data_range=1.0, multichannel=True)
                ssim_values.append(ssim_val)
            
            return np.mean(ssim_values)


class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity metric"""
    
    def __init__(self, net: str = 'alex'):
        super().__init__('lpips')
        self.lpips_fn = lpips.LPIPS(net=net, verbose=False)
        self.device = None
    
    def to(self, device):
        """Move to device"""
        self.device = device
        self.lpips_fn = self.lpips_fn.to(device)
        return self
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate LPIPS"""
        with torch.no_grad():
            # Ensure on correct device
            if self.device:
                predicted = predicted.to(self.device)
                target = target.to(self.device)
            
            # Calculate LPIPS
            lpips_val = self.lpips_fn(predicted, target).mean().item()
            
            return lpips_val


class EmbeddingStrength(BaseMetric):
    """Embedding strength metric"""
    
    def __init__(self):
        super().__init__('embedding_strength')
    
    def __call__(self, stego: torch.Tensor, cover: torch.Tensor) -> float:
        """Calculate embedding strength"""
        with torch.no_grad():
            # Calculate absolute difference
            diff = torch.abs(stego - cover)
            
            # Average embedding strength
            strength = diff.mean().item()
            
            return strength


class SignalToNoiseRatio(BaseMetric):
    """Signal-to-Noise Ratio metric"""
    
    def __init__(self):
        super().__init__('snr')
    
    def __call__(self, stego: torch.Tensor, cover: torch.Tensor) -> float:
        """Calculate SNR"""
        with torch.no_grad():
            # Signal power (cover image)
            signal_power = (cover ** 2).mean()
            
            # Noise power (difference)
            noise = stego - cover
            noise_power = (noise ** 2).mean()
            
            # SNR in dB
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            
            return snr.item()


class MetricCollection:
    """Collection of metrics"""
    
    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = {metric.name: metric for metric in metrics}
    
    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, outputs: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]):
        """Update metrics with batch data"""
        results = {}
        
        # Message metrics
        if 'decoded_messages' in outputs and 'messages' in targets:
            if 'message_accuracy' in self.metrics:
                acc = self.metrics['message_accuracy'](
                    outputs['decoded_messages'],
                    targets['messages']
                )
                self.metrics['message_accuracy'].update(acc)
                results['message_accuracy'] = acc
            
            if 'bit_error_rate' in self.metrics:
                ber = self.metrics['bit_error_rate'](
                    outputs['decoded_messages'],
                    targets['messages']
                )
                self.metrics['bit_error_rate'].update(ber)
                results['bit_error_rate'] = ber
            
            if 'message_success_rate' in self.metrics:
                msr = self.metrics['message_success_rate'](
                    outputs['decoded_messages'],
                    targets['messages']
                )
                self.metrics['message_success_rate'].update(msr)
                results['message_success_rate'] = msr
        
        # Image quality metrics
        if 'stego_images' in outputs and 'cover_images' in targets:
            if 'psnr' in self.metrics:
                psnr_val = self.metrics['psnr'](
                    outputs['stego_images'],
                    targets['cover_images']
                )
                self.metrics['psnr'].update(psnr_val)
                results['psnr'] = psnr_val
            
            if 'ssim' in self.metrics:
                ssim_val = self.metrics['ssim'](
                    outputs['stego_images'],
                    targets['cover_images']
                )
                self.metrics['ssim'].update(ssim_val)
                results['ssim'] = ssim_val
            
            if 'lpips' in self.metrics:
                lpips_val = self.metrics['lpips'](
                    outputs['stego_images'],
                    targets['cover_images']
                )
                self.metrics['lpips'].update(lpips_val)
                results['lpips'] = lpips_val
            
            if 'embedding_strength' in self.metrics:
                strength = self.metrics['embedding_strength'](
                    outputs['stego_images'],
                    targets['cover_images']
                )
                self.metrics['embedding_strength'].update(strength)
                results['embedding_strength'] = strength
            
            if 'snr' in self.metrics:
                snr_val = self.metrics['snr'](
                    outputs['stego_images'],
                    targets['cover_images']
                )
                self.metrics['snr'].update(snr_val)
                results['snr'] = snr_val
        
        return results
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def to(self, device):
        """Move metrics to device"""
        for metric in self.metrics.values():
            if hasattr(metric, 'to'):
                metric.to(device)
        return self


class ConfusionMatrix:
    """Confusion matrix for binary classification"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset confusion matrix"""
        self.tp = 0  # True positives
        self.tn = 0  # True negatives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
    
    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Update confusion matrix"""
        with torch.no_grad():
            # Convert to binary
            pred_binary = (torch.sigmoid(predicted) > 0.5).float()
            target_binary = (target > 0.5).float()
            
            # Calculate confusion matrix elements
            self.tp += ((pred_binary == 1) & (target_binary == 1)).sum().item()
            self.tn += ((pred_binary == 0) & (target_binary == 0)).sum().item()
            self.fp += ((pred_binary == 1) & (target_binary == 0)).sum().item()
            self.fn += ((pred_binary == 0) & (target_binary == 1)).sum().item()
    
    def precision(self) -> float:
        """Calculate precision"""
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)
    
    def recall(self) -> float:
        """Calculate recall"""
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    def f1_score(self) -> float:
        """Calculate F1 score"""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def accuracy(self) -> float:
        """Calculate accuracy"""
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total


def create_default_metrics(device: str = 'cpu') -> MetricCollection:
    """Create default metric collection"""
    metrics = [
        MessageAccuracy(),
        BitErrorRate(),
        MessageSuccessRate(),
        PSNR(),
        SSIM(),
        LPIPS(),
        EmbeddingStrength(),
        SignalToNoiseRatio()
    ]
    
    collection = MetricCollection(metrics)
    collection.to(device)
    
    return collection