#!/usr/bin/env python3
"""
Optimizer Module
Optimizer configurations for training
"""

import torch
from torch import optim
from typing import Any, List, Dict
import logging


def get_optimizer(model, config) -> torch.optim.Optimizer:
    """
    Get optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: Configuration object with optimizer settings
        
    Returns:
        PyTorch optimizer
    """
    optimizer_type = config.get('optimizer_type', 'adamw')
    lr = config.learning_rate
    weight_decay = config.weight_decay
    
    # Get parameters with custom weight decay
    params = get_parameter_groups(model, weight_decay)
    
    if optimizer_type == 'adam':
        return optim.Adam(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    elif optimizer_type == 'adamw':
        return AdamWOptimizer(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'sgd':
        return optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=0.99,
            eps=1e-8,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'lamb':
        return LAMBOptimizer(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'radam':
        return RAdamOptimizer(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'lookahead':
        base_optimizer = optim.Adam(params, lr=lr)
        return LookaheadOptimizer(base_optimizer, k=5, alpha=0.5)
    
    else:
        logging.warning(f"Unknown optimizer type: {optimizer_type}, using AdamW")
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_parameter_groups(model, weight_decay: float) -> List[Dict[str, Any]]:
    """
    Get parameter groups with custom weight decay
    
    Args:
        model: PyTorch model
        weight_decay: Base weight decay value
        
    Returns:
        List of parameter groups
    """
    # Don't apply weight decay to bias and normalization parameters
    no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm', 'GroupNorm']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    return optimizer_grouped_parameters


class AdamWOptimizer(optim.AdamW):
    """AdamW optimizer with gradient centralization"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, gradient_centralization=True):
        self.gradient_centralization = gradient_centralization
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
    
    def step(self, closure=None):
        """Performs a single optimization step with gradient centralization"""
        loss = None
        if closure is not None:
            loss = closure()
        
        if self.gradient_centralization:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Gradient centralization
                    if len(p.grad.shape) > 1:
                        p.grad.data.add_(-p.grad.data.mean(dim=tuple(range(1, len(p.grad.shape))), 
                                                           keepdim=True))
        
        super().step(closure)
        return loss


class LAMBOptimizer(optim.Optimizer):
    """Layer-wise Adaptive Moments optimizer for Batch training (LAMB)"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, adam=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in [torch.float16, torch.bfloat16]:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate
                if self.adam:
                    # Adam update
                    update = exp_avg / bias_correction1 / (exp_avg_sq.sqrt() / 
                            bias_correction2.sqrt() + group['eps'])
                else:
                    # LAMB update
                    update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Compute norms
                p_norm = p.data.pow(2).sum().sqrt()
                update_norm = update.pow(2).sum().sqrt()
                
                # Compute adaptive lr
                if p_norm > 0 and update_norm > 0:
                    adaptive_lr = group['lr'] * (p_norm / update_norm)
                else:
                    adaptive_lr = group['lr']
                
                # Update parameters
                p.data.add_(update, alpha=-adaptive_lr)
        
        return loss


class RAdamOptimizer(optim.Optimizer):
    """Rectified Adam optimizer"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                state['step'] += 1
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute the length of the approximated SMA
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / bias_correction2
                
                # Compute adaptive learning rate
                if rho_t > 5:
                    # Variance is tractable
                    var_rectified = ((rho_t - 4) * (rho_t - 2) * rho_inf / 
                                   ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    
                    step_size = group['lr'] * var_rectified / bias_correction1
                    
                    # Adam update
                    denom = exp_avg_sq.sqrt() + group['eps']
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Variance is not tractable, use SGD update
                    if self.degenerated_to_sgd:
                        step_size = group['lr'] / bias_correction1
                        p.data.add_(exp_avg, alpha=-step_size)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        
        return loss


class LookaheadOptimizer(optim.Optimizer):
    """Lookahead optimizer wrapper"""
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        
        # Initialize slow weights
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['slow_buffer'] = p.data.clone()
    
    def step(self, closure=None):
        """Performs k steps with base optimizer and lookahead update"""
        loss = self.base_optimizer.step(closure)
        
        # Lookahead update every k steps
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                
                if 'step_counter' not in param_state:
                    param_state['step_counter'] = 0
                
                param_state['step_counter'] += 1
                
                if param_state['step_counter'] % self.k == 0:
                    # Update slow weights
                    slow_buffer = param_state['slow_buffer']
                    slow_buffer.add_(p.data - slow_buffer, alpha=self.alpha)
                    p.data.copy_(slow_buffer)
        
        return loss
    
    def state_dict(self):
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)