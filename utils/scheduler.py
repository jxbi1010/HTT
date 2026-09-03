#!/usr/bin/env python3
"""
Learning Rate Scheduler Utilities

This module provides centralized learning rate scheduler implementations
commonly used across the tactile fusion training scripts.
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, Any
import math


class LinearWarmupCosineDecayLR(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    
    During warmup, the learning rate increases linearly from warmup_start_lr to the initial LR.
    After warmup, it decays using cosine annealing.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of steps for linear warmup
            total_steps: Total number of training steps
            warmup_start_lr: Starting learning rate for warmup (default: 0.0)
            eta_min: Minimum learning rate for cosine decay (default: 0.0)
            last_epoch: The index of the last epoch (default: -1)
        """
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        # Convert learning rates to floats (in case they're strings from OmegaConf)
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Compute learning rate using linear warmup followed by cosine decay.
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            warmup_factor = self.last_epoch / max(self.warmup_steps, 1)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay phase
            progress = (self.last_epoch - self.warmup_steps) / max(
                (self.total_steps - self.warmup_steps), 1
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(scheduler_type: str, optimizer: optim.Optimizer, **kwargs) -> optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler based on type.
    
    Args:
        scheduler_type: Type of scheduler ('cosine', 'cosine_warm_restarts', 'exponential', 'linear_warmup_cosine')
        optimizer: The optimizer whose learning rate will be scheduled
        **kwargs: Scheduler-specific parameters
            For 'linear_warmup_cosine': warmup_steps, total_steps, warmup_start_lr, eta_min
        
    Returns:
        Scheduler instance
        
    Raises:
        ValueError: If scheduler_type is not supported
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=kwargs.get('T_max', kwargs.get('epochs', 100)),
            eta_min=kwargs.get('eta_min', optimizer.param_groups[0]['lr'] * 0.01),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'cosine_warm_restarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 1),
            eta_min=kwargs.get('eta_min', optimizer.param_groups[0]['lr'] * 0.01),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=kwargs.get('gamma', 0.95),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'linear_warmup_cosine':
        return LinearWarmupCosineDecayLR(
            optimizer=optimizer,
            warmup_steps=kwargs.get('warmup_steps', kwargs.get('warmup_epochs', 0) * kwargs.get('steps_per_epoch', 1)),
            total_steps=kwargs.get('total_steps', kwargs.get('epochs', 100) * kwargs.get('steps_per_epoch', 1)),
            warmup_start_lr=kwargs.get('warmup_start_lr', 0.0),
            eta_min=kwargs.get('eta_min', optimizer.param_groups[0]['lr'] * 0.01),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                        f"Supported types: cosine, cosine_warm_restarts, exponential, linear_warmup_cosine")




def get_scheduler_info(scheduler: optim.lr_scheduler._LRScheduler) -> Dict[str, Any]:
    """
    Get information about a learning rate scheduler.
    
    Args:
        scheduler: The learning rate scheduler
        
    Returns:
        Dictionary containing scheduler information
    """
    info = {
        'type': type(scheduler).__name__,
        'current_lr': scheduler.get_last_lr(),
    }
    
    if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
        info.update({
            'T_max': scheduler.T_max,
            'eta_min': scheduler.eta_min,
            'last_epoch': scheduler.last_epoch
        })
    elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
        info.update({
            'T_0': scheduler.T_0,
            'T_mult': scheduler.T_mult,
            'eta_min': scheduler.eta_min,
            'last_epoch': scheduler.last_epoch
        })
    elif isinstance(scheduler, optim.lr_scheduler.ExponentialLR):
        info.update({
            'gamma': scheduler.gamma,
            'last_epoch': scheduler.last_epoch
        })
    elif isinstance(scheduler, LinearWarmupCosineDecayLR):
        info.update({
            'warmup_steps': scheduler.warmup_steps,
            'total_steps': scheduler.total_steps,
            'warmup_start_lr': scheduler.warmup_start_lr,
            'eta_min': scheduler.eta_min,
            'last_epoch': scheduler.last_epoch
        })

    
    return info


def print_scheduler_info(scheduler: optim.lr_scheduler._LRScheduler, prefix: str = "  ") -> None:
    """
    Print information about a learning rate scheduler.
    
    Args:
        scheduler: The learning rate scheduler
        prefix: Prefix for each line of output
    """
    info = get_scheduler_info(scheduler)
    
    print(f"{prefix}Scheduler Type: {info['type']}")
    print(f"{prefix}Current Learning Rate: {info['current_lr'][0]:.6f}")
    
    # Print scheduler-specific parameters
    for key, value in info.items():
        if key not in ['type', 'current_lr']:
            print(f"{prefix}{key}: {value}")




if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    
    # Create a dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("=== Testing Scheduler Factory ===")
    
    # Test different scheduler configurations
    schedulers = [
        ('cosine', {'T_max': 100, 'eta_min': 1e-6}),
        ('cosine_warm_restarts', {'T_0': 10, 'T_mult': 2}),
        ('exponential', {'gamma': 0.95}),
        ('linear_warmup_cosine', {'warmup_steps': 100, 'total_steps': 1000, 'warmup_start_lr': 0.0, 'eta_min': 1e-6})
    ]
    
    for scheduler_type, params in schedulers:
        print(f"\n=== {scheduler_type.upper()} Scheduler ===")
        scheduler = create_scheduler(scheduler_type, optimizer, **params)
        print_scheduler_info(scheduler)