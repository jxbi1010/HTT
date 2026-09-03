#!/usr/bin/env python3
"""
Shared evaluation and training loop utilities for downstream tasks.

Provides run_evaluation_loop used by run_spl.py (evaluate) and run_probe.py (validation/test).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, List, Union
from tqdm import tqdm

from utils.task_handlers import compute_loss, compute_force_dual_head_loss, calculate_force_metrics_detailed, compute_macro_f1


def run_evaluation_loop(
    loader,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    get_batch_data_fn: Callable[[Any], torch.Tensor],
    get_target_fn: Callable[[Any], torch.Tensor],
    criterion: nn.Module,
    task_type: str,
    output_dim: Optional[int] = None,
    force_loss_dims: Optional[int] = None,
    force_dual_head: bool = False,
    target_mean: Optional[Union[float, np.ndarray]] = None,
    target_std: Optional[Union[float, np.ndarray]] = None,
    use_amp: bool = False,
    device_type: str = 'cuda',
    l2_reg_fn: Optional[Callable[[], torch.Tensor]] = None,
    desc: str = "Evaluating",
    skip_batch_on_error: bool = False,
) -> Tuple[float, Dict[str, Any], List, List]:
    """
    Run evaluation loop over a dataloader.

    Args:
        loader: DataLoader to iterate over
        forward_fn: (data, targets) -> outputs. Receives pre-extracted data and targets.
                    Must return postprocessed outputs (2D for classification, correct shape for force).
        get_batch_data_fn: (batch) -> data tensor
        get_target_fn: (batch) -> targets tensor
        criterion: Loss function
        task_type: 'classification' or 'force'
        output_dim: Output dimension for force (used for 6D slice)
        force_loss_dims: When 3, use outputs[:, :3] and targets[:, :3] for loss (6D force)
        force_dual_head: When True, loss = loss_shear + loss_normal (2-head policy)
        target_mean: For force metrics denormalization
        target_std: For force metrics denormalization
        use_amp: Use autocast for forward
        device_type: 'cuda' or 'cpu' for autocast
        l2_reg_fn: Optional callable returning L2 regularization tensor to add to loss
        desc: Progress bar description
        skip_batch_on_error: If True, skip batch on get_batch_data/get_target error (probe style)

    Returns:
        (avg_loss, metrics_dict, all_predictions, all_targets)
        metrics_dict: classification -> {'accuracy': float}; force -> {'mae', 'rmse', 'normal_mae'?, 'shear_mae'?}
    """
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    all_predictions: List = []
    all_targets: List = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            try:
                data = get_batch_data_fn(batch)
                targets = get_target_fn(batch)
            except (ValueError, KeyError):
                if skip_batch_on_error:
                    continue
                raise

            batch_count += 1

            if use_amp:
                with torch.amp.autocast(device_type=device_type):
                    outputs = forward_fn(data, targets)
                    if task_type == 'force' and force_dual_head:
                        loss = compute_force_dual_head_loss(outputs, targets, criterion)
                    elif force_loss_dims == 3 and task_type == 'force':
                        loss = compute_loss(
                            outputs[:, :3], targets[:, :3], criterion, task_type
                        )
                    else:
                        loss = compute_loss(outputs, targets, criterion, task_type)
                    if l2_reg_fn is not None:
                        loss = loss + l2_reg_fn()
            else:
                outputs = forward_fn(data, targets)
                if task_type == 'force' and force_dual_head:
                    loss = compute_force_dual_head_loss(outputs, targets, criterion)
                elif force_loss_dims == 3 and task_type == 'force':
                    loss = compute_loss(
                        outputs[:, :3], targets[:, :3], criterion, task_type
                    )
                else:
                    loss = compute_loss(outputs, targets, criterion, task_type)
                if l2_reg_fn is not None:
                    loss = loss + l2_reg_fn()

            total_loss += loss.item()

            if task_type == 'force':
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else:
                # Classification / sliding: outputs already postprocessed to 2D
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)
                elif outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                if targets.dim() > 1:
                    targets = targets.squeeze()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

    num_iterations = batch_count if batch_count > 0 else 1
    avg_loss = total_loss / num_iterations

    if task_type == 'force':
        preds_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        if force_loss_dims == 3 and preds_array.shape[-1] >= 6:
            preds_array = preds_array[:, :3]
            targets_array = targets_array[:, :3]
        metrics = calculate_force_metrics_detailed(
            preds_array, targets_array,
            target_mean=target_mean, target_std=target_std,
            output_dim=3 if preds_array.shape[-1] == 3 else (output_dim or 3)
        )
        metrics_dict = {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'normal_mae': metrics.get('normal_mae'),
            'shear_mae': metrics.get('shear_mae'),
        }
    else:
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        metrics_dict = {'accuracy': accuracy}
        # Add macro-F1 + per-class breakdown for both classification and sliding.
        # Cheap (one numpy pass) and the only meaningful metric for the imbalanced
        # sliding task; for balanced classification it just confirms accuracy.
        if all_predictions and all_targets:
            num_classes = int(np.max(np.concatenate([all_predictions, all_targets]))) + 1
            num_classes = max(num_classes, 2)
            f1_metrics = compute_macro_f1(all_predictions, all_targets, num_classes=num_classes)
            metrics_dict.update(f1_metrics)

    return avg_loss, metrics_dict, all_predictions, all_targets
