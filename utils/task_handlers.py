#!/usr/bin/env python3
"""
Shared task handling utilities for classification and force tasks.

This module provides common functions for:
- Batch data extraction
- Target extraction
- Loss function creation
- Metrics calculation
- Forward pass handling

Used by both run_spl.py (supervised training) and run_probe.py (probe evaluation).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, Union, Sequence
from model.head import MLPHead, DualForceHead


def get_batch_data(batch: Dict[str, Any], 
                   task_type: str,
                   device: torch.device,
                   training_type: Optional[str] = None,
                   apply_fft: bool = False,
                   get_batch_data_fn: Optional[Callable] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract input data from batch dictionary based on task type.
    
    Args:
        batch: Batch dictionary from dataloader
        task_type: Task type ('classification' or 'force')
        device: Device to move data to
        training_type: Training type ('images' or 'tactile') for classification
        apply_fft: Whether to apply FFT transformation (for tactile data)
        get_batch_data_fn: Optional custom function to extract data
    
    Returns:
        Input data tensor(s). For force with 2 chunks, returns (sensor_0, sensor_1).
        For others, returns single tensor.
    """
    if get_batch_data_fn is not None:
        return get_batch_data_fn(batch)
    
    if task_type == 'sliding':
        # Sliding label prediction: same inputs as force — image (tactile_img) or taxel (tactile).
        if 'tactile_img' in batch:
            return batch['tactile_img'].to(device, non_blocking=True)
        if 'tactile' in batch:
            return batch['tactile'].to(device, non_blocking=True)
        raise ValueError("Batch must contain 'tactile_img' or 'tactile' key for sliding")

    if task_type == 'force':
        # Force: images (gsmini/9dtact -> tactile_img) or taxel (xela/tac02 -> tactile) or legacy sensor
        if 'tactile_img' in batch:
            return batch['tactile_img'].to(device, non_blocking=True)
        if 'tactile' in batch:
            data = batch['tactile'].to(device, non_blocking=True)
            if apply_fft and len(data.shape) == 3:
                from utils.fft import fft_to_frequency_features
                data = fft_to_frequency_features(data, dim=1)
            return data
        if 'sensor' in batch:
            data = batch['sensor'].to(device, non_blocking=True)
            if apply_fft and len(data.shape) == 3:  # Tactile data
                from utils.fft import fft_to_frequency_features
                data = fft_to_frequency_features(data, dim=1)
            return data
        else:
            raise ValueError("Batch must contain 'tactile_img', 'tactile', or 'sensor' key for force")
    
    elif task_type == 'pose':
        # Pose: Always uses both sensor chunks
        if 'sensor_0' in batch and 'sensor_1' in batch:
            sensor_0 = batch['sensor_0'].to(device, non_blocking=True)
            sensor_1 = batch['sensor_1'].to(device, non_blocking=True)
            return sensor_0, sensor_1
        else:
            raise ValueError("Batch must contain 'sensor_0' and 'sensor_1' keys for pose")
    
    else:  # classification
        # Classification: Access data based on training type
        if training_type == 'images':
            if 'images' in batch:
                return batch['images'].to(device, non_blocking=True)
            else:
                raise ValueError("Batch must contain 'images' key for image classification")
        else:  # tactile
            if 'tactile' in batch:
                data = batch['tactile'].to(device, non_blocking=True)
                if apply_fft:
                    from utils.fft import fft_to_frequency_features
                    data = fft_to_frequency_features(data, dim=1)
                return data
            else:
                raise ValueError("Batch must contain 'tactile' key for tactile classification")


def get_target(batch: Dict[str, Any],
                task_type: str,
                device: torch.device,
                get_target_fn: Optional[Callable] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Extract target from batch dictionary based on task type.
    
    Args:
        batch: Batch dictionary from dataloader
        task_type: Task type ('classification' or 'force')
        device: Device to move targets to
        get_target_fn: Optional custom function to extract target
    
    Returns:
        Target tensor(s). Returns single tensor.
    """
    if get_target_fn is not None:
        return get_target_fn(batch)
    
    if task_type == 'classification':
        if 'object_idx' in batch:
            return batch['object_idx'].to(device, non_blocking=True)
        else:
            raise ValueError("Batch must contain 'object_idx' key for classification")

    if task_type == 'sliding':
        if 'sliding_label' in batch:
            return batch['sliding_label'].to(device, non_blocking=True).long()
        raise ValueError("Batch must contain 'sliding_label' key for sliding task")

    # force (default for non-classification tactile regression)
    if task_type == 'force':
        if '6d_force' in batch:
            # Jayaram: first 3 dims (shear_x, shear_y, normal)
            return batch['6d_force'][:, :3].to(device, non_blocking=True)
        if 'force' in batch:
            return batch['force'].to(device, non_blocking=True)
        raise ValueError("Batch must contain '6d_force' or 'force' key for force")

    raise ValueError(f"Unsupported task_type for get_target: {task_type}")


def create_loss_function(task_type: str,
                         class_weights: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None
                         ) -> nn.Module:
    """
    Create appropriate loss function based on task type.

    Args:
        task_type: Task type ('classification', 'sliding', or 'force')
        class_weights: Optional per-class weights for CE (sliding/classification).
            Accepted as torch.Tensor, np.ndarray, or sequence of floats. Ignored for force.

    Returns:
        Loss function module
    """
    if task_type == 'force':
        return nn.MSELoss()
    # classification and sliding: optionally weighted CE for class imbalance.
    weight_t = None
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            weight_t = torch.as_tensor(class_weights, dtype=torch.float32)
        else:
            weight_t = class_weights.to(dtype=torch.float32)
    # Sliding frames that fall outside any labeled bracket are emitted as -1 by
    # the dataloader and must not contribute to the loss.
    ignore_index = -1 if task_type == 'sliding' else -100
    return nn.CrossEntropyLoss(weight=weight_t, ignore_index=ignore_index)


def compute_macro_f1(preds: Union[np.ndarray, Sequence[int]],
                     targets: Union[np.ndarray, Sequence[int]],
                     num_classes: int) -> Dict[str, float]:
    """
    Macro-averaged F1 + per-class precision/recall/F1 for multi-class classification.

    Macro F1 is the unweighted mean of per-class F1; the right metric when classes
    are imbalanced (e.g. sliding bracket labels are 87/11/2). Classes absent from
    both preds and targets contribute F1=0 by convention (not silently dropped),
    matching sklearn's default zero_division=0.

    Args:
        preds: predicted class indices, shape (N,)
        targets: true class indices, shape (N,)
        num_classes: number of classes (defines per-class output array length)

    Returns:
        dict with:
            'macro_f1': mean F1 across classes (percent, 0-100)
            'per_class_f1': list of per-class F1 (percent, 0-100), length num_classes
            'per_class_precision': list of per-class precision (percent), length num_classes
            'per_class_recall': list of per-class recall (percent), length num_classes
    """
    preds_arr = np.asarray(preds, dtype=np.int64).reshape(-1)
    tgts_arr = np.asarray(targets, dtype=np.int64).reshape(-1)
    per_p, per_r, per_f = [], [], []
    for c in range(num_classes):
        tp = int(((preds_arr == c) & (tgts_arr == c)).sum())
        fp = int(((preds_arr == c) & (tgts_arr != c)).sum())
        fn = int(((preds_arr != c) & (tgts_arr == c)).sum())
        prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_p.append(prec)
        per_r.append(rec)
        per_f.append(f1)
    macro_f1 = float(np.mean(per_f)) if per_f else 0.0
    return {
        'macro_f1': macro_f1,
        'per_class_f1': per_f,
        'per_class_precision': per_p,
        'per_class_recall': per_r,
    }


def forward_pass(model: nn.Module,
                 data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 task_type: str,
                 output_dim: Optional[int] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform forward pass through model based on task type.
    
    Args:
        model: Model to forward through
        data: Input data (single tensor or tuple of two tensors for pose/force with 2 chunks)
        task_type: Task type ('classification' or 'force')
        output_dim: Output dimension for force task (used to determine if output should be squeezed)
    
    Returns:
        Model output. Returns single tensor.
    """
    # Single input: classification or force
    outputs = model(data)
    if task_type == 'force' and output_dim == 1:
        # Squeeze if output_dim == 1 for force
        outputs = outputs.squeeze(-1)
    return outputs


def compute_loss(outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                 targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
                 criterion: nn.Module,
                 task_type: str) -> torch.Tensor:
    """
    Compute loss based on task type.
    
    Args:
        outputs: Model outputs
        targets: Target values
        criterion: Loss function
        task_type: Task type ('classification' or 'force')
    
    Returns:
        Loss tensor
    """
    # Classification / sliding / force: single loss
    # Ensure targets are 1D for classification-style tasks
    if task_type in ('classification', 'sliding'):
        # Ensure targets are 1D [batch_size] with class indices
        if targets.dim() > 1:
            targets = targets.squeeze()
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)  # Make it 1D if it's a scalar
        
        # Ensure outputs are 2D [batch_size, num_classes] for classification
        if outputs.dim() == 1:
            raise RuntimeError(f"Outputs for {task_type} should be 2D [batch_size, num_classes], got shape {outputs.shape}. "
                             f"Targets shape: {targets.shape}")
        if outputs.dim() > 2:
            outputs = outputs.view(outputs.size(0), -1)
    
    return criterion(outputs, targets)


def compute_force_dual_head_loss(outputs: torch.Tensor,
                                  targets: torch.Tensor,
                                  criterion: nn.Module) -> torch.Tensor:
    """
    Compute force loss as sum of shear loss (dims 0,1) and normal loss (dim 2).
    Used when force prediction uses DualForceHead (2-head policy).

    Args:
        outputs: [B, 3] predictions (shear dims 0,1; normal dim 2)
        targets: [B, 3] targets
        criterion: Loss function (e.g. MSELoss)

    Returns:
        loss_shear + loss_normal
    """
    # Ensure we use first 3 dims for 6D force
    out_3 = outputs[:, :3] if outputs.shape[-1] >= 3 else outputs
    tgt_3 = targets[:, :3] if targets.shape[-1] >= 3 else targets
    loss_shear = criterion(out_3[:, :2], tgt_3[:, :2])
    loss_normal = criterion(out_3[:, 2:3], tgt_3[:, 2:3])
    return loss_shear + loss_normal


def postprocess_outputs(outputs: torch.Tensor,
                        task_type: str,
                        output_dim: Optional[int] = None) -> torch.Tensor:
    """
    Post-process model outputs for loss/metrics computation.

    - Classification: ensure 2D [batch_size, num_classes] (mean over seq or view)
    - Force: squeeze last dim if output_dim == 1

    Args:
        outputs: Raw model outputs
        task_type: Task type ('classification' or 'force')
        output_dim: Output dimension for force (used for squeeze when == 1)

    Returns:
        Post-processed tensor
    """
    if task_type in ('classification', 'sliding'):
        if outputs.dim() == 3:
            outputs = outputs.mean(dim=1)  # [B, seq_len, num_classes] -> [B, num_classes]
        elif outputs.dim() > 2:
            outputs = outputs.view(outputs.size(0), -1)
    elif task_type == 'force' and output_dim == 1:
        outputs = outputs.squeeze(-1)
    return outputs


def calculate_force_metrics_detailed(
    preds: np.ndarray,
    targets: np.ndarray,
    target_mean: Optional[Union[float, np.ndarray, Sequence[float]]] = None,
    target_std: Optional[Union[float, np.ndarray, Sequence[float]]] = None,
    output_dim: int = 3,
) -> Dict[str, float]:
    """
    Calculate force metrics including normal/shear breakdown for 3D or 6D force.

    For 6D force, uses first 3 dims (shear_x, shear_y, normal). Dims 0,1 = shear, dim 2 = normal.

    Args:
        preds: Predictions array [N, 3] or [N, 6]
        targets: Targets array, same shape
        target_mean: Mean for denormalization (scalar or per-dim array of length 3)
        target_std: Std for denormalization (scalar or per-dim array of length 3)
        output_dim: 3 or 6; for 6, uses preds[:, :3] and targets[:, :3]

    Returns:
        Dict with: mae, rmse, and when output_dim in (3,6): normal_mae, normal_rmse,
        shear_mae, shear_rmse, force_mae, force_rmse
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if output_dim == 6 and preds.shape[-1] >= 6:
        preds = preds[:, :3]
        targets = targets[:, :3]

    if target_mean is not None and target_std is not None:
        mean = np.asarray(target_mean, dtype=np.float64)
        std = np.asarray(target_std, dtype=np.float64)
        if mean.ndim == 0:
            mean = float(mean)
            std = float(std)
        preds = preds * std + mean
        targets = targets * std + mean

    err = preds - targets
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    result = {'mae': mae, 'rmse': rmse}

    if output_dim in (3, 6) and preds.ndim >= 2 and preds.shape[-1] >= 3:
        err_shear = err[..., 0:2]
        err_normal = err[..., 2]
        shear_mae = float(np.mean(np.abs(err_shear)))
        shear_rmse = float(np.sqrt(np.mean(err_shear ** 2)))
        normal_mae = float(np.mean(np.abs(err_normal)))
        normal_rmse = float(np.sqrt(np.mean(err_normal ** 2)))
        result['normal_mae'] = normal_mae
        result['normal_rmse'] = normal_rmse
        result['shear_mae'] = shear_mae
        result['shear_rmse'] = shear_rmse
        result['force_mae'] = mae
        result['force_rmse'] = rmse

    return result


def calculate_metrics(outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                     targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
                     task_type: str,
                     normalize_targets: bool = False,
                     target_mean: Optional[float] = None,
                     target_std: Optional[float] = None,
                     translation_std: Optional[np.ndarray] = None,
                     rotation_std: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate task-specific metrics from outputs and targets.
    
    Args:
        outputs: Model outputs
        targets: Target values
        task_type: Task type ('classification' or 'force')
        normalize_targets: For force, whether targets are normalized
        target_mean: For force, mean for denormalization
        target_std: For force, std for denormalization
        translation_std: Unused (kept for API compatibility)
        rotation_std: Unused (kept for API compatibility)
    
    Returns:
        Dictionary with metrics:
        - For classification: {'accuracy': float}
        - For force: {'mae': float, 'rmse': float}
    """
    if task_type in ('classification', 'sliding'):
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        result: Dict[str, Any] = {'accuracy': accuracy, 'correct': correct, 'total': total}
        # Macro-F1 is the meaningful metric on imbalanced label sets (sliding) and
        # cheap to compute, so we always return it for both task types.
        num_classes = int(outputs.shape[-1]) if outputs.dim() >= 2 else int(predicted.max().item() + 1)
        f1_metrics = compute_macro_f1(
            predicted.detach().cpu().numpy(),
            targets.detach().cpu().numpy(),
            num_classes=num_classes,
        )
        result.update(f1_metrics)
        return result

    elif task_type == 'force':
        # Force metrics: MAE and RMSE
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Denormalize if needed
        if normalize_targets and target_mean is not None and target_std is not None:
            outputs_np = outputs_np * target_std + target_mean
            targets_np = targets_np * target_std + target_mean
        
        mae = np.mean(np.abs(outputs_np - targets_np))
        rmse = np.sqrt(np.mean((outputs_np - targets_np) ** 2))
        return {'mae': mae, 'rmse': rmse}
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def create_probe_head(task_type: str,
                     embed_dim: int,
                     num_classes: Optional[int] = None,
                     output_dim: int = 1,
                     device: torch.device = None,
                     force_use_single_sensor: bool = False) -> Union[nn.Module, Dict[str, nn.Module]]:
    """
    Create MLP probe head(s) based on task type.
    
    Args:
        task_type: Task type ('classification' or 'force')
        embed_dim: Embedding dimension
        num_classes: Number of classes (for classification)
        output_dim: Output dimension (for force, default 1; use 6 for 6D force)
        device: Device to move model to
        force_use_single_sensor: If True and task is force, use embed_dim (single sensor) instead of 2*embed_dim
    
    Returns:
        MLP head module.
    """
    if task_type in ('classification', 'sliding'):
        assert num_classes is not None, f"num_classes must be provided for {task_type} task"
        probe_mlp = MLPHead(embed_dim, num_classes, dropout=0.1)
        if device is not None:
            probe_mlp = probe_mlp.to(device)
        return probe_mlp

    else:  # force
        # For force: use 2-head policy (shear + normal) when output_dim in (3, 6)
        in_dim = embed_dim if force_use_single_sensor else (embed_dim * 2)
        if output_dim in (3, 6):
            probe_mlp = DualForceHead(in_dim, hidden_dim=256, dropout=0.1)
        else:
            probe_mlp = MLPHead(in_dim, output_dim, hidden_dim=256, dropout=0.1)
        if device is not None:
            probe_mlp = probe_mlp.to(device)
        return probe_mlp


def extract_features(encoder: nn.Module,
                     shared_trunk: Optional[nn.Module],
                     data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     task_type: str,
                     encoder_output_is_sequence: bool = True,
                     align_pooler: Optional[nn.Module] = None,
                     finetune: bool = False) -> torch.Tensor:
    """
    Extract features from encoder (and optionally shared_trunk) for probe evaluation.

    Optimized for pose tasks by processing both sensor chunks in a single batched forward pass when possible.

    Args:
        encoder: Encoder model
        shared_trunk: Optional shared trunk model
        data: Input data (single tensor or tuple of two tensors for pose/force with 2 chunks)
        task_type: Task type ('classification' or 'force')
        encoder_output_is_sequence: Whether encoder outputs sequence
        align_pooler: Optional AlignPooler. If provided, replaces the mean-over-sequence
            step with `align_pooler(features)` → [B, D]. Used by the force probe
            when the checkpoint was trained with align_token_mode=true.

    Returns:
        Feature tensor. For force with 2 chunks, returns concatenated features [B, 2*embed_dim].
        For others, returns [B, embed_dim].
    """
    # Default path: encoder is frozen, so we run under torch.no_grad() to save
    # compute and memory. When `finetune=True` (full-model probe), we need
    # gradient flow through the encoder + trunk + align_pooler — drop to a
    # null context so autograd records the graph.
    import contextlib
    ctx = contextlib.nullcontext() if finetune else torch.no_grad()
    with ctx:
        # Single input: classification or force with single chunk
        encoder_features = encoder(data)
        if isinstance(encoder_features, tuple):
            encoder_features = encoder_features[0]

        # Pass through shared trunk if provided
        if shared_trunk is not None:
            # Ensure encoder output is 3D [B, N, C] for shared trunk
            if len(encoder_features.shape) == 2:
                encoder_features = encoder_features.unsqueeze(1)
            trunk_features = shared_trunk(encoder_features)
            if isinstance(trunk_features, tuple):
                trunk_features = trunk_features[0]
            features = trunk_features
        else:
            # No shared trunk: use encoder features directly
            features = encoder_features

        # Reduce sequence dim to a single [B, D] vector. Use the align-pooler
        # (learnable summary token over the full sequence) when provided —
        # otherwise fall back to mean-pooling (legacy probe path).
        if encoder_output_is_sequence:
            if features.dim() == 3:
                if align_pooler is not None:
                    features = align_pooler(features)  # [B, N, C] -> [B, C]
                else:
                    features = torch.mean(features, dim=1)  # [B, N, C] -> [B, C]
            elif features.dim() == 2:
                pass
            else:
                if features.dim() > 2:
                    while features.dim() > 2:
                        features = features.mean(dim=1)

        return features


def get_task_display_name(task_type: str) -> str:
    """
    Get display name for task type.
    
    Args:
        task_type: Task type ('classification' or 'force')
    
    Returns:
        Display name string
    """
    return task_type.upper()


def validate_task_type(task_type: str):
    """
    Validate task type.
    
    Args:
        task_type: Task type to validate
    
    Raises:
        ValueError: If task_type is not valid
    """
    valid_types = ['classification', 'force', 'sliding']
    
    if task_type not in valid_types:
        raise ValueError(f"task_type must be one of {valid_types}, got {task_type}")

