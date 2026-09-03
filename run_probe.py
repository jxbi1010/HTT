#!/usr/bin/env python3
"""
Generalized Probe Evaluation System.

Supports classification, force, and sliding tasks:
- Classification: object classification (uses CrossEntropyLoss, accuracy metrics)
- Force: force estimation (uses MSELoss, MAE/RMSE metrics)
- Sliding: bracket labels 0/1/2 via same CrossEntropy as classification (all frames labeled)
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from datetime import datetime

from model.head import MLPHead
from utils.utils_model import set_all_seeds, adjust_state_dict_keys
from utils.task_handlers import (
    get_batch_data, get_target, create_loss_function, forward_pass, compute_loss,
    compute_force_dual_head_loss,
    calculate_metrics, postprocess_outputs, calculate_force_metrics_detailed,
    create_probe_head, extract_features, get_task_display_name, validate_task_type
)
from utils.downstream_loops import run_evaluation_loop
from utils.result_aggregation import aggregate_force_results, aggregate_classification_results
from data.gsmini_force_4probe_50each_dataloader import (
    create_force_4probe_50each_dataloader as create_force_dataloaders_4probe,
    load_force_stats as load_force_stats_4probe,
    load_force_stats_per_dim as load_force_stats_per_dim_4probe,
)
from data.taxel_force_4probe_50each_dataloader import (
    create_taxel_force_dataloader,
    load_force_stats_per_dim as load_taxel_force_stats_per_dim,
    TAXEL_DIMS as TAXEL_FORCE_DIMS,
)
from torch.utils.data import DataLoader, RandomSampler, IterableDataset


class ProbeDataloaderManager:
    """Manages dataloaders for different probe tasks."""
    
    def __init__(self, task_type: str, modality: str, config: OmegaConf):
        """
        Initialize dataloader manager.
        
        Args:
            task_type: Task type ('classification', 'force', or 'pose')
            modality: Modality name (e.g., '9dtact', 'xela', 'gsmini', 'tac02')
            config: Configuration object
        """
        self.task_type = task_type
        self.modality = modality
        self.config = config
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        
    def load_classification_dataloaders(self, probe_train_loader, probe_val_loader, 
                                       probe_test_loader, probe_train_dataset):
        """
        Load classification dataloaders (from existing loaders).
        
        Args:
            probe_train_loader: Training dataloader
            probe_val_loader: Validation dataloader
            probe_test_loader: Test dataloader
            probe_train_dataset: Training dataset (for getting num_classes)
        """
        self.train_loader = probe_train_loader
        self.val_loader = probe_val_loader
        self.test_loader = probe_test_loader
        self.train_dataset = probe_train_dataset
        
    def _load_apply_background_subtraction(self, config_dir: str = 'config/data') -> bool:
        """
        Load apply_background_subtraction from sensor config file.
        
        Args:
            config_dir: Directory containing sensor config YAML files (should be 'config/sensor' for sensor configs)
            
        Returns:
            apply_background_subtraction value (default: True)
        """
        # Try sensor config path first (config/sensor/{modality}_config.yaml)
        sensor_config_path = os.path.join('config/sensor', f"{self.modality}_config.yaml")
        if os.path.exists(sensor_config_path):
            try:
                with open(sensor_config_path, 'r') as f:
                    sensor_config = yaml.safe_load(f) or {}
                    if 'data' in sensor_config and 'apply_background_subtraction' in sensor_config['data']:
                        return bool(sensor_config['data']['apply_background_subtraction'])
            except Exception as e:
                print(f"  Warning: Error loading apply_background_subtraction from {sensor_config_path}: {e}, using default True")
        
        # Fallback to config_dir path (config/data/{modality}.yaml)
        config_path = os.path.join(config_dir, f"{self.modality}.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    if 'apply_background_subtraction' in config:
                        return bool(config['apply_background_subtraction'])
                    if 'data' in config and 'apply_background_subtraction' in config['data']:
                        return bool(config['data']['apply_background_subtraction'])
            except Exception as e:
                print(f"  Warning: Error loading apply_background_subtraction from {config_path}: {e}, using default True")
        
        # Default to True if not found
        return True
    
    def load_force_dataloaders(self, data_roots: Dict[str, str], 
                                 config_dir: str = 'config/data',
                                 random_seed: int = 42,
                                 mode_filter: Optional[str] = None,
                                 compute_friction_mu: Optional[bool] = None,
                                 friction_mu_eps: float = 1e-3,
                                 labeled_data_root: Optional[str] = None,
                                 load_sliding_labels: Optional[bool] = None,
                                 strict_labeled: bool = False):
        """
        Load force dataloaders (WebDataset-based).
        
        Args:
            data_roots: Dictionary mapping modality names to data root directories (H5 format)
            config_dir: Directory containing sensor config YAML files
            random_seed: Random seed for data splitting (default: 42, not used for WebDataset)
            mode_filter: For gsmini 4probe: "static" or "sliding" only; None for all
            compute_friction_mu: If True, batches include friction_mu; None = auto (True when mode_filter is sliding)
            friction_mu_eps: Added to |normal| (physical, after ref subtract+clip) for friction_mu
            labeled_data_root: Root for ``*.labeled.npz`` (default: sibling ``sliding_labeled`` of processed).
            load_sliding_labels: Precomputed bracket labels + mus/c_plus; None = auto (on when mode is sliding).
            strict_labeled: If True, skip episodes without a valid matching ``.labeled.npz``.
        """
        training_config = self.config.training
        
        batch_size = training_config.get('batch_size', 32)
        num_workers = training_config.get('num_workers', 4)
        
        # Load apply_background_subtraction from sensor config
        apply_background_subtraction = self._load_apply_background_subtraction(config_dir)
        
        # Get prefetch settings from config
        pin_memory = training_config.get('pin_memory', True)
        persistent_workers = training_config.get('persistent_workers', False) if num_workers > 0 else False
        prefetch_factor = training_config.get('prefetch_factor', 2) if num_workers > 0 else None
        timeout = training_config.get('timeout', 30) if num_workers > 0 else 0

        # Image-based force sensors: gsmini and 9dtact
        if self.modality in ('gsmini', '9dtact'):
            from data.dataset_paths import probe_processed_root
            default_root = probe_processed_root(self.modality, mode_filter)
            data_root = data_roots.get(self.modality, default_root)
            common = dict(
                data_root=data_root,
                modality=self.modality,
                batch_size=batch_size,
                config_dir=config_dir,
                apply_background_subtraction=apply_background_subtraction,
                apply_ref_force_subtraction=True,
                force_clip_min=-20.0,
                force_clip_max=20.0,
                mode_filter=mode_filter,
                compute_friction_mu=compute_friction_mu,
                friction_mu_eps=friction_mu_eps,
                labeled_data_root=labeled_data_root,
                load_sliding_labels=load_sliding_labels,
                strict_labeled=strict_labeled,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                timeout=timeout,
                seed=random_seed,
            )
            self.train_loader = create_force_dataloaders_4probe(split='train', **common)
            self.val_loader = create_force_dataloaders_4probe(split='val', **common)
            self.test_loader = create_force_dataloaders_4probe(split='test', **common)
        elif self.modality in TAXEL_FORCE_DIMS:
            # Taxel-based force sensors: xela and tac02
            taxel_dim = TAXEL_FORCE_DIMS[self.modality]
            from data.dataset_paths import probe_processed_root
            default_root = probe_processed_root(self.modality, mode_filter)
            data_root = data_roots.get(self.modality, default_root)
            common = dict(
                modality=self.modality,
                taxel_dim=taxel_dim,
                data_root=data_root,
                batch_size=batch_size,
                config_dir=config_dir,
                apply_ref_tactile_subtraction=True,
                apply_ref_force_subtraction=True,
                apply_force_normalization=True,
                force_clip_min=-20.0,
                force_clip_max=20.0,
                mode_filter=mode_filter,
                labeled_data_root=labeled_data_root,
                load_sliding_labels=load_sliding_labels,
                strict_labeled=strict_labeled,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                timeout=timeout,
                seed=random_seed,
            )
            self.train_loader = create_taxel_force_dataloader(split='train', **common)
            self.val_loader = create_taxel_force_dataloader(split='val', **common)
            self.test_loader = create_taxel_force_dataloader(split='test', **common)
        else:
            raise ValueError(
                f"Force dataloaders not supported for modality '{self.modality}'. "
                f"Supported: gsmini, 9dtact, xela, tac02."
            )

        # Store dataset reference (for getting stats)
        self.train_dataset = self.train_loader.dataset


class ProbeCheckpointLoader:
    """Helper class to load checkpoints and extract necessary information."""
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, device: torch.device = None):
        """
        Load checkpoint and return model state dict and config info.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load checkpoint on (default: CPU)
        
        Returns:
            Dictionary with checkpoint data
        """
        if device is None:
            device = torch.device('cpu')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint
    
    @staticmethod
    def get_config_paths(checkpoint: Dict[str, Any], 
                        default_pretrain_config: str = 'config/model/pretrain.yaml',
                        default_ssl_config: str = 'config/algo/mae_pretrain.yaml'):
        """
        Extract config paths from checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            default_pretrain_config: Default pretrain config path
            default_ssl_config: Default SSL config path
        
        Returns:
            Dictionary with config paths
        """
        config_paths = {
            'pretrain_config_path': checkpoint.get('pretrain_config_path', default_pretrain_config),
            'ssl_config_path': checkpoint.get('ssl_config_path', default_ssl_config),
        }
        return config_paths


def run_probe_mlp(encoder: nn.Module, 
                  shared_trunk: Optional[nn.Module], 
                  device: torch.device,
                  probe_train_loader,
                  probe_val_loader,
                  probe_test_loader,
                  task_type: str = 'classification',  # 'classification', 'force', or 'pose'
                  num_classes: Optional[int] = None,  # For classification
                  output_dim: int = 1,  # For force (default 1 for force)
                  embed_dim: int = 256,
                  encoder_output_is_sequence: bool = True,
                  probe_total_steps: int = 10000,
                  probe_val_interval: int = 50,  # Used when eval_steps not provided
                  eval_steps: Optional[int] = None,  # Step-based validation interval (overrides probe_val_interval)
                  probe_lr: float = 3e-4,
                  probe_seeds: List[int] = [10, 20, 30, 40, 50],
                  writer: Optional[SummaryWriter] = None,
                  modality: str = "",
                  get_batch_data_fn: Optional[Callable] = None,
                  get_target_fn: Optional[Callable] = None,
                  force_single_sensor: bool = False,  # For force: single sensor (e.g. Jayaram) -> embed_dim, not 2*embed_dim
                  normalize_targets: bool = False,  # For force: whether targets are normalized
                  target_mean: Optional[float] = None,  # For force: mean for denormalization
                  target_std: Optional[float] = None,
                  translation_std: Optional[np.ndarray] = None,  # For pose: translation std for denormalization
                  rotation_std: Optional[np.ndarray] = None,  # For pose: rotation std for denormalization
                  lr_scheduler_config: Optional[Dict[str, Any]] = None,
                  class_weights: Optional[Any] = None,
                  probe_patience: int = 5,
                  probe_weight_decay: float = 1e-4,
                  align_pooler: Optional[nn.Module] = None,
                  finetune: bool = False) -> Dict[str, Any]:
    """
    Run MLP probe evaluation on a frozen encoder and shared trunk.
    Supports classification, force, and pose tasks.
    
    Args:
        encoder: Frozen encoder model
        shared_trunk: Frozen shared trunk model (optional, if None, only encoder is used)
        device: Device to run on
        probe_train_loader: Training dataloader for probe
        probe_val_loader: Validation dataloader for probe
        probe_test_loader: Test dataloader for probe
        task_type: Task type ('classification', 'force', or 'pose')
        num_classes: Number of classes (for classification)
        output_dim: Output dimension (for force, default 1; not used for pose)
        embed_dim: Embedding dimension
        encoder_output_is_sequence: Whether encoder outputs sequence
        probe_total_steps: Total number of training steps
        probe_val_interval: Fallback validation interval (used when eval_steps is None)
        eval_steps: Step-based validation interval (overrides probe_val_interval when set)
        probe_lr: Learning rate for probe training
        probe_seeds: List of seeds for multiple runs
        writer: Optional TensorBoard writer
        modality: Modality name for logging (empty string for no prefix)
        get_batch_data_fn: Optional function to extract input data from batch
        get_target_fn: Optional function to extract target from batch
        normalize_targets: For force, whether targets are normalized
        target_mean: For force, mean for denormalization
        target_std: For force, std for denormalization
        translation_std: For pose, translation std for denormalization (delta_normalized * std = delta_raw)
        rotation_std: For pose, rotation std for denormalization (delta_normalized * std = delta_raw)
        lr_scheduler_config: Optional learning rate scheduler configuration dict
                           with 'type' and 'kwargs' keys
        
    Returns:
        Dictionary with probe results
    """
    validate_task_type(task_type)

    # Use eval_steps (step-based) when set, else probe_val_interval
    val_interval = eval_steps if eval_steps is not None else probe_val_interval
    
    if task_type in ('classification', 'sliding'):
        assert num_classes is not None, f"num_classes must be provided for {task_type} task"
    elif task_type == 'force':
        assert output_dim > 0, "output_dim must be positive for force task"
    
    modality_prefix = f"_{modality}" if modality else ""
    modality_display = f" - {modality}" if modality else ""
    task_display = get_task_display_name(task_type)
    
    mode_label = "FINE-TUNE" if finetune else "FROZEN"
    print(f"\n{task_display} PROBE{modality_display}  [{mode_label}]")

    # Frozen probe (default): encoder + shared_trunk + align_pooler are all
    # in eval() with requires_grad=False. Only the probe head trains.
    # finetune=True: unfreeze the backbone — all components train alongside
    # the head, single optimizer, same LR.
    if finetune:
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad = True
        if shared_trunk is not None:
            shared_trunk.train()
            for param in shared_trunk.parameters():
                param.requires_grad = True
        if align_pooler is not None:
            align_pooler.train()
            for param in align_pooler.parameters():
                param.requires_grad = True
    else:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        if shared_trunk is not None:
            shared_trunk.eval()
            for param in shared_trunk.parameters():
                param.requires_grad = False
        if align_pooler is not None:
            align_pooler.eval()
            for param in align_pooler.parameters():
                param.requires_grad = False
    
    # Use shared batch data and target extraction functions
    def _get_batch_data(batch):
        """Extract input data from batch dictionary."""
        return get_batch_data(
            batch, task_type, device,
            training_type=None,  # Not used in probe
            apply_fft=False,  # FFT not typically used in probe
            get_batch_data_fn=get_batch_data_fn
        )
    
    def _get_target(batch):
        """Extract target from batch dictionary."""
        return get_target(batch, task_type, device, get_target_fn=get_target_fn)
    
    def _probe_mlp_single_run(seed: int) -> Optional[Dict[str, float]]:
        """Run a single MLP probe training and testing with a given seed."""
        # Set seed for this run
        set_all_seeds(seed)
        
        # Create MLP probe(s) using shared function
        probe_head = create_probe_head(
            task_type, embed_dim, num_classes, output_dim, device,
            force_use_single_sensor=(task_type == 'force' and force_single_sensor),
        )  # sliding uses classification-style head (num_classes)
        criterion = create_loss_function(task_type, class_weights=class_weights)
        # Move CE class-weight buffer onto the probe device (CE looks at .weight on .device).
        if hasattr(criterion, 'weight') and criterion.weight is not None:
            criterion.weight = criterion.weight.to(device)
        
        probe_mlp = probe_head
        
        # Setup probe optimizer. In finetune mode, fold backbone params in too
        # (single param group, same LR). In frozen mode, only the head trains.
        optim_params = list(probe_mlp.parameters())
        if finetune:
            optim_params += list(encoder.parameters())
            if shared_trunk is not None:
                optim_params += list(shared_trunk.parameters())
            if align_pooler is not None:
                optim_params += list(align_pooler.parameters())
        probe_optimizer = optim.Adam(optim_params, lr=probe_lr, weight_decay=probe_weight_decay)
        
        # Setup learning rate scheduler
        probe_lr_scheduler = None
        if lr_scheduler_config is not None:
            scheduler_type = lr_scheduler_config.get('type', 'LinearWarmupCosineDecayLR')
            scheduler_kwargs = lr_scheduler_config.get('kwargs', {})
            
            # Use probe_total_steps for scheduler
            total_steps = probe_total_steps
            
            if scheduler_type == 'LinearWarmupCosineDecayLR':
                from utils.scheduler import LinearWarmupCosineDecayLR
                
                # Get warmup steps (step-based only, no epoch conversion)
                warmup_steps = scheduler_kwargs.get('warmup_steps')
                if warmup_steps is None:
                    warmup_steps = min(20, max(1, probe_total_steps // 10))
                else:
                    warmup_steps = int(warmup_steps)
                
                probe_lr_scheduler = LinearWarmupCosineDecayLR(
                    probe_optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    warmup_start_lr=float(scheduler_kwargs.get('warmup_start_lr', 0.0)),
                    eta_min=float(scheduler_kwargs.get('eta_min', 1e-6))
                )
        
        # Training loop for probe
        best_probe_loss = float('inf')
        best_probe_model_state = None
        # In finetune mode the backbone is also training, so we snapshot its
        # state alongside the head and restore both at the best-val step.
        best_encoder_state = None
        best_trunk_state = None
        best_pooler_state = None
        probe_patience_counter = 0
        
        # Logging buffers
        train_losses_log = []
        val_losses_log = []
        if task_type in ('classification', 'sliding'):
            train_accuracies_log = []
            val_accuracies_log = []
        elif task_type == 'force':
            train_maes_log = []
            train_rmses_log = []
            val_maes_log = []
            val_rmses_log = []
        
        # Create a seed-specific DataLoader for training to ensure different data order per seed
        # Get the dataset from the original DataLoader
        train_dataset = probe_train_loader.dataset
        
        # Check if dataset is an IterableDataset (e.g., WebDataset)
        # IterableDatasets don't support samplers - they handle iteration internally
        if isinstance(train_dataset, IterableDataset):
            # For IterableDatasets (like WebDataset), we can't use a sampler
            # WebDataset handles shuffling internally, so we just use the original dataloader
            # Note: We can't easily control seed-specific ordering for IterableDatasets
            seed_specific_train_loader = probe_train_loader
        else:
            # For regular Datasets, we can use RandomSampler with a seed
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(train_dataset, generator=generator)
            # Recreate the DataLoader with the seed-specific sampler
            seed_specific_train_loader = DataLoader(
                train_dataset,
                batch_size=probe_train_loader.batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=probe_train_loader.num_workers,
                collate_fn=probe_train_loader.collate_fn,
                pin_memory=probe_train_loader.pin_memory if hasattr(probe_train_loader, 'pin_memory') else False
            )
        
        # Create infinite iterator for training data
        train_epoch_counter = [0]  # Use list so mutable in closure

        def create_train_iterator():
            """Create a new training iterator with the current seed."""
            ds = seed_specific_train_loader.dataset
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(train_epoch_counter[0])
                train_epoch_counter[0] += 1
            return iter(seed_specific_train_loader)
        
        train_iter = create_train_iterator()
        
        progress_bar = tqdm(range(probe_total_steps), desc=f"Probe Training (seed {seed})")
        for step in progress_bar:
            # Training phase - single step
            probe_mlp.train()
            if finetune:
                encoder.train()
                if shared_trunk is not None:
                    shared_trunk.train()
                if align_pooler is not None:
                    align_pooler.train()
            
            # Get next batch (cycle if exhausted)
            try:
                batch = next(train_iter)
            except StopIteration:
                # Recreate iterator with seed to ensure consistent shuffling
                train_iter = create_train_iterator()
                batch = next(train_iter)
            
            try:
                data = _get_batch_data(batch)
                targets = _get_target(batch)
            except (ValueError, KeyError) as e:
                continue
            
            # Extract features from encoder + shared_trunk using shared function.
            # In frozen mode extract_features uses torch.no_grad() (encoder is frozen,
            # but features need to be regular tensors for autograd through the probe head).
            # In finetune mode it uses nullcontext so gradients flow through the backbone.
            features = extract_features(
                encoder, shared_trunk, data, task_type, encoder_output_is_sequence,
                align_pooler=align_pooler, finetune=finetune,
            )
            
            # Debug: Check feature dimensions match expected embed_dim
            if step == 0:
                actual_feature_dim = features.shape[-1]
                probe_mlp_to_check = probe_mlp
                expected_input_dim = probe_mlp_to_check.in_dim if hasattr(probe_mlp_to_check, 'in_dim') else None
                if expected_input_dim is None:
                    # Try to get from first layer
                    first_layer = list(probe_mlp_to_check.mlp.children())[0] if hasattr(probe_mlp_to_check, 'mlp') else None
                    if first_layer is not None and isinstance(first_layer, nn.Linear):
                        expected_input_dim = first_layer.in_features
                
                if actual_feature_dim != expected_input_dim:
                    raise RuntimeError(
                        f"Feature dimension mismatch! "
                        f"Expected {expected_input_dim} (from probe MLP), but got {actual_feature_dim} (from encoder). "
                        f"Features shape: {features.shape}, embed_dim parameter: {embed_dim}"
                    )
            
            # Forward pass through MLP probe(s)
            probe_optimizer.zero_grad()
            
            outputs = probe_mlp(features)
            outputs = postprocess_outputs(outputs, task_type, output_dim)
            
            # Force: use 2-head loss (shear + normal) when output_dim in (3, 6)
            if task_type == 'force' and output_dim in (3, 6):
                loss = compute_force_dual_head_loss(outputs, targets, criterion)
            else:
                loss = compute_loss(outputs, targets, criterion, task_type)

            # Track training accuracy for classification/sliding (rolling)
            if task_type in ('classification', 'sliding'):
                with torch.no_grad():
                    train_acc_step = (outputs.argmax(dim=-1) == targets).float().mean().item() * 100
                    if len(train_accuracies_log) == 0:
                        train_accuracies_log.append(train_acc_step)
                    else:
                        train_accuracies_log.append(0.9 * train_accuracies_log[-1] + 0.1 * train_acc_step)

            # Track training metrics for force task (before backward pass to avoid gradient issues)
            if task_type == 'force':
                # Compute training MAE/RMSE for monitoring (first 3 dims only when output_dim==6)
                with torch.no_grad():
                    # Denormalize if needed for metrics (per-dimension)
                    if normalize_targets and target_mean is not None and target_std is not None:
                        mean_t = torch.tensor(target_mean, dtype=outputs.dtype, device=outputs.device)
                        std_t = torch.tensor(target_std, dtype=outputs.dtype, device=outputs.device)
                        outputs_denorm = outputs * std_t + mean_t
                        targets_denorm = targets * std_t + mean_t
                    else:
                        outputs_denorm = outputs
                        targets_denorm = targets
                    if output_dim == 6:
                        outputs_denorm = outputs_denorm[:, :3]
                        targets_denorm = targets_denorm[:, :3]
                    train_mae = torch.mean(torch.abs(outputs_denorm - targets_denorm)).item()
                    train_rmse = torch.sqrt(torch.mean((outputs_denorm - targets_denorm) ** 2)).item()
                    
                    # Store in logs (rolling average)
                    if len(train_maes_log) == 0:
                        train_maes_log.append(train_mae)
                        train_rmses_log.append(train_rmse)
                    else:
                        train_maes_log.append(0.9 * train_maes_log[-1] + 0.1 * train_mae)
                        train_rmses_log.append(0.9 * train_rmses_log[-1] + 0.1 * train_rmse)
            
            # Backward pass
            loss.backward()
            probe_optimizer.step()
            
            # Update learning rate scheduler (step per batch)
            if probe_lr_scheduler is not None:
                probe_lr_scheduler.step()
            
            # Track training metrics (rolling average over recent steps)
            train_loss = loss.item()
            if len(train_losses_log) == 0:
                train_losses_log.append(train_loss)
            else:
                # Exponential moving average
                train_losses_log.append(0.9 * train_losses_log[-1] + 0.1 * train_loss)

            # Lightweight tqdm update every 10 steps (independent of validation cadence)
            if (step + 1) % 10 == 0:
                if task_type in ('classification', 'sliding'):
                    progress_bar.set_postfix({
                        'Train Loss': f'{train_losses_log[-1]:.4f}',
                        'Train Acc': f'{train_accuracies_log[-1]:.2f}%' if train_accuracies_log else 'N/A',
                    })
                elif task_type == 'force':
                    progress_bar.set_postfix({
                        'Train Loss': f'{train_losses_log[-1]:.4f}',
                        'Train MAE': f'{train_maes_log[-1]:.4f}' if train_maes_log else 'N/A',
                        'Train RMSE': f'{train_rmses_log[-1]:.4f}' if train_rmses_log else 'N/A',
                    })

            # Validation phase (every val_interval steps)
            if (step + 1) % val_interval == 0 or (step + 1) == probe_total_steps:
                probe_mlp.eval()
                # Always set backbone to eval() at val time so dropout/LN
                # behave correctly — even in finetune mode.
                encoder.eval()
                if shared_trunk is not None:
                    shared_trunk.eval()
                if align_pooler is not None:
                    align_pooler.eval()
                val_loss = 0.0
                n_val_batches = 0
                
                # Classification metrics
                val_correct = 0
                val_total = 0
                
                if probe_val_loader is not None:
                    def _val_forward_fn(data, targets):
                        # Val forward never needs gradient — always run frozen
                        # path through extract_features (finetune=False here).
                        features = extract_features(
                            encoder, shared_trunk, data, task_type, encoder_output_is_sequence,
                            align_pooler=align_pooler, finetune=False,
                        )
                        outputs = probe_mlp(features)
                        return postprocess_outputs(outputs, task_type, output_dim)

                    val_loss, val_metrics, _, _ = run_evaluation_loop(
                        loader=probe_val_loader,
                        forward_fn=_val_forward_fn,
                        get_batch_data_fn=_get_batch_data,
                        get_target_fn=_get_target,
                        criterion=criterion,
                        task_type=task_type,
                        output_dim=output_dim,
                        force_dual_head=(task_type == 'force' and output_dim in (3, 6)),
                        target_mean=target_mean if normalize_targets else None,
                        target_std=target_std if normalize_targets else None,
                        use_amp=False,
                        skip_batch_on_error=True,
                        desc=f"Val (step {step+1})",
                    )

                    val_losses_log.append(val_loss)
                    if task_type in ('classification', 'sliding'):
                        val_acc = val_metrics['accuracy']
                        val_accuracies_log.append(val_acc)
                    elif task_type == 'force':
                        val_maes_log.append(val_metrics['mae'])
                        val_rmses_log.append(val_metrics['rmse'])
                    
                    # Track best validation loss
                    if val_loss < best_probe_loss:
                        best_probe_loss = val_loss
                        best_probe_model_state = probe_mlp.state_dict().copy()
                        if finetune:
                            # Snapshot on CPU: a GPU-side clone doubles nothing for
                            # small encoders but adds +1.2 GiB for 305M baselines
                            # (AnyTouch OOM'd at first val, 2026-08-08).
                            best_encoder_state = {k: v.detach().to('cpu', copy=True) for k, v in encoder.state_dict().items()}
                            if shared_trunk is not None:
                                best_trunk_state = {k: v.detach().to('cpu', copy=True) for k, v in shared_trunk.state_dict().items()}
                            if align_pooler is not None:
                                best_pooler_state = {k: v.detach().to('cpu', copy=True) for k, v in align_pooler.state_dict().items()}
                        probe_patience_counter = 0
                    else:
                        probe_patience_counter += 1
                    
                    # Update progress bar
                    _log_this_val = (step + 1) % (10 * val_interval) == 0 or (step + 1) == probe_total_steps
                    if task_type in ('classification', 'sliding'):
                        progress_bar.set_postfix({
                            'Train Loss': f'{train_loss:.4f}',
                            'Val Loss': f'{val_loss:.4f}',
                            'Val Acc': f'{val_acc:.2f}%',
                        })
                        if _log_this_val:
                            tqdm.write(f"  [step {step+1:>5d}] val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")
                    elif task_type == 'force':
                        progress_bar.set_postfix({
                            'Train Loss': f'{train_loss:.4f}',
                            'Train MAE': f'{train_maes_log[-1]:.4f}' if train_maes_log else 'N/A',
                            'Val Loss': f'{val_loss:.4f}',
                            'Val MAE': f'{val_maes_log[-1]:.4f}' if val_maes_log else 'N/A',
                            'Val RMSE': f'{val_rmses_log[-1]:.4f}' if val_rmses_log else 'N/A',
                        })
                        if _log_this_val:
                            tqdm.write(
                                f"  [step {step+1:>5d}] val_loss={val_loss:.4f}"
                                f"  val_mae={val_maes_log[-1]:.4f}"
                                f"  val_rmse={val_rmses_log[-1]:.4f}"
                            )
                    
                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar(f"Probe_MLP{modality_prefix}/Train_Loss_Single_Trial", train_loss, step)
                        if task_type in ('classification', 'sliding'):
                            if probe_val_loader is not None:
                                writer.add_scalar(f"Probe_MLP{modality_prefix}/Val_Loss_Single_Trial", val_loss, step)
                                writer.add_scalar(f"Probe_MLP{modality_prefix}/Val_Accuracy_Single_Trial", val_acc, step)
                        elif task_type == 'force':
                            # Log training metrics
                            if train_maes_log:
                                writer.add_scalar(f"Probe_MLP{modality_prefix}/Train_MAE_Single_Trial", train_maes_log[-1], step)
                                writer.add_scalar(f"Probe_MLP{modality_prefix}/Train_RMSE_Single_Trial", train_rmses_log[-1], step)
                            # Log validation metrics
                            if probe_val_loader is not None and val_maes_log:
                                writer.add_scalar(f"Probe_MLP{modality_prefix}/Val_Loss_Single_Trial", val_loss, step)
                                writer.add_scalar(f"Probe_MLP{modality_prefix}/Val_MAE_Single_Trial", val_maes_log[-1], step)
                                writer.add_scalar(f"Probe_MLP{modality_prefix}/Val_RMSE_Single_Trial", val_rmses_log[-1], step)
                    
                    # Early stopping
                    if probe_patience_counter >= probe_patience:
                        break
                else:
                    # No validation loader, just update progress bar
                    progress_bar.set_postfix({
                        'Train Loss': f'{train_loss:.4f}',
                    })
                    
                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar(f"Probe_MLP{modality_prefix}/Train_Loss_Single_Trial", train_loss, step)
            else:
                # Not a validation step, just log training loss
                if writer and (step + 1) % 10 == 0:  # Log every 10 steps to avoid too much logging
                    writer.add_scalar(f"Probe_MLP{modality_prefix}/Train_Loss_Single_Trial", train_loss, step)
        
        # Load best model
        if best_probe_model_state is not None:
            probe_mlp.load_state_dict(best_probe_model_state)
        if finetune:
            if best_encoder_state is not None:
                encoder.load_state_dict(best_encoder_state)
            if best_trunk_state is not None and shared_trunk is not None:
                shared_trunk.load_state_dict(best_trunk_state)
            if best_pooler_state is not None and align_pooler is not None:
                align_pooler.load_state_dict(best_pooler_state)

        # SLIP-VIZ hook: persist best probe head + (finetuned) encoder/trunk for downstream viz.
        # Triggers when env var is set. Always overwrites; intended to be used with
        # --seeds <one_seed> so the saved state is the run we care about.
        _slip_save = os.environ.get('SLIP_VIZ_SAVE_PATH')
        if _slip_save:
            os.makedirs(os.path.dirname(os.path.abspath(_slip_save)), exist_ok=True)
            _payload = {
                'probe_head': probe_mlp.state_dict(),
                'encoder': encoder.state_dict(),
                'trunk': shared_trunk.state_dict() if shared_trunk is not None else None,
                'task_type': task_type,
                'embed_dim': embed_dim,
                'num_classes': num_classes,
                'seed': int(seed),
            }
            torch.save(_payload, _slip_save)
            print(f'[SLIP_VIZ_SAVE] wrote {_slip_save}')

        # Final test evaluation — backbone always in eval() for test pass.
        probe_mlp.eval()
        encoder.eval()
        if shared_trunk is not None:
            shared_trunk.eval()
        if align_pooler is not None:
            align_pooler.eval()
        test_preds = []
        test_targets = []

        if probe_test_loader is not None:
            def _test_forward_fn(data, targets):
                features = extract_features(
                    encoder, shared_trunk, data, task_type, encoder_output_is_sequence,
                    align_pooler=align_pooler, finetune=False,
                )
                outputs = probe_mlp(features)
                return postprocess_outputs(outputs, task_type, output_dim)

            test_loss, test_metrics, test_preds, test_targets = run_evaluation_loop(
                loader=probe_test_loader,
                forward_fn=_test_forward_fn,
                get_batch_data_fn=_get_batch_data,
                get_target_fn=_get_target,
                criterion=criterion,
                task_type=task_type,
                output_dim=output_dim,
                force_dual_head=(task_type == 'force' and output_dim in (3, 6)),
                target_mean=target_mean if normalize_targets else None,
                target_std=target_std if normalize_targets else None,
                use_amp=False,
                skip_batch_on_error=True,
                desc="Test",
            )

            
            # Calculate test metrics (from run_evaluation_loop)
            if task_type in ('classification', 'sliding'):
                test_acc = test_metrics['accuracy']
                test_macro_f1 = test_metrics.get('macro_f1')
                test_per_class_f1 = test_metrics.get('per_class_f1')

                if writer:
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Loss_Single_Trial', test_loss, probe_total_steps)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Accuracy_Single_Trial', test_acc, probe_total_steps)
                    if test_macro_f1 is not None:
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_MacroF1_Single_Trial', test_macro_f1, probe_total_steps)
                    if test_per_class_f1 is not None:
                        for c, f1c in enumerate(test_per_class_f1):
                            writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_F1_class{c}_Single_Trial', f1c, probe_total_steps)

                msg = f"  Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%"
                if test_macro_f1 is not None:
                    msg += f", Test macro-F1: {test_macro_f1:.2f}%"
                if test_per_class_f1 is not None:
                    msg += f", per-class F1: {[f'{f:.1f}' for f in test_per_class_f1]}"
                print(msg)

                result = {'test_accuracy': test_acc, 'test_loss': test_loss}
                if test_macro_f1 is not None:
                    result['test_macro_f1'] = test_macro_f1
                if test_per_class_f1 is not None:
                    result['test_per_class_f1'] = list(test_per_class_f1)
                return result
            elif task_type == 'force':
                result_force = {'test_mae': 0.0, 'test_rmse': 0.0, 'test_loss': test_loss}
                if test_preds:
                    # run_evaluation_loop returns list of per-sample arrays; stack to [N, 3]
                    test_outputs_tensor = torch.tensor(np.array(test_preds))
                    test_targets_tensor = torch.tensor(np.array(test_targets))
                    # For 6D force: metrics only on first 3 dims (normal + shear), no torque. When output_dim==3, already 3D.
                    if output_dim == 6:
                        test_outputs_tensor = test_outputs_tensor[:, :3]
                        test_targets_tensor = test_targets_tensor[:, :3]
                    metrics = calculate_metrics(
                        test_outputs_tensor, test_targets_tensor, task_type,
                        normalize_targets, target_mean, target_std
                    )
                    test_mae = metrics['mae']
                    test_rmse = metrics['rmse']
                    result_force['test_mae'] = test_mae
                    result_force['test_rmse'] = test_rmse

                    # For 3/6D force: detailed normal/shear metrics
                    det_metrics = calculate_force_metrics_detailed(
                        test_outputs_tensor.numpy(), test_targets_tensor.numpy(),
                        target_mean=target_mean if normalize_targets else None,
                        target_std=target_std if normalize_targets else None,
                        output_dim=output_dim
                    )
                    if 'normal_mae' in det_metrics:
                        result_force['test_force_mae'] = det_metrics['mae']
                        result_force['test_force_rmse'] = det_metrics['rmse']
                        result_force['test_normal_mae'] = det_metrics['normal_mae']
                        result_force['test_normal_rmse'] = det_metrics['normal_rmse']
                        result_force['test_shear_mae'] = det_metrics['shear_mae']
                        result_force['test_shear_rmse'] = det_metrics['shear_rmse']
                else:
                    test_mae = 0.0
                    test_rmse = 0.0

                if writer:
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Loss_Single_Trial', test_loss, probe_total_steps)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_MAE_Single_Trial', result_force['test_mae'], probe_total_steps)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_RMSE_Single_Trial', result_force['test_rmse'], probe_total_steps)
                    if output_dim in (3, 6) and 'test_normal_mae' in result_force:
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Normal_Force_MAE_Single_Trial', result_force['test_normal_mae'], probe_total_steps)
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Normal_Force_RMSE_Single_Trial', result_force['test_normal_rmse'], probe_total_steps)
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Shear_Force_MAE_Single_Trial', result_force['test_shear_mae'], probe_total_steps)
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Shear_Force_RMSE_Single_Trial', result_force['test_shear_rmse'], probe_total_steps)
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Force_MAE_Single_Trial', result_force['test_force_mae'], probe_total_steps)
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Force_RMSE_Single_Trial', result_force['test_force_rmse'], probe_total_steps)

                print(f"  Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
                if output_dim in (3, 6) and 'test_normal_mae' in result_force:
                    print(f"  Normal: {result_force['test_normal_mae']:.4f}, Shear: {result_force['test_shear_mae']:.4f}, Force(n+s): {result_force['test_force_mae']:.4f}")

                return result_force
        
        return None
    
    # Run multiple trials
    test_results_list = []
    for trial_idx, seed in enumerate(probe_seeds):
        print(f"\nTrial {trial_idx + 1}/{len(probe_seeds)} (seed={seed})")
        
        result = _probe_mlp_single_run(seed)
        if result is not None:
            test_results_list.append(result)
            if task_type in ('classification', 'sliding'):
                mae_str = ""
                if 'test_mae' in result and result['test_mae'] is not None:
                    unit = 'm'
                    mae_str = f", Test MAE: {result['test_mae']:.4f}{unit}"
                print(f"Trial {trial_idx + 1} (seed {seed}) - Test Accuracy: {result['test_accuracy']:.2f}%{mae_str}")
            elif task_type == 'force':
                line = f"Trial {trial_idx + 1} (seed {seed}) - Test MAE: {result['test_mae']:.4f}, Test RMSE: {result['test_rmse']:.4f}"
                if 'test_normal_mae' in result:
                    line += f" | Normal: {result['test_normal_mae']:.4f}, Shear: {result['test_shear_mae']:.4f}, Force (n+s): {result['test_force_mae']:.4f}"
                print(line)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"PROBE EVALUATION SUMMARY (MLP) - {task_display}{modality_display}")
    print(f"{'='*60}")
    
    if len(test_results_list) > 0:
        if task_type in ('classification', 'sliding'):
            class_stats = aggregate_classification_results(test_results_list)
            test_accuracies = class_stats.get('test_accuracy', {}).get('values', [])
            mean_acc = class_stats.get('test_accuracy', {}).get('mean', 0.0)
            std_acc = class_stats.get('test_accuracy', {}).get('std', 0.0)

            print(f"\nAll Test Accuracies:")
            for i, (seed, acc) in enumerate(zip(probe_seeds[:len(test_accuracies)], test_accuracies)):
                print(f"  Trial {i+1} (seed {seed}): {acc:.2f}%")

            print(f"\nStatistics:")
            print(f"  Mean Test Accuracy: {mean_acc:.2f}%")
            print(f"  Std Test Accuracy: {std_acc:.2f}%")

            f1_stats = class_stats.get('test_macro_f1', {})
            mean_f1 = f1_stats.get('mean')
            std_f1 = f1_stats.get('std')
            if mean_f1 is not None:
                print(f"  Mean Test Macro-F1: {mean_f1:.2f}%  (std {std_f1:.2f}%)")
                # Per-class F1 (only printed when present from individual runs)
                pc_lines = []
                for k in sorted(class_stats.keys()):
                    if k.startswith('test_per_class_f1_'):
                        cls_id = k.rsplit('_', 1)[-1]
                        pc_lines.append(f"    class {cls_id}: F1 = {class_stats[k]['mean']:.2f}% (std {class_stats[k]['std']:.2f})")
                if pc_lines:
                    print("  Per-class F1 (mean across seeds):")
                    print('\n'.join(pc_lines))

            # Log to TensorBoard
            if writer:
                writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Accuracy_Mean', mean_acc, 0)
                writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Accuracy_Std', std_acc, 0)
                for i, acc in enumerate(test_accuracies):
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Accuracy_Trial', acc, i)
                if mean_f1 is not None:
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_MacroF1_Mean', mean_f1, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_MacroF1_Std', std_f1, 0)
                for k in class_stats:
                    if k.startswith('test_per_class_f1_'):
                        cls_id = k.rsplit('_', 1)[-1]
                        writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_F1_class{cls_id}_Mean', class_stats[k]['mean'], 0)
            result_dict = {
                'test_accuracies': [float(acc) for acc in test_accuracies],
                'test_accuracy_mean': float(mean_acc),
                'test_accuracy_std': float(std_acc),
                'seeds': probe_seeds[:len(test_accuracies)],
                'num_trials': len(test_accuracies),
                'num_classes': int(num_classes),
                'probe_total_steps': probe_total_steps,
                'eval_steps': val_interval,
                'probe_lr': probe_lr,
                'probe_type': 'mlp',
                'task_type': task_type,
                'modality': modality if modality else None
            }
            if mean_f1 is not None:
                result_dict['test_macro_f1_mean'] = float(mean_f1)
                result_dict['test_macro_f1_std'] = float(std_f1)
                for k in class_stats:
                    if k.startswith('test_per_class_f1_'):
                        result_dict[f'{k}_mean'] = float(class_stats[k]['mean'])
                        result_dict[f'{k}_std'] = float(class_stats[k]['std'])

            return result_dict
        elif task_type == 'force':
            force_stats = aggregate_force_results(test_results_list)
            test_maes = force_stats.get('test_mae', {}).get('values', [])
            test_rmses = force_stats.get('test_rmse', {}).get('values', [])
            mean_mae = force_stats.get('test_mae', {}).get('mean', 0.0)
            std_mae = force_stats.get('test_mae', {}).get('std', 0.0)
            mean_rmse = force_stats.get('test_rmse', {}).get('mean', 0.0)
            std_rmse = force_stats.get('test_rmse', {}).get('std', 0.0)

            print(f"\nAll Test MAEs:")
            for i, (seed, mae) in enumerate(zip(probe_seeds[:len(test_maes)], test_maes)):
                print(f"  Trial {i+1} (seed {seed}): {mae:.4f}")

            print(f"\nAll Test RMSEs:")
            for i, (seed, rmse) in enumerate(zip(probe_seeds[:len(test_rmses)], test_rmses)):
                print(f"  Trial {i+1} (seed {seed}): {rmse:.4f}")

            print(f"\nStatistics:")
            print(f"  Mean Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            print(f"  Mean Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

            has_6d = 'test_normal_mae' in force_stats
            force_return = {
                'test_maes': [float(mae) for mae in test_maes],
                'test_rmses': [float(rmse) for rmse in test_rmses],
                'test_mae_mean': float(mean_mae),
                'test_mae_std': float(std_mae),
                'test_rmse_mean': float(mean_rmse),
                'test_rmse_std': float(std_rmse),
                'seeds': probe_seeds[:len(test_maes)],
                'num_trials': len(test_maes),
                'output_dim': int(output_dim),
                'probe_total_steps': probe_total_steps,
                'eval_steps': val_interval,
                'probe_lr': probe_lr,
                'probe_type': 'mlp',
                'task_type': task_type,
                'modality': modality if modality else None
            }
            if has_6d:
                s_norm = force_stats['test_normal_mae']
                s_norm_r = force_stats['test_normal_rmse']
                s_shear = force_stats['test_shear_mae']
                s_shear_r = force_stats['test_shear_rmse']
                s_force = force_stats['test_force_mae']
                s_force_r = force_stats['test_force_rmse']
                mean_normal_mae, std_normal_mae = s_norm['mean'], s_norm['std']
                mean_normal_rmse, std_normal_rmse = s_norm_r['mean'], s_norm_r['std']
                mean_shear_mae, std_shear_mae = s_shear['mean'], s_shear['std']
                mean_shear_rmse, std_shear_rmse = s_shear_r['mean'], s_shear_r['std']
                mean_force_mae, std_force_mae = s_force['mean'], s_force['std']
                mean_force_rmse, std_force_rmse = s_force_r['mean'], s_force_r['std']
                print(f"\nShear force (dims 0, 1):")
                print(f"  Mean Test Shear MAE: {mean_shear_mae:.4f} ± {std_shear_mae:.4f}")
                print(f"  Mean Test Shear RMSE: {mean_shear_rmse:.4f} ± {std_shear_rmse:.4f}")
                print(f"\nNormal force (dim 2):")
                print(f"  Mean Test Normal MAE: {mean_normal_mae:.4f} ± {std_normal_mae:.4f}")
                print(f"  Mean Test Normal RMSE: {mean_normal_rmse:.4f} ± {std_normal_rmse:.4f}")
                print(f"\nForce MAE (normal+shear, dims 0-2):")
                print(f"  Mean Test Force MAE: {mean_force_mae:.4f} ± {std_force_mae:.4f}")
                print(f"  Mean Test Force RMSE: {mean_force_rmse:.4f} ± {std_force_rmse:.4f}")
                if writer:
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Normal_Force_MAE_Mean', mean_normal_mae, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Normal_Force_MAE_Std', std_normal_mae, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Normal_Force_RMSE_Mean', mean_normal_rmse, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Normal_Force_RMSE_Std', std_normal_rmse, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Shear_Force_MAE_Mean', mean_shear_mae, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Shear_Force_MAE_Std', std_shear_mae, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Shear_Force_RMSE_Mean', mean_shear_rmse, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Shear_Force_RMSE_Std', std_shear_rmse, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Force_MAE_Mean', mean_force_mae, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Force_MAE_Std', std_force_mae, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Force_RMSE_Mean', mean_force_rmse, 0)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_Force_RMSE_Std', std_force_rmse, 0)
                force_return['test_normal_mae_mean'] = float(mean_normal_mae)
                force_return['test_normal_mae_std'] = float(std_normal_mae)
                force_return['test_normal_rmse_mean'] = float(mean_normal_rmse)
                force_return['test_normal_rmse_std'] = float(std_normal_rmse)
                force_return['test_shear_mae_mean'] = float(mean_shear_mae)
                force_return['test_shear_mae_std'] = float(std_shear_mae)
                force_return['test_shear_rmse_mean'] = float(mean_shear_rmse)
                force_return['test_shear_rmse_std'] = float(std_shear_rmse)
                force_return['test_force_mae_mean'] = float(mean_force_mae)
                force_return['test_force_mae_std'] = float(std_force_mae)
                force_return['test_force_rmse_mean'] = float(mean_force_rmse)
                force_return['test_force_rmse_std'] = float(std_force_rmse)

            # Log to TensorBoard (overall)
            if writer:
                writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_MAE_Mean', mean_mae, 0)
                writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_MAE_Std', std_mae, 0)
                writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_RMSE_Mean', mean_rmse, 0)
                writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_RMSE_Std', std_rmse, 0)
                for i, (mae, rmse) in enumerate(zip(test_maes, test_rmses)):
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_MAE_Trial', mae, i)
                    writer.add_scalar(f'Probe_MLP{modality_prefix}/Test_RMSE_Trial', rmse, i)

            return force_return
    else:
        print("No successful probe runs completed.")
        return None


# Example usage
if __name__ == "__main__":
    """
    Example: How to initialize dataloaders, load checkpoint, and run probe MLP.
    
    This example demonstrates:
    1. Classification probe (object classification)
    2. Force probe (force estimation)
    """
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    from model.create_model import create_pretrain_model, create_model
    from data.create_dataloaders import create_dataloaders_from_config, get_num_classes_from_dataset
    
    parser = argparse.ArgumentParser(description='Example probe evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pretrain checkpoint file')
    parser.add_argument('--pretrain_config', type=str, default='config/model/pretrain.yaml',
                       help='Path to pretrain model config (auto-selected if --modality is provided)')
    parser.add_argument('--probe_config', type=str, default='config/algo/probe.yaml',
                       help='Path to probe config file')

    parser.add_argument('--task', type=str, choices=['classification', 'force', 'sliding'],
                       default='classification',
                       help='Task type: classification, force, or sliding bracket labels (overrides config)')
    parser.add_argument('--modality', type=str, default='gsmini',
                       choices=['9dtact', 'gsmini', 'xela', 'tac02'],
                       help='Modality to probe (overrides config, auto-selects pretrain config)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10,20,30],
                       help='List of random seeds for probe evaluation (overrides config). Example: --seeds 10 20 30')
    parser.add_argument('--probe_data_percentage', type=float, default=1.0,
                       help='Percentage of training data to use for probe evaluation (0.0 to 1.0). Overrides config if provided.')
    parser.add_argument('--finetune', action='store_true',
                       help='Fine-tune the encoder + shared trunk (+ align_pooler if present) alongside the probe head. '
                            'Default: frozen-encoder linear probe.')

    args = parser.parse_args()
    
    print("Loading probe configuration...")
    
    if not os.path.exists(args.probe_config):
        raise FileNotFoundError(f"Probe config file not found: {args.probe_config}")
    
    with open(args.probe_config, 'r') as f:
        probe_config_dict = yaml.safe_load(f)
    
    probe_config = OmegaConf.create(probe_config_dict)
    
    # Override config with command-line arguments if provided
    if args.task is not None:
        probe_config.probe.task_type = args.task
    if args.modality is not None:
        probe_config.probe.modality = args.modality
    if args.seeds is not None:
        probe_config.probe.probe_seeds = args.seeds
    if args.probe_data_percentage is not None:
        probe_config.probe.probe_data_percentage = args.probe_data_percentage
    
    task_type = probe_config.probe.task_type
    modality = probe_config.probe.modality
    
    # Modality for model creation and data loading (same for force: gsmini)
    original_modality = modality
    data_modality = modality

    # Validate modality is set
    if original_modality is None:
        raise ValueError("Modality must be specified either in probe config or via --modality argument")
    
    # Automatic pretrain config selection based on modality
    
    # Mapping of modalities to model configs
    # Image modalities (9dtact, gsmini) use vision transformer (vit.yaml)
    # Tactile modalities: xela uses standard transformer, tac02 uses Conv2D transformer
    model_configs = {
        '9dtact': 'config/model/vit.yaml',      # Image modality -> vision transformer
        'gsmini': 'config/model/vit.yaml',      # Image modality (force uses gsmini; other modalities later)
        'xela': 'config/model/taxel_tf.yaml',   # Tactile modality -> standard tactile transformer
        'tac02': 'config/model/taxel_tf.yaml'    # Tactile modality -> Conv2D tactile transformer (for tacniq data)
    }
    
    # Use provided pretrain config for all modalities, or auto-select if not provided
    if args.pretrain_config is not None:
        pass  # use as-is
    else:
        if original_modality in model_configs:
            args.pretrain_config = model_configs[original_modality]
        else:
            raise ValueError(f"Unknown modality: {original_modality}. Cannot auto-select pretrain config.")

    eval_steps = getattr(probe_config.probe, 'eval_steps', None)
    probe_val_interval = getattr(probe_config.probe, 'probe_val_interval', 50)
    val_interval = eval_steps if eval_steps is not None else probe_val_interval

    run_name = probe_config.logging.get('run_name', None)
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{task_type}_{modality}_{timestamp}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Probe: task={task_type}, modality={original_modality}, seeds={list(probe_config.probe.probe_seeds)}, device={device}")
    
    print(f"Loading checkpoint: {args.checkpoint}")
    
    checkpoint_loader = ProbeCheckpointLoader()
    checkpoint = checkpoint_loader.load_checkpoint(args.checkpoint, device=device)
    
    # Get config paths from checkpoint
    config_paths = checkpoint_loader.get_config_paths(checkpoint)
    pretrain_config_path = args.pretrain_config or config_paths['pretrain_config_path']
    
    # Load pretrain config
    with open(pretrain_config_path, 'r') as f:
        pretrain_config = yaml.safe_load(f)
    
    config = OmegaConf.create(pretrain_config)
    
    print("Creating model and loading weights...")
    
    pretrain_config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Check checkpoint type: SSL, pretrain, or alignment
    # SSL checkpoints may store encoder_state_dict at top level (old format)
    # or nested inside model_state_dict (new base_trainer format)
    _nested = isinstance(checkpoint.get('model_state_dict'), dict) and \
              'encoder_state_dict' in checkpoint.get('model_state_dict', {})
    is_ssl_checkpoint = 'encoder_state_dict' in checkpoint or _nested
    is_alignment_checkpoint = 'pretrain_model_state_dict' in checkpoint
    is_pretrain_checkpoint = 'model_state_dict' in checkpoint and not _nested
    
    if is_ssl_checkpoint:
        ssl_config = checkpoint.get('config', {})
        if ssl_config and 'model' in ssl_config:
            model_config_dict = OmegaConf.to_container(ssl_config['model'], resolve=True) if hasattr(ssl_config['model'], '__dict__') else ssl_config['model']
        else:
            model_config_dict = pretrain_config_dict
        
        # Ensure architecture is set to encoder only
        model_config_dict = model_config_dict.copy()
        model_config_dict['architecture'] = ['encoder']
        
        # Determine if this is a vision or tactile model based on original modality
        vision_modalities = ['9dtact', 'gsmini']
        tactile_modalities = ['xela', 'tac02', 'tacniq']
        
        model_modality = original_modality
        
        if model_modality in vision_modalities:
                if model_config_dict.get('type') not in ['vit', 'vision']:
                    model_config_dict['type'] = 'vit'
                model_config_dict.setdefault('input_size', 224)
                model_config_dict.setdefault('patch_size', 16)
                
                # Try to infer num_frames from checkpoint weights if available
                num_frames_from_checkpoint = None
                if 'encoder_state_dict' in checkpoint:
                    encoder_state_dict = checkpoint['encoder_state_dict']
                    
                    # Method 1: Check pos_embed first (most reliable for actual num_frames)
                    # pos_embed accounts for all tokens including CLS token, so it's more accurate
                    pos_embed_key = None
                    num_frames_from_pos_embed = None
                    for key in ['vit.pos_embed', '_orig_mod.vit.pos_embed', 'pos_embed']:
                        if key in encoder_state_dict:
                            pos_embed_key = key
                            break
                    
                    if pos_embed_key:
                        pos_embed_shape = encoder_state_dict[pos_embed_key].shape
                        num_tokens = pos_embed_shape[1]
                        input_size = model_config_dict.get('input_size', 224)
                        patch_size = model_config_dict.get('patch_size', 16)
                        patches_per_frame = (input_size // patch_size) ** 2
                        num_frames_from_pos_embed = (num_tokens - 1) // patches_per_frame
                    
                    # Method 2: Check patch embedding weights as secondary check
                    # Patch embedding weight shape: [embed_dim, channels, tubelet_size, patch_h, patch_w] for 3D
                    # or [embed_dim, channels, patch_h, patch_w] for 2D
                    patch_embed_keys = ['vit.patch_embed.proj.weight', '_orig_mod.vit.patch_embed.proj.weight', 'patch_embed.proj.weight']
                    tubelet_size_from_checkpoint = None
                    for key in patch_embed_keys:
                        if key in encoder_state_dict:
                            patch_embed_shape = encoder_state_dict[key].shape
                            # Shape is [embed_dim, channels, tubelet_size, patch_h, patch_w] for 3D
                            if len(patch_embed_shape) == 5:
                                tubelet_size_from_checkpoint = patch_embed_shape[2]
                                model_config_dict['tubelet_size'] = tubelet_size_from_checkpoint
                            elif len(patch_embed_shape) == 4:
                                model_config_dict.setdefault('tubelet_size', 1)
                            break
                    
                    # Use pos_embed result (most reliable) if available and valid
                    if num_frames_from_pos_embed is not None and num_frames_from_pos_embed > 0:
                        num_frames_from_checkpoint = num_frames_from_pos_embed
                    # If pos_embed calculation gives 0 or invalid result, but we have 3D patch embedding,
                    # use tubelet_size as a hint (for classification, num_frames typically equals tubelet_size)
                    elif tubelet_size_from_checkpoint is not None and tubelet_size_from_checkpoint > 1:
                        if task_type == 'classification':
                            num_frames_from_checkpoint = tubelet_size_from_checkpoint
                
                # Set num_frames based on checkpoint inference or modality defaults
                if 'num_frames' not in model_config_dict or model_config_dict.get('num_frames') is None:
                    if task_type in ('force', 'sliding') and model_modality == 'gsmini':
                        num_frames = 2
                    elif num_frames_from_checkpoint is not None:
                        num_frames = num_frames_from_checkpoint
                    elif model_modality == '9dtact':
                        num_frames = model_config_dict.get('9dtact_num_frames', 2)
                    elif model_modality == 'gsmini':
                        num_frames = model_config_dict.get('gsmini_num_frames', 2)
                    else:
                        num_frames = 1
                        print(f"Warning: Unknown vision modality '{model_modality}', using default num_frames=1")
                    model_config_dict['num_frames'] = num_frames
                else:
                    if task_type in ('force', 'sliding') and model_modality == 'gsmini':
                        model_config_dict['num_frames'] = 2
                    elif num_frames_from_checkpoint is not None:
                        config_num_frames = model_config_dict.get('num_frames')
                        if config_num_frames != num_frames_from_checkpoint:
                            print(f"Warning: num_frames mismatch — config={config_num_frames}, checkpoint={num_frames_from_checkpoint}. Using checkpoint value.")
                            model_config_dict['num_frames'] = num_frames_from_checkpoint

                model_config_dict.setdefault('tubelet_size', 1)
        
        elif model_modality in tactile_modalities:
                if model_config_dict.get('type') == 'transformer' or 'tactile_dim' not in model_config_dict or model_config_dict.get('tactile_dim') is None:
                    if model_modality == 'xela':
                        tactile_dim = model_config_dict.get('xela_dim', 72)
                    elif model_modality in ['tac02', 'tacniq']:
                        tactile_dim = model_config_dict.get('tac_dim', 66)
                    else:
                        tactile_dim = model_config_dict.get('xela_dim', 72)
                        print(f"Warning: Unknown modality '{model_modality}', using default tactile_dim={tactile_dim}")
                    model_config_dict['tactile_dim'] = tactile_dim
        else:
            print(f"Warning: Unknown modality '{model_modality}', attempting to create model with existing config")
        
        # Create encoder
        encoder = create_model(model_config_dict).to(device)
        
        # Handle lazy initialization of patch_proj in ViTEncoder if needed
        # Support both flat (old) and nested-inside-model_state_dict (new base_trainer) formats
        if 'encoder_state_dict' in checkpoint:
            encoder_state_dict = checkpoint['encoder_state_dict']
        else:
            encoder_state_dict = checkpoint['model_state_dict']['encoder_state_dict']
        
        # Check for patch_proj in encoder state dict (for vision encoders)
        patch_proj_key = 'patch_proj.weight'
        if '_orig_mod.patch_proj.weight' in encoder_state_dict:
            patch_proj_key = '_orig_mod.patch_proj.weight'
        
        if patch_proj_key in encoder_state_dict and hasattr(encoder, 'patch_proj') and encoder.patch_proj is None:
            # Extract patch_dim from checkpoint weights
            patch_dim = encoder_state_dict[patch_proj_key].shape[1]
            embed_dim = encoder_state_dict[patch_proj_key].shape[0]
            # Initialize patch_proj with the correct dimensions
            encoder.patch_proj = nn.Linear(patch_dim, embed_dim).to(device)
            encoder.patch_dim = patch_dim
            print(f"Initialized patch_proj from checkpoint: patch_dim={patch_dim}, embed_dim={embed_dim}")
        
        # Load encoder weights
        encoder_state_dict = adjust_state_dict_keys(encoder_state_dict, encoder)
        encoder.load_state_dict(encoder_state_dict, strict=False)
        print("Encoder weights loaded successfully")
        
        # SSL models don't have shared trunk
        shared_trunk = None
        
        # Set encoder to eval mode
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Create a minimal wrapper to match the interface expected by probe function
        # Use original_modality for model operations
        class SSLModelWrapper(nn.Module):
            def __init__(self, encoder, modality):
                super().__init__()
                self.encoders = nn.ModuleDict({modality: encoder})
                self.modalities = [modality]
                self.shared_trunk = None
            
            def get_encoder(self, modality):
                return self.encoders[modality]
        
        model = SSLModelWrapper(encoder, original_modality)
        
        # For SSL checkpoints, encoder is already loaded, so skip the model loading section
        # Get encoder and shared trunk from the wrapper
        encoder = model.get_encoder(original_modality)
        shared_trunk = model.shared_trunk
    
    else:
        # Pretrain or alignment checkpoint: load full model
        model = create_pretrain_model(pretrain_config_dict).to(device)
        
        # Load model weights - handle both pretrain and alignment checkpoints
        # Pretrain checkpoints use 'model_state_dict', alignment checkpoints use 'pretrain_model_state_dict'
        model_state_dict = checkpoint.get('model_state_dict') or checkpoint.get('pretrain_model_state_dict')
        if model_state_dict is None:
            available_keys = [k for k in checkpoint.keys() if 'state_dict' in k]
            raise ValueError(f"Checkpoint does not contain 'model_state_dict' or 'pretrain_model_state_dict'. "
                            f"Available keys with 'state_dict': {available_keys}")
        
        # Adjust state dict keys to handle compiled/uncompiled models
        # This removes '_orig_mod.' prefix if model is not compiled but checkpoint is, or vice versa
        model_state_dict = adjust_state_dict_keys(model_state_dict, model)
        model.load_state_dict(model_state_dict, strict=False)
        print("Model weights loaded successfully")
        
        # Get encoder and shared trunk
        # Use original_modality for model operations
        encoder = model.get_encoder(original_modality)
        shared_trunk = model.shared_trunk
    
    # Set to eval mode
    encoder.eval()
    if shared_trunk is not None:
        shared_trunk.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    if shared_trunk is not None:
        for param in shared_trunk.parameters():
            param.requires_grad = False

    # ---------------------------------------------------------------- #
    # Align-token path for the FORCE probe only.
    # When the checkpoint was produced by align_token_mode=true joint
    # pretraining AND we're running the force task, build the modality's
    # frozen AlignPooler and load its weights. extract_features() will
    # use it in place of mean-pooling — the learnable align token attended
    # over the full unmasked encoder+trunk sequence during pretraining,
    # so it preserves fine-grained features the force probe needs.
    # Classification and sliding probes deliberately keep the mean-pool
    # path (unchanged behavior).
    align_pooler = None
    pooler_states = checkpoint.get('align_poolers_state_dict', {}) or {}
    if task_type == 'force' and pooler_states and original_modality in pooler_states:
        from model.predictor import AlignPooler
        align_cfg = checkpoint.get('align_token_cfg', {}) or {}
        embed_dim_for_pooler = None
        # Pull the trunk's hidden dim from the saved config if available;
        # otherwise infer from the saved pooler's align_token shape.
        saved_sd = pooler_states[original_modality]
        if 'align_token' in saved_sd:
            embed_dim_for_pooler = saved_sd['align_token'].shape[-1]
        if embed_dim_for_pooler is None:
            embed_dim_for_pooler = 192  # v4 default
        align_pooler = AlignPooler(
            embed_dim=embed_dim_for_pooler,
            depth=int(align_cfg.get('pooler_depth', 2)),
            num_heads=int(align_cfg.get('pooler_num_heads', 3)),
            mlp_ratio=float(align_cfg.get('pooler_mlp_ratio', 4.0)),
        ).to(device)
        align_pooler.load_state_dict(saved_sd, strict=False)
        align_pooler.eval()
        for p in align_pooler.parameters():
            p.requires_grad = False
        print(f"  Align-token mode: loaded AlignPooler[{original_modality}] for force probe.")

    print("Initializing dataloaders...")
    
    if task_type == 'classification':
        # Classification: Use existing dataloaders
        print(f"Loading classification dataloaders for {modality}...")
        
        # Get data config path from probe config - automatically select based on modality
        classification_config = probe_config.probe.classification
        
        # Try to get from data_configs mapping first
        if hasattr(classification_config, 'data_configs') and classification_config.data_configs is not None:
            data_configs_dict = OmegaConf.to_container(classification_config.data_configs, resolve=True)
            if modality in data_configs_dict:
                data_config_path = data_configs_dict[modality]
                print(f"Using data config for {modality}: {data_config_path}")
            else:
                # Fallback to default data_config
                data_config_path = classification_config.get('data_config', None)
                if data_config_path:
                    print(f"Warning: Modality '{modality}' not found in data_configs, using fallback: {data_config_path}")
                else:
                    raise ValueError(f"Modality '{modality}' not found in data_configs and no fallback data_config specified")
        else:
            # Fallback to single data_config field (backward compatibility)
            data_config_path = classification_config.get('data_config', None)
            if data_config_path is None:
                raise ValueError(f"No data_config specified for classification task. Please set classification.data_configs.{modality} or classification.data_config")
            print(f"Using data config (fallback): {data_config_path}")
        
        if not os.path.exists(data_config_path):
            raise FileNotFoundError(f"Data config file not found: {data_config_path}")
        
        # Load data config
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Merge with training config
        merged_data_config = data_config.copy()
        if 'data' not in merged_data_config:
            merged_data_config['data'] = {}
        
        # Set training parameters from probe config
        merged_data_config['data']['batch_size'] = probe_config.probe.batch_size
        merged_data_config['data']['num_workers'] = probe_config.probe.num_workers
        merged_data_config['data']['dataset_split_type'] = 'supervised'  # Use supervised split for probe evaluation
        merged_data_config['data']['train_data_percentage'] = probe_config.probe.probe_data_percentage  # Use probe_data_percentage for classification
        
        dataloader_config = OmegaConf.create(merged_data_config['data'])
        
        # Create dataloaders
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            create_dataloaders_from_config(dataloader_config)
        
        probe_train_loader = train_loader
        probe_val_loader = val_loader
        probe_test_loader = test_loader
        
        # Get number of classes
        num_classes = get_num_classes_from_dataset(train_dataset)
        print(f"Number of classes: {num_classes}")
        
        # Get embedding dimension
        # Compute from sample batch
        sample_batch = next(iter(train_loader))
        if 'tactile' in sample_batch:
            x_sample = sample_batch['tactile'][:1].to(device)
        elif 'images' in sample_batch:
            x_sample = sample_batch['images'][:1].to(device)
        else:
            raise ValueError("Cannot determine input type from batch")
        
        with torch.no_grad():
            encoder_out = encoder(x_sample)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            
            if len(encoder_out.shape) == 2:
                encoder_out = encoder_out.unsqueeze(1)
            
            # Pass through shared trunk if provided (skip for SSL models)
            if shared_trunk is not None:
                trunk_out = shared_trunk(encoder_out)
                if isinstance(trunk_out, tuple):
                    trunk_out = trunk_out[0]
                
                if len(trunk_out.shape) == 3:
                    embed_dim = trunk_out.shape[-1]
                    encoder_output_is_sequence = True
                else:
                    embed_dim = trunk_out.shape[-1]
                    encoder_output_is_sequence = False
            else:
                # No shared trunk (e.g., SSL models): use encoder output directly
                if len(encoder_out.shape) == 3:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = True
                else:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = False
        
        print(f"Embedding dimension: {embed_dim}")
        print(f"Encoder output is sequence: {encoder_output_is_sequence}")
        
        # Setup TensorBoard writer
        log_dir = probe_config.logging.log_dir
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_dir, run_name))
        
        
        def get_batch_data_fn(batch):
            if 'tactile' in batch:
                return batch['tactile'].to(device)
            elif 'images' in batch:
                return batch['images'].to(device)
            else:
                raise ValueError("Batch must contain 'tactile' or 'images' key")
        
        # Get scheduler config if available
        lr_scheduler_config = None
        if hasattr(probe_config.probe, 'lr_scheduler') and probe_config.probe.lr_scheduler is not None:
            lr_scheduler_config = OmegaConf.to_container(probe_config.probe.lr_scheduler, resolve=True)
        
        probe_results = run_probe_mlp(
            encoder=encoder,
            shared_trunk=shared_trunk,
            device=device,
            probe_train_loader=probe_train_loader,
            probe_val_loader=probe_val_loader,
            probe_test_loader=probe_test_loader,
            task_type='classification',
            num_classes=num_classes,
            embed_dim=embed_dim,
            encoder_output_is_sequence=encoder_output_is_sequence,
            probe_total_steps=probe_config.probe.probe_total_steps,
            eval_steps=getattr(probe_config.probe, 'eval_steps', None),
            probe_val_interval=getattr(probe_config.probe, 'probe_val_interval', 50),
            probe_lr=float(probe_config.probe.probe_lr),
            probe_seeds=list(probe_config.probe.probe_seeds),
            writer=writer,
            modality=modality,
            get_batch_data_fn=get_batch_data_fn,
            lr_scheduler_config=lr_scheduler_config,
            probe_patience=int(probe_config.probe.get('probe_patience', 5)),
            probe_weight_decay=float(probe_config.probe.get('probe_weight_decay', 1e-4)),
            finetune=args.finetune,
        )

        if probe_results:
            print(f"\nClassification Probe Results:")
            print(f"  Mean Test Accuracy: {probe_results['test_accuracy_mean']:.2f}%")
            print(f"  Std Test Accuracy: {probe_results['test_accuracy_std']:.2f}%")
        
        writer.close()
        
    elif task_type == 'force':
        _supported_force = ('gsmini', '9dtact', 'xela', 'tac02')
        if modality not in _supported_force:
            raise ValueError(
                f"Force probe supports: {_supported_force}. Got: {modality}"
            )
        print(f"Loading force dataloaders for {modality}...")
        
        # Get data roots from probe config
        data_roots_dict = probe_config.probe.force.data_roots
        data_roots = OmegaConf.to_container(data_roots_dict, resolve=True)
        data_roots = dict(data_roots)
        # Fill in default data roots for known modalities (released force split)
        from data.dataset_paths import force_processed_root
        for _mod in ('gsmini', '9dtact', 'xela', 'tac02'):
            if _mod not in data_roots:
                data_roots[_mod] = force_processed_root(_mod)
        
        # Create a minimal config with training defaults from probe config
        temp_config = OmegaConf.create({
            'training': {
                'batch_size': probe_config.probe.batch_size,
                'num_workers': probe_config.probe.num_workers,
                'probe_data_percentage': probe_config.probe.probe_data_percentage,
            }
        })
        # Merge with existing config
        config = OmegaConf.merge(config, temp_config)
        
        dataloader_manager = ProbeDataloaderManager(
            task_type='force',
            modality=modality,
            config=config
        )
        
        # Load force dataloaders
        config_dir = probe_config.probe.force.config_dir
        # Get data split seed from config or use first probe_seed as default
        data_split_seed = probe_config.probe.force.get('data_split_seed', None)
        if data_split_seed is None:
            # Use first probe_seed for data splitting if not specified
            data_split_seed = probe_config.probe.probe_seeds[0] if probe_config.probe.probe_seeds else 42
        if probe_config.probe.force.get('mode_filter'):
            print(f"Mode filter: {probe_config.probe.force.mode_filter} only")
        print(f"Using data split seed: {data_split_seed}")
        dataloader_manager.load_force_dataloaders(
            data_roots=data_roots,
            config_dir=config_dir,
            random_seed=data_split_seed,
            mode_filter=probe_config.probe.force.get('mode_filter'),
            compute_friction_mu=probe_config.probe.force.get('compute_friction_mu'),
            friction_mu_eps=float(
                1e-3
                if probe_config.probe.force.get('friction_mu_eps') is None
                else probe_config.probe.force.get('friction_mu_eps')
            ),
            labeled_data_root=probe_config.probe.force.get('labeled_data_root'),
            load_sliding_labels=probe_config.probe.force.get('load_sliding_labels'),
            strict_labeled=bool(probe_config.probe.force.get('strict_labeled', False)),
        )
        
        probe_train_loader = dataloader_manager.train_loader
        probe_val_loader = dataloader_manager.val_loader
        probe_test_loader = dataloader_manager.test_loader
        
        # Get embedding dimension
        sample_batch = next(iter(probe_train_loader))
        _is_taxel_force = modality in TAXEL_FORCE_DIMS
        if _is_taxel_force:
            if 'tactile' not in sample_batch:
                raise ValueError(f"Taxel force probe requires 'tactile' key in batch. Got: {list(sample_batch.keys())}")
            x_sample = sample_batch['tactile'][:1].to(device)
        else:
            if 'tactile_img' not in sample_batch:
                raise ValueError(f"Image force probe requires 'tactile_img' key. Got: {list(sample_batch.keys())}")
            x_sample = sample_batch['tactile_img'][:1].to(device)

        with torch.no_grad():
            encoder_out = encoder(x_sample)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]

            if len(encoder_out.shape) == 2:
                encoder_out = encoder_out.unsqueeze(1)

            # Pass through shared trunk if provided (skip for SSL models)
            if shared_trunk is not None:
                trunk_out = shared_trunk(encoder_out)
                if isinstance(trunk_out, tuple):
                    trunk_out = trunk_out[0]

                if len(trunk_out.shape) == 3:
                    embed_dim = trunk_out.shape[-1]
                    encoder_output_is_sequence = True
                else:
                    embed_dim = trunk_out.shape[-1]
                    encoder_output_is_sequence = False
            else:
                # No shared trunk (e.g., SSL models): use encoder output directly
                if len(encoder_out.shape) == 3:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = True
                else:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = False

        print(f"Embedding dimension: {embed_dim}")
        print(f"Encoder output is sequence: {encoder_output_is_sequence}")

        # Load per-dimension force stats for MAE/RMSE denormalization (first 3 dims: shear_x, shear_y, normal)
        config_dir = probe_config.probe.force.config_dir
        if _is_taxel_force:
            force_mean, force_std = load_taxel_force_stats_per_dim(modality, config_dir=config_dir, dims=3)
        else:
            force_mean, force_std = load_force_stats_per_dim_4probe(modality, config_dir=config_dir, dims=3)
        print(f"Force stats (per-dim for denorm): Mean={force_mean}, Std={force_std}")

        # Setup TensorBoard writer
        log_dir = probe_config.logging.log_dir
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_dir, run_name))


        # 4probe: single sensor (tactile only), predict first 3 force dims (normal + shear), dataloader normalizes
        output_dim = 3
        normalize_targets = True  # Dataloader normalizes; denormalize for MAE/RMSE in raw units (N)
        _tactile_key = 'tactile' if _is_taxel_force else 'tactile_img'
        def get_batch_data_fn(batch):
            return batch[_tactile_key].to(device, non_blocking=True)
        def get_target_fn(batch):
            return batch['6d_force'][:, :3].to(device, non_blocking=True)
        
        # Get scheduler config if available
        lr_scheduler_config = None
        if hasattr(probe_config.probe, 'lr_scheduler') and probe_config.probe.lr_scheduler is not None:
            lr_scheduler_config = OmegaConf.to_container(probe_config.probe.lr_scheduler, resolve=True)
        
        probe_results = run_probe_mlp(
            encoder=encoder,
            shared_trunk=shared_trunk,
            device=device,
            probe_train_loader=probe_train_loader,
            probe_val_loader=probe_val_loader,
            probe_test_loader=probe_test_loader,
            task_type='force',
            output_dim=output_dim,
            embed_dim=embed_dim,
            encoder_output_is_sequence=encoder_output_is_sequence,
            probe_total_steps=probe_config.probe.probe_total_steps,
            eval_steps=getattr(probe_config.probe, 'eval_steps', None),
            probe_val_interval=getattr(probe_config.probe, 'probe_val_interval', 50),
            probe_lr=float(probe_config.probe.probe_lr),
            probe_seeds=list(probe_config.probe.probe_seeds),
            writer=writer,
            modality=modality,
            get_batch_data_fn=get_batch_data_fn,
            get_target_fn=get_target_fn,
            force_single_sensor=True,
            normalize_targets=normalize_targets,
            target_mean=force_mean,
            target_std=force_std,
            lr_scheduler_config=lr_scheduler_config,
            probe_patience=int(probe_config.probe.get('probe_patience', 5)),
            probe_weight_decay=float(probe_config.probe.get('probe_weight_decay', 1e-4)),
            align_pooler=align_pooler,
            finetune=args.finetune,
        )

        if probe_results:
            print(f"\nForce Probe Results:")
            print(f"  Mean Test MAE: {probe_results['test_mae_mean']:.4f} ± {probe_results['test_mae_std']:.4f}")
            print(f"  Mean Test RMSE: {probe_results['test_rmse_mean']:.4f} ± {probe_results['test_rmse_std']:.4f}")
            if 'test_normal_mae_mean' in probe_results:
                print(f"  Shear force (dims 0,1)   - MAE: {probe_results['test_shear_mae_mean']:.4f} ± {probe_results['test_shear_mae_std']:.4f}, RMSE: {probe_results['test_shear_rmse_mean']:.4f} ± {probe_results['test_shear_rmse_std']:.4f}")
                print(f"  Normal force (dim 2)     - MAE: {probe_results['test_normal_mae_mean']:.4f} ± {probe_results['test_normal_mae_std']:.4f}, RMSE: {probe_results['test_normal_rmse_mean']:.4f} ± {probe_results['test_normal_rmse_std']:.4f}")
                print(f"  Force MAE (normal+shear) - MAE: {probe_results['test_force_mae_mean']:.4f} ± {probe_results['test_force_mae_std']:.4f}, RMSE: {probe_results['test_force_rmse_mean']:.4f} ± {probe_results['test_force_rmse_std']:.4f}")
        
        writer.close()
    
    elif task_type == 'sliding':
        # Image modalities (gsmini, 9dtact) emit `tactile_img`; taxel modalities
        # (xela, tac02) emit `tactile`. Both dataloaders now emit `sliding_label`.
        if modality not in ('gsmini', '9dtact', *TAXEL_FORCE_DIMS):
            raise ValueError(
                f"Sliding probe supports gsmini, 9dtact, and taxel modalities {tuple(TAXEL_FORCE_DIMS)}; "
                f"got '{modality}'."
            )
        is_taxel_sliding = modality in TAXEL_FORCE_DIMS
        tactile_key = 'tactile' if is_taxel_sliding else 'tactile_img'
        print(f"Loading sliding (bracket label) dataloaders for {modality}...")
        # Sliding uses the released slip split; probe.force.sliding_data_roots
        # (optional) overrides, probe.force.data_roots points at the force split
        # and is intentionally NOT reused here.
        sliding_roots_cfg = probe_config.probe.force.get('sliding_data_roots')
        data_roots = dict(OmegaConf.to_container(sliding_roots_cfg, resolve=True)) \
            if sliding_roots_cfg else {}
        if modality not in data_roots:
            from data.dataset_paths import slip_processed_root
            data_roots[modality] = slip_processed_root(modality)
        temp_config = OmegaConf.create({
            'training': {
                'batch_size': probe_config.probe.batch_size,
                'num_workers': probe_config.probe.num_workers,
                'probe_data_percentage': probe_config.probe.probe_data_percentage,
            }
        })
        config = OmegaConf.merge(config, temp_config)
        dataloader_manager = ProbeDataloaderManager(
            task_type='force',
            modality=modality,
            config=config
        )
        config_dir = probe_config.probe.force.config_dir
        data_split_seed = probe_config.probe.force.get('data_split_seed', None)
        if data_split_seed is None:
            data_split_seed = probe_config.probe.probe_seeds[0] if probe_config.probe.probe_seeds else 42
        mf = probe_config.probe.force.get('mode_filter')
        if mf != 'sliding':
            # Sliding task can only train on episodes that have a matching
            # `.labeled.npz`; static episodes would surface as all -1 labels.
            mf = 'sliding'
        print(f"Mode filter: {mf} (sliding task)")
        print(f"Using data split seed: {data_split_seed}")
        lsl = probe_config.probe.force.get('load_sliding_labels')
        if lsl is None:
            lsl = True
        dataloader_manager.load_force_dataloaders(
            data_roots=data_roots,
            config_dir=config_dir,
            random_seed=data_split_seed,
            mode_filter=mf,
            compute_friction_mu=probe_config.probe.force.get('compute_friction_mu'),
            friction_mu_eps=float(
                1e-3
                if probe_config.probe.force.get('friction_mu_eps') is None
                else probe_config.probe.force.get('friction_mu_eps')
            ),
            labeled_data_root=probe_config.probe.force.get('labeled_data_root'),
            load_sliding_labels=lsl,
            strict_labeled=bool(probe_config.probe.force.get('strict_labeled', False)),
        )
        probe_train_loader = dataloader_manager.train_loader
        probe_val_loader = dataloader_manager.val_loader
        probe_test_loader = dataloader_manager.test_loader
        sample_batch = next(iter(probe_train_loader))
        if tactile_key not in sample_batch or 'sliding_label' not in sample_batch:
            raise ValueError(
                f"Sliding probe requires batches with '{tactile_key}' and 'sliding_label'. "
                f"Got: {list(sample_batch.keys())}"
            )
        x_sample = sample_batch[tactile_key][:1].to(device)
        with torch.no_grad():
            encoder_out = encoder(x_sample)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            if len(encoder_out.shape) == 2:
                encoder_out = encoder_out.unsqueeze(1)
            if shared_trunk is not None:
                trunk_out = shared_trunk(encoder_out)
                if isinstance(trunk_out, tuple):
                    trunk_out = trunk_out[0]
                if len(trunk_out.shape) == 3:
                    embed_dim = trunk_out.shape[-1]
                    encoder_output_is_sequence = True
                else:
                    embed_dim = trunk_out.shape[-1]
                    encoder_output_is_sequence = False
            else:
                if len(encoder_out.shape) == 3:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = True
                else:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = False
        print(f"Embedding dimension: {embed_dim}")
        num_classes = int(probe_config.probe.force.get('num_sliding_classes', 3))
        print(f"Sliding classes: {num_classes}")

        # Compute inverse-frequency class weights from cached labels (fix imbalance:
        # bracket labels are ~88% gross / 11% static / 2% incipient, so unweighted CE
        # collapses to "always predict gross.")
        from utils.sliding_labels import compute_sliding_class_weights
        labeled_root_cfg = probe_config.probe.force.get('labeled_data_root')
        labeled_root = labeled_root_cfg or os.path.join(
            os.path.dirname(data_roots[modality]) if data_roots.get(modality) else f'data/{modality}_force_4probe_50each',
            'sliding_labeled',
        )
        try:
            class_weights = compute_sliding_class_weights(labeled_root, num_classes=num_classes)
            print(f"Sliding class weights (from {labeled_root}): {class_weights.tolist()}")
        except Exception as e:
            print(f"Warning: could not compute class weights ({e}); using uniform weights")
            class_weights = None

        log_dir = probe_config.logging.log_dir
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_dir, run_name))
        lr_scheduler_config = None
        if hasattr(probe_config.probe, 'lr_scheduler') and probe_config.probe.lr_scheduler is not None:
            lr_scheduler_config = OmegaConf.to_container(probe_config.probe.lr_scheduler, resolve=True)
        probe_results = run_probe_mlp(
            encoder=encoder,
            shared_trunk=shared_trunk,
            device=device,
            probe_train_loader=probe_train_loader,
            probe_val_loader=probe_val_loader,
            probe_test_loader=probe_test_loader,
            task_type='sliding',
            num_classes=num_classes,
            embed_dim=embed_dim,
            encoder_output_is_sequence=encoder_output_is_sequence,
            probe_total_steps=probe_config.probe.probe_total_steps,
            eval_steps=getattr(probe_config.probe, 'eval_steps', None),
            probe_val_interval=getattr(probe_config.probe, 'probe_val_interval', 50),
            probe_lr=float(probe_config.probe.probe_lr),
            probe_seeds=list(probe_config.probe.probe_seeds),
            writer=writer,
            modality=modality,
            lr_scheduler_config=lr_scheduler_config,
            class_weights=class_weights,
            probe_patience=int(probe_config.probe.get('probe_patience', 5)),
            probe_weight_decay=float(probe_config.probe.get('probe_weight_decay', 1e-4)),
            finetune=args.finetune,
        )
        if probe_results:
            print(f"\nSliding Probe Results:")
            print(f"  Mean Test Accuracy: {probe_results['test_accuracy_mean']:.2f}%")
            print(f"  Std Test Accuracy: {probe_results['test_accuracy_std']:.2f}%")
            if 'test_macro_f1_mean' in probe_results:
                print(f"  Mean Test Macro-F1: {probe_results['test_macro_f1_mean']:.2f}%  (std {probe_results['test_macro_f1_std']:.2f}%)")
        writer.close()

    else:
        raise ValueError(f"Unknown task type: {task_type}. Use 'classification', 'force', or 'sliding'.")
    
    print("Probe evaluation completed.")
