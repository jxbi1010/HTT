"""
SITR Model Loader and Encoder Wrapper

This module provides utilities to load SITR models from the gsrl repository
and wrap them as encoders compatible with the probe evaluation system (run_probe.py).

Usage:
    # Option 1: Load encoder and use with run_probe_mlp manually
    from load_sitr import load_sitr_encoder
    
    encoder = load_sitr_encoder(use_cls_token_only=True)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Use with run_probe_mlp (from run_probe.py)
    probe_results = run_probe_mlp(
        encoder=encoder,
        shared_trunk=None,  # SITR doesn't use shared trunk
        device=device,
        probe_train_loader=train_loader,
        probe_val_loader=val_loader,
        probe_test_loader=test_loader,
        task_type='classification',  # or 'force', 'pose'
        num_classes=num_classes,
        embed_dim=embed_dim,  # inferred from encoder output
        encoder_output_is_sequence=False,  # False if use_cls_token_only=True
        ...
    )
    
    # Option 2: Use run_sitr_probe for complete workflow
    from load_sitr import run_sitr_probe
    
    probe_results = run_sitr_probe(
        task_type='classification',
        modality='gsmini',
        probe_config=probe_config,
        device=device,
        use_cls_token_only=True,
        ...
    )
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
import warnings
from typing import Optional, Dict, Any
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.environ.get('SITR_REPO', 'third_party/SITR'))

# Suppress timm deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

from models.networks import SITR_base

# Import probe-related functions from run_probe.py
from run_probe import run_probe_mlp, ProbeDataloaderManager
from data.create_dataloaders import create_dataloaders_from_config, get_num_classes_from_dataset
from utils.task_handlers import get_batch_data
# NOTE: data layer was refactored to the 4probe layout — `load_force_stats`
# now lives in `data.gsmini_force_4probe_50each_dataloader`. Pose support
# (`data.pose_dataloader`) was removed entirely. Import lazily so the file
# still imports when only force / classification / sliding are used.
def load_force_stats(modality, config_dir="config/data"):
    from data.gsmini_force_4probe_50each_dataloader import load_force_stats as _impl
    return _impl(modality, config_dir=config_dir)

def load_pose_stats(*args, **kwargs):
    raise RuntimeError(
        "Pose support removed during the 4probe data refactor — `load_pose_stats` "
        "is no longer available. Use task='force'|'classification'|'sliding'."
    )


class SITREncoderWrapper(nn.Module):
    """
    Wrapper for SITR_base model to make it compatible with probe evaluation.
    
    This wrapper uses the forward_encoder method of SITR to extract encoder features.
    The encoder outputs sequence features [B, N, C] where N = num_patches + 1 (includes cls token at position 0).
    
    For downstream tasks, you can choose to:
    1. Use only the cls token (recommended) - set use_cls_token_only=True
    2. Average all tokens (including cls) - set use_cls_token_only=False
    
    Compatible with run_probe.py and extract_features function.
    """
    
    def __init__(self, sitr_model, use_cls_token_only=True):
        """
        Initialize the wrapper.
        
        Args:
            sitr_model: SITR_base model instance
            use_cls_token_only: If True, return only cls token [B, C]. 
                               If False, return all tokens [B, N, C] for averaging later.
                               Default: True (recommended for transformers)
        """
        super().__init__()
        self.sitr_model = sitr_model
        self.num_calibration = 0
        self.use_cls_token_only = use_cls_token_only
    
    def forward(self, x):
        """
        Forward pass through SITR encoder.
        
        Args:
            x: Input tensor [B, C, H, W] for images or [B, T, C, H, W] for video
        
        Returns:
            If use_cls_token_only=True:
                - For images [B, C, H, W]: returns [B, C]
                - For video [B, T, C, H, W]: returns [B, T, C] (can be averaged over time by extract_features)
            If use_cls_token_only=False:
                - For images [B, C, H, W]: returns [B, N, C] where N = num_patches + 1
                - For video [B, T, C, H, W]: returns [B, T, N, C] (can be averaged over time by extract_features)
        """
        # Handle video input [B, T, C, H, W] by reshaping to [B*T, C, H, W]
        original_shape = x.shape
        if len(original_shape) == 5:
            # Video input: [B, T, C, H, W] -> [B*T, C, H, W]
            B, T, C, H, W = original_shape
            x = x.view(B * T, C, H, W)
            is_video = True
        else:
            is_video = False
        
        # Forward through encoder: returns [B*T, N, C] where N = num_patches + 1
        # Position 0 is the cls token
        features = self.sitr_model.forward_encoder(x, c=None)
        
        if self.use_cls_token_only:
            # Extract only cls token (position 0): [B*T, N, C] -> [B*T, C]
            features = features[:, 0, :]
        # else: features is [B*T, N, C]
        
        # Reshape back if video input
        if is_video:
            if self.use_cls_token_only:
                # [B*T, C] -> [B, T, C]
                # extract_features will average over time dimension if encoder_output_is_sequence=True
                features = features.view(B, T, -1)
            else:
                # [B*T, N, C] -> [B, T, N, C]
                # extract_features will average over time and sequence dimensions if encoder_output_is_sequence=True
                features = features.view(B, T, features.shape[1], -1)
        
        return features



def count_parameters(model):
    """Count and print model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"\nTotal number of parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 80)

    return total_params, trainable_params, non_trainable_params


def load_sitr_encoder(return_wrapped=True, num_calibration=0, use_cls_token_only=True):
    """
    Load SITR model and optionally wrap it as an encoder.
    
    Args:
        return_wrapped: If True, return SITREncoderWrapper; if False, return raw SITR_base
        num_calibration: Number of calibration frames (default: 0, no calibration)
        use_cls_token_only: If True, wrapper returns only cls token [B, C] (recommended).
                           If False, wrapper returns all tokens [B, N, C] for averaging.
                           Default: True
    
    Returns:
        SITREncoderWrapper (if return_wrapped=True) or SITR_base (if return_wrapped=False)
    """
    
    # Create SITR_base model with specified num_calibration
    base = SITR_base(num_calibration=num_calibration)

    count_parameters(base)

    # Load checkpoint if available
    base_checkpoint_path = os.environ.get(
        'SITR_CHECKPOINT',
        os.path.join(os.environ.get('SITR_REPO', 'third_party/SITR'), 'checkpoints/SITR_B18.pth'))
    if os.path.exists(base_checkpoint_path):
        print(f"\nLoading checkpoint from {base_checkpoint_path}...")
        checkpoint = torch.load(base_checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict (strict=False to handle num_calibration mismatch)
        base.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print(f"\nCheckpoint not found at {base_checkpoint_path}")
        print("Using randomly initialized weights.")

    if return_wrapped:
        # Return wrapped encoder for probe evaluation
        encoder = SITREncoderWrapper(base, use_cls_token_only=use_cls_token_only)
        token_strategy = "cls token only" if use_cls_token_only else "all tokens (for averaging)"
        print(f"\nSITR model wrapped as encoder for probe evaluation.")
        print(f"  Token strategy: {token_strategy}")
        
        return encoder
    else:
        return base


def run_sitr_probe(
    task_type: str,
    modality: str,
    probe_config: OmegaConf,
    device: torch.device,
    use_cls_token_only: bool = True,
    num_calibration: int = 0,
    run_name: Optional[str] = None,
    config: Optional[OmegaConf] = None,
    finetune: bool = False,
) -> Dict[str, Any]:
    """
    Run probe evaluation using SITR model as encoder.
    
    This function handles the complete workflow:
    1. Loads SITR encoder
    2. Sets up dataloaders based on task type
    3. Runs probe training and evaluation
    4. Returns results
    
    Args:
        task_type: Task type ('classification', 'force', or 'pose')
        modality: Modality name (e.g., '9dtact', 'xela', 'gsmini', 'tac02')
        probe_config: Probe configuration (OmegaConf object)
        device: Device to run on
        use_cls_token_only: If True, use only CLS token (recommended). If False, use all tokens.
        num_calibration: Number of calibration frames for SITR (default: 0)
        run_name: Optional run name for logging (default: auto-generated)
        config: Optional full config object (for compatibility with run_probe.py)
    
    Returns:
        Dictionary with probe results
    """
    print(f"\n{'='*80}")
    print("SITR PROBE EVALUATION")
    print(f"{'='*80}")
    print(f"Task: {task_type}")
    print(f"Modality: {modality}")
    print(f"Using CLS token only: {use_cls_token_only}")
    
    # ============================================================================
    # STEP 1: Load SITR encoder
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 1: Loading SITR model")
    print(f"{'='*80}")
    
    encoder = load_sitr_encoder(
        return_wrapped=True,
        num_calibration=num_calibration,
        use_cls_token_only=use_cls_token_only
    ).to(device)
    
    # Freeze (default) or unfreeze for full-model finetune. run_probe_mlp's
    # own train/eval-mode + requires_grad logic will take over once finetune
    # is plumbed through; this initial block just sets the starting state.
    if finetune:
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad = True
        print("SITR encoder UNFROZEN — running full-model finetune.")
    else:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
    
    # SITR doesn't have shared trunk
    shared_trunk = None
    
    # Get embedding dimension from a dummy forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_output = encoder(dummy_input)
        if len(dummy_output.shape) == 2:
            embed_dim = dummy_output.shape[1]
            encoder_output_is_sequence = False
        else:
            embed_dim = dummy_output.shape[-1]
            encoder_output_is_sequence = True
    
    print(f"SITR encoder loaded successfully")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Output is sequence: {encoder_output_is_sequence}")
    print(f"  Using CLS token only: {use_cls_token_only}")
    
    # ============================================================================
    # STEP 2: Initialize dataloaders
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 2: Initializing dataloaders")
    print(f"{'='*80}")
    
    # Create config if not provided
    if config is None:
        config = OmegaConf.create({
            'training': {
                'batch_size': probe_config.probe.batch_size,
                'num_workers': probe_config.probe.num_workers,
                'probe_data_percentage': probe_config.probe.probe_data_percentage
            }
        })
    
    # Setup TensorBoard writer
    if run_name is None:
        from datetime import datetime
        run_name = f"sitr_{task_type}_{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log_dir = probe_config.logging.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, run_name))
    
    if task_type == 'classification':
        # Classification: Use existing dataloaders
        print(f"Loading classification dataloaders for {modality}...")
        
        # Get data config path from probe config
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
        merged_data_config['data']['train_data_percentage'] = probe_config.probe.probe_data_percentage
        
        # Disable background subtraction and normalization for SITR
        merged_data_config['data']['apply_background_subtraction'] = False
        merged_data_config['data']['normalize_images'] = False
        merged_data_config['data']['normalize_tactile'] = False
        print("SITR dataloader settings: background subtraction=False, normalization=False")
        
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
        
        print(f"Embedding dimension (SITR): {embed_dim}")
        print(f"Encoder output is sequence (SITR): {encoder_output_is_sequence}")
        
        # Get scheduler config if available
        lr_scheduler_config = None
        if hasattr(probe_config.probe, 'lr_scheduler') and probe_config.probe.lr_scheduler is not None:
            lr_scheduler_config = OmegaConf.to_container(probe_config.probe.lr_scheduler, resolve=True)
        
        # ============================================================================
        # STEP 3: Run classification probe
        # ============================================================================
        print(f"\n{'='*80}")
        print("STEP 3: Running classification probe")
        print(f"{'='*80}")
        print("Progress bar will show: Train Loss, Val Loss, Val Accuracy")
        print(f"Running {len(probe_config.probe.probe_seeds)} trial(s) with seeds: {probe_config.probe.probe_seeds}")
        
        # Use `eval_steps` (canonical key) as the validation interval.
        # The previous min(..., 10) clamp produced 200 val passes per
        # 2000-step run, which is way too expensive for a ~80M-param backbone.
        _vi = probe_config.probe.get('probe_val_interval',
                                      probe_config.probe.get('eval_steps', 100))
        val_interval = int(_vi)
        print(f"Validation interval: {val_interval} steps")
        print("=" * 80)
        
        # Patch tqdm to print validation metrics
        from tqdm import tqdm
        original_set_postfix = tqdm.set_postfix
        
        def enhanced_set_postfix(self, *args, **kwargs):
            """Enhanced set_postfix that also prints validation metrics."""
            result = original_set_postfix(self, *args, **kwargs)
            # Check if this is a validation update (has Val Loss or Val Acc)
            if kwargs:
                val_loss = kwargs.get('Val Loss', None)
                val_acc = kwargs.get('Val Acc', None)
                train_loss = kwargs.get('Train Loss', None)
                if val_loss is not None or val_acc is not None:
                    # This is a validation step - print the metrics
                    if val_loss and val_acc:
                        print(f"  [Validation] Step {self.n}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}")
                    elif val_loss:
                        print(f"  [Validation] Step {self.n}: Train Loss: {train_loss}, Val Loss: {val_loss}")
            return result
        
        # Temporarily patch tqdm
        tqdm.set_postfix = enhanced_set_postfix
        
        # Default `get_batch_data` in task_handlers has no 'classification'
        # branch — without an explicit fn here, every train batch raises
        # ValueError and the try/except silently `continue`s, leaving the
        # model untrained (Test Acc=0%). Provide an explicit reader.
        def cls_get_batch_data_fn(batch):
            if 'tactile_img' in batch:
                return batch['tactile_img'].to(device, non_blocking=True)
            if 'images' in batch:
                return batch['images'].to(device, non_blocking=True)
            if 'tactile' in batch:
                return batch['tactile'].to(device, non_blocking=True)
            raise ValueError(f"Cls batch missing image key. Got: {list(batch.keys())}")

        try:
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
                probe_val_interval=val_interval,
                probe_lr=float(probe_config.probe.probe_lr),
                probe_seeds=list(probe_config.probe.probe_seeds),
                writer=writer,
                modality=modality,
                get_batch_data_fn=cls_get_batch_data_fn,
                lr_scheduler_config=lr_scheduler_config,
                finetune=finetune,
            )
        finally:
            # Restore original set_postfix
            tqdm.set_postfix = original_set_postfix

        if probe_results:
            print(f"\nClassification Probe Results:")
            print(f"  Mean Test Accuracy: {probe_results['test_accuracy_mean']:.4f} ± {probe_results['test_accuracy_std']:.4f}")
        
        writer.close()
        return probe_results
        
    elif task_type == 'force':
        # Force: Use force dataloaders
        print(f"Loading force dataloaders for {modality}...")
        
        # Get data roots from probe config
        data_roots_dict = probe_config.probe.force.data_roots
        data_roots = OmegaConf.to_container(data_roots_dict, resolve=True)
        
        if modality not in data_roots:
            raise ValueError(f"Data root not provided for modality: {modality} in probe config")
        
        # Create a minimal config with training defaults from probe config
        temp_config = OmegaConf.create({
            'training': {
                'batch_size': probe_config.probe.batch_size,
                'num_workers': probe_config.probe.num_workers,
                'probe_data_percentage': probe_config.probe.probe_data_percentage,
                'min_force': probe_config.probe.force.get('min_force', 0.1),
                'max_force': probe_config.probe.force.get('max_force', 9.9),
                'apply_background_subtraction': False  # Disable background subtraction for SITR
            }
        })
        # Merge with existing config
        config = OmegaConf.merge(config, temp_config)
        
        dataloader_manager = ProbeDataloaderManager(
            task_type='force',
            modality=modality,
            config=config
        )
        
        # Override _load_apply_background_subtraction to return False
        dataloader_manager._load_apply_background_subtraction = lambda config_dir: False
        print("SITR dataloader settings: background subtraction=False, normalization=False")
        
        # Load force dataloaders
        config_dir = probe_config.probe.force.config_dir
        # Get data split seed from config or use first probe_seed as default
        data_split_seed = probe_config.probe.force.get('data_split_seed', None)
        if data_split_seed is None:
            # Use first probe_seed for data splitting if not specified
            data_split_seed = probe_config.probe.probe_seeds[0] if probe_config.probe.probe_seeds else 42
        print(f"Using data split seed: {data_split_seed}")
        dataloader_manager.load_force_dataloaders(
            data_roots=data_roots,
            config_dir=config_dir,
            random_seed=data_split_seed
        )
        
        probe_train_loader = dataloader_manager.train_loader
        probe_val_loader = dataloader_manager.val_loader
        probe_test_loader = dataloader_manager.test_loader
        
        print(f"Embedding dimension (SITR): {embed_dim}")
        print(f"Encoder output is sequence (SITR): {encoder_output_is_sequence}")
        
        # Force stats moved below (per-dim, 3D — for the 4probe layout).
        
        # ============================================================================
        # STEP 3: Run force probe
        # ============================================================================
        print(f"\n{'='*80}")
        print("STEP 3: Running force probe")
        print(f"{'='*80}")
        print("Progress bar will show: Train Loss, Train MAE, Val Loss, Val MAE, Val RMSE")
        print(f"Running {len(probe_config.probe.probe_seeds)} trial(s) with seeds: {probe_config.probe.probe_seeds}")
        
        # Use `eval_steps` (canonical key) as the validation interval.
        # The previous min(..., 10) clamp produced 200 val passes per
        # 2000-step run, which is way too expensive for a ~80M-param backbone.
        _vi = probe_config.probe.get('probe_val_interval',
                                      probe_config.probe.get('eval_steps', 100))
        val_interval = int(_vi)
        print(f"Validation interval: {val_interval} steps")
        print("=" * 80)
        
        # Patch tqdm to print validation metrics
        from tqdm import tqdm
        original_set_postfix = tqdm.set_postfix
        
        def enhanced_set_postfix(self, *args, **kwargs):
            """Enhanced set_postfix that also prints validation metrics."""
            result = original_set_postfix(self, *args, **kwargs)
            # Check if this is a validation update (has Val Loss, Val MAE, etc.)
            if kwargs:
                val_loss = kwargs.get('Val Loss', None)
                val_mae = kwargs.get('Val MAE', None)
                val_rmse = kwargs.get('Val RMSE', None)
                train_loss = kwargs.get('Train Loss', None)
                train_mae = kwargs.get('Train MAE', None)
                if val_loss is not None:
                    # This is a validation step - print the metrics
                    metrics_str = f"Train Loss: {train_loss}"
                    if train_mae:
                        metrics_str += f", Train MAE: {train_mae}"
                    metrics_str += f", Val Loss: {val_loss}"
                    if val_mae:
                        metrics_str += f", Val MAE: {val_mae}"
                    if val_rmse:
                        metrics_str += f", Val RMSE: {val_rmse}"
                    print(f"  [Validation] Step {self.n}: {metrics_str}")
            return result
        
        # Temporarily patch tqdm
        tqdm.set_postfix = enhanced_set_postfix
        
        # 4probe vision-modality force: single image input (`tactile_img`),
        # 3D target = `6d_force[:, :3]` (shear_x, shear_y, normal).
        # The old `get_batch_data(batch, 'force', device)` expected the
        # sensor_0/sensor_1 dual layout that doesn't exist in 4probe data.
        def get_batch_data_fn(batch):
            return batch['tactile_img'].to(device, non_blocking=True)
        def get_target_fn(batch):
            return batch['6d_force'][:, :3].to(device, non_blocking=True)

        normalize_targets = probe_config.probe.force.normalize_targets
        # Hardcode 3 to match the 4probe data shape (probe.yaml has 1, which
        # is stale from the pre-4probe layout). Also need per-dim force stats
        # for denormalization.
        output_dim = 3
        from data.gsmini_force_4probe_50each_dataloader import load_force_stats_per_dim
        config_dir = probe_config.probe.force.config_dir
        force_mean, force_std = load_force_stats_per_dim(modality, config_dir=config_dir, dims=3)
        print(f"Force stats (per-dim, 3D): Mean={force_mean}, Std={force_std}")
        
        # Get scheduler config if available
        lr_scheduler_config = None
        if hasattr(probe_config.probe, 'lr_scheduler') and probe_config.probe.lr_scheduler is not None:
            lr_scheduler_config = OmegaConf.to_container(probe_config.probe.lr_scheduler, resolve=True)
        
        try:
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
                probe_val_interval=val_interval,
                probe_lr=float(probe_config.probe.probe_lr),
                probe_seeds=list(probe_config.probe.probe_seeds),
                writer=writer,
                modality=modality,
                get_batch_data_fn=get_batch_data_fn,
                get_target_fn=get_target_fn,
                normalize_targets=normalize_targets,
                target_mean=force_mean if normalize_targets else None,
                target_std=force_std if normalize_targets else None,
                lr_scheduler_config=lr_scheduler_config,
                finetune=finetune,
                force_single_sensor=True,  # SITR is a single-image encoder
            )
        finally:
            # Restore original set_postfix
            tqdm.set_postfix = original_set_postfix

        if probe_results:
            print(f"\nForce Probe Results:")
            print(f"  Mean Test MAE: {probe_results['test_mae_mean']:.4f} ± {probe_results['test_mae_std']:.4f}")
            print(f"  Mean Test RMSE: {probe_results['test_rmse_mean']:.4f} ± {probe_results['test_rmse_std']:.4f}")
        
        writer.close()
        return probe_results
        
    elif task_type == 'pose':
        # Pose: Use pose dataloaders
        print(f"Loading pose dataloaders for {modality}...")
        
        # Get data roots from probe config
        data_roots_dict = probe_config.probe.pose.data_roots
        data_roots = OmegaConf.to_container(data_roots_dict, resolve=True)
        
        if modality not in data_roots:
            raise ValueError(f"Data root not provided for modality: {modality} in probe config")
        
        # Create a minimal config with training defaults from probe config
        temp_config = OmegaConf.create({
            'training': {
                'batch_size': probe_config.probe.batch_size,
                'num_workers': probe_config.probe.num_workers,
                'probe_data_percentage': probe_config.probe.probe_data_percentage,
                'apply_background_subtraction': False  # Disable background subtraction for SITR
            }
        })
        # Merge with existing config
        config = OmegaConf.merge(config, temp_config)
        
        dataloader_manager = ProbeDataloaderManager(
            task_type='pose',
            modality=modality,
            config=config
        )
        
        # Override _load_apply_background_subtraction to return False
        dataloader_manager._load_apply_background_subtraction = lambda config_dir: False
        print("SITR dataloader settings: background subtraction=False, normalization=False")
        
        # Load pose dataloaders
        config_dir = probe_config.probe.pose.config_dir
        # Get data split seed from config or use first probe_seed as default
        data_split_seed = probe_config.probe.pose.get('data_split_seed', None)
        if data_split_seed is None:
            # Use first probe_seed for data splitting if not specified
            data_split_seed = probe_config.probe.probe_seeds[0] if probe_config.probe.probe_seeds else 42
        print(f"Using data split seed: {data_split_seed}")
        dataloader_manager.load_pose_dataloaders(
            data_roots=data_roots,
            config_dir=config_dir,
            random_seed=data_split_seed
        )
        
        probe_train_loader = dataloader_manager.train_loader
        probe_val_loader = dataloader_manager.val_loader
        probe_test_loader = dataloader_manager.test_loader
        
        print(f"Embedding dimension (SITR): {embed_dim}")
        print(f"Encoder output is sequence (SITR): {encoder_output_is_sequence}")
        print(f"Note: For pose tasks, concatenated embedding dimension will be {embed_dim * 2}")
        
        # Load pose stats for denormalization
        translation_std, rotation_std = load_pose_stats(modality, config_dir=config_dir)
        print(f"Pose stats - Translation std: {translation_std}, Rotation std: {rotation_std}")
        
        # ============================================================================
        # STEP 3: Run pose probe
        # ============================================================================
        print(f"\n{'='*80}")
        print("STEP 3: Running pose probe")
        print(f"{'='*80}")
        print("Progress bar will show: Train Loss, Val Loss, Val Translation MAE, Val Rotation MAE")
        print(f"Running {len(probe_config.probe.probe_seeds)} trial(s) with seeds: {probe_config.probe.probe_seeds}")
        
        # Use `eval_steps` (canonical key) as the validation interval.
        # The previous min(..., 10) clamp produced 200 val passes per
        # 2000-step run, which is way too expensive for a ~80M-param backbone.
        _vi = probe_config.probe.get('probe_val_interval',
                                      probe_config.probe.get('eval_steps', 100))
        val_interval = int(_vi)
        print(f"Validation interval: {val_interval} steps")
        print("=" * 80)
        
        # Patch tqdm to print validation metrics
        from tqdm import tqdm
        original_set_postfix = tqdm.set_postfix
        
        def enhanced_set_postfix(self, *args, **kwargs):
            """Enhanced set_postfix that also prints validation metrics."""
            result = original_set_postfix(self, *args, **kwargs)
            # Check if this is a validation update (has Val Loss, Val Trans MAE, etc.)
            if kwargs:
                val_loss = kwargs.get('Val Loss', None)
                val_trans_mae = kwargs.get('Val Trans MAE', None)
                val_rot_mae = kwargs.get('Val Rot MAE', None)
                train_loss = kwargs.get('Train Loss', None)
                if val_loss is not None:
                    # This is a validation step - print the metrics
                    metrics_str = f"Train Loss: {train_loss}, Val Loss: {val_loss}"
                    if val_trans_mae:
                        metrics_str += f", Val Trans MAE: {val_trans_mae}"
                    if val_rot_mae:
                        metrics_str += f", Val Rot MAE: {val_rot_mae}"
                    print(f"  [Validation] Step {self.n}: {metrics_str}")
            return result
        
        # Temporarily patch tqdm
        tqdm.set_postfix = enhanced_set_postfix
        
        # Use get_batch_data from task_handlers which correctly handles sensor_0/sensor_1 format
        def get_batch_data_fn(batch):
            return get_batch_data(batch, task_type, device)
        
        # Get scheduler config if available
        lr_scheduler_config = None
        if hasattr(probe_config.probe, 'lr_scheduler') and probe_config.probe.lr_scheduler is not None:
            lr_scheduler_config = OmegaConf.to_container(probe_config.probe.lr_scheduler, resolve=True)
        
        try:
            probe_results = run_probe_mlp(
                encoder=encoder,
                shared_trunk=shared_trunk,
                device=device,
                probe_train_loader=probe_train_loader,
                probe_val_loader=probe_val_loader,
                probe_test_loader=probe_test_loader,
                task_type='pose',
                embed_dim=embed_dim,
                encoder_output_is_sequence=encoder_output_is_sequence,
                probe_total_steps=probe_config.probe.probe_total_steps,
                probe_val_interval=val_interval,
                probe_lr=float(probe_config.probe.probe_lr),
                probe_seeds=list(probe_config.probe.probe_seeds),
                writer=writer,
                modality=modality,
                get_batch_data_fn=get_batch_data_fn,
                translation_std=translation_std,
                rotation_std=rotation_std,
                lr_scheduler_config=lr_scheduler_config,
                finetune=finetune,
            )
        finally:
            # Restore original set_postfix
            tqdm.set_postfix = original_set_postfix
        
        if probe_results:
            print(f"\nPose Probe Results:")
            print(f"  Translation - Mean Test MAE: {probe_results['test_translation_mae_mean']:.4f} ± {probe_results['test_translation_mae_std']:.4f}")
            print(f"  Translation - Mean Test RMSE: {probe_results['test_translation_rmse_mean']:.4f} ± {probe_results['test_translation_rmse_std']:.4f}")
            print(f"  Rotation - Mean Test MAE: {probe_results['test_rotation_mae_mean']:.4f} ± {probe_results['test_rotation_mae_std']:.4f}")
            print(f"  Rotation - Mean Test RMSE: {probe_results['test_rotation_rmse_mean']:.4f} ± {probe_results['test_rotation_rmse_std']:.4f}")
        
        writer.close()
        return probe_results
        
    else:
        raise ValueError(f"Unknown task type: {task_type}. Must be 'classification', 'force', or 'pose'.")


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='SITR Probe Evaluation')
    parser.add_argument('--task', type=str, required=False, default='classification',
                       choices=['classification', 'force', 'pose', 'sliding'],
                       help='Task type: classification, force, pose, or sliding')
    parser.add_argument('--modality', type=str, required=False, default='gsmini',
                       choices=['9dtact', 'gsmini', 'xela', 'tac02'],
                       help='Modality to probe')
    parser.add_argument('--probe_config', type=str, default='config/algo/probe.yaml',
                       help='Path to probe config file (default: config/algo/probe.yaml)')
    parser.add_argument('--use_cls_token_only', action='store_true', default=False,
                       help='Use only CLS token (default: True, recommended). Set to False to use all tokens.')
    parser.add_argument('--no_cls_token_only', dest='use_cls_token_only', action='store_false',
                       help='Use all tokens instead of CLS token only')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10],
                       help='List of random seeds for probe evaluation (overrides config). Example: --seeds 10 20 30')
    parser.add_argument('--probe_data_percentage', type=float, default=0.25,
                       help='Percentage of training data to use for probe evaluation (0.0 to 1.0). Overrides config if provided.')
    parser.add_argument('--finetune', action='store_true',
                       help='Fine-tune the SITR encoder alongside the probe head. Default: frozen-encoder linear probe.')

    args = parser.parse_args()
    
    # ============================================================================
    # STEP 0: Load probe configuration
    # ============================================================================
    print(f"\n{'='*80}")
    print("SITR PROBE EVALUATION")
    print(f"{'='*80}")
    print(f"Task: {args.task}")
    print(f"Modality: {args.modality}")
    print(f"Using CLS token only: {args.use_cls_token_only}")
    
    if not os.path.exists(args.probe_config):
        raise FileNotFoundError(f"Probe config file not found: {args.probe_config}")
    
    with open(args.probe_config, 'r') as f:
        probe_config_dict = yaml.safe_load(f)
    
    probe_config = OmegaConf.create(probe_config_dict)
    
    # Override config with command-line arguments
    probe_config.probe.task_type = args.task
    probe_config.probe.modality = args.modality
    
    if args.seeds is not None:
        probe_config.probe.probe_seeds = args.seeds
        print(f"Overriding probe seeds from command line: {args.seeds}")
    if args.probe_data_percentage is not None:
        probe_config.probe.probe_data_percentage = args.probe_data_percentage
        print(f"Overriding probe_data_percentage from command line: {args.probe_data_percentage}")
    
    task_type = args.task
    modality = args.modality
    
    # Legacy: the pre-refactor force/pose dataloaders used "gelsight" instead
    # of "gsmini". The 4probe dataloaders (current) use "gsmini" everywhere,
    # so the remap is no longer needed and would break data-root lookup.
    
    # Generate run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sitr_{task_type}_{args.modality}_{timestamp}"
    print(f"Run name: {run_name}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================================
    # STEP 1: Run SITR probe evaluation
    # ============================================================================
    try:
        if task_type == 'sliding':
            # Delegate sliding to the unified baseline runner (it knows how to
            # build sliding dataloaders, compute class weights, and report
            # macro-F1; SITR is just one of two supported backbones there).
            from load_baselines import run_baseline_probe
            probe_results = run_baseline_probe(
                backbone='sitr',
                task_type='sliding',
                modality=args.modality,  # use raw, not the gelsight-mapped name
                probe_config=probe_config,
                device=device,
                run_name=run_name,
                use_cls_token_only=args.use_cls_token_only,
                num_calibration=0,
                finetune=args.finetune,
            )
        else:
            probe_results = run_sitr_probe(
                task_type=task_type,
                modality=modality,
                probe_config=probe_config,
                device=device,
                use_cls_token_only=args.use_cls_token_only,
                num_calibration=0,
                run_name=run_name,
                config=None,
                finetune=args.finetune,
            )
        
        print(f"\n{'='*80}")
        print("SITR PROBE EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n{'='*80}")
        print("SITR PROBE EVALUATION FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
