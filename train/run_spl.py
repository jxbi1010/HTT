#!/usr/bin/env python3
"""
Unified Classification Training Script

This script provides a unified interface for training different types of classification models
based on configuration files. It supports:
- Image classification (9DTact, GsMini)
- Tactile classification (Tacniq, Xela)

"""

import os
import argparse
import yaml
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
import random
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from omegaconf import OmegaConf

# Import dataloader creation function
# Make the repo root importable when running this script directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.create_dataloaders import create_dataloaders_from_config, get_num_classes_from_dataset
from data.gsmini_force_4probe_50each_dataloader import (
    create_force_4probe_50each_dataloader as create_force_dataloaders_4probe,
    load_force_stats_per_dim as load_force_stats_per_dim_4probe,
)
from data.taxel_force_4probe_50each_dataloader import (
    create_taxel_force_dataloader,
    load_force_stats_per_dim as load_taxel_force_stats_per_dim,
    TAXEL_DIMS as TAXEL_FORCE_DIMS,
)
from torch.utils.data import DataLoader

# Import model factories
from model.create_model import create_model

# Import utilities
from utils.base_trainer import BaseTrainer
from utils.scheduler import create_scheduler, print_scheduler_info
from utils.augmentation import create_augmentation_from_config
from utils.utils_model import set_all_seeds
from utils.task_handlers import (
    get_batch_data, get_target, create_loss_function, forward_pass, compute_loss,
    compute_force_dual_head_loss,
    calculate_metrics, postprocess_outputs, calculate_force_metrics_detailed,
    get_task_display_name, validate_task_type
)
from utils.dataset_utils import get_dataset_length, get_dataloader_length
from utils.downstream_loops import run_evaluation_loop
from utils.result_aggregation import aggregate_force_results, aggregate_classification_results


class UnifiedConfig:
    """Unified configuration class for all training types."""
    
    def __init__(self, config_path: str, model_config_path: str = None, algo_config_path: str = None, 
                 train_data_percentage: float = None, random_seed: int = None, task_type: str = 'classification'):
        self.config_path = config_path
        self.model_config_path = model_config_path
        self.algo_config_path = algo_config_path
        self.train_data_percentage_override = train_data_percentage
        self.random_seed_override = random_seed
        self.task_type = task_type  # 'classification' or 'force'
        self.load_config()
        self.determine_training_type()
    
    def load_config(self):
        """Load configuration from YAML files."""
        # Load main config (sensor config)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            with open(self.config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.Loader)
        
        # Load model config if provided
        model_config = {}
        if self.model_config_path and os.path.exists(self.model_config_path):
            try:
                with open(self.model_config_path, 'r') as f:
                    model_config = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                with open(self.model_config_path, 'r') as f:
                    model_config = yaml.load(f, Loader=yaml.Loader)
        
        # Load algorithm config (supervised.yaml) if provided
        algo_config = {}
        if self.algo_config_path and os.path.exists(self.algo_config_path):
            try:
                with open(self.algo_config_path, 'r') as f:
                    algo_config = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                with open(self.algo_config_path, 'r') as f:
                    algo_config = yaml.load(f, Loader=yaml.Loader)
        
        # Merge configs
        self.config = config
        self.model_config = model_config
        
        # Update config with model parameters
        if 'model' not in self.config:
            self.config['model'] = {}
        self.config['model'].update(model_config)
        
        # Merge algorithm config (supervised.yaml) - takes precedence for training and logging
        if algo_config:
            # Map algorithm.optimizer to training
            if 'algorithm' in algo_config and 'optimizer' in algo_config['algorithm']:
                if 'training' not in self.config:
                    self.config['training'] = {}
                optimizer = algo_config['algorithm']['optimizer']
                self.config['training']['learning_rate'] = optimizer.get('lr')
                self.config['training']['weight_decay'] = optimizer.get('weight_decay', 0.0)
            
            # Map algorithm.lr_scheduler to training.scheduler
            if 'algorithm' in algo_config and 'lr_scheduler' in algo_config['algorithm']:
                if 'training' not in self.config:
                    self.config['training'] = {}
                if 'scheduler' not in self.config['training']:
                    self.config['training']['scheduler'] = {}
                scheduler = algo_config['algorithm']['lr_scheduler']
                self.config['training']['scheduler']['type'] = scheduler.get('type')
                self.config['training']['scheduler'].update(scheduler.get('kwargs', {}))
            
            # Map algorithm.kwargs.l2_lambda to training.l2_lambda
            if 'algorithm' in algo_config and 'kwargs' in algo_config['algorithm']:
                if 'training' not in self.config:
                    self.config['training'] = {}
                if 'l2_lambda' in algo_config['algorithm']['kwargs']:
                    self.config['training']['l2_lambda'] = algo_config['algorithm']['kwargs']['l2_lambda']
            
            # Merge training section (takes precedence over sensor config)
            if 'training' in algo_config:
                if 'training' not in self.config:
                    self.config['training'] = {}
                self.config['training'].update(algo_config['training'])
            
            # Merge logging section (takes precedence over sensor config)
            if 'logging' in algo_config:
                if 'logging' not in self.config:
                    self.config['logging'] = {}
                self.config['logging'].update(algo_config['logging'])
            
            # Merge data section (takes precedence over sensor config)
            if 'data' in algo_config:
                if 'data' not in self.config:
                    self.config['data'] = {}
                self.config['data'].update(algo_config['data'])
            
            # Merge force section (for force tasks)
            if 'force' in algo_config:
                self.config['force'] = algo_config['force']
            
        
        # Override train_data_percentage if provided
        if self.train_data_percentage_override is not None:
            if 'data' not in self.config:
                self.config['data'] = {}
            self.config['data']['train_data_percentage'] = self.train_data_percentage_override
        
        # Override random_seed if provided
        if self.random_seed_override is not None:
            if 'data' not in self.config:
                self.config['data'] = {}
            self.config['data']['random_seed'] = self.random_seed_override
        
        # Set attributes from configs with proper type conversion
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for param_key, param_value in sub_value.items():
                            converted_value = self._convert_value(param_value)
                            setattr(self, f"{key}_{sub_key}_{param_key}", converted_value)
                    else:
                        converted_value = self._convert_value(sub_value)
                        setattr(self, f"{key}_{sub_key}", converted_value)
            else:
                converted_value = self._convert_value(value)
                setattr(self, key, converted_value)
    
    def _convert_value(self, value):
        """Convert string values to appropriate types."""
        if isinstance(value, str):
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return int(value)
            except ValueError:
                pass
        return value
    
    def determine_training_type(self):
        """Determine the type of training based on config."""
        data_root = getattr(self, 'data_data_root', '')
        config_path = getattr(self, 'config_path', '')
        
        # Detect training type based on config file name first, then data root
        if 'gsmini_config' in config_path.lower():
            self.training_type = 'images'
            self.dataset_type = 'tacniq_gsmini'
        elif 'tac_config' in config_path.lower():
            self.training_type = 'tactile'
            self.dataset_type = 'tacniq_gsmini'
        elif '9dtact_config' in config_path.lower():
            self.training_type = 'images'
            self.dataset_type = 'xela_9dtact'
        elif 'xela_config' in config_path.lower():
            self.training_type = 'tactile'
            self.dataset_type = 'xela_9dtact'
        else:
            raise ValueError(f"Unsupported dataset type: {data_root}")

        
    
    def get(self, key, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_attr(self, attr_name, default=None):
        """Get attribute if it exists, otherwise return None to let dataloader use its defaults."""
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        return default


class UnifiedTrainer(BaseTrainer):
    """Unified trainer class for classification and force tasks."""

    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config
        self.task_type = config.task_type  # 'classification' or 'force'
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.best_accuracy = 0.0  # For classification
        self.best_mae = float('inf')  # For force
        self.best_normal_mae = None  # For force (output_dim==3)
        self.best_shear_mae = None   # For force (output_dim==3)
        self.best_val_loss = float('inf')  # For early stopping based on validation loss
        self.best_epoch = 0  # Track the epoch with best validation loss (for checkpoints)
        self.best_step = 0  # Track the step with best validation loss (for display)
        self.patience_counter = 0
        
        # Set random seeds if random_seed is available
        random_seed = getattr(config, 'data_random_seed', None)
        if random_seed is not None:
            set_all_seeds(random_seed)

        # Setup components
        self.setup_logging()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_augmentation()
    
    def setup_logging(self):
        """Setup logging and checkpoint directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.config.get('logging', {}).get('log_dir', 'logs')
        save_dir = self.config.get('logging', {}).get('save_dir', 'checkpoints')

        run_name = self.config.get('logging', {}).get('run_name', None)
        run_suffix = run_name if run_name else f"run_{timestamp}"

        super().setup_logging(log_dir, save_dir, run_suffix)
        self.run_log_dir = self.log_path
        self.run_save_dir = os.path.abspath(os.path.join(save_dir, run_suffix))
        os.makedirs(self.run_save_dir, exist_ok=True)
        print(f"Run: {run_suffix} | logs: {self.run_log_dir}")
    
    def setup_data(self):
        """Setup data loaders based on task type (classification, force, or sliding)."""
        if self.task_type in ('force', 'sliding'):
            # Force / sliding: gsmini 4probe .npz episodes (sliding predicts bracket labels)
            print(f"Setting up {'sliding' if self.task_type == 'sliding' else 'force'} data loaders...")
            
            # Get force config from merged config
            force_config = self.config.get('force', {})
            if not force_config:
                raise ValueError("Force configuration not found. Please ensure 'force' section exists in algo config (spl.yaml)")
            
            # Get modality from config (overridden by --modality in main())
            modality = force_config.get('modality', 'gsmini')
            is_image_modality = modality in ('gsmini', '9dtact')
            is_taxel_modality = modality in TAXEL_FORCE_DIMS
            if not (is_image_modality or is_taxel_modality):
                raise ValueError(
                    f"Force/sliding task not supported for modality '{modality}'. "
                    f"Supported: gsmini, 9dtact, xela, tac02."
                )

            config_dir = force_config.get('config_dir', 'config/data')

            # Get apply_background_subtraction from config (default to True)
            apply_background_subtraction = self.config.get_attr('data_apply_background_subtraction', True)
            # Also check force_config for this parameter (allows override in algo config)
            if 'apply_background_subtraction' in force_config:
                apply_background_subtraction = force_config.get('apply_background_subtraction', True)

            # Derive data_root from modality + task so --modality override works
            # without editing data_root in spl.yaml: force → released force/
            # split, sliding → released slip/ split. Honor an explicit override
            # only when it already names the selected modality.
            from data.dataset_paths import force_processed_root, slip_processed_root
            if self.task_type == 'sliding':
                default_root = slip_processed_root(modality)
            else:
                default_root = force_processed_root(modality)
            cfg_data_root = force_config.get('data_root')
            root_parts = cfg_data_root.replace('\\', '/').split('/') if cfg_data_root else []
            if any(p == modality or p == f'{modality}_force_4probe_50each' for p in root_parts):
                data_root = cfg_data_root
            else:
                data_root = default_root

            mode_filter = force_config.get('mode_filter')
            if self.task_type == 'sliding' and mode_filter != 'sliding':
                # Sliding task can only train on episodes that have a matching
                # `.labeled.npz`; static episodes would surface as all -1 labels.
                mode_filter = 'sliding'

            # Get num_workers and prefetch settings from config
            batch_size = int(
                force_config.get('batch_size')
                or self.config.get_attr('data_batch_size')
                or self.config.get('training', {}).get('batch_size', 32)
            )
            num_workers = self.config.get_attr('data_num_workers', 0)
            pin_memory = self.config.get_attr('data_pin_memory', True)
            persistent_workers = self.config.get_attr('data_persistent_workers', False) if num_workers > 0 else False
            default_prefetch = 2 if is_image_modality else 1
            prefetch_factor = self.config.get_attr('data_prefetch_factor', default_prefetch) if num_workers > 0 else None
            timeout = self.config.get_attr('data_loader_timeout', 30) if num_workers > 0 else 0

            data_split_seed = force_config.get('data_split_seed', self.config.get_attr('random_seed', 42))
            load_sliding_labels_kw = force_config.get('load_sliding_labels')
            if self.task_type == 'sliding' and load_sliding_labels_kw is None:
                load_sliding_labels_kw = True

            if is_image_modality:
                # gsmini / 9dtact: processed .npz episodes, tactile_img -> 3D force (normal+shear)
                common = dict(
                    data_root=data_root,
                    modality=modality,
                    batch_size=batch_size,
                    config_dir=config_dir,
                    apply_background_subtraction=apply_background_subtraction,
                    apply_ref_force_subtraction=force_config.get('apply_ref_force_subtraction', True),
                    force_clip_min=force_config.get('force_clip_min', -20.0),
                    force_clip_max=force_config.get('force_clip_max', 20.0),
                    mode_filter=mode_filter,
                    compute_friction_mu=force_config.get('compute_friction_mu'),
                    friction_mu_eps=float(
                        1e-3
                        if force_config.get('friction_mu_eps') is None
                        else force_config.get('friction_mu_eps')
                    ),
                    labeled_data_root=force_config.get('labeled_data_root'),
                    load_sliding_labels=load_sliding_labels_kw,
                    strict_labeled=bool(force_config.get('strict_labeled', False)),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    timeout=timeout,
                    seed=data_split_seed,
                )
                self.train_loader = create_force_dataloaders_4probe(split='train', **common)
                self.val_loader = create_force_dataloaders_4probe(split='val', **common)
                self.test_loader = create_force_dataloaders_4probe(split='test', **common)
                self._tactile_key = 'tactile_img'
            else:
                # xela / tac02: taxel .npz episodes, tactile chunk -> 3D force (normal+shear)
                taxel_dim = TAXEL_FORCE_DIMS[modality]
                common = dict(
                    modality=modality,
                    taxel_dim=taxel_dim,
                    data_root=data_root,
                    batch_size=batch_size,
                    chunk_size=force_config.get('tactile_chunk_size', 20),
                    config_dir=config_dir,
                    apply_ref_tactile_subtraction=force_config.get('apply_ref_tactile_subtraction', True),
                    apply_ref_force_subtraction=force_config.get('apply_ref_force_subtraction', True),
                    apply_force_normalization=force_config.get('apply_force_normalization', True),
                    force_clip_min=force_config.get('force_clip_min', -20.0),
                    force_clip_max=force_config.get('force_clip_max', 20.0),
                    mode_filter=mode_filter,
                    stride=force_config.get('stride', 1),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    timeout=timeout,
                    seed=data_split_seed,
                )
                self.train_loader = create_taxel_force_dataloader(split='train', **common)
                self.val_loader = create_taxel_force_dataloader(split='val', **common)
                self.test_loader = create_taxel_force_dataloader(split='test', **common)
                self._tactile_key = 'tactile'
            
            self.train_dataset = self.train_loader.dataset
            self.val_dataset = self.val_loader.dataset
            self.test_dataset = self.test_loader.dataset
            
            self.modality = modality
            if self.task_type == 'sliding':
                self.num_classes = int(force_config.get('num_sliding_classes', 3))
                self.output_dim = self.num_classes
                self.force_mean, self.force_std = None, None
            else:
                if is_taxel_modality:
                    self.force_mean, self.force_std = load_taxel_force_stats_per_dim(
                        modality, config_dir=config_dir, dims=3
                    )
                else:
                    self.force_mean, self.force_std = load_force_stats_per_dim_4probe(
                        modality, config_dir=config_dir, dims=3
                    )
                self.num_classes = None  # Not applicable for force
                self.output_dim = 3  # normal + shear
            
            print(f"Data setup completed:")
            print(f"  Train samples: {self._get_dataset_length(self.train_dataset, 'train_dataset')}")
            print(f"  Validation samples: {self._get_dataset_length(self.val_dataset, 'val_dataset')}")
            print(f"  Test samples: {self._get_dataset_length(self.test_dataset, 'test_dataset')}")
            if self.task_type == 'sliding':
                print(f"  Sliding classes: {self.num_classes} (bracket labels 0..C-1; -1 ignored)")
            else:
                print(f"  Force stats (per-dim for denorm): Mean={self.force_mean}, Std={self.force_std}")
        
        else:
            # Classification: Use existing dataloaders
            num_workers = self.config.get_attr('data_num_workers', 0)
            
            # Add training_type to config for dataloader creation. Sensor configs
            # (e.g. gsmini_config.yaml) only define a subset of these fields, so
            # access via get_attr with sensible defaults — matches the probe path.
            config_with_training_type = OmegaConf.create({
                'training_type': self.config.training_type,
                'dataset_type': self.config.dataset_type,
                'data_root': self.config.data_data_root,
                'batch_size': self.config.get_attr('data_batch_size', 256),
                'train_ratio': self.config.get_attr('data_train_ratio', 0.8),
                'val_ratio': self.config.get_attr('data_val_ratio', 0.1),
                'test_ratio': self.config.get_attr('data_test_ratio', 0.1),
                'num_workers': num_workers,
                'random_seed': self.config.get_attr('data_random_seed', 42),
                'image_chunk_size': self.config.get_attr('data_image_chunk_size'),
                'tactile_chunk_size': self.config.get_attr('data_tactile_chunk_size'),
                'chunk_selection_mode': self.config.get_attr('data_chunk_selection_mode'),
                'overlap_step': self.config.get_attr('data_overlap_step'),
                'apply_background_subtraction': self.config.get_attr('data_apply_background_subtraction'),
                'normalize_images': self.config.get_attr('data_normalize_images'),
                'normalize_tactile': self.config.get_attr('data_normalize_tactile'),
                'pin_memory': self.config.get_attr('data_pin_memory'),
                'persistent_workers': self.config.get_attr('data_persistent_workers'),
                'skip_validation': self.config.get_attr('data_skip_validation'),
                'train_data_percentage': self.config.get_attr('data_train_data_percentage', 1.0),
                'dataset_split_type': 'supervised'  # Use supervised split for SPL training
            })
            
            # Create dataloaders using the standalone function
            self.train_loader, self.val_loader, self.test_loader, self.train_dataset, self.val_dataset, self.test_dataset = create_dataloaders_from_config(config_with_training_type)
            
            # Get number of classes
            self.num_classes = get_num_classes_from_dataset(self.train_dataset)
            self.output_dim = None  # Not applicable for classification
            
            n_train = self._get_dataset_length(self.train_dataset, 'train_dataset')
            n_val = self._get_dataset_length(self.val_dataset, 'val_dataset') if self.val_dataset is not None else 0
            n_test = self._get_dataset_length(self.test_dataset, 'test_dataset')
            print(f"Data: train={n_train}, val={n_val}, test={n_test}, classes={self.num_classes}")
    
    def setup_model(self):
        """Setup model based on task type (classification, force, or pose)."""
        model_config = self.config.config.get('model', {}).copy()

        self.apply_fft = self.config.get_attr('data_apply_fft', False)
        
        if self.task_type in ('force', 'sliding'):
            # Force / sliding: encoder only, MLP head added below
            model_config['architecture'] = ['encoder']
            
            # For force/pose, we need to determine the encoder embed dimension
            # This will be set after creating the encoder
        else:
            # Classification: Use encoder+mlp_head
            model_config['architecture'] = ['encoder', 'classifier']  # 'classifier' kept for backward compatibility
            model_config['num_classes'] = self.num_classes
        
        # For force/pose tasks, infer data type from actual batch (not from config training_type)
        # since these dataloaders can use different modalities than the sensor config suggests
        actual_training_type = self.config.training_type
        sample_data = None  # Initialize for use in tactile_dim inference
        if self.task_type in ('force', 'sliding'):
            # Sample a batch to determine actual data type
            # Use a try-except to handle empty dataloaders gracefully
            try:
                temp_iter = iter(self.train_loader)
                sample_batch = next(temp_iter)
            except StopIteration:
                raise RuntimeError(
                    f"Train dataloader for {self.task_type} task is empty. "
                    f"Please check that tar files exist and contain data for the {self.config.get('force', {}).get('modality', 'unknown')} modality."
                )
            if self.task_type in ('force', 'sliding'):
                tactile_key = getattr(self, '_tactile_key', 'tactile_img')
                if tactile_key not in sample_batch:
                    raise ValueError(
                        f"Force/sliding task requires '{tactile_key}' key in batch. "
                        f"Got: {list(sample_batch.keys())}"
                    )
                sample_data = sample_batch[tactile_key][:1]
            
            # Determine if data is images (5D: [B, T, C, H, W]) or tactile (3D: [B, L, F])
            if len(sample_data.shape) == 5:
                actual_training_type = 'images'
                if model_config.get('type') not in ['vit', 'vision', 'resnet18']:
                    model_config['type'] = 'vit'
            elif len(sample_data.shape) == 3:
                actual_training_type = 'tactile'
                model_config['type'] = 'transformer'
                model_config.pop('num_frames', None)
                model_config.pop('input_size', None)
            else:
                print(f"Warning: Unknown data shape {sample_data.shape}, using config training_type: {self.config.training_type}")
        
        # Handle vision-specific configuration
        if actual_training_type == 'images':
            is_resnet18 = model_config.get('type') == 'resnet18'
            if not is_resnet18 and model_config.get('type') not in ['vit', 'vision']:
                model_config['type'] = 'vit'

            if 'num_frames' not in model_config or model_config.get('num_frames') is None:
                image_chunk_size = None
                if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'image_chunk_size'):
                    image_chunk_size = self.train_dataset.image_chunk_size
                if image_chunk_size is None:
                    image_chunk_size = self.config.get_attr('data_image_chunk_size')

                if image_chunk_size is not None:
                    num_frames = image_chunk_size
                elif self.task_type in ('force', 'sliding') and getattr(self, 'modality', None) == 'gsmini':
                    num_frames = 2
                elif self.config.dataset_type == 'xela_9dtact':
                    num_frames = model_config.get('9dtact_num_frames', 2)
                elif self.config.dataset_type == 'tacniq_gsmini':
                    num_frames = model_config.get('gsmini_num_frames', 2)
                else:
                    num_frames = 1
                    print(f"Warning: Unknown dataset_type '{self.config.dataset_type}', using default num_frames=1")
                model_config['num_frames'] = num_frames

            if not is_resnet18:
                model_config.setdefault('input_size', 224)
                model_config.setdefault('patch_size', 16)
                model_config.setdefault('tubelet_size', 1)
        # Handle tactile-specific configuration
        if actual_training_type == 'tactile':
            # Get encoder type from config
            encoder_type = model_config.get('type', 'transformer')
            
            # For force/pose tasks, infer tactile_dim from actual data (since dataloader can use different modalities)
            if 'tactile_dim' not in model_config or model_config.get('tactile_dim') is None:
                if self.task_type in ('force', 'sliding') and len(sample_data.shape) == 3:
                    tactile_dim = sample_data.shape[-1]
                elif self.config.dataset_type == 'xela_9dtact':
                    tactile_dim = model_config.get('xela_dim', 72)
                elif self.config.dataset_type == 'tacniq_gsmini':
                    tactile_dim = model_config.get('tac_dim', 66)
                else:
                    tactile_dim = model_config.get('xela_dim', 72)
                    print(f"Warning: Unknown dataset_type '{self.config.dataset_type}', using default tactile_dim={tactile_dim}")
                model_config['tactile_dim'] = tactile_dim

            if self.apply_fft:
                if encoder_type == 'transformer':
                    orig = model_config.get('tactile_dim')
                    if orig:
                        model_config['tactile_dim'] = orig * 2
                elif encoder_type == 'transformer3d':
                    orig_in = model_config.get('input_dim')
                    orig_td = model_config.get('tactile_dim')
                    if orig_in:
                        model_config['input_dim'] = orig_in * 2
                    if orig_td:
                        model_config['tactile_dim'] = orig_td * 2
                else:
                    print(f"Warning: FFT enabled but model type '{encoder_type}' not recognized.")
        
        # Create model using unified create_model function
        if self.task_type in ('force', 'sliding'):
            # Force: DualForceHead or scalar MLP. Sliding: MLPHead(embed_dim, num_classes).
            encoder = create_model(model_config)
            encoder = encoder.to(self.device)
            
            # Get encoder embed dimension by forward pass
            sample_batch = next(iter(self.train_loader))
            tactile_key = getattr(self, '_tactile_key', 'tactile_img')
            if tactile_key not in sample_batch:
                raise ValueError(
                    f"Force/sliding task requires '{tactile_key}' key in batch. "
                    f"Got: {list(sample_batch.keys())}"
                )
            x_sample = sample_batch[tactile_key][:1].to(self.device)
            
            with torch.no_grad():
                encoder_out = encoder(x_sample)
                if isinstance(encoder_out, tuple):
                    encoder_out = encoder_out[0]
                
                # Handle sequence output (average if needed)
                if len(encoder_out.shape) == 3:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = True
                else:
                    embed_dim = encoder_out.shape[-1]
                    encoder_output_is_sequence = False
            
            from model.head import DualForceHead, MLPHead
            if self.task_type == 'sliding':
                mlp_head = MLPHead(embed_dim, self.num_classes, hidden_dim=256, dropout=0.1).to(self.device)
            elif self.output_dim in (3, 6):
                mlp_head = DualForceHead(embed_dim, hidden_dim=256, dropout=0.1).to(self.device)
            else:
                mlp_head = MLPHead(embed_dim, self.output_dim, hidden_dim=256, dropout=0.1).to(self.device)
            
            # Create wrapper model
            class EncoderMLPWrapper(nn.Module):
                def __init__(self, encoder, mlp_head, encoder_output_is_sequence):
                    super().__init__()
                    self.encoder = encoder
                    self.mlp_head = mlp_head
                    self.encoder_output_is_sequence = encoder_output_is_sequence
                
                def forward(self, x):
                    encoder_out = self.encoder(x)
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]
                    if self.encoder_output_is_sequence:
                        encoder_out = torch.mean(encoder_out, dim=1)
                    return self.mlp_head(encoder_out)
            
            self.model = EncoderMLPWrapper(encoder, mlp_head, encoder_output_is_sequence)
            self.encoder_output_is_sequence = encoder_output_is_sequence
            
        else:
            # Classification: Create encoder + mlp_head
            self.model = create_model(model_config)
            self.model = self.model.to(self.device)
            self.encoder_output_is_sequence = False  # Will be determined if needed
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Set compilation flag
        self.compile_model = self.config.get('training', {}).get('compile_model', False)
        
        print(f"Model: {model_config.get('type', type(self.model).__name__)} | params: {trainable_params:,} trainable / {total_params:,} total")

        if self.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Loss function - use shared function. For the imbalanced sliding task,
        # plug in inverse-frequency class weights computed from cached labels.
        sliding_class_weights = None
        if self.task_type == 'sliding':
            try:
                from utils.sliding_labels import compute_sliding_class_weights
                force_cfg = getattr(self.config, 'force', None) or getattr(self.config, 'training', {})
                labeled_root = (
                    getattr(force_cfg, 'labeled_data_root', None)
                    if hasattr(force_cfg, 'labeled_data_root') else None
                )
                if labeled_root is None:
                    # Per-modality default: released slip split labels. Don't trust
                    # force_cfg.data_root — it may be stale from spl.yaml.
                    from data.dataset_paths import slip_labeled_root
                    labeled_root = slip_labeled_root(self.modality)
                sliding_class_weights = compute_sliding_class_weights(
                    labeled_root, num_classes=int(getattr(self, 'num_classes', 3))
                )
                print(f"Sliding class weights (from {labeled_root}): {sliding_class_weights.tolist()}")
            except Exception as e:
                print(f"Warning: could not compute sliding class weights ({e}); using uniform weights")
                sliding_class_weights = None

        self.criterion = create_loss_function(self.task_type, class_weights=sliding_class_weights)
        if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self.device)
    
    def compute_l2_regularization(self):
        """Compute L2 regularization loss for all model weights."""
        l2_reg = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
    
    def setup_optimizer(self):
        """Setup optimizer, scheduler, AMP, and gradient settings."""

        optimizer_type = self.config.get('algorithm', {}).get('optimizer', {}).get('type', 'Adam')
        learning_rate = float(self.config.get('training', {}).get('learning_rate', 1e-4))
        weight_decay = float(self.config.get('training', {}).get('weight_decay', 1e-4))
        optimizer_kwargs = dict(self.config.get('algorithm', {}).get('optimizer', {}).get('kwargs', {}))

        self.optimizer = self._build_optimizer(
            self.model.named_parameters(),
            optimizer_type=optimizer_type,
            lr=learning_rate,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        )

        # Build scheduler_cfg in the format _build_scheduler expects
        sched_raw = self.config.get('training', {}).get('scheduler', {}) or {}
        steps_per_epoch = self._get_dataloader_length(self.train_loader, "train_loader")
        epochs = int(self.config.get('training', {}).get('num_epochs', 200))
        total_steps = epochs * steps_per_epoch
        scheduler_cfg = {'type': sched_raw.get('type', 'cosine'), 'kwargs': dict(sched_raw)}
        self.lr_scheduler, self.scheduler_per_iter = self._build_scheduler(
            self.optimizer, scheduler_cfg, total_steps, steps_per_epoch
        )
        # Keep self.scheduler as alias used by print_scheduler_info / test code
        self.scheduler = self.lr_scheduler

        self.setup_amp(self.config.get('training', {}).get('use_amp', False), self.device_type)
        self.gradient_accumulation_steps = int(
            self.config.get('training', {}).get('gradient_accumulation_steps', 1)
        )

        clip_grad_value = self.config.get('training', {}).get('gradient_clip_val')
        self.clip_grad = float(clip_grad_value) if clip_grad_value is not None else None

        self.l2_lambda = float(self.config.get('training', {}).get('l2_lambda', 0.0))
        print(f"Optimizer: {type(self.optimizer).__name__}, lr={learning_rate}, wd={weight_decay}, amp={self.use_amp}")
        if self.lr_scheduler:
            print_scheduler_info(self.lr_scheduler)
    
    def setup_augmentation(self):
        """Setup data augmentation if configured."""
        if hasattr(self.config, 'augmentation_enabled') and self.config.augmentation_enabled:
            augmentation_config = self.config.config.get('augmentation', {}).copy()
            if 'params' in augmentation_config and augmentation_config['params']:
                if 'seed' not in augmentation_config['params']:
                    augmentation_config['params'] = augmentation_config['params'].copy()
                    augmentation_config['params']['seed'] = getattr(self.config, 'data_random_seed', 42)
            self.augmentation = create_augmentation_from_config(augmentation_config)
        else:
            self.augmentation = None
    
    def _get_dataset_length(self, dataset, dataset_name="dataset"):
        """Get dataset length using shared util (cached on self)."""
        return get_dataset_length(dataset, dataset_name, cache=self.__dict__)

    def _get_dataloader_length(self, dataloader, loader_name="dataloader"):
        """Get dataloader length using shared util (cached on self)."""
        return get_dataloader_length(dataloader, loader_name, cache=self.__dict__)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        # Set epoch for IterableDataset (e.g. force 4probe) so all workers use same shuffle
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0  # Count batches for averaging
        
        # Force metrics
        train_preds = []
        train_targets = []
        
        # Use correct batch count for tqdm total (IterableDataset DataLoader may report wrong len)
        total_batches = self._get_dataloader_length(self.train_loader, "train_loader")
        step_start = epoch * getattr(self, 'train_iters_per_epoch', total_batches)
        step_end = min(step_start + total_batches, getattr(self, 'max_steps', step_start + total_batches))
        progress_bar = tqdm(self.train_loader, desc=f"Step {step_start+1}-{step_end}", total=total_batches)
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_count += 1
            # Extract data and targets using shared functions
            data = get_batch_data(
                batch, self.task_type, self.device,
                training_type=self.config.training_type,
                apply_fft=self.apply_fft
            )
            targets = get_target(batch, self.task_type, self.device)
            
            # Apply data augmentation (only during training, and only for single-input tasks)
            if self.augmentation is not None:
                if isinstance(data, tuple):
                    pass  # Tuple format: no augmentation
                else:
                    # Single input: apply augmentation
                    data = self.augmentation(data, training=True)
            
            # Forward pass using shared function
            if self.use_amp:
                with autocast(self.device_type):
                    outputs = forward_pass(self.model, data, self.task_type, self.output_dim)
                    outputs = postprocess_outputs(outputs, self.task_type, self.output_dim)
                    if self.task_type == 'force' and self.output_dim in (3, 6):
                        loss = compute_force_dual_head_loss(outputs, targets, self.criterion)
                    else:
                        loss = compute_loss(outputs, targets, self.criterion, self.task_type)
                    # Add L2 regularization
                    if self.l2_lambda > 0:
                        l2_loss = self.compute_l2_regularization()
                        loss = loss + l2_loss
                    # Capture unscaled loss inside autocast context (convert to float32 for stability)
                    unscaled_loss_value = float(loss.detach().cpu().item())
            else:
                outputs = forward_pass(self.model, data, self.task_type, self.output_dim)
                outputs = postprocess_outputs(outputs, self.task_type, self.output_dim)
                if self.task_type == 'force' and self.output_dim in (3, 6):
                    loss = compute_force_dual_head_loss(outputs, targets, self.criterion)
                else:
                    loss = compute_loss(outputs, targets, self.criterion, self.task_type)
                # Add L2 regularization
                if self.l2_lambda > 0:
                    l2_reg = self.compute_l2_regularization()
                    loss = loss + l2_reg
                # Capture unscaled loss for statistics BEFORE scaling
                unscaled_loss_value = float(loss.detach().item())
            
            self.grad_update_step(
                loss, self.model.parameters(), batch_idx,
                clip_val=self.clip_grad or 0.0,
                step_scheduler=self.scheduler_per_iter,
            )
            
            # Statistics (use the unscaled loss value captured before scaling)
            total_loss += unscaled_loss_value
            
            if self.task_type == 'force':
                # Store predictions and targets for metrics
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(targets.detach().cpu().numpy())
                
                # Calculate MAE for progress bar
                if train_preds:
                    preds_array = np.concatenate(train_preds)
                    targets_array = np.concatenate(train_targets)
                    metrics = calculate_force_metrics_detailed(
                        preds_array, targets_array,
                        target_mean=self.force_mean, target_std=self.force_std,
                        output_dim=self.output_dim
                    )
                    current_mae = metrics['mae']
                else:
                    current_mae = 0.0
                
                progress_bar.set_postfix({
                    'Loss': f'{unscaled_loss_value:.4f}',
                    'MAE': f'{current_mae:.4f}'
                })
            else:
                # Classification
                # Ensure outputs are 2D [batch_size, num_classes]
                if outputs.dim() == 3:
                    # If 3D [batch_size, seq_len, num_classes], average over sequence
                    outputs = outputs.mean(dim=1)  # [batch_size, num_classes]
                elif outputs.dim() > 2:
                    # Reshape to 2D
                    outputs = outputs.view(outputs.size(0), -1)
                
                # Ensure targets are 1D [batch_size]
                if targets.dim() > 1:
                    targets = targets.squeeze()
                
                # Get predicted class indices
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update progress bar
                current_accuracy = 100. * correct / total if total > 0 else 0.0
                progress_bar.set_postfix({
                    'Loss': f'{unscaled_loss_value:.4f}',
                    'Acc': f'{current_accuracy:.2f}%'
                })
            
            # Increment global step (approximate: one step per batch)
            if hasattr(self, "global_step"):
                self.global_step += 1
                global_step = self.global_step
            else:
                # Fallback: epoch-based global step
                num_iterations_per_epoch = self._get_dataloader_length(self.train_loader, "train_loader")
                global_step = epoch * max(num_iterations_per_epoch, 1) + batch_idx

            # Log to TensorBoard
            if batch_idx % self.config.get('logging', {}).get('log_interval', 10) == 0:
                self.writer.add_scalar('Train/Loss', unscaled_loss_value, global_step)
                if self.task_type == 'force':
                    if train_preds:
                        preds_array = np.concatenate(train_preds)
                        targets_array = np.concatenate(train_targets)
                        metrics = calculate_force_metrics_detailed(
                            preds_array, targets_array,
                            target_mean=self.force_mean, target_std=self.force_std,
                            output_dim=self.output_dim
                        )
                        self.writer.add_scalar('Train/MAE', metrics['mae'], global_step)
                        self.writer.add_scalar('Train/RMSE', metrics['rmse'], global_step)
                else:
                    # Classification
                    current_accuracy = 100. * correct / total if total > 0 else 0.0
                    self.writer.add_scalar('Train/Accuracy', current_accuracy, global_step)
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Train/LearningRate', current_lr, global_step)

            # Step-based validation and early stopping
            if (
                hasattr(self, "eval_steps")
                and self.eval_steps is not None
                and self.eval_steps > 0
                and hasattr(self, "val_loader")
                and self.val_loader is not None
                and hasattr(self, "global_step")
                and self.global_step > 0
                and self.global_step % self.eval_steps == 0
            ):
                # Run validation
                if self.task_type == 'force':
                    val_loss, val_mae, val_rmse, _, _, val_normal_mae, val_shear_mae = self.evaluate(epoch, self.val_loader, 'val')
                    val_accuracy = None
                else:
                    # Classification: returns (loss, accuracy, predictions, targets)
                    eval_result = self.evaluate(epoch, self.val_loader, 'val')
                    val_loss, val_accuracy, _, _ = eval_result
                    val_mae = val_rmse = None

                # Check for best model (based on validation loss)
                is_best_loss = val_loss < self.best_val_loss
                if self.task_type == 'force':
                    is_best_metric = val_mae < self.best_mae if val_mae is not None else False
                    if is_best_metric:
                        self.best_mae = val_mae
                        if val_normal_mae is not None:
                            self.best_normal_mae = val_normal_mae
                        if val_shear_mae is not None:
                            self.best_shear_mae = val_shear_mae
                else:
                    is_best_metric = val_accuracy > self.best_accuracy if val_accuracy is not None else False
                    if is_best_metric:
                        self.best_accuracy = val_accuracy

                if is_best_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_step = self.global_step
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save checkpoint based on validation loss
                if self.task_type == 'force':
                    metric_value = val_mae
                else:
                    metric_value = val_accuracy
                self.save_checkpoint(epoch, metric_value, is_best_loss)

                # Print validation results with step-based info
                step_str = f"Step {self.global_step:6d}/{self.max_steps if hasattr(self, 'max_steps') else self.global_step}"
                if self.task_type == 'force':
                    print(
                        f"{step_str}: Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f} "
                        f"(Best Val Loss: {self.best_val_loss:.4f} @ Step {self.best_step}, Patience evals: {self.patience_counter}/{getattr(self, 'patience', 5)})"
                    )
                else:
                    print(
                        f"{step_str}: Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% "
                        f"(Best Val Loss: {self.best_val_loss:.4f} @ Step {self.best_step}, Patience evals: {self.patience_counter}/{getattr(self, 'patience', 5)})"
                    )

                # Early stopping (based on validation loss)
                if self.patience_counter >= getattr(self, "patience", 5):
                    print(f"Early stopping triggered: validation loss did not improve for {getattr(self, 'patience', 5)} validation check(s)")
                    print(f"Best validation loss: {self.best_val_loss:.4f}")
                    self.early_stop = True
                    break
        
        # Use actual batch count if available, otherwise use cached length
        if batch_count == 0:
            num_iterations_per_epoch = self._get_dataloader_length(self.train_loader, "train_loader")
            if num_iterations_per_epoch == 0:
                num_iterations_per_epoch = 1  # Prevent division by zero
        else:
            num_iterations_per_epoch = batch_count
        
        avg_loss = total_loss / num_iterations_per_epoch
        
        if self.task_type == 'force':
            # Calculate force metrics
            if train_preds:
                preds_array = np.concatenate(train_preds)
                targets_array = np.concatenate(train_targets)
                metrics = calculate_force_metrics_detailed(
                    preds_array, targets_array,
                    target_mean=self.force_mean, target_std=self.force_std,
                    output_dim=self.output_dim
                )
                mae, rmse = metrics['mae'], metrics['rmse']
            else:
                mae, rmse = 0.0, 0.0
            return avg_loss, mae, rmse
        else:
            accuracy = 100. * correct / total if total > 0 else 0.0
            return avg_loss, accuracy
    
    def evaluate(self, epoch, loader, split='val'):
        """Evaluate the model."""
        self.model.eval()

        def get_batch_data_fn(batch):
            return get_batch_data(
                batch, self.task_type, self.device,
                training_type=self.config.training_type,
                apply_fft=self.apply_fft
            )

        def get_target_fn(batch):
            return get_target(batch, self.task_type, self.device)

        def forward_fn(data, targets):
            outputs = forward_pass(self.model, data, self.task_type, self.output_dim)
            return postprocess_outputs(outputs, self.task_type, self.output_dim)

        l2_reg_fn = self.compute_l2_regularization if self.l2_lambda > 0 else None

        avg_loss, metrics_dict, all_predictions, all_targets = run_evaluation_loop(
            loader=loader,
            forward_fn=forward_fn,
            get_batch_data_fn=get_batch_data_fn,
            get_target_fn=get_target_fn,
            criterion=self.criterion,
            task_type=self.task_type,
            output_dim=self.output_dim,
            force_dual_head=(self.task_type == 'force' and self.output_dim in (3, 6)),
            target_mean=self.force_mean if self.task_type == 'force' else None,
            target_std=self.force_std if self.task_type == 'force' else None,
            use_amp=self.use_amp,
            device_type=self.device_type,
            l2_reg_fn=l2_reg_fn,
            desc=f"Evaluating {split}",
            skip_batch_on_error=False,
        )

        # Log to TensorBoard (if writer is available)
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.add_scalar(f'{split.title()}/Loss', avg_loss, epoch)
            if self.task_type == 'force':
                mae = metrics_dict['mae']
                rmse = metrics_dict['rmse']
                self.writer.add_scalar(f'{split.title()}/MAE', mae, epoch)
                self.writer.add_scalar(f'{split.title()}/RMSE', rmse, epoch)
                if metrics_dict.get('normal_mae') is not None:
                    self.writer.add_scalar(f'{split.title()}/Normal_Force_MAE', metrics_dict['normal_mae'], epoch)
                    self.writer.add_scalar(f'{split.title()}/Shear_Force_MAE', metrics_dict['shear_mae'], epoch)
            else:
                self.writer.add_scalar(f'{split.title()}/Accuracy', metrics_dict['accuracy'], epoch)

        if self.task_type == 'force':
            return avg_loss, metrics_dict['mae'], metrics_dict['rmse'], all_predictions, all_targets, metrics_dict.get('normal_mae'), metrics_dict.get('shear_mae')
        else:
            return avg_loss, metrics_dict['accuracy'], all_predictions, all_targets
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Save model checkpoint."""
        self.current_epoch = epoch
        checkpoint = self._base_checkpoint(self.model.state_dict())
        checkpoint.update({
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'best_val_loss': self.best_val_loss,
            'config_path': self.config.config_path,
            'model_config_path': self.config.model_config_path,
            'algo_config_path': self.config.algo_config_path,
        })
        filename = f'checkpoint_epoch_{epoch}.pth'
        self._save_to_disk(checkpoint, self.run_save_dir, filename, is_best=is_best, config=self.config.config)
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")

        # Base epoch/early-stopping config
        base_epochs = int(self.config.get('training', {}).get('num_epochs', 200))
        patience = int(self.config.get('training', {}).get('patience', 5))
        save_interval = self.config.get('logging', {}).get('save_interval', 10)

        # Step-based training configuration
        max_steps = int(self.config.get('training', {}).get('max_steps', 10_000))
        eval_steps = int(self.config.get('training', {}).get('eval_steps', 200))

        # Estimate number of iterations per epoch
        train_iters_per_epoch = self._get_dataloader_length(self.train_loader, "train_loader")
        if train_iters_per_epoch <= 0:
            train_iters_per_epoch = 1

        # Derive total epochs and validation interval from step targets
        epochs = min(base_epochs, int(math.ceil(max_steps / train_iters_per_epoch)))
        eval_interval = max(1, int(round(eval_steps / train_iters_per_epoch)))

        # Global step counter and step-based control
        self.global_step = 0
        self.max_steps = max_steps
        self.patience = patience
        self.eval_steps = eval_steps

        self.train_iters_per_epoch = train_iters_per_epoch
        print(f"Training: {max_steps} steps ({train_iters_per_epoch}/epoch), val every {eval_steps} steps, patience={patience}")
        
        for epoch in range(epochs):
            if hasattr(self, "max_steps") and hasattr(self, "global_step") and self.global_step >= self.max_steps:
                print(f"Reached target training steps ({self.global_step} >= {self.max_steps}). Stopping training.")
                break
            if hasattr(self, "early_stop") and self.early_stop:
                print("Early stopping flag is set. Stopping training loop.")
                break

            self.current_epoch = epoch
            
            # Training
            if self.task_type == 'force':
                train_loss, train_mae, train_rmse = self.train_epoch(epoch)
                train_accuracy = None
                train_trans_mae = train_trans_rmse = train_rot_mae = train_rot_rmse = None
            else:
                train_loss, train_accuracy = self.train_epoch(epoch)
                train_mae = train_rmse = None
                train_trans_mae = train_trans_rmse = train_rot_mae = train_rot_rmse = None
            
            # Validation (based on frequency)
            val_loss, val_accuracy, val_mae, val_rmse = None, None, None, None
            val_trans_mae = val_trans_rmse = val_rot_mae = val_rot_rmse = None
            use_step_based_eval = hasattr(self, "eval_steps") and self.eval_steps is not None and self.eval_steps > 0
            if self.val_loader is not None and (not use_step_based_eval) and (epoch % eval_interval == 0 or epoch == epochs - 1):
                if self.task_type == 'force':
                    val_loss, val_mae, val_rmse, _, _, val_normal_mae, val_shear_mae = self.evaluate(epoch, self.val_loader, 'val')
                    val_accuracy = None
                else:
                    # Classification: returns (loss, accuracy, predictions, targets)
                    eval_result = self.evaluate(epoch, self.val_loader, 'val')
                    val_loss, val_accuracy, _, _ = eval_result
                    val_mae = None
                    val_rmse = None
                    val_trans_mae = val_trans_rmse = val_rot_mae = val_rot_rmse = None
                
                # Check for best model (based on validation loss)
                is_best_loss = val_loss < self.best_val_loss
                if self.task_type == 'force':
                    is_best_metric = val_mae < self.best_mae if val_mae is not None else False
                    if is_best_metric:
                        self.best_mae = val_mae
                        if val_normal_mae is not None:
                            self.best_normal_mae = val_normal_mae
                        if val_shear_mae is not None:
                            self.best_shear_mae = val_shear_mae
                else:
                    is_best_metric = val_accuracy > self.best_accuracy if val_accuracy is not None else False
                    if is_best_metric:
                        self.best_accuracy = val_accuracy
                
                if is_best_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_step = self.global_step
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint based on validation loss
                if self.task_type == 'force':
                    metric_value = val_mae
                else:
                    metric_value = val_accuracy
                if epoch % save_interval == 0 or is_best_loss or epoch == epochs - 1:
                    self.save_checkpoint(epoch, metric_value, is_best_loss)
                
                # Print validation results
                current_step = getattr(self, "global_step", 0)
                if self.task_type == 'force':
                    print(f"Step {current_step:6d}/{self.max_steps if hasattr(self, 'max_steps') else current_step}: "
                          f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f} "
                          f"(Best Val Loss: {self.best_val_loss:.4f} @ Step {self.best_step}, Patience evals: {self.patience_counter}/{patience})")
                else:
                    # Classification
                    print(f"Step {current_step:6d}/{self.max_steps if hasattr(self, 'max_steps') else current_step}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% "
                          f"(Best Val Loss: {self.best_val_loss:.4f} @ Step {self.best_step}, Patience evals: {self.patience_counter}/{patience})")
                
                # Early stopping (based on validation loss)
                if self.patience_counter >= patience:
                    print(f"Early stopping triggered: validation loss did not improve for {patience} validation check(s)")
                    print(f"Best validation loss: {self.best_val_loss:.4f}")
                    break
            else:
                # No validation or not validation epoch, save checkpoint periodically
                if self.task_type == 'force':
                    metric_value = train_mae
                else:
                    metric_value = train_accuracy
                if epoch % save_interval == 0 or epoch == epochs - 1:
                    self.save_checkpoint(epoch, metric_value, False)
                
                step_str = f"Step {getattr(self, 'global_step', 0):6d}/{self.max_steps if hasattr(self, 'max_steps') else getattr(self, 'global_step', 0)}"
                if self.task_type == 'force':
                    if val_loss is not None and val_mae is not None:
                        print(
                            f"{step_str}: Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, "
                            f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f} "
                            f"(Best Val Loss: {self.best_val_loss:.4f} @ Step {self.best_step}, Patience evals: {self.patience_counter}/{patience})"
                        )
                    else:
                        print(
                            f"{step_str}: Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}"
                        )
                else:
                    if val_loss is not None and val_accuracy is not None:
                        print(
                            f"{step_str}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% "
                            f"(Best Val Loss: {self.best_val_loss:.4f} @ Step {self.best_step}, Patience evals: {self.patience_counter}/{patience})"
                        )
                    else:
                        print(
                            f"{step_str}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
                        )
            
            # Update scheduler and log learning rate
            if self.lr_scheduler and not self.scheduler_per_iter:
                self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LearningRate', current_lr, self.global_step)
        
        # Final test evaluation
        print(f"\n{'='*60}")
        print("FINAL TEST EVALUATION")
        print(f"{'='*60}")
        
        # Load best model if available
        best_model_path = os.path.join(self.run_save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("Loading best model for final evaluation...")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch'))
            best_val_loss = checkpoint.get('best_val_loss', 'N/A')
            best_step = checkpoint.get('best_step', (checkpoint.get('best_epoch', 0) * checkpoint.get('train_iters_per_epoch', 1)))
            print(f"Loaded best model from step {best_step} (best val loss: {best_val_loss:.4f})")
        
        if self.task_type == 'force':
            test_loss, test_mae, test_rmse, test_predictions, test_targets, test_normal_mae, test_shear_mae = self.evaluate(
                self.current_epoch, self.test_loader, 'test'
            )
            test_accuracy = None
            test_trans_mae = test_trans_rmse = test_rot_mae = test_rot_rmse = None
            
            print(f"Final Test Results:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  MAE: {test_mae:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
            if test_normal_mae is not None and test_shear_mae is not None:
                print(f"  Normal Force MAE: {test_normal_mae:.4f}")
                print(f"  Shear Force MAE: {test_shear_mae:.4f}")
            print(f"  Best validation loss: {self.best_val_loss:.4f} @ Step {self.best_step}")
            print(f"  Best validation MAE: {self.best_mae:.4f}")
            if self.best_normal_mae is not None and self.best_shear_mae is not None:
                print(f"  Best validation Normal Force MAE: {self.best_normal_mae:.4f}")
                print(f"  Best validation Shear Force MAE: {self.best_shear_mae:.4f}")
        else:
            # Classification or sliding: returns (loss, accuracy, predictions, targets)
            eval_result = self.evaluate(self.current_epoch, self.test_loader, 'test')
            test_loss, test_accuracy, test_predictions, test_targets = eval_result
            test_rmse = None
            
            print(f"Final Test Results:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Accuracy: {test_accuracy:.2f}%")
            print(f"  Best validation loss: {self.best_val_loss:.4f} @ Step {self.best_step}")
            print(f"  Best validation accuracy: {self.best_accuracy:.2f}%")
            
            if hasattr(self.train_dataset, 'object_to_idx'):
                class_names = list(self.train_dataset.object_to_idx.keys())
                # Get all class indices (0 to num_classes-1) to ensure all classes are included
                all_labels = list(range(len(class_names)))
                report = classification_report(test_targets, test_predictions, 
                                              labels=all_labels,
                                              target_names=class_names,
                                              zero_division=0)
                cm = confusion_matrix(test_targets, test_predictions, labels=all_labels)
                class_names_cm = class_names
            else:
                # Get unique labels from data
                unique_labels = np.unique(np.concatenate([test_targets, test_predictions]))
                unique_labels = np.sort(unique_labels)
                class_names = [f"Class_{i}" for i in unique_labels]
                report = classification_report(test_targets, test_predictions,
                                              labels=unique_labels.tolist(),
                                              target_names=class_names,
                                              zero_division=0)
                cm = confusion_matrix(test_targets, test_predictions, labels=unique_labels)
                class_names_cm = class_names
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names_cm, yticklabels=class_names_cm)
            plt.title('Confusion Matrix - Final Test Results')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            cm_path = os.path.join(self.run_save_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")
        
        # Save comprehensive final results
        final_results = {
            'task_type': self.task_type,
            'test_results': {
                'test_loss': test_loss,
                'best_validation_loss': self.best_val_loss,
                'best_validation_loss_step': self.best_step,
                'best_validation_loss_epoch': self.best_epoch,
                'total_steps': self.global_step,
                'early_stopped': self.patience_counter >= int(self.config.get('training', {}).get('patience', 7))
            },
            'training_config': {
                'training_type': self.config.training_type,
                'dataset_type': self.config.dataset_type,
                'num_classes': self.num_classes,
                'model_type': type(self.model).__name__,
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'batch_size': getattr(self.config, 'data_batch_size', 32),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
                'scheduler_type': type(self.scheduler).__name__,
                'use_amp': self.use_amp,
                'compile_model': self.compile_model,
                'clip_grad': self.clip_grad
            },
            'data_info': {
                'train_samples': self._get_dataset_length(self.train_dataset, 'train_dataset'),
                'val_samples': self._get_dataset_length(self.val_dataset, 'val_dataset') if hasattr(self, 'val_dataset') and self.val_dataset is not None else 0,
                'test_samples': self._get_dataset_length(self.test_dataset, 'test_dataset')
            },
            'config_files': {
                'main_config': self.config.config_path,
                'model_config': self.config.model_config_path,
                'algo_config': self.config.algo_config_path
            },
            'paths': {
                'log_dir': self.run_log_dir,
                'save_dir': self.run_save_dir
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add task-specific results
        if self.task_type == 'force':
            final_results['test_results']['test_mae'] = test_mae
            final_results['test_results']['test_rmse'] = test_rmse
            final_results['test_results']['best_validation_mae'] = self.best_mae
            if test_normal_mae is not None and test_shear_mae is not None:
                final_results['test_results']['test_normal_force_mae'] = test_normal_mae
                final_results['test_results']['test_shear_force_mae'] = test_shear_mae
                final_results['test_results']['best_validation_normal_force_mae'] = self.best_normal_mae
                final_results['test_results']['best_validation_shear_force_mae'] = self.best_shear_mae
            final_results['test_predictions'] = test_predictions.tolist() if isinstance(test_predictions, np.ndarray) else test_predictions
            final_results['test_targets'] = test_targets.tolist() if isinstance(test_targets, np.ndarray) else test_targets
        else:
            final_results['test_results']['test_accuracy'] = test_accuracy
            final_results['test_results']['best_validation_accuracy'] = self.best_accuracy
            final_results['test_predictions'] = test_predictions.tolist() if isinstance(test_predictions, np.ndarray) else test_predictions
            final_results['test_targets'] = test_targets.tolist() if isinstance(test_targets, np.ndarray) else test_targets
            final_results['classification_report'] = report
            final_results['confusion_matrix'] = cm.tolist()
            final_results['class_names'] = class_names_cm  # Always defined now
            final_results['paths']['confusion_matrix'] = cm_path
        
        results_path = os.path.join(self.run_save_dir, 'final_results.yaml')
        with open(results_path, 'w') as f:
            yaml.dump(final_results, f, default_flow_style=False)
        
        print(f"\nTraining completed! Results saved to: {self.run_save_dir}")
        print(f"  - Final results: {results_path}")
        if self.task_type in ('classification', 'sliding'):
            print(f"  - Confusion matrix: {cm_path}")
        print(f"  - Best model: {best_model_path}")
        print(f"  - TensorBoard logs: {self.run_log_dir}")
        self.writer.close()
        
        # Return results dictionary for multi-seed runs
        if self.task_type == 'force':
            out = {
                'task_type': self.task_type,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_loss': test_loss,
                'best_val_mae': self.best_mae,
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch,
                'best_step': self.best_step,
                'total_steps': self.global_step,
                'train_iters_per_epoch': getattr(self, 'train_iters_per_epoch', 1),
                'early_stopped': self.patience_counter >= int(self.config.get('training', {}).get('patience', 7))
            }
            if test_normal_mae is not None and test_shear_mae is not None:
                out['test_normal_force_mae'] = test_normal_mae
                out['test_shear_force_mae'] = test_shear_mae
                out['best_val_normal_force_mae'] = self.best_normal_mae
                out['best_val_shear_force_mae'] = self.best_shear_mae
            return out
        else:
            # Classification
            result = {
                'task_type': self.task_type,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'best_val_accuracy': self.best_accuracy,
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch,
                'best_step': self.best_step,
                'total_steps': self.global_step,
                'train_iters_per_epoch': getattr(self, 'train_iters_per_epoch', 1),
                'early_stopped': self.patience_counter >= int(self.config.get('training', {}).get('patience', 7))
            }
            return result


def evaluate_checkpoint(checkpoint_path: str, split: str = 'test', device: str = 'auto'):
    """
    Evaluate a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        split: Which split to evaluate on ('train', 'val', or 'test'). Default is 'test'
        device: Device to use ('cuda', 'cpu', or 'auto'). Default is 'auto'
    """
    print(f"\n{'='*60}")
    print("CHECKPOINT EVALUATION")
    print(f"{'='*60}")
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract metadata
    config_path = checkpoint.get('config_path')
    model_config_path = checkpoint.get('model_config_path')
    
    if config_path is None:
        raise ValueError("Checkpoint does not contain config_path. Cannot reconstruct configuration.")
    
    print(f"Config path: {config_path}")
    if model_config_path:
        print(f"Model config path: {model_config_path}")
    
    # Reconstruct configuration
    # Try to infer algo_config_path from checkpoint or use default
    algo_config_path = checkpoint.get('algo_config_path', 'config/algo/supervised.yaml')
    config = UnifiedConfig(config_path, model_config_path, algo_config_path)
    
    print("\n" + "="*60)
    print("LOADING MODEL AND DATA")
    print("="*60)
    
    # Setup data (this will create new dataloaders)
    trainer = UnifiedTrainer(config)
    
    # Load model weights
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model = trainer.model.to(device)
    
    ckpt_step = checkpoint.get('best_step', checkpoint.get('epoch', 'unknown'))
    print(f"Loaded model from step {ckpt_step}")
    print(f"Checkpoint accuracy: {checkpoint.get('accuracy', 'unknown'):.2f}%")
    
    # Select dataloader based on split
    if split == 'train':
        loader = trainer.train_loader
        print(f"Evaluating on training set ({trainer._get_dataset_length(trainer.train_dataset, 'train_dataset')} samples)")
    elif split == 'val':
        if trainer.val_loader is None:
            print("Warning: No validation set available. Using test set instead.")
            loader = trainer.test_loader
            split = 'test'
        else:
            loader = trainer.val_loader
            print(f"Evaluating on validation set ({trainer._get_dataset_length(trainer.val_dataset, 'val_dataset')} samples)")
    else:  # test
        loader = trainer.test_loader
        print(f"Evaluating on test set ({trainer._get_dataset_length(trainer.test_dataset, 'test_dataset')} samples)")
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    eval_loss, eval_accuracy, predictions, targets = trainer.evaluate(
        checkpoint.get('epoch', 0), loader, split
    )
    
    print(f"\n{split.upper()} Results:")
    print(f"  Loss: {eval_loss:.4f}")
    print(f"  Accuracy: {eval_accuracy:.2f}%")
    
    # Generate classification report
    if hasattr(trainer.train_dataset, 'object_to_idx'):
        class_names = list(trainer.train_dataset.object_to_idx.keys())
        # Get all class indices (0 to num_classes-1) to ensure all classes are included
        all_labels = list(range(len(class_names)))
    else:
        class_names = [f"Class_{i}" for i in range(trainer.num_classes)]
        all_labels = list(range(trainer.num_classes))
    
    print(f"\nClassification Report:")
    report = classification_report(targets, predictions, 
                                   labels=all_labels,
                                   target_names=class_names,
                                   zero_division=0)
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(targets, predictions, labels=all_labels)
    
    # Create output directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    output_dir = os.path.join(checkpoint_dir, f"evaluation_{split}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {split.upper()} Results')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Save evaluation results
    eval_results = {
        'checkpoint_path': checkpoint_path,
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
        'checkpoint_accuracy': checkpoint.get('accuracy', 'unknown'),
        'split': split,
        'evaluation_results': {
            'loss': eval_loss,
            'accuracy': eval_accuracy
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'predictions': predictions,
        'targets': targets,
        'model_info': {
            'model_type': checkpoint.get('model_type'),
            'total_params': checkpoint.get('total_params'),
            'trainable_params': checkpoint.get('trainable_params')
        },
        'training_info': {
            'training_type': checkpoint.get('training_type'),
            'dataset_type': checkpoint.get('dataset_type'),
            'num_classes': checkpoint.get('num_classes'),
            'batch_size': checkpoint.get('batch_size')
        },
        'config_files': {
            'main_config': config_path,
            'model_config': model_config_path,
            'algo_config': algo_config_path
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(output_dir, f'evaluation_results_{split}.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(eval_results, f, default_flow_style=False)
    
    print(f"Evaluation results saved to: {results_path}")
    
    return eval_loss, eval_accuracy, predictions, targets


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Unified Classification Training and Evaluation')
    parser.add_argument('--algo_config', type=str, required=False, default="config/algo/spl.yaml",
                       help='Path to algorithm configuration file (e.g., spl.yaml)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')

    parser.add_argument('--modality', type=str, default='gsmini',
                       choices=['9dtact', 'gsmini', 'xela', 'tac02'],
                       help='Modality to use (auto-selects sensor and model configs, overrides modality in spl.yaml)')

    parser.add_argument('--task_type', type=str, default='force',
                       choices=['force', 'classification', 'sliding'],
                       help='Task type: classification, force, or sliding (bracket labels on gsmini)')
    
    # Training arguments
    parser.add_argument('--train_data_percentage', type=float, default=1.0,
                       help='Percentage of training data to use (0.0 to 1.0). Overrides config file value.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10,20,30],
                       help='List of random seeds to run. If not provided, uses seed from config (single run).')
    
    # Evaluation arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file for evaluation mode')
    parser.add_argument('--eval_split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate on (train, val, or test). Default is test.')
    
    args = parser.parse_args()
    
    # Automatic config selection based on modality
    
    # Mapping of modalities to sensor configs (gsmini = GsMini/GelSight sensor)
    sensor_configs = {
        '9dtact': 'config/sensor/9dtact_config.yaml',
        'gsmini': 'config/sensor/gsmini_config.yaml',
        'xela': 'config/sensor/xela_config.yaml',
        'tac02': 'config/sensor/tac_config.yaml'
    }
    
    # Mapping of modalities to model configs
    # Image modalities (9dtact, gsmini) use vision transformer (vit.yaml)
    # Tactile modalities: xela uses standard transformer, tac02 uses Conv2D transformer
    model_configs = {
        '9dtact': 'config/model/vit.yaml',      # Image modality -> vision transformer
        'gsmini': 'config/model/vit.yaml',      # GsMini/GelSight (classification + force)
        'xela': 'config/model/taxel_tf.yaml',   # Tactile modality -> standard tactile transformer
        'tac02': 'config/model/taxel_tf.yaml'    # Tactile modality -> Conv2D tactile transformer (for tacniq data)
    }
    
    # Auto-select sensor config and model config based on modality
    if args.modality not in sensor_configs:
        raise ValueError(f"Unknown modality: {args.modality}. Cannot auto-select sensor config.")
    if args.modality not in model_configs:
        raise ValueError(f"Unknown modality: {args.modality}. Cannot auto-select model config.")
    
    config_path = sensor_configs[args.modality]
    model_config_path = model_configs[args.modality]
    
    # Load algo config to check force.backbone and apply modality override
    if os.path.exists(args.algo_config):
        try:
            with open(args.algo_config, 'r') as f:
                algo_config_dict = yaml.safe_load(f) or {}
            
            # Force task: optional backbone override (e.g. resnet18 instead of vit)
            if args.task_type in ('force', 'sliding') and algo_config_dict.get('force', {}).get('backbone') == 'resnet18':
                model_config_path = 'config/model/resnet18.yaml'
                print(f"Force backbone override: using ResNet-18 (force.backbone: resnet18)")
            
            # Override modality in force/pose sections if they exist
            modality_override_applied = False
            if 'force' in algo_config_dict and 'modality' in algo_config_dict['force']:
                algo_config_dict['force']['modality'] = args.modality
                modality_override_applied = True

            if modality_override_applied:
                temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.dump(algo_config_dict, temp_config_file, default_flow_style=False)
                temp_config_file.close()
                args.algo_config = temp_config_file.name
        except Exception as e:
            print(f"Warning: Could not override modality in algo config: {e}")
            print(f"  Continuing with original config. Modality may need to be set manually in {args.algo_config}")
    
    print(f"Configs: sensor={config_path}, model={model_config_path}")
    
    # Check if this is evaluation mode
    if args.checkpoint is not None:
        # Evaluation mode
        print("="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            split=args.eval_split,
            device=args.device
        )
    else:
        # Training mode
        # Set device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        print(f"Using device: {device}")
        
        # Determine seeds to run
        if args.seeds is not None:
            seeds = args.seeds
            print(f"Running multiple seeds: {seeds}")
        else:
            # Single run - use seed from config
            seeds = [None]  # Will use config seed
        
        # Collect results from all seeds
        all_results = []
        
        for seed_idx, seed in enumerate(seeds):
            seed_str = str(seed) if seed is not None else "from config"
            print(f"\n--- Run {seed_idx + 1}/{len(seeds)} (seed={seed_str}) ---")
            
            # Load configuration with overrides
            config = UnifiedConfig(
                config_path, 
                model_config_path, 
                args.algo_config,
                train_data_percentage=args.train_data_percentage,
                random_seed=seed,
                task_type=args.task_type
            )
            
            # Create trainer
            trainer = UnifiedTrainer(config)
            
            # Start training and collect results
            results = trainer.train()
            
            # Add seed to results
            if seed is not None:
                results['seed'] = seed
            else:
                # Get seed from config
                results['seed'] = getattr(config, 'data_random_seed', None)
            
            all_results.append(results)
        
        # If multiple seeds, calculate and report statistics
        if len(seeds) > 1:
            print(f"\n{'='*80}")
            print("MULTI-SEED RESULTS SUMMARY")
            print(f"{'='*80}\n")
            
            # Get task type from first result (all should have same task type)
            task_type = all_results[0].get('task_type', 'classification')
            if task_type is None:
                # Try to infer from available keys
                if 'test_mae' in all_results[0]:
                    task_type = 'force'
                else:
                    task_type = 'classification'
            
            # Extract metrics based on task type
            test_losses = [r['test_loss'] for r in all_results]
            val_losses = [r['best_val_loss'] for r in all_results]
            best_steps = [r.get('best_step', r.get('best_epoch', 0) * r.get('train_iters_per_epoch', 1)) for r in all_results]
            
            if task_type == 'force':
                # Force metrics via shared aggregation
                force_stats = aggregate_force_results(all_results)
                test_loss_mean = np.mean(test_losses)
                test_loss_std = np.std(test_losses)
                val_loss_mean = np.mean(val_losses)
                val_loss_std = np.std(val_losses)
                best_step_mean = np.mean(best_steps)
                best_step_std = np.std(best_steps)

                print(f"Number of runs: {len(seeds)}")
                print(f"Task type: {task_type}")
                print(f"Seeds used: {[r['seed'] for r in all_results]}")
                if args.train_data_percentage is not None:
                    print(f"Train data percentage: {args.train_data_percentage * 100:.1f}%")

                if 'test_mae' in force_stats:
                    s = force_stats['test_mae']
                    print(f"\nTest MAE:")
                    print(f"  Mean: {s['mean']:.4f}")
                    print(f"  Std:  {s['std']:.4f}")
                    print(f"  Min:  {s['min']:.4f}")
                    print(f"  Max:  {s['max']:.4f}")
                    print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")

                if 'test_normal_mae' in force_stats:
                    s = force_stats['test_normal_mae']
                    print(f"\nTest Normal Force MAE:")
                    print(f"  Mean: {s['mean']:.4f}")
                    print(f"  Std:  {s['std']:.4f}")
                    print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")
                if 'test_shear_mae' in force_stats:
                    s = force_stats['test_shear_mae']
                    print(f"\nTest Shear Force MAE:")
                    print(f"  Mean: {s['mean']:.4f}")
                    print(f"  Std:  {s['std']:.4f}")
                    print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")

                if 'test_rmse' in force_stats:
                    s = force_stats['test_rmse']
                    print(f"\nTest RMSE:")
                    print(f"  Mean: {s['mean']:.4f}")
                    print(f"  Std:  {s['std']:.4f}")
                    print(f"  Min:  {s['min']:.4f}")
                    print(f"  Max:  {s['max']:.4f}")
                    print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")

                print(f"\nTest Loss:")
                print(f"  Mean: {test_loss_mean:.4f}")
                print(f"  Std:  {test_loss_std:.4f}")
                print(f"  Min:  {min(test_losses):.4f}")
                print(f"  Max:  {max(test_losses):.4f}")

                if 'best_val_mae' in force_stats:
                    s = force_stats['best_val_mae']
                    print(f"\nBest Validation MAE:")
                    print(f"  Mean: {s['mean']:.4f}")
                    print(f"  Std:  {s['std']:.4f}")
                    print(f"  Min:  {s['min']:.4f}")
                    print(f"  Max:  {s['max']:.4f}")
                    print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")

                print(f"\nBest Validation Loss:")
                print(f"  Mean: {val_loss_mean:.4f}")
                print(f"  Std:  {val_loss_std:.4f}")
                print(f"  Min:  {min(val_losses):.4f}")
                print(f"  Max:  {max(val_losses):.4f}")

                print(f"\nBest Step:")
                print(f"  Mean: {best_step_mean:.1f}")
                print(f"  Std:  {best_step_std:.1f}")
                print(f"  Min:  {min(best_steps)}")
                print(f"  Max:  {max(best_steps)}")

                # Save summary results
                summary_results = {
                    'task_type': task_type,
                    'num_runs': len(seeds),
                    'seeds': [r['seed'] for r in all_results],
                    'train_data_percentage': args.train_data_percentage,
                    'test_loss': {
                        'mean': float(test_loss_mean),
                        'std': float(test_loss_std),
                        'min': float(min(test_losses)),
                        'max': float(max(test_losses)),
                        'values': [float(loss) for loss in test_losses]
                    },
                    'best_val_loss': {
                        'mean': float(val_loss_mean),
                        'std': float(val_loss_std),
                        'min': float(min(val_losses)),
                        'max': float(max(val_losses)),
                        'values': [float(loss) for loss in val_losses]
                    },
                    'best_step': {
                        'mean': float(best_step_mean),
                        'std': float(best_step_std),
                        'min': int(min(best_steps)),
                        'max': int(max(best_steps)),
                        'values': [int(s) for s in best_steps]
                    },
                    'individual_results': all_results,
                    'config_files': {
                        'main_config': config_path,
                        'model_config': model_config_path,
                        'algo_config': args.algo_config
                    },
                    'timestamp': datetime.now().isoformat()
                }
                for key in ('test_mae', 'test_rmse', 'best_val_mae'):
                    if key in force_stats:
                        summary_results[key] = force_stats[key]
                if 'test_normal_mae' in force_stats:
                    summary_results['test_normal_force_mae'] = force_stats['test_normal_mae']
                if 'test_shear_mae' in force_stats:
                    summary_results['test_shear_force_mae'] = force_stats['test_shear_mae']
            else:
                # Classification metrics via shared aggregation
                class_stats = aggregate_classification_results(all_results)
                test_loss_mean = np.mean(test_losses)
                test_loss_std = np.std(test_losses)
                val_loss_mean = np.mean(val_losses)
                val_loss_std = np.std(val_losses)
                best_step_mean = np.mean(best_steps)
                best_step_std = np.std(best_steps)

                print(f"Number of runs: {len(seeds)}")
                print(f"Task type: {task_type}")
                print(f"Seeds used: {[r['seed'] for r in all_results]}")
                if args.train_data_percentage is not None:
                    print(f"Train data percentage: {args.train_data_percentage * 100:.1f}%")

                if 'test_accuracy' in class_stats:
                    s = class_stats['test_accuracy']
                    print(f"\nTest Accuracy:")
                    print(f"  Mean: {s['mean']:.2f}%")
                    print(f"  Std:  {s['std']:.2f}%")
                    print(f"  Min:  {s['min']:.2f}%")
                    print(f"  Max:  {s['max']:.2f}%")
                    print(f"  Values: {[f'{v:.2f}' for v in s['values']]}")

                print(f"\nTest Loss:")
                print(f"  Mean: {test_loss_mean:.4f}")
                print(f"  Std:  {test_loss_std:.4f}")
                print(f"  Min:  {min(test_losses):.4f}")
                print(f"  Max:  {max(test_losses):.4f}")

                if 'best_val_accuracy' in class_stats:
                    s = class_stats['best_val_accuracy']
                    print(f"\nBest Validation Accuracy:")
                    print(f"  Mean: {s['mean']:.2f}%")
                    print(f"  Std:  {s['std']:.2f}%")
                    print(f"  Min:  {s['min']:.2f}%")
                    print(f"  Max:  {s['max']:.2f}%")
                    print(f"  Values: {[f'{v:.2f}' for v in s['values']]}")

                print(f"\nBest Validation Loss:")
                print(f"  Mean: {val_loss_mean:.4f}")
                print(f"  Std:  {val_loss_std:.4f}")
                print(f"  Min:  {min(val_losses):.4f}")
                print(f"  Max:  {max(val_losses):.4f}")

                print(f"\nBest Step:")
                print(f"  Mean: {best_step_mean:.1f}")
                print(f"  Std:  {best_step_std:.1f}")
                print(f"  Min:  {min(best_steps)}")
                print(f"  Max:  {max(best_steps)}")

                # Save summary results
                summary_results = {
                    'task_type': task_type,
                    'num_runs': len(seeds),
                    'seeds': [r['seed'] for r in all_results],
                    'train_data_percentage': args.train_data_percentage,
                    'test_loss': {
                        'mean': float(test_loss_mean),
                        'std': float(test_loss_std),
                        'min': float(min(test_losses)),
                        'max': float(max(test_losses)),
                        'values': [float(loss) for loss in test_losses]
                    },
                    'best_val_loss': {
                        'mean': float(val_loss_mean),
                        'std': float(val_loss_std),
                        'min': float(min(val_losses)),
                        'max': float(max(val_losses)),
                        'values': [float(loss) for loss in val_losses]
                    },
                    'best_step': {
                        'mean': float(best_step_mean),
                        'std': float(best_step_std),
                        'min': int(min(best_steps)),
                        'max': int(max(best_steps)),
                        'values': [int(s) for s in best_steps]
                    },
                    'individual_results': all_results,
                    'config_files': {
                        'main_config': config_path,
                        'model_config': model_config_path,
                        'algo_config': args.algo_config
                    },
                    'timestamp': datetime.now().isoformat()
                }
                if 'test_accuracy' in class_stats:
                    summary_results['test_accuracy'] = class_stats['test_accuracy']
                if 'best_val_accuracy' in class_stats:
                    summary_results['best_val_accuracy'] = class_stats['best_val_accuracy']
            
            # Save to a summary file (use the last trainer's save directory as base)
            summary_dir = os.path.dirname(trainer.run_save_dir)
            os.makedirs(summary_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"multi_seed_summary_{timestamp}.yaml"
            summary_path = os.path.join(summary_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                yaml.dump(summary_results, f, default_flow_style=False)
            
            print(f"\n{'='*80}")
            print(f"Summary results saved to: {summary_path}")
            print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


# # Single run with custom train_data_percentage
# python train/run_spl.py --train_data_percentage 0.5

# # Multiple seeds with custom train_data_percentage
# python train/run_spl.py --train_data_percentage 0.5 --seeds 10 20 30 40 50

# # Multiple seeds using config file train_data_percentage
# python train/run_spl.py --seeds 10 20 30