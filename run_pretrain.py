#!/usr/bin/env python3
"""
Multimodal Pretraining script for tactile and vision data.
Trains multiple encoders and decoders with a shared trunk, switching between modalities.
"""

import os
import argparse
import yaml
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, List

# Import dataloader creation function
from data.create_dataloaders import create_dataloaders_from_config

from model.create_model import create_pretrain_model

# Import utilities
from utils.utils_logger import get_pylogger
from utils.augmentation import create_augmentation_from_config
from utils.config_printer import (
    print_config_files,
    print_merged_config,
)
from utils.utils_model import set_all_seeds, adjust_state_dict_keys
from utils.ssl_utils import random_masking, compute_mae_loss, compute_weighted_mae_loss, compute_mae_gradient_loss, SIGReg
from utils.base_trainer import BaseTrainer

log = get_pylogger(__name__)


class DatasetManager:
    """Manages dataset loading and cleanup for different modalities."""
    
    def __init__(self, modality_to_data_config: Dict[str, str]):
        """
        Initialize dataset manager.
        
        Args:
            modality_to_data_config: Dictionary mapping modality names to data config paths
        """
        self.modality_to_data_config = modality_to_data_config
        self.all_loaders = {}  # Store loaders for all modalities
        self.all_datasets = {}  # Store datasets for all modalities
        self.modality_iterators = {}  # Store iterators for each modality's train loader
    
    def load_all_datasets(self, pretrain_config: OmegaConf):
        """
        Load datasets for all modalities at once.
        
        Args:
            pretrain_config: Pretrain configuration object
        """
        for modality in self.modality_to_data_config.keys():
            # Load data config
            data_config_path = self.modality_to_data_config[modality]
            if not os.path.exists(data_config_path):
                raise FileNotFoundError(f"Data config file not found: {data_config_path}")
            
            try:
                with open(data_config_path, 'r') as f:
                    data_config = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                with open(data_config_path, 'r') as f:
                    data_config = yaml.load(f, Loader=yaml.Loader)
            
            # Merge with pretrain config training parameters
            merged_data_config = data_config.copy()
            if 'data' not in merged_data_config:
                merged_data_config['data'] = {}
            
            # Override with pretrain config if available
            if hasattr(pretrain_config, 'training'):
                training_config = pretrain_config.training
                if 'batch_size' in training_config:
                    merged_data_config['data']['batch_size'] = training_config.batch_size
                if 'num_workers' in training_config:
                    merged_data_config['data']['num_workers'] = training_config.num_workers
            
            # Create dataloader config
            dataloader_config = OmegaConf.create(merged_data_config['data'])
            # Explicitly set dataset_split_type to 'pretrain' for pretraining
            dataloader_config['dataset_split_type'] = 'pretrain'
            
            # Get probe_data_percentage from training config (default 1.0 for full dataset)
            probe_data_percentage = pretrain_config.training.get('probe_data_percentage', 1.0)
            train_data_percentage = 1.0  # Always use full dataset for pretraining
            
            # Create dataloaders
            train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
                create_dataloaders_from_config(dataloader_config)
            
            # Create probe dataloaders if needed
            if probe_data_percentage < 1.0:
                probe_dataloader_config = dataloader_config.copy()
                probe_dataloader_config['train_data_percentage'] = probe_data_percentage
                probe_train_loader, _, _, probe_train_dataset, _, _ = \
                    create_dataloaders_from_config(probe_dataloader_config)
                probe_val_loader = val_loader
                probe_test_loader = test_loader
            else:
                probe_train_loader = train_loader
                probe_val_loader = val_loader
                probe_test_loader = test_loader
                probe_train_dataset = train_dataset
            
            # Store loaders and datasets for this modality
            self.all_loaders[modality] = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader,
                'probe_train': probe_train_loader,
                'probe_val': probe_val_loader,
                'probe_test': probe_test_loader
            }
            self.all_datasets[modality] = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset,
                'probe_train': probe_train_dataset
            }
            
            # Create infinite iterator for training
            self.modality_iterators[modality] = iter(train_loader)
        
        log.info(f"Loaded datasets for all modalities: {list(self.all_loaders.keys())}")
    
    def load_single_modality_dataset(self, modality: str, pretrain_config: OmegaConf):
        """
        Load dataset for a single modality only.
        
        Args:
            modality: Modality name to load
            pretrain_config: Pretrain configuration object
        """
        if modality not in self.modality_to_data_config:
            raise ValueError(f"Modality '{modality}' not found in modality_to_data_config")
        
        # Check if already loaded
        if modality in self.all_loaders:
            log.info(f"Dataset for modality '{modality}' already loaded, skipping")
            return
        
        # Load data config
        data_config_path = self.modality_to_data_config[modality]
        if not os.path.exists(data_config_path):
            raise FileNotFoundError(f"Data config file not found: {data_config_path}")
        
        try:
            with open(data_config_path, 'r') as f:
                data_config = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            with open(data_config_path, 'r') as f:
                data_config = yaml.load(f, Loader=yaml.Loader)
        
        # Merge with pretrain config training parameters
        merged_data_config = data_config.copy()
        if 'data' not in merged_data_config:
            merged_data_config['data'] = {}
        
        # Override with pretrain config if available
        if hasattr(pretrain_config, 'training'):
            training_config = pretrain_config.training
            if 'batch_size' in training_config:
                merged_data_config['data']['batch_size'] = training_config.batch_size
            if 'num_workers' in training_config:
                merged_data_config['data']['num_workers'] = training_config.num_workers
        
        # Create dataloader config
        dataloader_config = OmegaConf.create(merged_data_config['data'])
        # Explicitly set dataset_split_type to 'pretrain' for pretraining
        dataloader_config['dataset_split_type'] = 'pretrain'
        
        # Get probe_data_percentage from training config (default 1.0 for full dataset)
        probe_data_percentage = pretrain_config.training.get('probe_data_percentage', 1.0)
        train_data_percentage = 1.0  # Always use full dataset for pretraining
        
        # Create dataloaders
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            create_dataloaders_from_config(dataloader_config)
        
        # Create probe dataloaders if needed
        if probe_data_percentage < 1.0:
            probe_dataloader_config = dataloader_config.copy()
            probe_dataloader_config['train_data_percentage'] = probe_data_percentage
            probe_train_loader, _, _, probe_train_dataset, _, _ = \
                create_dataloaders_from_config(probe_dataloader_config)
            probe_val_loader = val_loader
            probe_test_loader = test_loader
        else:
            probe_train_loader = train_loader
            probe_val_loader = val_loader
            probe_test_loader = test_loader
            probe_train_dataset = train_dataset
        
        # Store loaders and datasets for this modality
        self.all_loaders[modality] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'probe_train': probe_train_loader,
            'probe_val': probe_val_loader,
            'probe_test': probe_test_loader
        }
        self.all_datasets[modality] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'probe_train': probe_train_dataset
        }
        
        # Create infinite iterator for training
        self.modality_iterators[modality] = iter(train_loader)
        
        log.info(f"Loaded dataset for modality: {modality}")
    
    def clear_single_modality_dataset(self, modality: str):
        """
        Clear dataset for a single modality and free memory.
        
        Args:
            modality: Modality name to clear
        """
        if modality not in self.all_loaders:
            log.info(f"Dataset for modality '{modality}' not loaded, nothing to clear")
            return
        
        # Close iterator if exists
        if modality in self.modality_iterators:
            del self.modality_iterators[modality]
        
        # Close dataloaders for this modality
        if modality in self.all_loaders:
            for loader in self.all_loaders[modality].values():
                if loader is not None:
                    try:
                        loader._iterator = None
                    except:
                        pass
        
        # Delete references
        if modality in self.all_loaders:
            del self.all_loaders[modality]
        if modality in self.all_datasets:
            del self.all_datasets[modality]
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        log.info(f"Cleared dataset for modality: {modality}")
    
    def get_next_batch(self, modality: str):
        """
        Get next batch from a modality's train loader.
        Automatically restarts iterator if exhausted.
        
        Args:
            modality: Modality name
            
        Returns:
            Batch data
        """
        try:
            batch = next(self.modality_iterators[modality])
        except StopIteration:
            # Restart iterator
            self.modality_iterators[modality] = iter(self.all_loaders[modality]['train'])
            batch = next(self.modality_iterators[modality])
        return batch
    
    def get_loaders(self, modality: str = None):
        """
        Get loaders for a specific modality or all modalities.
        
        Args:
            modality: Modality name (if None, returns all)
            
        Returns:
            Loaders dictionary
        """
        if modality is None:
            return self.all_loaders
        return self.all_loaders.get(modality)
    
    def get_datasets(self, modality: str = None):
        """
        Get datasets for a specific modality or all modalities.
        
        Args:
            modality: Modality name (if None, returns all)
            
        Returns:
            Datasets dictionary
        """
        if modality is None:
            return self.all_datasets
        return self.all_datasets.get(modality)
    
    def clear_all_datasets(self):
        """Clear all datasets and free memory."""
        # Close all iterators
        self.modality_iterators = {}
        
        # Close all dataloaders
        for modality_loaders in self.all_loaders.values():
            for loader in modality_loaders.values():
                if loader is not None:
                    try:
                        loader._iterator = None
                    except:
                        pass
        
        # Delete references
        self.all_loaders = {}
        self.all_datasets = {}
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        log.info("Cleared all datasets")


class PretrainConfig:
    """Configuration class for multimodal pretraining."""
    
    def __init__(self, pretrain_config_path: str, ssl_config_path: str, 
                 modality_to_data_config: Dict[str, str]):
        """
        Initialize pretrain configuration.
        
        Args:
            pretrain_config_path: Path to pretrain model config file
            ssl_config_path: Path to SSL algorithm config file
            modality_to_data_config: Dictionary mapping modality names to data config paths
        """
        self.pretrain_config_path = pretrain_config_path
        self.ssl_config_path = ssl_config_path
        self.modality_to_data_config = modality_to_data_config
        self.load_configs()
    
    def load_configs(self):
        """Load configurations from multiple YAML files."""
        # Print config file paths
        print_config_files(
            {
                'pretrain': self.pretrain_config_path,
                'ssl': self.ssl_config_path,
                'data': self.modality_to_data_config
            },
            title="PRETRAIN CONFIGURATION LOADING"
        )
        
        # Load pretrain model configuration
        if not os.path.exists(self.pretrain_config_path):
            raise FileNotFoundError(f"Pretrain config file not found: {self.pretrain_config_path}")
        
        try:
            with open(self.pretrain_config_path, 'r') as f:
                pretrain_config = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            with open(self.pretrain_config_path, 'r') as f:
                pretrain_config = yaml.load(f, Loader=yaml.Loader)
        
        # Load SSL algorithm configuration
        if not os.path.exists(self.ssl_config_path):
            raise FileNotFoundError(f"SSL config file not found: {self.ssl_config_path}")
        
        try:
            with open(self.ssl_config_path, 'r') as f:
                ssl_config = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            with open(self.ssl_config_path, 'r') as f:
                ssl_config = yaml.load(f, Loader=yaml.Loader)
        
        # Merge configurations
        merged_config = {}
        merged_config.update(pretrain_config)
        merged_config['algorithm'] = ssl_config.get('algorithm', {})
        merged_config['training'] = ssl_config.get('training', {})
        merged_config['logging'] = ssl_config.get('logging', {})
        
        # Merge augmentation if present
        if 'augmentation' in ssl_config:
            merged_config['augmentation'] = ssl_config['augmentation']
        
        # Convert to OmegaConf for easier access
        self.config = OmegaConf.create(merged_config)
        
        # Print merged configuration
        print_merged_config(self.config)


class PretrainTrainer(BaseTrainer):
    """Trainer for multimodal pretraining with shared trunk."""

    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.config = config.config
        self.pretrain_config_path = config.pretrain_config_path
        self.ssl_config_path = config.ssl_config_path
        self.modality_to_data_config = config.modality_to_data_config

        # Force GPU usage
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)
        else:
            raise RuntimeError("CUDA is not available. This training requires GPU acceleration.")
        
        # Set random seeds
        set_all_seeds(self.config.training.get('random_seed', 42))
        
        # Setup dataset manager
        self.dataset_manager = DatasetManager(self.modality_to_data_config)
        
        # Setup model
        self.setup_model()
        
        # Setup SSL algorithm
        self.setup_ssl_algorithm()
        
        # Setup logging
        self.setup_logging()
        
        # Setup augmentation
        self.setup_augmentation()
        
        # Setup optimizer (for all modalities)
        self.setup_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.modality_losses = {}  # Track losses per modality
    
    def setup_model(self):
        """Setup the pretrain model with multiple encoders, shared trunk, and decoders."""
        # Convert config to dict for create_pretrain_model
        pretrain_config_dict = OmegaConf.to_container(self.config, resolve=True)
        
        # Validate num_frames for vision encoders before model creation
        encoders_config = pretrain_config_dict.get('encoders', {})
        global_num_frames = pretrain_config_dict.get('num_frames', None)
        
        print("\nValidating encoder configurations...")
        for modality_name, encoder_config in encoders_config.items():
            encoder_type = encoder_config.get('type', 'tactile')
            if encoder_type in ['vit', 'vision']:
                # Vision encoder requires num_frames
                encoder_num_frames = encoder_config.get('num_frames', global_num_frames)
                if encoder_num_frames is None:
                    raise ValueError(
                        f"Vision encoder '{modality_name}' requires 'num_frames' to be specified "
                        f"either in its encoder config or globally in pretrain config"
                    )
                print(f"  {modality_name} encoder: num_frames={encoder_num_frames}")
        
        # Create model
        self.model = create_pretrain_model(pretrain_config_dict).to(self.device)
        
        # Log num_frames for each encoder after creation
        print(f"\nEncoder configurations after model creation:")
        for modality in self.model.modalities:
            encoder = self.model.encoders[modality]
            if hasattr(encoder, 'num_frames'):
                print(f"  {modality}: num_frames={encoder.num_frames}")
            elif hasattr(encoder, 'vit') and hasattr(encoder.vit, 'num_frames'):
                print(f"  {modality}: num_frames={encoder.vit.num_frames}")
            else:
                print(f"  {modality}: num_frames=N/A (tactile encoder)")
        
        # Compile model for optimization (PyTorch 2.0+)
        if self.config.training.compile_model and hasattr(torch, 'compile'):
            for modality in self.model.modalities:
                self.model.encoders[modality] = torch.compile(self.model.encoders[modality])
                self.model.decoders[modality] = torch.compile(self.model.decoders[modality])
            self.model.shared_trunk = torch.compile(self.model.shared_trunk)
        
        print(f"\nCreated pretrain model with modalities: {self.model.modalities}")
        print(f"Shared trunk embed_dim: {self.model.embed_dim}")
    
    def setup_ssl_algorithm(self):
        """Setup MAE SSL algorithm."""
        algorithm_config = self.config.algorithm
        # Default mask_ratio from algorithm config (used as fallback only)
        default_mask_ratio = algorithm_config.kwargs.get('mask_ratio', 0.75)
        self.norm_pix_loss = algorithm_config.kwargs.get('norm_pix_loss', False)
        
        # Get mask_ratio for each modality from encoder configs
        # Each encoder should specify its own mask_ratio in the model config
        # Falls back to algorithm config default only if encoder config doesn't specify it
        self.modality_mask_ratios = {}
        self.modality_use_weighted_mae_loss = {}
        self.modality_saliency_weight = {}
        self.modality_use_gradient_loss = {}
        self.modality_gradient_loss_coef = {}

        encoders_config = self.config.get('encoders', {})

        image_modalities = ['9dtact', 'gsmini', 'gelsight']
        
        for modality in self.model.modalities:
            is_image_modality = modality in image_modalities
            
            if modality in encoders_config:
                # Get mask_ratio from encoder config (required per encoder)
                encoder_mask_ratio = encoders_config[modality].get('mask_ratio', None)
                if encoder_mask_ratio is not None:
                    self.modality_mask_ratios[modality] = float(encoder_mask_ratio)
                else:
                    # Fallback to default from algorithm config if encoder doesn't specify
                    self.modality_mask_ratios[modality] = default_mask_ratio
                    log.warning(f"Encoder '{modality}' does not specify mask_ratio, using default: {default_mask_ratio}")
            else:
                # Fallback to default from algorithm config if encoder config not found
                self.modality_mask_ratios[modality] = default_mask_ratio
                log.warning(f"Encoder config not found for '{modality}', using default mask_ratio: {default_mask_ratio}")
            
            if is_image_modality:
                self.modality_use_weighted_mae_loss[modality] = algorithm_config.kwargs.get('use_weighted_mae_loss', True)
                self.modality_saliency_weight[modality] = algorithm_config.kwargs.get('saliency_weight', 5.0)
                gradient_loss_coef = algorithm_config.kwargs.get('gradient_loss_coef', 0.0)
                self.modality_gradient_loss_coef[modality] = gradient_loss_coef
                self.modality_use_gradient_loss[modality] = gradient_loss_coef > 0.0
            else:
                self.modality_use_weighted_mae_loss[modality] = False
                self.modality_saliency_weight[modality] = 5.0
                self.modality_use_gradient_loss[modality] = False
                self.modality_gradient_loss_coef[modality] = 0.0
        
        # Log mask ratios and settings for each modality
        print(f"\nMask ratios per modality:")
        for modality, mask_ratio in self.modality_mask_ratios.items():
            is_image = modality in image_modalities
            print(f"  {modality}: {mask_ratio} ({'image' if is_image else 'tactile'})")
            if is_image:
                print(f"    Weighted MAE loss: {self.modality_use_weighted_mae_loss[modality]}")
                if self.modality_use_weighted_mae_loss[modality]:
                    print(f"    Saliency weight: {self.modality_saliency_weight[modality]}")
                print(f"    Gradient loss: {self.modality_use_gradient_loss[modality]}")
                if self.modality_use_gradient_loss[modality]:
                    print(f"    Gradient loss coef: {self.modality_gradient_loss_coef[modality]}")
            else:
                print(f"    Using standard MAE loss (weighted loss disabled for tactile)")
        
        # Keep backward compatibility: set self.mask_ratio to default (for any code that might still use it)
        self.mask_ratio = default_mask_ratio

        # SIGReg auxiliary loss
        sigreg_coef = algorithm_config.kwargs.get('sigreg_loss_coef', 0.0)
        self.sigreg_loss_coef = float(sigreg_coef)
        if self.sigreg_loss_coef > 0.0:
            knots = int(algorithm_config.kwargs.get('sigreg_knots', 17))
            num_proj = int(algorithm_config.kwargs.get('sigreg_num_proj', 1024))
            self.sigreg = SIGReg(knots=knots, num_proj=num_proj).to(self.device)
            print(f"\nSIGReg enabled: coef={self.sigreg_loss_coef}, knots={knots}, num_proj={num_proj}")
        else:
            self.sigreg = None
    
    def setup_logging(self):
        """Setup logging and checkpointing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.config.logging.get('run_name', None)
        log_suffix = run_name if run_name else f"pretrain_multimodal_{timestamp}"
        super().setup_logging(
            self.config.logging.log_dir,
            self.config.logging.save_dir,
            log_suffix,
        )
        # Early stopping per modality
        self.patience = self.config.training.get('patience', 10)
        self.modality_best_losses = {}
        self.modality_patience_counters = {}
        self.modality_converged = {}
        self.modality_epochs = {}
    
    def setup_augmentation(self):
        """Setup data augmentation."""
        self.augmentation_enabled = self.config.get('augmentation', {}).get('enabled', False)
        if self.augmentation_enabled:
            augmentation_config = self.config.augmentation.copy()
            if 'params' in augmentation_config and augmentation_config['params']:
                if 'seed' not in augmentation_config['params']:
                    augmentation_config['params'] = augmentation_config['params'].copy()
                    augmentation_config['params']['seed'] = self.config.training.get('random_seed', 42)
            self.augmentation = create_augmentation_from_config(augmentation_config)
        else:
            self.augmentation = None
    
    def _get_batch_data(self, batch):
        """Extract input data from batch dictionary."""
        if 'tactile' in batch:
            return batch['tactile'].to(self.device)
        elif 'images' in batch:
            return batch['images'].to(self.device)
        else:
            raise ValueError("Batch must contain either 'tactile' or 'images' key")
    
    def mae_forward(self, x, modality: str):
        """
        MAE forward pass for a specific modality.
        
        Args:
            x: Input data
            modality: Modality name
            
        Returns:
            pred: Reconstructed patches/tokens
            mask: Binary mask
            target: Target patches/tokens
        """
        batch_size = x.shape[0]
        
        # Get encoder for this modality
        encoder = self.model.get_encoder(modality)
        
        # Patchify to get tokens for masking
        patches = encoder.patchify(x)  # [B, N, patch_dim]
        num_tokens = patches.size(1)
        
        # Get modality-specific mask_ratio
        mask_ratio = self.modality_mask_ratios.get(modality, self.mask_ratio)
        
        ids_keep, mask, ids_restore = random_masking(batch_size, num_tokens, x.device, mask_ratio)
        
        # Forward encoder with masking
        encoder_out = encoder(patches, masks=ids_keep)
        
        # Forward through shared trunk
        trunk_output = self.model.shared_trunk((encoder_out, mask, ids_restore))
        if isinstance(trunk_output, tuple):
            trunk_features, _, _ = trunk_output
        else:
            trunk_features = trunk_output
        
        # Forward decoder
        decoder = self.model.get_decoder(modality)
        pred = decoder(trunk_features, ids_restore)
        
        # Target is the original patches
        target = patches

        return pred, mask, target, trunk_features
    
    def train_step(self, modality: str, step: int):
        """
        Train one step for a specific modality.
        
        Args:
            modality: Modality name
            step: Current training step
            
        Returns:
            Loss value
        """
        # Get next batch from this modality
        batch = self.dataset_manager.get_next_batch(modality)
        x = self._get_batch_data(batch)
        
        # Apply augmentation
        if self.augmentation is not None:
            x = self.augmentation(x, training=True)
        
        # Forward pass
        forward_context = autocast('cuda') if self.use_amp else torch.enable_grad()
        gradient_loss_value = 0.0
        sigreg_loss_value = 0.0
        with forward_context:
            pred, mask, target, trunk_features = self.mae_forward(x, modality)
            
            # Compute MAE loss (weighted for image modalities if enabled)
            use_weighted_mae = self.modality_use_weighted_mae_loss.get(modality, False)
            if use_weighted_mae:
                saliency_weight = self.modality_saliency_weight.get(modality, 5.0)
                loss = compute_weighted_mae_loss(pred, target, mask, self.norm_pix_loss, saliency_weight)
            else:
                loss = compute_mae_loss(pred, target, mask, self.norm_pix_loss)
            
            # Add gradient loss for image modalities if enabled
            use_gradient_loss = self.modality_use_gradient_loss.get(modality, False)
            if use_gradient_loss:
                try:
                    # Get patch size from encoder
                    patch_size = 16  # Default
                    encoder = self.model.get_encoder(modality)
                    if hasattr(encoder, 'vit') and hasattr(encoder.vit, 'patch_embed'):
                        patch_size_obj = encoder.vit.patch_embed.patch_size
                        if isinstance(patch_size_obj, tuple):
                            patch_size = patch_size_obj[-1]  # Use spatial patch size
                        else:
                            patch_size = patch_size_obj
                    
                    # Get image dimensions from original input
                    # Handle both video [B, T, C, H, W] and image [B, C, H, W] formats
                    if len(x.shape) == 5:  # [B, T, C, H, W]
                        B, T, C, H, W = x.shape
                        num_frames = T
                    elif len(x.shape) == 4:  # [B, C, H, W]
                        B, C, H, W = x.shape
                        num_frames = None
                    else:
                        # Skip gradient loss for non-image data
                        num_frames = None
                        H, W = None, None
                    
                    if H is not None and W is not None:
                        gradient_loss_coef = self.modality_gradient_loss_coef.get(modality, 0.0)
                        gradient_loss = compute_mae_gradient_loss(
                            pred_patches=pred,
                            target_patches=target,
                            mask=mask,
                            patch_size=patch_size,
                            img_size=(H, W),
                            gradient_loss_coef=gradient_loss_coef,
                            num_frames=num_frames
                        )
                        loss = loss + gradient_loss
                        gradient_loss_value = gradient_loss.item()
                except Exception as e:
                    log.warning(f"Could not compute gradient loss for {modality}: {e}")
                    # Continue with just MAE loss
                    gradient_loss_value = 0.0

            # SIGReg auxiliary loss on shared trunk features [B, L, D] → permute to [L, B, D]
            if self.sigreg is not None:
                sigreg_loss = self.sigreg(trunk_features.permute(1, 0, 2))
                loss = loss + self.sigreg_loss_coef * sigreg_loss
                sigreg_loss_value = sigreg_loss.item()
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            log.warning(f"Invalid loss detected at step {step} for {modality}: {loss.item()}")
            return None

        loss_value = self.grad_update_step(
            loss, self.modality_clip_params[modality], step,
            clip_val=float(self.config.training.get('gradient_clip_val', 0)),
            step_scheduler=True,  # pretrain is step-based: always step scheduler
        )
        
        # Log to TensorBoard
        if step % self.config.logging.log_interval == 0:
            self.writer.add_scalar(f'{modality}/Train_Loss', loss_value, step)
            if gradient_loss_value > 0:
                self.writer.add_scalar(f'{modality}/Gradient_Loss', gradient_loss_value, step)
            if sigreg_loss_value > 0:
                self.writer.add_scalar(f'{modality}/SIGReg_Loss', sigreg_loss_value, step)
            self.writer.add_scalar(f'{modality}/LearningRate',
                                 self.optimizer.param_groups[0]['lr'], step)
        
        return loss_value
    
    def _get_dataloader_length(self, dataloader, loader_name="dataloader"):
        """
        Get the length of a dataloader, handling both regular and iterable datasets.
        Results are cached to avoid recalculating.
        
        Args:
            dataloader: The dataloader to get length for
            loader_name: Name of the dataloader for logging purposes
            
        Returns:
            int: Number of batches in the dataloader
        """
        # Check cache first
        cache_key = f'_len_{loader_name}'
        if hasattr(self, cache_key):
            return getattr(self, cache_key)
        
        try:
            # Try to get length directly (works for regular datasets)
            length = len(dataloader)
            setattr(self, cache_key, length)
            return length
        except TypeError:
            # For iterable datasets (like WebDataset), len() is not available
            # Try to get length from dataset if it has __len__
            if hasattr(dataloader.dataset, '__len__'):
                try:
                    dataset_len = len(dataloader.dataset)
                    # Calculate number of batches
                    batch_size = dataloader.batch_size if hasattr(dataloader, 'batch_size') else 1
                    length = (dataset_len + batch_size - 1) // batch_size  # Ceiling division
                    log.info(f"Calculated {loader_name} length from dataset: {length} batches (dataset size: {dataset_len})")
                    setattr(self, cache_key, length)
                    return length
                except (TypeError, AttributeError) as e:
                    log.debug(f"Could not get length from dataset.__len__(): {e}")
                    pass
            
            # Last resort: calculate by iterating once (with warning)
            log.warning(f"Could not determine length of {loader_name} (iterable dataset). "
                       f"Calculating by iterating once with num_workers=0 (this may take a moment for large datasets)...")
            
            # Create a temporary dataloader with num_workers=0 to avoid multiprocessing issues
            from torch.utils.data import DataLoader
            collate_fn = getattr(dataloader, 'collate_fn', None)
            temp_dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=False
            )
            
            try:
                length = sum(1 for _ in temp_dataloader)
                if length == 0:
                    log.error(f"Calculated {loader_name} length is 0 - no batches found!")
                    default_length = 1000
                    log.warning(f"Using default length estimate: {default_length} batches for {loader_name}")
                    setattr(self, cache_key, default_length)
                    return default_length
                log.info(f"Calculated {loader_name} length by iteration: {length} batches")
                setattr(self, cache_key, length)
                return length
            except Exception as e:
                log.error(f"Failed to calculate {loader_name} length by iteration: {e}")
                default_length = 1000
                log.warning(f"Using default estimate for {loader_name} length: {default_length} batches")
                setattr(self, cache_key, default_length)
                return default_length
    
    def validate_modality(self, modality: str, val_loader):
        """Validate for a specific modality."""
        self.model.encoders[modality].eval()
        self.model.decoders[modality].eval()
        self.model.shared_trunk.eval()
        
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation - {modality}"):
                x = self._get_batch_data(batch)
                forward_context = autocast('cuda') if self.use_amp else torch.no_grad()
                with forward_context:
                    pred, mask, target, _ = self.mae_forward(x, modality)
                    
                    # Compute MAE loss (weighted for image modalities if enabled)
                    use_weighted_mae = self.modality_use_weighted_mae_loss.get(modality, False)
                    if use_weighted_mae:
                        saliency_weight = self.modality_saliency_weight.get(modality, 5.0)
                        loss = compute_weighted_mae_loss(pred, target, mask, self.norm_pix_loss, saliency_weight)
                    else:
                        loss = compute_mae_loss(pred, target, mask, self.norm_pix_loss)
                    
                    # Add gradient loss for image modalities if enabled (same as training)
                    use_gradient_loss = self.modality_use_gradient_loss.get(modality, False)
                    if use_gradient_loss:
                        try:
                            # Get patch size from encoder
                            patch_size = 16  # Default
                            encoder = self.model.get_encoder(modality)
                            if hasattr(encoder, 'vit') and hasattr(encoder.vit, 'patch_embed'):
                                patch_size_obj = encoder.vit.patch_embed.patch_size
                                if isinstance(patch_size_obj, tuple):
                                    patch_size = patch_size_obj[-1]
                                else:
                                    patch_size = patch_size_obj
                            
                            # Get image dimensions
                            if len(x.shape) == 5:  # [B, T, C, H, W]
                                B, T, C, H, W = x.shape
                                num_frames = T
                            elif len(x.shape) == 4:  # [B, C, H, W]
                                B, C, H, W = x.shape
                                num_frames = None
                            else:
                                # Skip for non-image data
                                num_frames = None
                                H, W = None, None
                            
                            if H is not None and W is not None:
                                gradient_loss_coef = self.modality_gradient_loss_coef.get(modality, 0.0)
                                gradient_loss = compute_mae_gradient_loss(
                                    pred_patches=pred,
                                    target_patches=target,
                                    mask=mask,
                                    patch_size=patch_size,
                                    img_size=(H, W),
                                    gradient_loss_coef=gradient_loss_coef,
                                    num_frames=num_frames
                                )
                                loss = loss + gradient_loss
                        except Exception as e:
                            # Skip gradient loss on validation errors
                            pass
                
                total_loss += loss.item()
                batch_count += 1
        
        # Use actual batch count if available, otherwise try to get from cached length
        if batch_count == 0:
            log.error(f"No batches processed during validation for {modality}! Check your dataloader configuration.")
            num_val_iterations = self._get_dataloader_length(val_loader, f"val_loader_{modality}")
            if num_val_iterations == 0:
                num_val_iterations = 1  # Prevent division by zero
                log.warning(f"Using fallback value of 1 for validation loss calculation for {modality}.")
        else:
            num_val_iterations = batch_count
        
        val_loss = total_loss / num_val_iterations
        # Restore train mode
        self.model.encoders[modality].train()
        self.model.decoders[modality].train()
        self.model.shared_trunk.train()
        return val_loss
    
    def setup_optimizer(self):
        """Setup optimizer for all modalities (all encoders + decoders + shared trunk)."""
        algorithm_config = self.config.algorithm

        # Collect all trainable named parameters across encoders, trunk, decoders
        all_named = []
        for modality in self.model.modalities:
            all_named += [(f'enc_{modality}.{n}', p)
                          for n, p in self.model.encoders[modality].named_parameters()]
            all_named += [(f'dec_{modality}.{n}', p)
                          for n, p in self.model.decoders[modality].named_parameters()]
        all_named += [(f'trunk.{n}', p)
                      for n, p in self.model.shared_trunk.named_parameters()]

        self.optimizer = self._build_optimizer(
            all_named,
            optimizer_type=algorithm_config.optimizer.type,
            lr=algorithm_config.optimizer.lr,
            weight_decay=algorithm_config.optimizer.weight_decay,
            **algorithm_config.optimizer.get('kwargs', {}),
        )

        total_steps = self.config.training.get('total_steps', 100000)
        self.lr_scheduler, self.scheduler_per_iter = self._build_scheduler(
            self.optimizer,
            algorithm_config.get('lr_scheduler'),
            total_steps=total_steps,
        )

        self.setup_amp(self.config.training.use_amp)
        self.gradient_accumulation_steps = self.config.training.gradient_accumulation_steps

        # Precompute per-modality clip param lists (avoid rebuilding every train step)
        self.modality_clip_params = {
            m: (list(self.model.encoders[m].parameters()) +
                list(self.model.decoders[m].parameters()) +
                list(self.model.shared_trunk.parameters()))
            for m in self.model.modalities
        }

        # Set all submodels to train mode once here
        for m in self.model.modalities:
            self.model.encoders[m].train()
            self.model.decoders[m].train()
        self.model.shared_trunk.train()
    
    def train(self, resume_from_checkpoint: str = None):
        """
        Main training loop with step-based training and random modality sampling.
        
        Args:
            resume_from_checkpoint: Optional path to checkpoint file to resume from
        """
        modalities = self.model.modalities
        
        # Get step-based training parameters from config
        total_steps = self.config.training.get('total_steps', 100000)
        val_interval = self.config.training.get('val_interval', 500)
        checkpoint_save_interval = self.config.training.get('checkpoint_save_interval', 5000)
        
        # Load checkpoint if resuming
        start_step = 1
        if resume_from_checkpoint is not None:
            start_step = self.load_checkpoint(resume_from_checkpoint) + 1  # Resume from next step
            print(f"\nResuming training from step {start_step} (checkpoint was at step {start_step - 1})")
        else:
            print("Starting multimodal pretraining with step-based training...")
        
        print(f"Total training steps: {total_steps:,}")
        print(f"Starting from step: {start_step:,}")
        print(f"Validation interval: every {val_interval} steps")
        print(f"Checkpoint save interval: every {checkpoint_save_interval} steps")
        print(f"Modalities: {self.model.modalities}")
        
        # Load all datasets at the beginning
        print(f"\n{'='*80}")
        print("LOADING ALL MODALITIES")
        print(f"{'='*80}")
        self.dataset_manager.load_all_datasets(self.config)
        
        # Get modality weights from config (if specified) or use dataset sizes
        modality_weights_config = self.config.training.get('modality_weights', None)
        
        if modality_weights_config is not None:
            # Use custom weights from config
            modality_weights = []
            for modality in modalities:
                weight = modality_weights_config.get(modality, 1.0)
                modality_weights.append(weight)
            print(f"\nUsing custom modality weights from config:")
            for modality, weight in zip(modalities, modality_weights):
                print(f"  {modality}: {weight}")
        else:
            # Compute weights based on dataset sizes
            modality_weights = []
            modality_sizes = {}
            print(f"\nComputing modality weights based on dataset sizes:")
            for modality in modalities:
                train_dataset = self.dataset_manager.get_datasets(modality)['train']
                try:
                    # Try to get length from dataset (works for both regular and IterableDataset with __len__)
                    dataset_size = len(train_dataset)
                except (TypeError, AttributeError):
                    # Fallback: try to get from dataloader
                    train_loader = self.dataset_manager.get_loaders(modality)['train']
                    try:
                        dataset_size = len(train_loader)
                    except (TypeError, AttributeError):
                        # Last resort: use a default estimate
                        log.warning(f"Could not determine dataset size for {modality}, using default estimate")
                        dataset_size = 10000  # Default estimate
                
                modality_sizes[modality] = dataset_size
                modality_weights.append(dataset_size)
                print(f"  {modality}: {dataset_size} training samples")
        
        # Normalize weights to probabilities
        total_weight = sum(modality_weights)
        modality_probs = [w / total_weight for w in modality_weights]
        print(f"\nModality sampling probabilities:")
        for modality, prob in zip(modalities, modality_probs):
            print(f"  {modality}: {prob:.2%}")
        
        # Training statistics — bounded deque to avoid unbounded growth
        step_losses = {modality: deque(maxlen=200) for modality in modalities}
        if resume_from_checkpoint is None:
            self.current_epoch = 0  # Keep for compatibility with checkpointing
            # Initialize tracking for each modality only if not resuming
            for modality in modalities:
                self.modality_best_losses[modality] = float('inf')
                self.modality_patience_counters[modality] = 0
                self.modality_converged[modality] = False
                self.modality_epochs[modality] = 0
        
        # Main training loop
        print(f"\n{'='*80}")
        if resume_from_checkpoint is not None:
            print("RESUMING TRAINING")
        else:
            print("STARTING TRAINING")
        print(f"{'='*80}")
        
        progress_bar = tqdm(range(start_step, total_steps + 1), desc="Training", initial=start_step - 1, total=total_steps)
        
        for step in progress_bar:
            self.current_step = step
            
            # Sample modality with weights (custom or dataset-size based)
            modality = np.random.choice(modalities, p=modality_probs)
            
            # Train one step
            loss_value = self.train_step(modality, step)
            
            if loss_value is not None:
                step_losses[modality].append(loss_value)
                
                # Update progress bar
                avg_loss = sum(step_losses[modality]) / len(step_losses[modality]) if step_losses[modality] else 0.0
                progress_bar.set_postfix({
                    'Modality': modality,
                    'Loss': f'{loss_value:.4f}',
                    'Avg Loss (100)': f'{avg_loss:.4f}'
                })
            
            # Validate on all modalities every val_interval steps
            if step % val_interval == 0:
                print(f"\nStep {step}: Validating on all modalities...")
                for val_modality in modalities:
                    val_loaders = self.dataset_manager.get_loaders(val_modality)
                    val_loss = self.validate_modality(val_modality, val_loaders['val'])
                    
                    # Update best loss for this modality
                    is_best = val_loss < self.modality_best_losses[val_modality]
                    if is_best:
                        self.modality_best_losses[val_modality] = val_loss
                        print(f"  {val_modality}: Val Loss: {val_loss:.4f} (NEW BEST)")
                    else:
                        print(f"  {val_modality}: Val Loss: {val_loss:.4f} (Best: {self.modality_best_losses[val_modality]:.4f})")
                    
                    # Log to TensorBoard (grouped under same modality section)
                    # Use step as the global step to ensure proper time series tracking
                    self.writer.add_scalar(f'{val_modality}/Val_Loss', val_loss, global_step=step)
                
                # Flush writer to ensure all validation logs are written immediately
                self.writer.flush()
                
                # Save checkpoint
                self.current_epoch = step // val_interval  # For checkpoint naming
                if step % checkpoint_save_interval == 0:
                    self.save_checkpoint()
            
            # Clear GPU cache periodically
            if step % 100 == 0:
                torch.cuda.empty_cache()
        
        # Print final summary
        print(f"\n{'='*80}")
        print("PRETRAINING SUMMARY")
        print(f"{'='*80}")
        for modality in modalities:
            avg_train_loss = sum(step_losses[modality]) / len(step_losses[modality]) if step_losses[modality] else 0.0
            print(f"{modality}: Best Val Loss: {self.modality_best_losses[modality]:.4f}, "
                  f"Avg Train Loss: {avg_train_loss:.4f}, "
                  f"Steps trained: {len(step_losses[modality])}")
        
        print("\nPretraining completed!")
                
        # Clear all datasets
        self.dataset_manager.clear_all_datasets()
        
        self.writer.close()
    

    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = self._base_checkpoint(self.model.state_dict())
        checkpoint.update({
            'config': self.config,
            'pretrain_config_path': self.pretrain_config_path,
            'ssl_config_path': self.ssl_config_path,
            'modality_to_data_config': self.modality_to_data_config,
            'modality_best_losses': self.modality_best_losses,
            'modality_patience_counters': self.modality_patience_counters,
            'modality_converged': self.modality_converged,
            'modality_epochs': self.modality_epochs,
            'log_path': self.log_path,
        })
        checkpoint_dir = os.path.join(
            self.config.logging.save_dir, os.path.basename(self.log_path)
        )
        self._save_to_disk(
            checkpoint, checkpoint_dir,
            f'checkpoint_step_{self.current_step}.pth',
            is_best=is_best,
            config=self.config,
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and restore full training state. Returns step to resume from."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"\n{'='*60}\nLOADING CHECKPOINT: {checkpoint_path}\n{'='*60}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._load_model_weights(checkpoint, self.model)
        self._restore_common_state(checkpoint)

        # Pretrain-specific: restore modality tracking
        for key in ('modality_best_losses', 'modality_patience_counters',
                    'modality_converged', 'modality_epochs'):
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
        print(f"  Modality best losses: {self.modality_best_losses}")

        # Restore TensorBoard writer to original run directory
        saved_log_path = checkpoint.get('log_path')
        if saved_log_path and os.path.exists(saved_log_path):
            if self.writer:
                self.writer.close()
            self.log_path = saved_log_path
            self.writer = SummaryWriter(self.log_path)
            print(f"  TensorBoard resumed → {self.log_path}")
        else:
            # Fallback: infer from checkpoint directory name
            ckpt_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
            inferred = os.path.join(self.config.logging.get('log_dir', 'logs/pretrain'), ckpt_dir_name)
            if os.path.exists(inferred):
                if self.writer:
                    self.writer.close()
                self.log_path = inferred
                self.writer = SummaryWriter(self.log_path)
                print(f"  TensorBoard inferred → {self.log_path}")

        print(f"\nResuming from step {self.current_step}")
        return self.current_step
    


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Multimodal pretraining with shared trunk')
    parser.add_argument('--pretrain_config', type=str, required=False, default="config/model/pretrain_medium.yaml",
                       help='Path to pretrain model config file (e.g., config/model/pretrain.yaml)')
    parser.add_argument('--ssl_config', type=str, required=False, default="config/algo/pretrain.yaml",
                       help='Path to SSL algorithm config file')
    parser.add_argument('--checkpoint', type=str, required=False, default=None,
                       help='Path to checkpoint file to load and run probe evaluation (e.g., checkpoints/pretrain/pretrain_multimodal_20251204_004617/best_model.pth)')
    parser.add_argument('--resume', type=str, required=False, default=None,
                       help='Path to checkpoint file to resume training from (e.g., checkpoints/pretrain/pretrain_multimodal_20251204_004617/checkpoint_step_5000.pth)')
    
    args = parser.parse_args()
    
    # Use fixed default paths for data configs
    default_data_configs = {
        '9dtact': 'config/sensor/9dtact_config.yaml',
        'xela': 'config/sensor/xela_config.yaml',
        'gsmini': 'config/sensor/gsmini_config.yaml',
        'tac02': 'config/sensor/tac_config.yaml'
    }
    
    # Map modalities to data configs (only include configs that exist)
    modality_to_data_config = {}
    for modality, config_path in default_data_configs.items():
        if os.path.exists(config_path):
            modality_to_data_config[modality] = config_path
        else:
            print(f"Warning: Data config not found for {modality}: {config_path}")
    
    if not modality_to_data_config:
        raise ValueError("At least one data config must be provided. Check that config files exist in config/sensor/")
    
    # Load configuration
    config = PretrainConfig(
        pretrain_config_path=args.pretrain_config,
        ssl_config_path=args.ssl_config,
        modality_to_data_config=modality_to_data_config
    )
    
    # Create trainer
    trainer = PretrainTrainer(config)
    
    # Start training (with optional resume)
    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
