#!/usr/bin/env python3
"""
Standalone dataloader creation function.
This module provides a unified interface for creating dataloaders based on configuration.
"""

import os
from typing import Tuple, Optional, Dict, Any
from omegaconf import OmegaConf

# Import dataloaders (WebDataset-based)
from data.xela_9dtact_dataloader_webdataset import create_xela_dataloaders
from data.tacniq_gsmini_dataloader_webdataset import create_tacniq_dataloaders


def create_dataloaders_from_config(config: OmegaConf) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Create dataloaders based on configuration.
    
    Args:
        config: Configuration object containing data parameters
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
        
    Raises:
        ValueError: If dataset_type or training_type is not supported
        FileNotFoundError: If data_root path does not exist
    """
    
    # Extract configuration parameters
    dataset_type = config.get('dataset_type', config.get('data.dataset_type'))
    training_type = config.get('training_type', config.get('data.training_type'))
    data_root = config.get('data_root', config.get('data.data_root'))
    
    # Validate required parameters
    if not dataset_type:
        raise ValueError("dataset_type must be specified in config")
    if not data_root:
        raise ValueError("data_root must be specified in config")
    
    # Check if data root exists
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root path does not exist: {data_root}")
    
    print(f"Creating dataloaders for {training_type} training with {dataset_type} dataset")
    print(f"  Data root: {data_root}")
    
    # Extract common parameters
    batch_size = config.get('batch_size', config.get('data.batch_size'))
    num_workers = config.get('num_workers', config.get('data.num_workers', 4))  # Default to 4 if not specified
    
    # Ensure num_workers is an integer
    if num_workers is not None:
        num_workers = int(num_workers)
    else:
        num_workers = 4  # Default fallback
    
    # Extract optional parameters with defaults
    image_chunk_size = config.get('image_chunk_size', config.get('data.image_chunk_size', 2))
    tactile_chunk_size = config.get('tactile_chunk_size', config.get('data.tactile_chunk_size', 20))
    apply_background_subtraction = config.get('apply_background_subtraction', config.get('data.apply_background_subtraction'))
    normalize_images = config.get('normalize_images', config.get('data.normalize_images'))
    normalize_tactile = config.get('normalize_tactile', config.get('data.normalize_tactile'))
    dataset_split_type = config.get('dataset_split_type', config.get('data.dataset_split_type', 'pretrain'))
    difference_threshold = config.get('difference_threshold', config.get('data.difference_threshold'))
    
    # Determine load modalities based on training type
    # Note: WebDataset dataloaders don't return 'bg' data
    if training_type == 'images':
        load_modalities = ['images']
    elif training_type == 'tactile':
        load_modalities = ['tactile']
    elif training_type == 'multimodal':
        load_modalities = ['images', 'tactile']
    else:
        raise ValueError(f"Unsupported training_type: {training_type}")
    
    # Create dataloaders based on dataset type
    # WebDataset dataloaders read from pre-split tar files, so no train/val/test ratios needed
    if dataset_type == 'xela_9dtact':
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_xela_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            image_chunk_size=image_chunk_size,
            tactile_chunk_size=tactile_chunk_size,
            num_workers=num_workers,
            dataset_split_type=dataset_split_type,
            load_modalities=load_modalities,
            apply_background_subtraction=apply_background_subtraction,
            normalize_images=normalize_images,
            normalize_tactile=normalize_tactile,
            difference_threshold=difference_threshold
        )
        
    elif dataset_type == 'tacniq_gsmini':
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_tacniq_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            image_chunk_size=image_chunk_size,
            tactile_chunk_size=tactile_chunk_size,
            num_workers=num_workers,
            dataset_split_type=dataset_split_type,
            load_modalities=load_modalities,
            apply_background_subtraction=apply_background_subtraction,
            normalize_images=normalize_images,
            normalize_tactile=normalize_tactile,
            difference_threshold=difference_threshold
        )
    
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")
    
    # Print dataloader information
    print(f"Dataloaders created successfully:")
    # WebDataset is an IterableDataset, so len() is not available
    print(f"  Train split: {train_dataset.full_split_name}")
    if val_dataset is not None:
        print(f"  Validation split: {val_dataset.full_split_name}")
    print(f"  Test split: {test_dataset.full_split_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Training type: {training_type}")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Load modalities: {load_modalities}")
    print(f"  Note: WebDataset is an IterableDataset, so len() is not available")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def get_num_classes_from_dataset(train_dataset: Any) -> int:
    """
    Extract number of classes from dataset.
    
    Args:
        train_dataset: Training dataset object
        
    Returns:
        Number of classes
    """
    if hasattr(train_dataset, 'num_classes'):
        return train_dataset.num_classes
    elif hasattr(train_dataset, 'object_to_idx'):
        return len(train_dataset.object_to_idx)
    else:
        # Try to infer from dataset attributes
        if hasattr(train_dataset, 'classes'):
            return len(train_dataset.classes)
        elif hasattr(train_dataset, 'labels'):
            return len(set(train_dataset.labels))
        else:
            return 20  # Default fallback


def create_dataloaders_from_config_file(config_path: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Create dataloaders from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = OmegaConf.create(config_dict)
    return create_dataloaders_from_config(config)


# Example usage
if __name__ == "__main__":
    # Example configuration
    example_config = OmegaConf.create({
        'dataset_type': 'xela_9dtact',
        'training_type': 'multimodal',
        'data_root': 'data/xela_9dtact_tar',  # Updated to tar directory
        'batch_size': 32,
        'image_chunk_size': 5,
        'tactile_chunk_size': 50,
        'num_workers': 4,
        'dataset_split_type': 'pretrain'
    })
    
    try:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders_from_config(example_config)
        num_classes = get_num_classes_from_dataset(train_dataset)
        print(f"Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
