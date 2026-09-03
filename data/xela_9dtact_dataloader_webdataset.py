#!/usr/bin/env python3
"""
Xela 9DTact Dataset Dataloader using WebDataset for TAR format.

This dataloader uses WebDataset for safe and efficient tar file loading.
All data is processed and saved in tar files - no episode structure needed.
"""

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import cv2
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import random
from collections import defaultdict, deque
import multiprocessing as mp
import io
import hashlib
from PIL import Image
import webdataset as wds
from webdataset import WebDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def xela_collate_fn(batch):
    """Custom collate function for xela_9dtact multimodal data."""
    images_data = []
    tactile_data = []
    object_idx_data = []
    
    for sample in batch:
        if 'images' in sample:
            images_data.append(sample['images'])
        if 'tactile' in sample:
            tactile_data.append(sample['tactile'])
        object_idx_data.append(sample['object_idx'])
    
    result = {
        'object_idx': torch.tensor(object_idx_data, dtype=torch.long),
    }
    
    if images_data:
        result['images'] = torch.stack(images_data, dim=0)
    if tactile_data:
        result['tactile'] = torch.stack(tactile_data, dim=0)
    
    return result


# Set multiprocessing start method to 'spawn'
if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


class Xela9DTactDatasetWebDataset(IterableDataset):
    """Dataset for Xela 9DTact using WebDataset for safe tar file loading."""
    
    # Each xela chunk in the tar file contains 10 frames (xela is 10x frequency of images)
    XELA_FRAMES_PER_CHUNK = 10
    
    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 image_chunk_size: int = 2,
                 tactile_chunk_size: int = 20,
                 apply_background_subtraction: Optional[bool] = None,
                 normalize_images: Optional[bool] = None,
                 normalize_tactile: Optional[bool] = None,
                 load_modalities: List[str] = ["tactile"],
                 difference_threshold: Optional[float] = None,
                 dataset_split_type: str = "pretrain"):
        """
        Initialize Xela 9DTact Dataset using WebDataset.
        
        Reads directly from pre-split tar files (e.g., pretrain_train.tar, pretrain_val.tar).
        No additional splitting is performed - the tar files already contain the correct split.
        Chunks are non-overlapping (no sliding window).
        
        Args:
            data_root: Path to directory containing tar files
            split: Dataset split ("train", "val", or "test")
            image_chunk_size: Number of image frames per chunk
            tactile_chunk_size: Number of tactile timesteps per chunk
            apply_background_subtraction: Whether to subtract background. If None, read from config/sensor/9dtact_config.yaml
            normalize_images: Whether to normalize image data. If None, read from config/sensor/9dtact_config.yaml
            normalize_tactile: Whether to normalize tactile data. If None, read from config/sensor/9dtact_config.yaml
            load_modalities: List of modalities to load ["images", "tactile"]
            difference_threshold: Threshold for masking images
            dataset_split_type: "pretrain" or "supervised"
        """
        # Accept either a dir that directly holds the tar shards (legacy) or
        # the released dataset root (shards under {pretrain,classification}/<pair>/).
        try:
            from data.dataset_paths import resolve_tar_dir
            data_root = resolve_tar_dir(str(data_root), dataset_split_type, "xela_9dtact")
        except FileNotFoundError:
            pass  # fall through; shard discovery below reports the error
        self.data_root = Path(data_root)
        self.split = split
        self.image_chunk_size = image_chunk_size
        self.tactile_chunk_size = tactile_chunk_size
        # Follow config defaults unless explicitly overridden
        cfg = _load_config_settings(None)
        self.apply_background_subtraction = cfg['apply_background_subtraction'] if apply_background_subtraction is None else bool(apply_background_subtraction)
        self.normalize_images = cfg['normalize_images'] if normalize_images is None else bool(normalize_images)
        self.normalize_tactile = cfg['normalize_tactile'] if normalize_tactile is None else bool(normalize_tactile)
        self.load_modalities = load_modalities
        self.difference_threshold = difference_threshold
        self.dataset_split_type = dataset_split_type
        
        # Construct full split name: {dataset_split_type}_{split}
        # e.g., "pretrain_train", "pretrain_val", "supervised_test"
        self.full_split_name = f"{dataset_split_type}_{split}"
        
        # Find tar files for this specific split
        self._find_tar_files()
        
        # Load normalization stats
        self._load_normalization_stats()
        
        # Load background
        self._load_reference_background()
        
        # Compute object mapping
        self._compute_object_mapping()
        
        # Create episode_name -> object_name mapping from manifest
        self._compute_episode_to_object_mapping()
        
        # Compute total length (cached)
        self._total_length = None
        
        print(f"{self.full_split_name.upper()} Dataset (WebDataset):")
        print(f"  Image chunk size: {self.image_chunk_size}")
        print(f"  Tactile chunk size: {self.tactile_chunk_size}")
        print(f"  Modalities: {', '.join(self.load_modalities)}")
        print(f"  Normalize images: {self.normalize_images}")
        print(f"  Normalize tactile: {self.normalize_tactile}")
        print(f"  Background subtraction: {self.apply_background_subtraction}")
    
    def _find_tar_files(self):
        """Find tar files (shards) for the specific split.
        
        Looks for sharded tar files named: {dataset_split_type}_{split}_XXXX.tar
        e.g., pretrain_train_0000.tar, pretrain_train_0001.tar, etc.
        Falls back to single tar file: {dataset_split_type}_{split}.tar if shards not found.
        
        Supports both:
        1. Sharded format: pretrain_train_0000.tar, pretrain_train_0001.tar, ...
        2. Single file format: pretrain_train.tar (for backward compatibility)
        """
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # First, try to find sharded tar files: {dataset_split_type}_{split}_XXXX.tar
        shard_pattern = f"{self.full_split_name}_*.tar"
        shard_files = sorted(list(self.data_root.glob(shard_pattern)))
        
        if shard_files:
            # Found sharded files
            self.tar_files = [str(f) for f in shard_files]
            self.tar_file = self.tar_files[0]  # Keep for backward compatibility
            print(f"Found {len(self.tar_files)} shard(s) for split '{self.full_split_name}':")
            for tar_file in self.tar_files:
                print(f"  {Path(tar_file).name}")
            return
        
        # Fallback: look for single interleaved tar file: {dataset_split_type}_{split}.tar
        tar_file = self.data_root / f"{self.full_split_name}.tar"
        
        if tar_file.exists():
            self.tar_files = [str(tar_file)]
            self.tar_file = str(tar_file)  # Keep for backward compatibility
            print(f"Found single tar file: {self.tar_file}")
            return
        
        # Neither sharded nor single file found
        raise FileNotFoundError(
            f"Tar file(s) not found for split '{self.full_split_name}' in {self.data_root}\n"
            f"  Expected sharded format: {self.full_split_name}_XXXX.tar (e.g., {self.full_split_name}_0000.tar)\n"
            f"  Or single file format: {self.full_split_name}.tar"
        )
    
    def _load_normalization_stats(self):
        """Load normalization statistics."""
        config_dir = Path(__file__).parent.parent / 'config' / 'data'
        
        # Image stats
        if self.normalize_images:
            sensor_config_path = config_dir / '9dtact.yaml'
            if sensor_config_path.exists():
                with open(sensor_config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    self.image_mean = config.get('image_mean', 0.0)
                    self.image_std = config.get('image_std', 1.0)
            else:
                self.image_mean = 0.0
                self.image_std = 1.0
        else:
            self.image_mean = None
            self.image_std = None
        
        # Tactile stats
        if self.normalize_tactile:
            sensor_config_path = config_dir / 'xela.yaml'
            if sensor_config_path.exists():
                with open(sensor_config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    tactile_mean = config.get('tactile_mean', [0.0] * 72)
                    tactile_std = config.get('tactile_std', [1.0] * 72)
                    self.tactile_mean = np.array(tactile_mean)
                    self.tactile_std = np.array(tactile_std)
                    self._tactile_mean_tensor = torch.from_numpy(self.tactile_mean).float()
                    self._tactile_std_tensor = torch.from_numpy(self.tactile_std).float()
            else:
                self.tactile_mean = None
                self.tactile_std = None
                self._tactile_mean_tensor = None
                self._tactile_std_tensor = None
        else:
            self.tactile_mean = None
            self.tactile_std = None
            self._tactile_mean_tensor = None
            self._tactile_std_tensor = None
    
    def _load_reference_background(self):
        """Load background data."""
        need_bg = (self.apply_background_subtraction or 
                  self.difference_threshold is not None)
        
        if not need_bg:
            self.reference_bg = None
            self.reference_bg_tactile = None
            return
        
        bg_data_dir = Path('data/bg_data')
        
        # Image background
        if "images" in self.load_modalities or self.apply_background_subtraction:
            bg_image_path = bg_data_dir / '9dtact.png'
            if bg_image_path.exists():
                bg_image = Image.open(bg_image_path)
                bg_array = np.array(bg_image, dtype=np.uint8)
                # Background image is treated as RGB (PIL loads RGB).
                # (Swapping removed per request: assume bg PNG is already in correct RGB order.)
                self.reference_bg = torch.from_numpy(bg_array).float() / 255.0
            else:
                self.reference_bg = None
        
        # Tactile background
        if "tactile" in self.load_modalities or self.apply_background_subtraction:
            bg_tactile_path = bg_data_dir / 'xela.npy'
            if bg_tactile_path.exists():
                bg_array = np.load(bg_tactile_path)
                self.reference_bg_tactile = torch.from_numpy(bg_array).float()
            else:
                self.reference_bg_tactile = None
    
    def _compute_object_mapping(self):
        """Compute object mapping from tar file metadata.
        
        Reads from all shards (if multiple) or single tar file to get object names.
        """
        # Load metadata from tar file(s) to get object names
        object_names = set()
        
        # Get list of tar files (sharded or single)
        tar_files = getattr(self, 'tar_files', [self.tar_file])
        
        try:
            import tarfile
            # Read from all shards to get complete object mapping
            for tar_file_path in tar_files:
                try:
                    with tarfile.open(tar_file_path, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name == '_manifest.json':
                                manifest_data = tar.extractfile(member)
                                manifest = json.load(manifest_data)
                                for episode_meta in manifest.get('episodes', []):
                                    object_name = episode_meta.get('object', 'unknown')
                                    object_names.add(object_name)
                                break
                except Exception as e:
                    # Continue to next shard if one fails
                    print(f"Warning: Could not load manifest from {Path(tar_file_path).name}: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not load object mapping from manifests: {e}")
            object_names.add('unknown')
        
        if not object_names:
            # Fallback if no objects found
            object_names.add('unknown')
        
        self.object_names = sorted(list(object_names))
        self.object_to_idx = {name: idx for idx, name in enumerate(self.object_names)}
        print(f"Found {len(self.object_names)} unique objects: {self.object_names}")
    
    def _compute_episode_to_object_mapping(self):
        """Create mapping from episode_name to object_name from manifest files."""
        self.episode_to_object = {}
        
        # Get list of tar files (sharded or single)
        tar_files = getattr(self, 'tar_files', [self.tar_file])
        
        try:
            import tarfile
            # Read from all shards to get complete episode-to-object mapping
            for tar_file_path in tar_files:
                try:
                    with tarfile.open(tar_file_path, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name == '_manifest.json':
                                manifest_data = tar.extractfile(member)
                                if manifest_data:
                                    manifest = json.load(manifest_data)
                                    for episode_meta in manifest.get('episodes', []):
                                        episode_name = episode_meta.get('episode_name', '')
                                        object_name = episode_meta.get('object', 'unknown')
                                        if episode_name:
                                            self.episode_to_object[episode_name] = object_name
                                break
                except Exception as e:
                    # Continue to next shard if one fails
                    print(f"Warning: Could not load episode-to-object mapping from {Path(tar_file_path).name}: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not load episode-to-object mapping from manifests: {e}")
        
        print(f"Created episode-to-object mapping for {len(self.episode_to_object)} episodes")
    
    def _compute_total_length(self):
        """Compute total number of chunks that can be created from the dataset.
        
        Reads manifests from all shards and calculates total chunks based on chunk sizes.
        Returns the total number of samples (chunks) that will be yielded.
        """
        if self._total_length is not None:
            return self._total_length
        
        total_chunks = 0
        tar_files = getattr(self, 'tar_files', [self.tar_file])
        
        # Calculate how many tactile chunks are needed per sample
        num_tactile_chunks_needed = 0
        if "tactile" in self.load_modalities:
            num_tactile_chunks_needed = (self.tactile_chunk_size + self.XELA_FRAMES_PER_CHUNK - 1) // self.XELA_FRAMES_PER_CHUNK
        
        try:
            import tarfile
            for tar_file_path in tar_files:
                try:
                    with tarfile.open(tar_file_path, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name == '_manifest.json':
                                manifest_data = tar.extractfile(member)
                                manifest = json.load(manifest_data)
                                
                                for episode_meta in manifest.get('episodes', []):
                                    num_images = episode_meta.get('num_images', 0)
                                    num_xela_chunks = episode_meta.get('num_xela_chunks', 0)
                                    
                                    # Calculate chunks for images
                                    if "images" in self.load_modalities:
                                        image_chunks = num_images // self.image_chunk_size if num_images >= self.image_chunk_size else 0
                                    else:
                                        image_chunks = float('inf')  # No limit from images
                                    
                                    # Calculate chunks for tactile
                                    if "tactile" in self.load_modalities:
                                        if num_tactile_chunks_needed > 0:
                                            tactile_chunks = num_xela_chunks // num_tactile_chunks_needed if num_xela_chunks >= num_tactile_chunks_needed else 0
                                        else:
                                            tactile_chunks = 0
                                    else:
                                        tactile_chunks = float('inf')  # No limit from tactile
                                    
                                    # Total chunks for this episode is the minimum (both modalities must be satisfied)
                                    if "images" in self.load_modalities and "tactile" in self.load_modalities:
                                        # Multimodal: need both, so take minimum
                                        episode_chunks = min(image_chunks, tactile_chunks)
                                    elif "images" in self.load_modalities:
                                        episode_chunks = image_chunks
                                    elif "tactile" in self.load_modalities:
                                        episode_chunks = tactile_chunks
                                    else:
                                        episode_chunks = 0
                                    
                                    total_chunks += episode_chunks
                                break
                except Exception as e:
                    print(f"Warning: Could not load manifest from {Path(tar_file_path).name} for length calculation: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not compute total length from manifests: {e}")
            # Fallback: return a large number to indicate unknown
            self._total_length = 1000000
            return self._total_length
        
        self._total_length = total_chunks
        return total_chunks
    
    def __len__(self):
        """Return the total number of chunks in the dataset.
        
        This enables len(dataset) and is used to compute iterations per epoch.
        """
        return self._compute_total_length()
    
    def _decode_image(self, data):
        """Decode JPEG image from bytes (returns RGB).
        
        NOTE: For 9dtact, the stored JPEG channel order is effectively swapped; to get
        correct RGB tensors (and correct background subtraction), we swap R<->B here
        as part of data loading/processing. Visualization should display the batch
        "as-is" without further channel changes.
        """
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None
        img_np = np.array(img, dtype=np.uint8)
        # Fix channel order at load-time (RGB <-> BGR swap)
        return img_np[:, :, [2, 1, 0]]
    
    def _decode_npy(self, data):
        """Decode numpy array from bytes."""
        return np.load(io.BytesIO(data))
    
    
    def _extract_episode_and_index(self, filename):
        """Extract episode name and index from filename.
        
        Returns: (episode_name, index) or (None, None) if parsing fails
        
        Pattern: episode_XXX_YYYYY.9dtact.jpg or episode_XXX_YYYYY.xela.npy
        e.g., episode_1_000000.9dtact.jpg -> (episode_1, 0)
        """
        # Pattern: episode_XXX_YYYYY.9dtact.jpg or episode_XXX_YYYYY.xela.npy
        parts = filename.split('_')
        if len(parts) >= 3:
            episode_name = f"{parts[0]}_{parts[1]}"
            # Index is before the first dot (e.g., "000000.9dtact.jpg" -> "000000")
            index_str = parts[-1].split('.')[0]
            try:
                index = int(index_str)
                return episode_name, index
            except ValueError:
                return None, None
        return None, None
    
    def _process_sample(self, sample):
        """Process a single WebDataset sample (automatically grouped by basename).
        
        WebDataset automatically groups files with the same basename, so this sample
        contains both the .9dtact.jpg and .xela.npy files for the same timestep.
        
        Note: WebDataset uses keys like '9dtact.jpg' and 'xela.npy' (not full filenames).
        The full basename is in '__key__' field.
        
        Returns: dict with 'episode_name', 'index', 'image', 'tactile_chunk', 'object_name'
        or None if processing fails.
        """
        result = {}
        episode_name = None
        index = None
        object_name = 'unknown'
        
        # Get the basename from __key__ (WebDataset provides this)
        # e.g., __key__ = "episode_951_000000" for files episode_951_000000.9dtact.jpg and episode_951_000000.xela.npy
        basename = sample.get('__key__', '')
        if basename:
            episode_name, index = self._extract_episode_and_index(basename)
        
        # Extract image if requested
        if "images" in self.load_modalities:
            # WebDataset uses key '9dtact.jpg' (not 'episode_XXX_YYYYY.9dtact.jpg')
            if '9dtact.jpg' in sample:
                image = self._decode_image(sample['9dtact.jpg'])
                if image is not None:
                    result['image'] = image
                    # If episode_name not extracted from __key__, try to extract from basename
                    if episode_name is None and basename:
                        episode_name, index = self._extract_episode_and_index(basename)
        
        # Extract tactile if requested
        if "tactile" in self.load_modalities:
            # WebDataset uses key 'xela.npy' (not 'episode_XXX_YYYYY.xela.npy')
            if 'xela.npy' in sample:
                tactile_chunk = self._decode_npy(sample['xela.npy'])
                if tactile_chunk is not None:
                    result['tactile_chunk'] = tactile_chunk
                    # If episode_name not extracted from __key__, try to extract from basename
                    if episode_name is None and basename:
                        episode_name, index = self._extract_episode_and_index(basename)
        
        # Extract metadata if available
        # WebDataset might use 'meta.json' as key
        for key in ['meta.json', '_meta.json']:
            if key in sample:
                try:
                    meta = json.loads(sample[key].decode('utf-8'))
                    object_name = meta.get('object', 'unknown')
                    if episode_name is None:
                        episode_name = meta.get('episode_name', None)
                except:
                    pass
        
        # If object_name is still 'unknown', try to get it from episode_to_object mapping
        if object_name == 'unknown' and episode_name and hasattr(self, 'episode_to_object'):
            object_name = self.episode_to_object.get(episode_name, 'unknown')
        
        if episode_name is None:
            return None
        
        result['episode_name'] = episode_name
        result['index'] = index
        result['object_name'] = object_name
        
        return result
    
    def _process_sequence(self, image_sequence, tactile_sequence, object_name):
        """Process a sequence of images and tactile data into tensors."""
        result = {}
        
        # Process images: stack actual consecutive frames
        if "images" in self.load_modalities and image_sequence:
            # Stack actual consecutive images (not repeated frames)
            images_array = np.stack(image_sequence, axis=0)  # [T, H, W, C]
            
            # Ensure we have the right chunk size
            num_images = len(images_array)
            if num_images >= self.image_chunk_size:
                images_chunk = images_array[:self.image_chunk_size]
            else:
                # Pad with last frame if needed
                padding = np.tile(images_array[-1:], (self.image_chunk_size - num_images, 1, 1, 1))
                images_chunk = np.concatenate([images_array, padding], axis=0)
            
            # Convert to tensor and preprocess
            images_tensor = torch.from_numpy(images_chunk).float() / 255.0
            
            # Background subtraction
            if self.apply_background_subtraction and self.reference_bg is not None:
                images_tensor -= self.reference_bg.unsqueeze(0)
                # images_tensor = torch.abs(images_tensor)
            
            # Difference threshold masking
            if self.reference_bg is not None and self.difference_threshold is not None:
                bg_expanded = self.reference_bg.unsqueeze(0)
                differences = torch.abs(images_tensor - bg_expanded)
                mean_differences = differences.mean(dim=(1, 2, 3))
                mask = mean_differences < self.difference_threshold
                images_tensor = images_tensor * (~mask).float().unsqueeze(1).unsqueeze(2).unsqueeze(3)
            
            # Normalize
            if self.normalize_images and self.image_std is not None:
                images_tensor = images_tensor / (self.image_std + 1e-6)
                # Note: No clamping for images (values don't reach clamp limits in practice)
            
            # Convert to CHW
            images_tensor = images_tensor.permute(0, 3, 1, 2)
            result['images'] = images_tensor
        
        # Process tactile: concatenate consecutive chunks
        # Each chunk from the tar file contains XELA_FRAMES_PER_CHUNK (10) frames
        # We need to stack the right number of chunks to get tactile_chunk_size frames
        if "tactile" in self.load_modalities and tactile_sequence:
            # Calculate how many chunks we need
            num_chunks_needed = (self.tactile_chunk_size + self.XELA_FRAMES_PER_CHUNK - 1) // self.XELA_FRAMES_PER_CHUNK  # Ceiling division
            
            # Take only the needed number of chunks
            chunks_to_use = tactile_sequence[:num_chunks_needed]
            
            # Concatenate the chunks
            if chunks_to_use:
                tactile_data = np.concatenate(chunks_to_use, axis=0)  # [T_total, features]
                
                # Ensure we have the right chunk size
                num_tactile = len(tactile_data)
                if num_tactile >= self.tactile_chunk_size:
                    tactile_chunk = tactile_data[:self.tactile_chunk_size]
                else:
                    # Pad with last frame
                    padding = np.tile(tactile_data[-1:], (self.tactile_chunk_size - num_tactile, 1))
                    tactile_chunk = np.concatenate([tactile_data, padding], axis=0)
            else:
                # No tactile data available, create empty tensor with default feature dim (72 for xela)
                tactile_chunk = np.zeros((self.tactile_chunk_size, 72))
            
            # Convert to tensor and preprocess
            tactile_tensor = torch.from_numpy(tactile_chunk).float()
            
            # Background subtraction
            if self.apply_background_subtraction and self.reference_bg_tactile is not None:
                tactile_tensor = tactile_tensor - self.reference_bg_tactile.unsqueeze(0)
                tactile_tensor = torch.abs(tactile_tensor)
            
            # Normalize
            if self.normalize_tactile and self._tactile_std_tensor is not None:
                tactile_tensor = tactile_tensor / (self._tactile_std_tensor + 1e-6)
                tactile_tensor = torch.clamp(tactile_tensor, min=-10.0, max=10.0)
            
            result['tactile'] = tactile_tensor
        
        # Add object_idx
        result['object_idx'] = self.object_to_idx.get(object_name, 0)
        
        return result
    
    def __iter__(self):
        """Create iterator using WebDataset with streaming sliding window approach.
        
        Uses WebDataset as a true stream (no O(N) memory load). WebDataset automatically
        groups files with the same basename, so each sample contains both image and tactile
        data for the same timestep. Supports multiprocessing via workersplitter.
        """
        # Ensure image_chunk_size is set (handle None case)
        if self.image_chunk_size is None:
            self.image_chunk_size = 2  # Default value
        if self.tactile_chunk_size is None:
            self.tactile_chunk_size = 20  # Default value
        
        # Use tar files (sharded or single file)
        # WebDataset automatically handles multiple shards and distributes them across workers
        # When using multiple shards with multiple workers, each worker gets different shards
        # This is more efficient than using a single shard with multiple workers
        tar_files = getattr(self, 'tar_files', [self.tar_file])
        
        if len(tar_files) > 1:
            # Multiple shards: WebDataset will automatically distribute across workers
            # Enable shuffling for better data distribution (use positive integer for buffer size)
            dataset = wds.WebDataset(
                tar_files,
                shardshuffle=100,  # Shuffle shards with buffer size of 100
                empty_check=False  # Allow empty shards (shouldn't happen with proper sharding)
            )
        else:
            # Single shard: use empty_check=False to avoid errors with multiple workers
            dataset = wds.WebDataset(
                tar_files,
                shardshuffle=0,  # Disable shuffling for single shard (0 = no shuffle)
                empty_check=False  # Allow empty shards when using multiple workers with single shard
            )
        
        # Process samples - WebDataset automatically groups by basename
        # The _process_sample function will only extract requested modalities
        dataset = dataset.map(self._process_sample)
        
        # Sequence buffers using deque
        image_seq = deque(maxlen=self.image_chunk_size)
        tactile_seq = deque(maxlen=self.image_chunk_size)
        current_episode = None
        prev_index = None
        object_name = 'unknown'
        
        # Stream through samples
        for sample_data in dataset:
            # Skip None results (samples that couldn't be processed)
            if sample_data is None:
                continue
            episode_name = sample_data['episode_name']
            index = sample_data['index']
            obj_name = sample_data.get('object_name', 'unknown')
            
            # Check if we're starting a new episode or non-consecutive sequence
            is_new_episode = current_episode != episode_name
            is_non_consecutive = (prev_index is not None and index != prev_index + 1)
            
            if is_new_episode or is_non_consecutive:
                # Yield previous sequence if we have frames
                if len(image_seq) > 0 or len(tactile_seq) > 0:
                    # Process what we have so far
                    seq_images = list(image_seq)[:self.image_chunk_size] if len(image_seq) >= self.image_chunk_size else list(image_seq)
                    seq_tactile = list(tactile_seq)[:self.image_chunk_size] if len(tactile_seq) >= self.image_chunk_size else list(tactile_seq)
                    if seq_images or seq_tactile:
                        result = self._process_sequence(seq_images, seq_tactile, object_name)
                        if result:
                            result['episode_name'] = current_episode
                            result['start_index'] = prev_index
                            yield result
                
                # Start new sequence
                image_seq.clear()
                tactile_seq.clear()
                current_episode = episode_name
                prev_index = index
                object_name = obj_name
            else:
                # Continue current sequence
                prev_index = index
                if object_name == 'unknown' and obj_name != 'unknown':
                    object_name = obj_name
            
            # Add to sequence buffers
            if 'image' in sample_data and sample_data['image'] is not None:
                image_seq.append(sample_data['image'])
            if 'tactile_chunk' in sample_data and sample_data['tactile_chunk'] is not None:
                tactile_seq.append(sample_data['tactile_chunk'])
            
            # Yield sequence when buffer is full (no overlap - clear buffers after yielding)
            # For images: check if we have enough image frames
            # For tactile: check if we have enough chunks (each chunk has XELA_FRAMES_PER_CHUNK frames)
            num_tactile_chunks_needed = (self.tactile_chunk_size + self.XELA_FRAMES_PER_CHUNK - 1) // self.XELA_FRAMES_PER_CHUNK if "tactile" in self.load_modalities else 0
            
            has_enough_images = len(image_seq) >= self.image_chunk_size if "images" in self.load_modalities else True
            has_enough_tactile = len(tactile_seq) >= num_tactile_chunks_needed if "tactile" in self.load_modalities else True
            
            if has_enough_images and has_enough_tactile:
                # Take the needed number of items
                images_to_use = list(image_seq)[:self.image_chunk_size] if "images" in self.load_modalities else []
                tactile_to_use = list(tactile_seq)[:num_tactile_chunks_needed] if "tactile" in self.load_modalities else []
                
                result = self._process_sequence(images_to_use, tactile_to_use, object_name)
                if result:
                    result['episode_name'] = current_episode
                    result['start_index'] = prev_index
                    yield result

                # Clear buffers after yielding (no overlap between chunks)
                image_seq.clear()
                tactile_seq.clear()
        
        # Yield remaining sequence
        if len(image_seq) > 0 or len(tactile_seq) > 0:
            seq_images = list(image_seq)[:self.image_chunk_size] if len(image_seq) >= self.image_chunk_size else list(image_seq)
            # For tactile, calculate how many chunks we need
            num_tactile_chunks_needed = (self.tactile_chunk_size + self.XELA_FRAMES_PER_CHUNK - 1) // self.XELA_FRAMES_PER_CHUNK if "tactile" in self.load_modalities else 0
            seq_tactile = list(tactile_seq)[:num_tactile_chunks_needed] if len(tactile_seq) >= num_tactile_chunks_needed else list(tactile_seq)
            if seq_images or seq_tactile:
                result = self._process_sequence(seq_images, seq_tactile, object_name)
                if result:
                    result['episode_name'] = current_episode
                    result['start_index'] = prev_index
                    yield result

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
        """Get a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # WebDataset handles shuffling internally
            num_workers=num_workers,
            collate_fn=xela_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None
        )


def create_xela_dataloaders(data_root: str,
                           batch_size: int = 32,
                           image_chunk_size: int = 5,
                           tactile_chunk_size: int = 50,
                           num_workers: int = 4,
                           dataset_split_type: str = "pretrain",
                           **kwargs):
    """Create train, validation, and test dataloaders using WebDataset.
    
    Reads directly from pre-split interleaved tar files. Each tar file contains both
    9dtact images and xela tactile data interleaved. Files are named with matching
    basenames so WebDataset automatically groups them together.
    
    The tar files should be named:
    - pretrain_train.tar, pretrain_val.tar, pretrain_test.tar
    - supervised_train.tar, supervised_val.tar, supervised_test.tar
    
    Args:
        data_root: Path to directory containing tar files
        batch_size: Batch size for dataloaders
        image_chunk_size: Number of image frames per chunk
        tactile_chunk_size: Number of tactile timesteps per chunk
        num_workers: Number of worker processes
        dataset_split_type: "pretrain" or "supervised"
        **kwargs: Additional arguments passed to dataset
    """
    
    train_dataset = Xela9DTactDatasetWebDataset(
        data_root=data_root,
        split="train",
        image_chunk_size=image_chunk_size,
        tactile_chunk_size=tactile_chunk_size,
        dataset_split_type=dataset_split_type,
        **kwargs
    )
    
    val_dataset = Xela9DTactDatasetWebDataset(
        data_root=data_root,
        split="val",
        image_chunk_size=image_chunk_size,
        tactile_chunk_size=tactile_chunk_size,
        dataset_split_type=dataset_split_type,
        **kwargs
    )
    
    test_dataset = Xela9DTactDatasetWebDataset(
        data_root=data_root,
        split="test",
        image_chunk_size=image_chunk_size,
        tactile_chunk_size=tactile_chunk_size,
        dataset_split_type=dataset_split_type,
        **kwargs
    )
    
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, num_workers=num_workers)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, num_workers=num_workers)
    test_loader = test_dataset.get_dataloader(batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def test_dataloader(data_root: str = 'data/xela_9dtact_tar',
                    batch_size: int = 4,
                    dataset_split_type: str = 'pretrain',
                    load_modalities: List[str] = ['images', 'tactile'],
                    num_workers: int = 0,
                    num_batches: int = 3):
    """Test function to initialize dataloader and sample batched data."""
    print("=" * 60)
    print("Testing Xela 9DTact WebDataset Dataloader")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset split type: {dataset_split_type}")
    print(f"  Load modalities: {load_modalities}")
    print(f"  Num workers: {num_workers}")
    print(f"  Num batches to sample: {num_batches}")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_xela_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_chunk_size=5,
        tactile_chunk_size=50,
        num_workers=num_workers,
        dataset_split_type=dataset_split_type,
        load_modalities=load_modalities,
        apply_background_subtraction=True,
        normalize_images=False,
        normalize_tactile=True
    )
    
    print(f"\nDataset info:")
    print(f"  Train split: {train_dataset.full_split_name}")
    print(f"  Val split: {val_dataset.full_split_name}")
    print(f"  Test split: {test_dataset.full_split_name}")
    print(f"  Note: WebDataset is an IterableDataset, so len() is not available")
    print()
    
    # Sample batches from train loader
    print("=" * 60)
    print(f"Sampling {num_batches} batch(es) from train loader:")
    print("=" * 60)
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Keys in batch: {list(batch.keys())}")
        
        # Print details for each key
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"\n  {key} (Tensor):")
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                print(f"    Min: {value.min().item():.6f}")
                print(f"    Max: {value.max().item():.6f}")
                
                # Only compute mean/std for floating point tensors
                if value.dtype.is_floating_point:
                    print(f"    Mean: {value.mean().item():.6f}")
                    print(f"    Std: {value.std().item():.6f}")
                
                # For images, print more details
                if key == 'images':
                    print(f"    Image tensor details:")
                    print(f"      Batch size: {value.shape[0]}")
                    print(f"      Sequence length: {value.shape[1]}")
                    print(f"      Channels: {value.shape[2]}")
                    print(f"      Height: {value.shape[3]}")
                    print(f"      Width: {value.shape[4]}")
                
                # For tactile, print more details
                if key == 'tactile':
                    print(f"    Tactile tensor details:")
                    print(f"      Batch size: {value.shape[0]}")
                    print(f"      Sequence length: {value.shape[1]}")
                    print(f"      Features: {value.shape[2]}")
                
                # For object_idx
                if key == 'object_idx':
                    print(f"    Object indices:")
                    print(f"      Batch size: {value.shape[0]}")
                    print(f"      Unique objects: {torch.unique(value).tolist()}")
                    print(f"      Object counts: {torch.bincount(value).tolist()}")
            
            elif isinstance(value, (list, tuple)):
                print(f"\n  {key}: {type(value).__name__}")
                print(f"    Length: {len(value)}")
                if len(value) > 0:
                    print(f"    First item: {value[0]}")
        
        print()
    
    print("=" * 60)
    print("✓ Dataloader test completed successfully!")
    print("=" * 60)


def _load_config_settings(config_path: Optional[str] = None) -> Dict[str, bool]:
    """Load background subtraction and normalization settings from config file.
    
    Args:
        config_path: Path to config file. If None, tries to find 9dtact_config.yaml in config/sensor/
    
    Returns:
        Dict with 'apply_background_subtraction', 'normalize_images', 'normalize_tactile'
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'sensor' / '9dtact_config.yaml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"9DTact config file not found: {config_path}. "
            f"Expected normalize settings to come from YAML."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Prefer the 'data' section (matches repo configs). Fallback to root keys if needed.
    data_cfg = config.get('data', config)

    required_keys = ['apply_background_subtraction', 'normalize_images', 'normalize_tactile']
    missing = [k for k in required_keys if k not in data_cfg]
    if missing:
        raise KeyError(
            f"Missing required key(s) {missing} in {config_path} "
            f"(looked in {'data' if 'data' in config else 'root'} section)."
        )

    return {
        'apply_background_subtraction': bool(data_cfg['apply_background_subtraction']),
        'normalize_images': bool(data_cfg['normalize_images']),
        'normalize_tactile': bool(data_cfg['normalize_tactile']),
    }


def _to_uint8_display(img: np.ndarray, *, grayscale: bool) -> np.ndarray:
    """Convert image to uint8 for display using linear rescaling.
    
    Applies linear rescaling to map any input range to [0, 255] without clipping.
    This is for visualization only and does not affect data processing.
    
    Args:
        img: Input image array (any dtype, any range)
        grayscale: Whether to convert to grayscale if 3-channel
    
    Returns:
        uint8 array with values linearly rescaled to [0, 255]
    """
    if img.ndim == 3 and grayscale:
        img = img.mean(axis=2)

    if img.dtype == np.uint8:
        return img

    # Compute min and max for linear rescaling
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    
    # Handle constant image case (min == max)
    if img_max == img_min:
        # If all values are the same, set to middle gray (128) or 0 if negative
        if img_min >= 0:
            return np.full_like(img, 128, dtype=np.uint8)
        else:
            return np.zeros_like(img, dtype=np.uint8)
    
    # Linear rescaling: map [img_min, img_max] to [0, 255]
    # Formula: (img - img_min) / (img_max - img_min) * 255
    img_normalized = (img - img_min) / (img_max - img_min)
    img_scaled = img_normalized * 255.0
    
    return np.rint(img_scaled).astype(np.uint8)


def visualize_samples(data_root: str = 'data/xela_9dtact_tar',
                      batch_size: int = 4,
                      dataset_split_type: str = 'pretrain',
                      load_modalities: List[str] = ['images', 'tactile'],
                      num_workers: int = 0,
                      num_samples: int = 4,
                      output_path: str = 'data/xela_sample_visualization.png',
                      normalize_images: Optional[bool] = None,
                      normalize_tactile: Optional[bool] = None,
                      apply_background_subtraction: Optional[bool] = None,
                      grayscale: bool = True,
                      config_path: Optional[str] = None,
                      skip_batches: int = 300,
                      seed: int = 20,
                      show_background: bool = True):
    """Visualize samples from the dataloader and save to a PNG file.
    
    Args:
        data_root: Path to tar files directory
        batch_size: Batch size for dataloader
        dataset_split_type: 'pretrain' or 'supervised'
        load_modalities: List of modalities to load ['images', 'tactile']
        num_workers: Number of worker processes
        num_samples: Number of samples to visualize
        output_path: Path to save visualization PNG
        normalize_images: Whether images are normalized. If None, loads from config file.
        normalize_tactile: Whether tactile data is normalized. If None, loads from config file.
        apply_background_subtraction: Whether to subtract background. If None, loads from config file.
        grayscale: Display as grayscale (default: True)
        config_path: Path to config file. If None, uses config/sensor/9dtact_config.yaml
        show_background: Whether to show background image in visualization (default: True)
    """
    # Load settings from config file if not explicitly provided
    config_settings = _load_config_settings(config_path)
    
    if apply_background_subtraction is None:
        apply_background_subtraction = config_settings['apply_background_subtraction']
    if normalize_images is None:
        normalize_images = config_settings['normalize_images']
    if normalize_tactile is None:
        normalize_tactile = config_settings['normalize_tactile']
    
    print("=" * 60)
    print("Visualizing Xela 9DTact Samples")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset split type: {dataset_split_type}")
    print(f"  Load modalities: {load_modalities}")
    print(f"  Num workers: {num_workers}")
    print(f"  Num samples: {num_samples}")
    print(f"  Output path: {output_path}")
    print(f"  Background subtraction: {apply_background_subtraction} (from config)")
    print(f"  Normalize images: {normalize_images} (from config)")
    print(f"  Normalize tactile: {normalize_tactile} (from config)")
    print(f"  Grayscale: {grayscale}")
    print(f"  Show background image: {show_background}")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_xela_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_chunk_size=5,
        tactile_chunk_size=50,
        num_workers=num_workers,
        dataset_split_type=dataset_split_type,
        load_modalities=load_modalities,
        apply_background_subtraction=apply_background_subtraction,
        normalize_images=normalize_images,
        normalize_tactile=normalize_tactile
    )
    
    # Get a batch (optionally skip some to sample different data)
    print("Loading batch...")
    it = iter(train_loader)
    if seed:
        random.seed(seed)
    if skip_batches and skip_batches > 0:
        for _ in range(skip_batches):
            try:
                next(it)
            except StopIteration:
                it = iter(train_loader)
                break
    batch = next(it)
    
    # Determine number of samples to visualize
    actual_num_samples = min(num_samples, batch_size)
    if 'images' in batch:
        actual_num_samples = min(actual_num_samples, batch['images'].shape[0])
    elif 'tactile' in batch:
        actual_num_samples = min(actual_num_samples, batch['tactile'].shape[0])
    
    # Optionally load background image for visualization (independent of subtraction setting)
    bg_img_u8 = None
    if show_background and ("images" in load_modalities):
        bg_path = Path("data/bg_data") / "9dtact.png"
        if bg_path.exists():
            try:
                bg_pil = Image.open(bg_path)
                bg_np = np.array(bg_pil, dtype=np.uint8)  # PIL loads RGB
                # Background is already in RGB format, no conversion needed
                bg_img_u8 = _to_uint8_display(bg_np, grayscale=grayscale)
            except Exception as e:
                print(f"  Warning: could not load background image {bg_path}: {e}")

    # Create figure
    has_images = 'images' in batch and 'images' in load_modalities
    has_tactile = 'tactile' in batch and 'tactile' in load_modalities

    extra_rows = 1 if (has_images and bg_img_u8 is not None) else 0
    
    if has_images and has_tactile:
        # Create grid: rows = samples, cols = image frames + tactile
        num_image_frames = batch['images'].shape[1] if has_images else 0
        fig = plt.figure(figsize=(4 * (num_image_frames + 1), 3 * (actual_num_samples + extra_rows)))
        gs = gridspec.GridSpec(actual_num_samples + extra_rows, num_image_frames + 1, figure=fig, hspace=0.3, wspace=0.3)
    elif has_images:
        num_image_frames = batch['images'].shape[1]
        fig = plt.figure(figsize=(4 * num_image_frames, 3 * (actual_num_samples + extra_rows)))
        gs = gridspec.GridSpec(actual_num_samples + extra_rows, num_image_frames, figure=fig, hspace=0.3, wspace=0.3)
    elif has_tactile:
        fig = plt.figure(figsize=(12, 3 * actual_num_samples))
        gs = gridspec.GridSpec(actual_num_samples, 1, figure=fig, hspace=0.3, wspace=0.3)
    else:
        print("No images or tactile data to visualize!")
        return

    # Plot background row if available
    if extra_rows == 1:
        ax_bg = fig.add_subplot(gs[0, :])
        if grayscale:
            ax_bg.imshow(bg_img_u8, cmap='gray', vmin=0, vmax=255)
        else:
            ax_bg.imshow(bg_img_u8)
        ax_bg.set_title("Background (9dtact.png)", fontsize=10)
        ax_bg.axis('off')
    
    for sample_idx in range(actual_num_samples):
        # Visualize images
        row_idx = sample_idx + extra_rows
        if has_images:
            images = batch['images'][sample_idx]  # [T, C, H, W]
            num_frames = images.shape[0]
            
            for frame_idx in range(num_frames):
                img = images[frame_idx]  # [C, H, W]
                
                # Convert to numpy: [C, H, W] -> [H, W, C]
                img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                
                # Apply linear rescaling to [0, 255] for visualization (does not affect data processing)
                img_u8 = _to_uint8_display(img_np, grayscale=grayscale)
                
                # Display
                ax = fig.add_subplot(gs[row_idx, frame_idx])
                if grayscale:
                    if img_u8.ndim == 2:
                        ax.imshow(img_u8, cmap='gray', vmin=0, vmax=255)
                    else:
                        # Convert 3-channel to grayscale for display
                        img_gray = img_u8.mean(axis=2) if img_u8.ndim == 3 else img_u8
                        ax.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
                else:
                    if img_u8.ndim == 2:
                        ax.imshow(img_u8, cmap='gray', vmin=0, vmax=255)
                    else:
                        ax.imshow(img_u8)
                ax.set_title(f'Sample {sample_idx+1}, Frame {frame_idx+1}', fontsize=10)
                ax.axis('off')
        
        # Visualize tactile
        if has_tactile:
            tactile = batch['tactile'][sample_idx]  # [T, D]
            tactile_np = tactile.cpu().numpy()
            
            # Create tactile visualization
            if has_images:
                ax = fig.add_subplot(gs[row_idx, -1])
            else:
                ax = fig.add_subplot(gs[row_idx, 0])
            
            # Plot tactile data as heatmap or line plot
            # For heatmap: time (T) x features (D)
            im = ax.imshow(tactile_np.T, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Tactile Feature')
            ax.set_title(f'Sample {sample_idx+1} - Tactile', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Add object info if available
            if 'object_idx' in batch:
                obj_idx = batch['object_idx'][sample_idx].item()
                if hasattr(train_dataset, 'object_names'):
                    obj_name = train_dataset.object_names[obj_idx] if obj_idx < len(train_dataset.object_names) else f'Object {obj_idx}'
                    ax.text(0.02, 0.98, f'Object: {obj_name}', 
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Xela 9DTact Dataset Visualization ({dataset_split_type})', fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()
    
    print("=" * 60)
    print("✓ Visualization completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test/Visualize Xela 9DTact WebDataset dataloader with tar files')
    parser.add_argument('--data-root', type=str, default='data/xela_9dtact_tar',
                       help='Path to tar files directory')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--dataset-split-type', type=str, default='pretrain', choices=['pretrain', 'supervised'],
                       help='Dataset split type: pretrain or supervised')
    parser.add_argument('--load-modalities', type=str, nargs='+', default=['images', 'tactile'],
                       choices=['images', 'tactile'],
                       help='Modalities to load')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of worker processes (0 for single-threaded)')
    parser.add_argument('--num-batches', type=int, default=3,
                       help='Number of batches to sample (for test mode)')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'visualize'],
                       help='Mode: test (print stats) or visualize (save images)')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of samples to visualize (for visualize mode)')
    parser.add_argument('--output-path', type=str, default='data/xela_sample_visualization.png',
                       help='Output path for visualization PNG')
    # Override switches: default None means "use config"
    norm_img_group = parser.add_mutually_exclusive_group()
    norm_img_group.add_argument('--normalize-images', dest='normalize_images', action='store_true',
                                help='Enable image normalization (overrides config)')
    norm_img_group.add_argument('--no-normalize-images', dest='normalize_images', action='store_false',
                                help='Disable image normalization (overrides config)')
    parser.set_defaults(normalize_images=None)

    norm_tac_group = parser.add_mutually_exclusive_group()
    norm_tac_group.add_argument('--normalize-tactile', dest='normalize_tactile', action='store_true',
                                help='Enable tactile normalization (overrides config)')
    norm_tac_group.add_argument('--no-normalize-tactile', dest='normalize_tactile', action='store_false',
                                help='Disable tactile normalization (overrides config)')
    parser.set_defaults(normalize_tactile=None)

    bg_group = parser.add_mutually_exclusive_group()
    bg_group.add_argument('--background-subtraction', dest='apply_background_subtraction', action='store_true',
                          help='Enable background subtraction (overrides config)')
    bg_group.add_argument('--no-background-subtraction', dest='apply_background_subtraction', action='store_false',
                          help='Disable background subtraction (overrides config)')
    parser.set_defaults(apply_background_subtraction=None)
    parser.add_argument('--rgb', action='store_true',
                       help='Display as RGB (default: grayscale)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to config file (default: config/sensor/9dtact_config.yaml)')
    parser.add_argument('--skip-batches', type=int, default=400,
                       help='Skip N batches before visualizing (useful to sample different data)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed for reproducibility (affects shard shuffle / skipping behavior)')
    parser.add_argument('--no-show-background', dest='show_background', action='store_false',
                       help='Do not include background image in visualization')
    parser.set_defaults(show_background=True)
    
    args = parser.parse_args()
    
    if args.mode == 'visualize':
        # None => use config; otherwise override
        normalize_images = args.normalize_images
        normalize_tactile = args.normalize_tactile
        apply_bg_sub = args.apply_background_subtraction
        
        visualize_samples(
            data_root=args.data_root,
            batch_size=args.batch_size,
            dataset_split_type=args.dataset_split_type,
            load_modalities=args.load_modalities,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            output_path=args.output_path,
            normalize_images=normalize_images,
            normalize_tactile=normalize_tactile,
            apply_background_subtraction=apply_bg_sub,
            grayscale=not args.rgb,
            config_path=args.config_path,
            skip_batches=args.skip_batches,
            seed=args.seed,
            show_background=args.show_background
        )
    else:
        test_dataloader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            dataset_split_type=args.dataset_split_type,
            load_modalities=args.load_modalities,
            num_workers=args.num_workers,
            num_batches=args.num_batches
        )
