#!/usr/bin/env python3
"""
Data augmentation utilities for tactile and multimodal data.

This module provides various augmentation techniques specifically designed for:
- Tactile sensor data (time series)
- Image data (if needed for multimodal training)
- Combined multimodal augmentations

The augmentations are designed to be realistic and preserve the physical meaning
of tactile sensor readings while increasing data diversity.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Optional, Union, Tuple
import torch.nn.functional as F


def create_temporal_mask(x: torch.Tensor, mask_ratio: float = 0.2, mask_length: int = 1) -> torch.Tensor:
    """
    Create random, non-overlapping temporal masks for 3D input tensors (batch, length, features).
    Uses fixed-length masks with random positions, ensuring no overlap between masks.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, feature_dim]
        mask_ratio: Ratio of sequence length to mask (default: 0.2)
        mask_length: Fixed length of each mask segment (default: 1)
        
    Returns:
        mask: Boolean mask tensor of shape [batch_size, seq_len] where True indicates masked positions
    """
    batch_size, seq_len, feature_dim = x.shape
    device = x.device   
    
    total_masked_positions = int(seq_len * mask_ratio)
    num_masks_needed = total_masked_positions // mask_length

    # --- New Logic for Non-Overlapping Selection ---
    max_start_pos = seq_len - mask_length
    
    if max_start_pos < 0 or num_masks_needed == 0:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    max_possible_masks = seq_len // mask_length
    num_masks = min(num_masks_needed, max_possible_masks)
    
    # Create an index pool for non-overlapping *mask segments*.
    # The segments are 0, 1, 2, ... (max_possible_masks - 1)
    # Start position of segment 'i' is i * mask_length  
    segment_indices = torch.arange(max_possible_masks, device=device)
    
    # Generate random selection for each batch item
    rand_floats = torch.rand(batch_size, max_possible_masks, device=device)
    perm_indices = torch.argsort(rand_floats, dim=1)
    
    # Select the first 'num_masks' segments randomly
    selected_segment_indices = segment_indices[perm_indices[:, :num_masks]] # [batch_size, num_masks]
    
    # Convert the selected segment index back into the absolute sequence start position
    selected_starts = selected_segment_indices * mask_length # [batch_size, num_masks]

    # --- Mask Creation (Same as before, but with non-overlapping 'selected_starts') ---

    # Initialize mask
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Calculate all absolute indices for the mask segments
    offset_indices = torch.arange(mask_length, device=device) # [mask_length]
    # selected_starts [batch_size, num_masks, 1] + offset_indices [mask_length] -> [batch_size, num_masks, mask_length]
    absolute_indices = selected_starts.unsqueeze(-1) + offset_indices 
    
    # Clamp is not strictly necessary here because of how max_possible_masks is calculated,
    # but it acts as a safeguard.
    absolute_indices = torch.clamp(absolute_indices, min=0, max=seq_len - 1)
    
    # Flatten the indices for scatter_
    indices_to_mask = absolute_indices.view(batch_size, -1)

    # Apply the mask using scatter_
    # This sets the positions specified by indices_to_mask to True
    mask = mask.scatter_(1, indices_to_mask, True)

    return mask


def create_spatio_temporal_mask_3d(x: torch.Tensor, mask_ratio: float = 0.2, mask_t: int = 1, mask_h: int = 1, mask_w: int = 1) -> torch.Tensor:
    """
    Create random, non-overlapping 3D spatio-temporal masks for input tensors 
    of shape [batch_size, T, H, W, C].

    The masking is applied by selecting non-overlapping 3D blocks (T_block, H_block, W_block) 
    and setting them to True (masked).

    Args:
        x: Input tensor of shape [batch_size, T, H, W, C].
        mask_ratio: Target ratio of the total T*H*W volume to mask (e.g., 0.2).
        mask_t, mask_h, mask_w: Fixed dimensions of the 3D masking block.
        
    Returns:
        mask: Boolean mask tensor of shape [batch_size, T, H, W] where True indicates 
              masked positions.
    """
    batch_size, seq_t, seq_h, seq_w, feature_dim = x.shape
    device = x.device

    # 1. Define the virtual grid (non-overlapping 3D blocks)
    # T_grid: Number of blocks along the T-dimension
    T_grid = seq_t // mask_t
    H_grid = seq_h // mask_h
    W_grid = seq_w // mask_w

    num_total_blocks = T_grid * H_grid * W_grid
    
    if num_total_blocks == 0:
        return torch.zeros(batch_size, seq_t, seq_h, seq_w, dtype=torch.bool, device=device)

    # 2. Calculate number of blocks to mask
    # This ensures the masked volume roughly satisfies the mask_ratio.
    num_blocks_to_mask = int(num_total_blocks * mask_ratio)
    num_blocks_to_mask = min(num_blocks_to_mask, num_total_blocks) # Safety clamp

    # 3. Randomly select the blocks to mask
    # Create an index pool for all blocks in the T*H*W grid
    block_indices = torch.arange(num_total_blocks, device=device) # [Num_Blocks]
    
    # Generate random selection for each batch item
    # rand_floats: [batch_size, Num_Blocks]
    rand_floats = torch.rand(batch_size, num_total_blocks, device=device)
    
    # perm_indices: Indices that sort rand_floats. The first 'num_blocks_to_mask'
    # indices are the ones we select randomly.
    perm_indices = torch.argsort(rand_floats, dim=1)
    
    # Select the indices of the blocks to be masked
    # selected_mask_block_indices: [batch_size, num_blocks_to_mask]
    selected_mask_block_indices = block_indices[perm_indices[:, :num_blocks_to_mask]]

    # 4. Create the mask tensor in block space
    # mask_block_space: [batch_size, Num_Blocks] -> True for blocks to mask
    mask_block_space = torch.zeros(batch_size, num_total_blocks, dtype=torch.bool, device=device)
    
    # Use scatter to mark the selected blocks as True
    mask_block_space.scatter_(1, selected_mask_block_indices, True)

    # 5. Reshape and expand the mask to (T, H, W) resolution
    
    # Reshape: [batch_size, T_grid, H_grid, W_grid]
    mask_block_space = mask_block_space.view(batch_size, T_grid, H_grid, W_grid)

    # Expand: Repeat the True/False values across the block dimensions
    # torch.repeat_interleave is used to expand from grid space to full resolution.
    
    # Expand T-dimension: [B, T_grid, H_grid, W_grid] -> [B, T, H_grid, W_grid]
    mask = torch.repeat_interleave(mask_block_space, repeats=mask_t, dim=1)
    
    # Expand H-dimension: [B, T, H_grid, W_grid] -> [B, T, H, W_grid]
    mask = torch.repeat_interleave(mask, repeats=mask_h, dim=2)
    
    # Expand W-dimension: [B, T, H, W_grid] -> [B, T, H, W]
    mask = torch.repeat_interleave(mask, repeats=mask_w, dim=3)
    
    # Final clamping to handle cases where T, H, or W are not perfectly divisible by mask_t, mask_h, mask_w.
    mask = mask[:, :seq_t, :seq_h, :seq_w]
    
    return mask

def apply_mask(x: torch.Tensor, mask: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """
    Apply a temporal mask to the input tensor.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, features]
        mask: Boolean mask tensor of shape [batch_size, seq_len]. True indicates masked positions.
        mask_value: Value to use for masked positions (default: 0.0)
        
    Returns:
        masked_x: Tensor with masked positions set to mask_value
    """
    masked_x = x.clone()
    mask_expanded = mask.unsqueeze(-1)
    mask_expanded = mask_expanded.expand_as(x)
    masked_x[mask_expanded] = mask_value
    return masked_x

class TactileAugmentation:
    """
    Tactile data augmentation class for time series tactile sensor data.
    
    Supports various augmentation techniques that are physically meaningful
    for tactile sensor readings.
    """
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 dropout_prob: float = 0.1,
                 scaling_range: Tuple[float, float] = (0.9, 1.1),
                 time_warp_range: Tuple[float, float] = (0.95, 1.05),
                 temporal_masking_ratio: float = 0.2,
                 temporal_mask_length: int = 1,
                 enable_all: bool = True,
                 augmentation_prob: float = 0.5,
                 h: Optional[int] = None,
                 w: Optional[int] = None,
                 input_dim: Optional[int] = None,
                 mask_t: int = 1,
                 mask_h: int = 1,
                 mask_w: int = 1,
                 **kwargs):
        """
        Initialize tactile augmentation parameters.
        
        Args:
            noise_std: Standard deviation for Gaussian noise injection
            dropout_prob: Probability of dropping out tactile readings
            scaling_range: Range for random scaling of tactile values
            time_warp_range: Range for temporal warping (stretch/compress)
            temporal_masking_ratio: Ratio of sequence length to mask
            temporal_mask_length: Fixed length of each mask segment
            enable_all: Whether to enable all augmentations by default
            augmentation_prob: Probability of applying any augmentation (0.0-1.0)
            h: Spatial height dimension for spatio-temporal masking
            w: Spatial width dimension for spatio-temporal masking
            input_dim: Number of feature channels per spatial location
            mask_t: Temporal mask dimension for spatio-temporal masking
            mask_h: Height mask dimension for spatio-temporal masking
            mask_w: Width mask dimension for spatio-temporal masking
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.scaling_range = scaling_range
        self.time_warp_range = time_warp_range
        self.temporal_masking_ratio = temporal_masking_ratio
        self.temporal_mask_length = temporal_mask_length
        self.enable_all = enable_all
        self.augmentation_prob = augmentation_prob
        self.h = h
        self.w = w
        self.input_dim = input_dim
        self.mask_t = mask_t
        self.mask_h = mask_h
        self.mask_w = mask_w
        
        # Individual augmentation flags
        self.enable_noise = kwargs.get('enable_noise', enable_all)
        self.enable_dropout = kwargs.get('enable_dropout', enable_all)
        self.enable_scaling = kwargs.get('enable_scaling', enable_all)
        self.enable_time_warp = kwargs.get('enable_time_warp', enable_all)
        self.enable_temporal_masking = kwargs.get('enable_temporal_masking', enable_all)
        self.enable_spatio_temporal_masking = kwargs.get('enable_spatio_temporal_masking', False)

        
    def add_noise(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to tactile readings.
        
        Args:
            tactile: Tactile tensor of shape [batch_size, seq_len, features] or [seq_len, features]
            
        Returns:
            Augmented tactile tensor with noise
        """
        if not self.enable_noise:
            return tactile
            
        # Generate noise with same shape as tactile data
        noise = torch.randn_like(tactile) * self.noise_std
        return tactile + noise
    
    def apply_dropout(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Randomly zero out some tactile readings (simulate sensor failures).
        
        Args:
            tactile: Tactile tensor of shape [batch_size, seq_len, features] or [seq_len, features]
            
        Returns:
            Augmented tactile tensor with some readings zeroed
        """
        if not self.enable_dropout:
            return tactile
            
        # Create dropout mask
        dropout_mask = torch.rand_like(tactile) > self.dropout_prob
        return tactile * dropout_mask.float()
    
    def apply_scaling(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Randomly scale tactile values (simulate different contact forces).
        
        Args:
            tactile: Tactile tensor of shape [batch_size, seq_len, features] or [seq_len, features]
            
        Returns:
            Augmented tactile tensor with scaled values
        """
        if not self.enable_scaling:
            return tactile
        seq_len = tactile.shape[1]
        final_scale_factor = random.uniform(self.scaling_range[0], self.scaling_range[1])
        time_steps = torch.arange(seq_len, dtype=tactile.dtype, device=tactile.device)
        normalized_time = time_steps / (seq_len - 1)
        scaling_multiplier = 1.0 + (final_scale_factor - 1.0) * normalized_time
        scaling_multiplier = scaling_multiplier.view(1, seq_len, 1)

        return tactile * scaling_multiplier

    
    def apply_masking(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Apply non-overlapping temporal masking to tactile data.
        
        Args:
            tactile: Tactile tensor of shape [batch_size, seq_len, features]
            
        Returns:
            Masked tactile tensor with zeros at masked positions
        """
        if not self.enable_temporal_masking:
            return tactile
        
        mask = create_temporal_mask(tactile, self.temporal_masking_ratio, self.temporal_mask_length)
        return apply_mask(tactile, mask)

    def apply_spatio_temporal_masking(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Apply non-overlapping spatio-temporal masking to tactile data.
        
        Args:
            tactile: Tactile tensor of shape [batch_size, seq_len, features]
            
        Returns:
            Masked tactile tensor with zeros at masked positions
        """
        if not self.enable_spatio_temporal_masking:
            return tactile
        
        if self.h is None or self.w is None or self.input_dim is None:
            raise ValueError("h, w, and input_dim must be provided for spatio-temporal masking")
        
        if tactile.dim() == 3:
            batch_size, seq_len, num_features = tactile.shape
            expected_features = self.h * self.w * self.input_dim
            
            if num_features != expected_features:
                raise ValueError(f"Expected {expected_features} features (h={self.h} * w={self.w} * input_dim={self.input_dim}), but got {num_features}")
            
            # Reshape to [batch_size, T, H, W, C]
            tactile_5d = tactile.view(batch_size, seq_len, self.h, self.w, self.input_dim)
        elif tactile.dim() == 5:
            tactile_5d = tactile
        else:
            raise ValueError(f"Tactile tensor must be 3D or 5D, but got {tactile.dim()} dimensions")
        
        # Create 4D spatio-temporal mask
        mask = create_spatio_temporal_mask_3d(
            tactile_5d, 
            mask_ratio=self.temporal_masking_ratio, 
            mask_t=self.mask_t, 
            mask_h=self.mask_h, 
            mask_w=self.mask_w
        )
        
        # Apply mask (mask is [batch_size, T, H, W])
        masked_tactile_5d = tactile_5d.clone()
        mask_expanded = mask.unsqueeze(-1)  # [batch_size, T, H, W, 1]
        mask_expanded = mask_expanded.expand_as(tactile_5d)  # [batch_size, T, H, W, C]
        masked_tactile_5d[mask_expanded] = 0.0
        
        if tactile.dim() == 3:
            # Reshape back to [batch_size, seq_len, features]
            masked_tactile = masked_tactile_5d.view(batch_size, seq_len, num_features)
        elif tactile.dim() == 5:
            masked_tactile = masked_tactile_5d
        
        return masked_tactile
    
    def apply_time_warping(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal warping to simulate different interaction speeds.
        
        Args:
            tactile: Tactile tensor of shape [batch_size, seq_len, features] or [seq_len, features]
            
        Returns:
            Augmented tactile tensor with temporal warping
        """
        if not self.enable_time_warp:
            return tactile
            

        batch_size, seq_len, features = tactile.shape
        
        # Generate a random warp factor
        warp_factor = random.uniform(self.time_warp_range[0], self.time_warp_range[1])
        new_len = int(seq_len * warp_factor)

        # Permute to [batch_size, features, seq_len] for F.interpolate
        # F.interpolate works on the last dimension by default for 3D tensors
        tactile_permuted = tactile.permute(0, 2, 1)
        
        # Apply resampling
        warped_tactile_permuted = F.interpolate(
            tactile_permuted, 
            size=new_len, 
            mode='linear', 
            align_corners=False # Generally recommended for signal data
        )
        
        # Resize back to the original sequence length
        # This stretches or squashes the warped signal
        final_warped_permuted = F.interpolate(
            warped_tactile_permuted,
            size=seq_len,
            mode='linear',
            align_corners=False
        )
        
        # Permute back to [batch_size, seq_len, features]
        warped_tactile = final_warped_permuted.permute(0, 2, 1)

            
        return warped_tactile
    
    def __call__(self, tactile: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply all enabled augmentations to tactile data.
        
        Args:
            tactile: Tactile tensor
            training: Whether in training mode (augmentations only applied during training)
            
        Returns:
            Augmented tactile tensor
        """
        if not training:
            return tactile
        
        # Determine if we should apply any augmentation based on probability
        should_augment = random.random() < self.augmentation_prob
        
        if not should_augment:
            return tactile
        
        # Apply augmentations in sequence
        tactile = self.add_noise(tactile)
        tactile = self.apply_dropout(tactile)
        tactile = self.apply_scaling(tactile)
        tactile = self.apply_time_warping(tactile)
        tactile = self.apply_masking(tactile)
        tactile = self.apply_spatio_temporal_masking(tactile)
        
        return tactile


class ImageAugmentation:
    """
    Image augmentation class for visual data (if needed for multimodal training).
    
    Provides standard computer vision augmentations suitable for tactile images.
    """
    
    def __init__(self, 
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 rotation_range: Tuple[float, float] = (-10, 10),
                 enable_all: bool = True,
                 augmentation_prob: float = 0.5,
                 **kwargs):
        """
        Initialize image augmentation parameters.
        
        Args:
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            rotation_range: Range for rotation in degrees
            enable_all: Whether to enable all augmentations by default
            augmentation_prob: Probability of applying any augmentation (0.0-1.0)
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.enable_all = enable_all
        self.augmentation_prob = augmentation_prob
        
        # Individual augmentation flags
        self.enable_brightness = kwargs.get('enable_brightness', enable_all)
        self.enable_contrast = kwargs.get('enable_contrast', enable_all)
        self.enable_rotation = kwargs.get('enable_rotation', enable_all)
    
    def adjust_brightness(self, images: torch.Tensor) -> torch.Tensor:
        """Adjust brightness of images."""
        if not self.enable_brightness:
            return images
            
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        # Handle different tensor shapes
        if images.dim() == 4:  # [batch_size, channels, height, width]
            return F.adjust_brightness(images, brightness_factor)
        elif images.dim() == 3:  # [channels, height, width]
            return F.adjust_brightness(images, brightness_factor)
        else:
            return images
    
    def adjust_contrast(self, images: torch.Tensor) -> torch.Tensor:
        """Adjust contrast of images."""
        if not self.enable_contrast:
            return images
            
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # Handle different tensor shapes
        if images.dim() == 4:  # [batch_size, channels, height, width]
            return F.adjust_contrast(images, contrast_factor)
        elif images.dim() == 3:  # [channels, height, width]
            return F.adjust_contrast(images, contrast_factor)
        else:
            return images
    
    def rotate(self, images: torch.Tensor) -> torch.Tensor:
        """Rotate images."""
        if not self.enable_rotation:
            return images
            
        angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        
        # Handle different tensor shapes
        if images.dim() == 4:  # [batch_size, channels, height, width]
            return F.rotate(images, angle)
        elif images.dim() == 3:  # [channels, height, width]
            return F.rotate(images, angle)
        else:
            return images
    
    def __call__(self, images: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply all enabled augmentations to image data.
        
        Args:
            images: Image tensor
            training: Whether in training mode
            
        Returns:
            Augmented image tensor
        """
        if not training:
            return images
        
        # Check if augmentation should be applied based on probability
        should_augment = random.random() <= self.augmentation_prob
        
        if not should_augment:
            return images
            
        # Apply augmentations in sequence
        images = self.adjust_brightness(images)
        images = self.adjust_contrast(images)
        images = self.rotate(images)
        
        return images


class MultimodalAugmentation:
    """
    Combined augmentation for multimodal data (tactile + visual).
    
    Ensures augmentations are applied consistently across modalities.
    """
    
    def __init__(self, tactile_config: Dict = None, image_config: Dict = None):
        """
        Initialize multimodal augmentation.
        
        Args:
            tactile_config: Configuration for tactile augmentations
            image_config: Configuration for image augmentations
        """
        tactile_config = tactile_config or {}
        image_config = image_config or {}
        
        self.tactile_aug = TactileAugmentation(**tactile_config)
        self.image_aug = ImageAugmentation(**image_config)
    
    def __call__(self, batch: Dict, training: bool = True) -> Dict:
        """
        Apply augmentations to multimodal batch.
        
        Args:
            batch: Dictionary containing 'tactile' and/or 'images' keys
            training: Whether in training mode
            
        Returns:
            Augmented batch dictionary
        """
        augmented_batch = batch.copy()
        
        # Apply tactile augmentation
        if 'tactile' in batch:
            augmented_batch['tactile'] = self.tactile_aug(batch['tactile'], training)
        
        # Apply image augmentation
        if 'images' in batch:
            augmented_batch['images'] = self.image_aug(batch['images'], training)
        
        return augmented_batch


def create_augmentation_from_config(config: Dict) -> Union[TactileAugmentation, ImageAugmentation, MultimodalAugmentation]:
    """
    Create augmentation instance from configuration dictionary.
    
    Args:
        config: Configuration dictionary with augmentation parameters
    
    Returns:
        Appropriate augmentation instance
    """
    augmentation_type = config.get('type', 'tactile')
    
    if augmentation_type == 'tactile':
        return TactileAugmentation(**config.get('params', {}))
    elif augmentation_type == 'image':
        return ImageAugmentation(**config.get('params', {}))
    elif augmentation_type == 'multimodal':
        return MultimodalAugmentation(
            tactile_config=config.get('tactile_params', {}),
            image_config=config.get('image_params', {})
        )
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


if __name__ == "__main__":
    random_tensor = torch.ones(1, 4, 4, 4, 1)
    mask = create_spatio_temporal_mask_3d(random_tensor, mask_ratio=0.2, mask_t=2, mask_h=2, mask_w=1)
    masked_tensor = apply_mask(random_tensor, mask)
    print(masked_tensor.squeeze())