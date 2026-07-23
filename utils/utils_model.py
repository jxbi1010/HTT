"""
Utility functions for model operations and reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Integer seed value for all random number generators
    """
    # 1. Python's built-in random module
    random.seed(seed)
    
    # 2. NumPy's random number generator
    np.random.seed(seed)
    
    # 3. PyTorch's random number generators
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
    # 4. For deterministic GPU algorithms (can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. Environment variable for Python hash-based functions
    os.environ['PYTHONHASHSEED'] = str(seed)


def adjust_state_dict_keys(state_dict, model):
    """
    Adjust state_dict keys to match the model's expected keys.
    Handles the case where model is compiled with torch.compile() which adds '_orig_mod.' 
    in the key path (can be at the start or in the middle, e.g., 'encoders.9dtact._orig_mod.vit.blocks...').
    
    Args:
        state_dict: The state_dict from checkpoint
        model: The model to load into
        
    Returns:
        Adjusted state_dict with keys matching the model
    """
    # Get the expected keys from the model
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())
    
    # Check if model expects _orig_mod anywhere in keys (model is compiled)
    model_has_orig_mod = any('_orig_mod.' in key for key in model_keys)
    
    # Check if checkpoint keys have _orig_mod anywhere
    checkpoint_has_orig_mod = any('_orig_mod.' in key for key in checkpoint_keys)
    
    # If model is compiled but checkpoint doesn't have _orig_mod, add it
    # This is complex - we need to add it in the right place in the path
    # For now, we'll add it at the start (simple case)
    if model_has_orig_mod and not checkpoint_has_orig_mod:
        adjusted_state_dict = {}
        for key, value in state_dict.items():
            # Add _orig_mod at the start
            adjusted_key = f'_orig_mod.{key}'
            adjusted_state_dict[adjusted_key] = value
        print(f"Adjusted state_dict keys: added '_orig_mod.' prefix to match compiled model")
        return adjusted_state_dict
    
    # If model is not compiled but checkpoint has _orig_mod, remove it from anywhere in the path
    elif not model_has_orig_mod and checkpoint_has_orig_mod:
        adjusted_state_dict = {}
        for key, value in state_dict.items():
            # Remove _orig_mod. from anywhere in the key path
            # Replace '_orig_mod.' with empty string (handles both at start and in middle)
            adjusted_key = key.replace('_orig_mod.', '')
            adjusted_state_dict[adjusted_key] = value
        print(f"Adjusted state_dict keys: removed '_orig_mod.' from key paths to match uncompiled model")
        return adjusted_state_dict
    
    # Keys already match, return as is
    return state_dict

