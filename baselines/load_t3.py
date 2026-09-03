"""
Test script to measure inference time for T3 model with encoder, trunk, and decoder

Author: Test Script
"""

import torch
import time
from omegaconf import OmegaConf
import sys
import os
sys.path.append(os.environ.get('T3_REPO', 'third_party/t3'))

from t3.models import T3

def load_model(task_type="pose", device=None):
    """
    Load T3 model for pose estimation, object classification, or force estimation.
    
    Args:
        task_type (str): One of "pose", "classification", or "force"
        device (torch.device, optional): Device to move model to. If None, uses CPU.
    
    Returns:
        model: Loaded T3 model in evaluation mode
        encoder_domain (str): Name of the encoder domain
        decoder_domain (str): Name of the decoder domain
        forward_mode (str): "single_tower" or "multi_tower"
    """
    # Select configuration file and settings based on task type
    if task_type == "pose":
        config_path = os.path.join(os.environ.get("T3_REPO", "third_party/t3"), "configs/network/finetune_exp_pose_regression.yaml")
        expected_decoder = "pose_estimation_3d"
        forward_mode = "multi_tower"
    elif task_type == "classification":
        config_path = os.path.join(os.environ.get("T3_REPO", "third_party/t3"), "configs/network/finetune_exp_cls.yaml")
        expected_decoder = "cls_cnc"
        forward_mode = "single_tower"
    elif task_type == "force":
        config_path = os.path.join(os.environ.get("T3_REPO", "third_party/t3"), "configs/network/pretrain1_mae.yaml")
        expected_decoder = "mae_recon_single"
        forward_mode = "multi_tower"
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Must be 'pose', 'classification', or 'force'"
        )
    
    print(f"Loading configuration for {task_type} from {config_path}")
    network_cfg = OmegaConf.load(config_path)
    
    # Wrap in a config structure as expected by T3 with proper namespace for interpolation
    cfg = OmegaConf.create({
        'network': network_cfg,
        'encoders': network_cfg.encoders,
        'shared_trunk': network_cfg.shared_trunk,
        'decoders': network_cfg.decoders
    })
    
    # Create model
    print("Initializing T3 model...")
    model = T3(cfg)
    
    # Set to evaluation mode
    model.eval()
    
    # Select encoder domain - prefer 'mini' if it exists, otherwise use first available
    available_encoders = list(model.encoders.keys())
    print(f"Available encoders: {available_encoders}")
    if 'mini' in available_encoders:
        encoder_domain = 'mini'
    else:
        encoder_domain = available_encoders[0]
    
    # Verify expected decoder exists
    if expected_decoder not in model.decoders:
        available_decoders = list(model.decoders.keys())
        raise ValueError(
            f"Expected decoder '{expected_decoder}' not found. "
            f"Available decoders: {available_decoders}"
        )
    decoder_domain = expected_decoder
    
    # Set domains and forward mode
    model.set_domains(encoder_domain, decoder_domain, forward_mode)
    
    print(f"Using encoder: {encoder_domain}")
    print(f"Using decoder: {decoder_domain}")
    print(f"Using forward mode: {forward_mode}")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    # Print model summary
    model.model_summary()
    
    return model, encoder_domain, decoder_domain, forward_mode

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load T3 model for different tasks")
    parser.add_argument(
        "--task",
        type=str,
        choices=["pose", "classification", "force"],
        default="force",
        help="Task type: 'pose', 'classification', or 'force'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda' or 'cpu'). If not specified, uses CPU."
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, encoder_domain, decoder_domain, forward_mode = load_model(
        task_type=args.task,
        device=device
    )
    
    print(f"\nModel loaded successfully!")
    print(f"Task: {args.task}")
    print(f"Encoder: {encoder_domain}")
    print(f"Decoder: {decoder_domain}")
    print(f"Forward mode: {forward_mode}")
