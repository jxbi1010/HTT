"""
General utility for printing training and model configurations.
Can be used with OmegaConf, regular dictionaries, or any config-like object.
"""

from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf


def print_config_files(config_paths: Dict[str, Optional[str]], title: str = "CONFIGURATION LOADING"):
    """
    Print config file paths in a formatted way.
    
    Args:
        config_paths: Dictionary mapping config names to file paths (e.g., {'ssl': 'path/to/ssl.yaml'})
        title: Title for the section header
    """
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
    for config_name, config_path in config_paths.items():
        if config_path:
            print(f"{config_name.capitalize()} config file: {config_path}")
    
    print(f"{'='*60}")


def print_merged_config(config: Union[OmegaConf, Dict[str, Any]], title: str = "Merged configuration"):
    """
    Print merged configuration in YAML format.
    
    Args:
        config: Configuration object (OmegaConf or dict)
        title: Title for the section
    """
    print(f"\n{title}:")
    if isinstance(config, OmegaConf):
        print(OmegaConf.to_yaml(config))
    else:
        # Convert dict to OmegaConf for nice YAML formatting
        print(OmegaConf.to_yaml(OmegaConf.create(config)))
    print(f"{'='*60}")


def print_model_config(
    model_config_dict: Dict[str, Any],
    encoder=None,
    decoder=None,
    device=None,
    title: str = "NETWORK CONFIGURATION"
):
    """
    Print comprehensive model/network configuration.
    
    Args:
        model_config_dict: Dictionary containing model configuration
        encoder: Encoder model (optional, for parameter counting)
        decoder: Decoder model (optional, for parameter counting)
        device: Device string (optional)
        title: Title for the section header
    """
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
    # Encoder configuration
    print(f"\n[ENCODER]")
    model_type = model_config_dict.get('type', 'Unknown')
    # Normalize model type
    if model_type in ['vit', 'vision']:
        model_type_display = 'vision'
    else:
        model_type_display = model_type
    print(f"  Type: {model_type_display}")
    
    # Get encoder dimensions from encoder object if available
    if encoder is not None:
        embed_dim = getattr(encoder, 'embed_dim', None)
        hidden_dim = getattr(encoder, 'hidden_dim', None)
        if embed_dim is not None:
            print(f"  Embedding dimension (embed_dim): {embed_dim}")
        if hidden_dim is not None and hidden_dim != embed_dim:
            print(f"  Hidden dimension (output dim): {hidden_dim}")
        elif hidden_dim is not None:
            print(f"  Hidden dimension: {hidden_dim}")
    
    # Encoder-specific parameters
    if model_type == 'transformer':
        print(f"  Tactile dimension: {model_config_dict.get('tactile_dim', 'Unknown')}")
        # Hidden dimension already printed above if available from encoder
        if encoder is None or not hasattr(encoder, 'hidden_dim'):
            print(f"  Hidden dimension: {model_config_dict.get('hidden_dim', 'Unknown')}")
        print(f"  Patch size: {model_config_dict.get('patch_size', 'Unknown')}")
        print(f"  Stride: {model_config_dict.get('stride', 'Unknown')}")
        # Try to get from encoder if available
        num_layers = None
        num_heads = None
        if encoder is not None:
            if hasattr(encoder, 'transformer'):
                num_layers = len(encoder.transformer.layers) if hasattr(encoder.transformer, 'layers') else getattr(encoder, 'num_layers', None)
            num_heads = getattr(encoder, 'num_heads', None) or (getattr(encoder.transformer, 'layers', [None])[0].self_attn.num_heads if hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'layers') and len(encoder.transformer.layers) > 0 else None)
        print(f"  Transformer layers: {num_layers if num_layers is not None else model_config_dict.get('transformer_layers', model_config_dict.get('num_layers', 'Unknown'))}")
        print(f"  Number of heads: {num_heads if num_heads is not None else model_config_dict.get('num_heads', 'Unknown')}")
        print(f"  Dropout: {model_config_dict.get('dropout', 'Unknown')}")
    elif model_type == 'transformer3d':
        print(f"  Tactile dimension: {model_config_dict.get('tactile_dim', 'Unknown')}")
        # Hidden dimension already printed above if available from encoder
        if encoder is None or not hasattr(encoder, 'hidden_dim'):
            print(f"  Hidden dimension: {model_config_dict.get('hidden_dim', 'Unknown')}")
        print(f"  Spatial dimensions: h={model_config_dict.get('h', 'Unknown')}, w={model_config_dict.get('w', 'Unknown')}")
        print(f"  Input dimension: {model_config_dict.get('input_dim', 'Unknown')}")
        print(f"  Patch sizes: t={model_config_dict.get('patch_t', 'Unknown')}, h={model_config_dict.get('patch_h', 'Unknown')}, w={model_config_dict.get('patch_w', 'Unknown')}")
        print(f"  Strides: t={model_config_dict.get('stride_t', 'Unknown')}, h={model_config_dict.get('stride_h', 'Unknown')}, w={model_config_dict.get('stride_w', 'Unknown')}")
        print(f"  Transformer layers: {model_config_dict.get('transformer_layers', model_config_dict.get('num_layers', 'Unknown'))}")
        print(f"  Number of heads: {model_config_dict.get('num_heads', 'Unknown')}")
        print(f"  Dropout: {model_config_dict.get('dropout', 'Unknown')}")
    elif model_type == 'lstm':
        print(f"  Tactile dimension: {model_config_dict.get('tactile_dim', 'Unknown')}")
        # Hidden dimension already printed above if available from encoder
        if encoder is None or not hasattr(encoder, 'hidden_dim'):
            print(f"  Hidden dimension: {model_config_dict.get('hidden_dim', 'Unknown')}")
        print(f"  LSTM layers: {model_config_dict.get('lstm_layers', 'Unknown')}")
        print(f"  Bidirectional: {model_config_dict.get('bidirectional', 'Unknown')}")
        print(f"  Dropout: {model_config_dict.get('dropout', 'Unknown')}")
    elif model_type in ['vit', 'vision']:
        # Vision Transformer specific parameters
        # Try to get from encoder object first, fallback to config
        patch_size = None
        input_size = None
        num_frames = None
        depth = None
        num_heads = None
        model_size = None
        
        if encoder is not None:
            # Get from ViTEncoder attributes
            patch_size = getattr(encoder, 'patch_size', None) or (getattr(encoder.vit, 'patch_embed', None) and getattr(encoder.vit.patch_embed, 'patch_size', None))
            if patch_size is not None and isinstance(patch_size, (tuple, list)):
                patch_size = patch_size[0] if len(patch_size) > 0 else patch_size
            
            input_size = getattr(encoder, 'input_size', None) or getattr(encoder.vit, 'img_size', None)
            num_frames = getattr(encoder, 'num_frames', None) or getattr(encoder.vit, 'num_frames', None)
            
            # Get depth from encoder
            if hasattr(encoder, 'vit') and hasattr(encoder.vit, 'blocks'):
                depth = len(encoder.vit.blocks)
            elif hasattr(encoder, 'vit') and hasattr(encoder.vit, 'depth'):
                depth = encoder.vit.depth
            
            # Get num_heads from encoder
            if hasattr(encoder, 'vit') and hasattr(encoder.vit, 'blocks') and len(encoder.vit.blocks) > 0:
                first_block = encoder.vit.blocks[0]
                if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'num_heads'):
                    num_heads = first_block.attn.num_heads
            elif hasattr(encoder, 'vit') and hasattr(encoder.vit, 'num_heads'):
                num_heads = encoder.vit.num_heads
            
            # Get model_size if available
            model_size = getattr(encoder, 'model_size', None)
        
        # Print values (prefer model values, fallback to config)
        print(f"  Patch size: {patch_size if patch_size is not None else model_config_dict.get('patch_size', 'Unknown')}")
        print(f"  Input size: {input_size if input_size is not None else model_config_dict.get('input_size', model_config_dict.get('img_size', 'Unknown'))}")
        print(f"  Number of frames: {num_frames if num_frames is not None else model_config_dict.get('num_frames', 'Unknown')}")
        # Embedding and hidden dimensions already printed above, skip here
        print(f"  Depth: {depth if depth is not None else model_config_dict.get('num_layers', model_config_dict.get('depth', 'Unknown'))}")
        print(f"  Number of heads: {num_heads if num_heads is not None else model_config_dict.get('num_heads', 'Unknown')}")
        print(f"  Model size: {model_size if model_size is not None else model_config_dict.get('model_size', 'Unknown')}")
        print(f"  Tubelet size: {model_config_dict.get('tubelet_size', 'Unknown')}")
        print(f"  MLP ratio: {model_config_dict.get('mlp_ratio', 'Unknown')}")
        print(f"  Dropout: {model_config_dict.get('dropout', 'Unknown')}")
    
    # Count encoder parameters
    if encoder is not None:
        encoder_params = sum(p.numel() for p in encoder.parameters())
        encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  Total parameters: {encoder_params:,}")
        print(f"  Trainable parameters: {encoder_trainable:,}")
    
    # Decoder configuration
    print(f"\n[DECODER]")
    if decoder is not None:
        decoder_type = decoder.__class__.__name__ if hasattr(decoder, '__class__') else 'Unknown'
        print(f"  Type: {decoder_type}")
        
        # Get decoder config
        decoder_config = model_config_dict.get('decoder', {})
        if decoder_config:
            print(f"  Decoder embed dimension: {decoder_config.get('decoder_embed_dim', 'Unknown')}")
            print(f"  Decoder depth: {decoder_config.get('decoder_depth', 'Unknown')}")
            print(f"  Decoder num heads: {decoder_config.get('decoder_num_heads', 'Unknown')}")
            print(f"  MLP ratio: {decoder_config.get('mlp_ratio', 'Unknown')}")
        
        # Try to get decoder attributes
        if hasattr(decoder, 'decoder_embed_dim'):
            print(f"  Decoder embed dimension (from model): {decoder.decoder_embed_dim}")
        if hasattr(decoder, 'decoder_depth'):
            print(f"  Decoder depth (from model): {decoder.decoder_depth}")
        if hasattr(decoder, 'decoder_num_heads'):
            print(f"  Decoder num heads (from model): {decoder.decoder_num_heads}")
        
        # Count decoder parameters
        decoder_params = sum(p.numel() for p in decoder.parameters())
        decoder_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"  Total parameters: {decoder_params:,}")
        print(f"  Trainable parameters: {decoder_trainable:,}")
    else:
        print(f"  Status: Not configured")
        decoder_params = 0
        decoder_trainable = 0
    
    # Total network parameters
    if encoder is not None:
        total_params = encoder_params + decoder_params
        total_trainable = encoder_trainable + decoder_trainable
        print(f"\n[TOTAL NETWORK]")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total trainable parameters: {total_trainable:,}")
    
    # Device information
    if device is not None or encoder is not None:
        print(f"\n[DEVICE]")
        if device is not None:
            print(f"  Device: {device}")
        if encoder is not None:
            try:
                print(f"  Encoder on: {next(encoder.parameters()).device}")
            except StopIteration:
                pass
        if decoder is not None:
            try:
                print(f"  Decoder on: {next(decoder.parameters()).device}")
            except StopIteration:
                pass
    
    print(f"{'='*60}\n")


def print_training_info(
    device: str,
    algorithm: str = None,
    dataset_type: str = None,
    random_seed: int = None,
    title: str = "TRAINER INITIALIZATION"
):
    """
    Print basic training information.
    
    Args:
        device: Device string (e.g., 'cuda' or 'cpu')
        algorithm: Algorithm name (optional)
        dataset_type: Dataset type (optional)
        random_seed: Random seed (optional)
        title: Title for the section header
    """
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"Device: {device}")
    if algorithm:
        print(f"Algorithm: {algorithm}")
    if dataset_type:
        print(f"Dataset: {dataset_type}")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")
    print(f"{'='*60}")

