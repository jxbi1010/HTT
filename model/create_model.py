"""
Unified model creation function for tactile and vision models.
Supports various architectures: encoder-only, encoder+decoder, encoder+mlp_head,
and encoder+shared_trunk+decoder.

Examples:
    # Tactile encoder only
    config = {
        'type': 'tactile',
        'architecture': ['encoder'],
        'tactile_dim': 72,
        'hidden_dim': 256,
        'patch_size': 4,
        'stride': 4,
        'num_heads': 8,
        'transformer_layers': 4,
        'dropout': 0.1
    }
    model = create_model(config)
    
    # Vision encoder + decoder (MAE)
    config = {
        'type': 'vision',
        'architecture': ['encoder', 'decoder'],
        'input_size': 224,
        'num_frames': 25,
        'hidden_dim': 768,
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_depth': 8,
            'decoder_num_heads': 16
        }
    }
    model = create_model(config)
    
    # Tactile encoder + shared_trunk + decoder
    config = {
        'type': 'tactile',
        'architecture': ['encoder', 'shared_trunk', 'decoder'],
        'tactile_dim': 72,
        'hidden_dim': 256,
        'shared_trunk': {
            'depth': 9,
            'num_heads': 8
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_depth': 8
        }
    }
    model = create_model(config)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union

from model.head import MLPHead
from model.taxel_networks import create_taxel_model
from model.vision_networks import create_vision_model
from model.resnet_encoder import create_resnet18_encoder
from model.shared_trunk import TransformerTrunk


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Unified model creation function for tactile and vision models.
    
    Architecture combinations:
    - ['encoder'] -> encoder only
    - ['encoder', 'decoder'] -> encoder-decoder
    - ['encoder', 'classifier'] -> encoder-mlp_head (for classification, 'classifier' kept for backward compatibility)
    - ['encoder', 'shared_trunk', 'decoder'] -> encoder-shared_trunk-decoder
    
    Args:
        model_config: Configuration dictionary with type, architecture, and model parameters
    
    Returns:
        nn.Module: The created model
    """
    # Normalize model type
    model_type = model_config.get('type', 'tactile')
    if model_type in ['vit', 'vision']:
        model_type = 'vision'
    elif model_type == 'resnet18':
        pass  # keep as resnet18
    elif model_type in ['tactile', 'taxel', 'transformer', 'transformer3d']:
        model_type = 'tactile'
    elif 'tactile_dim' in model_config:
        model_type = 'tactile'
    elif 'input_size' in model_config:
        model_type = 'vision'
    
    architecture = model_config.get('architecture', ['encoder'])
    if not isinstance(architecture, list):
        architecture = [architecture]
    
    # Create encoder
    encoder_config = model_config.copy()
    encoder_config['architecture'] = ['encoder']
    
    if model_type == 'tactile':
        encoder = create_taxel_model(encoder_config)
        if hasattr(encoder, 'encoder'):
            encoder = encoder.encoder
    elif model_type == 'resnet18':
        encoder = create_resnet18_encoder(encoder_config)
    else:
        encoder = create_vision_model(encoder_config)
    
    # Get encoder output dimension
    if model_type == 'vision' and hasattr(encoder, 'hidden_dim'):
        encoder_embed_dim = encoder.hidden_dim
    elif model_type == 'resnet18':
        encoder_embed_dim = getattr(encoder, 'embed_dim', encoder_config.get('embed_dim', 512))
    else:
        encoder_embed_dim = getattr(encoder, 'embed_dim', model_config.get('hidden_dim'))
    
    # Return encoder only if no other components
    if architecture == ['encoder']:
        return encoder
    
    # Create mlp_head if needed (architecture string 'classifier' kept for backward compatibility)
    mlp_head = None
    if 'classifier' in architecture:
        num_classes = model_config.get('num_classes')
        if num_classes is None:
            raise ValueError("'num_classes' must be provided in model_config for mlp_head architecture")
        
        mlp_head_hidden_dim = model_config.get('classifier', {}).get('hidden_dim', encoder_embed_dim)
        mlp_head_dropout = model_config.get('classifier', {}).get('dropout', 0.1)
        
        mlp_head = MLPHead(
            in_dim=encoder_embed_dim,
            out_dim=num_classes,
            hidden_dim=mlp_head_hidden_dim,
            dropout=mlp_head_dropout
        )
    
    # Create shared trunk if needed
    shared_trunk = None
    if 'shared_trunk' in architecture:
        trunk_config = model_config.get('shared_trunk', {})
        shared_trunk = TransformerTrunk(
            embed_dim=trunk_config.get('embed_dim', encoder_embed_dim),
            depth=trunk_config.get('depth'),
            num_heads=trunk_config.get('num_heads', model_config.get('num_heads')),
            pooling_type=trunk_config.get('pooling_type', 'none'),
        )
    
    # Create decoder if needed
    decoder = None
    if 'decoder' in architecture:
        decoder_config = model_config.get('decoder', {})
        
        if model_type == 'tactile':
            from model.taxel_networks import TactileTransformerDecoder
            patch_dim = encoder.patch_size * model_config.get('tactile_dim')
            decoder = TactileTransformerDecoder(
                input_embed_dim=encoder_embed_dim,
                decoder_embed_dim=decoder_config.get('decoder_embed_dim'),
                decoder_depth=decoder_config.get('decoder_depth'),
                decoder_num_heads=decoder_config.get('decoder_num_heads'),
                patch_dim=patch_dim,
            )
        else:
            from model.vision_networks import ViTDecoder
            def get_decoder_param(param_name):
                return decoder_config.get(param_name, model_config.get(param_name))
            
            decoder = ViTDecoder(
                embed_dim=encoder_embed_dim,
                num_frames=model_config.get('num_frames'),
                tubelet_size=model_config.get('tubelet_size'),
                depth=get_decoder_param('decoder_depth'),
                num_heads=get_decoder_param('decoder_num_heads'),
            )
    
    # Return appropriate wrapper
    if architecture == ['encoder', 'classifier']:
        return EncoderMLPHeadWrapper(encoder, mlp_head, encoder_embed_dim)
    elif architecture == ['encoder', 'decoder']:
        return EncoderDecoderWrapper(encoder, decoder)
    elif architecture == ['encoder', 'shared_trunk', 'decoder']:
        return EncoderTrunkDecoderWrapper(encoder, shared_trunk, decoder, encoder_embed_dim)
    
    return encoder


class EncoderDecoderWrapper(nn.Module):
    """Wrapper for encoder-decoder models."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = getattr(encoder, 'embed_dim', None)
    
    def forward(self, x, masks=None, ids_restore=None):
        """
        Forward pass through encoder-decoder.
        
        Args:
            x: Input data
            masks: Optional mask indices for MAE
            ids_restore: Optional restore indices for MAE decoder
        
        Returns:
            If ids_restore is provided: decoder output
            Else: encoder output
        """
        if ids_restore is not None:
            # MAE mode: encoder -> decoder
            encoded = self.encoder(x, masks=masks)
            decoded = self.decoder(encoded, ids_restore)
            return decoded
        else:
            # Encoder-only mode
            return self.encoder(x, masks=masks)


class EncoderMLPHeadWrapper(nn.Module):
    """Wrapper for encoder-mlp_head models."""
    
    def __init__(self, encoder: nn.Module, mlp_head: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.mlp_head = mlp_head
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """Forward pass: encoder -> mlp_head."""
        features = self.encoder(x)
        
        # Handle different encoder output formats
        if isinstance(features, tuple):
            # MAE format: (features, mask, ids_restore)
            features = features[0]
        
        # Pool features if needed (for sequence outputs)
        if features.dim() == 3:
            # [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim]
            if hasattr(self.encoder, 'use_cls_token') and self.encoder.use_cls_token:
                features = features[:, 0, :]  # Use CLS token
            else:
                features = features.mean(dim=1)  # Global average pooling
        
        logits = self.mlp_head(features)
        return logits

class EncoderTrunkDecoderWrapper(nn.Module):
    """Wrapper for encoder-shared_trunk-decoder models."""
    
    def __init__(self, encoder: nn.Module, shared_trunk: nn.Module, decoder: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.shared_trunk = shared_trunk
        self.decoder = decoder
        self.embed_dim = embed_dim
    
    def forward(self, x, masks=None, ids_restore=None):
        """
        Forward pass: encoder -> shared_trunk -> decoder.
        
        Args:
            x: Input data (can be raw data or patched data depending on encoder)
            masks: Optional mask indices for MAE [batch_size, len_keep]
            ids_restore: Required restore indices for decoder [batch_size, num_total]
        
        Returns:
            Decoder output [batch_size, num_total, patch_dim]
        """
        if ids_restore is None:
            raise ValueError("ids_restore is required for encoder-shared_trunk-decoder forward pass")
        
        # Encoder forward - returns features [batch_size, len_keep, embed_dim]
        encoded = self.encoder(x, masks=masks)
        
        # Shared trunk forward
        # Encoders return features directly, not tuples
        # But shared_trunk can handle both regular features and MAE tuple format
        # For MAE, we wrap features in tuple to preserve mask/ids_restore info
        if masks is not None:
            # MAE mode: pass as tuple to preserve mask and ids_restore
            trunk_output = self.shared_trunk((encoded, masks, ids_restore))
            # trunk_output should be (features, mask, ids_restore)
            if isinstance(trunk_output, tuple):
                trunk_features, _, _ = trunk_output
            else:
                trunk_features = trunk_output
        else:
            # Regular mode: just process features
            trunk_features = self.shared_trunk(encoded)
        
        # Decoder forward
        decoded = self.decoder(trunk_features, ids_restore)
        return decoded


def create_pretrain_model(pretrain_config: Dict[str, Any]) -> nn.Module:
    """
    Create a multimodal pretraining model with multiple encoders, shared trunk, and multiple decoders.
    
    This function creates a model from the pretrain.yaml config format, which includes:
    - Multiple encoders (one per modality, e.g., '9dtact', 'tac02')
    - One shared trunk (shared across all modalities)
    - Multiple decoders (one per encoder/modality)
    
    Args:
        pretrain_config: Configuration dictionary with:
            - encoders: Dict[str, Dict] - Dictionary of encoder configs keyed by modality name
            - shared_trunk: Dict - Shared trunk configuration
            - decoders: Dict[str, Dict] - Dictionary of decoder configs keyed by modality name
            - hidden_dim: int - Common hidden dimension
            - num_frames: int - Number of frames (for vision encoders)
            - Other common parameters
    
    Returns:
        nn.Module: PretrainModelWrapper that can switch between modalities
    
    Example:
        config = {
            'encoders': {
                '9dtact': {'type': 'vit', 'hidden_dim': 192, ...},
                'tac02': {'type': 'transformer', 'tactile_dim': 66, ...}
            },
            'shared_trunk': {'embed_dim': 192, 'depth': 9, ...},
            'decoders': {
                '9dtact': {'decoder_type': 'vit', 'decoder_embed_dim': 192, ...},
                'tac02': {'decoder_type': 'transformer', 'decoder_embed_dim': 192, ...}
            },
            'hidden_dim': 192,
            'num_frames': 1
        }
        model = create_pretrain_model(config)
    """
    encoders_config = pretrain_config.get('encoders', {})
    shared_trunk_config = pretrain_config.get('shared_trunk', {})
    decoders_config = pretrain_config.get('decoders', {})
    
    if not encoders_config:
        raise ValueError("'encoders' must be provided in pretrain_config")
    if not shared_trunk_config:
        raise ValueError("'shared_trunk' must be provided in pretrain_config")
    if not decoders_config:
        raise ValueError("'decoders' must be provided in pretrain_config")
    
    # Get common parameters
    hidden_dim = pretrain_config.get('hidden_dim')
    num_frames = pretrain_config.get('num_frames', 1)
    
    # Create encoders
    encoders = nn.ModuleDict()
    encoder_embed_dims = {}
    
    for modality_name, encoder_config in encoders_config.items():
        # Merge common config with encoder-specific config
        full_encoder_config = pretrain_config.copy()
        full_encoder_config.update(encoder_config)
        full_encoder_config['architecture'] = ['encoder']
        
        # Determine encoder type
        encoder_type = encoder_config.get('type', 'tactile')
        if encoder_type in ['vit', 'vision']:
            encoder_type = 'vision'
        elif encoder_type in ['tactile', 'taxel', 'transformer', 'transformer3d']:
            encoder_type = 'tactile'
        elif 'tactile_dim' in encoder_config:
            encoder_type = 'tactile'
        elif 'input_size' in encoder_config:
            encoder_type = 'vision'
        
        # Create encoder
        if encoder_type == 'tactile':
            encoder = create_taxel_model(full_encoder_config)
            if hasattr(encoder, 'encoder'):
                encoder = encoder.encoder
        else:
            encoder = create_vision_model(full_encoder_config)
            if hasattr(encoder, 'encoder'):
                encoder = encoder.encoder
        
        # Get encoder embed dimension
        if encoder_type == 'vision' and hasattr(encoder, 'hidden_dim'):
            encoder_embed_dim = encoder.hidden_dim
        else:
            encoder_embed_dim = getattr(encoder, 'embed_dim', encoder_config.get('hidden_dim', hidden_dim))
        
        encoders[modality_name] = encoder
        encoder_embed_dims[modality_name] = encoder_embed_dim
    
    # Create shared trunk
    trunk_embed_dim = shared_trunk_config.get('embed_dim', hidden_dim)
    shared_trunk = TransformerTrunk(
        embed_dim=trunk_embed_dim,
        depth=shared_trunk_config.get('depth'),
        num_heads=shared_trunk_config.get('num_heads', pretrain_config.get('num_heads')),
        mlp_ratio=shared_trunk_config.get('mlp_ratio', 4.0),
        pooling_type=shared_trunk_config.get('pooling_type', 'none'),
    )
    
    # Create decoders
    decoders = nn.ModuleDict()
    
    for modality_name, decoder_config in decoders_config.items():
        if modality_name not in encoders:
            raise ValueError(f"Decoder '{modality_name}' has no corresponding encoder")
        
        encoder = encoders[modality_name]
        encoder_embed_dim = encoder_embed_dims[modality_name]
        
        # Determine decoder type from config or infer from encoder
        decoder_type = decoder_config.get('decoder_type')
        if decoder_type is None:
            # Infer from encoder type
            encoder_config = encoders_config[modality_name]
            encoder_type = encoder_config.get('type', 'tactile')
            if encoder_type in ['vit', 'vision']:
                decoder_type = 'vit'
            else:
                decoder_type = 'transformer'
        
        # Create decoder
        if decoder_type == 'transformer':
            from model.taxel_networks import TactileTransformerDecoder
            encoder_config = encoders_config[modality_name]
            # Get patch_dim from encoder config
            # patch_dim = patch_size * tactile_dim (where tactile_dim is input_dim)
            if 'tactile_dim' in encoder_config:
                tactile_dim = encoder_config['tactile_dim']
                patch_size = encoder_config.get('patch_size', 1)
                patch_dim = patch_size * tactile_dim
            elif hasattr(encoder, 'patch_size'):
                # Try to infer from encoder attributes
                patch_size = encoder.patch_size
                # For TactileTransformerEncoder, input_dim is not stored, so we need config
                # This is a fallback that may not work - prefer using tactile_dim in config
                raise ValueError(
                    f"Cannot determine patch_dim for decoder '{modality_name}'. "
                    f"Please specify 'tactile_dim' in encoder config."
                )
            else:
                raise ValueError(f"Cannot determine patch_dim for decoder '{modality_name}'")
            
            decoder = TactileTransformerDecoder(
                input_embed_dim=encoder_embed_dim,
                decoder_embed_dim=decoder_config.get('decoder_embed_dim', hidden_dim),
                decoder_depth=decoder_config.get('decoder_depth'),
                decoder_num_heads=decoder_config.get('decoder_num_heads', pretrain_config.get('num_heads')),
                patch_dim=patch_dim,
            )
        else:  # decoder_type == 'vit'
            from model.vision_networks import ViTDecoder
            def get_decoder_param(param_name):
                return decoder_config.get(param_name, pretrain_config.get(param_name))
            
            # Get num_frames from encoder config if available, otherwise use global
            encoder_config = encoders_config[modality_name]
            modality_num_frames = encoder_config.get('num_frames', num_frames)
            
            decoder = ViTDecoder(
                embed_dim=encoder_embed_dim,
                num_frames=modality_num_frames,
                tubelet_size=encoder_config.get('tubelet_size'),
                depth=decoder_config.get('decoder_depth'),
                num_heads=decoder_config.get('decoder_num_heads', pretrain_config.get('num_heads')),
            )
        
        decoders[modality_name] = decoder
    
    # Return wrapper
    return PretrainModelWrapper(encoders, shared_trunk, decoders, trunk_embed_dim)


class PretrainModelWrapper(nn.Module):
    """
    Wrapper for multimodal pretraining model with multiple encoders, shared trunk, and multiple decoders.
    
    This wrapper allows switching between different modalities during training.
    Each modality has its own encoder and decoder, but they all share the same trunk.
    """
    
    def __init__(self, encoders: nn.ModuleDict, shared_trunk: nn.Module, 
                 decoders: nn.ModuleDict, embed_dim: int):
        super().__init__()
        self.encoders = encoders
        self.shared_trunk = shared_trunk
        self.decoders = decoders
        self.embed_dim = embed_dim
        
        # Verify that encoder and decoder keys match
        encoder_keys = set(encoders.keys())
        decoder_keys = set(decoders.keys())
        if encoder_keys != decoder_keys:
            raise ValueError(
                f"Encoder and decoder keys must match. "
                f"Encoders: {encoder_keys}, Decoders: {decoder_keys}"
            )
    
    def forward(self, x, modality: str, masks=None, ids_restore=None):
        """
        Forward pass through encoder -> shared_trunk -> decoder for a specific modality.
        
        Args:
            x: Input data for the specified modality
            modality: Name of the modality (must be a key in encoders/decoders)
            masks: Optional mask indices for MAE [batch_size, len_keep]
            ids_restore: Required restore indices for decoder [batch_size, num_total]
        
        Returns:
            Decoder output [batch_size, num_total, patch_dim]
        """
        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}. Available: {list(self.encoders.keys())}")
        
        if ids_restore is None:
            raise ValueError("ids_restore is required for pretrain model forward pass")
        
        # Get encoder and decoder for this modality
        encoder = self.encoders[modality]
        decoder = self.decoders[modality]
        
        # Encoder forward - returns features [batch_size, len_keep, embed_dim]
        encoded = encoder(x, masks=masks)
        
        # Shared trunk forward
        # For MAE, we wrap features in tuple to preserve mask/ids_restore info
        if masks is not None:
            # MAE mode: pass as tuple to preserve mask and ids_restore
            trunk_output = self.shared_trunk((encoded, masks, ids_restore))
            # trunk_output should be (features, mask, ids_restore)
            if isinstance(trunk_output, tuple):
                trunk_features, _, _ = trunk_output
            else:
                trunk_features = trunk_output
        else:
            # Regular mode: just process features
            trunk_features = self.shared_trunk(encoded)
        
        # Decoder forward
        decoded = decoder(trunk_features, ids_restore)
        return decoded
    
    def get_encoder(self, modality: str) -> nn.Module:
        """Get encoder for a specific modality."""
        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}. Available: {list(self.encoders.keys())}")
        return self.encoders[modality]
    
    def get_decoder(self, modality: str) -> nn.Module:
        """Get decoder for a specific modality."""
        if modality not in self.decoders:
            raise ValueError(f"Unknown modality: {modality}. Available: {list(self.decoders.keys())}")
        return self.decoders[modality]
    
    @property
    def modalities(self):
        """Return list of available modalities."""
        return list(self.encoders.keys())

