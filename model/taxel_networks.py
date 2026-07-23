"""
Neural network architectures for tactile data processing.
Includes both encoders and decoders for tactile data reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pos_encoding import get_1d_sinusoidal_pe
from model.head import MLPHead

def init_weights(module: nn.Module):
    """
    Initialize weights to prevent NaN losses.
    
    Args:
        module: PyTorch module to initialize weights for.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TactileTransformerEncoder(nn.Module):
    """Transformer encoder for tactile data (alternative to LSTM) using
    non-learned sinusoidal positional encoding."""
    
    def __init__(self, input_dim, hidden_dim, patch_size, stride, max_len=1000, num_heads=8, num_layers=2, dropout=0.1, use_cls_token=False):
        super(TactileTransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim * patch_size, hidden_dim) # Corrected projection dim
        
        # --- IMPROVEMENT: Use non-learned sinusoidal positional encoding ---
        self.max_len = max_len
        self.pos_encoding = get_1d_sinusoidal_pe(max_len + 1, hidden_dim).unsqueeze(0)  # Add batch dimension: (1, max_len+1, hidden_dim) - +1 for CLS token
        self.register_buffer('pos_encoding_buffer', self.pos_encoding) 
        # Register as buffer to save state but not train (non-learned)
        
        self.patch_size = patch_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token
        
        # CLS token: learnable classification token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            torch.nn.init.normal_(self.cls_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = hidden_dim
        
        init_weights(self)

    def patchify(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transforms a time series tensor (Batch, Len, Feature) into a sequence of patches (tokens).
        (Implementation remains the same as provided)
        """
        batch_size, seq_len, num_features = data.shape

        data_reshaped = data.permute(0, 2, 1).reshape(batch_size * num_features, seq_len)
            
        patches_unfolded = data_reshaped.unfold(dimension=1, size=self.patch_size, step=self.stride)

        patches_unfolded = patches_unfolded.reshape(batch_size, num_features, -1, self.patch_size)

        patches = patches_unfolded.permute(0, 2, 3, 1)

        patched_data = patches.reshape(batch_size, -1, self.patch_size * num_features)
        
        return patched_data 
    
    def forward(self, x, masks=None):
        """
        Forward pass for tactile transformer encoder.
        
        Args:
            x: Input data. 
               If masks is None: Raw tactile data [B, L, F] -> will be patchified.
               If masks is not None: Already-patched tokens [B, num_patches, patch_dim].
            masks: Indices of tokens to keep [B, len_keep] (for MAE). If None, use all tokens.
        
        Returns:
            output: Encoded features [B, num_tokens, embed_dim] or [B, num_tokens+1, embed_dim] with CLS
        """
        if masks is None:
            # Case 1: No masking (e.g., classification or inference)
            # Assume x is raw data [B, L, F] and needs patchification
            # Note: If x is already patches, patchify might produce unexpected results, 
            # but standard usage implies raw data when masks=None.
            patches = self.patchify(x)
            patches_visible = patches
            
            # Use all tokens
            num_tokens = patches_visible.size(1)
            batch_size = patches_visible.size(0)
            
            # Project input to hidden dimension
            x_proj = self.input_projection(patches_visible)
            
            # Add CLS token if enabled
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x_proj = torch.cat([cls_tokens, x_proj], dim=1)
                # num_tokens incremented implicitly in pos encoding slicing
            
            # Add sequential positional encoding
            # CLS token gets position 0, regular tokens get positions 1, 2, ..., num_tokens
            # We need num_tokens from patches (before CLS) for slicing buffer
            pos_encoding = self.pos_encoding_buffer[:, :num_tokens].to(x_proj.device)
            
            if self.use_cls_token:
                # If CLS is used, pos_encoding_buffer[0] is for CLS (pos 0)
                # and pos_encoding_buffer[1:] are for tokens (pos 1..N)
                # The buffer is already created with max_len + 1
                # So we just take the first num_tokens + 1 elements
                pos_encoding = self.pos_encoding_buffer[:, :num_tokens+1].to(x_proj.device)
            else:
                pos_encoding = self.pos_encoding_buffer[:, :num_tokens].to(x_proj.device)
                
            x_proj = x_proj + pos_encoding
            
        else:
            # Case 2: Masking enabled (e.g., MAE training)
            # Assume x is already patches [B, N, P*F]
            patches = x
            
            ids_keep_expanded = masks.unsqueeze(-1).expand(-1, -1, patches.size(-1))
            patches_visible = torch.gather(patches, dim=1, index=ids_keep_expanded)
            
            num_tokens = patches_visible.size(1)
            batch_size = patches_visible.size(0)
            
            # Project input to hidden dimension
            x_proj = self.input_projection(patches_visible)
            
            # Add CLS token if enabled
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x_proj = torch.cat([cls_tokens, x_proj], dim=1)
            
            # Add positional encoding based on original positions
            # batch_size_orig = masks.size(0) # Same as batch_size
            
            # Get full PE buffer
            pos_encoding = self.pos_encoding_buffer[:, :self.max_len + 1].to(x_proj.device)
            
            # Select PEs for visible tokens
            # masks contains indices of kept tokens (0-indexed in original sequence)
            # If CLS is used, token i maps to position i+1 (since CLS is at 0)
            
            if self.use_cls_token:
                # CLS token gets position 0
                cls_pos = pos_encoding[:, 0:1, :].expand(batch_size, -1, -1)
                
                # Regular tokens: shift indices by 1
                masks_shifted = masks + 1
                pos_encoding_selected = torch.gather(
                    pos_encoding.expand(batch_size, -1, -1),
                    dim=1,
                    index=masks_shifted.unsqueeze(-1).expand(-1, -1, x_proj.size(-1))
                )
                
                pos_encoding_full = torch.cat([cls_pos, pos_encoding_selected], dim=1)
                x_proj = x_proj + pos_encoding_full
            else:
                # No CLS, direct mapping
                pos_encoding_selected = torch.gather(
                    pos_encoding.expand(batch_size, -1, -1),
                    dim=1,
                    index=masks.unsqueeze(-1).expand(-1, -1, x_proj.size(-1))
                )
                x_proj = x_proj + pos_encoding_selected
        
        # Transformer encoding
        output = self.transformer(x_proj)
        
        return output


class TactileTransformerDecoder(nn.Module):
    """Transformer decoder for tactile data."""
    
    def __init__(
        self,
        input_embed_dim=128,
        num_patches=None,  # Will be inferred during forward pass if None
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        patch_dim=None,  # Dimension of each patch/token for reconstruction
        norm_layer=nn.LayerNorm,
    ):
        """
        Args:
            input_embed_dim: Embedding dimension from encoder
            num_patches: Number of patches/tokens (can be None for variable length)
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads
            mlp_ratio: MLP ratio in decoder blocks
            patch_dim: Dimension of reconstructed patch/token
            norm_layer: Normalization layer type
        """
        super(TactileTransformerDecoder, self).__init__()
        
        self.input_embed_dim = input_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.patch_dim = patch_dim
        
        # Decoder embedding
        self.decoder_embed = nn.Linear(input_embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional encoding (non-learned sinusoidal, same as transformer encoder)
        self.max_patches = num_patches if num_patches is not None else 1000
        self.pos_encoding = get_1d_sinusoidal_pe(self.max_patches, decoder_embed_dim).unsqueeze(0)  # Add batch dimension: (1, max_patches, decoder_embed_dim)
        self.register_buffer('decoder_pos_embed', self.pos_encoding)
        
        # Decoder transformer blocks
        # Use the simple DecoderBlock (self-attention only) from decoder_block
        # Import the module and access the first DecoderBlock definition
        from model.layers import decoder_block as db_module
        
        # The first DecoderBlock in decoder_block.py is the simple self-attention one
        # We need to create our own simple decoder block to avoid the cross-attention version
        # Define a simple decoder block inline (similar to the first DecoderBlock)
        from model.layers.attention import Attention
        from model.layers.drop_path import DropPath
        from model.layers.layer_scale import LayerScale
        from model.layers.mlp import Mlp
        
        class SimpleDecoderBlock(nn.Module):
            """Simple decoder block with self-attention only (for MAE)."""
            def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, 
                        proj_bias=False, ffn_bias=True, drop_path=0.0, 
                        norm_layer=nn.LayerNorm, act_layer=nn.GELU, init_values=1):
                super().__init__()
                self.norm1 = norm_layer(dim)
                self.attn = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                )
                self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
                self.norm2 = norm_layer(dim)
                mlp_hidden_dim = int(dim * mlp_ratio)
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    bias=ffn_bias,
                )
                self.layer_scale = LayerScale(init_values) if init_values > 0 else nn.Identity()
            
            def forward(self, x):
                x = x + self.drop_path(self.layer_scale(self.attn(self.norm1(x))))
                x = x + self.drop_path(self.layer_scale(self.mlp(self.norm2(x))))
                return x
        
        decoder_blocks = []
        for _ in range(decoder_depth):
            decoder_blocks.append(
                SimpleDecoderBlock(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    norm_layer=norm_layer,
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        # Decoder norm
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Decoder prediction head (outputs patch_dim per token)
        if patch_dim is not None:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)
        else:
            self.decoder_pred = None
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights."""
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional encoding is non-learned (sinusoidal), so no initialization needed
        
        # Initialize prediction head if exists
        if self.decoder_pred is not None:
            torch.nn.init.normal_(self.decoder_pred.weight, std=0.02)
            torch.nn.init.constant_(self.decoder_pred.bias, 0)
    
    def forward(self, x, ids_restore):
        """
        Forward pass through decoder.

        Args:
            x: [N, L, D] encoded features from encoder
            ids_restore: [N, L_total] indices to restore original order
        
        Returns:
            [N, L_total, patch_dim] reconstructed patches
        """
        batch_size = x.shape[0]
        num_visible = x.shape[1]
        num_total = ids_restore.shape[1]
        
        # Embed tokens
        x = self.decoder_embed(x)  # [N, L, decoder_embed_dim]
        
        # Append mask tokens
        num_masked = num_total - num_visible
        mask_tokens = self.mask_token.repeat(batch_size, num_masked, 1)  # [N, L_masked, decoder_embed_dim]
        x_full = torch.cat([x, mask_tokens], dim=1)  # [N, L_total, decoder_embed_dim]
        
        # Unshuffle to restore original order
        x_unshuffled = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )
        
        # Add positional encoding (non-learned sinusoidal, same as transformer encoder)
        # Ensure the buffer is on the correct device and truncate to num_total
        pos_encoding = self.decoder_pos_embed[:, :num_total, :].to(x_unshuffled.device)  # [1, L_total, decoder_embed_dim]
        x_unshuffled = x_unshuffled + pos_encoding
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x_unshuffled = blk(x_unshuffled)
        x_unshuffled = self.decoder_norm(x_unshuffled)
        
        # Predictor projection
        if self.decoder_pred is not None:
            pred = self.decoder_pred(x_unshuffled)  # [N, L_total, patch_dim]
        else:
            pred = x_unshuffled
        
        return pred


def create_taxel_model(model_config):
    """
    Create a tactile model based on configuration.
    
    Args:
        model_config: Configuration dictionary containing:
            - architecture: List of modules ['encoder'] or ['encoder', 'decoder']
            - tactile_dim: Feature dimension per timestep
            - hidden_dim: Hidden dimension
            - dropout: Dropout rate
            - patch_size: Patch size for patching
            - stride: Stride for patching
            - num_heads: Number of attention heads
            - transformer_layers: Number of transformer layers
            - decoder: Decoder config (required if 'decoder' in architecture)
    
    Returns:
        nn.Module: Encoder or encoder+decoder wrapper
    """
    architecture = model_config.get('architecture', ['encoder'])
    hidden_dim = model_config.get('hidden_dim')
    
    # Create encoder
    tactile_dim = model_config.get('tactile_dim')
    encoder = TactileTransformerEncoder(
        input_dim=tactile_dim,
        hidden_dim=hidden_dim,
        patch_size=model_config.get('patch_size'),
        stride=model_config.get('stride'),
        num_heads=model_config.get('num_heads'),
        num_layers=model_config.get('transformer_layers'),
    )
    
    # Return encoder only if no decoder
    if 'decoder' not in architecture:
        return encoder
    
    # Create decoder
    decoder_config = model_config.get('decoder', {})
    patch_dim = model_config.get('patch_size') * tactile_dim
    
    decoder = TactileTransformerDecoder(
        input_embed_dim=hidden_dim,
        decoder_embed_dim=decoder_config.get('decoder_embed_dim'),
        decoder_depth=decoder_config.get('decoder_depth'),
        decoder_num_heads=decoder_config.get('decoder_num_heads'),
        patch_dim=patch_dim,
    )
    
    # Return encoder+decoder wrapper
    class EncoderDecoderWrapper(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.embed_dim = getattr(encoder, 'embed_dim', hidden_dim)
        
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
                # Encoder-only mode (e.g., classification)
                return self.encoder(x, masks=masks)
    
    return EncoderDecoderWrapper(encoder, decoder)




