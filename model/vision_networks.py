"""
Neural network architectures for gelsight video data processing.
"""

from inspect import FrameInfo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.vision_transformer import VisionTransformer, vit_tiny, vit_small, vit_base, vit_large, apply_masks
from model.pos_encoding import get_1d_sinusoidal_pe
from model.head import MLPHead


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for video data using local VisionTransformer.
    
    Now accepts patched data [batch_size, num_patches, patch_dim] directly
    instead of 5D video data [batch_size, num_frames, channels, height, width].
    """
    
    def __init__(self, input_size=224, patch_size=16, num_frames=None, 
                 hidden_dim=None, num_layers=None, num_heads=None, dropout=0.1, 
                 model_size='tiny', tubelet_size=None, mlp_ratio=4.0):
        super(ViTEncoder, self).__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.patch_dim = 3 * tubelet_size * patch_size**2
        
        # Create VisionTransformer based on model size
        if model_size == 'tiny':
            self.vit = vit_tiny(
                patch_size=patch_size,
                depth=num_layers,
                img_size=input_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                in_chans=3,
                drop_path_rate=dropout,
                pos_embed_fn="learned",
                mlp_ratio=mlp_ratio
            )
            self.embed_dim = 192
        elif model_size == 'small':
            self.vit = vit_small(
                patch_size=patch_size,
                img_size=input_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                in_chans=3,
                drop_path_rate=dropout,
                pos_embed_fn="learned",
                mlp_ratio=mlp_ratio
            )
            self.embed_dim = 384
        elif model_size == 'base':
            self.vit = vit_base(
                patch_size=patch_size,
                img_size=input_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                in_chans=3,
                drop_path_rate=dropout,
                pos_embed_fn="learned",
                mlp_ratio=mlp_ratio
            )
            self.embed_dim = 768
        elif model_size == 'large':
            self.vit = vit_large(
                patch_size=patch_size,
                img_size=input_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                in_chans=3,
                drop_path_rate=dropout,
                pos_embed_fn="learned",
                mlp_ratio=mlp_ratio
            )
            self.embed_dim = 1024
        else:
            # Custom configuration
            self.vit = VisionTransformer(
                img_size=input_size,
                patch_size=patch_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                in_chans=3,
                embed_dim=hidden_dim,
                depth=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path_rate=dropout,
                pos_embed_fn="learned"
            )
            self.embed_dim = hidden_dim

        # VisionTransformer builds a patch_embed (PatchEmbed/PatchEmbed3D) internally, but
        # ViTEncoder bypasses it by projecting pre-extracted patches directly with patch_proj.
        # Cache the attributes patchify() needs, then replace patch_embed with Identity
        # to eliminate ~295K dead parameters per encoder.
        self._patch_size_cached = self.vit.patch_embed.patch_size
        self._tubelet_size_cached = getattr(self.vit, 'tubelet_size', tubelet_size)
        self.vit.patch_embed = nn.Identity()

        self.patch_proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Projection layer to match desired hidden_dim
        if self.embed_dim != hidden_dim:
            self.projection = nn.Linear(self.embed_dim, hidden_dim)
        else:
            self.projection = nn.Identity()
        
        
    def patchify(self, video_data):
        """
        Extract raw patches from video data for MAE targets.
        
        Args:
            video_data: [batch_size, num_frames, 3, H, W]
            
        Returns:
            patches: [batch_size, num_patches, patch_dim]
        """
        # video_data: [B, T, C, H, W]
        batch_size, num_frames, channels, height, width = video_data.shape
        
        # Get patch parameters from cached attributes (patch_embed is now an Identity)
        patch_size = self._patch_size_cached
        if isinstance(patch_size, tuple):
            if len(patch_size) == 3:
                patch_t, patch_h, patch_w = patch_size
            else:
                patch_t, patch_h, patch_w = self._tubelet_size_cached, patch_size[0], patch_size[1]
        else:
            patch_t, patch_h, patch_w = self._tubelet_size_cached, patch_size, patch_size
            
        # Reshape to [B, C, T, H, W] for unfolding
        x = video_data.permute(0, 2, 1, 3, 4).contiguous()
        
        # Unfold
        # Note: We assume stride = patch_size (non-overlapping patches) for standard MAE
        x = x.unfold(2, patch_t, patch_t)
        x = x.unfold(3, patch_h, patch_h)
        x = x.unfold(4, patch_w, patch_w)
        
        # [B, C, T', H', W', pt, ph, pw]
        # Permute to [B, T', H', W', C, pt, ph, pw]
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        
        # Flatten to [B, num_patches, patch_dim]
        patches = x.view(batch_size, -1, channels * patch_t * patch_h * patch_w)
        
        return patches

    def forward(self, patched_data, masks=None):
        """
        Forward pass with patched data or raw video data.
        
        Args:
            patched_data: Either:
                - [batch_size, num_patches, patch_dim] - already patched data
                - [batch_size, num_frames, channels, height, width] - raw video data (will be patchified)
            masks: Optional indices of tokens to keep [batch_size, len_keep] (for MAE)
        
        Returns:
            If masks is not None: [batch_size, len_keep, embed_dim] - visible tokens
            If masks is None: [batch_size, embed_dim] - pooled features
        """
        # Handle both 5D video data and 3D patched data
        if patched_data.dim() == 5:
            # Raw video data: [batch_size, num_frames, channels, height, width]
            # Patchify it first
            patched_data = self.patchify(patched_data)
        elif patched_data.dim() != 3:
            raise ValueError(
                f"Expected 3D input [batch_size, num_patches, patch_dim] or "
                f"5D input [batch_size, num_frames, channels, height, width], "
                f"got {patched_data.shape}"
            )
        
        batch_size, num_patches, input_patch_dim = patched_data.shape
        
        # Project patches to embed_dim
        x = self.patch_proj(patched_data)  # [batch_size, num_patches, embed_dim]
        batch_size, num_patches, embed_dim = x.shape
        
        if self.vit.pos_embed_fn == "learned":
            pos_embed_full = self.vit.pos_embed.float()
        elif hasattr(self.vit.pos_embed, 'pos_embed'):
            # Handle cases where the sinusoidal PE is wrapped in a module
            pos_embed_full = self.vit.pos_embed.pos_embed.float()
        else:
            # For sinusoidal PE without a stored buffer, call the generator
            pos_embed_full = self.vit.pos_embed(x.device).float().unsqueeze(0)


        # Learned PE has shape [1, num_patches, D] — no CLS slot, so use directly.
        # Sinusoidal PE may be a callable or a buffer; either way truncate to num_patches.
        if pos_embed_full.shape[1] > num_patches:
            pos_embed = pos_embed_full[:, :num_patches, :]
        else:
            pos_embed = pos_embed_full
        
        # Add positional encoding
        x = x + pos_embed.to(x.device)
        
        # Apply masks if provided
        if masks is not None:
            # MAE mode: return unpooled visible tokens
            # masks: [batch_size, len_keep] (indices to keep)
            if isinstance(masks, torch.Tensor):
                masks_list = [masks]
            else:
                masks_list = masks
            
            # Apply masks using the same logic as VisionTransformer
            x = apply_masks(x, masks_list, concat=False)
            x = x[0]  # Get first (and only) masked tensor
            
            # Pass through transformer blocks
            for blk in self.vit.blocks:
                x = blk(x)
            x = self.vit.norm(x)
            
            # Project if needed
            if hasattr(self, 'projection') and not isinstance(self.projection, nn.Identity):
                x = self.projection(x)
            
            return x
        else:
            # Classification mode: return pooled feature vector
            # Pass through transformer blocks
            for blk in self.vit.blocks:
                x = blk(x)
            x = self.vit.norm(x)
            
            # Global average pooling over patches
            pooled_features = x.mean(dim=1)  # [batch_size, embed_dim]
            
            # Project to hidden dimension
            projected_features = self.projection(pooled_features)  # [batch_size, hidden_dim]
            
            return projected_features


class ViTDecoder(nn.Module):
    """
    Vision Transformer decoder for MAE reconstruction of masked patches.
    Updated to match TactileMAEDecoder structure for compatibility.
    """
    
    def __init__(self, embed_dim, num_frames, channels= 3, height= 224, width= 224,
                 patch_size=16, tubelet_size=None, depth=None, num_heads=None, dropout=0.1, mlp_ratio=4.0):
        """
        Args:
            embed_dim: Dimension of encoder embeddings (input to decoder, D_enc)
            ... (other params)
            depth: MAE typically uses a shallower decoder (e.g., 4 or 8 layers)
        """
        super(ViTDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.channels = channels
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.decoder_depth = depth
        
        # Calculate number of patches
        # Assuming non-overlapping patches/tubelets
        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size
        self.num_patches_t = num_frames // tubelet_size
        
        self.num_patches_per_frame = self.num_patches_h * self.num_patches_w
        self.total_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w
        
        # Patch dimension (flattened patch size) - the target output dimension
        self.patch_dim = channels * tubelet_size * patch_size * patch_size
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Decoder positional encoding (non-learned sinusoidal)
        self.pos_encoding = get_1d_sinusoidal_pe(self.total_patches, embed_dim).unsqueeze(0)
        self.register_buffer('decoder_pos_embed', self.pos_encoding)
        
        # Decoder transformer blocks
        # Define a simple decoder block inline (similar to TactileMAEDecoder)
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
        for _ in range(depth):
            decoder_blocks.append(
                SimpleDecoderBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    norm_layer=nn.LayerNorm,
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        # Decoder norm
        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        
        # Decoder prediction head (outputs patch_dim per token)
        self.decoder_pred = nn.Linear(self.embed_dim, self.patch_dim, bias=True)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights."""
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional encoding is non-learned (sinusoidal), so no initialization needed
        
        # Initialize prediction head
        torch.nn.init.normal_(self.decoder_pred.weight, std=0.02)
        torch.nn.init.constant_(self.decoder_pred.bias, 0)
        
    def forward(self, x, ids_restore):
        """
        Forward pass through ViT decoder.
        
        Args:
            x: Encoded visible tokens, shape [batch_size, N_visible, embed_dim]
            ids_restore: Indices to restore full sequence, shape [batch_size, N_total]
            
        Returns:
            predicted_pixels: [batch_size, N_total, patch_dim]
        """
        batch_size = x.shape[0]
        num_visible = x.shape[1]
        num_total = ids_restore.shape[1]
        
        # Append mask tokens
        num_masked = num_total - num_visible
        mask_tokens = self.mask_token.repeat(batch_size, num_masked, 1)  # [B, N_masked, D_dec]
        x_full = torch.cat([x, mask_tokens], dim=1)  # [B, N_total, D_dec]
        
        # Unshuffle to restore original order
        x_unshuffled = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )
        
        # Add positional encoding (non-learned sinusoidal)
        # Ensure the buffer is on the correct device and truncate to num_total
        pos_encoding = self.decoder_pos_embed[:, :num_total, :].to(x_unshuffled.device)
        x_unshuffled = x_unshuffled + pos_encoding
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x_unshuffled = blk(x_unshuffled)
        x_unshuffled = self.decoder_norm(x_unshuffled)
        
        # Predictor projection
        predicted_pixels = self.decoder_pred(x_unshuffled)  # [B, N_total, patch_dim]
        
        return predicted_pixels

    def get_reconstructed_video(self, predicted_pixels):
        """Reshape predicted patches [B, N_total, patch_dim] back to [B, T, C, H, W]."""
        batch_size = predicted_pixels.shape[0]

        # Each patch covers (tubelet_size, patch_size, patch_size) voxels.
        # Layout: [B, num_patches_t * num_patches_h * num_patches_w, C * tubelet_size * ph * pw]
        predicted_patches = predicted_pixels.reshape(
            batch_size,
            self.num_patches_t, self.num_patches_h, self.num_patches_w,
            self.channels, self.tubelet_size, self.patch_size, self.patch_size,
        )

        # Permute to [B, C, num_patches_t, tubelet_size, num_patches_h, patch_size, num_patches_w, patch_size]
        # then reshape to [B, C, T, H, W]
        out = predicted_patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        out = out.reshape(batch_size, self.channels, self.num_frames, self.height, self.width)

        # Return [B, T, C, H, W] to match input convention
        return out.permute(0, 2, 1, 3, 4).contiguous()


def create_vision_model(model_config):
    """Create a vision model based on architecture configuration.
    
    Args:
        model_config: Configuration dictionary containing:
            - architecture: List of modules ['encoder'] or ['encoder', 'decoder']
            - input_size: Input image size
            - num_frames: Number of frames
            - hidden_dim: Hidden dimension
            - dropout: Dropout rate
            - patch_size: Patch size
            - num_layers: Number of layers
            - num_heads: Number of attention heads
            - model_size: Model size preset
            - tubelet_size: Tubelet size for video
            - decoder: Decoder config (required if 'decoder' in architecture)
    
    Returns:
        nn.Module: Encoder or encoder+decoder wrapper
    """
    architecture = model_config.get('architecture', ['encoder'])
    
    # Create encoder
    encoder = ViTEncoder(
        input_size=model_config.get('input_size', 224),
        patch_size=model_config.get('patch_size', 16),
        num_frames=model_config.get('num_frames'),
        hidden_dim=model_config.get('hidden_dim'),
        num_layers=model_config.get('num_layers'),
        num_heads=model_config.get('num_heads'),
        model_size=model_config.get('model_size', 'tiny'),
        tubelet_size=model_config.get('tubelet_size', 1),
        mlp_ratio=model_config.get('mlp_ratio', 4.0)
    )
    
    # Return encoder only if no decoder
    if 'decoder' not in architecture:
        return encoder
    
    # Create decoder
    decoder_config = model_config.get('decoder', {})
    
    def get_decoder_param(param_name, default_value):
        return decoder_config.get(param_name, model_config.get(param_name, default_value))
    
    decoder = ViTDecoder(
        embed_dim=model_config.get('hidden_dim'),
        num_frames=model_config.get('num_frames'),
        tubelet_size=model_config.get('tubelet_size'),
        depth=get_decoder_param('decoder_depth'),
        num_heads=get_decoder_param('decoder_num_heads'),
    )
    
    # Return encoder+decoder wrapper
    class EncoderDecoderWrapper(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.embed_dim = getattr(encoder, 'embed_dim', model_config.get('hidden_dim'))
        
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
