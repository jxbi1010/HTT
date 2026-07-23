# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

# from utils.logging import get_pylogger
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention, CrossAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

# logger = get_pylogger(__name__)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias: bool = False,
        ffn_bias: bool = True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        ffn_layer="mlp",
        init_values=1,
    ):
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
        if ffn_layer == "mlp":
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=ffn_bias,
            )
        else:
            raise NotImplementedError

        self.layer_scale = LayerScale(init_values) if init_values > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.layer_scale(self.mlp(self.norm2(x))))
        return x


class MAEDecoder(nn.Module):
    """MAE Decoder for masked autoencoder."""
    
    def __init__(
        self,
        in_chans=3,
        img_size=224,
        input_embed_dim=768,
        patch_size=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.decoder_embed_dim = decoder_embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Decoder embedding
        self.decoder_embed = nn.Linear(input_embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        
        # Decoder transformer blocks
        decoder_blocks = []
        for _ in range(decoder_depth):
            decoder_blocks.append(
                DecoderBlock(
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
        
        # Decoder prediction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights."""
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize positional embedding
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # Initialize prediction head
        torch.nn.init.normal_(self.decoder_pred.weight, std=0.02)
        torch.nn.init.constant_(self.decoder_pred.bias, 0)
    
    def forward(self, x, ids_restore):
        """
        Forward pass through decoder.
        Args:
            x: [N, L, D] encoded features
            ids_restore: [N, L] indices to restore original order
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Unshuffle
        
        # Add cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x


class IJEPADecoder(nn.Module):
    """I-JEPA Decoder for joint embedding predictive architecture."""
    
    def __init__(
        self,
        in_chans=3,
        img_size=224,
        input_embed_dim=768,
        patch_size=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # For now, use the same implementation as MAE
        # This can be customized later for I-JEPA specific requirements
        self.mae_decoder = MAEDecoder(
            in_chans=in_chans,
            img_size=img_size,
            input_embed_dim=input_embed_dim,
            patch_size=patch_size,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
    
    def forward(self, x, ids_restore):
        return self.mae_decoder(x, ids_restore)


class VJEPADecoder(nn.Module):
    """V-JEPA Decoder for video joint embedding predictive architecture."""
    
    def __init__(
        self,
        in_chans=3,
        img_size=224,
        input_embed_dim=768,
        patch_size=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # For now, use the same implementation as MAE
        # This can be customized later for V-JEPA specific requirements
        self.mae_decoder = MAEDecoder(
            in_chans=in_chans,
            img_size=img_size,
            input_embed_dim=input_embed_dim,
            patch_size=patch_size,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
    
    def forward(self, x, ids_restore):
        return self.mae_decoder(x, ids_restore)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias: bool = False,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        self_attn_class: Callable[..., nn.Module] = Attention,
        cross_attn_class: Callable[..., nn.Module] = CrossAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = self_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.q_norm2 = norm_layer(dim)
        self.kv_norm2 = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, q, kv):
        def self_attn_residual_func(q: Tensor) -> Tensor:
            return self.ls1(self.self_attn(self.norm1(q)))

        def cross_attn_residual_func(q: Tensor, kv: Tensor) -> Tensor:
            return self.ls2(self.cross_attn(self.q_norm2(q), self.kv_norm2(kv)))

        def ffn_residual_func(q: Tensor) -> Tensor:
            return self.ls3(self.mlp(self.norm3(q)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            q = drop_add_residual_stochastic_depth(
                [q],
                residual_func=self_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            q = drop_add_residual_stochastic_depth(
                [q, kv],
                residual_func=cross_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            q = drop_add_residual_stochastic_depth(
                [q],
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            q = q + self.drop_path1(self_attn_residual_func(q))
            q = q + self.drop_path2(cross_attn_residual_func(q, kv))
            q = q + self.drop_path3(ffn_residual_func(q))
        else:
            q = q + self_attn_residual_func(q)
            q = q + cross_attn_residual_func(q, kv)
            q = q + ffn_residual_func(q)

        return q


def drop_add_residual_stochastic_depth(
    xs: List[Tensor],
    residual_func: Callable[[List[Tensor]], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    q = xs[0]
    b, _, _ = q.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=q.device))[:sample_subset_size]
    xs_subset = [x[brange] for x in xs]

    # 2) apply residual_func to get residual
    residual = residual_func(*xs_subset)

    q_flat = q.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    q_plus_residual = torch.index_add(
        q_flat, 0, brange, residual.to(dtype=q.dtype), alpha=residual_scale_factor
    )
    return q_plus_residual.view_as(q)


def get_branges_scales(xs: List[Tensor], sample_drop_ratio: float = 0.0):
    q = xs[0]
    b, _, _ = q.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=q.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(
    x: Tensor,
    brange: Tensor,
    residual: Tensor,
    residual_scale_factor: float,
    scaling_vector=None,
):
    if scaling_vector is None:
        x_plus_residual = torch.index_add(
            x.flatten(1),
            0,
            brange,
            residual.flatten(1).to(dtype=x.dtype),
            alpha=residual_scale_factor,
        )
    else:
        x_plus_residual = scaled_index_add(
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}
