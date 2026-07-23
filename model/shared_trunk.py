"""
Trunk definition for Transferable Tactile Transformer (T3)

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""

import os
import torch
from torch import nn
from typing import Literal
import timm.models.vision_transformer as timm_vit
import logging

class Trunk(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        kwargs = {}

        if os.path.exists(path):
            logging.info(f"Loading trunk from weights from {path}")
            self.load_state_dict(torch.load(path, **kwargs))
        else:
            logging.warning(f"Trunk weights not found at {path}. Skipping")


class TransformerTrunk(Trunk):
    """ 
    Transformer with only intermediate blocks and a final normalization layer
    """
    def __init__(self, embed_dim=768, depth=9, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 pooling_type: Literal['none', 'global', 'cls'] = 'none',
                 **kwargs):
        super().__init__()

        self.blocks = nn.ModuleList([
            timm_vit.Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pooling_type = pooling_type

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        is_mae = False
        if isinstance(x, tuple):
            (x, mask, ids_restore) = x
            is_mae = True
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.pooling_type == 'none':
            x = self.norm(x)
        elif self.pooling_type == 'global':
            x = x[:, 1:, :].mean(dim=1)
        elif self.pooling_type == 'cls':
            x = self.norm(x)
            x = x[:, 0]

        if is_mae:
            return (x, mask, ids_restore)
        else: 
            return x
