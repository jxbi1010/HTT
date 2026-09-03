import torch
import torch.nn as nn
from model.layers.attention import CrossAttention,Attention
from model.layers.drop_path import DropPath
from model.layers.layer_scale import LayerScale
from model.layers.mlp import Mlp
from model.pos_encoding import get_1d_sinusoidal_pe


class PredictorBlock(nn.Module):
    """
    Cross-attention transformer block for embedding prediction.
    Follows standard MAE decoder approach:
    - Q: visible target tokens + mask tokens (restored to original order)
    - K/V: source embeddings (context from other modality)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        # norm for cross-attention
        self.norm_q = nn.LayerNorm(dim)
        # Norm for context (source embeddings)
        self.norm_kv = nn.LayerNorm(dim)
        # Cross-attention: Q from target tokens, K/V from source embeddings
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
        )
        self.drop_path = DropPath(0.0)
        # Norm for MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            bias=True,
        )
        self.layer_scale = LayerScale(1)
    
    def forward(self, q, kv):
        """
        Args:
            q: Query tokens [B, N_target, dim] - visible target tokens + mask tokens (restored order)
            kv: Context embeddings [B, N_source, dim] - source embeddings (K/V)
        
        Returns:
            Updated query tokens [B, N_target, dim]
        """
        # Cross-attention: Q from target tokens, K/V from source embeddings
        q = q + self.drop_path(self.layer_scale(
            self.cross_attn(self.norm_q(q), self.norm_kv(kv))
        ))

        # Self-attention MLP on query tokens
        q = q + self.drop_path(self.layer_scale(self.mlp(self.norm2(q))))
        return q


class EmbeddingPredictor(nn.Module):
    """
    Predictor that takes embeddings from one modality and predicts embeddings of another modality.
    Follows standard MAE decoder approach:
    - Query (Q): visible target embeddings + mask tokens (restored to original order)
    - Key/Value (K/V): source embeddings from the other modality
    """
    
    def __init__(self, embed_dim: int, 
                 num_tokens: int, depth: int = 3, num_heads: int = 3, 
                 mlp_ratio: float = 4.0):
        """
        Args:
            embed_dim: Dimension of embeddings (same for input and output)
            num_tokens: Number of tokens/patches in target modality
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio in transformer blocks
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # Mask token for target modality
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Positional encoding for target modality
        self.pos_encoding = get_1d_sinusoidal_pe(num_tokens, self.embed_dim).unsqueeze(0)
        self.register_buffer('decoder_pos_embed', self.pos_encoding)
        
        self.blocks = nn.ModuleList([
            PredictorBlock(self.embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.project_output = nn.Linear(self.embed_dim, self.embed_dim)
        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, target_embeddings_visible: torch.Tensor, target_ids_restore: torch.Tensor,
                source_embeddings: torch.Tensor, num_target_tokens: int):
        """
        Predict target embeddings from source embeddings using visible target embeddings as query.
        Follows standard MAE decoder approach: Q = visible target tokens + mask tokens, K/V = source embeddings.
        
        Args:
            target_embeddings_visible: [B, N_visible, embed_dim] - visible target embeddings (after masking)
            target_ids_restore: [B, N_target] - indices to restore original order
            source_embeddings: [B, N_source, embed_dim] - embeddings from source modality (used as K/V)
            num_target_tokens: Total number of target tokens (visible + masked)
        
        Returns:
            predicted_embeddings: [B, N_target, embed_dim] - predicted target embeddings
        """
        batch_size = target_embeddings_visible.shape[0]
        num_visible = target_embeddings_visible.shape[1]
        num_masked = num_target_tokens - num_visible
        
        # Create mask tokens for masked positions
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)  # [B, N_masked, embed_dim]
        
        # Concatenate visible embeddings with mask tokens: [visible, masked]
        x_full = torch.cat([target_embeddings_visible, mask_tokens], dim=1)  # [B, N_target, embed_dim]
        
        # Restore to original order using ids_restore
        # ids_restore tells us where each token should go in the original sequence
        x_full = torch.gather(x_full, dim=1, index=target_ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        # Add positional encoding
        if num_target_tokens > self.num_tokens:
            # Expand positional encoding if needed
            new_pos_encoding = get_1d_sinusoidal_pe(num_target_tokens, self.embed_dim).unsqueeze(0)
            pos_embed = new_pos_encoding.to(x_full.device)
        else:
            pos_embed = self.decoder_pos_embed[:, :num_target_tokens, :]
        
        x_full = x_full + pos_embed
        
        # Apply cross-attention transformer blocks
        # x_full: query tokens (visible target + mask tokens, restored to original order) [B, N_target, embed_dim]
        # source_embeddings: context embeddings (source modality) [B, N_source, embed_dim]
        for block in self.blocks:
            x_full = block(x_full, source_embeddings)  # Q from target tokens, K/V from source embeddings
        
        x_full = self.project_output(x_full)

        return x_full


class MeanEmbeddingPredictor(nn.Module):
    """Predictor that produces a single summary vector matching the target's
    mean-pooled embedding. Uses a learnable query that cross-attends over the
    source embeddings; no target tokens are fed in (the target side is consumed
    only by the loss, as the mean of all target tokens)."""

    def __init__(self, embed_dim: int, depth: int = 3, num_heads: int = 3,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            PredictorBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.project_output = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, source_embeddings: torch.Tensor) -> torch.Tensor:
        """source_embeddings: [B, N_src, D] → returns [B, D]."""
        B = source_embeddings.shape[0]
        q = self.query.expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, source_embeddings)
        q = self.norm_out(q)
        return self.project_output(q.squeeze(1))


class AlignPooler(nn.Module):
    """Per-modality align-token pooler.

    Holds a single learnable query (the "align token") that cross-attends over
    the FULL unmasked sequence of encoder+trunk tokens from its modality. The
    output [B, D] vector is what downstream cross-modal alignment and the force
    probe consume — because the query sees every token, fine-grained features
    (e.g. local taxel forces) are preserved in the summary.
    """

    def __init__(self, embed_dim: int, depth: int = 2, num_heads: int = 3,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable align token. Shape (1, 1, D) — broadcast over batch at runtime.
        self.align_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            PredictorBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.align_token, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, D] (full unmasked encoder+trunk output) → [B, D]."""
        B = tokens.shape[0]
        q = self.align_token.expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, tokens)
        q = self.norm_out(q)
        return q.squeeze(1)


class AlignProjector(nn.Module):
    """Per-direction cross-modal projector.

    Maps a source modality's align token → predicted target modality align token.
    Small (LayerNorm + 2-layer MLP with residual) since both sides are single
    [B, D] vectors; cross-attention isn't needed here.
    """

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))