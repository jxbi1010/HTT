"""
MLP and ResNet head models for classification and regression tasks.
"""

import torch
import torch.nn as nn

class MLPHead(nn.Module):
    """
    3-layer MLP head for classification or regression.
    
    Architecture:
        Input -> Linear -> GELU -> Dropout
        -> Linear -> GELU -> Dropout  
        -> Linear -> Output
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output dimension (number of classes for classification, or output size for regression)
        hidden_dim: Hidden layer dimension (default: in_dim)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = in_dim
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.mlp = nn.Sequential(
            # nn.LayerNorm(in_dim), # Added: Stabilizes pretrained features
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
        )

        
        if out_dim == 1:
            self._init_weights_kaiming()
        else:
            self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _init_weights_kaiming(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, in_dim] or [batch_size, seq_len, in_dim]
        
        Returns:
            output: [batch_size, out_dim] or [batch_size, seq_len, out_dim]
                   For classification: logits (no activation)
                   For regression: continuous values (no activation)
        """
        # Handle sequence input (if x is 3D, apply MLP to each timestep)
        if x.dim() == 3:
            batch_size, seq_len, in_dim = x.shape
            x = x.view(-1, in_dim)  # [batch_size * seq_len, in_dim]
            output = self.mlp(x)  # [batch_size * seq_len, out_dim]
            output = output.view(batch_size, seq_len, self.out_dim)  # [batch_size, seq_len, out_dim]
        else:
            output = self.mlp(x)  # [batch_size, out_dim]
        
        return output


class DualForceHead(nn.Module):
    """
    Two-head MLP for force prediction: one head predicts shear (dims 0,1), another predicts normal (dim 2).
    Total loss = loss_shear + loss_normal.

    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 3  # shear (2) + normal (1)
        self.shear_head = MLPHead(in_dim, 2, hidden_dim=hidden_dim, dropout=dropout)
        self.normal_head = MLPHead(in_dim, 1, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns concatenated [shear, normal] = [B, 3].

        Args:
            x: Input features [batch_size, in_dim] or [batch_size, seq_len, in_dim]

        Returns:
            output: [batch_size, 3] or [batch_size, seq_len, 3]
        """
        shear = self.shear_head(x)  # [B, 2] or [B, seq, 2]
        normal = self.normal_head(x)  # [B, 1] or [B, seq, 1]
        return torch.cat([shear, normal], dim=-1)