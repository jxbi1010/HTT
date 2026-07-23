import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_1d_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """
    Generates 1D Sinusoidal Positional Encoding.
    
    Args:
        max_len (int): The length of the sequence.
        d_model (int): The dimension of the embedding.
        
    Returns:
        torch.Tensor: Shape (max_len, d_model)
    """
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    # Calculate the division term for the frequency
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    # For odd d_model, we need to slice div_term to match the number of odd-indexed columns
    pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
    
    return pe


def get_3d_sinusoidal_pe_for_patches(T_prime: int, H_prime: int, W_prime: int, hidden_dim: int) -> torch.Tensor:
    """
    Generates 3D Sinusoidal Positional Encoding by concatenating T, H, and W embeddings.
    
    Args:
        T_prime (int): Number of patches in time dimension.
        H_prime (int): Number of patches in height dimension.
        W_prime (int): Number of patches in width dimension.
        hidden_dim (int): Hidden dimension of the model.
        
    Returns:
        torch.Tensor: Sinusoidal PE for the current patch sequence. Shape (1, T'*H'*W', hidden_dim).
    """
    
    # Allocate dimensions for T, H, W. The remainder goes to D_t.
    pe_t = get_1d_sinusoidal_pe(T_prime, hidden_dim)  # (T', d_dim)
    pe_h = get_1d_sinusoidal_pe(H_prime, hidden_dim)  # (H', d_dim)
    pe_w = get_1d_sinusoidal_pe(W_prime, hidden_dim)  # (W', d_dim)
    
    # 2. Create 3D grid of positional embeddings (outer product equivalent)
    # Expand PEs to match the final T' x H' x W' grid size
    pe_t_3d = pe_t.view(T_prime, 1, 1, hidden_dim).expand(-1, H_prime, W_prime, -1)
    pe_h_3d = pe_h.view(1, H_prime, 1, hidden_dim).expand(T_prime, -1, W_prime, -1)
    pe_w_3d = pe_w.view(1, 1, W_prime, hidden_dim).expand(T_prime, H_prime, -1, -1)

    pe_3d = pe_t_3d + pe_h_3d + pe_w_3d  # (T', H', W', hidden_dim)
    
    # 4. Flatten and reshape to (1, Num_Tokens, D) for addition to the token sequence
    pe_final = pe_3d.flatten(0, 2).unsqueeze(0).contiguous()
    
    return pe_final