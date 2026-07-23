"""
Shared utilities for Self-Supervised Learning (SSL) training.
Contains reusable functions for MAE training and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_masking(batch_size: int, num_tokens: int, device: torch.device, mask_ratio: float):
    """
    Perform random masking for MAE.
    
    Args:
        batch_size: Batch size
        num_tokens: Number of tokens/patches in the sequence
        device: Device to create tensors on
        mask_ratio: Masking ratio
        
    Returns:
        ids_keep: Indices of tokens to keep [B, len_keep]
        mask: Binary mask (0=keep, 1=remove) [B, L]
        ids_restore: Indices to restore original order [B, L]
    """
    L = num_tokens
    len_keep = int(L * (1 - mask_ratio))
    
    # Generate random noise for each sample in batch
    noise = torch.rand(batch_size, L, device=device)  # [B, L]
    ids_shuffle = torch.argsort(noise, dim=1)  # [B, L]
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # [B, L]
    ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
    
    # Generate mask: 0 is keep, 1 is remove
    mask = torch.ones(batch_size, L, device=device)
    mask[:, :len_keep] = 0
    # Unshuffle mask to original order
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return ids_keep, mask, ids_restore




def get_gradients(x: torch.Tensor) -> tuple:
    """
    Compute image gradients using Sobel filters.
    
    Args:
        x: Input image tensor [B, C, H, W]
        
    Returns:
        grad_x: Gradient in x direction [B, C, H, W]
        grad_y: Gradient in y direction [B, C, H, W]
    """
    # Standard Sobel filters for x and y directions
    # Create base kernels [1, 1, 3, 3]
    kx_base = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky_base = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Expand kernels to match number of channels for grouped convolution
    # When using groups=C, we need [C, 1, 3, 3] kernel shape
    C = x.shape[1]
    kx = kx_base.repeat(C, 1, 1, 1).to(x.device)  # [C, 1, 3, 3]
    ky = ky_base.repeat(C, 1, 1, 1).to(x.device)  # [C, 1, 3, 3]
    
    # Compute gradients per channel using grouped convolution
    grad_x = F.conv2d(x, kx, padding=1, groups=C)
    grad_y = F.conv2d(x, ky, padding=1, groups=C)
    
    return grad_x, grad_y


def image_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert an image to patches matching the format used by patches_to_image.
    
    Args:
        x: Image tensor of shape (B, C, H, W)
        patch_size: Size of the patches
        
    Returns:
        A tensor of shape (B, N, patch_size * patch_size * C), where N is the number of patches
    """
    import einops
    B, C, H, W = x.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), "Image dimensions must be divisible by patch size."
    
    # Rearrange to patches: (B, C, H, W) -> (B, N, patch_size * patch_size * C)
    patches = einops.rearrange(
        x,
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=patch_size,
        p2=patch_size,
        h=H // patch_size,
        w=W // patch_size
    )
    return patches


def compute_mae_gradient_loss(
    pred_patches: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    img_size: tuple,
    gradient_loss_coef: float = 1.0,
    num_frames: int = None
) -> torch.Tensor:
    """
    Compute masked gradient prediction loss for MAE.
    
    This loss encourages the model to learn edge and texture features by predicting
    image gradients. The key difference from the original implementation is that:
    1. It only computes loss on MASKED patches (where mask == 1)
    2. It works directly with patches to maintain mask alignment
    3. It normalizes gradients before computing MSE
    
    Args:
        pred_patches: Predicted patches [B, L, D] where D = patch_size * patch_size * C (or C * patch_t * patch_h * patch_w for video)
        target_patches: Target patches [B, L, D]
        mask: Binary mask [B, L] where 1 = masked (remove), 0 = keep
        patch_size: Size of spatial patches (patch_h = patch_w = patch_size)
        img_size: Tuple (H, W) of image dimensions
        gradient_loss_coef: Coefficient to weight the gradient loss
        num_frames: Number of frames in video (None for single image)
        
    Returns:
        loss: Masked gradient loss (scaled by coefficient)
    """
    if gradient_loss_coef == 0.0:
        return torch.tensor(0.0, device=pred_patches.device)
    
    # Reconstruct images from patches
    from utils import patches_to_image
    
    # Try to reconstruct images directly (for single image data)
    try:
        pred_img = patches_to_image(pred_patches, patch_size, img_size)  # [B, C, H, W]
        target_img = patches_to_image(target_patches, patch_size, img_size)  # [B, C, H, W]
    except (AssertionError, ValueError) as e:
        # Reconstruction failed - likely due to video patches with temporal dimension
        # For video data, patches have dimension C * patch_t * patch_h * patch_w
        # but patches_to_image expects C * patch_h * patch_w
        # Extract first frame from video patches
        error_msg = str(e)
        if "Invalid number of patches" in error_msg:
            # This is video data - extract first frame patches
            B, L, D = pred_patches.shape
            H, W = img_size
            
            # Calculate spatial grid dimensions
            n_h, n_w = H // patch_size, W // patch_size
            num_spatial_patches = n_h * n_w
            
            # Calculate temporal patches
            num_temporal_patches = L // num_spatial_patches
            
            if num_temporal_patches > 1 and L == num_temporal_patches * num_spatial_patches:
                # Extract first frame patches (first num_spatial_patches)
                pred_first_frame = pred_patches[:, :num_spatial_patches, :]  # [B, H'*W', D]
                target_first_frame = target_patches[:, :num_spatial_patches, :]  # [B, H'*W', D]
                mask_first_frame = mask[:, :num_spatial_patches]  # [B, H'*W']
                
                # Extract spatial part from patch dimension
                # D = C * patch_t * patch_h * patch_w, we need C * patch_h * patch_w
                # Try common patch_t values (1, 2, 4)
                patch_t_candidates = [1, 2, 4]
                pred_spatial = None
                target_spatial = None
                
                for patch_t in patch_t_candidates:
                    if D % patch_t != 0:
                        continue
                    
                    C_times_patch_hw = D // patch_t
                    # Check if this can be divided by patch_size^2
                    if C_times_patch_hw % (patch_size * patch_size) != 0:
                        continue
                    
                    C = C_times_patch_hw // (patch_size * patch_size)
                    
                    # Reshape to extract first temporal slice
                    try:
                        pred_reshaped = pred_first_frame.view(B, num_spatial_patches, C, patch_t, patch_size, patch_size)
                        target_reshaped = target_first_frame.view(B, num_spatial_patches, C, patch_t, patch_size, patch_size)
                        
                        # Take first temporal slice (t=0)
                        pred_spatial = pred_reshaped[:, :, :, 0, :, :].contiguous()  # [B, H'*W', C, patch_h, patch_w]
                        target_spatial = target_reshaped[:, :, :, 0, :, :].contiguous()
                        
                        # Flatten spatial dimensions
                        pred_spatial = pred_spatial.view(B, num_spatial_patches, C * patch_size * patch_size)
                        target_spatial = target_spatial.view(B, num_spatial_patches, C * patch_size * patch_size)
                        
                        # Try to reconstruct - if this works, we found the right patch_t
                        pred_img = patches_to_image(pred_spatial, patch_size, img_size)
                        target_img = patches_to_image(target_spatial, patch_size, img_size)
                        mask = mask_first_frame  # Update mask to first frame only
                        break
                    except:
                        continue
                
                if pred_spatial is None:
                    # Couldn't extract first frame, skip gradient loss
                    return torch.tensor(0.0, device=pred_patches.device)
            else:
                # Can't determine structure, skip gradient loss
                return torch.tensor(0.0, device=pred_patches.device)
        elif "must be divisible" in error_msg:
            # Different error, skip gradient loss
            return torch.tensor(0.0, device=pred_patches.device)
        else:
            # Re-raise if it's a different error
            raise
    
    # Compute Sobel gradients
    # Compute Sobel gradients
    pred_gx, pred_gy = get_gradients(pred_img)  # [B, C, H, W] each
    target_gx, target_gy = get_gradients(target_img)  # [B, C, H, W] each
    
    # Normalize gradients to stabilize training
    # Normalize by L2 norm per channel to keep gradients in reasonable range
    def normalize_gradients(gx, gy):
        # Compute magnitude per channel
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)  # [B, C, H, W]
        # Normalize by mean magnitude per channel to keep scale consistent
        mean_mag = magnitude.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        scale = mean_mag + 1e-8
        gx_norm = gx / scale
        gy_norm = gy / scale
        return gx_norm, gy_norm
    
    pred_gx_norm, pred_gy_norm = normalize_gradients(pred_gx, pred_gy)
    target_gx_norm, target_gy_norm = normalize_gradients(target_gx, target_gy)
    
    # Concatenate gradients: [B, 2*C, H, W]
    pred_grad = torch.cat([pred_gx_norm, pred_gy_norm], dim=1)
    target_grad = torch.cat([target_gx_norm, target_gy_norm], dim=1)
    
    # Convert gradient images back to patches to apply mask
    pred_grad_patches = image_to_patches(pred_grad, patch_size)  # [B, L, 2*C*patch_size*patch_size]
    target_grad_patches = image_to_patches(target_grad, patch_size)  # [B, L, 2*C*patch_size*patch_size]
    
    # Compute MSE per patch
    grad_loss_per_patch = F.mse_loss(pred_grad_patches, target_grad_patches, reduction='none')  # [B, L, D]
    grad_loss_per_patch = grad_loss_per_patch.mean(dim=-1)  # [B, L] - mean over patch dimension
    
    # Apply mask: only compute loss on masked patches (where mask == 1)
    # mask: 0 = keep (visible), 1 = remove (masked)
    masked_grad_loss = grad_loss_per_patch * mask  # [B, L]
    
    # Average over masked patches only
    num_masked = mask.sum()  # Total number of masked patches
    if num_masked > 0:
        grad_loss = masked_grad_loss.sum() / num_masked
    else:
        # No masked patches (shouldn't happen in MAE, but handle gracefully)
        grad_loss = torch.tensor(0.0, device=pred_patches.device)
    
    return gradient_loss_coef * grad_loss




def compute_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, 
                     norm_pix_loss: bool = False) -> torch.Tensor:
    """
    Compute MAE reconstruction loss.
    
    Args:
        pred: Predicted patches/tokens [B, L, D]
        target: Target patches/tokens [B, L, D]
        mask: Binary mask [B, L] (0=keep, 1=remove)
        norm_pix_loss: Whether to normalize pixel values
        
    Returns:
        loss: Reconstruction loss on masked tokens
    """
    # Verify shapes match
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    assert mask.shape[0] == pred.shape[0] and mask.shape[1] == pred.shape[1], \
        f"Mask shape {mask.shape} doesn't match pred shape {pred.shape}"
    
    # Normalize if requested (only normalize target, model learns to predict normalized values)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
    
    # Compute MSE loss per token
    loss = F.mse_loss(pred, target, reduction='none')  # [B, L, D]
    loss = loss.mean(dim=-1)  # [B, L] - mean loss per token
    
    # Only compute loss on masked tokens
    mask_sum = mask.sum()
    if mask_sum == 0 or mask_sum == mask.numel():
        loss = loss.mean()
    else:
        loss = (loss * mask).sum() / mask_sum  # Mean loss on removed tokens
    
    return loss

def compute_weighted_mae_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor, 
    norm_pix_loss: bool = False,
    saliency_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute MAE reconstruction loss with saliency weighting.
    """
    # 1. Standard Normalization
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
    
    # 2. Compute per-token MSE
    loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # [B, L]
    
    # 3. Compute Energy-based Weights
    # We use the absolute sum of the target patches to identify contact
    energy = target.abs().mean(dim=-1) # [B, L]
    
    # Create weight map: high weight for contact, low for background
    # This prevents the background from "diluting" the contact features
    # relative threshold: patches with energy> mean are important
    threshold = energy.mean(dim=1, keepdim=True)
    weights = torch.where(energy > threshold, saliency_weight, 0.1)
    
    # 4. Combine Mask and Weights
    mask_sum = (mask * weights).sum()
    if mask_sum == 0:
        return loss.mean()
    
    # Only compute loss on masked tokens, but scale by saliency
    weighted_masked_loss = (loss * mask * weights).sum() / mask_sum
    
    return weighted_masked_loss

class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer.

    Regularizes a feature distribution toward an isotropic Gaussian using the
    Epps-Pulley characteristic-function statistic.  Intended as an auxiliary
    training loss on the shared trunk representations.

    Args:
        knots:    Number of quadrature points for the integral over t ∈ [0, 3].
        num_proj: Number of random projection directions sampled each forward call.

    Usage:
        sigreg = SIGReg(knots=17, num_proj=1024).to(device)
        # features: [B, L, D]  →  permute to [L, B, D] before passing in
        loss = sigreg(features.permute(1, 0, 2))
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: Feature tensor of shape (T, B, D), where T = sequence length,
                  B = batch size, D = feature dimension.
        Returns:
            Scalar loss value.
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()