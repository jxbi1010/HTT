"""
ResNet-18 backbone encoder for image/video input.
Accepts [B, T, C, H, W], runs ResNet18 on each frame, returns [B, T, embed_dim].
Used for force prediction and other downstream tasks.
"""

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    resnet18 = None
    ResNet18_Weights = None


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 encoder that accepts 5D input [batch, time, channels, height, width].
    Runs ResNet18 on each frame (or on mean frame if pool_time=True), returns
    [B, T, 512] (pool_time=False) or [B, 512] (pool_time=True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 512,
        num_frames: int = 2,
        pool_time: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        if resnet18 is None:
            raise ImportError("torchvision is required for ResNet18Encoder. Install with: pip install torchvision")
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.pool_time = pool_time
        self.in_channels = in_channels

        # Load ResNet18 (features only: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool)
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained and ResNet18_Weights is not None else None
            resnet = resnet18(weights=weights)
        except TypeError:
            # Older torchvision: resnet18(pretrained=bool)
            resnet = resnet18(pretrained=pretrained)
        # Remove fc so we get 512-d feature vector
        resnet.fc = nn.Identity()
        if in_channels != 3:
            # Replace first conv for different input channels
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = resnet

    def forward(self, x, masks=None):
        """
        Args:
            x: [B, T, C, H, W] or [B, C, H, W]
            masks: unused (for API compatibility with ViT encoder)
        Returns:
            [B, T, embed_dim] if pool_time=False, else [B, embed_dim]
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # (B*T, C, H, W)
            x = x.view(B * T, C, H, W)
            features = self.backbone(x)  # (B*T, 512)
            features = features.view(B, T, -1)
            if self.pool_time:
                features = features.mean(dim=1)  # (B, 512)
            return features
        elif x.dim() == 4:
            features = self.backbone(x)  # (B, 512)
            return features
        else:
            raise ValueError(f"ResNet18Encoder expected 4D or 5D input, got shape {x.shape}")

    @property
    def hidden_dim(self):
        return self.embed_dim


def create_resnet18_encoder(config: dict) -> ResNet18Encoder:
    """Build ResNet18Encoder from config dict."""
    return ResNet18Encoder(
        in_channels=config.get("in_channels", 3),
        embed_dim=config.get("embed_dim", 512),
        num_frames=config.get("num_frames", 2),
        pool_time=config.get("pool_time", False),
        pretrained=config.get("pretrained", False),
    )
