"""HTT (Heterogeneous Tactile Transformer) — load the multimodal pretrained backbone and extract features.

This is the single public entry point for *using* the model. It wraps three
things:

  1. `load_model(...)`   -> build the architecture from the shipped config and
                            load the pretrained weights, returning the
                            per-modality encoder, the shared trunk, and a
                            `preprocess(raw)` function.
  2. `preprocess(raw)`   -> turn a raw sensor reading into a model-ready tensor,
                            matching the pretraining distribution *exactly*
                            (see docs/PREPROCESSING.md — this matters).
  3. `encode(...)`       -> encoder -> shared_trunk -> pooled `[B, 192]` feature.

For the quickest possible use, `HTT` bundles all three:

    from htt import HTT
    tf = HTT(modality="gsmini")          # loads ckpt once
    emb = tf(raw_uint8_HxWx3)                       # -> [1, 192] tensor

Supported modalities:
    vision : "gsmini", "9dtact"   (GelSight-style RGB tactile images)
    taxel  : "xela",   "tac02"    (tactile sensor arrays)

All four share one 9-layer trunk and produce a fixed `[B, 192]` embedding you
can feed to any downstream head (classification, force regression, ...).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from model.create_model import create_pretrain_model

# --------------------------------------------------------------------------- #
# Paths — everything resolves relative to this file so the package works from
# any working directory.
# --------------------------------------------------------------------------- #
PKG_ROOT = Path(__file__).resolve().parent

VISION_MODALITIES = ("gsmini", "9dtact")
TAXEL_MODALITIES = ("xela", "tac02")
ALL_MODALITIES = VISION_MODALITIES + TAXEL_MODALITIES

EMBED_DIM = 192

DEFAULT_CKPT = PKG_ROOT / "checkpoints" / "htt_4sensors_best.pth"
DEFAULT_PRETRAIN_CONFIG = PKG_ROOT / "config" / "model" / "pretrain.yaml"

# Per-modality data config (holds the per-channel / per-dim std used at training).
_SENSOR_CONFIG = {
    "gsmini": PKG_ROOT / "config" / "data" / "gsmini.yaml",
    "9dtact": PKG_ROOT / "config" / "data" / "9dtact.yaml",
    "xela":   PKG_ROOT / "config" / "data" / "xela.yaml",
    "tac02":  PKG_ROOT / "config" / "data" / "tacniq.yaml",  # tac02 uses the tacniq stats
}

# No-contact reference ("background") reading per modality. These are the ones
# used during pretraining; for your own sensor/lighting, recompute your own.
_DEFAULT_BG = {
    "gsmini": PKG_ROOT / "assets" / "bg_data" / "gsmini.png",
    "9dtact": PKG_ROOT / "assets" / "bg_data" / "9dtact.png",
    "xela":   PKG_ROOT / "assets" / "bg_data" / "xela.npy",
    "tac02":  PKG_ROOT / "assets" / "bg_data" / "tac02.npy",
}

# Frame counts the encoders expect (see config/model/pretrain.yaml).
_VISION_T = 2   # image_chunk_size
_TAXEL_T = 20   # tactile_chunk_size


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(
    ckpt_path: os.PathLike | str = DEFAULT_CKPT,
    modality: str = "gsmini",
    device: Optional[torch.device] = None,
    config_path: os.PathLike | str = DEFAULT_PRETRAIN_CONFIG,
    bg_path: Optional[os.PathLike | str] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, Callable]:
    """Build the architecture, load pretrained weights, return an inference bundle.

    Args:
        ckpt_path:   path to the checkpoint (.pth). Defaults to the shipped slim
                     checkpoint (encoders + trunk + decoders).
        modality:    one of "gsmini", "9dtact", "xela", "tac02".
        device:      torch device. Defaults to cuda if available, else cpu.
        config_path: model architecture config (defaults to the shipped one).
        bg_path:     override the no-contact reference reading for preprocessing.

    Returns:
        (encoder, shared_trunk, preprocess). encoder/trunk are in eval() mode
        with requires_grad=False. `preprocess(raw)` maps a raw reading to a
        model-ready tensor on `device`.
    """
    if modality not in ALL_MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}. Supported: {ALL_MODALITIES}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    model = create_pretrain_model(cfg).to(device).eval()

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)  # slim + full ckpts both work
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    encoder = model.encoders[modality]
    shared_trunk = model.shared_trunk
    for p in encoder.parameters():
        p.requires_grad = False
    for p in shared_trunk.parameters():
        p.requires_grad = False

    preprocess = build_preprocess(modality, device, bg_path=bg_path)
    return encoder, shared_trunk, preprocess


# --------------------------------------------------------------------------- #
# Preprocessing — mirrors the pretraining dataloaders exactly.
# See docs/PREPROCESSING.md for the why behind every step.
# --------------------------------------------------------------------------- #
def _load_std(modality: str) -> torch.Tensor:
    with open(_SENSOR_CONFIG[modality]) as f:
        cfg = yaml.safe_load(f)
    if modality in VISION_MODALITIES:
        std = cfg.get("image_std_per_channel")
        if std is None:
            raise ValueError(f"`image_std_per_channel` missing in {_SENSOR_CONFIG[modality]}")
        return torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    std = cfg.get("tactile_std")
    if std is None:
        raise ValueError(f"`tactile_std` missing in {_SENSOR_CONFIG[modality]}")
    return torch.tensor(std, dtype=torch.float32).view(1, 1, -1)


def _load_bg_vision(bg_path) -> torch.Tensor:
    from PIL import Image
    bg = np.asarray(Image.open(bg_path), dtype=np.uint8)
    return torch.from_numpy(bg).float().permute(2, 0, 1) / 255.0  # [3, H, W] in [0,1]


def _load_bg_taxel(bg_path) -> torch.Tensor:
    return torch.from_numpy(np.load(bg_path)).float().reshape(-1)  # [tactile_dim]


def build_preprocess(
    modality: str,
    device: torch.device,
    bg_path: Optional[os.PathLike | str] = None,
) -> Callable:
    """Return `preprocess(raw) -> Tensor` ready for the encoder.

    vision: raw uint8 [H,W,3] or [T,H,W,3] -> float32 [1, 2, 3, H, W]
    taxel:  raw float [tactile_dim] or [T,tactile_dim] -> float32 [1, 20, tactile_dim]
    """
    std = _load_std(modality).to(device)
    bg_path = Path(bg_path) if bg_path is not None else _DEFAULT_BG[modality]
    if not os.path.exists(bg_path):
        raise FileNotFoundError(
            f"Reference background not found at {bg_path}. Provide a no-contact "
            f"frame via bg_path=..., or restore assets/bg_data/."
        )

    if modality in VISION_MODALITIES:
        bg = _load_bg_vision(bg_path).to(device)  # [3, H, W]

        def preprocess(raw: np.ndarray) -> torch.Tensor:
            arr = np.asarray(raw)
            if arr.ndim == 3:            # single frame -> add T
                arr = arr[None]
            if arr.dtype != np.uint8:
                raise ValueError(
                    f"Vision input must be uint8 (got {arr.dtype}). Pass raw camera "
                    "output; do NOT pre-divide by 255."
                )
            t = torch.from_numpy(arr).float().to(device)       # [T,H,W,3]
            t = t.permute(0, 3, 1, 2).contiguous() / 255.0     # [T,3,H,W] in [0,1]
            t = t - bg.unsqueeze(0)                             # bg subtract
            t = t / (std + 1e-6)                               # per-channel std
            t = _fit_frames(t, _VISION_T)                      # pad/trunc to T=2
            return t.unsqueeze(0)                              # [1, T, 3, H, W]

        return preprocess

    # taxel path
    bg = _load_bg_taxel(bg_path).to(device)  # [tactile_dim]

    def preprocess(raw: np.ndarray) -> torch.Tensor:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:               # single timestep -> add T
            arr = arr[None]
        t = torch.from_numpy(arr).float().to(device)   # [T, tactile_dim]
        t = t - bg.unsqueeze(0)                         # bg subtract
        t = torch.abs(t)                                # taxel rectifies (vision does NOT)
        t = t / (std.squeeze(0) + 1e-6)                 # per-dim std
        t = torch.clamp(t, min=-10.0, max=10.0)         # outlier safety cap
        t = _fit_frames(t, _TAXEL_T)                    # pad/trunc to T=20
        return t.unsqueeze(0)                           # [1, T, tactile_dim]

    return preprocess


def _fit_frames(t: torch.Tensor, target_t: int) -> torch.Tensor:
    """Pad (repeat last) or truncate along dim 0 to `target_t` frames."""
    n = t.shape[0]
    if n < target_t:
        pad = t[-1:].repeat((target_t - n,) + (1,) * (t.dim() - 1))
        return torch.cat([t, pad], dim=0)
    if n > target_t:
        return t[:target_t]
    return t


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def encode(
    encoder: torch.nn.Module,
    shared_trunk: torch.nn.Module,
    x: torch.Tensor,
    pool: str = "mean",
) -> torch.Tensor:
    """encoder -> shared_trunk -> optional pooling.

    Args:
        x:    preprocessed tensor (output of `preprocess`).
        pool: "mean" -> [B, 192]; "none" -> [B, N, 192].
    """
    with torch.no_grad():
        feats = encoder(x)
        if isinstance(feats, tuple):        # some encoders return (tokens, ...)
            feats = feats[0]
        if feats.dim() == 2:
            feats = feats.unsqueeze(1)      # [B, 1, D]
        trunk_out = shared_trunk(feats)
        if isinstance(trunk_out, tuple):
            trunk_out = trunk_out[0]
        if pool == "mean":
            return trunk_out.mean(dim=1)    # [B, D]
        if pool == "none":
            return trunk_out                # [B, N, D]
        raise ValueError(f"pool must be 'mean' or 'none', got {pool!r}")


# --------------------------------------------------------------------------- #
# Ergonomic wrapper
# --------------------------------------------------------------------------- #
class HTT:
    """One-call feature extractor: preprocessing + encoder + trunk bundled.

        tf = HTT(modality="gsmini")
        emb = tf(raw_uint8_HxWx3)      # [1, 192]
        seq = tf(raw, pool="none")     # [1, N, 192]
    """

    def __init__(
        self,
        modality: str = "gsmini",
        ckpt_path: os.PathLike | str = DEFAULT_CKPT,
        device: Optional[torch.device] = None,
        config_path: os.PathLike | str = DEFAULT_PRETRAIN_CONFIG,
        bg_path: Optional[os.PathLike | str] = None,
    ):
        self.modality = modality
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder, self.trunk, self.preprocess = load_model(
            ckpt_path=ckpt_path, modality=modality, device=self.device,
            config_path=config_path, bg_path=bg_path,
        )
        self.embed_dim = EMBED_DIM

    def __call__(self, raw, pool: str = "mean") -> torch.Tensor:
        x = self.preprocess(raw)
        return encode(self.encoder, self.trunk, x, pool=pool)

    def embed(self, raw, pool: str = "mean") -> torch.Tensor:
        """Alias for __call__."""
        return self(raw, pool=pool)
