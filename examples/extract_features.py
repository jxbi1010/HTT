"""CLI demo: turn a raw tactile sensor reading into a [B, 192] feature vector.

    # vision (gsmini / 9dtact): pass a PNG/JPEG, resized to 224x224 internally
    python examples/extract_features.py --modality gsmini --input path/to/frame.png

    # taxel (xela / tac02): pass a .npy array of shape [T, tactile_dim] or [tactile_dim]
    python examples/extract_features.py --modality xela --input path/to/reading.npy

    # no --input: runs on the bundled REAL sample (assets/samples/<modality>_sample.npz)
    python examples/extract_features.py --modality gsmini

The output is the mean-pooled shared-trunk feature — feed it to any downstream
head. See docs/PREPROCESSING.md for the exact raw-input contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable when running this script directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htt import (
    ALL_MODALITIES,
    VISION_MODALITIES,
    DEFAULT_CKPT,
    load_model,
    encode,
)

PKG_ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in examples/)

# Canonical raw shapes the model was pretrained on.
_VISION_HW = 224
_TAXEL_DIM = {"xela": 72, "tac02": 66}


def _load_sample(modality: str, is_vision: bool):
    """Pull a real raw reading from the bundled sample episode, if present."""
    path = PKG_ROOT / "assets" / "samples" / f"{modality}_sample.npz"
    if not path.exists():
        return None
    ep = np.load(path, allow_pickle=True)
    k = int(ep["peak_local"]) if "peak_local" in ep else 0
    if is_vision:
        return np.asarray(ep["tactile_img"])[k].astype(np.uint8)      # [224,224,3]
    tac = np.asarray(ep["tactile"], dtype=np.float32)                 # [T, D]
    return tac[max(0, k - 19): k + 1]                                 # [<=20, D]


def _load_vision_input(path: str) -> np.ndarray:
    """Load a PNG/JPEG as uint8 [224, 224, 3] (resized, alpha dropped)."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((_VISION_HW, _VISION_HW))
    return np.asarray(img, dtype=np.uint8)


def _synth_vision() -> np.ndarray:
    return np.random.randint(0, 256, (_VISION_HW, _VISION_HW, 3), dtype=np.uint8)


def _synth_taxel(modality: str) -> np.ndarray:
    return np.random.randn(20, _TAXEL_DIM[modality]).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modality", default="gsmini", choices=ALL_MODALITIES)
    p.add_argument("--input", default=None,
                   help="PNG/JPEG (vision) or .npy (taxel). Omit for a synthetic smoke test.")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--pool", default="mean", choices=["mean", "none"])
    args = p.parse_args()

    is_vision = args.modality in VISION_MODALITIES
    if args.input is None:
        raw = _load_sample(args.modality, is_vision)
        if raw is not None:
            print(f"[info] no --input given; using the bundled REAL {args.modality} "
                  f"sample (shape {raw.shape}).")
        else:
            raw = _synth_vision() if is_vision else _synth_taxel(args.modality)
            print(f"[warn] no --input and no bundled sample; using SYNTHETIC "
                  f"{args.modality} reading of shape {raw.shape}.")
    elif is_vision:
        raw = _load_vision_input(args.input)
    else:
        raw = np.load(args.input)

    encoder, trunk, preprocess = load_model(ckpt_path=args.ckpt, modality=args.modality)
    x = preprocess(raw)
    feats = encode(encoder, trunk, x, pool=args.pool)

    print(f"modality      : {args.modality}")
    print(f"model input   : {tuple(x.shape)}")
    print(f"feature shape : {tuple(feats.shape)}")
    print(f"feature norm  : {feats.norm().item():.4f}")


if __name__ == "__main__":
    main()
