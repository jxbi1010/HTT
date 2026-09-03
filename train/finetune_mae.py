"""MAE-finetune the pretrained backbone onto YOUR tactile sensor.

This adapts the shared trunk + a per-modality encoder/decoder to a new sensor
using masked-autoencoder reconstruction (the same objective used in
pretraining). Two sensor types:

  --sensor_type vision
      RGB tactile images (GelSight-style). The encoder + decoder are
      initialized from a pretrained vision encoder (`--init_encoder`, default
      gsmini) so you inherit its features. Inputs are resized to 224x224 and
      (by default) divided by the pretrained per-channel std so your data
      matches the distribution the encoder already knows.

  --sensor_type taxel
      Tactile arrays / force fields, flattened to a `--tactile_dim`-length
      vector per timestep. A FRESH taxel encoder + decoder is trained from
      scratch, while the pretrained 9-layer trunk is inherited (and, by
      default, given a 10x smaller LR). Inputs are abs()'d, divided by a
      per-dim std computed over your corpus, and clamped.

The trunk is always initialized from the checkpoint. Use --freeze_trunk to keep
it fixed while the encoder/decoder adapt.

--------------------------------------------------------------------------------
DATA FORMAT
--------------------------------------------------------------------------------
Point --data_dir at a directory of per-episode array files (*.npy or *.npz):

  vision : each array is [T_episode, H, W, 3]   (uint8 or float)
  taxel  : each array is [T_episode, ...]        (flattened to [T_episode, D])

For .npz files, select the array with --data_key (default: "arr_0", or the
sole key if there is only one). Episodes are split 80/20 train/val by file
index; fixed-length windows (--chunk_size) are sampled from each.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python train/finetune_mae.py --sensor_type vision --data_dir /path/to/vision_episodes
  python train/finetune_mae.py --sensor_type taxel  --tactile_dim 72 \
                         --data_dir /path/to/taxel_episodes

Output: checkpoints/finetune_<name>/<timestamp>/{best.pth, checkpoint_step_*.pth}
        (+ taxel_std.npy for taxel runs) and TensorBoard logs under logs/.
Load a finetuned run for inference by pointing htt.load_model at the
saved encoder/trunk (see docs/TRAINING.md).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Make the repo root importable when running this script directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.create_model import create_pretrain_model
from model.taxel_networks import TactileTransformerEncoder, TactileTransformerDecoder
from utils.ssl_utils import compute_mae_loss, random_masking

PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = (PKG_ROOT / "checkpoints" / "htt_4sensors_best.pth")
DEFAULT_PRETRAIN_CFG = (PKG_ROOT / "config" / "model" / "pretrain.yaml")


# --------------------------------------------------------------------------- #
# Dataset — a dependency-free directory-of-arrays loader.
# --------------------------------------------------------------------------- #
def _load_episode(path: str, data_key: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    if data_key in keys:
        return z[data_key]
    if len(keys) == 1:
        return z[keys[0]]
    raise KeyError(f"{path}: --data_key {data_key!r} not in {keys}; pass a valid --data_key.")


class ArrayEpisodeDataset(Dataset):
    """Sliding fixed-length windows over per-episode array files in a directory."""

    def __init__(self, data_dir: str, data_key: str, chunk_size: int,
                 is_val: bool, val_ratio: float = 0.2, seed: int = 42):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npy")) +
                       glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npy/.npz episode files under {data_dir}")
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(files))
        n_val = max(1, int(round(len(files) * val_ratio)))
        val_idx = set(perm[:n_val].tolist())
        selected = [f for i, f in enumerate(files) if (i in val_idx) == is_val]
        if not selected:
            raise ValueError(f"No {'val' if is_val else 'train'} episodes after split "
                             f"({len(files)} files, val_ratio={val_ratio}).")

        self.data_key = data_key
        self.chunk_size = chunk_size
        self._episodes: List[np.ndarray] = []
        self._index: List[Tuple[int, int]] = []  # (episode_idx, start)
        for f in selected:
            arr = _load_episode(f, data_key)
            if arr.shape[0] < chunk_size:
                continue
            ei = len(self._episodes)
            self._episodes.append(arr)
            for s in range(0, arr.shape[0] - chunk_size + 1):
                self._index.append((ei, s))
        if not self._index:
            raise ValueError(f"No windows of length {chunk_size} in "
                             f"{'val' if is_val else 'train'} episodes.")
        print(f"  [{'val' if is_val else 'train'}] {len(self._episodes)} episodes, "
              f"{len(self._index)} windows of length {chunk_size}")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ei, s = self._index[idx]
        arr = self._episodes[ei][s:s + self.chunk_size]
        return {"x": torch.from_numpy(np.ascontiguousarray(arr)).float()}


def make_loader(data_dir, data_key, chunk_size, batch_size, num_workers, is_val):
    ds = ArrayEpisodeDataset(data_dir, data_key, chunk_size, is_val=is_val)
    return DataLoader(ds, batch_size=batch_size, shuffle=not is_val,
                      num_workers=num_workers, pin_memory=True, drop_last=not is_val)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def load_vision_std(init_encoder: str) -> torch.Tensor:
    cfg_path = PKG_ROOT / "config" / "data" / f"{init_encoder}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return torch.tensor(cfg["image_std_per_channel"], dtype=torch.float32).view(1, 1, 3, 1, 1)


def preprocess_vision(batch_x, target_hw, image_std=None):
    """[B, T, H, W, 3] -> [B, T, 3, target_hw, target_hw], resized + optional /std."""
    if batch_x.max() > 1.5:            # looks like uint8-range input
        batch_x = batch_x / 255.0
    B, T, H, W, C = batch_x.shape
    x = batch_x.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, H, W)
    x = F.interpolate(x, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    x = x.view(B, T, C, target_hw, target_hw)
    if image_std is not None:
        x = x / (image_std.to(x.device) + 1e-6)
    return x


def compute_taxel_std(data_dir, data_key, cap: int = 30_000) -> torch.Tensor:
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")) +
                   glob.glob(os.path.join(data_dir, "*.npz")))
    chunks = []
    for f in files:
        arr = _load_episode(f, data_key)
        chunks.append(arr.reshape(arr.shape[0], -1))
    flat = np.concatenate(chunks, axis=0)
    if len(flat) > cap:
        idx = np.random.RandomState(0).choice(len(flat), cap, replace=False)
        flat = flat[idx]
    std = flat.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    return torch.from_numpy(std)


def preprocess_taxel(batch_x, std):
    """[B, T, ...] -> [B, T, D] flattened + abs + /std + clamp."""
    B, T = batch_x.shape[:2]
    x = batch_x.reshape(B, T, -1).abs()
    x = x / (std.to(x.device).view(1, 1, -1) + 1e-6)
    return torch.clamp(x, min=-10.0, max=10.0)


# --------------------------------------------------------------------------- #
# Model building
# --------------------------------------------------------------------------- #
def build_base(ckpt_path, cfg_path, device):
    cfg = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
    model = create_pretrain_model(cfg).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    return model


def build_finetune_model(args, base, device):
    trunk = base.shared_trunk
    if args.sensor_type == "vision":
        encoder = base.encoders[args.init_encoder]
        decoder = base.decoders[args.init_encoder]
    else:  # taxel — fresh init
        encoder = TactileTransformerEncoder(
            input_dim=args.tactile_dim, hidden_dim=192, patch_size=4, stride=4,
            num_heads=3, num_layers=3, dropout=0.1, use_cls_token=False,
        ).to(device)
        decoder = TactileTransformerDecoder(
            input_embed_dim=192, decoder_embed_dim=192, decoder_depth=3,
            decoder_num_heads=3, patch_dim=4 * args.tactile_dim,
        ).to(device)
    return encoder, trunk, decoder


def mae_forward(x, encoder, trunk, decoder, mask_ratio):
    patches = encoder.patchify(x)                     # [B, N, patch_dim]
    B, N, _ = patches.shape
    ids_keep, mask, ids_restore = random_masking(B, N, x.device, mask_ratio)
    enc_out = encoder(patches, masks=ids_keep)        # [B, len_keep, D]
    trunk_out = trunk((enc_out, mask, ids_restore))
    if isinstance(trunk_out, tuple):
        trunk_out = trunk_out[0]
    pred = decoder(trunk_out, ids_restore)            # [B, N, patch_dim]
    return pred, patches, mask


# --------------------------------------------------------------------------- #
# Optimizer / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(encoder, trunk, decoder, lr, trunk_lr_mult, weight_decay, freeze_trunk):
    encdec = list(encoder.parameters()) + list(decoder.parameters())
    if freeze_trunk:
        for p in trunk.parameters():
            p.requires_grad = False
        print(f"[optim] trunk FROZEN, enc+dec @ lr={lr:.1e}")
        return torch.optim.AdamW([{"params": encdec, "lr": lr, "name": "enc_dec"}],
                                 weight_decay=weight_decay)
    print(f"[optim] trunk @ lr*{trunk_lr_mult}, enc+dec @ lr={lr:.1e}")
    return torch.optim.AdamW([
        {"params": list(trunk.parameters()), "lr": lr * trunk_lr_mult, "name": "trunk"},
        {"params": encdec, "lr": lr, "name": "enc_dec"},
    ], weight_decay=weight_decay)


def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.99 + 0.01
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Train / val
# --------------------------------------------------------------------------- #
def _prep(args, x_raw, image_std, taxel_std):
    if args.sensor_type == "vision":
        return preprocess_vision(x_raw, args.target_hw, image_std=image_std)
    return preprocess_taxel(x_raw, taxel_std)


@torch.no_grad()
def validate(args, loader, encoder, trunk, decoder, device, image_std, taxel_std):
    encoder.eval(); trunk.eval(); decoder.eval()
    tot, n = 0.0, 0
    for batch in loader:
        x = _prep(args, batch["x"].to(device), image_std, taxel_std)
        pred, target, mask = mae_forward(x, encoder, trunk, decoder, args.mask_ratio)
        loss = compute_mae_loss(pred, target, mask, norm_pix_loss=False)
        tot += loss.item() * x.shape[0]; n += x.shape[0]
    encoder.train(); decoder.train()
    if not args.freeze_trunk:
        trunk.train()
    return tot / max(n, 1)


def save_ckpt(path, encoder, trunk, decoder, step, best_val, args):
    torch.save({
        "sensor_type": args.sensor_type,
        "step": step,
        "best_val_mae": best_val,
        "encoder_state_dict": encoder.state_dict(),
        "trunk_state_dict": trunk.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "args": vars(args),
    }, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    base = build_base(args.init_ckpt, args.init_pretrain_config, device)
    encoder, trunk, decoder = build_finetune_model(args, base, device)

    image_std, taxel_std = None, None
    if args.sensor_type == "vision":
        if not args.no_image_std_normalize:
            image_std = load_vision_std(args.init_encoder)
            print(f"[vision] using {args.init_encoder} image_std normalization")
    else:
        print("[taxel] computing per-dim std over corpus...")
        taxel_std = compute_taxel_std(args.data_dir, args.data_key)
        print(f"  std mean={taxel_std.mean():.4f} min={taxel_std.min():.4f} max={taxel_std.max():.4f}")

    print(f"[data] loading from {args.data_dir}")
    train_loader = make_loader(args.data_dir, args.data_key, args.chunk_size,
                               args.batch_size, args.num_workers, is_val=False)
    val_loader = make_loader(args.data_dir, args.data_key, args.chunk_size,
                             args.batch_size, args.num_workers, is_val=True)

    optim = build_optimizer(encoder, trunk, decoder, args.lr, args.trunk_lr_mult,
                            args.weight_decay, args.freeze_trunk)
    sched = build_scheduler(optim, args.total_steps, args.warmup_steps)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = PKG_ROOT / "checkpoints" / f"finetune_{args.name}" / ts
    log_root = PKG_ROOT / "logs" / f"finetune_{args.name}" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_root))
    with open(out_root / "args.yaml", "w") as f:
        yaml.safe_dump(vars(args), f)
    if taxel_std is not None:
        np.save(out_root / "taxel_std.npy", taxel_std.cpu().numpy())
    print(f"[out] {out_root}")

    step, epoch, best_val = 0, 0, float("inf")
    encoder.train(); decoder.train()
    trunk.train() if not args.freeze_trunk else trunk.eval()
    t0 = time.time()
    while step < args.total_steps:
        for batch in train_loader:
            if step >= args.total_steps:
                break
            x = _prep(args, batch["x"].to(device), image_std, taxel_std)
            pred, target, mask = mae_forward(x, encoder, trunk, decoder, args.mask_ratio)
            loss = compute_mae_loss(pred, target, mask, norm_pix_loss=False)
            optim.zero_grad(); loss.backward()
            clip = [p for p in (list(encoder.parameters()) + list(trunk.parameters())
                                + list(decoder.parameters())) if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(clip, max_norm=1.0)
            optim.step(); sched.step()

            if step % args.log_every == 0:
                writer.add_scalar("train/mae", loss.item(), step)
                its = (step + 1) / max(time.time() - t0, 1e-6)
                print(f"  step {step:6d}/{args.total_steps} ep{epoch} "
                      f"mae={loss.item():.4f} {its:.1f}it/s")
            if step > 0 and step % args.val_every == 0:
                v = validate(args, val_loader, encoder, trunk, decoder, device, image_std, taxel_std)
                writer.add_scalar("val/mae", v, step)
                print(f"  [val] step {step} val_mae={v:.4f}")
                if v < best_val:
                    best_val = v
                    save_ckpt(out_root / "best.pth", encoder, trunk, decoder, step, best_val, args)
                    print("  [val] new best -> best.pth")
            if step > 0 and step % args.save_every == 0:
                save_ckpt(out_root / f"checkpoint_step_{step}.pth", encoder, trunk, decoder,
                          step, best_val, args)
            step += 1
        epoch += 1

    save_ckpt(out_root / f"checkpoint_step_{step}.pth", encoder, trunk, decoder, step, best_val, args)
    print(f"[done] {step} steps in {(time.time()-t0)/60:.1f} min. best_val={best_val:.4f}")
    writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sensor_type", choices=["vision", "taxel"], required=True)
    p.add_argument("--data_dir", required=True, help="Directory of *.npy/*.npz episode files.")
    p.add_argument("--data_key", default="arr_0", help="Array key inside .npz files.")
    p.add_argument("--name", default=None, help="Run name (default: sensor_type).")
    # vision
    p.add_argument("--init_encoder", default="gsmini", choices=["gsmini", "9dtact"],
                   help="[vision] pretrained vision encoder to initialize from.")
    p.add_argument("--target_hw", type=int, default=224)
    p.add_argument("--no_image_std_normalize", action="store_true",
                   help="[vision] disable /image_std normalization.")
    # taxel
    p.add_argument("--tactile_dim", type=int, default=None,
                   help="[taxel] flattened per-timestep input dimension (required for taxel).")
    # shared
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Frames per window (default: 2 vision, 20 taxel).")
    p.add_argument("--mask_ratio", type=float, default=None,
                   help="MAE mask ratio (default: 0.75 vision, 0.6 taxel).")
    p.add_argument("--init_ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--init_pretrain_config", default=str(DEFAULT_PRETRAIN_CFG))
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--trunk_lr_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--freeze_trunk", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Fill sensor-type-dependent defaults.
    if args.name is None:
        args.name = args.sensor_type
    if args.chunk_size is None:
        args.chunk_size = 2 if args.sensor_type == "vision" else 20
    if args.mask_ratio is None:
        args.mask_ratio = 0.75 if args.sensor_type == "vision" else 0.6
    if args.sensor_type == "taxel" and args.tactile_dim is None:
        p.error("--tactile_dim is required for --sensor_type taxel")
    train(args)


if __name__ == "__main__":
    main()
