"""Downstream example: predict 3D contact force from HTT embeddings (vision).

Pipeline:  RGB tactile frame --HTT backbone--> [B, 192] --DualForceHead--> (Fx, Fy, Fz)

A small `DualForceHead` (a 2-layer MLP, ~0.2 M params) was trained on FROZEN
HTT features to regress contact force. Pretrained heads for the two vision
modalities ship in `examples/force_heads/`. This script runs one on a bundled
real sample episode and reports predicted vs. ground-truth force.

    python examples/predict_force.py --modality gsmini
    python examples/predict_force.py --modality 9dtact
    python examples/predict_force.py --modality 9dtact --episode my_episode.npz

Force prediction is provided for the vision sensors (gsmini, 9dtact). Taxel
force regression is not shipped in this release.

Notes
-----
Force regression uses PER-EPISODE reference subtraction (the `ref_frame` stored
with each recording), which is what the head was trained on — NOT the global
background used by `htt.preprocess`. The head outputs a *normalized* force that
is de-normalized with the stats in `config/data/force.yaml`.

Episode format (`.npz`):
  tactile_img [T, 224, 224, 3] uint8, ref_frame [224, 224, 3],
  6d_force [T, 6], ref_force [6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from htt import load_model, encode, VISION_MODALITIES
from model.head import DualForceHead

PKG_ROOT = _REPO


def load_force_stats(modality: str):
    with open(PKG_ROOT / "config" / "data" / "force.yaml") as f:
        cfg = yaml.safe_load(f)
    mean = np.asarray(cfg[modality]["force_mean"], dtype=np.float64)[:3]
    std = np.asarray(cfg[modality]["force_std"], dtype=np.float64)[:3]
    return mean, std


def load_image_std(modality: str) -> torch.Tensor:
    with open(PKG_ROOT / "config" / "data" / f"{modality}.yaml") as f:
        cfg = yaml.safe_load(f)
    return torch.tensor(cfg["image_std_per_channel"], dtype=torch.float32).view(1, 3, 1, 1)


@torch.no_grad()
def infer_vision(modality, ep, encoder, trunk, head, device):
    imgs = np.asarray(ep["tactile_img"])                           # [T, H, W, 3] uint8
    ref = np.asarray(ep["ref_frame"], dtype=np.float32)            # [H, W, 3]
    if ref.max() > 2.0:
        ref = ref / 255.0
    std = load_image_std(modality)                                 # [1, 3, 1, 1]
    ref_t = torch.from_numpy(ref).float().permute(2, 0, 1)         # [3, H, W]
    T = imgs.shape[0]
    preds = np.zeros((T, 3))
    for t in range(T):
        x = torch.from_numpy(imgs[t].copy()).float().permute(2, 0, 1) / 255.0  # [3,H,W]
        x = (x - ref_t).unsqueeze(0) / (std + 1e-6)                # [1,3,H,W]
        x = x.repeat(2, 1, 1, 1).unsqueeze(0).to(device)          # [1,2,3,H,W] same frame doubled
        feats = encode(encoder, trunk, x, pool="mean")            # [1, 192]
        preds[t] = head(feats).cpu().numpy().squeeze(0)
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modality", default="gsmini", choices=list(VISION_MODALITIES))
    p.add_argument("--episode", default=None,
                   help="Episode .npz (default: assets/samples/<modality>_sample.npz).")
    p.add_argument("--ckpt", default=None, help="Backbone checkpoint (default: shipped).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep_path = Path(args.episode) if args.episode else \
        PKG_ROOT / "assets" / "samples" / f"{args.modality}_sample.npz"
    ep = np.load(ep_path, allow_pickle=True)

    # Backbone (encoder + trunk) and force head.
    load_kw = {"modality": args.modality, "device": device}
    if args.ckpt:
        load_kw["ckpt_path"] = args.ckpt
    encoder, trunk, _ = load_model(**load_kw)
    head = DualForceHead(192, hidden_dim=256).to(device).eval()
    head.load_state_dict(torch.load(
        PKG_ROOT / "examples" / "force_heads" / f"{args.modality}_force_head.pth",
        map_location=device, weights_only=False))

    # Inference (normalized) -> de-normalize to Newtons.
    preds_norm = infer_vision(args.modality, ep, encoder, trunk, head, device)
    f_mean, f_std = load_force_stats(args.modality)
    preds = preds_norm * f_std + f_mean                            # [T, 3] Newtons

    force = np.asarray(ep["6d_force"], dtype=np.float64)
    ref_force = np.asarray(ep["ref_force"], dtype=np.float64)
    gt = force[:, :3] - ref_force[None, :3]                        # [T, 3] Newtons

    l2 = np.linalg.norm(preds - gt, axis=1)
    peak = int(np.argmax(np.abs(gt[:, 2])))
    print(f"modality        : {args.modality}   ({ep_path.name}, {gt.shape[0]} frames)")
    print(f"mean L2 error   : {l2.mean():.3f} N  (over the episode)")
    print(f"peak-contact frame #{peak}:")
    print(f"  ground truth  (Fx, Fy, Fz) = ({gt[peak,0]:+.2f}, {gt[peak,1]:+.2f}, {gt[peak,2]:+.2f}) N")
    print(f"  prediction    (Fx, Fy, Fz) = ({preds[peak,0]:+.2f}, {preds[peak,1]:+.2f}, {preds[peak,2]:+.2f}) N")


if __name__ == "__main__":
    main()
