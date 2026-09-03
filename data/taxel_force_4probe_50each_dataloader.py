#!/usr/bin/env python3
"""
Dataloader for taxel-based tactile + 6D force data (4-probe, 50 each) from processed .npz episodes.
Supports tac02 (66-dim) and xela (72-dim) sensors.

Each .npz contains: ref_tactile (taxel_dim,), ref_force (6,), tactile (T, taxel_dim), 6d_force (T, 6),
probe (int), mode (str "static" or "sliding").
Force convention: first 3 dims are [shear_x, shear_y, normal].

Yields windowed samples: tactile chunk (chunk_size, taxel_dim) and 6d_force (6,) at the last timestep.
Applies reference subtraction to both tactile and force.

Batch: tactile [B, chunk_size, taxel_dim], 6d_force [B, 6], probe [B,], mode [list of str]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def compute_and_save_taxel_force_stats(
    data_root: str,
    modality: str,
    config_dir: str = "config/data",
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute and save per-dim force mean/std from .npz episodes."""
    data_root = Path(data_root)
    episode_paths = sorted(data_root.rglob("*.npz"))
    if not episode_paths:
        raise ValueError(f"No .npz files found in {data_root}")

    all_force = []
    for path in episode_paths:
        try:
            data = np.load(path, allow_pickle=True)
            force = np.asarray(data["6d_force"], dtype=np.float64)
            all_force.append(force)
        except Exception:
            continue
    if not all_force:
        raise ValueError("No valid force data found")

    all_force = np.concatenate(all_force, axis=0)
    force_mean = np.mean(all_force, axis=0).astype(np.float64)
    force_std = np.std(all_force, axis=0).astype(np.float64)
    force_std = np.where(force_std > 1e-8, force_std, 1.0)

    out_path = Path(output_path) if output_path else Path(config_dir) / "force.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = yaml.safe_load(f) or {}
    existing[modality] = {
        "force_mean": force_mean.tolist(),
        "force_std": force_std.tolist(),
    }
    with open(out_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    print(f"Saved {modality} force stats to {out_path}")
    return force_mean, force_std


class TaxelForce4Probe50EachDataset(IterableDataset):
    """
    Iterable dataset over processed taxel + force .npz episodes (4-probe, 50 each).
    Yields windowed chunks: tactile (chunk_size, taxel_dim), 6d_force (6,) at last step.
    """

    def __init__(
        self,
        modality: str,
        taxel_dim: int,
        data_root: Optional[str] = None,
        split: str = "train",
        chunk_size: int = 20,
        config_dir: str = "config/data",
        apply_ref_tactile_subtraction: bool = True,
        apply_ref_force_subtraction: bool = True,
        apply_force_normalization: bool = True,
        apply_tactile_abs: bool = True,
        apply_tactile_normalization: bool = True,
        apply_tactile_clamp: bool = True,
        tactile_clamp_min: float = -10.0,
        tactile_clamp_max: float = 10.0,
        force_clip_min: Optional[float] = -20.0,
        force_clip_max: Optional[float] = 20.0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        mode_filter: Optional[str] = None,
        stride: int = 1,
        labeled_data_root: Optional[str] = None,
        load_sliding_labels: Optional[bool] = None,
        strict_labeled: bool = False,
    ):
        """
        Args:
            modality: 'tac02' or 'xela'
            taxel_dim: Number of taxel values (66 for tac02, 72 for xela)
            data_root: Path to processed dir with p*_*/*.npz subfolders.
            chunk_size: Number of timesteps per sample (default 20, matches classification)
            stride: Step between consecutive windows (default 1 = every timestep)
            mode_filter: 'static', 'sliding', or None for all
            apply_tactile_abs / apply_tactile_normalization / apply_tactile_clamp:
                Match the preprocessing applied by the pretrain dataloader
                (data/xela_9dtact_dataloader_webdataset.py:595-606): after
                ref subtraction, rectify with abs(), divide by per-dim
                tactile_std (from config/data/<modality>.yaml), and clamp to
                [-10, 10]. Defaults are True so the dataloader output is
                in-distribution for the alignscratch-pretrained encoder.

                NOTE: this was added on 2026-05-23. Earlier runs (the §7
                alignscratch FT-probe force/sliding numbers and all SPL
                force/sliding numbers in results/BREAKDOWN_3SEED_ALL.md)
                used the old behavior (all three flags effectively False —
                only ref subtraction). For exact reproducibility of those
                numbers, pass `apply_tactile_abs=False,
                apply_tactile_normalization=False,
                apply_tactile_clamp=False`. Re-running the alignscratch
                FT-probe with the new defaults is expected to improve
                xela force (~0.1-0.2 N MAE tighter) and may help xela
                sliding; tac02 is largely insensitive (its dynamic range
                is small enough that std/clamp have little effect).
            tactile_clamp_min / tactile_clamp_max: clamp bounds; defaults
                match the pretrain dataloader.
            labeled_data_root: Root for ``*.labeled.npz`` (default: sibling ``sliding_labeled`` of processed).
            load_sliding_labels: If True, emit ``sliding_label`` per chunk (label at chunk's last timestep);
                None = auto (on when mode_filter is 'sliding').
            strict_labeled: If True, skip episodes without a valid matching ``.labeled.npz``.
        """
        self.modality = modality
        self.taxel_dim = taxel_dim
        self.chunk_size = chunk_size
        self.stride = stride
        self.apply_ref_tactile_subtraction = apply_ref_tactile_subtraction
        self.apply_ref_force_subtraction = apply_ref_force_subtraction
        self.apply_force_normalization = apply_force_normalization
        self.apply_tactile_abs = apply_tactile_abs
        self.apply_tactile_normalization = apply_tactile_normalization
        self.apply_tactile_clamp = apply_tactile_clamp
        self.tactile_clamp_min = tactile_clamp_min
        self.tactile_clamp_max = tactile_clamp_max
        self.force_clip_min = force_clip_min
        self.force_clip_max = force_clip_max

        if data_root is None:
            data_root = f"data/{modality}_force_4probe_50each/processed"
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")

        if load_sliding_labels is None:
            load_sliding_labels = mode_filter == "sliding"
        self.load_sliding_labels = load_sliding_labels
        self.strict_labeled = strict_labeled
        if labeled_data_root is not None:
            self.labeled_data_root = Path(labeled_data_root).resolve()
        else:
            self.labeled_data_root = (self.data_root.parent / "sliding_labeled").resolve()

        self._load_force_stats(config_dir)
        self._load_tactile_std(config_dir)

        all_paths = sorted(self.data_root.glob("p*_*/*.npz"))
        if mode_filter is not None:
            suffix = f"_{mode_filter}"
            all_paths = [p for p in all_paths if p.parent.name.endswith(suffix)]
            if not all_paths:
                raise ValueError(f"No episodes for mode_filter='{mode_filter}' in {self.data_root}")
        self.episode_paths = all_paths
        if not self.episode_paths:
            raise ValueError(f"No .npz episodes found in {self.data_root}")

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.episode_paths))
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if split == "train":
            self._paths = [self.episode_paths[i] for i in idx[:n_train]]
        elif split == "val":
            self._paths = [self.episode_paths[i] for i in idx[n_train:n_train + n_val]]
        else:
            self._paths = [self.episode_paths[i] for i in idx[n_train + n_val:]]

        if not self._paths:
            raise ValueError(f"No episodes in split '{split}'")

        self._shuffle_seed = seed
        self._epoch = 0

    def _load_force_stats(self, config_dir: str):
        force_path = Path(config_dir) / "force.yaml"
        if not force_path.exists():
            force_path = Path("config/data/force.yaml")
        self.force_mean = np.zeros(6)
        self.force_std = np.ones(6)
        if force_path.exists():
            try:
                with open(force_path) as f:
                    cfg = yaml.safe_load(f) or {}
                if self.modality in cfg:
                    m = cfg[self.modality]
                    if "force_mean" in m:
                        self.force_mean = np.array(m["force_mean"], dtype=np.float64)
                    if "force_std" in m:
                        self.force_std = np.array(m["force_std"], dtype=np.float64)
            except Exception:
                pass

    def _load_tactile_std(self, config_dir: str):
        """Per-dim tactile std from config/data/<modality>.yaml. Mirrors
        xela_9dtact_dataloader_webdataset.py's _tactile_std_tensor."""
        # tac02 sensor config lives at config/data/tacniq.yaml
        cfg_name = "tacniq" if self.modality == "tac02" else self.modality
        cfg_path = Path(config_dir) / f"{cfg_name}.yaml"
        if not cfg_path.exists():
            cfg_path = Path(f"config/data/{cfg_name}.yaml")
        self.tactile_std = np.ones(self.taxel_dim, dtype=np.float32)
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                if "tactile_std" in cfg and isinstance(cfg["tactile_std"], list):
                    arr = np.array(cfg["tactile_std"], dtype=np.float32)
                    if arr.shape == (self.taxel_dim,):
                        self.tactile_std = arr
            except Exception:
                pass

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _labeled_npz_path(self, episode_path: Path) -> Path:
        """Map ``.../processed/p1_sliding/foo.npz`` → ``.../sliding_labeled/p1_sliding/foo.labeled.npz``."""
        try:
            rel = episode_path.resolve().relative_to(self.data_root.resolve())
        except ValueError:
            rel = Path(episode_path.name)
        return self.labeled_data_root / rel.with_suffix(".labeled.npz")

    def __iter__(self):
        rng = np.random.RandomState(self._shuffle_seed + self._epoch)
        order = rng.permutation(len(self._paths))
        paths = [self._paths[i] for i in order]

        worker_info = get_worker_info()
        if worker_info is not None:
            per = int(math.ceil(len(paths) / worker_info.num_workers))
            start = worker_info.id * per
            paths = paths[start:start + per]

        for path in paths:
            try:
                data = np.load(path, allow_pickle=True)
                ref_tactile = np.asarray(data["ref_tactile"], dtype=np.float32)
                ref_force = np.asarray(data["ref_force"], dtype=np.float64) if "ref_force" in data else np.zeros(6)
                tactile = np.asarray(data["tactile"], dtype=np.float32)
                force = np.asarray(data["6d_force"], dtype=np.float64)
                probe = int(data["probe"]) if "probe" in data else 0
                mode = str(data["mode"]) if "mode" in data else "unknown"
            except Exception:
                continue

            T = tactile.shape[0]
            if T == 0 or force.shape[0] != T:
                continue

            lb_arr = None
            if self.load_sliding_labels:
                lp = self._labeled_npz_path(path)
                if lp.exists():
                    try:
                        L = np.load(lp, allow_pickle=True)
                        lb_arr = np.asarray(L["sliding_labels_bracket"], dtype=np.int64).reshape(-1)
                        if lb_arr.shape[0] != T:
                            lb_arr = None
                    except Exception:
                        lb_arr = None
                if self.strict_labeled and lb_arr is None:
                    continue

            # Pad start with reference frame if needed for first window.
            # Padded steps map to label -1 (ignored) since they're synthetic.
            if T < self.chunk_size:
                pad = self.chunk_size - T
                tactile_padded = np.concatenate([np.tile(ref_tactile[None], (pad, 1)), tactile], axis=0)
                force_padded = np.concatenate([np.tile(ref_force[None], (pad, 1)), force], axis=0)
                T_padded = self.chunk_size
            else:
                pad = 0
                tactile_padded = tactile
                force_padded = force
                T_padded = T

            for start in range(0, T_padded - self.chunk_size + 1, self.stride):
                chunk = tactile_padded[start:start + self.chunk_size].copy()
                last_idx_padded = start + self.chunk_size - 1
                f = force_padded[last_idx_padded].copy()

                if self.apply_ref_tactile_subtraction:
                    chunk = chunk - ref_tactile[None]
                # Match the pretrain dataloader's tactile preprocessing
                # (xela_9dtact_dataloader_webdataset.py:600-606) so the chunk is
                # in-distribution for the alignscratch-pretrained encoder.
                if self.apply_tactile_abs:
                    chunk = np.abs(chunk)
                if self.apply_tactile_normalization:
                    chunk = chunk / (self.tactile_std[None] + 1e-6)
                if self.apply_tactile_clamp:
                    chunk = np.clip(chunk, self.tactile_clamp_min, self.tactile_clamp_max)
                if self.apply_ref_force_subtraction:
                    f = f - ref_force

                f = np.where(np.isfinite(f), f, 0.0).astype(np.float32)
                if self.force_clip_min is not None and self.force_clip_max is not None:
                    f[:3] = np.clip(f[:3], self.force_clip_min, self.force_clip_max)
                if self.apply_force_normalization:
                    f = ((f - self.force_mean) / (self.force_std + 1e-6)).astype(np.float32)

                out = {
                    "tactile": torch.from_numpy(chunk),
                    "6d_force": torch.tensor(f, dtype=torch.float32),
                    "probe": probe,
                    "mode": mode,
                }
                if self.load_sliding_labels:
                    last_idx_orig = last_idx_padded - pad
                    if lb_arr is not None and 0 <= last_idx_orig < lb_arr.shape[0]:
                        out["sliding_label"] = torch.tensor(int(lb_arr[last_idx_orig]), dtype=torch.long)
                    else:
                        out["sliding_label"] = torch.tensor(-1, dtype=torch.long)
                yield out

    def __len__(self):
        total = 0
        for path in self._paths:
            try:
                data = np.load(path, allow_pickle=True)
                T = int(data["tactile"].shape[0])
                total += max(0, (T - self.chunk_size) // self.stride + 1)
            except Exception:
                continue
        return total


def taxel_force_collate_fn(batch):
    tactile = torch.stack([item["tactile"] for item in batch])
    force_6d = torch.stack([item["6d_force"] for item in batch])
    probe = torch.tensor([item["probe"] for item in batch], dtype=torch.long)
    mode = [item["mode"] for item in batch]
    out = {"tactile": tactile, "6d_force": force_6d, "probe": probe, "mode": mode}
    if batch and "sliding_label" in batch[0]:
        out["sliding_label"] = torch.stack([item["sliding_label"] for item in batch])
    return out


def create_taxel_force_dataloader(
    modality: str,
    taxel_dim: int,
    data_root: Optional[str] = None,
    split: str = "train",
    batch_size: int = 32,
    chunk_size: int = 20,
    config_dir: str = "config/data",
    apply_ref_tactile_subtraction: bool = True,
    apply_ref_force_subtraction: bool = True,
    apply_force_normalization: bool = True,
    apply_tactile_abs: bool = True,
    apply_tactile_normalization: bool = True,
    apply_tactile_clamp: bool = True,
    tactile_clamp_min: float = -10.0,
    tactile_clamp_max: float = 10.0,
    force_clip_min: Optional[float] = -20.0,
    force_clip_max: Optional[float] = 20.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    timeout: int = 30,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    mode_filter: Optional[str] = None,
    stride: int = 1,
    labeled_data_root: Optional[str] = None,
    load_sliding_labels: Optional[bool] = None,
    strict_labeled: bool = False,
    **kwargs,
) -> DataLoader:
    dataset = TaxelForce4Probe50EachDataset(
        modality=modality,
        taxel_dim=taxel_dim,
        data_root=data_root,
        split=split,
        chunk_size=chunk_size,
        config_dir=config_dir,
        apply_ref_tactile_subtraction=apply_ref_tactile_subtraction,
        apply_ref_force_subtraction=apply_ref_force_subtraction,
        apply_force_normalization=apply_force_normalization,
        apply_tactile_abs=apply_tactile_abs,
        apply_tactile_normalization=apply_tactile_normalization,
        apply_tactile_clamp=apply_tactile_clamp,
        tactile_clamp_min=tactile_clamp_min,
        tactile_clamp_max=tactile_clamp_max,
        force_clip_min=force_clip_min,
        force_clip_max=force_clip_max,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        mode_filter=mode_filter,
        stride=stride,
        labeled_data_root=labeled_data_root,
        load_sliding_labels=load_sliding_labels,
        strict_labeled=strict_labeled,
    )
    prefetch = prefetch_factor if num_workers > 0 else None
    persistent = persistent_workers if num_workers > 0 else False
    worker_timeout = timeout if num_workers > 0 else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        timeout=worker_timeout,
        collate_fn=taxel_force_collate_fn,
    )


def load_force_stats_per_dim(
    modality: str, config_dir: str = "config/data", dims: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    force_path = Path(config_dir) / "force.yaml"
    if not force_path.exists():
        force_path = Path("config/data/force.yaml")
    if not force_path.exists():
        return np.zeros(dims), np.ones(dims)
    try:
        with open(force_path) as f:
            cfg = yaml.safe_load(f) or {}
        if modality not in cfg:
            return np.zeros(dims), np.ones(dims)
        m = cfg[modality]
        fm = np.array(m.get("force_mean", [0.0] * 6), dtype=np.float64)
        fs = np.array(m.get("force_std", [1.0] * 6), dtype=np.float64)
        if len(fm) < dims or len(fs) < dims:
            return np.zeros(dims), np.ones(dims)
        return fm[:dims].copy(), np.maximum(fs[:dims].copy(), 1e-9)
    except Exception:
        return np.zeros(dims), np.ones(dims)


# Taxel dimensions per modality
TAXEL_DIMS = {"tac02": 66, "xela": 72}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=True, choices=list(TAXEL_DIMS.keys()))
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--compute_force_stats", action="store_true")
    args = parser.parse_args()

    taxel_dim = TAXEL_DIMS[args.modality]

    if args.compute_force_stats:
        root = args.data_root or f"data/{args.modality}_force_4probe_50each/processed"
        compute_and_save_taxel_force_stats(root, args.modality)
        exit(0)

    loader = create_taxel_force_dataloader(
        modality=args.modality,
        taxel_dim=taxel_dim,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    print("Keys:", list(batch.keys()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")
    print(f"Expected tactile shape: ({args.batch_size}, 20, {taxel_dim})")
