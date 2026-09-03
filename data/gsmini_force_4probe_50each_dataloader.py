#!/usr/bin/env python3
"""
Dataloader for GS Mini tactile + 6D force data (4-probe, 50 each) from processed .npz episodes.

Each .npz contains: ref_frame (224,224,C), ref_force (6,), tactile_img (T,224,224,C), 6d_force (T,6),
probe (int 1-4), mode (str "static" or "sliding").
Force convention: first 3 dims are [shear_x, shear_y, normal] (dims 0,1 = shear, dim 2 = normal).
Yields per-frame samples: tactile image (ref-frame subtracted, repeated to 2 on time dim) and 6d force (ref-force subtracted).
Batch: tactile_img [B, 2, 3, 224, 224], 6d_force [B, 6], probe [B], mode [B], and optionally friction_mu [B]
when loading sliding data: μ = ‖shear‖/(|n|+eps) on **ref-subtracted, clipped** shear/normal (no z-score);
6d_force is still z-score normalized when enabled.

For ``mode_filter=='sliding'``, optional precomputed arrays from ``sliding_labeled/.../*.labeled.npz``
(see ``utils.sliding_labels`` export): ``sliding_label`` (bracket 0/1/2, or -1 if missing),
``labeled_mus``, ``labeled_c_plus`` (NaN when file missing and not strict).
Force dims 0,1,2 are clipped to [-20, 20] by default.
By default applies reference frame subtraction and reference force subtraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
import math
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from utils.force_friction import friction_coefficient_from_shear_normal


def compute_and_save_gsmini_force_stats(
    data_root: str,
    config_dir: str = "config/data",
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force mean, std, min, max from raw 6d_force in processed GS Mini .npz episodes.
    Overwrites config/data/force.yaml with only the gsmini section (no merge).
    Load from this file on subsequent runs; call this only when stats need to be recomputed.

    Args:
        data_root: Path to processed dir (flat *.npz or p*_*/*.npz subfolders).
        config_dir: Directory for output (default: config/data).
        output_path: Override path to save (default: config_dir/force.yaml).

    Returns:
        (force_mean, force_std, force_min, force_max) each shape (6,).
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")

    # Discover all .npz (works for flat or p*_*/*.npz structure)
    episode_paths = sorted(data_root.rglob("*.npz"))
    episode_paths = [p for p in episode_paths if p.name.endswith(".npz")]
    if not episode_paths:
        raise ValueError(f"No .npz files found in {data_root}")

    all_force = []
    for path in episode_paths:
        try:
            data = np.load(path, allow_pickle=True)
            force = np.asarray(data["6d_force"], dtype=np.float64)
        except Exception:
            continue
        if force.shape[0] == 0:
            continue
        force = np.where(np.isfinite(force), force, 0.0)
        all_force.append(force)
    if not all_force:
        raise ValueError("No valid force data found in any episode")

    all_force = np.concatenate(all_force, axis=0)
    force_mean = np.mean(all_force, axis=0).astype(np.float64)
    force_std = np.std(all_force, axis=0).astype(np.float64)
    force_std = np.where(force_std > 1e-8, force_std, 1.0)
    force_min = np.min(all_force, axis=0).astype(np.float64)
    force_max = np.max(all_force, axis=0).astype(np.float64)

    out_path = Path(output_path) if output_path else Path(config_dir) / "force.yaml"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite with only gsmini section (no merge)
    cfg = {
        "gsmini": {
            "force_mean": force_mean.tolist(),
            "force_std": force_std.tolist(),
            "force_min": force_min.tolist(),
            "force_max": force_max.tolist(),
        }
    }

    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Computed gsmini force stats from {len(episode_paths)} episodes, saved to {out_path}")
    return force_mean, force_std, force_min, force_max

class GsminiForce4Probe50EachDataset(IterableDataset):
    """
    Iterable dataset over processed GS Mini + force .npz episodes (4-probe, 50 each).
    Yields per-frame: tactile image (2, 3, 224, 224), 6d_force (6,), probe, mode with ref subtraction.
    """

    MODALITY = "gsmini"

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = "train",
        config_dir: str = "config/data",
        apply_background_subtraction: bool = True,
        apply_image_normalization: bool = True,
        apply_ref_force_subtraction: bool = True,
        apply_force_normalization: bool = True,
        force_clip_min: Optional[float] = -20.0,
        force_clip_max: Optional[float] = 20.0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        mode_filter: Optional[str] = None,
        compute_friction_mu: Optional[bool] = None,
        friction_mu_eps: float = 1e-3,
        labeled_data_root: Optional[str] = None,
        load_sliding_labels: Optional[bool] = None,
        strict_labeled: bool = False,
        modality: Optional[str] = None,
    ):
        """
        Args:
            data_root: Path to processed dir (contains p1_static/, p1_sliding/, etc. with *.npz).
                       If None, uses data/gsmini_force_4probe_50each/processed
            split: 'train', 'val', or 'test'
            config_dir: Directory for gsmini.yaml (image norm) and force.yaml (force norm)
            apply_background_subtraction: Subtract per-episode ref_frame from tactile (default True)
            apply_ref_force_subtraction: Subtract per-episode ref_force from 6d_force (default True)
            apply_force_normalization: Normalize force with (f - mean) / std from force.yaml (default True)
            force_clip_min: Clip force (dims 0,1,2: shear_x, shear_y, normal) to this lower bound (default -20). None to disable.
            force_clip_max: Clip force (dims 0,1,2) to this upper bound (default 20). None to disable.
            train_ratio: Fraction of episodes for train (rest split into val/test)
            val_ratio: Fraction of episodes for val (test = 1 - train_ratio - val_ratio)
            seed: Random seed for deterministic train/val/test split
            mode_filter: If set, use only episodes of this mode. "static" = only p*_static/,
                        "sliding" = only p*_sliding/. None = use all.
            compute_friction_mu: If True, each sample includes ``friction_mu``: ‖shear‖/(|n|+eps) on
                ref-subtracted, clipped **physical** shear/normal (before z-score). If None, defaults
                to True when ``mode_filter == 'sliding'``.
            friction_mu_eps: Added to |normal| (physical units after clip) in the ``friction_mu`` denominator.
            labeled_data_root: Root for exported ``*.labeled.npz`` (default: ``<parent of data_root>/sliding_labeled``).
            load_sliding_labels: If True, load bracket labels + ``mus``/``c_plus`` from the matching ``.labeled.npz``.
                If None, defaults to True when ``mode_filter == 'sliding'``.
            strict_labeled: If True, skip an episode when the expected ``.labeled.npz`` is missing or length-mismatched.
        """
        if modality is not None:
            self.MODALITY = modality
        self.split = split
        self.config_dir = Path(config_dir)
        self.apply_background_subtraction = apply_background_subtraction
        self.apply_image_normalization = apply_image_normalization
        self.apply_ref_force_subtraction = apply_ref_force_subtraction
        self.apply_force_normalization = apply_force_normalization
        self.force_clip_min = force_clip_min
        self.force_clip_max = force_clip_max

        if data_root is None:
            data_root = f"data/{self.MODALITY}_force_4probe_50each/processed"
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")

        self._load_normalization_config()
        if self.mean is not None and self.std is not None:
            self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(1, 3, 1, 1)
            self.std_tensor = torch.tensor(self.std, dtype=torch.float32).view(1, 3, 1, 1)
        else:
            self.mean_tensor = None
            self.std_tensor = None
        self.inv_255 = 1.0 / 255.0

        self._load_force_stats()

        # Discover episodes in subfolders (p1_static, p1_sliding, p2_static, etc.)
        all_paths = sorted(self.data_root.glob("p*_*/*.npz"))
        if mode_filter is not None:
            suffix = f"_{mode_filter}"
            self.episode_paths = [p for p in all_paths if p.parent.name.endswith(suffix)]
            if not self.episode_paths:
                raise ValueError(
                    f"No .npz files found for mode_filter='{mode_filter}' in {self.data_root} "
                    f"(expect subfolders p*_{mode_filter}/)"
                )
        else:
            self.episode_paths = all_paths
        if not self.episode_paths:
            raise ValueError(f"No .npz files found in {self.data_root} (expect p1_static/, p1_sliding/, etc.)")

        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.episode_paths))
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if split == "train":
            self._paths = [self.episode_paths[i] for i in indices[:n_train]]
        elif split == "val":
            self._paths = [self.episode_paths[i] for i in indices[n_train : n_train + n_val]]
        else:
            self._paths = [self.episode_paths[i] for i in indices[n_train + n_val :]]

        if not self._paths:
            raise ValueError(f"No episodes in split '{split}' (total {n} episodes)")

        self._total_length = None
        self._shuffle_seed = seed
        self._epoch = 0

        if compute_friction_mu is None:
            compute_friction_mu = mode_filter == "sliding"
        self.compute_friction_mu = compute_friction_mu
        self.friction_mu_eps = float(friction_mu_eps)

        if load_sliding_labels is None:
            load_sliding_labels = mode_filter == "sliding"
        self.load_sliding_labels = load_sliding_labels
        self.strict_labeled = strict_labeled
        if labeled_data_root is None:
            self.labeled_data_root = (self.data_root.parent / "sliding_labeled").resolve()
        else:
            self.labeled_data_root = Path(labeled_data_root).resolve()

    def _labeled_npz_path(self, episode_npz: Path) -> Path:
        """Map ``.../processed/p1_sliding/foo.npz`` → ``.../sliding_labeled/p1_sliding/foo.labeled.npz``."""
        rel = episode_npz.resolve().relative_to(self.data_root.resolve())
        return self.labeled_data_root / rel.with_suffix(".labeled.npz")

    def _load_normalization_config(self):
        """Load image mean/std from {modality}.yaml."""
        self.mean = np.array([0.0, 0.0, 0.0])
        self.std = np.array([1.0, 1.0, 1.0])
        cfg_path = self.config_dir / f"{self.MODALITY}.yaml"
        if not cfg_path.exists():
            cfg_path = Path(f"config/data/{self.MODALITY}.yaml")
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                if "image_mean_per_channel" in cfg and "image_std_per_channel" in cfg:
                    self.mean = np.array(cfg["image_mean_per_channel"], dtype=np.float32)
                    self.std = np.array(cfg["image_std_per_channel"], dtype=np.float32)
                elif "image_mean" in cfg and "image_std" in cfg:
                    m, s = float(cfg["image_mean"]), float(cfg["image_std"])
                    self.mean = np.array([m, m, m])
                    self.std = np.array([s, s, s])
            except Exception:
                pass

    def _load_force_stats(self):
        """Load 6D force mean/std from force.yaml (gsmini modality)."""
        force_path = self.config_dir / "force.yaml"
        if not force_path.exists():
            force_path = Path("config/data/force.yaml")
        self.force_mean = np.zeros(6)
        self.force_std = np.ones(6)
        if force_path.exists():
            try:
                with open(force_path) as f:
                    cfg = yaml.safe_load(f) or {}
                if self.MODALITY in cfg:
                    m = cfg[self.MODALITY]
                    if "force_mean" in m and isinstance(m["force_mean"], list):
                        self.force_mean = np.array(m["force_mean"], dtype=np.float64)
                    elif "force_mean" in m:
                        self.force_mean = np.full(6, float(m["force_mean"]))
                    if "force_std" in m and isinstance(m["force_std"], list):
                        self.force_std = np.array(m["force_std"], dtype=np.float64)
                    elif "force_std" in m:
                        self.force_std = np.full(6, float(m["force_std"]))
            except Exception:
                pass

    def _process_image(self, image: np.ndarray, bg_frame: Optional[np.ndarray]) -> torch.Tensor:
        """[H,W,C] uint8 -> [2, C, H, W] float tensor, normalized and optionally bg-subtracted."""
        x = torch.from_numpy(image).float()
        x = x.permute(2, 0, 1).contiguous()
        x = x * self.inv_255
        x = x.unsqueeze(0)
        if self.apply_background_subtraction and bg_frame is not None:
            bg = torch.from_numpy(bg_frame).float()
            bg = bg.permute(2, 0, 1).unsqueeze(0) * self.inv_255
            x = x - bg
        if self.apply_image_normalization and self.std_tensor is not None:
            x = x / (self.std_tensor + 1e-6)
        x = x.repeat(2, 1, 1, 1)
        return x

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffle. Call before each epoch so all workers use the same permutation."""
        self._epoch = epoch

    def __iter__(self):
        # Deterministic shuffle per epoch so all workers see the same path order (no duplicates)
        rng = np.random.RandomState(self._shuffle_seed + self._epoch)
        order = rng.permutation(len(self._paths))
        paths_to_use = [self._paths[i] for i in order]

        # With num_workers > 0, each worker must iterate only its shard to avoid duplicates
        worker_info = get_worker_info()
        if worker_info is not None:
            per_worker = int(math.ceil(len(paths_to_use) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(paths_to_use))
            paths_to_use = paths_to_use[start:end]

        for path in paths_to_use:
            try:
                data = np.load(path, allow_pickle=True)
                ref_frame = data["ref_frame"] if "ref_frame" in data else data["bg_frame"]
                ref_force = data["ref_force"] if "ref_force" in data else np.zeros(6, dtype=np.float64)
                tactile = data["tactile_img"]
                force = np.asarray(data["6d_force"], dtype=np.float64)
                probe = int(data["probe"]) if "probe" in data else 0
                mode_raw = data["mode"] if "mode" in data else "unknown"
                mode = str(mode_raw) if hasattr(mode_raw, "item") else str(mode_raw)
            except Exception:
                continue
            T = tactile.shape[0]
            if T == 0 or force.shape[0] != T:
                continue

            lb_arr = mu_arr = cp_arr = None
            if self.load_sliding_labels:
                lp = self._labeled_npz_path(path)
                if lp.exists():
                    try:
                        L = np.load(lp, allow_pickle=True)
                        lb_arr = np.asarray(L["sliding_labels_bracket"], dtype=np.int64).reshape(-1)
                        mu_arr = np.asarray(L["mus"], dtype=np.float32).reshape(-1)
                        cp_arr = np.asarray(L["c_plus"], dtype=np.float32).reshape(-1)
                        if lb_arr.shape[0] != T or mu_arr.shape[0] != T or cp_arr.shape[0] != T:
                            lb_arr = mu_arr = cp_arr = None
                    except Exception:
                        lb_arr = mu_arr = cp_arr = None
                if self.strict_labeled and lb_arr is None:
                    continue

            for t in range(T):
                # Tactile: ref-frame subtraction, then repeat to (2, 3, 224, 224)
                tactile_t = self._process_image(tactile[t], ref_frame)
                # Force: ref-force subtraction
                f = force[t].copy()
                if self.apply_ref_force_subtraction:
                    f = f - ref_force
                f = np.where(np.isfinite(f), f, 0.0).astype(np.float32)
                # Clip normal and shear (dims 0, 1, 2) to [force_clip_min, force_clip_max]
                if self.force_clip_min is not None and self.force_clip_max is not None:
                    f[:3] = np.clip(f[:3], float(self.force_clip_min), float(self.force_clip_max))
                # μ from physical shear/normal after ref subtract + clip (not z-score normalized)
                friction_mu_tensor = None
                if self.compute_friction_mu:
                    mu = friction_coefficient_from_shear_normal(
                        float(f[0]),
                        float(f[1]),
                        float(f[2]),
                        eps=self.friction_mu_eps,
                    )
                    friction_mu_tensor = torch.tensor(mu, dtype=torch.float32)
                # Z-score normalize for model targets / training
                if self.apply_force_normalization:
                    f = (f - self.force_mean) / (self.force_std + 1e-6)
                out = {
                    "tactile_img": tactile_t,
                    "6d_force": torch.tensor(f, dtype=torch.float32),
                    "probe": probe,
                    "mode": mode,
                }
                if friction_mu_tensor is not None:
                    out["friction_mu"] = friction_mu_tensor
                if self.load_sliding_labels:
                    if lb_arr is not None:
                        out["sliding_label"] = torch.tensor(int(lb_arr[t]), dtype=torch.long)
                        out["labeled_mus"] = torch.tensor(float(mu_arr[t]), dtype=torch.float32)
                        out["labeled_c_plus"] = torch.tensor(float(cp_arr[t]), dtype=torch.float32)
                    else:
                        out["sliding_label"] = torch.tensor(-1, dtype=torch.long)
                        out["labeled_mus"] = torch.tensor(float("nan"), dtype=torch.float32)
                        out["labeled_c_plus"] = torch.tensor(float("nan"), dtype=torch.float32)
                yield out

    def __len__(self):
        if self._total_length is not None:
            return self._total_length
        total = 0
        for path in self._paths:
            try:
                data = np.load(path, allow_pickle=True)
                T = data["tactile_img"].shape[0]
                total += T
            except Exception:
                continue
        self._total_length = total
        return total


def force_4probe_collate_fn(batch):
    """Collate: batch of tactile_img (2,3,224,224), 6d_force (6,), probe, mode -> stacked tensors."""
    tactile_img = torch.stack([item["tactile_img"] for item in batch])
    force_6d = torch.stack([item["6d_force"] for item in batch])
    probe = torch.tensor([item["probe"] for item in batch], dtype=torch.long)
    mode = [item["mode"] for item in batch]
    out = {
        "tactile_img": tactile_img,
        "6d_force": force_6d,
        "probe": probe,
        "mode": mode,
    }
    if batch and "friction_mu" in batch[0]:
        out["friction_mu"] = torch.stack([item["friction_mu"] for item in batch])
    if batch and "sliding_label" in batch[0]:
        out["sliding_label"] = torch.stack([item["sliding_label"] for item in batch])
        out["labeled_mus"] = torch.stack([item["labeled_mus"] for item in batch])
        out["labeled_c_plus"] = torch.stack([item["labeled_c_plus"] for item in batch])
    return out


def create_force_4probe_50each_dataloader(
    data_root: Optional[str] = None,
    modality: str = "gsmini",
    batch_size: int = 32,
    split: str = "train",
    config_dir: str = "config/data",
    apply_background_subtraction: bool = True,
    apply_image_normalization: bool = True,
    apply_ref_force_subtraction: bool = True,
    apply_force_normalization: bool = True,
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
    compute_friction_mu: Optional[bool] = None,
    friction_mu_eps: float = 1e-3,
    labeled_data_root: Optional[str] = None,
    load_sliding_labels: Optional[bool] = None,
    strict_labeled: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for GS Mini 4-probe 50-each force data (.npz episodes).
    Batch keys: tactile_img (B, 2, 3, 224, 224), 6d_force (B, 6), probe (B,), mode (list of B strs).
    When ``compute_friction_mu`` is True, or when it is omitted and ``mode_filter`` is ``sliding``,
    batches also include friction_mu (B,): ‖shear‖/(|n|+eps) on physical shear/normal after ref subtract
    and clip (not z-score); 6d_force remains normalized when enabled.
    When ``load_sliding_labels`` is True (default when ``mode_filter`` is ``sliding``), batches include
    ``sliding_label`` (B,) int64 (bracket 0/1/2, or -1 if no ``.labeled.npz``), ``labeled_mus`` and
    ``labeled_c_plus`` (B,) float32 (NaN when labels missing).
    Force (dims 0,1 = shear, dim 2 = normal) are clipped to [force_clip_min, force_clip_max] (default [-20, 20]).
    mode_filter: "static" or "sliding" to use only that mode; None for all.
    Ignores other kwargs (e.g. min_force, max_force) for compatibility.
    """
    dataset = GsminiForce4Probe50EachDataset(
        data_root=data_root,
        split=split,
        config_dir=config_dir,
        apply_background_subtraction=apply_background_subtraction,
        apply_image_normalization=apply_image_normalization,
        apply_ref_force_subtraction=apply_ref_force_subtraction,
        apply_force_normalization=apply_force_normalization,
        force_clip_min=force_clip_min,
        force_clip_max=force_clip_max,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        mode_filter=mode_filter,
        compute_friction_mu=compute_friction_mu,
        friction_mu_eps=friction_mu_eps,
        labeled_data_root=labeled_data_root,
        load_sliding_labels=load_sliding_labels,
        strict_labeled=strict_labeled,
        modality=modality,
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
        collate_fn=force_4probe_collate_fn,
    )


def load_force_stats(modality: str, config_dir: str = "config/data") -> Tuple[float, float, float, float]:
    """
    Same interface as force_dataloader_webdataset.load_force_stats.
    Returns scalar stats (L2 of 6D mean/std) for compatibility.
    """
    if modality not in (GsminiForce4Probe50EachDataset.MODALITY, "9dtact"):
        return (0.0, 1.0, 0.0, 1.0)
    force_path = Path(config_dir) / "force.yaml"
    if not force_path.exists():
        force_path = Path("config/data/force.yaml")
    if not force_path.exists():
        return (0.0, 1.0, 0.0, 1.0)
    try:
        with open(force_path) as f:
            cfg = yaml.safe_load(f) or {}
        if modality not in cfg:
            return (0.0, 1.0, 0.0, 1.0)
        m = cfg[modality]
        fm = m.get("force_mean", 0.0)
        fs = m.get("force_std", 1.0)
        if isinstance(fm, (list, tuple)):
            fm = float(np.linalg.norm(fm))
        if isinstance(fs, (list, tuple)):
            fs = float(np.linalg.norm(fs)) or 1.0
        return (float(fm), float(fs), float(fm), float(fs))
    except Exception:
        pass
    return (0.0, 1.0, 0.0, 1.0)


def load_force_stats_per_dim(
    modality: str, config_dir: str = "config/data", dims: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-dimension force mean and std for denormalizing MAE/RMSE to raw units (N).
    Returns (mean, std) as shape (dims,) for first dims (shear_x, shear_y, normal).
    """
    if modality not in (GsminiForce4Probe50EachDataset.MODALITY, "9dtact"):
        return np.zeros(dims), np.ones(dims)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--compute_force_stats", action="store_true", help="Compute and save gsmini force stats from data_root to config/data/force.yaml")
    args = parser.parse_args()

    if args.compute_force_stats:
        data_root = args.data_root or "data/gsmini_force_4probe_50each/processed"
        compute_and_save_gsmini_force_stats(data_root)
        exit(0)
    loader = create_force_4probe_50each_dataloader(
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
        elif isinstance(v, list):
            print(f"  {k}: list of {len(v)} strs")
        else:
            print(f"  {k}: {type(v).__name__}")
    assert batch["tactile_img"].shape == (args.batch_size, 2, 3, 224, 224), batch["tactile_img"].shape
    assert batch["6d_force"].shape == (args.batch_size, 6), batch["6d_force"].shape
    assert batch["probe"].shape == (args.batch_size,), batch["probe"].shape
    assert len(batch["mode"]) == args.batch_size, len(batch["mode"])
