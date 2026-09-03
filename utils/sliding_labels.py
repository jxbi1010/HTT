"""
Pipeline: friction μ series → Page CUSUM (C⁺) → per-frame sliding labels (incipient / gross).

Two label tracks (both computed by default in :func:`label_episode`):

1. **CUSUM** (``labels``): C⁺ thresholds + persistence + hysteresis — see :func:`generate_sliding_labels`.

2. **Sliding signature** (``labels_signature``): clip μ outliers, normal-force gating, and
   median-based sliding reference — see :func:`label_mostly_sliding_episode`.

3. **Bracket** (``labels_bracket``): first C⁺ > ``h_trigger`` through last contact
   (``abs(fz) > abs(fz_threshold)``), with incipient frames before start — see
   :func:`label_sliding_bracket`.

Label semantics — CUSUM:
  0 = no event (or below incipient threshold)
  1 = incipient slip (warning): C⁺ > h_min
  2 = gross sliding: C⁺ > h_max for ``persistence`` consecutive frames (overrides 1)

Label semantics — signature method:
  0 = no contact (fz < fz_threshold, same convention as input fz)
  1 = incipient / low-force contact
  2 = gross sliding vs sliding reference

All exported per-frame label arrays (CUSUM, signature, bracket) are **int32 in {0, 1, 2}** for every
frame (no ``-1`` sentinels). If a track is skipped (CLI), it falls back to the CUSUM track.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from utils.cusum import cusum_two_sided

EpisodePath = Union[str, Path]


def compute_mu_fz_series_from_episode(
    episode: EpisodePath,
    clip_lo: float = -20.0,
    clip_hi: float = 20.0,
    friction_mu_eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load one episode: per-frame μ (same as dataloader) and fz = normal force (dim 2) after
    ref subtract + clip. Single pass over the file.
    """
    from utils.force_friction import friction_coefficient_from_shear_normal

    episode = Path(episode)
    data = np.load(episode, allow_pickle=True)
    ref_force = data["ref_force"] if "ref_force" in data else np.zeros(6, dtype=np.float64)
    force = np.asarray(data["6d_force"], dtype=np.float64)
    T = force.shape[0]
    mus: List[float] = []
    fz_list: List[float] = []
    for t in range(T):
        f = force[t].copy() - ref_force
        f = np.where(np.isfinite(f), f, 0.0).astype(np.float32)
        f[:3] = np.clip(f[:3], clip_lo, clip_hi)
        mus.append(
            float(
                friction_coefficient_from_shear_normal(
                    float(f[0]), float(f[1]), float(f[2]), eps=friction_mu_eps
                )
            )
        )
        fz_list.append(float(f[2]))
    mus_arr = np.array(mus, dtype=np.float64)
    fz_arr = np.array(fz_list, dtype=np.float64)
    meta = {
        "probe": data.get("probe", None),
        "mode": str(data.get("mode", "")) if "mode" in data else None,
        "T": T,
    }
    return mus_arr, fz_arr, meta


def sanitize_sliding_class_labels(arr: np.ndarray) -> np.ndarray:
    """
    Force per-frame class ids into ``{0, 1, 2}`` (training expects no ``-1``).

    Maps negatives to ``0`` and values ``> 2`` to ``2``. Empty arrays unchanged.
    """
    if arr.size == 0:
        return np.asarray(arr, dtype=np.int32)
    x = np.asarray(arr, dtype=np.int64).ravel()
    x = np.clip(x, 0, 2)
    return x.astype(np.int32)


def label_mostly_sliding_episode(
    mus: np.ndarray,
    fz: np.ndarray,
    fz_threshold: float = 0.1,
    mu_clip_percentile: float = 95.0,
    gross_ratio: float = 0.4,
) -> np.ndarray:
    """
    Heuristic labels for mostly-sliding episodes: clip μ spikes, derive a sliding reference from
    the upper half of clean μ, gate on normal force fz (dim 2, physical after ref subtract + clip).

    0 = no contact (|fz| < fz_threshold)
    1 = incipient / low-force contact
    2 = gross sliding (clean_mus > gross_ratio * sliding_reference)

    Ensures a frame before a 2 transition cannot stay 0 (promotes to 1).
    """
    mus = np.asarray(mus, dtype=np.float64).ravel()
    fz = np.asarray(fz, dtype=np.float64).ravel()
    T = mus.size
    if fz.size != T:
        raise ValueError(f"mus length {T} != fz length {fz.size}")
    if T == 0:
        return np.array([], dtype=np.int32)

    mu_limit = float(np.percentile(mus, mu_clip_percentile))
    clean_mus = np.clip(mus, 0.0, mu_limit)

    med = float(np.median(clean_mus))
    upper = clean_mus[clean_mus > med]
    if upper.size > 0:
        sliding_reference = float(np.median(upper))
    else:
        sliding_reference = med
    if sliding_reference <= 1e-12:
        sliding_reference = max(float(np.mean(clean_mus)), 1e-12)

    labels = np.zeros(T, dtype=np.int32)
    threshold = gross_ratio * sliding_reference

    for t in range(T):
        if fz[t] < fz_threshold:
            labels[t] = 0
        elif clean_mus[t] > threshold:
            labels[t] = 2
        else:
            labels[t] = 1

    for t in range(1, T):
        if labels[t] == 2 and labels[t - 1] == 0:
            labels[t - 1] = 1

    return labels


def label_sliding_bracket(
    c_plus: np.ndarray,
    fz: np.ndarray,
    h_trigger: float = 5.0,
    fz_threshold: float = -1.5,
    incipient_window: int = 5,
) -> np.ndarray:
    """
    Label the interval from first C⁺ slip crossing to last contact frame.

    0 = idle / outside bracket
    1 = incipient (``incipient_window`` frames before first C⁺ > ``h_trigger``)
    2 = sliding (from first trigger through last contact: ``abs(fz) > abs(fz_threshold)``)

    ``fz_threshold`` is often negative in this dataset; contact uses ``abs(fz) > abs(fz_threshold)``.
    """
    c_plus = np.asarray(c_plus, dtype=np.float64).ravel()
    fz = np.asarray(fz, dtype=np.float64).ravel()
    T = c_plus.size
    if fz.size != T:
        raise ValueError(f"c_plus length {T} != fz length {fz.size}")
    labels = np.zeros(T, dtype=np.int32)
    if T == 0:
        return labels

    start_indices = np.where(c_plus > h_trigger)[0]
    if len(start_indices) == 0:
        return labels

    t_start = int(start_indices[0])
    abs_fz = np.abs(fz)
    thr = float(np.abs(fz_threshold))
    contact_indices = np.where(abs_fz > thr)[0]
    if len(contact_indices) == 0:
        return labels

    t_stop = int(contact_indices[-1])
    if t_stop > t_start:
        labels[t_start : t_stop + 1] = 2
        t_incipient = max(0, t_start - incipient_window)
        labels[t_incipient:t_start] = 1

    np.clip(labels, 0, 2, out=labels)
    return labels


def generate_sliding_labels(
    mus: np.ndarray,
    c_plus: np.ndarray,
    h_min: float = 1.0,
    h_max: float = 5.0,
    persistence: int = 3,
) -> np.ndarray:
    """
    mus: array of friction coefficients (per frame)
    c_plus: array of CUSUM upper statistics (same length)
    h_min: threshold to flag incipient slip (warning)
    h_max: threshold for gross sliding; must hold for ``persistence`` consecutive frames
    persistence: number of consecutive frames with C⁺ > h_max to set label 2

    Returns:
        labels: int array, shape (T,) with values in {0, 1, 2}.
    """
    mus = np.asarray(mus, dtype=np.float64).ravel()
    c_plus = np.asarray(c_plus, dtype=np.float64).ravel()
    T = mus.size
    if c_plus.size != T:
        raise ValueError(f"mus length {T} != c_plus length {c_plus.size}")
    if T == 0:
        return np.array([], dtype=np.int32)

    labels = np.zeros(T, dtype=np.int32)

    for t in range(T):
        if c_plus[t] > h_min:
            labels[t] = 1

        if persistence <= 0:
            continue
        if t >= persistence - 1:
            window = c_plus[t - persistence + 1 : t + 1]
            if window.size == persistence and np.all(window > h_max):
                labels[t] = 2

    mu_bar = float(np.mean(mus))
    threshold = mu_bar * 0.8
    for t in range(1, T):
        if labels[t - 1] == 2 and mus[t] > threshold:
            if labels[t] == 0:
                labels[t] = 2

    return labels


@dataclass
class EpisodeLabelingResult:
    """Outputs of :func:`label_episode`."""

    episode: str
    mus: np.ndarray
    fz: np.ndarray
    c_plus: np.ndarray
    c_minus: np.ndarray
    labels: np.ndarray
    labels_signature: np.ndarray
    labels_bracket: np.ndarray
    meta: Dict[str, Any]
    n0: int
    mu0: float
    sigma: float
    params: Dict[str, Any] = field(default_factory=dict)
    params_signature: Dict[str, Any] = field(default_factory=dict)
    params_bracket: Dict[str, Any] = field(default_factory=dict)


def label_episode(
    episode: EpisodePath,
    *,
    clip_lo: float = -20.0,
    clip_hi: float = 20.0,
    friction_mu_eps: float = 1e-3,
    baseline_frac: float = 0.25,
    cusum_k: float = 0.5,
    h_min: float = 1.0,
    h_max: float = 5.0,
    persistence: int = 3,
    fz_threshold: float = 0.1,
    mu_clip_percentile: float = 95.0,
    gross_ratio: float = 0.4,
    compute_signature: bool = True,
    h_trigger: float = 5.0,
    fz_threshold_bracket: float = -1.5,
    incipient_window: int = 5,
    compute_bracket: bool = True,
) -> EpisodeLabelingResult:
    """
    Full pipeline for one processed ``.npz`` episode: μ (+ fz) → CUSUM labels and optional
    sliding-signature labels (:func:`label_mostly_sliding_episode`).

    CUSUM baseline (μ₀, σ) is taken from the first ``max(8, int(T * baseline_frac))`` frames
    of the μ series, matching the interactive CUSUM demo.
    """
    episode = Path(episode)
    mus, fz, meta = compute_mu_fz_series_from_episode(
        episode,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        friction_mu_eps=friction_mu_eps,
    )
    T = int(meta["T"])
    if T == 0:
        empty = np.array([])
        return EpisodeLabelingResult(
            episode=str(episode),
            mus=mus,
            fz=empty,
            c_plus=empty,
            c_minus=empty,
            labels=np.array([], dtype=np.int32),
            labels_signature=np.array([], dtype=np.int32),
            labels_bracket=np.array([], dtype=np.int32),
            meta=meta,
            n0=0,
            mu0=0.0,
            sigma=1.0,
            params={
                "baseline_frac": baseline_frac,
                "cusum_k": cusum_k,
                "h_min": h_min,
                "h_max": h_max,
                "persistence": persistence,
            },
            params_signature={},
            params_bracket={},
        )

    n0 = max(8, int(T * baseline_frac))
    mu0 = float(np.mean(mus[:n0]))
    sigma = float(np.std(mus[:n0], ddof=1)) if n0 > 1 else 1.0
    if sigma <= 1e-12:
        sigma = 1.0

    c_plus, c_minus = cusum_two_sided(
        mus, mu0=mu0, sigma=sigma, k=cusum_k, standardized=True
    )
    labels = sanitize_sliding_class_labels(
        generate_sliding_labels(mus, c_plus, h_min=h_min, h_max=h_max, persistence=persistence)
    )

    params = {
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
        "friction_mu_eps": friction_mu_eps,
        "baseline_frac": baseline_frac,
        "cusum_k": cusum_k,
        "h_min": h_min,
        "h_max": h_max,
        "persistence": persistence,
    }

    params_signature = {
        "fz_threshold": fz_threshold,
        "mu_clip_percentile": mu_clip_percentile,
        "gross_ratio": gross_ratio,
    }
    if compute_signature:
        labels_signature = sanitize_sliding_class_labels(
            label_mostly_sliding_episode(
                mus,
                fz,
                fz_threshold=fz_threshold,
                mu_clip_percentile=mu_clip_percentile,
                gross_ratio=gross_ratio,
            )
        )
    else:
        labels_signature = labels.copy()

    if compute_bracket:
        labels_bracket = sanitize_sliding_class_labels(
            label_sliding_bracket(
                c_plus,
                fz,
                h_trigger=h_trigger,
                fz_threshold=fz_threshold_bracket,
                incipient_window=incipient_window,
            )
        )
        start_idx = np.where(c_plus > h_trigger)[0]
        abs_fz = np.abs(fz)
        thr_b = float(np.abs(fz_threshold_bracket))
        contact_idx = np.where(abs_fz > thr_b)[0]
        params_bracket = {
            "h_trigger": h_trigger,
            "fz_threshold": fz_threshold_bracket,
            "incipient_window": incipient_window,
            "t_start": int(start_idx[0]) if len(start_idx) else None,
            "t_stop": int(contact_idx[-1]) if len(contact_idx) else None,
        }
    else:
        labels_bracket = labels.copy()
        params_bracket = {}

    return EpisodeLabelingResult(
        episode=str(episode.resolve()),
        mus=mus,
        fz=fz,
        c_plus=c_plus,
        c_minus=c_minus,
        labels=labels,
        labels_signature=labels_signature,
        labels_bracket=labels_bracket,
        meta=meta,
        n0=n0,
        mu0=mu0,
        sigma=sigma,
        params=params,
        params_signature=params_signature,
        params_bracket=params_bracket,
    )


def iter_episode_paths(
    data_root: EpisodePath,
    *,
    subfolder_glob: str = "p*_sliding",
    extension: str = ".npz",
) -> Iterator[Path]:
    """Yield ``*.npz`` paths under ``data_root/<subfolder_glob>/*/`` (e.g. ``p1_sliding/*.npz``)."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    for sub in sorted(root.glob(subfolder_glob)):
        if not sub.is_dir():
            continue
        for p in sorted(sub.glob(f"*{extension}")):
            if p.is_file():
                yield p


def label_episodes(
    data_root: EpisodePath,
    *,
    subfolder_glob: str = "p*_sliding",
    **label_episode_kwargs: Any,
) -> List[EpisodeLabelingResult]:
    """
    Run :func:`label_episode` on every ``.npz`` under ``data_root`` matching ``subfolder_glob``.
    """
    results: List[EpisodeLabelingResult] = []
    for path in iter_episode_paths(data_root, subfolder_glob=subfolder_glob):
        results.append(label_episode(path, **label_episode_kwargs))
    return results


def summarize_bracket_labels_dataset(
    data_root: EpisodePath,
    *,
    subfolder_glob: str = "p*_sliding",
    **label_episode_kwargs: Any,
) -> Tuple[Dict[int, int], int, int, List[Tuple[str, Dict[int, int], int]]]:
    """
    Run bracket labeling on every sliding episode and aggregate label counts.

    Returns:
        total_counts: ``{0: n0, 1: n1, 2: n2}`` over all frames
        total_frames: sum of T over episodes
        n_episodes: number of ``.npz`` files processed
        per_episode: list of ``(episode_name, {0:..,1:..,2:..}, T)``
    """
    kw = dict(label_episode_kwargs)
    kw.setdefault("compute_signature", False)

    total_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    total_frames = 0
    per_episode: List[Tuple[str, Dict[int, int], int]] = []

    for path in iter_episode_paths(data_root, subfolder_glob=subfolder_glob):
        r = label_episode(path, **kw)
        lb = r.labels_bracket
        T = int(lb.size)
        if T == 0:
            per_episode.append((path.name, {0: 0, 1: 0, 2: 0}, 0))
            continue
        ep_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        for k in (0, 1, 2):
            c = int(np.sum(lb == k))
            ep_counts[k] = c
            total_counts[k] += c
        total_frames += T
        per_episode.append((path.name, ep_counts, T))

    n_episodes = len(per_episode)
    return total_counts, total_frames, n_episodes, per_episode


def save_labeled_episode_npz(
    result: EpisodeLabelingResult,
    out_path: EpisodePath,
    *,
    extra_keys: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save μ, CUSUM, labels, and small meta into a single ``.npz`` (does not copy raw images/forces).

    Keys (load with ``numpy.load``):

    - ``mus`` (T,), ``fz`` (T,), ``c_plus`` (T,), ``c_minus`` (T,)
    - ``sliding_labels`` (CUSUM), ``sliding_labels_signature``, ``sliding_labels_bracket`` (each ``(T,)`` int32 in ``{0,1,2}``)
    - ``n0``, ``mu0``, ``sigma`` (CUSUM baseline scalars)
    - ``episode_path`` (str array), ``labeling_meta`` (JSON string with params)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode": result.episode,
        "params": result.params,
        "params_signature": result.params_signature,
        "params_bracket": result.params_bracket,
    }
    payload: Dict[str, Any] = {
        "mus": result.mus,
        "fz": result.fz,
        "c_plus": result.c_plus,
        "c_minus": result.c_minus,
        "sliding_labels": result.labels,
        "sliding_labels_signature": result.labels_signature,
        "sliding_labels_bracket": result.labels_bracket,
        "episode_path": np.array(result.episode),
        "labeling_meta": np.array(json.dumps(meta)),
        "n0": np.int32(result.n0),
        "mu0": np.float64(result.mu0),
        "sigma": np.float64(result.sigma),
    }
    if extra_keys:
        payload.update(extra_keys)
    np.savez_compressed(out_path, **payload)


def export_all_sliding_episodes_to_dir(
    data_root: EpisodePath,
    save_dir: EpisodePath,
    *,
    subfolder_glob: str = "p*_sliding",
    **label_episode_kwargs: Any,
) -> List[Path]:
    """
    Label every sliding ``.npz`` under ``data_root`` and write one ``.labeled.npz`` per episode,
    preserving subfolders (e.g. ``save_dir/p1_sliding/foo.labeled.npz``).

    Returns paths written.
    """
    root = Path(data_root).resolve()
    base_out = Path(save_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for path in iter_episode_paths(root, subfolder_glob=subfolder_glob):
        r = label_episode(path, **label_episode_kwargs)
        rel = path.resolve().relative_to(root)
        out = base_out / rel.with_suffix(".labeled.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_labeled_episode_npz(r, out)
        written.append(out)
    return written


def compute_sliding_class_weights(
    labeled_data_root: EpisodePath,
    *,
    num_classes: int = 3,
    label_key: str = "sliding_labels_bracket",
    glob_pattern: str = "**/*.labeled.npz",
    eps: float = 1.0,
) -> np.ndarray:
    """
    Inverse-frequency class weights for the sliding bracket task.

    Scans every ``*.labeled.npz`` under ``labeled_data_root``, sums per-class
    frame counts, and returns ``w[c] = total / (num_classes * (count[c] + eps))``
    normalized so weights average to 1.

    The ``+eps`` smooths empty classes (the incipient class is sparse) and the
    final normalization keeps the per-batch CE magnitude in roughly the same
    range as unweighted CE so the existing learning rate transfers.

    Args:
        labeled_data_root: directory containing ``*.labeled.npz`` exports
        num_classes: number of classes (default 3 for static/incipient/gross)
        label_key: which label track to use; bracket is what training reads
        glob_pattern: rglob pattern for label files
        eps: additive smoothing on per-class counts

    Returns:
        ``np.ndarray`` of shape ``(num_classes,)``, dtype float32, mean ≈ 1.
    """
    root = Path(labeled_data_root)
    counts = np.zeros(num_classes, dtype=np.float64)
    n_files = 0
    for f in root.rglob(glob_pattern):
        d = np.load(f, allow_pickle=True)
        if label_key not in d.files:
            continue
        lb = np.asarray(d[label_key]).reshape(-1).astype(np.int64)
        valid = (lb >= 0) & (lb < num_classes)
        if not valid.any():
            continue
        for c in range(num_classes):
            counts[c] += int((lb[valid] == c).sum())
        n_files += 1
    if n_files == 0:
        raise FileNotFoundError(
            f"No '{glob_pattern}' files with key '{label_key}' under {root}"
        )
    total = counts.sum()
    if total <= 0:
        raise ValueError(f"All-zero class counts under {root} (label_key={label_key})")
    w = total / (num_classes * (counts + eps))
    w = w / w.mean()
    return w.astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Label sliding episodes: μ → CUSUM → incipient (1) / gross (2) labels."
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Single .npz episode (if set, --data-root batch mode is skipped)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/gsmini_force_4probe_50each/processed",
        help="Root with p1_sliding/, p2_sliding/, ...",
    )
    parser.add_argument(
        "--subfolder-glob",
        type=str,
        default="p*_sliding",
        help="Only subfolders matching this glob are scanned",
    )
    parser.add_argument("--friction-mu-eps", type=float, default=1e-3, dest="friction_mu_eps")
    parser.add_argument("--baseline-frac", type=float, default=0.25)
    parser.add_argument("--cusum-k", type=float, default=0.5, dest="cusum_k")
    parser.add_argument("--h-min", type=float, default=1.0, dest="h_min")
    parser.add_argument("--h-max", type=float, default=5.0, dest="h_max")
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument(
        "--fz-threshold",
        type=float,
        default=0.1,
        help="Signature method: no contact if fz < this (normal force, physical)",
    )
    parser.add_argument(
        "--mu-clip-percentile",
        type=float,
        default=95.0,
        dest="mu_clip_percentile",
        help="Signature method: clip μ at this percentile before reference",
    )
    parser.add_argument(
        "--gross-ratio",
        type=float,
        default=0.4,
        dest="gross_ratio",
        help="Signature method: gross sliding if clean_mus > ratio * sliding_reference",
    )
    parser.add_argument(
        "--no-signature",
        action="store_true",
        help="Skip sliding-signature labels (only CUSUM)",
    )
    parser.add_argument(
        "--h-trigger",
        type=float,
        default=5.0,
        dest="h_trigger",
        help="Bracket method: first frame with C+ > this starts sliding segment",
    )
    parser.add_argument(
        "--fz-threshold-bracket",
        type=float,
        default=-1.5,
        dest="fz_threshold_bracket",
        help="Bracket method: contact if abs(fz) > abs(this); often negative in this dataset",
    )
    parser.add_argument(
        "--incipient-window",
        type=int,
        default=5,
        dest="incipient_window",
        help="Bracket method: label 1 for this many frames before C+ trigger",
    )
    parser.add_argument(
        "--no-bracket",
        action="store_true",
        help="Skip bracket labels (sliding interval from C+ trigger to last contact)",
    )
    parser.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help="If set with --episode, write labels to this .npz path",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set with --data-root, save one .labeled.npz per episode, mirroring p1_sliding/... subfolders",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Same as batch --data-root but uses --save-dir default if not set (sliding_labeled next to processed)",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print t, mu, fz, C_plus, bracket + other labels per frame",
    )
    parser.add_argument(
        "--summarize-bracket",
        action="store_true",
        help="Process all sliding episodes under --data-root and print global bracket label ratios",
    )
    args = parser.parse_args()

    common = dict(
        friction_mu_eps=args.friction_mu_eps,
        baseline_frac=args.baseline_frac,
        cusum_k=args.cusum_k,
        h_min=args.h_min,
        h_max=args.h_max,
        persistence=args.persistence,
        fz_threshold=args.fz_threshold,
        mu_clip_percentile=args.mu_clip_percentile,
        gross_ratio=args.gross_ratio,
        compute_signature=not args.no_signature,
        h_trigger=args.h_trigger,
        fz_threshold_bracket=args.fz_threshold_bracket,
        incipient_window=args.incipient_window,
        compute_bracket=not args.no_bracket,
    )

    default_export_dir = None
    if args.export_all and not args.save_dir:
        default_export_dir = str(Path(args.data_root).resolve().parent / "sliding_labeled")
        args.save_dir = default_export_dir

    if args.summarize_bracket:
        if args.no_bracket:
            raise SystemExit("--summarize-bracket requires bracket labels (do not use --no-bracket)")
        root = Path(args.data_root)
        total_counts, total_frames, n_episodes, per_episode = summarize_bracket_labels_dataset(
            root,
            subfolder_glob=args.subfolder_glob,
            **common,
        )
        print("Bracket labeling (label_sliding_bracket) — dataset summary")
        print(f"  data_root: {root.resolve()}")
        print(f"  subfolder_glob: {args.subfolder_glob}")
        print(f"  episodes (.npz): {n_episodes}")
        print(f"  total frames: {total_frames}")
        print(f"  params: h_trigger={args.h_trigger}, fz_threshold_bracket={args.fz_threshold_bracket}, incipient_window={args.incipient_window}")
        if total_frames == 0:
            raise SystemExit("No frames found.")
        print("  Global counts: 0=%d, 1=%d, 2=%d" % (total_counts[0], total_counts[1], total_counts[2]))
        print(
            "  Global ratios:  0=%.4f, 1=%.4f, 2=%.4f"
            % (
                total_counts[0] / total_frames,
                total_counts[1] / total_frames,
                total_counts[2] / total_frames,
            )
        )
        raise SystemExit(0)

    if args.episode:
        r = label_episode(args.episode, **common)
        header = {
            "episode": r.episode,
            "T": r.meta["T"],
            "n0": r.n0,
            "mu0": r.mu0,
            "sigma": r.sigma,
            "bracket": r.params_bracket,
        }
        print(json.dumps(header, indent=2))
        counts = {int(k): int(v) for k, v in zip(*np.unique(r.labels, return_counts=True))}
        print("CUSUM label counts (0=none, 1=incipient, 2=gross):", counts)
        if r.labels_signature.size and r.labels_signature.min() >= 0:
            sig_counts = {
                int(k): int(v) for k, v in zip(*np.unique(r.labels_signature, return_counts=True))
            }
            print("Signature label counts (0=no contact, 1=incipient, 2=gross):", sig_counts)
        if r.labels_bracket.size and r.labels_bracket.min() >= 0:
            br_counts = {
                int(k): int(v) for k, v in zip(*np.unique(r.labels_bracket, return_counts=True))
            }
            print("Bracket label counts (0=idle, 1=incipient, 2=sliding):", br_counts)
        if args.save_npz:
            save_labeled_episode_npz(r, args.save_npz)
            print("saved:", args.save_npz)
        if args.print_table:
            print(
                "t\tmu\t\tfz\t\tC_plus\t\tC_minus\t\tlabel_bracket\tlabel_cusum\tlabel_sig"
            )
            for t in range(len(r.mus)):
                sig = (
                    int(r.labels_signature[t])
                    if r.labels_signature.size and r.labels_signature.min() >= 0
                    else -1
                )
                br = (
                    int(r.labels_bracket[t])
                    if r.labels_bracket.size and r.labels_bracket.min() >= 0
                    else -1
                )
                print(
                    f"{t}\t{r.mus[t]:.6f}\t{r.fz[t]:.6f}\t{r.c_plus[t]:.6f}\t{r.c_minus[t]:.6f}\t{br}\t{r.labels[t]}\t{sig}"
                )
    else:
        root = Path(args.data_root)
        results = label_episodes(root, subfolder_glob=args.subfolder_glob, **common)
        print(f"Labeled {len(results)} episodes under {root}")
        for r in results:
            counts_arr = np.unique(r.labels, return_counts=True)
            counts = {int(k): int(v) for k, v in zip(*counts_arr)}
            rel = Path(r.episode).name
            if r.labels_signature.size and r.labels_signature.min() >= 0:
                sig_c = {
                    int(k): int(v) for k, v in zip(*np.unique(r.labels_signature, return_counts=True))
                }
            else:
                sig_c = {}
            if r.labels_bracket.size and r.labels_bracket.min() >= 0:
                br_c = {
                    int(k): int(v) for k, v in zip(*np.unique(r.labels_bracket, return_counts=True))
                }
            else:
                br_c = {}
            print(
                f"  {rel}: T={r.meta['T']} CUSUM={counts} sig={sig_c} bracket={br_c}"
            )
            if args.save_dir:
                ep = Path(r.episode).resolve()
                root_res = root.resolve()
                try:
                    rel = ep.relative_to(root_res)
                except ValueError:
                    rel = Path(ep.name)
                out = Path(args.save_dir) / rel.with_suffix(".labeled.npz")
                out.parent.mkdir(parents=True, exist_ok=True)
                save_labeled_episode_npz(r, out)
        if args.save_dir:
            print(f"wrote labeled npz files under {Path(args.save_dir).resolve()}")
            if args.export_all and default_export_dir:
                print(f"  (default export dir from --export-all)")
