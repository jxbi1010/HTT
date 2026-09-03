"""
CUSUM (Cumulative Sum) control scheme for detecting shifts in the process mean (Page, 1954).

Standard two-sided standardized CUSUM:
  z_t = (x_t - μ₀) / σ
  C⁺_t = max(0, C⁺_{t-1} + z_t - k)
  C⁻_t = max(0, C⁻_{t-1} - z_t - k)

where μ₀ is the in-control mean, σ is the within-series scale, and k is the slack (allowance),
often k = 0.5σ in original units or k = 0.5 when z is N(0,1) under control.

A mean increase pushes C⁺ up; a decrease pushes C⁻ up. Threshold h (not computed here) is
compared to C⁺/C⁻ for signaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

Array = Union[np.ndarray, list]


def cusum_two_sided(
    x: Array,
    mu0: Optional[float] = None,
    sigma: Optional[float] = None,
    k: float = 0.5,
    standardized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-sided Page CUSUM for each time index.

    Args:
        x: 1D sequence of length T.
        mu0: Target / in-control mean. If None, uses mean of ``x`` (Phase-I style baseline).
        sigma: Scale (e.g. std). If None, uses std of ``x`` with ddof=1, or 1.0 if degenerate.
        k: Slack parameter. If ``standardized`` is True, ``k`` is in units of z (default 0.5).
           If False, ``k`` is subtracted in the same units as ``x - mu0`` (typical: k = 0.5 * sigma).
        standardized: If True, use z_t = (x_t - mu0) / sigma before CUSUM; if False, use x_t - mu0
            and subtract ``k`` directly (set k = 0.5 * sigma for the usual rule).

    Returns:
        c_plus: shape (T,), upper CUSUM at each time step.
        c_minus: shape (T,), lower CUSUM at each time step.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = x.size
    if T == 0:
        return np.array([]), np.array([])

    if mu0 is None:
        mu0 = float(np.mean(x))
    if sigma is None:
        sigma = float(np.std(x, ddof=1)) if T > 1 else 0.0
    if sigma <= 1e-12:
        sigma = 1.0

    c_plus = np.zeros(T, dtype=np.float64)
    c_minus = np.zeros(T, dtype=np.float64)
    cp = 0.0
    cm = 0.0

    for t in range(T):
        if standardized:
            z = (x[t] - mu0) / sigma
            delta = k
        else:
            z = x[t] - mu0
            delta = k

        cp = max(0.0, cp + z - delta)
        cm = max(0.0, cm - z - delta)
        c_plus[t] = cp
        c_minus[t] = cm

    return c_plus, c_minus


def cumulative_sum_deviations(x: Array, mu0: Optional[float] = None) -> np.ndarray:
    """
    Plain cumulative sum S_t = sum_{i=1..t} (x_i - μ₀), useful as a simple baseline shift tracker.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if mu0 is None:
        mu0 = float(np.mean(x))
    return np.cumsum(x - mu0)


def compute_mu_series_from_episode(
    episode: Path,
    clip_lo: float = -20.0,
    clip_hi: float = 20.0,
    friction_mu_eps: float = 1e-3,
) -> Tuple[np.ndarray, dict]:
    """
    Load a .npz episode and return friction μ_t at each frame (dataloader-aligned).

    Returns:
        mus: shape (T,)
        meta: probe, mode, ref_force, etc.
    """
    from utils.force_friction import friction_coefficient_from_shear_normal

    data = np.load(episode, allow_pickle=True)
    ref_force = data["ref_force"] if "ref_force" in data else np.zeros(6, dtype=np.float64)
    force = np.asarray(data["6d_force"], dtype=np.float64)
    T = force.shape[0]
    mus = []
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
    mus = np.array(mus)
    meta = {
        "probe": data.get("probe", None),
        "mode": str(data.get("mode", "")) if "mode" in data else None,
        "T": T,
    }
    return mus, meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-frame friction μ and Page CUSUM on a sliding episode.")
    parser.add_argument(
        "--episode",
        type=str,
        default="./data/gsmini_force_4probe_50each/processed/p1_sliding/0_press_10.npz",
        help="Path to a processed .npz episode",
    )
    parser.add_argument("--friction-mu-eps", type=float, default=1e-3, dest="friction_mu_eps")
    parser.add_argument("--k", type=float, default=0.5, help="CUSUM slack (standardized units)")
    parser.add_argument(
        "--baseline-frac",
        type=float,
        default=0.25,
        help="Fraction of frames from the start used to estimate μ₀, σ for CUSUM",
    )
    args = parser.parse_args()
    episode = Path(args.episode)

    mus, meta = compute_mu_series_from_episode(episode, friction_mu_eps=args.friction_mu_eps)
    T = meta["T"]

    mus_min, mus_max = float(mus.min()), float(mus.max())
    mus_mean, mus_std = float(mus.mean()), float(mus.std())

    # Phase I: baseline for CUSUM from first portion of the μ series
    n0 = max(8, int(T * args.baseline_frac))
    mu0 = float(np.mean(mus[:n0]))
    sigma = float(np.std(mus[:n0], ddof=1)) if n0 > 1 else 1.0
    if sigma <= 1e-12:
        sigma = 1.0

    c_plus, c_minus = cusum_two_sided(mus, mu0=mu0, sigma=sigma, k=args.k, standardized=True)

    print("=" * 72)
    print("Friction μ (physical): ref_force subtract → clip [-20, 20] on shear/normal →")
    print(f"  μ_t = ‖shear‖ / (|normal| + {args.friction_mu_eps})  (no z-score)")
    print("Page CUSUM on μ_t: z_t = (μ_t - μ₀) / σ,  C⁺, C⁻ with k =", args.k)
    print("=" * 72)
    print(f"Episode: {episode}")
    print(f"T = {T}, probe = {meta['probe']}, mode = {meta['mode']}")
    print(f"μ series — min={mus_min:.6f}, max={mus_max:.6f}, mean={mus_mean:.6f}, std={mus_std:.6f}")
    print(
        f"CUSUM baseline (first {n0} frames): μ₀ = {mu0:.6f}, σ = {sigma:.6f}"
    )
    print(f"CUSUM — max(C⁺) = {float(c_plus.max()):.6f}, max(C⁻) = {float(c_minus.max()):.6f} "
          f"(final C⁺={c_plus[-1]:.6f}, C⁻={c_minus[-1]:.6f})")
    print()
    print("t\tmu\t\tC_plus\t\tC_minus")
    for t in range(T):
        print(f"{t}\t{mus[t]:.6f}\t{c_plus[t]:.6f}\t{c_minus[t]:.6f}")
