"""
Result aggregation utilities for multi-run/multi-trial downstream evaluation.

Provides aggregate_force_results and aggregate_classification_results to compute
mean, std, min, max across runs (run_spl multi-seed, run_probe multi-trial).
"""

from typing import Dict, List, Any
import numpy as np


def _aggregate_values(values: List[float]) -> Dict[str, Any]:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'values': [float(v) for v in values],
    }


def aggregate_force_results(run_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate force metrics across multiple runs (e.g. multi-seed or multi-trial).

    Handles both run_spl format (test_normal_force_mae, test_shear_force_mae) and
    run_probe format (test_normal_mae, test_shear_mae, test_force_mae, etc.).

    Args:
        run_results: List of dicts from each run. Each dict may contain:
            - test_mae, test_rmse (always)
            - best_val_mae (run_spl)
            - test_normal_force_mae, test_shear_force_mae (run_spl)
            - test_normal_mae, test_shear_mae, test_force_mae (run_probe)
            - test_normal_rmse, test_shear_rmse, test_force_rmse (run_probe)

    Returns:
        Dict mapping metric name to stats dict: {mean, std, min, max, values}.
        Keys: test_mae, test_rmse, best_val_mae, test_normal_mae, test_shear_mae,
        test_normal_rmse, test_shear_rmse, test_force_mae, test_force_rmse.
    """
    stats: Dict[str, Dict[str, Any]] = {}

    # Core metrics
    for key in ('test_mae', 'test_rmse', 'best_val_mae'):
        vals = []
        for r in run_results:
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
        if vals:
            stats[key] = _aggregate_values(vals)

    # Normal/shear: run_spl uses test_normal_force_mae, test_shear_force_mae
    # run_probe uses test_normal_mae, test_shear_mae
    normal_mae_src = ('test_normal_mae', 'test_normal_force_mae')
    shear_mae_src = ('test_shear_mae', 'test_shear_force_mae')
    for out_key, src_keys in [('test_normal_mae', normal_mae_src), ('test_shear_mae', shear_mae_src)]:
        vals = []
        for r in run_results:
            v = r.get(src_keys[0]) or r.get(src_keys[1])
            if v is not None:
                vals.append(float(v))
        if vals:
            stats[out_key] = _aggregate_values(vals)

    # run_probe 6D: test_normal_rmse, test_shear_rmse, test_force_mae, test_force_rmse
    for key in ('test_normal_rmse', 'test_shear_rmse', 'test_force_mae', 'test_force_rmse'):
        vals = []
        for r in run_results:
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
        if vals:
            stats[key] = _aggregate_values(vals)

    return stats


def aggregate_classification_results(
    run_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate classification / sliding metrics across multiple runs.

    Args:
        run_results: List of dicts from each run. Each dict may contain:
            - test_accuracy, best_val_accuracy
            - test_macro_f1, best_val_macro_f1
            - test_per_class_f1: list of per-class F1 (length = num_classes)

    Returns:
        Dict mapping metric name to stats dict: {mean, std, min, max, values}.
        Adds per-class F1 stats under keys ``test_per_class_f1_<c>`` for each class.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for key in ('test_accuracy', 'best_val_accuracy', 'test_macro_f1', 'best_val_macro_f1'):
        vals = []
        for r in run_results:
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
        if vals:
            stats[key] = _aggregate_values(vals)

    # Per-class F1 (sliding has 3 classes; classification has many). Only emit when
    # all runs report the same num_classes — otherwise skip silently.
    per_class_lists: List[List[float]] = []
    for r in run_results:
        pc = r.get('test_per_class_f1')
        if pc is not None:
            per_class_lists.append([float(x) for x in pc])
    if per_class_lists and all(len(x) == len(per_class_lists[0]) for x in per_class_lists):
        for c in range(len(per_class_lists[0])):
            stats[f'test_per_class_f1_{c}'] = _aggregate_values([x[c] for x in per_class_lists])

    return stats
