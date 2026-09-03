"""Default locations of the HTT dataset splits.

The public dataset (https://huggingface.co/datasets/AllenBi21/HTT-dataset) is
expected at ``./HTT-dataset`` relative to the repo root:

    hf download AllenBi21/HTT-dataset --repo-type dataset --local-dir HTT-dataset

Set the ``HTT_DATA_ROOT`` environment variable (or symlink ``HTT-dataset``) to
point somewhere else. All defaults below derive from it; explicit ``data_root``
entries in configs always win over these defaults.

Layout (see the dataset's FORMAT.md):
    <root>/pretrain/<pair>/pretrain_{train,val,test}_{0000..0003}.tar
    <root>/classification/<pair>/supervised_{train,val,test}_{0000..0003}.tar
    <root>/force/<sensor>/processed/p{1..4}_static/*.npz
    <root>/slip/<sensor>/{processed,sliding_labeled}/p{1..4}_sliding/*
"""
import os

DATASET_ROOT = os.environ.get("HTT_DATA_ROOT", "HTT-dataset")

PAIRS = ("xela_9dtact", "tacniq_gsmini")
SENSORS = ("xela", "tac02", "9dtact", "gsmini")


def pretrain_pair_root(pair: str) -> str:
    return os.path.join(DATASET_ROOT, "pretrain", pair)


def classification_pair_root(pair: str) -> str:
    return os.path.join(DATASET_ROOT, "classification", pair)


def force_processed_root(sensor: str) -> str:
    return os.path.join(DATASET_ROOT, "force", sensor, "processed")


def slip_processed_root(sensor: str) -> str:
    return os.path.join(DATASET_ROOT, "slip", sensor, "processed")


def slip_labeled_root(sensor: str) -> str:
    return os.path.join(DATASET_ROOT, "slip", sensor, "sliding_labeled")


def probe_processed_root(sensor: str, mode_filter) -> str:
    """Task-appropriate npz root: slip layout for sliding, force layout otherwise."""
    return slip_processed_root(sensor) if mode_filter == "sliding" else force_processed_root(sensor)


def resolve_tar_dir(data_root: str, dataset_split_type: str, pair: str) -> str:
    """Resolve the directory holding the WebDataset tar shards.

    Accepts either a directory that directly contains the shards (legacy
    single-dir layout) or the dataset root of the released 4-split layout,
    where pretrain and supervised shards live under ``pretrain/<pair>`` and
    ``classification/<pair>`` respectively.
    """
    prefix = "pretrain" if dataset_split_type == "pretrain" else "supervised"
    import glob
    if glob.glob(os.path.join(data_root, f"{prefix}_*.tar")):
        return data_root
    sub = "pretrain" if dataset_split_type == "pretrain" else "classification"
    candidate = os.path.join(data_root, sub, pair)
    if glob.glob(os.path.join(candidate, f"{prefix}_*.tar")):
        return candidate
    raise FileNotFoundError(
        f"No {prefix}_*.tar shards found under {data_root} or {candidate}. "
        f"Download the dataset first (see data/dataset_paths.py docstring)."
    )
