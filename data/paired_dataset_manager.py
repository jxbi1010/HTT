"""Paired multimodal dataset manager for joint pretraining.

Loads the paired WebDataset shards for each modality pair and hands out
batches round-robin per pair. Data roots default to the released dataset
layout (see data/dataset_paths.py).
"""
import logging
import os

import torch
import yaml
from omegaconf import OmegaConf

from data.create_dataloaders import create_dataloaders_from_config
from data.dataset_paths import pretrain_pair_root

log = logging.getLogger(__name__)


class PairedDatasetManager:
    """Manages paired multimodal dataset loading."""

    # Map modality pairs to dataset types
    PAIR_TO_DATASET_TYPE = {
        ('9dtact', 'xela'): 'xela_9dtact',
        ('xela', '9dtact'): 'xela_9dtact',
        ('gsmini', 'tac02'): 'tacniq_gsmini',
        ('tac02', 'gsmini'): 'tacniq_gsmini',
    }

    PAIR_DATA_ROOTS = {
        "xela_9dtact": pretrain_pair_root("xela_9dtact"),
        "tacniq_gsmini": pretrain_pair_root("tacniq_gsmini"),
    }

    def __init__(self, data_config_path: str, alignment_modality_pairs: list):
        """
        Args:
            data_config_path: Path to multimodal data config file (for training)
            alignment_modality_pairs: List of modality pairs, e.g.,
                [['9dtact', 'xela'], ['gsmini', 'tac02']]
        """
        self.data_config_path = data_config_path
        self.alignment_modality_pairs = alignment_modality_pairs
        self.pair_loaders = {}  # Dict[pair_key, {'train_loader', 'val_loader', 'train_iterator', ...}]
        self.train_loader = None  # For backward compatibility
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_iterator = None

    def load_dataset(self, alignment_config):
        """Build a per-pair dataloader from the base data config + pair root."""
        with open(self.data_config_path, "r") as f:
            base_data_config = yaml.safe_load(f) or {}
        for pair in self.alignment_modality_pairs:
            mod0, mod1 = pair
            pair_key = f"{mod0}_{mod1}"
            dataset_type = self.PAIR_TO_DATASET_TYPE.get((mod0, mod1))
            if dataset_type is None:
                raise ValueError(f"Unknown pair {pair}")
            data_root = self.PAIR_DATA_ROOTS.get(dataset_type)
            if data_root is None or not os.path.exists(data_root):
                raise FileNotFoundError(
                    f"Data root for {dataset_type} not found: {data_root}\n"
                    f"Download the dataset first: hf download AllenBi21/HTT-dataset "
                    f"--repo-type dataset --local-dir HTT-dataset (or set HTT_DATA_ROOT)."
                )
            cfg = dict(base_data_config)
            data_section = dict(cfg.get("data", {}))
            data_section["dataset_type"] = dataset_type
            data_section["data_root"] = data_root
            data_section["dataset_split_type"] = "pretrain"
            if hasattr(alignment_config, "training"):
                t = alignment_config.training
                if "batch_size" in t:
                    data_section["batch_size"] = t.batch_size
                if "num_workers" in t:
                    data_section["num_workers"] = t.num_workers
                if "train_data_percentage" in t:
                    data_section["train_data_percentage"] = t.train_data_percentage
            cfg["data"] = data_section
            dl_cfg = OmegaConf.create(cfg["data"])
            train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = \
                create_dataloaders_from_config(dl_cfg)
            self.pair_loaders[pair_key] = {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
                "train_iterator": iter(train_loader),
                "train_dataset": train_ds,
                "val_dataset": val_ds,
                "test_dataset": test_ds,
            }
            log.info(f"Loaded dataset for pair {pair_key} ({dataset_type}) from {data_root}")

        # Set default loaders to first pair (for backward compatibility)
        first_pair_key = f"{self.alignment_modality_pairs[0][0]}_{self.alignment_modality_pairs[0][1]}"
        first = self.pair_loaders[first_pair_key]
        self.train_loader = first["train_loader"]
        self.val_loader = first["val_loader"]
        self.test_loader = first["test_loader"]
        self.train_dataset = first["train_dataset"]
        self.val_dataset = first["val_dataset"]
        self.test_dataset = first["test_dataset"]
        self.train_iterator = first["train_iterator"]

    def get_next_batch(self, pair: list = None):
        """Get next batch from the train loader of a specific pair (or the first)."""
        if pair is None:
            try:
                return next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                return next(self.train_iterator)

        mod0, mod1 = pair
        pair_key = f"{mod0}_{mod1}"
        if pair_key not in self.pair_loaders:
            raise ValueError(
                f"No dataloader found for pair {pair}. Available pairs: {list(self.pair_loaders.keys())}")
        pair_loaders = self.pair_loaders[pair_key]
        try:
            return next(pair_loaders["train_iterator"])
        except StopIteration:
            pair_loaders["train_iterator"] = iter(pair_loaders["train_loader"])
            return next(pair_loaders["train_iterator"])

    def clear_dataset(self):
        """Clear dataset and free memory."""
        for loaders in self.pair_loaders.values():
            loaders["train_iterator"] = None
        self.pair_loaders = {}
        self.train_iterator = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        log.info("Cleared dataset")
