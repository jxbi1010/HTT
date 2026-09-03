#!/usr/bin/env python3
"""
Shared dataset and dataloader utilities.

Provides length computation for both regular and iterable datasets (e.g., WebDataset).
Used by run_spl.py and other scripts.
"""

from typing import Optional
from torch.utils.data import DataLoader, IterableDataset


def get_dataset_length(
    dataset,
    dataset_name: str = "dataset",
    default_estimate: int = 10000,
    cache: Optional[dict] = None,
) -> int:
    """
    Get the length of a dataset, handling both regular and iterable datasets.

    Args:
        dataset: The dataset to get length for
        dataset_name: Name of the dataset for logging purposes
        default_estimate: Default value if length cannot be determined
        cache: Optional dict to cache results (key: f'_dataset_len_{dataset_name}')

    Returns:
        int: Number of samples in the dataset
    """
    cache_key = f'_dataset_len_{dataset_name}'
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    try:
        length = len(dataset)
        if cache is not None:
            cache[cache_key] = length
        return length
    except (TypeError, AttributeError):
        if hasattr(dataset, '__len__'):
            try:
                length = len(dataset)
                if cache is not None:
                    cache[cache_key] = length
                return length
            except (TypeError, AttributeError):
                pass

    print(f"Warning: Could not determine length of {dataset_name} (iterable dataset). Using default estimate: {default_estimate}")
    if cache is not None:
        cache[cache_key] = default_estimate
    return default_estimate


def get_dataloader_length(
    dataloader,
    loader_name: str = "dataloader",
    default_estimate: int = 1000,
    cache: Optional[dict] = None,
) -> int:
    """
    Get the length of a dataloader, handling both regular and iterable datasets.

    Args:
        dataloader: The dataloader to get length for
        loader_name: Name of the dataloader for logging purposes
        default_estimate: Default value if length cannot be determined
        cache: Optional dict to cache results (key: f'_len_{loader_name}')

    Returns:
        int: Number of batches in the dataloader
    """
    cache_key = f'_len_{loader_name}'
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # For IterableDataset: PyTorch DataLoader.__len__ may return dataset length (samples)
    # instead of batch count in some versions. Always compute batches from dataset + batch_size.
    dataset = getattr(dataloader, 'dataset', None)
    if dataset is not None and isinstance(dataset, IterableDataset) and hasattr(dataset, '__len__'):
        try:
            dataset_len = len(dataset)
            batch_size = getattr(dataloader, 'batch_size', 1)
            drop_last = getattr(dataloader, 'drop_last', False)
            if drop_last and batch_size > 0:
                length = dataset_len // batch_size
            else:
                length = (dataset_len + batch_size - 1) // batch_size
            if cache is not None:
                cache[cache_key] = length
            return length
        except (TypeError, AttributeError):
            pass

    try:
        length = len(dataloader)
        if cache is not None:
            cache[cache_key] = length
        return length
    except TypeError:
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
            try:
                dataset_len = len(dataloader.dataset)
                batch_size = getattr(dataloader, 'batch_size', 1)
                drop_last = getattr(dataloader, 'drop_last', False)
                if drop_last and batch_size > 0:
                    length = dataset_len // batch_size
                else:
                    length = (dataset_len + batch_size - 1) // batch_size
                if cache is not None:
                    cache[cache_key] = length
                return length
            except (TypeError, AttributeError):
                pass

    # Last resort: calculate by iterating once
    collate_fn = getattr(dataloader, 'collate_fn', None)
    temp_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )

    try:
        length = sum(1 for _ in temp_dataloader)
        if length == 0:
            print(f"Warning: Calculated {loader_name} length is 0. Using default estimate: {default_estimate}")
            length = default_estimate
        if cache is not None:
            cache[cache_key] = length
        return length
    except Exception:
        print(f"Warning: Failed to calculate {loader_name} length. Using default estimate: {default_estimate}")
        if cache is not None:
            cache[cache_key] = default_estimate
        return default_estimate
