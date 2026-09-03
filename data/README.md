# Data loading

Dataloaders for the released
[HTT dataset](https://huggingface.co/datasets/AllenBi21/HTT-dataset)
(**1.59M frames**, 4.3 GB, four task splits):

```bash
hf download AllenBi21/HTT-dataset --repo-type dataset --local-dir HTT-dataset
```

| split | contents | loader |
|---|---|---|
| `pretrain/` | paired episodes, both pairs, label-free | `xela_9dtact_dataloader_webdataset.py`, `tacniq_gsmini_dataloader_webdataset.py` |
| `classification/` | 20-object supervised paired episodes | same tar loaders, `dataset_split_type="supervised"` |
| `force/` | static probe episodes with 6D F/T | `taxel_force_4probe_50each_dataloader.py`, `gsmini_force_4probe_50each_dataloader.py` |
| `slip/` | sliding episodes + 3-class bracket labels | same npz loaders, `mode_filter="sliding"` |

All default paths resolve through **`dataset_paths.py`**: the dataset is
expected at `./HTT-dataset` relative to the repo root; set the
`HTT_DATA_ROOT` env var (or symlink) to relocate it. Explicit `data_root`
entries in configs always win over these defaults. The tar loaders accept
either a directory that directly holds the shards or the dataset root (they
resolve `pretrain/<pair>` vs `classification/<pair>` from the split type).

`paired_dataset_manager.py` feeds `train/run_pretrain_joint.py` batches from
both sensor pairs round-robin; `create_dataloaders.py` is the config-driven
factory used by the training scripts.

Formats, episode structure, and per-split statistics: see the dataset card
and its `FORMAT.md` / `STATS.json` on the Hub.
