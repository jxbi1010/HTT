# Checkpoints

The pretrained weights are **not committed to git** (`.pth` is gitignored). Place
the checkpoint file here before running anything:

```
checkpoints/htt_4sensors_best.pth
```

This is the default path used by `htt.load_model`,
`examples/extract_features.py`, and `train/finetune_mae.py`.

## What it is

| | |
|---|---|
| Name | `htt_4sensors_best.pth` |
| Size | ~69 MB |
| Contents | `model_state_dict` = 4 encoders (`xela`, `tac02`, `9dtact`, `gsmini`) + shared 9-layer trunk + 4 decoders (17.1 M params) |
| Embedding dim | 192 |
| Source | joint multimodal MAE + cross-modal alignment pretraining |

It is a **slim** checkpoint: only the model weights are kept (optimizer /
scheduler / predictor states from training were dropped). It works for both
inference and finetuning — `train/finetune_mae.py` only reads `model_state_dict`.

## How it was produced

Exported from the full training checkpoint by keeping only the model weights:

```python
import torch
full = torch.load("checkpoint_step_34000.pth", map_location="cpu", weights_only=False)
slim = {"model_state_dict": full["model_state_dict"],
        "step": 34000, "embed_dim": 192,
        "modalities": ["xela", "tac02", "9dtact", "gsmini"]}
torch.save(slim, "htt_4sensors_best.pth")
```

## Download

Hosted on the Hugging Face Hub (single repo for dataset & model):
**https://huggingface.co/datasets/AllenBi21/HTT-dataset**

```bash
# Option A — hf CLI (pip install huggingface_hub)
hf download AllenBi21/HTT-dataset htt_4sensors_best.pth --repo-type dataset --local-dir checkpoints

# Option B — curl (direct link)
curl -L -o checkpoints/htt_4sensors_best.pth \
  https://huggingface.co/datasets/AllenBi21/HTT-dataset/resolve/main/htt_4sensors_best.pth
```

```python
# Option C — from Python
from huggingface_hub import hf_hub_download
hf_hub_download("AllenBi21/HTT-dataset", "htt_4sensors_best.pth",
                repo_type="dataset", local_dir="checkpoints")
```

Integrity (SHA-256):

```
024f4c3a067168197d0a6996bbca5c03e744ed5abd1d35a666dbf78e7ac673f0  htt_4sensors_best.pth
```
