# External baselines (SITR / T3)

`load_baselines.py` runs external vision-tactile backbones through the **same
probe pipeline** as HTT (`train/run_probe.py`), so the numbers are directly
comparable. Both are RGB-image backbones — only `gsmini` and `9dtact` are
supported; taxel modalities raise a clear error.

## Setup

Clone the upstream repos and weights under `third_party/` (gitignored), or
point the env vars anywhere else:

| What | Where | Env var |
|---|---|---|
| SITR repo | `third_party/SITR` | `SITR_REPO` |
| SITR checkpoint `SITR_B18.pth` | `<SITR_REPO>/checkpoints/` | `SITR_CHECKPOINT` |
| T3 repo | `third_party/t3` | `T3_REPO` |
| T3 `t3_medium` weights (`trunk.pth`, `encoders/`, `decoders/`) | `<T3_REPO>/checkpoints/t3_medium` | `T3_WEIGHTS_DIR` |

Weights come from the respective authors' releases (both on Hugging Face).

## Run

```python
import torch
from omegaconf import OmegaConf
from baselines.load_baselines import run_baseline_probe

results = run_baseline_probe(
    backbone='sitr',                 # or 't3'
    task_type='classification',      # or 'force'
    modality='gsmini',               # gsmini | 9dtact
    probe_config=OmegaConf.load('config/algo/probe.yaml'),
    device=torch.device('cuda'),
)
```

Or just grab a frozen, wrapped encoder that behaves like an HTT encoder
(`[B, T, 3, 224, 224] -> [B, T, embed_dim]`):

```python
from baselines.load_baselines import load_baseline_encoder
encoder = load_baseline_encoder('t3', device=torch.device('cuda'))
```

`load_sitr.py` / `load_t3.py` hold the per-backbone loading details.
