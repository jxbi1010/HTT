# Finetuning onto your own sensor

`finetune_mae.py` adapts the pretrained backbone to a **new tactile sensor**
using the same masked-autoencoder objective used in pretraining. You inherit the
shared 9-layer trunk (and, for vision, a pretrained encoder) and let the model
adapt to your data.

This is the supported training path in this release. Full multimodal
*pretraining from scratch* (joint MAE + cross-modal alignment on paired
WebDataset shards) is out of scope for v1 — it needs the paired multi-sensor
datasets, which are not distributed here.

## Data format

Point `--data_dir` at a directory of per-episode array files (`*.npy` or
`*.npz`), one file per episode:

| Sensor type | Array shape per file | Notes |
|---|---|---|
| `vision` | `[T_episode, H, W, 3]` | uint8 or float; any `H×W` (resized to 224) |
| `taxel`  | `[T_episode, ...]` | flattened to `[T_episode, tactile_dim]` internally |

For `.npz`, select the array with `--data_key` (default `arr_0`, or the sole key
if there's only one). Episodes are split 80/20 train/val by file index; fixed
`--chunk_size` windows are sampled from each.

## Vision sensor

```bash
python finetune_mae.py --sensor_type vision \
    --data_dir /path/to/vision_episodes \
    --init_encoder gsmini            # or 9dtact — which pretrained encoder to start from
```

- Encoder **and** decoder are initialized from the chosen pretrained vision
  encoder, so you inherit its features.
- Inputs are resized to `224×224` and divided by that encoder's
  `image_std_per_channel` (disable with `--no_image_std_normalize`).
- Defaults: `chunk_size=2`, `mask_ratio=0.75`.

## Taxel sensor

```bash
python finetune_mae.py --sensor_type taxel \
    --tactile_dim 72 \               # your sensor's per-timestep dimension (required)
    --data_dir /path/to/taxel_episodes \
    --freeze_trunk                   # recommended: keep the pretrained trunk fixed
```

- A **fresh** taxel encoder + decoder is trained from scratch (your sensor's
  layout is new), while the pretrained trunk is inherited.
- A per-dim std is computed over your corpus and saved as `taxel_std.npy` next to
  the checkpoint — **reuse it at inference**, never recompute.
- Defaults: `chunk_size=20`, `mask_ratio=0.6`.

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--total_steps` | 10000 | training steps |
| `--batch_size` | 64 | |
| `--lr` | 3e-4 | encoder/decoder LR |
| `--trunk_lr_mult` | 0.1 | trunk LR = `lr × this` (ignored if `--freeze_trunk`) |
| `--freeze_trunk` | off | freeze the shared trunk entirely |
| `--val_every` / `--save_every` | 500 / 2000 | validation / checkpoint cadence |

## Outputs

```
checkpoints/finetune_<name>/<timestamp>/
├── args.yaml                 # exact CLI args
├── best.pth                  # best-val encoder + trunk + decoder state dicts
├── checkpoint_step_*.pth
└── taxel_std.npy             # taxel runs only
```
Logs (TensorBoard) go to `logs/finetune_<name>/<timestamp>/`.

## Using a finetuned checkpoint for inference

A finetune checkpoint stores `encoder_state_dict` / `trunk_state_dict`
separately (not a full `PretrainModelWrapper`). To extract features:

```python
import torch
from omegaconf import OmegaConf
from model.create_model import create_pretrain_model
from htt import encode, build_preprocess

ckpt = torch.load("checkpoints/finetune_vision/<ts>/best.pth", map_location="cpu",
                  weights_only=False)

# Rebuild the matching encoder + trunk (vision example: init_encoder was gsmini)
cfg  = OmegaConf.to_container(OmegaConf.load("config/model/pretrain.yaml"), resolve=True)
base = create_pretrain_model(cfg)
encoder = base.encoders["gsmini"]
trunk   = base.shared_trunk
encoder.load_state_dict(ckpt["encoder_state_dict"])
trunk.load_state_dict(ckpt["trunk_state_dict"])
encoder.eval(); trunk.eval()

# For a taxel run, build a matching TactileTransformerEncoder(input_dim=tactile_dim, ...)
# and preprocess with the saved taxel_std.npy (abs → /std → clamp).
```
