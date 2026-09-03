# Training

Two training paths:

1. **[Full pretraining & evaluation](#full-pretraining--evaluation)** on the
   released [HTT dataset](https://huggingface.co/datasets/AllenBi21/HTT-dataset)
   — reproduce the released checkpoint and the downstream task numbers.
2. **[Finetuning onto your own sensor](#finetuning-onto-your-own-sensor)** with
   `train/finetune_mae.py` — adapt the released backbone to a new sensor from a
   directory of raw episodes.

## Full pretraining & evaluation

Download the dataset to `./HTT-dataset` (or symlink / set `HTT_DATA_ROOT`):

```bash
hf download AllenBi21/HTT-dataset --repo-type dataset --local-dir HTT-dataset
```

**Pretraining (ours).** Joint MAE + cross-modal alignment over both sensor
pairs, alignment active from step 0 with a 10k-step coefficient ramp,
asymmetric per-direction coefficients, and encoder stop-grad from the
alignment branch (encoders learn from reconstruction only):

```bash
python train/run_pretrain_joint.py \
  --pretrain_config config/model/pretrain.yaml \
  --ssl_config     config/algo/pretrain_joint.yaml
```

Checkpoints land in `checkpoints/pretrain_joint/<run>/` every 2k steps; the
released `htt_4sensors_best.pth` is the (slimmed) step-34k checkpoint of this
exact recipe. (The MAE-only trainer lives in `train/pretrain_base.py`, inherited by the joint script.)

**Downstream tasks.** `train/run_probe.py` evaluates a pretrained checkpoint on
`classification` (20 objects, paired shards), `force` (6D force, static probe
episodes), and `sliding` (3-class slip brackets, sliding episodes) for any of
the four sensors. `--finetune` updates encoder + trunk together with the head
(the paper protocol); omit it for a frozen-encoder linear/MLP probe:

```bash
python train/run_probe.py --task force --modality xela \
  --checkpoint checkpoints/htt_4sensors_best.pth \
  --pretrain_config config/model/pretrain.yaml \
  --probe_config config/algo/probe.yaml --finetune
```

Results and logs land under `logs/{classification,force,sliding}_probe/`.
Force training/eval uses **static-mode episodes only** (the dataset's `force/`
split); slip uses the `slip/` split — always report **macro-F1** for slip.

**Baselines.** `train/run_spl.py --task_type {force,sliding,classification}
--modality <sensor>` trains the same encoders supervised-from-scratch
(configs: `config/model/{taxel_tf,vit}.yaml`, `config/algo/spl.yaml`).
`baselines/load_baselines.py` runs SITR / T3 through the identical probe pipeline —
clone those repos + weights under `third_party/` or set `SITR_REPO` /
`SITR_CHECKPOINT` / `T3_REPO` / `T3_WEIGHTS_DIR`.

---

# Finetuning onto your own sensor

`train/finetune_mae.py` adapts the pretrained backbone to a **new tactile sensor**
using the same masked-autoencoder objective used in pretraining. You inherit the
shared 9-layer trunk (and, for vision, a pretrained encoder) and let the model
adapt to your data.

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
python train/finetune_mae.py --sensor_type vision \
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
python train/finetune_mae.py --sensor_type taxel \
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
