# HTT — Heterogeneous Tactile Transformer

A **multimodal tactile representation model**. One shared transformer backbone
encodes four different tactile sensors into a common `192`-dimensional embedding
space, pretrained with masked-autoencoder reconstruction and cross-modal
alignment. Drop in a raw sensor reading, get a feature vector you can feed to any
downstream head (classification, force / slip estimation, policy learning, …).

This is the **full release**: inference API, pretraining + downstream
training/evaluation code, supervised and external (SITR / T3) baselines, and
the **[HTT dataset](https://huggingface.co/datasets/AllenBi21/HTT-dataset)**
(~1.59M frames across 4 task splits) on the Hugging Face Hub.

Supported sensors:

| Modality | Type | Raw input |
|---|---|---|
| `gsmini` | vision (GelSight Mini) | uint8 image `[224, 224, 3]` |
| `9dtact` | vision (9DTact) | uint8 image `[224, 224, 3]` |
| `xela`   | taxel array | float `[T, 72]` |
| `tac02`  | taxel array | float `[T, 66]` |

All four map to a single `[B, 192]` embedding through one shared 9-layer trunk.

---

## Install

```bash
# Python 3.10 or 3.11 recommended
python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt
```

## Get the checkpoint

Model weights are hosted on the Hugging Face Hub
([AllenBi21/HTT](https://huggingface.co/AllenBi21/HTT), ~69 MB, not in git):

```bash
hf download AllenBi21/HTT htt_4sensors_best.pth --local-dir checkpoints
```

This places the file at `checkpoints/htt_4sensors_best.pth` (the default path
used by `htt.load_model`, `examples/extract_features.py`, and `train/finetune_mae.py`). See
[`checkpoints/README.md`](checkpoints/README.md) for curl/Python alternatives and
the SHA-256.

## Quickstart

**Python API** — the one-call `HTT` wrapper handles preprocessing:

```python
import numpy as np
from htt import HTT

tf = HTT(modality="gsmini")           # loads the checkpoint once
frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)   # your sensor frame
emb = tf(frame)                                 # -> torch.Tensor [1, 192]
```

Taxel is identical:

```python
tf = HTT(modality="xela")
reading = np.random.randn(20, 72).astype(np.float32)   # your [T, 72] reading
emb = tf(reading)                                       # -> [1, 192]
```

Lower-level API if you want the encoder/trunk handles:

```python
from htt import load_model, encode
encoder, trunk, preprocess = load_model(modality="gsmini")
emb = encode(encoder, trunk, preprocess(frame))         # [1, 192]
seq = encode(encoder, trunk, preprocess(frame), pool="none")  # [1, N, 192]
```

**Command line**:

```bash
python examples/extract_features.py --modality gsmini --input path/to/frame.png
python examples/extract_features.py --modality xela   --input path/to/reading.npy
python examples/extract_features.py --modality gsmini            # uses the bundled real sample
python examples/quickstart.py
```

The repo ships one **real sample recording** per modality under
`assets/samples/` (a short trimmed episode with reference frame + ground-truth
force), so every demo runs out of the box with no data setup.

## Dataset

The full training/evaluation dataset (**1.59M frames**, 4.3 GB) is hosted at
**[AllenBi21/HTT-dataset](https://huggingface.co/datasets/AllenBi21/HTT-dataset)**:
paired pretraining episodes for both sensor pairs (label-free), 20-object
classification episodes, 6D-force probe episodes (static mode), and slip-stage
sliding episodes — see the dataset card and its `FORMAT.md` for details.

```bash
hf download AllenBi21/HTT-dataset --repo-type dataset --local-dir HTT-dataset
```

All training scripts and configs expect it at `./HTT-dataset` (symlink is fine;
or set the `HTT_DATA_ROOT` environment variable — see `data/dataset_paths.py`).

## Training & evaluation

### Pretrain HTT (ours)

Joint MAE + cross-modal alignment over both sensor pairs — the exact recipe
behind the released checkpoint (`htt_4sensors_best.pth` = step 34k of this run;
~9 h for the full 60k steps on one RTX 4090, ~22 GiB):

```bash
python train/run_pretrain_joint.py \
  --pretrain_config config/model/pretrain.yaml \
  --ssl_config     config/algo/pretrain_joint.yaml
```

### Downstream tasks (classification / force / slip)

Finetune-probe a pretrained checkpoint per task × sensor (2-layer MLP head, or
a dual force head; `--finetune` also updates encoder + trunk — the paper
protocol):

```bash
CKPT=checkpoints/htt_4sensors_best.pth
for mod in 9dtact xela gsmini tac02; do
  for task in classification force sliding; do
    python train/run_probe.py --task $task --modality $mod \
      --checkpoint $CKPT --pretrain_config config/model/pretrain.yaml \
      --probe_config config/algo/probe.yaml --finetune
  done
done
```

Force uses the static-mode episodes only (the dataset's `force/` split); slip
uses the sliding episodes (`slip/` split) with 3-class bracket labels.

### Supervised-from-scratch baseline (SPL)

Per-sensor supervised training with the same encoders, no pretraining:

```bash
python train/run_spl.py --task_type force   --modality xela --seeds 10
python train/run_spl.py --task_type sliding --modality xela --seeds 10
```

For slip, always compare **macro-F1** (accuracy is inflated by the majority
class).

### External baselines (SITR / T3)

`baselines/load_baselines.py` runs the same probe pipeline on external vision-tactile
backbones (image sensors only). Clone the upstream repos + weights under
`third_party/` (or point `SITR_REPO` / `SITR_CHECKPOINT` / `T3_REPO` /
`T3_WEIGHTS_DIR` env vars at them):

- SITR — repo + `SITR_B18.pth` checkpoint from the SITR authors
- T3 — repo + `t3_medium` weights (`trunk.pth`, `encoders/`, `decoders/`)

```python
from baselines.load_baselines import run_baseline_probe
run_baseline_probe(backbone='sitr', task_type='classification', modality='gsmini', ...)
```

## How it works

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the 4-encoder + shared-trunk
  design and the inference forward pass.
- **[docs/PREPROCESSING.md](docs/PREPROCESSING.md)** — the exact raw-input
  contract per modality. **Read this** before feeding your own data; the model
  silently returns garbage on out-of-distribution inputs.
- **[docs/TRAINING.md](docs/TRAINING.md)** — full pretraining + downstream
  evaluation on the released dataset, and finetuning the backbone onto your
  own sensor with `train/finetune_mae.py`.

### Finetune onto a new sensor (short version)

```bash
# vision sensor — init from a pretrained vision encoder
python train/finetune_mae.py --sensor_type vision --data_dir /path/to/episodes

# taxel sensor — fresh encoder, inherited (frozen) trunk
python train/finetune_mae.py --sensor_type taxel --tactile_dim 72 \
    --data_dir /path/to/episodes --freeze_trunk
```

Data is a directory of per-episode `*.npy`/`*.npz` arrays — see
[docs/TRAINING.md](docs/TRAINING.md).

### Downstream example: predict contact force

A worked example of using the embedding for a task. Small `DualForceHead` MLPs
(trained on frozen HTT features, shipped in `examples/force_heads/`) regress 3D
contact force `(Fx, Fy, Fz)` for the **vision** sensors (gsmini, 9dtact). Runs
on the bundled real samples:

```bash
python examples/predict_force.py --modality gsmini
python examples/predict_force.py --modality 9dtact
```

```
modality        : gsmini   (gsmini_sample.npz, 16 frames)
mean L2 error   : 0.86 N  (over the episode)
peak-contact frame #8:
  ground truth  (Fx, Fy, Fz) = (-1.21, -1.40, -17.75) N
  prediction    (Fx, Fy, Fz) = (-1.15, -1.40, -17.98) N
```

The vision force heads land around 1 N mean L2. Force regression uses
per-episode reference subtraction — see the script header for details.

## Repository layout

```
htt.py                      # library: load_model / preprocess / encode / HTT
train/
├── run_pretrain_joint.py   # HTT pretraining — the released checkpoint's recipe
├── pretrain_base.py        # trainer library inherited by run_pretrain_joint
├── run_probe.py            # downstream tasks: classification / force / sliding
├── run_spl.py              # supervised-from-scratch baseline
└── finetune_mae.py         # finetune the backbone onto your own sensor
baselines/
└── load_baselines.py       # SITR / T3 external baselines (+ load_sitr / load_t3)
examples/
├── quickstart.py           # minimal end-to-end example
├── extract_features.py     # CLI: raw reading -> [B, 192] feature
├── predict_force.py        # downstream force-prediction example
└── force_heads/            # pretrained force-regression heads (one per modality)
model/                      # architecture (encoders, shared trunk, decoders, predictors)
data/                       # dataloaders for every dataset split (+ dataset_paths.py)
utils/                      # training loops, schedulers, metrics, MAE utilities
config/
├── model/pretrain.yaml     # architecture config (also taxel_tf / vit for SPL)
├── algo/                   # pretrain_joint / probe / spl configs
├── sensor/                 # per-pair data configs for the tar splits
└── data/*.yaml             # per-modality normalization + force stats
assets/
├── bg_data/                # per-sensor background references for preprocessing
└── samples/                # one real sample recording per modality
checkpoints/                # place htt_4sensors_best.pth here (see checkpoints/README.md)
HTT-dataset/                # place (or symlink) the downloaded dataset here
docs/                       # ARCHITECTURE / PREPROCESSING / TRAINING
```

## License

Released under **CC BY-NC 4.0** (non-commercial). Portions are derived from
Meta's V-JEPA / DINOv2 (Apache-2.0 and CC-BY-NC-4.0). See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for the full terms and attributions.
