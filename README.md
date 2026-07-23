# HTT — Heterogeneous Tactile Transformer

A **multimodal tactile representation model**. One shared transformer backbone
encodes four different tactile sensors into a common `192`-dimensional embedding
space, pretrained with masked-autoencoder reconstruction and cross-modal
alignment. Drop in a raw sensor reading, get a feature vector you can feed to any
downstream head (classification, force / slip estimation, policy learning, …).

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
used by `htt.load_model`, `extract_features.py`, and `finetune_mae.py`). See
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
python extract_features.py --modality gsmini --input path/to/frame.png
python extract_features.py --modality xela   --input path/to/reading.npy
python extract_features.py --modality gsmini            # uses the bundled real sample
python examples/quickstart.py
```

The repo ships one **real sample recording** per modality under
`assets/samples/` (a short trimmed episode with reference frame + ground-truth
force), so every demo runs out of the box with no data setup.

## How it works

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the 4-encoder + shared-trunk
  design and the inference forward pass.
- **[docs/PREPROCESSING.md](docs/PREPROCESSING.md)** — the exact raw-input
  contract per modality. **Read this** before feeding your own data; the model
  silently returns garbage on out-of-distribution inputs.
- **[docs/TRAINING.md](docs/TRAINING.md)** — finetune the backbone onto your own
  sensor with `finetune_mae.py`.

### Finetune onto a new sensor (short version)

```bash
# vision sensor — init from a pretrained vision encoder
python finetune_mae.py --sensor_type vision --data_dir /path/to/episodes

# taxel sensor — fresh encoder, inherited (frozen) trunk
python finetune_mae.py --sensor_type taxel --tactile_dim 72 \
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
htt.py                  # library: load_model / preprocess / encode / HTT
extract_features.py     # CLI: raw reading -> [B, 192] feature
finetune_mae.py         # finetune the backbone onto your own sensor
model/                  # architecture (encoders, shared trunk, decoders, layers)
utils/                  # MAE utilities + checkpoint helpers
config/
├── model/pretrain.yaml # architecture config
└── data/*.yaml         # per-modality normalization + force stats
assets/
├── bg_data/            # per-sensor background references for preprocessing
└── samples/            # one real sample recording per modality
checkpoints/            # place htt_4sensors_best.pth here (see checkpoints/README.md)
examples/
├── quickstart.py       # minimal end-to-end example
├── predict_force.py    # downstream force-prediction example
└── force_heads/        # pretrained force-regression heads (one per modality)
docs/                   # ARCHITECTURE / PREPROCESSING / TRAINING
```

## License

Released under **CC BY-NC 4.0** (non-commercial). Portions are derived from
Meta's V-JEPA / DINOv2 (Apache-2.0 and CC-BY-NC-4.0). See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for the full terms and attributions.
