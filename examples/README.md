# Examples

Every example runs **out of the box** — the repo ships one real sample
recording per sensor under `assets/samples/` (a short trimmed episode with
reference frame and ground-truth force). You only need the checkpoint
(`checkpoints/htt_4sensors_best.pth`, see the root README).

## `quickstart.py` — minimal end-to-end

```bash
python examples/quickstart.py
```

Loads the model once and prints a `[1, 192]` embedding for a vision frame and
a taxel reading — the whole API in ~20 lines.

## `extract_features.py` — CLI feature extractor

```bash
python examples/extract_features.py --modality gsmini --input path/to/frame.png
python examples/extract_features.py --modality xela   --input path/to/reading.npy
python examples/extract_features.py --modality gsmini   # bundled real sample
```

Raw reading in, `[B, 192]` feature out (`--output feats.npy` to save). Read
[docs/PREPROCESSING.md](../docs/PREPROCESSING.md) before feeding your own
data — the model silently returns garbage on out-of-distribution inputs.

## `predict_force.py` — downstream force prediction

```bash
python examples/predict_force.py --modality gsmini
python examples/predict_force.py --modality 9dtact
```

Small `DualForceHead` MLPs (shipped in `force_heads/`, trained on frozen HTT
features) regress 3D contact force from the embedding — the vision heads land
around 1 N mean L2 on the bundled episodes. A worked template for putting
your own task head on top of HTT features.
