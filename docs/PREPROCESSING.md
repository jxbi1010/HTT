# Preprocessing — the raw-input contract

The model has only ever seen inputs from the pretraining distribution. The
forward pass will run on *any* correctly-shaped tensor and return numbers, but
if you skip a step the features are silently garbage. `htt.preprocess`
(and `HTT.__call__`) apply exactly the steps below — use them.

## Vision modalities (`gsmini`, `9dtact`)

Raw input: **uint8** `[H, W, 3]` (single frame) or `[T, H, W, 3]`, at the
sensor's native resolution (the shipped backgrounds are `224×224`).

```
uint8 [H,W,3]
  → float / 255.0                 # to [0, 1] RGB
  → x − reference_bg              # per-recording background subtraction  (REQUIRED)
  → x / image_std_per_channel     # per-channel std normalization         (REQUIRED)
  → permute to CHW                # [3, H, W]
  → pad/truncate to T = 2 frames  # duplicate a single frame if needed
  → [1, 2, 3, H, W]
```

- **No mean subtraction, no extra `/255`.** Only background subtraction and
  per-channel std scaling.
- `reference_bg` is a **no-contact** frame for that recording. `assets/bg_data/`
  ships the backgrounds used in pretraining (`gsmini.png`, `9dtact.png`). If you
  change sensor / lighting / mounting, **recompute your own** and pass it via
  `bg_path=...`.
- `image_std_per_channel` lives in `config/data/{gsmini,9dtact}.yaml`.

## Taxel modalities (`xela`, `tac02`)

Raw input: **float** `[tactile_dim]` (single timestep) or `[T, tactile_dim]`,
with `tactile_dim = 72` (xela) or `66` (tac02).

```
float [T, D]
  → x − reference_bg_tactile      # per-recording background subtraction
  → abs(x)                        # rectify   ← taxel does this; vision does NOT
  → x / tactile_std               # per-dim std normalization
  → clamp(−10, 10)                # outlier safety cap
  → pad/truncate to T = 20        # repeat last timestep if needed
  → [1, 20, D]
```

- `reference_bg_tactile` ships as `assets/bg_data/{xela,tac02}.npy`.
- `tactile_std` lives in `config/data/xela.yaml` and `config/data/tacniq.yaml`
  (`tac02` uses the `tacniq` stats).

## Gotchas

| Don't | Why |
|---|---|
| Skip background subtraction or std normalization | Encoder silently returns collapsed/noisy features |
| Add `/255` mean-subtraction to vision | Model was not trained with it |
| Forget `abs()` on taxel | Vision does **not** rectify; taxel **does** |
| Feed vision with `T ≠ 2` or taxel with `T ≠ 20` | `preprocess` pads/truncates for you — but feed real frames when you have them; one duplicated frame degrades quality |
| Reuse pretraining backgrounds on a different sensor | Backgrounds are per-sensor; recompute yours |

## Where this is implemented

`htt.build_preprocess(modality, device, bg_path=None)` returns the
`preprocess(raw)` closure. It mirrors the pretraining dataloaders exactly, so
training and inference see identical distributions.
