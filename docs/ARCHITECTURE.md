# Architecture

HTT (Heterogeneous Tactile Transformer) is a **multimodal masked-autoencoder backbone**. Four
per-modality encoders feed a single shared transformer trunk that produces a
common `192`-dim embedding space. During pretraining, per-modality decoders
reconstruct masked inputs (MAE) and a cross-modal alignment objective pulls the
paired modalities together in the shared space.

```
                    raw sensor reading
                           │
                 ┌─────────┴──────────┐
                 │   preprocess()     │   (per-modality; see PREPROCESSING.md)
                 └─────────┬──────────┘
                           │
     ┌──────────┬──────────┼──────────┬──────────┐
     │          │          │          │          │
  xela enc   tac02 enc  9dtact enc  gsmini enc          per-modality encoders
 (taxel TF) (taxel TF)   (ViT)       (ViT)              depth 3, 3 heads, dim 192
     │          │          │          │
     └──────────┴────┬─────┴──────────┘
                     │  tokens [B, N, 192]
              ┌──────┴───────┐
              │ shared trunk │                          9-layer transformer
              │  (9 layers)  │                          3 heads, dim 192
              └──────┬───────┘
                     │  [B, N, 192]
        ┌────────────┴─────────────┐
        │                          │
   mean-pool over N           per-modality decoder
   → [B, 192]  ◀── inference   → MAE reconstruction (training only)
```

## Components

| Component | Type | Key config |
|---|---|---|
| `encoders["xela"]`   | Tactile transformer | `tactile_dim=72`, patch/stride 4, depth 3, 3 heads |
| `encoders["tac02"]`  | Tactile transformer | `tactile_dim=66`, patch/stride 4, depth 3, 3 heads |
| `encoders["9dtact"]` | Vision transformer (ViT) | `num_frames=2`, `tubelet_size=2`, depth 3, 3 heads |
| `encoders["gsmini"]` | Vision transformer (ViT) | `num_frames=2`, `tubelet_size=2`, depth 3, 3 heads |
| `shared_trunk`       | Transformer trunk | `depth=9`, `embed_dim=192`, 3 heads, `pooling=none` |
| `decoders[*]`        | MAE decoders | depth 3 — **reconstruction only, unused at inference** |

- **Hidden dim** everywhere: `192`. **Embedding dim** (trunk output): `192`.
- **Total weights** in the shipped checkpoint: ~17.1 M params (encoders + trunk +
  decoders).
- Vision encoders expect **T = 2 frames** at **224×224**; taxel encoders expect
  **T = 20 timesteps** of a `tactile_dim`-length vector.

## Where it lives in code

- `model/create_model.py::create_pretrain_model(cfg)` builds the whole
  `PretrainModelWrapper` from `config/model/pretrain.yaml`.
  - `.encoders` — `nn.ModuleDict` keyed by modality
  - `.shared_trunk` — `model/shared_trunk.py::TransformerTrunk`
  - `.decoders` — `nn.ModuleDict` keyed by modality
- `model/taxel_networks.py` — `TactileTransformerEncoder` / `TactileTransformerDecoder`
- `model/vision_networks.py` + `model/vision_transformer.py` — ViT encoder/decoder
- `model/layers/` — attention, blocks, patch embedding (shared building blocks)

## Inference forward pass

```python
feats  = encoder(x)              # [B, N, 192]  (patchify + embed happen inside)
tokens = shared_trunk(feats)     # [B, N, 192]
emb    = tokens.mean(dim=1)      # [B, 192]     ← the representation you use
```

`htt.encode(encoder, trunk, x)` does exactly this. The decoders and
the cross-modal predictors used during training are **not** needed to extract
features and are not built by the inference path.
