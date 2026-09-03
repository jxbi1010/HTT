# Training & evaluation

Everything here runs **from the repo root** and expects the
[HTT dataset](https://huggingface.co/datasets/AllenBi21/HTT-dataset) at
`./HTT-dataset` (symlink is fine, or set `HTT_DATA_ROOT`):

```bash
hf download AllenBi21/HTT-dataset --repo-type dataset --local-dir HTT-dataset
```

## Pretrain HTT

`run_pretrain_joint.py` is the entry point behind the released checkpoint:
joint MAE reconstruction + cross-modal alignment over both sensor pairs,
alignment active from step 0 with a 10k-step coefficient ramp, asymmetric
per-direction coefficients, and encoder stop-grad from the alignment branch
(encoders learn from reconstruction only).

```bash
python train/run_pretrain_joint.py \
  --pretrain_config config/model/pretrain.yaml \
  --ssl_config     config/algo/pretrain_joint.yaml
```

- ~9 h for the full 60k steps on one RTX 4090 (~22 GiB); checkpoints every
  2k steps under `checkpoints/pretrain_joint/<run>/`.
- The released `htt_4sensors_best.pth` is the (slimmed) **step-34k**
  checkpoint of this exact config.
- `pretrain_base.py` is the trainer library this script inherits — it is not
  an entry point.

## Downstream tasks (classification / force / slip)

`run_probe.py` evaluates a pretrained checkpoint per task × sensor. With
`--finetune` the encoder + trunk are updated together with the task head
(the paper protocol); without it, a frozen-encoder MLP probe.

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

- `classification` — 20 objects, paired supervised shards.
- `force` — 6D force; uses **static-mode episodes only** (the dataset's
  `force/` split), matching the paper.
- `sliding` — 3-class slip brackets on the `slip/` split; report
  **macro-F1** (accuracy is inflated by the majority class).
- Logs and results land under `logs/{classification,force,sliding}_probe/`.

## Supervised-from-scratch baseline (SPL)

Same encoder architectures, no pretraining:

```bash
python train/run_spl.py --task_type force   --modality xela --seeds 10
python train/run_spl.py --task_type sliding --modality xela --seeds 10
```

Configs: `config/model/{taxel_tf,vit}.yaml`, `config/algo/spl.yaml`.

## Finetune onto your own sensor

`finetune_mae.py` adapts the released backbone to a new tactile sensor from a
directory of raw episodes — see
[docs/TRAINING.md](../docs/TRAINING.md#finetuning-onto-your-own-sensor) for
the data format, flags, and how to use the finetuned checkpoint.

```bash
python train/finetune_mae.py --sensor_type taxel --tactile_dim 72 \
    --data_dir /path/to/episodes --freeze_trunk
```
