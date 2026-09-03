"""
Joint multimodal pretraining: MAE reconstruction + cross-modal alignment.

Strategy A: every step samples one paired batch (mod0, mod1) and computes 4 losses:
    L = MAE(mod0) + MAE(mod1) + alpha * [Align(mod0->mod1) + Align(mod1->mod0)]

No knowledge-distillation losses (joint training has no "old" model to preserve).

This is a separate entry point — `run_pretrain.py` is unchanged.

Usage:
    python train/run_pretrain_joint.py \\
        --pretrain_config config/model/pretrain.yaml \\
        --ssl_config     config/algo/pretrain_joint.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import deque
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make the repo root importable when running this script directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.predictor import (
    AlignPooler,
    AlignProjector,
    EmbeddingPredictor,
    MeanEmbeddingPredictor,
)
from data.paired_dataset_manager import PairedDatasetManager
from train.pretrain_base import PretrainConfig, PretrainTrainer
from utils.ssl_utils import compute_mae_loss, random_masking

log = logging.getLogger(__name__)


class JointPretrainTrainer(PretrainTrainer):
    """MAE reconstruction + cross-modal alignment, jointly optimized.

    Inherits model setup, MAE forward, augmentation, optimizer-builder helpers,
    and checkpoint utilities from PretrainTrainer. Replaces the dataset manager,
    optimizer (predictor params), train step, validation, and main training
    loop. KD losses from run_alignment are intentionally dropped.
    """

    # ------------------------------------------------------------------ #
    # Init: skip the parent's DatasetManager, install PairedDatasetManager
    # ------------------------------------------------------------------ #
    def __init__(self, config: PretrainConfig):
        # Mirror BaseTrainer init without going through PretrainTrainer.__init__
        # (which constructs the per-modality DatasetManager we don't want).
        from utils.base_trainer import BaseTrainer
        from utils.utils_model import set_all_seeds

        BaseTrainer.__init__(self)
        self.config = config.config
        self.pretrain_config_path = config.pretrain_config_path
        self.ssl_config_path = config.ssl_config_path
        self.modality_to_data_config = config.modality_to_data_config

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for joint pretraining.")
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)

        set_all_seeds(self.config.training.get("random_seed", 42))

        # --- Joint-training specific config --------------------------------
        # PretrainConfig only merges algorithm/training/logging from the ssl yaml;
        # top-level keys like `alignment_modality_pairs` are dropped, so re-read
        # the joint yaml directly to recover them.
        import yaml as _yaml
        with open(self.ssl_config_path, "r") as _f:
            _joint_yaml = _yaml.safe_load(_f) or {}
        pairs = _joint_yaml.get("alignment_modality_pairs", None)
        algo_kwargs = self.config.algorithm.get("kwargs", {})
        if pairs is None:
            raise ValueError(
                "Joint pretrain requires `alignment_modality_pairs` at the "
                "top level of the algo config (e.g. [[9dtact, xela], [gsmini, tac02]])."
            )
        self.alignment_modality_pairs: List[List[str]] = [list(p) for p in pairs]
        for pair in self.alignment_modality_pairs:
            if len(pair) != 2:
                raise ValueError(f"Each pair must have 2 modalities, got {pair}")
        self.alignment_modalities: List[str] = sorted(
            {m for pair in self.alignment_modality_pairs for m in pair}
        )

        # `alignment_loss_coef` accepts either a scalar (applied to every
        # direction) or a dict keyed by direction name like "9dtact_to_xela".
        # Per-direction coefs are needed to give asymmetric pull between
        # vision→taxel (the unstable direction per CLAUDE.md §6) and
        # taxel→vision; both should normally still be low.
        raw_coef = algo_kwargs.get("alignment_loss_coef", 1.0)
        if isinstance(raw_coef, (int, float)):
            self.alignment_loss_coef = float(raw_coef)
            self.alignment_loss_coef_per_dir: Dict[str, float] = {}
        else:
            coef_map = (
                OmegaConf.to_container(raw_coef, resolve=True)
                if not isinstance(raw_coef, dict) else dict(raw_coef)
            )
            self.alignment_loss_coef_per_dir = {
                str(k): float(v) for k, v in coef_map.items()
            }
            # Used only for the warmup ramp & for logging / printing the
            # "headline" coef. Pick the max so the ramp completes when any
            # direction has reached its target weight.
            self.alignment_loss_coef = max(
                self.alignment_loss_coef_per_dir.values()
            ) if self.alignment_loss_coef_per_dir else 0.0

        # If True, detach the encoder output before the shared trunk on the
        # ALIGNMENT branch. This routes alignment gradients only into the
        # shared trunk + predictors/projectors/poolers; the per-modality
        # encoders are then updated exclusively by MAE. Default False keeps
        # prior v2–v6/lowcoef behaviour (encoder still sees alignment grad).
        self.alignment_stop_grad_encoders = bool(
            algo_kwargs.get("alignment_stop_grad_encoders", False)
        )

        # Steps to train pure MAE before the alignment loss is added in. The
        # alignment branch is skipped when step < alignment_warmup_steps; after
        # that, alignment_loss_coef is applied as configured.
        self.alignment_warmup_steps = int(
            algo_kwargs.get("alignment_warmup_steps", 0)
        )

        # Rebuttal C3 control: shuffle the TARGET side of every alignment pair
        # within the batch (torch.randperm) so source/target frames are no
        # longer temporally matched. Applied during TRAINING only — validation
        # still measures alignment loss on true pairs so the val curves stay
        # comparable to the real run. If shuffled-pair pretraining matches
        # true-pair pretraining downstream, the alignment loss is a generic
        # regularizer rather than cross-modal supervision.
        self.alignment_shuffle_pairs = bool(
            algo_kwargs.get("alignment_shuffle_pairs", False)
        )

        # Additional linear ramp on the effective alignment coefficient after
        # the MAE-only warmup ends. Spreads the initial alignment-gradient
        # shock over `alignment_coef_warmup_steps` steps so the predictor can
        # absorb the bulk of the easy mapping work at low coef before the
        # encoder feels the full pull. 0 (default) = no ramp, backwards-compat.
        self.alignment_coef_warmup_steps = int(
            algo_kwargs.get("alignment_coef_warmup_steps", 0)
        )

        # Per-modality MAE loss weights. Higher = that modality contributes more
        # to the total loss. Default 1.0 keeps prior behaviour.
        mae_weights = algo_kwargs.get("mae_weights", {}) or {}
        if not isinstance(mae_weights, dict):
            mae_weights = OmegaConf.to_container(mae_weights, resolve=True)
        self.mae_weights: Dict[str, float] = {
            m: float(mae_weights.get(m, 1.0)) for m in self.alignment_modalities
        }

        # Whether to stop-grad the alignment target embedding. Default True
        # (SimSiam/BYOL/I-JEPA convention) to avoid collapse where the encoder
        # minimises alignment MSE by trivialising the target.
        self.alignment_detach_target = bool(
            algo_kwargs.get("alignment_detach_target", True)
        )

        # Per-modality {source, target} mask ratios used by the alignment forward
        ratios = algo_kwargs.get("alignment_mask_ratios", {})
        ratios = OmegaConf.to_container(ratios, resolve=True) if ratios else {}
        self.alignment_mask_ratios: Dict[str, Dict[str, float]] = {}
        for m in self.alignment_modalities:
            r = ratios.get(m)
            if not isinstance(r, dict) or "source" not in r or "target" not in r:
                raise ValueError(
                    f"alignment_mask_ratios must specify source/target for every "
                    f"alignment modality. Missing/invalid for '{m}'."
                )
            self.alignment_mask_ratios[m] = {
                "source": float(r["source"]),
                "target": float(r["target"]),
            }

        self.predictor_cfg = algo_kwargs.get(
            "predictor", {"depth": 3, "num_heads": 3, "mlp_ratio": 4.0}
        )
        self.predictor_cfg = (
            OmegaConf.to_container(self.predictor_cfg, resolve=True)
            if not isinstance(self.predictor_cfg, dict)
            else self.predictor_cfg
        )

        # Alignment target representation:
        #   "per_token" — predictor outputs per-token preds, masked-MSE against full target (v1–v6 default).
        #   "mean"      — predictor outputs a single [B, D] summary, MSE against mean-pooled target tokens.
        self.alignment_target_mode = str(
            algo_kwargs.get("alignment_target_mode", "per_token")
        )
        if self.alignment_target_mode not in ("per_token", "mean"):
            raise ValueError(
                f"alignment_target_mode must be 'per_token' or 'mean', "
                f"got '{self.alignment_target_mode}'."
            )

        # Phase-2 align-token mode (iteration 1):
        #   - Per-modality AlignPooler holds a learnable align token that cross-
        #     attends over the FULL unmasked encoder+trunk tokens of its modality
        #     (so fine-grained features survive).
        #   - Per-direction AlignProjector maps source align token -> predicted
        #     target align token; loss is MSE against the detached target align
        #     token.
        #   - When enabled, replaces the per-token / mean EmbeddingPredictors for
        #     alignment. MAE branches and decoders are untouched.
        self.align_token_mode = bool(algo_kwargs.get("align_token_mode", False))
        align_cfg = algo_kwargs.get("align_token", {}) or {}
        if not isinstance(align_cfg, dict):
            align_cfg = OmegaConf.to_container(align_cfg, resolve=True)
        self.align_token_cfg = {
            "pooler_depth":      int(align_cfg.get("pooler_depth", 2)),
            "pooler_num_heads":  int(align_cfg.get("pooler_num_heads", 3)),
            "pooler_mlp_ratio":  float(align_cfg.get("pooler_mlp_ratio", 4.0)),
            "projector_mlp_ratio": float(align_cfg.get("projector_mlp_ratio", 4.0)),
        }
        # Iter-2 knobs:
        #   align_use_projector=false  → drop AlignProjector entirely; src_align
        #       is fed directly into the alignment loss (no learnable cross-modal
        #       MLP that can trivially fit the detached target).
        #   align_loss_type='cosine_mse' → L2-normalize src/tgt align tokens
        #       before MSE. Equivalent to minimizing (2 - 2*cos_sim); direction
        #       must align, magnitude is free. MAE branches still prevent the
        #       trivial zero-magnitude collapse.
        self.align_use_projector = bool(algo_kwargs.get("align_use_projector", True))
        self.align_loss_type = str(algo_kwargs.get("align_loss_type", "mse"))
        if self.align_loss_type not in ("mse", "cosine_mse", "infonce"):
            raise ValueError(
                f"align_loss_type must be 'mse'|'cosine_mse'|'infonce', got '{self.align_loss_type}'."
            )
        # InfoNCE temperature for align_loss_type='infonce'. Smaller tau = harder
        # discrimination; 0.07 is the standard CLIP/MoCo value.
        self.align_infonce_tau = float(algo_kwargs.get("align_infonce_tau", 0.07))

        # --- Paired dataset manager (replaces per-modality DatasetManager) -
        # Paired loaders need a multimodal data config (training_type=multimodal,
        # so the WebDataset emits both `images` and `tactile`).
        paired_data_config = _joint_yaml.get(
            "paired_data_config", "config/sensor/multimodal_alignment_config.yaml",
        )
        if not os.path.exists(paired_data_config):
            raise FileNotFoundError(
                f"Joint pretrain needs a multimodal data config; missing: {paired_data_config}"
            )
        self.dataset_manager = PairedDatasetManager(
            data_config_path=paired_data_config,
            alignment_modality_pairs=self.alignment_modality_pairs,
        )

        # --- Reuse parent setup for model + SSL params ---------------------
        self.setup_model()
        self.setup_ssl_algorithm()
        self.setup_logging()
        self.setup_augmentation()

        # Build predictors / align-token modules AFTER model exists (need encoder embed_dim).
        # In align_token_mode we *replace* the per-token EmbeddingPredictors with
        # AlignPoolers (one per modality) + AlignProjectors (one per direction).
        if self.align_token_mode:
            self.predictors = torch.nn.ModuleDict()  # kept empty for code-path uniformity
            self.setup_align_token_modules()
        else:
            self.setup_predictors()
            self.align_poolers = torch.nn.ModuleDict()
            self.align_projectors = torch.nn.ModuleDict()

        # Optimizer must include predictor / align-token params.
        self.setup_optimizer()

        self.current_epoch = 0
        self.current_step = 0

        # Per-modality / per-pair tracking (carried through checkpoints)
        self.modality_best_losses = {m: float("inf") for m in self.model.modalities}
        self.modality_patience_counters = {m: 0 for m in self.model.modalities}
        self.modality_converged = {m: False for m in self.model.modalities}
        self.modality_epochs = {m: 0 for m in self.model.modalities}

    # ------------------------------------------------------------------ #
    # Predictors
    # ------------------------------------------------------------------ #
    def setup_predictors(self):
        embed_dim = self.model.embed_dim
        self.predictors = torch.nn.ModuleDict()
        # Per-direction config support. If self.predictor_cfg has the legacy
        # flat keys (`depth`/`num_heads`/`mlp_ratio`), treat as a single shared
        # cfg for all directions. Otherwise treat as a dict keyed by direction
        # name (e.g. "9dtact_to_xela").
        cfg = self.predictor_cfg or {}
        _flat_keys = {"depth", "num_heads", "mlp_ratio"}
        is_per_direction = bool(cfg) and not (set(cfg.keys()) & _flat_keys)
        default = {"depth": 3, "num_heads": 3, "mlp_ratio": 4.0}
        for pair in self.alignment_modality_pairs:
            mod0, mod1 = pair
            for src, tgt in [(mod0, mod1), (mod1, mod0)]:
                key = f"{src}_to_{tgt}"
                if is_per_direction:
                    dcfg = cfg.get(key, default)
                else:
                    dcfg = cfg if cfg else default
                if self.alignment_target_mode == "mean":
                    self.predictors[key] = MeanEmbeddingPredictor(
                        embed_dim=embed_dim,
                        depth=int(dcfg.get("depth", 3)),
                        num_heads=int(dcfg.get("num_heads", 3)),
                        mlp_ratio=float(dcfg.get("mlp_ratio", 4.0)),
                    ).to(self.device)
                else:
                    self.predictors[key] = EmbeddingPredictor(
                        embed_dim=embed_dim,
                        num_tokens=1000,  # over-provision; predictor resizes PE dynamically
                        depth=int(dcfg.get("depth", 3)),
                        num_heads=int(dcfg.get("num_heads", 3)),
                        mlp_ratio=float(dcfg.get("mlp_ratio", 4.0)),
                    ).to(self.device)
        if self.config.training.get("compile_model", False) and hasattr(torch, "compile"):
            for k in list(self.predictors.keys()):
                self.predictors[k] = torch.compile(self.predictors[k])
        print(f"\nCreated {len(self.predictors)} predictors:")
        for k, p in self.predictors.items():
            n_params = sum(t.numel() for t in p.parameters())
            print(f"  {k}: {n_params/1e6:.2f}M params")

    # ------------------------------------------------------------------ #
    # Align-token modules (phase-2 alignment path)
    # ------------------------------------------------------------------ #
    def setup_align_token_modules(self):
        """Build one AlignPooler per alignment modality and one AlignProjector
        per direction (src -> tgt and tgt -> src for every pair). Each pooler
        owns a learnable align-token parameter that cross-attends over the
        unmasked encoder+trunk token sequence of its modality."""
        embed_dim = self.model.embed_dim
        cfg = self.align_token_cfg

        self.align_poolers = torch.nn.ModuleDict()
        for m in self.alignment_modalities:
            self.align_poolers[m] = AlignPooler(
                embed_dim=embed_dim,
                depth=cfg["pooler_depth"],
                num_heads=cfg["pooler_num_heads"],
                mlp_ratio=cfg["pooler_mlp_ratio"],
            ).to(self.device)

        self.align_projectors = torch.nn.ModuleDict()
        if self.align_use_projector:
            for pair in self.alignment_modality_pairs:
                mod0, mod1 = pair
                for src, tgt in [(mod0, mod1), (mod1, mod0)]:
                    key = f"{src}_to_{tgt}"
                    self.align_projectors[key] = AlignProjector(
                        embed_dim=embed_dim,
                        mlp_ratio=cfg["projector_mlp_ratio"],
                    ).to(self.device)

        print(f"\n[align_token_mode] Created {len(self.align_poolers)} poolers + "
              f"{len(self.align_projectors)} projectors "
              f"(loss={self.align_loss_type}, use_projector={self.align_use_projector}):")
        for m, p in self.align_poolers.items():
            n_params = sum(t.numel() for t in p.parameters())
            print(f"  pooler[{m}]: {n_params/1e6:.3f}M params")
        for k, p in self.align_projectors.items():
            n_params = sum(t.numel() for t in p.parameters())
            print(f"  projector[{k}]: {n_params/1e6:.3f}M params")

    # ------------------------------------------------------------------ #
    # Optimizer (override): include predictor / align-token params
    # ------------------------------------------------------------------ #
    def setup_optimizer(self):
        algo = self.config.algorithm

        all_named = []
        for m in self.model.modalities:
            all_named += [(f"enc_{m}.{n}", p)
                          for n, p in self.model.encoders[m].named_parameters()]
            all_named += [(f"dec_{m}.{n}", p)
                          for n, p in self.model.decoders[m].named_parameters()]
        all_named += [(f"trunk.{n}", p)
                      for n, p in self.model.shared_trunk.named_parameters()]
        for k, predictor in self.predictors.items():
            all_named += [(f"pred_{k}.{n}", p)
                          for n, p in predictor.named_parameters()]
        for k, pooler in self.align_poolers.items():
            all_named += [(f"align_pool_{k}.{n}", p)
                          for n, p in pooler.named_parameters()]
        for k, proj in self.align_projectors.items():
            all_named += [(f"align_proj_{k}.{n}", p)
                          for n, p in proj.named_parameters()]

        self.optimizer = self._build_optimizer(
            all_named,
            optimizer_type=algo.optimizer.type,
            lr=algo.optimizer.lr,
            weight_decay=algo.optimizer.weight_decay,
            **algo.optimizer.get("kwargs", {}),
        )

        total_steps = self.config.training.get("total_steps", 100000)
        self.lr_scheduler, self.scheduler_per_iter = self._build_scheduler(
            self.optimizer, algo.get("lr_scheduler"), total_steps=total_steps,
        )

        self.setup_amp(self.config.training.use_amp)
        self.gradient_accumulation_steps = self.config.training.gradient_accumulation_steps

        # All-modality clip params (joint step touches everything)
        self._all_clip_params = []
        for m in self.model.modalities:
            self._all_clip_params += list(self.model.encoders[m].parameters())
            self._all_clip_params += list(self.model.decoders[m].parameters())
        self._all_clip_params += list(self.model.shared_trunk.parameters())
        for predictor in self.predictors.values():
            self._all_clip_params += list(predictor.parameters())
        for pooler in self.align_poolers.values():
            self._all_clip_params += list(pooler.parameters())
        for proj in self.align_projectors.values():
            self._all_clip_params += list(proj.parameters())

        # Set everything to train mode
        for m in self.model.modalities:
            self.model.encoders[m].train()
            self.model.decoders[m].train()
        self.model.shared_trunk.train()
        for predictor in self.predictors.values():
            predictor.train()
        for pooler in self.align_poolers.values():
            pooler.train()
        for proj in self.align_projectors.values():
            proj.train()

    # ------------------------------------------------------------------ #
    # Helpers — alignment forward (mirrors run_alignment.alignment_forward)
    # ------------------------------------------------------------------ #
    def _get_paired_batch_data(self, batch, modality: str) -> torch.Tensor:
        """Pull modality tensor from a paired batch. Tries modality-specific key
        first, then falls back to 'images'/'tactile'."""
        if modality in batch:
            return batch[modality].to(self.device, non_blocking=True)
        if modality in ("9dtact", "gsmini") and "images" in batch:
            return batch["images"].to(self.device, non_blocking=True)
        if modality in ("xela", "tac02") and "tactile" in batch:
            return batch["tactile"].to(self.device, non_blocking=True)
        raise KeyError(
            f"Paired batch missing data for '{modality}'. Got keys: {list(batch.keys())}"
        )

    def _encode_with_mask(
        self,
        x: torch.Tensor,
        modality: str,
        mask_ratio: float,
        stop_grad_encoder: bool = False,
    ):
        """Encoder + shared trunk on `x` with random masking (or no mask if 0).

        When `stop_grad_encoder` is True, the encoder output is detached before
        being passed into the shared trunk. The trunk still receives gradient,
        but the per-modality encoder does not — used by the alignment branch
        when `alignment_stop_grad_encoders=True`.
        """
        encoder = self.model.get_encoder(modality)
        patches = encoder.patchify(x)
        B, N, _ = patches.shape
        if mask_ratio > 0.0:
            ids_keep, mask, ids_restore = random_masking(B, N, x.device, mask_ratio)
        else:
            ids_keep = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            mask = torch.zeros(B, N, device=x.device)
            ids_restore = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        encoder_out = encoder(patches, masks=ids_keep)
        if stop_grad_encoder:
            encoder_out = encoder_out.detach()
        trunk_out = self.model.shared_trunk((encoder_out, mask, ids_restore))
        if isinstance(trunk_out, tuple):
            trunk_out = trunk_out[0]
        return trunk_out, mask, ids_restore, ids_keep, N

    def _compute_align_token(
        self,
        x: torch.Tensor,
        modality: str,
        stop_grad_encoder: bool = False,
    ) -> torch.Tensor:
        """Encode a modality unmasked through encoder + trunk and pool to a single
        align token via that modality's learnable AlignPooler. The pooler's
        learnable query attends over ALL N tokens, so fine-grained features
        survive the summary."""
        tokens, _, _, _, _ = self._encode_with_mask(
            x, modality, 0.0, stop_grad_encoder=stop_grad_encoder,
        )  # [B, N, D]
        return self.align_poolers[modality](tokens)  # [B, D]

    def _alignment_forward(self, batch, source_modality: str, target_modality: str):
        """Predict target trunk-embeddings from source trunk-embeddings.

        Returns three tensors plus None for the mask in mean/align_token modes:
          per_token mode    : (pred [B, N_t, D], target_full [B, N_t, D], target_mask [B, N_t])
          mean mode         : (pred [B, D],      target_mean [B, D],      None)
          align_token mode  : (pred [B, D],      target_align [B, D],     None)
        """
        source_data = self._get_paired_batch_data(batch, source_modality)
        target_data = self._get_paired_batch_data(batch, target_modality)

        # C3 shuffled-pair control: permute the target batch during training so
        # the pairing is temporally mismatched. Trunk train() state distinguishes
        # the training step from validate_pair (which sets eval()).
        if self.alignment_shuffle_pairs and self.model.shared_trunk.training:
            perm = torch.randperm(target_data.shape[0], device=target_data.device)
            target_data = target_data[perm]

        # ----- Phase-2 align-token path -----
        if self.align_token_mode:
            # Source side: unmasked encode + pool -> learnable align token.
            # When alignment_stop_grad_encoders is on, the encoder cannot
            # receive gradient from this branch (only the trunk + pooler do).
            src_align = self._compute_align_token(
                source_data, source_modality,
                stop_grad_encoder=self.alignment_stop_grad_encoders,
            )

            # Target side: same, but detached (and under no_grad) by default so
            # the encoder cannot collapse the target representation.
            if self.alignment_detach_target:
                with torch.no_grad():
                    tgt_align = self._compute_align_token(target_data, target_modality)
                tgt_align = tgt_align.detach()
            else:
                tgt_align = self._compute_align_token(target_data, target_modality)

            # With projector (iter 1): src_align -> learnable map -> pred_align.
            # Without projector (iter 2): pred_align = src_align; alignment is
            # forced to live in the encoder/pooler, not a learnable bypass.
            key = f"{source_modality}_to_{target_modality}"
            if self.align_use_projector and key in self.align_projectors:
                pred_align = self.align_projectors[key](src_align)
            else:
                pred_align = src_align
            return pred_align, tgt_align, None

        src_ratio = self.alignment_mask_ratios[source_modality]["source"]

        # Source: encode with source-side mask; predictor uses these as K/V.
        # alignment_stop_grad_encoders → only the trunk + predictor learn here.
        source_emb, _, _, _, _ = self._encode_with_mask(
            source_data, source_modality, src_ratio,
            stop_grad_encoder=self.alignment_stop_grad_encoders,
        )

        # Target: encode with NO mask to get the full ground-truth embeddings.
        # Detach by default so the alignment MSE cannot collapse the encoder by
        # making the target representation trivial (cf. SimSiam/BYOL/I-JEPA).
        if self.alignment_detach_target:
            with torch.no_grad():
                target_emb_full, _, _, _, _ = self._encode_with_mask(
                    target_data, target_modality, 0.0,
                )
            target_emb_full = target_emb_full.detach()
        else:
            target_emb_full, _, _, _, _ = self._encode_with_mask(
                target_data, target_modality, 0.0,
            )

        predictor = self.predictors[f"{source_modality}_to_{target_modality}"]

        if self.alignment_target_mode == "mean":
            pred = predictor(source_embeddings=source_emb)
            target_mean = target_emb_full.mean(dim=1)
            return pred, target_mean, None

        tgt_ratio = self.alignment_mask_ratios[target_modality]["target"]
        # Target masking for prediction: visible target embeddings = predictor query.
        target_visible_emb, target_mask, target_ids_restore, _, num_target_tokens = \
            self._encode_with_mask(target_data, target_modality, tgt_ratio)

        pred_emb = predictor(
            target_embeddings_visible=target_visible_emb,
            target_ids_restore=target_ids_restore,
            source_embeddings=source_emb,
            num_target_tokens=num_target_tokens,
        )
        return pred_emb, target_emb_full, target_mask

    def _alignment_loss(self, pred_emb, tgt_emb, mask):
        # Align-token mode loss variants. The first two have been shown
        # (iter 1, iter 2) to collapse: the pooler/projector can trivially
        # match a detached target without forcing the encoder to learn real
        # cross-modal features. The 'infonce' variant penalises that
        # collapse by requiring discrimination across the batch.
        loss_type = getattr(self, "align_loss_type", "mse")

        if mask is None and loss_type == "infonce":
            # InfoNCE / CLIP-style contrastive loss over a batch.
            # pred_emb [B, D] = source align token (with grad)
            # tgt_emb  [B, D] = target align token (detached)
            # Positive pair: (pred[i], tgt[i]); negatives: (pred[i], tgt[j!=i]).
            # The pooler+encoder cannot win by emitting one universal direction
            # because that drives every off-diagonal logit equal to the
            # diagonal → CE loss saturates at log(B).
            tau = getattr(self, "align_infonce_tau", 0.07)
            pred_n = F.normalize(pred_emb, p=2, dim=-1, eps=1e-6)
            tgt_n  = F.normalize(tgt_emb,  p=2, dim=-1, eps=1e-6)
            logits = pred_n @ tgt_n.transpose(0, 1) / tau           # [B, B]
            labels = torch.arange(logits.size(0), device=logits.device)
            return F.cross_entropy(logits, labels)

        if mask is None and loss_type == "cosine_mse":
            # MSE on L2-normalized vectors, summed over feature dim so the
            # loss is in [0, 4] (peer to plain MSE on raw embeddings).
            pred_n = F.normalize(pred_emb, p=2, dim=-1, eps=1e-6)
            tgt_n  = F.normalize(tgt_emb,  p=2, dim=-1, eps=1e-6)
            return ((pred_n - tgt_n) ** 2).sum(dim=-1).mean()
        if mask is None:
            return F.mse_loss(pred_emb, tgt_emb)
        per_token = F.mse_loss(pred_emb, tgt_emb, reduction="none").mean(dim=-1)
        denom = mask.sum()
        if denom.item() > 0:
            return (per_token * mask).sum() / denom
        return per_token.mean()

    # ------------------------------------------------------------------ #
    # Joint train step
    # ------------------------------------------------------------------ #
    def joint_train_step(self, pair: Sequence[str], step: int):
        mod0, mod1 = pair
        batch = self.dataset_manager.get_next_batch(list(pair))

        # Augmentation per modality
        if self.augmentation is not None:
            for m in (mod0, mod1):
                if m in batch:
                    batch[m] = self.augmentation(batch[m], training=True)
            if "images" in batch:
                batch["images"] = self.augmentation(batch["images"], training=True)
            if "tactile" in batch:
                batch["tactile"] = self.augmentation(batch["tactile"], training=True)

        x0 = self._get_paired_batch_data(batch, mod0)
        x1 = self._get_paired_batch_data(batch, mod1)

        forward_context = autocast("cuda") if self.use_amp else torch.enable_grad()
        align_active = step >= self.alignment_warmup_steps
        if not align_active:
            eff_align_coef = 0.0
        elif self.alignment_coef_warmup_steps > 0:
            progress = (step - self.alignment_warmup_steps) / float(
                self.alignment_coef_warmup_steps
            )
            eff_align_coef = self.alignment_loss_coef * min(1.0, progress)
        else:
            eff_align_coef = self.alignment_loss_coef
        w0 = self.mae_weights.get(mod0, 1.0)
        w1 = self.mae_weights.get(mod1, 1.0)
        with forward_context:
            # MAE on each modality (reuse parent's mae_forward)
            pred0, mask0, target0, _ = self.mae_forward(x0, mod0)
            mae0 = compute_mae_loss(pred0, target0, mask0, self.norm_pix_loss)

            pred1, mask1, target1, _ = self.mae_forward(x1, mod1)
            mae1 = compute_mae_loss(pred1, target1, mask1, self.norm_pix_loss)

            # Alignment in both directions (gated by warmup)
            if align_active:
                p_01, t_01, m_01 = self._alignment_forward(batch, mod0, mod1)
                align_01 = self._alignment_loss(p_01, t_01, m_01)

                p_10, t_10, m_10 = self._alignment_forward(batch, mod1, mod0)
                align_10 = self._alignment_loss(p_10, t_10, m_10)

                # Per-direction coef support: if alignment_loss_coef was a dict
                # in the config, look up per-direction weights; otherwise both
                # directions use the same scalar. The warmup ramp factor is
                # shared across directions (eff_align_coef / alignment_loss_coef).
                if self.alignment_loss_coef_per_dir:
                    ramp = (eff_align_coef / self.alignment_loss_coef) if self.alignment_loss_coef > 0 else 0.0
                    c01 = ramp * self.alignment_loss_coef_per_dir.get(
                        f"{mod0}_to_{mod1}", 0.0
                    )
                    c10 = ramp * self.alignment_loss_coef_per_dir.get(
                        f"{mod1}_to_{mod0}", 0.0
                    )
                else:
                    c01 = c10 = eff_align_coef
                align_term = c01 * align_01 + c10 * align_10
            else:
                # Skip alignment forwards entirely during MAE-only warmup to
                # save compute and avoid creating dangling parameter grads on
                # predictors (which their optimizer state would still touch).
                align_01 = torch.zeros((), device=self.device)
                align_10 = torch.zeros((), device=self.device)
                align_term = torch.zeros((), device=self.device)

            total_loss = (w0 * mae0 + w1 * mae1) + align_term

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            log.warning(f"Invalid loss at step {step} for pair {pair}: {total_loss.item()}")
            return None

        loss_value = self.grad_update_step(
            total_loss,
            self._all_clip_params,
            step,
            clip_val=float(self.config.training.get("gradient_clip_val", 0)),
            step_scheduler=True,
        )

        # Log component losses
        if step % self.config.logging.log_interval == 0:
            self.writer.add_scalar(f"{mod0}/Train_MAE", mae0.item(), step)
            self.writer.add_scalar(f"{mod1}/Train_MAE", mae1.item(), step)
            self.writer.add_scalar(f"align/{mod0}_to_{mod1}", align_01.item(), step)
            self.writer.add_scalar(f"align/{mod1}_to_{mod0}", align_10.item(), step)
            self.writer.add_scalar(f"align/coef_effective", eff_align_coef, step)
            if align_active and self.alignment_loss_coef_per_dir:
                self.writer.add_scalar(
                    f"align/coef_eff_{mod0}_to_{mod1}", float(c01), step,
                )
                self.writer.add_scalar(
                    f"align/coef_eff_{mod1}_to_{mod0}", float(c10), step,
                )
            self.writer.add_scalar(f"joint/Train_Loss", loss_value, step)
            self.writer.add_scalar(
                "joint/LearningRate", self.optimizer.param_groups[0]["lr"], step,
            )

        return {
            "total": loss_value,
            "mae0": mae0.item(), "mae1": mae1.item(),
            "align_01": align_01.item(), "align_10": align_10.item(),
        }

    # ------------------------------------------------------------------ #
    # Validation: per-modality MAE + per-direction alignment on val loaders
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate_pair(self, pair: Sequence[str], val_loader):
        mod0, mod1 = pair
        for m in self.model.modalities:
            self.model.encoders[m].eval()
            self.model.decoders[m].eval()
        self.model.shared_trunk.eval()
        for predictor in self.predictors.values():
            predictor.eval()

        for pooler in self.align_poolers.values():
            pooler.eval()
        for proj in self.align_projectors.values():
            proj.eval()

        mae0_sum = mae1_sum = a01_sum = a10_sum = 0.0
        n = 0
        for batch in val_loader:
            x0 = self._get_paired_batch_data(batch, mod0)
            x1 = self._get_paired_batch_data(batch, mod1)
            forward_context = autocast("cuda") if self.use_amp else torch.no_grad()
            with forward_context:
                pred0, mask0, target0, _ = self.mae_forward(x0, mod0)
                mae0 = compute_mae_loss(pred0, target0, mask0, self.norm_pix_loss)
                pred1, mask1, target1, _ = self.mae_forward(x1, mod1)
                mae1 = compute_mae_loss(pred1, target1, mask1, self.norm_pix_loss)
                p_01, t_01, m_01 = self._alignment_forward(batch, mod0, mod1)
                a01 = self._alignment_loss(p_01, t_01, m_01)
                p_10, t_10, m_10 = self._alignment_forward(batch, mod1, mod0)
                a10 = self._alignment_loss(p_10, t_10, m_10)
            mae0_sum += mae0.item()
            mae1_sum += mae1.item()
            a01_sum += a01.item()
            a10_sum += a10.item()
            n += 1

        # Restore train mode
        for m in self.model.modalities:
            self.model.encoders[m].train()
            self.model.decoders[m].train()
        self.model.shared_trunk.train()
        for predictor in self.predictors.values():
            predictor.train()
        for pooler in self.align_poolers.values():
            pooler.train()
        for proj in self.align_projectors.values():
            proj.train()

        n = max(n, 1)
        return {
            f"{mod0}_mae": mae0_sum / n,
            f"{mod1}_mae": mae1_sum / n,
            f"align_{mod0}_to_{mod1}": a01_sum / n,
            f"align_{mod1}_to_{mod0}": a10_sum / n,
        }

    # ------------------------------------------------------------------ #
    # Main training loop (replaces parent's modality-sampled loop)
    # ------------------------------------------------------------------ #
    def train(self, resume_from_checkpoint: str = None, fork_run: bool = False):
        total_steps = self.config.training.get("total_steps", 100000)
        val_interval = self.config.training.get("val_interval", 500)
        ckpt_interval = self.config.training.get("checkpoint_save_interval", 5000)

        start_step = 1
        if resume_from_checkpoint is not None:
            start_step = self.load_checkpoint(resume_from_checkpoint, fork_run=fork_run) + 1
            print(f"\nResuming from step {start_step}")
        else:
            print("Starting joint MAE + alignment pretraining…")

        print(f"Total steps: {total_steps:,}")
        print(f"Pairs: {self.alignment_modality_pairs}")
        if self.alignment_loss_coef_per_dir:
            print(f"Alignment loss coef (per direction):")
            for k, v in self.alignment_loss_coef_per_dir.items():
                print(f"  {k}: {v}")
        else:
            print(f"Alignment loss coef: {self.alignment_loss_coef}")
        print(
            f"Alignment stop-grad encoders: {self.alignment_stop_grad_encoders} "
            f"(True → alignment trains trunk+predictors only; encoder learns from MAE)"
        )
        print(f"Val interval: {val_interval} | Ckpt interval: {ckpt_interval}")

        # Load paired datasets
        print(f"\n{'='*80}\nLOADING PAIRED DATASETS\n{'='*80}")
        self.dataset_manager.load_dataset(self.config)

        # Pair sampling: weighted by paired-dataset size, or uniform.
        pair_keys = [f"{p[0]}_{p[1]}" for p in self.alignment_modality_pairs]
        pair_sizes = []
        for pair, key in zip(self.alignment_modality_pairs, pair_keys):
            ds = self.dataset_manager.pair_loaders[key]["train_dataset"]
            try:
                pair_sizes.append(len(ds))
            except (TypeError, AttributeError):
                pair_sizes.append(1)
        total = sum(pair_sizes) or 1
        pair_probs = [s / total for s in pair_sizes]
        print(f"Pair sampling probabilities:")
        for pair, prob, sz in zip(self.alignment_modality_pairs, pair_probs, pair_sizes):
            print(f"  {pair}: {prob:.2%} (n={sz})")

        # Running loss buffers
        loss_buf = {p: deque(maxlen=200) for p in pair_keys}

        progress = tqdm(
            range(start_step, total_steps + 1),
            desc="Joint Pretraining",
            initial=start_step - 1,
            total=total_steps,
        )

        for step in progress:
            self.current_step = step

            pair_idx = int(np.random.choice(len(self.alignment_modality_pairs), p=pair_probs))
            pair = self.alignment_modality_pairs[pair_idx]
            losses = self.joint_train_step(pair, step)

            if losses is not None:
                pkey = f"{pair[0]}_{pair[1]}"
                loss_buf[pkey].append(losses["total"])
                avg = sum(loss_buf[pkey]) / len(loss_buf[pkey])
                progress.set_postfix({
                    "pair": pkey,
                    "L": f"{losses['total']:.3f}",
                    "mae0": f"{losses['mae0']:.3f}",
                    "mae1": f"{losses['mae1']:.3f}",
                    "a01": f"{losses['align_01']:.3f}",
                    "a10": f"{losses['align_10']:.3f}",
                    "avg": f"{avg:.3f}",
                })

            # Validation
            if step % val_interval == 0:
                print(f"\nStep {step}: validating on each pair…")
                for pair, key in zip(self.alignment_modality_pairs, pair_keys):
                    val_loader = self.dataset_manager.pair_loaders[key]["val_loader"]
                    if val_loader is None:
                        continue
                    metrics = self.validate_pair(pair, val_loader)
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"val/{k}", v, step)
                    msg = f"  {pair}: " + ", ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items()
                    )
                    # Track per-modality "best MAE" for compatibility with parent's tracker
                    for m in pair:
                        cur = metrics.get(f"{m}_mae")
                        if cur is not None and cur < self.modality_best_losses[m]:
                            self.modality_best_losses[m] = cur
                            msg += f"  [NEW BEST {m}]"
                    print(msg)
                self.writer.flush()

                self.current_epoch = step // val_interval
                if step % ckpt_interval == 0:
                    self.save_checkpoint()

            if step % 100 == 0:
                torch.cuda.empty_cache()

        # Final summary
        print(f"\n{'='*80}\nJOINT PRETRAINING SUMMARY\n{'='*80}")
        for m in self.model.modalities:
            print(f"  {m}: best val MAE = {self.modality_best_losses[m]:.4f}")
        print("Done.")

        self.dataset_manager.clear_dataset()
        self.writer.close()

    # ------------------------------------------------------------------ #
    # Checkpoint: include predictor states + alignment metadata
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, is_best=False):
        checkpoint = self._base_checkpoint(self.model.state_dict())
        checkpoint.update({
            "config": self.config,
            "pretrain_config_path": self.pretrain_config_path,
            "ssl_config_path": self.ssl_config_path,
            "modality_to_data_config": self.modality_to_data_config,
            "modality_best_losses": self.modality_best_losses,
            "modality_patience_counters": self.modality_patience_counters,
            "modality_converged": self.modality_converged,
            "modality_epochs": self.modality_epochs,
            "log_path": self.log_path,
            "alignment_modality_pairs": self.alignment_modality_pairs,
            "alignment_loss_coef": self.alignment_loss_coef,
            "alignment_mask_ratios": self.alignment_mask_ratios,
            "predictors_state_dict": {
                k: v.state_dict() for k, v in self.predictors.items()
            },
            # Phase-2 align-token state (empty dicts when align_token_mode=false).
            "align_token_mode":     self.align_token_mode,
            "align_token_cfg":      self.align_token_cfg,
            "align_use_projector":  self.align_use_projector,
            "align_loss_type":      self.align_loss_type,
            "align_poolers_state_dict": {
                k: v.state_dict() for k, v in self.align_poolers.items()
            },
            "align_projectors_state_dict": {
                k: v.state_dict() for k, v in self.align_projectors.items()
            },
        })
        ckpt_dir = os.path.join(
            self.config.logging.save_dir, os.path.basename(self.log_path),
        )
        self._save_to_disk(
            checkpoint, ckpt_dir,
            f"checkpoint_step_{self.current_step}.pth",
            is_best=is_best,
            config=self.config,
        )

    def load_checkpoint(self, checkpoint_path: str, fork_run: bool = False) -> int:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"\n{'='*60}\nLOADING JOINT CHECKPOINT: {checkpoint_path}\n{'='*60}")
        if fork_run:
            print("  Fork mode: will skip log_path / TensorBoard restore.")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._load_model_weights(checkpoint, self.model)

        # Try optimizer restore; predictor-class changes (e.g. per_token → mean
        # predictor) make the saved param groups incompatible, so fall back to
        # restoring only the LR scheduler / scaler / step counters in that case.
        try:
            self._restore_common_state(checkpoint)
        except (ValueError, RuntimeError) as e:
            log.warning(
                f"  Optimizer-state restore failed ({type(e).__name__}: {e}); "
                "restoring scheduler / scaler / step counters only and starting "
                "Adam moments from zero."
            )
            if "lr_scheduler_state_dict" in checkpoint and self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                print("  LR scheduler state restored.")
            if "scaler_state_dict" in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print("  Grad scaler state restored.")
            self.current_epoch = checkpoint.get("epoch", 0)
            self.current_step  = checkpoint.get("step", 0)

        # Restore predictor weights if present (skip silently if loading from a
        # non-joint pretrain checkpoint, or if the saved predictor class differs
        # from the current one — predictors stay freshly initialized in those cases).
        pred_states = checkpoint.get("predictors_state_dict", {})
        for k, sd in pred_states.items():
            if k in self.predictors:
                try:
                    cur_keys = set(self.predictors[k].state_dict().keys())
                    saved_keys = set(sd.keys())
                    overlap = cur_keys & saved_keys
                    if not overlap:
                        log.info(
                            f"  Skipping predictor '{k}': no overlapping keys "
                            f"(saved={len(saved_keys)}, current={len(cur_keys)})."
                        )
                        continue
                    self.predictors[k].load_state_dict(sd, strict=False)
                except Exception as e:
                    log.warning(f"Could not restore predictor '{k}': {e}")

        # Restore align-token states if present in the checkpoint.
        # Missing => fresh init (e.g. resuming a non-align-token checkpoint into
        # align_token_mode for phase-2 training).
        pooler_states = checkpoint.get("align_poolers_state_dict", {}) or {}
        for k, sd in pooler_states.items():
            if k in self.align_poolers:
                try:
                    self.align_poolers[k].load_state_dict(sd, strict=False)
                except Exception as e:
                    log.warning(f"Could not restore align_pooler '{k}': {e}")
        proj_states = checkpoint.get("align_projectors_state_dict", {}) or {}
        for k, sd in proj_states.items():
            if k in self.align_projectors:
                try:
                    self.align_projectors[k].load_state_dict(sd, strict=False)
                except Exception as e:
                    log.warning(f"Could not restore align_projector '{k}': {e}")
        if self.align_token_mode and not pooler_states:
            print("  align_token_mode=true but checkpoint has no align-token state; "
                  "poolers/projectors stay freshly initialized.")

        for key in ("modality_best_losses", "modality_patience_counters",
                    "modality_converged", "modality_epochs"):
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
        print(f"  Modality best losses: {self.modality_best_losses}")

        saved_log_path = checkpoint.get("log_path")
        if saved_log_path and os.path.exists(saved_log_path) and not fork_run:
            if self.writer:
                self.writer.close()
            self.log_path = saved_log_path
            self.writer = SummaryWriter(self.log_path)
            print(f"  TensorBoard resumed → {self.log_path}")
        elif fork_run:
            print(f"  Fork mode: keeping fresh log_path → {self.log_path}")

        print(f"\nResuming from step {self.current_step}")
        return self.current_step


def main():
    parser = argparse.ArgumentParser(
        description="Joint multimodal pretraining (MAE + cross-modal alignment)."
    )
    parser.add_argument("--pretrain_config", type=str,
                        default="config/model/pretrain.yaml",
                        help="Path to pretrain MODEL config (encoders/decoders/trunk).")
    parser.add_argument("--ssl_config", type=str,
                        default="config/algo/pretrain_joint.yaml",
                        help="Path to joint-pretrain ALGO config.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional checkpoint to resume from.")
    parser.add_argument("--fork-run", dest="fork_run", action="store_true",
                        help="When resuming, skip restoring the saved log_path "
                             "so the new run gets its own checkpoint directory.")
    args = parser.parse_args()

    default_data_configs = {
        "9dtact": "config/sensor/9dtact_config.yaml",
        "xela":   "config/sensor/xela_config.yaml",
        "gsmini": "config/sensor/gsmini_config.yaml",
        "tac02":  "config/sensor/tac_config.yaml",
    }
    modality_to_data_config = {
        m: p for m, p in default_data_configs.items() if os.path.exists(p)
    }
    if not modality_to_data_config:
        raise ValueError("No sensor configs found under config/sensor/.")

    config = PretrainConfig(
        pretrain_config_path=args.pretrain_config,
        ssl_config_path=args.ssl_config,
        modality_to_data_config=modality_to_data_config,
    )
    trainer = JointPretrainTrainer(config)
    trainer.train(resume_from_checkpoint=args.resume, fork_run=args.fork_run)


if __name__ == "__main__":
    main()
