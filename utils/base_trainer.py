"""
BaseTrainer — shared training infrastructure for all trainer classes.

Provides: optimizer building (with weight-decay param groups), scheduler
building, AMP + gradient accumulation, a single grad_update_step(), logging
setup, and checkpoint save/restore helpers.  Each script subclasses this and
overrides only what is genuinely script-specific.
"""

import os
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import yaml

from utils.scheduler import LinearWarmupCosineDecayLR, create_scheduler
from utils.utils_model import adjust_state_dict_keys


class BaseTrainer:
    """Shared infrastructure inherited by PretrainTrainer, SSLTrainer, SPLTrainer."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer: optim.Optimizer = None
        self.lr_scheduler = None
        self.scaler: GradScaler = None
        self.use_amp: bool = False
        self.gradient_accumulation_steps: int = 1
        self.scheduler_per_iter: bool = False  # True only for LinearWarmupCosineDecayLR
        self.writer: SummaryWriter = None
        self.log_path: str = None
        self.current_epoch: int = 0
        self.current_step: int = 0

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #

    def _build_optimizer(
        self,
        named_params,          # iterable of (name, param) — e.g. model.named_parameters()
        optimizer_type: str,   # e.g. "AdamW"
        lr: float,
        weight_decay: float,
        **kwargs,
    ) -> optim.Optimizer:
        """
        Build an optimizer with weight-decay param groups:
        - 2-D+ parameters (weight matrices) receive weight_decay
        - 1-D parameters (biases, norms) always have weight_decay=0
        """
        params = {n: p for n, p in named_params if p.requires_grad}
        decay    = [p for p in params.values() if p.dim() >= 2]
        no_decay = [p for p in params.values() if p.dim() < 2]
        groups = [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt_cls = getattr(optim, optimizer_type)
        return opt_cls(groups, lr=float(lr), weight_decay=float(weight_decay), **kwargs)

    # ------------------------------------------------------------------ #
    # Scheduler
    # ------------------------------------------------------------------ #

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_cfg,          # dict or OmegaConf node; None → no scheduler
        total_steps: int,
        steps_per_epoch: int = 1,
    ):
        """
        Build an LR scheduler.  Returns (scheduler, per_iter: bool).
        per_iter=True means the caller must step() after every optimizer step;
        per_iter=False means the caller steps once per epoch.

        scheduler_cfg keys:
          type   : scheduler name (case-insensitive)
          kwargs : extra args forwarded to the scheduler
        """
        if not scheduler_cfg:
            return None, False

        stype = str(scheduler_cfg.get("type", "")).lower()
        kw = dict(scheduler_cfg.get("kwargs", {}))

        # Normalise string numbers that come from OmegaConf
        for k, v in kw.items():
            if isinstance(v, str):
                try:
                    kw[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    pass

        if stype in ("linearwarmupcosinedecaylr", "linear_warmup_cosine", "warmup_cosine"):
            warmup_steps = kw.get("warmup_steps") or (
                int(kw.get("warmup_epochs", 10)) * steps_per_epoch
            )
            scheduler = LinearWarmupCosineDecayLR(
                optimizer,
                warmup_steps=int(warmup_steps),
                total_steps=int(total_steps),
                warmup_start_lr=float(kw.get("warmup_start_lr", 0.0)),
                eta_min=float(kw.get("eta_min", 1e-6)),
            )
            self._log_scheduler("LinearWarmupCosineDecayLR", warmup_steps, total_steps, kw)
            return scheduler, True  # per-iter stepping

        # Fallback: standard PyTorch schedulers via create_scheduler
        stype_map = {
            "cosineannealinglr": "cosine",
            "cosine_annealing_lr": "cosine",
            "cosineannealingwarmrestarts": "cosine_warm_restarts",
            "cosine_warm_restarts": "cosine_warm_restarts",
        }
        normalized = stype_map.get(stype, stype)
        if "T_max" not in kw:
            kw["T_max"] = total_steps
        scheduler = create_scheduler(normalized, optimizer, **kw)
        print(f"Learning rate scheduler: {scheduler_cfg.get('type')}")
        return scheduler, False  # per-epoch stepping

    def _log_scheduler(self, name, warmup_steps, total_steps, kw):
        print(f"Learning rate scheduler: {name}")
        print(f"  Warmup steps : {warmup_steps}")
        print(f"  Total steps  : {total_steps}")
        print(f"  warmup_start_lr: {kw.get('warmup_start_lr', 0.0)}")
        print(f"  eta_min        : {kw.get('eta_min', 1e-6)}")

    # ------------------------------------------------------------------ #
    # AMP + gradient accumulation
    # ------------------------------------------------------------------ #

    def setup_amp(self, use_amp: bool, device_type: str = "cuda"):
        self.use_amp = use_amp
        self.scaler = GradScaler(device_type) if use_amp else None

    # ------------------------------------------------------------------ #
    # Single gradient-update step
    # ------------------------------------------------------------------ #

    def grad_update_step(
        self,
        loss: torch.Tensor,
        params,             # iterable of parameters to clip (or None to skip clipping)
        step: int,          # global step counter (0-indexed)
        clip_val: float = 0.0,
        step_scheduler: bool = False,
    ) -> float:
        """
        Scale loss for accumulation → backward → (clip) → optimizer step → zero_grad.
        Optionally steps the LR scheduler after the optimizer update.

        Returns the unscaled loss value (float) for logging.
        """
        scaled = loss / self.gradient_accumulation_steps
        if self.use_amp:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (step + 1) % self.gradient_accumulation_steps == 0:
            if clip_val > 0 and params is not None:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(params, clip_val)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            if step_scheduler and self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return loss.item()

    # ------------------------------------------------------------------ #
    # TensorBoard logging setup
    # ------------------------------------------------------------------ #

    def setup_logging(self, log_dir: str, save_dir: str, run_name: str):
        """Create TensorBoard writer at <log_dir>/<run_name>."""
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(self.log_path)

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #

    def _base_checkpoint(self, model_state_dict: dict) -> dict:
        """Return the state dict keys that every checkpoint shares."""
        ckpt = {
            "epoch": self.current_epoch,
            "step":  self.current_step,
            "model_state_dict": model_state_dict,
        }
        if self.optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.lr_scheduler:
            ckpt["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self.scaler:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        return ckpt

    def _save_to_disk(
        self,
        checkpoint: dict,
        checkpoint_dir: str,
        filename: str,
        is_best: bool = False,
        config=None,
    ) -> str:
        """Save checkpoint dict to disk; optionally also write best_model.pth."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Persist config yaml alongside checkpoint
        if config is not None:
            config_text = OmegaConf.to_yaml(config) if hasattr(config, "_metadata") else yaml.dump(config)
            with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
                f.write(config_text)

        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Best model saved  → {best_path}")
        else:
            print(f"Checkpoint saved  → {path}")
        return path

    def _restore_common_state(self, checkpoint: dict):
        """Restore optimizer / scheduler / scaler / counters from a checkpoint."""
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  Optimizer state restored.")
        if "lr_scheduler_state_dict" in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            print("  LR scheduler state restored.")
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("  Grad scaler state restored.")
        self.current_epoch = checkpoint.get("epoch", 0)
        self.current_step  = checkpoint.get("step",  0)

    def _load_model_weights(self, checkpoint: dict, model: torch.nn.Module, strict: bool = False):
        """Adjust and load model_state_dict from checkpoint into model."""
        state = checkpoint.get("model_state_dict")
        if state is None:
            raise ValueError("Checkpoint has no 'model_state_dict'.")
        state = adjust_state_dict_keys(state, model)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            print(f"  Missing keys   : {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Model weights restored.")
