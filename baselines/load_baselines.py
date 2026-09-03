"""
Baseline encoder interface for probe evaluation.

Unifies external vision baselines (SITR, T3) behind the same probe training pipeline
used for native encoders in run_probe.py. After loading, the encoder is frozen and
passed to run_probe_mlp together with the dataloaders produced by ProbeDataloaderManager,
so classification and force probe results are directly comparable.

Two entry points:

    # 1. Just get a wrapped encoder (mirrors the output of run_probe.py's encoder build)
    encoder, embed_dim, is_seq = load_baseline_encoder('sitr', device=device)

    # 2. Full probe run against a modality (classification or force)
    results = run_baseline_probe(
        backbone='sitr',                # or 't3'
        task_type='classification',     # or 'force'
        modality='gsmini',              # gsmini | 9dtact  (taxel modalities unsupported)
        probe_config=OmegaConf.load('config/algo/probe.yaml'),
        device=torch.device('cuda'),
    )

Both SITR and T3 are RGB-image backbones, so only image modalities (gsmini, 9dtact)
are supported. Taxel modalities (xela, tac02) raise a clear error.
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

# Make the repo root importable when running this script directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.run_probe import run_probe_mlp
from data.create_dataloaders import create_dataloaders_from_config, get_num_classes_from_dataset
from data.gsmini_force_4probe_50each_dataloader import (
    create_force_4probe_50each_dataloader as create_force_dataloaders_4probe,
    load_force_stats_per_dim as load_force_stats_per_dim_4probe,
)

# Upstream baseline repos — clone them under third_party/ (see README
# "Baselines") or point the env vars elsewhere:
#   SITR: https://github.com/hmarticorena/SITR   (checkpoint SITR_B18.pth)
#   T3:   https://github.com/alanzjl/t3          (t3_medium weights)
_SITR_REPO = os.environ.get('SITR_REPO', 'third_party/SITR')
_T3_REPO = os.environ.get('T3_REPO', 'third_party/t3')
_SITR_CHECKPOINT = os.environ.get(
    'SITR_CHECKPOINT', os.path.join(_SITR_REPO, 'checkpoints/SITR_B18.pth'))

# T3 pretrained components (encoder/<name>.pth + trunk.pth + decoders/<name>.pth).
# The finetune_exp_cls.yaml config matches t3_medium (embed_dim=768, enc_depth=3,
# trunk_depth=9). Override via `t3_weights_dir=` on load_baseline_encoder or the
# T3_WEIGHTS_DIR env var if needed.
_T3_WEIGHTS_DIR_CANDIDATES = tuple(p for p in (
    os.environ.get('T3_WEIGHTS_DIR'),
    os.path.join(_T3_REPO, 'checkpoints/t3_medium'),
) if p)

IMAGE_MODALITIES = ('gsmini', '9dtact')
TAXEL_MODALITIES = ('xela', 'tac02')

# ----- Per-backbone preprocessing (matches each repo's training convention) -----
# SITR: trained on (sample - reference) in 0-255 pixel space, then channelwise
# (x - mean) / std with these stats. Our dataloader supplies (sample/255 - ref/255)
# so we pre-scale by 255.0 before applying the mean/std.
# (see gsrl/dataloaders.py sample_mu / sample_std)
SITR_PRE_SCALE = 255.0
SITR_MEAN = (-1.2223, -1.8114, -1.7090)
SITR_STD = (11.7932, 12.7956, 13.6452)
SITR_NEEDS_BG_SUBTRACT = True

# T3: no background subtraction; raw [0,1] RGB with per-dataset (x - mean) / std.
# See t3/t3/data_loader.py:20-68.
# T3 computes `img_norm` per dataset. For the Mini sensor specifically, the T3
# repo uses the values from eval_ds_cls_mini.yaml / eval_ds_pose_mini.yaml. We
# default to those for modality 'gsmini'. For other modalities (e.g. 9dtact, not
# a native T3 sensor) we fall back to ImageNet -- the T3 framework's own default
# when no img_norm is supplied. Pass `img_norm_mean`/`img_norm_std` to
# load_t3_encoder / load_baseline_encoder to override (e.g. with statistics
# computed on our own raw gsmini frames for strictest convention matching).
T3_PRE_SCALE = 1.0
T3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
T3_IMAGENET_STD = (0.229, 0.224, 0.225)
T3_NEEDS_BG_SUBTRACT = False

# Per-modality T3 img_norm: source listed alongside for traceability.
# T3's convention is per-dataset stats on raw [0,1] RGB. We computed these on
# our own raw (non-subtracted) tactile frames from the 4probe_50each datasets
# (80 episodes, ~40M pixels, per-channel mean/std). Using T3's cnc_Mini defaults
# here gave inputs several sigma outside T3's training distribution -> bad probe.
T3_IMG_NORM_BY_MODALITY: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    'gsmini': {
        'mean': (0.34855, 0.33071, 0.22839),
        'std':  (0.22232, 0.19994, 0.16457),
    },
    '9dtact': {
        'mean': (0.41965, 0.42487, 0.39501),
        'std':  (0.24389, 0.24589, 0.23036),
    },
}


# AnyTouch (ICLR'25, GeWu-Lab): CLIP-ViT-L/14 touch tower pretrained on TacQuad
# (GelSight Mini native, id 3) + stage-1 corpora. Its probe convention
# (run_probe_*.sh): raw [0,1] RGB resized to 224, ImageNet mean/std, NO bg
# subtraction, --use_same_patchemb (video Conv3d patch embedding applied to the
# static image repeated 3x) and --use_sensor_token (5 learned tokens inserted
# after CLS). We replicate that convention exactly.
ANYTOUCH_CKPT = 'third_party/anytouch/checkpoint.pth'
ANYTOUCH_CLIP_CONFIG = 'third_party/anytouch/AnyTouch/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
ANYTOUCH_MEAN = (0.485, 0.456, 0.406)
ANYTOUCH_STD = (0.229, 0.224, 0.225)
ANYTOUCH_NEEDS_BG_SUBTRACT = False
# Sensor-token slots: 0=GelSight'17, 1=DIGIT, 2=ObjectFolder-Real, 3=GelSight
# Mini (TacQuad+Octopi), 4=DuraGel, 5 was used by AnyTouch's own new-sensor
# probe (obj2). 9DTact is unseen -> fresh slot 6, mirroring their new-sensor
# convention (token zero-initialized; trained during --finetune probes).
ANYTOUCH_SENSOR_SLOT = {'gsmini': 3, '9dtact': 6}


class _ImageNorm(nn.Module):
    """`(pre_scale * x - mean) / std` with per-channel buffers, applied to [B, C, H, W]."""

    def __init__(self, pre_scale: float, mean, std):
        super().__init__()
        self.pre_scale = float(pre_scale)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.pre_scale - self.mean) / self.std


# -----------------------------------------------------------------------------
# Shared encoder wrapper
# -----------------------------------------------------------------------------
class BaselineEncoderWrapper(nn.Module):
    """
    Wraps an image-encoder callable so it plugs into run_probe_mlp.

    The underlying `encoder_fn` must accept [B, C, H, W] and return either
    [B, D] (pooled) or [B, N, D] (sequence, with CLS at position 0 when
    `use_cls_token_only=True`).

    This class handles 5D video inputs [B, T, C, H, W] by folding T into B,
    running the encoder per-frame, and re-folding so run_probe_mlp's
    extract_features sees a consistent [B, T, D] or [B, T, N, D] tensor.
    """

    def __init__(
        self,
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        inner_module: Optional[nn.Module] = None,
        use_cls_token_only: bool = True,
        image_norm: Optional[_ImageNorm] = None,
    ):
        super().__init__()
        self._encoder_fn = encoder_fn
        # Register the inner module so .to(device)/.eval()/.parameters() propagate.
        if inner_module is not None:
            self._inner = inner_module
        self.use_cls_token_only = use_cls_token_only
        self.image_norm = image_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_video = x.dim() == 5
        if is_video:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        if self.image_norm is not None:
            x = self.image_norm(x)

        feats = self._encoder_fn(x)
        if isinstance(feats, tuple):
            feats = feats[0]

        # feats is now [B*T, D] or [B*T, N, D]
        if feats.dim() == 3 and self.use_cls_token_only:
            feats = feats[:, 0, :]  # CLS token

        if is_video:
            if feats.dim() == 2:
                feats = feats.view(B, T, -1)
            else:
                N, D = feats.shape[1], feats.shape[2]
                feats = feats.view(B, T, N, D)
        return feats


# -----------------------------------------------------------------------------
# SITR loader
# -----------------------------------------------------------------------------
def _load_sitr_raw(num_calibration: int = 18) -> nn.Module:
    """Load SITR_base from the SITR_B18 checkpoint.

    The published checkpoint was trained with num_calibration=18 — its
    c_patch_embed.proj.weight has shape (768, 54, 16, 16) = 18 RGB calibration
    images stacked on the channel dim. Constructing the model with
    num_calibration=0 silently drops those weights and the model never sees
    the expected [CLS, sample_patches, calib_patches] token layout, which
    collapses downstream probe performance. Default to 18 so the calibration
    branch is preserved.
    """
    if _SITR_REPO not in sys.path:
        sys.path.append(_SITR_REPO)
    warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

    from models.networks import SITR_base  # type: ignore

    base = SITR_base(num_calibration=num_calibration)
    if os.path.exists(_SITR_CHECKPOINT):
        print(f"[SITR] Loading checkpoint {_SITR_CHECKPOINT}  (num_calibration={num_calibration})")
        ckpt = torch.load(_SITR_CHECKPOINT, map_location='cpu')
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        missing, unexpected = base.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[SITR] {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[SITR] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    else:
        print(f"[SITR] WARNING: checkpoint not found at {_SITR_CHECKPOINT}. Using random weights.")
    return base


class _SITRWithZeroCalib(nn.Module):
    """SITR encoder wrapper that supplies a constant zero-impression calibration.

    Without any per-sensor calibration data, we feed the model 18 "blank
    calibration" frames — i.e. `(ref - ref) = 0` in pixel space, then
    SITR-normalized to `(0 - mean) / std`. Each calibration "image" becomes
    a constant `-mean/std ≈ (0.10, 0.14, 0.13)` per RGB channel, matching
    the model's expected input layout `[B, 54, H, W]` for num_calibration=18.

    This is the natural zero-shot use of SITR when only the no-contact
    reference frame is available — analogous to telling the model "no
    additional calibration impressions, only the reference."
    """

    def __init__(self, base: nn.Module, img_size: int = 224):
        super().__init__()
        self.base = base
        # Per-channel value of (0 - mean) / std for an unpressed calibration.
        # Broadcast to [1, num_calibration*3, H, W].
        mean = torch.tensor(SITR_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(SITR_STD, dtype=torch.float32).view(1, 3, 1, 1)
        per_channel = (-mean / std).repeat(1, base.num_calibration, 1, 1)  # [1, K*3, 1, 1]
        calib = per_channel.expand(1, base.num_calibration * 3, img_size, img_size).contiguous()
        # Register as buffer so it moves with .to(device). Not trainable.
        self.register_buffer('zero_calib', calib, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        c = self.zero_calib.expand(B, -1, -1, -1)
        return self.base.forward_encoder(x, c=c)


def load_sitr_encoder(
    num_calibration: int = 18,
    use_cls_token_only: bool = True,
) -> BaselineEncoderWrapper:
    base = _load_sitr_raw(num_calibration=num_calibration)
    wrapped = _SITRWithZeroCalib(base, img_size=224) if num_calibration > 0 else None

    if wrapped is not None:
        def _forward(x: torch.Tensor) -> torch.Tensor:
            return wrapped(x)
        inner = wrapped
    else:
        def _forward(x: torch.Tensor) -> torch.Tensor:
            return base.forward_encoder(x, c=None)
        inner = base

    return BaselineEncoderWrapper(
        encoder_fn=_forward,
        inner_module=inner,
        use_cls_token_only=use_cls_token_only,
        image_norm=_ImageNorm(SITR_PRE_SCALE, SITR_MEAN, SITR_STD),
    )


# -----------------------------------------------------------------------------
# T3 loader
# -----------------------------------------------------------------------------
# For linear probing we always want the finetune-style ViTEncoder + TransformerTrunk
# (not MAEViTEncoder, which takes mask_ratio and returns masked-token features),
# regardless of whether we then regress force or classify. One config covers both.
_T3_FINETUNE_CONFIG = 'configs/network/finetune_exp_cls.yaml'

_T3_MODALITY_DOMAIN = {
    'gsmini': 'mini',
    '9dtact': 'mini',  # fallback; T3's 'mini' encoder is the closest RGB gel-sensor tower
}


def _resolve_t3_weights_dir(weights_dir: Optional[str]) -> str:
    candidates = [weights_dir] if weights_dir else list(_T3_WEIGHTS_DIR_CANDIDATES)
    for d in candidates:
        if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, 'trunk.pth')):
            return d
    raise FileNotFoundError(
        f"No T3 pretrained components found. Looked in: {candidates}. "
        "Pass t3_weights_dir=<path> pointing at a directory containing "
        "trunk.pth, encoders/<name>.pth, decoders/<name>.pth."
    )


def _load_t3_raw(modality: str, weights_dir: Optional[str] = None):
    if _T3_REPO not in sys.path:
        sys.path.append(_T3_REPO)
    from t3.models import T3  # type: ignore

    cfg_path = os.path.join(_T3_REPO, _T3_FINETUNE_CONFIG)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"T3 config not found: {cfg_path}")

    print(f"[T3] Loading config {cfg_path}")
    network_cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.create({
        'network': network_cfg,
        'encoders': network_cfg.encoders,
        'shared_trunk': network_cfg.shared_trunk,
        'decoders': network_cfg.decoders,
    })

    model = T3(cfg)
    model.eval()

    # Pick encoder domain from modality, or fall back to first available.
    available = list(model.encoders.keys())
    domain = _T3_MODALITY_DOMAIN.get(modality)
    if domain not in available:
        if 'mini' in available:
            domain = 'mini'
        else:
            domain = available[0]
        print(f"[T3] Modality '{modality}' -> encoder domain '{domain}' (available: {available}).")
    else:
        print(f"[T3] Using encoder domain '{domain}'.")

    # set_domains requires a decoder too; pick any decoder the config exposes
    # (we won't actually use it — we just need the trunk's forward).
    decoder_domain = next(iter(model.decoders.keys()))
    model.set_domains(domain, decoder_domain, forward_mode='single_tower')

    # Load pretrained components: encoder/<domain>.pth, trunk.pth, decoders/<*>.pth.
    # Without this, the encoder+trunk have random init and the probe scores ~chance.
    resolved_dir = _resolve_t3_weights_dir(weights_dir)
    print(f"[T3] Loading pretrained components from {resolved_dir}")
    model.load_components(resolved_dir)
    return model, domain


def _resolve_t3_img_norm(
    modality: str,
    override_mean: Optional[Tuple[float, float, float]] = None,
    override_std: Optional[Tuple[float, float, float]] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], str]:
    """Pick the T3 (mean, std) for a given modality and report the source used."""
    if override_mean is not None and override_std is not None:
        return tuple(override_mean), tuple(override_std), 'override'
    stats = T3_IMG_NORM_BY_MODALITY.get(modality)
    if stats is not None:
        return stats['mean'], stats['std'], f'T3 Mini defaults ({modality})'
    return T3_IMAGENET_MEAN, T3_IMAGENET_STD, 'T3 framework fallback (ImageNet)'


def load_t3_encoder(
    task_type: str = 'classification',
    modality: str = 'gsmini',
    use_cls_token_only: bool = True,
    img_norm_mean: Optional[Tuple[float, float, float]] = None,
    img_norm_std: Optional[Tuple[float, float, float]] = None,
    weights_dir: Optional[str] = None,
) -> BaselineEncoderWrapper:
    model, domain = _load_t3_raw(modality=modality, weights_dir=weights_dir)
    encoder_tower = model.encoders[domain]
    trunk = model.trunk

    mean, std, src = _resolve_t3_img_norm(modality, img_norm_mean, img_norm_std)
    print(f"[T3] img_norm source: {src}  mean={mean}  std={std}")

    def _forward(x: torch.Tensor) -> torch.Tensor:
        # Full T3 feature path up to the decoder: encoder -> trunk.
        # Encoder is 3 blocks; trunk is 9 blocks — skipping trunk was the bug that
        # made features near-random. Decoder is dropped; the probe head replaces it.
        tokens = encoder_tower(x)         # [B, N+1, D]
        tokens = trunk(tokens)            # [B, N+1, D] (TransformerTrunk w/ pooling='none')
        # Re-normalize per-token. The trunk's final LayerNorm has tiny learned
        # weights (mean ~0.07), so its output has cross-sample std ~0.003. Probe
        # MLP gradients are too small to learn from features of that magnitude
        # (~5% acc on 20-way cls). T3's own decoder learns to scale it back up;
        # we apply a non-affine LN here so the linear probe sees unit-variance
        # features per token.
        tokens = nn.functional.layer_norm(tokens, (tokens.shape[-1],))
        return tokens

    # Register the full T3 model so its buffers/params move with .to(device).
    return BaselineEncoderWrapper(
        encoder_fn=_forward,
        inner_module=model,
        use_cls_token_only=use_cls_token_only,
        image_norm=_ImageNorm(T3_PRE_SCALE, mean, std),
    )


# -----------------------------------------------------------------------------
# AnyTouch loader
# -----------------------------------------------------------------------------
class _AnyTouchTouchTower(nn.Module):
    """AnyTouch touch tower rebuilt for probing, replicating the official
    `main_probe.py --load_from_align --use_same_patchemb --use_sensor_token`
    path (load_model_from_multi_clip + TactileProbe.emb_forward/touch_forward).

    Input: [B, 3, 224, 224], already ImageNet-normalized by the wrapper.
    Output: [B, 768] CLS-pooled projection (their TAG-probe pooling).
    """

    def __init__(self, ckpt_path: str, clip_config_dir: str, sensor_slot: int):
        super().__init__()
        from transformers import AutoConfig
        from transformers.models.clip.modeling_clip import CLIPVisionTransformer

        cfg = AutoConfig.from_pretrained(clip_config_dir)
        vcfg = cfg.vision_config
        self.touch_model = CLIPVisionTransformer(vcfg)
        # --use_same_patchemb: video Conv3d patch embedding replaces the 2D one.
        self.touch_model.embeddings.patch_embedding = nn.Conv3d(
            in_channels=vcfg.num_channels,
            out_channels=vcfg.hidden_size,
            kernel_size=(3, vcfg.patch_size, vcfg.patch_size),
            stride=(3, vcfg.patch_size, vcfg.patch_size),
            bias=False,
        )
        self.sensor_token = nn.Parameter(torch.zeros(10, 5, vcfg.hidden_size))
        self.touch_projection = nn.Linear(vcfg.hidden_size, cfg.projection_dim, bias=False)
        self.sensor_slot = int(sensor_slot)

        # load_model_from_multi_clip mapping: strip 'touch_mae_model.', keep
        # touch_model/touch_projection/sensor_token, map video_patch_embedding
        # onto embeddings.patch_embedding.
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
        new_sd = {}
        for k, v in raw.items():
            if (('touch_model' in k or 'touch_projection' in k or 'sensor_token' in k)
                    and 'sensor_token_proj' not in k):
                new_sd[k.replace('touch_mae_model.', '')] = v
            if 'video_patch_embedding' in k:
                nk = k.replace('touch_mae_model.', '').replace(
                    'video_patch_embedding', 'touch_model.embeddings.patch_embedding')
                new_sd[nk] = v
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        n_loaded = len(new_sd) - len(unexpected)
        print(f"[AnyTouch] loaded {n_loaded} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)}, "
              f"sensor_slot={self.sensor_slot})")
        if len(missing) > 5:
            print(f"[AnyTouch] WARNING many missing keys, e.g. {sorted(missing)[:5]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb_mod = self.touch_model.embeddings
        # Official probe: static image -> unsqueeze(1).repeat(1,3,1,1,1) then the
        # Conv3d tubelet patch embedding. Replicated VERBATIM (incl. the axis
        # layout) so numbers are comparable to the released evaluation.
        x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        patch_embeds = emb_mod.patch_embedding(x.to(emb_mod.patch_embedding.weight.dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)          # [B, 256, D]

        pos_ids = getattr(emb_mod, 'position_ids', None)
        if pos_ids is None:
            pos_ids = torch.arange(
                emb_mod.position_embedding.num_embeddings, device=x.device
            ).unsqueeze(0)
        pos_emb = emb_mod.position_embedding(pos_ids)                    # [1, 257, D]

        embeddings = patch_embeds + pos_emb[:, 1:, :]
        class_embeds = emb_mod.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(B, 1, -1)
        sensor_emb = self.sensor_token[self.sensor_slot].unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)     # [B, 262, D]

        h = self.touch_model.pre_layrnorm(h)
        h = self.touch_model.encoder(inputs_embeds=h, return_dict=True).last_hidden_state
        pooled = self.touch_model.post_layernorm(h[:, 0, :])
        return self.touch_projection(pooled)                             # [B, 768]


def load_anytouch_encoder(
    modality: str = 'gsmini',
    sensor_slot: Optional[int] = None,
    ckpt_path: str = ANYTOUCH_CKPT,
    clip_config_dir: str = ANYTOUCH_CLIP_CONFIG,
) -> BaselineEncoderWrapper:
    if sensor_slot is None:
        if modality not in ANYTOUCH_SENSOR_SLOT:
            raise ValueError(
                f"AnyTouch is optical-only; no sensor slot for '{modality}'.")
        sensor_slot = ANYTOUCH_SENSOR_SLOT[modality]
    tower = _AnyTouchTouchTower(ckpt_path, clip_config_dir, sensor_slot)

    return BaselineEncoderWrapper(
        encoder_fn=tower,
        inner_module=tower,
        use_cls_token_only=True,  # tower already returns [B, D]; flag is inert
        image_norm=_ImageNorm(1.0, ANYTOUCH_MEAN, ANYTOUCH_STD),
    )


# -----------------------------------------------------------------------------
# Unified loader + embedding probe
# -----------------------------------------------------------------------------
def load_baseline_encoder(
    name: str,
    task_type: str = 'classification',
    modality: str = 'gsmini',
    device: Optional[torch.device] = None,
    use_cls_token_only: bool = True,
    **kwargs,
) -> BaselineEncoderWrapper:
    """Dispatch on backbone name; returns a frozen eval-mode wrapper on the given device."""
    name = name.lower()
    if name == 'sitr':
        encoder = load_sitr_encoder(
            num_calibration=kwargs.get('num_calibration', 0),
            use_cls_token_only=use_cls_token_only,
        )
    elif name == 't3':
        encoder = load_t3_encoder(
            task_type=task_type,
            modality=modality,
            use_cls_token_only=use_cls_token_only,
            img_norm_mean=kwargs.get('img_norm_mean'),
            img_norm_std=kwargs.get('img_norm_std'),
            weights_dir=kwargs.get('t3_weights_dir'),
        )
    elif name == 'anytouch':
        encoder = load_anytouch_encoder(
            modality=modality,
            sensor_slot=kwargs.get('sensor_slot'),
        )
    else:
        raise ValueError(f"Unknown baseline backbone '{name}'. Supported: 'sitr', 't3', 'anytouch'.")

    if device is not None:
        encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def probe_embed_info(
    encoder: nn.Module,
    modality: str,
    device: torch.device,
    input_size: int = 224,
    num_frames: int = 2,
) -> Tuple[int, bool]:
    """Forward a dummy batch to discover (embed_dim, encoder_output_is_sequence)."""
    if modality in ('gsmini', '9dtact'):
        dummy = torch.randn(1, num_frames, 3, input_size, input_size, device=device)
    else:
        dummy = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        out = encoder(dummy)
    if isinstance(out, tuple):
        out = out[0]

    if out.dim() == 2:
        return out.shape[-1], False
    # out is [B, T, D] (video+cls) or [B, N, D] (image+tokens) or [B, T, N, D] (video+tokens)
    return out.shape[-1], True


# -----------------------------------------------------------------------------
# Probe runner
# -----------------------------------------------------------------------------
def _backbone_dataloader_flags(backbone: str) -> Tuple[bool, bool]:
    """Return (apply_background_subtraction, apply_image_normalization) matching the backbone.

    Image normalization is always off (the baseline wrapper applies its own mean/std);
    background subtraction is on for SITR (it's pretrained on differenced images) and
    off for T3 (it's pretrained on raw RGB with ImageNet stats).
    """
    name = backbone.lower()
    if name == 'sitr':
        return (SITR_NEEDS_BG_SUBTRACT, False)
    if name == 't3':
        return (T3_NEEDS_BG_SUBTRACT, False)
    if name == 'anytouch':
        return (ANYTOUCH_NEEDS_BG_SUBTRACT, False)
    raise ValueError(f"Unknown baseline backbone '{backbone}'.")


def run_baseline_probe(
    backbone: str,
    task_type: str,
    modality: str,
    probe_config: OmegaConf,
    device: torch.device,
    run_name: Optional[str] = None,
    use_cls_token_only: bool = True,
    apply_background_subtraction: Optional[bool] = None,
    apply_image_normalization: bool = False,
    finetune: bool = False,
    **backbone_kwargs,
) -> Dict[str, Any]:
    """
    Run a probe experiment with a baseline encoder.

    Mirrors run_probe.py's classification/force flow: builds dataloaders directly
    (with backbone-matched preprocessing flags) and calls run_probe_mlp on the
    frozen baseline encoder. The encoder is frozen identically to the native path,
    so results are directly comparable to the linear-probe numbers from run_probe.py.

    Supported task_type: 'classification', 'force', 'sliding'.
    Supported modality:  'gsmini', '9dtact' (sliding requires gsmini).

    apply_background_subtraction: None -> use backbone default (SITR: True, T3: False).
    apply_image_normalization: leave False so the baseline's own normalization applies.
    """
    if task_type not in ('classification', 'force', 'sliding'):
        raise ValueError(f"task_type must be 'classification', 'force', or 'sliding', got '{task_type}'.")
    # Sliding labels exist under {modality}_force_4probe_50each/sliding_labeled
    # for all 4 modalities — 9dtact / gsmini sliding probes are supported by
    # the underlying `_run_sliding` helper as long as the data root resolves.
    if task_type == 'sliding' and modality not in IMAGE_MODALITIES:
        raise ValueError(f"Sliding baseline probe supports only vision modalities {IMAGE_MODALITIES}, got '{modality}'.")
    if modality in TAXEL_MODALITIES:
        raise ValueError(
            f"Baseline backbone '{backbone}' is RGB-image only; modality '{modality}' is taxel. "
            f"Supported modalities: {IMAGE_MODALITIES}."
        )
    if modality not in IMAGE_MODALITIES:
        raise ValueError(f"Unknown modality '{modality}'. Supported: {IMAGE_MODALITIES}.")

    default_bg, _ = _backbone_dataloader_flags(backbone)
    if apply_background_subtraction is None:
        apply_background_subtraction = default_bg

    print("=" * 80)
    print(f"BASELINE PROBE  backbone={backbone}  task={task_type}  modality={modality}")
    print(f"  apply_background_subtraction={apply_background_subtraction}  "
          f"apply_image_normalization={apply_image_normalization}  (baseline applies its own mean/std)")
    print("=" * 80)

    # 1. Encoder
    encoder = load_baseline_encoder(
        name=backbone,
        task_type=task_type,
        modality=modality,
        device=device,
        use_cls_token_only=use_cls_token_only,
        **backbone_kwargs,
    )
    embed_dim, is_seq = probe_embed_info(encoder, modality=modality, device=device)
    print(f"Encoder embed_dim={embed_dim}, output_is_sequence={is_seq}")

    # `load_baseline_encoder` already calls .eval() + requires_grad=False.
    # When finetune is requested, undo that here — run_probe_mlp will then
    # train/eval the backbone alongside the probe head.
    if finetune:
        encoder.train()
        for p in encoder.parameters():
            p.requires_grad = True
        print(f"Baseline encoder UNFROZEN — running full-model finetune.")

    # 2. Run name / writer
    if run_name is None:
        run_name = f"{backbone}_{task_type}_{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = probe_config.logging.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, run_name))

    # 3. LR scheduler passthrough
    lr_scheduler_config = None
    if getattr(probe_config.probe, 'lr_scheduler', None) is not None:
        lr_scheduler_config = OmegaConf.to_container(probe_config.probe.lr_scheduler, resolve=True)

    # 4. Task-specific dataloaders + probe call
    try:
        if task_type == 'classification':
            probe_results = _run_classification(
                encoder=encoder,
                embed_dim=embed_dim,
                is_seq=is_seq,
                modality=modality,
                probe_config=probe_config,
                device=device,
                writer=writer,
                apply_background_subtraction=apply_background_subtraction,
                normalize_images=apply_image_normalization,
                lr_scheduler_config=lr_scheduler_config,
                finetune=finetune,
            )
        elif task_type == 'force':
            probe_results = _run_force(
                encoder=encoder,
                embed_dim=embed_dim,
                is_seq=is_seq,
                modality=modality,
                probe_config=probe_config,
                device=device,
                writer=writer,
                apply_background_subtraction=apply_background_subtraction,
                apply_image_normalization=apply_image_normalization,
                lr_scheduler_config=lr_scheduler_config,
                finetune=finetune,
            )
        else:
            probe_results = _run_sliding(
                encoder=encoder,
                embed_dim=embed_dim,
                is_seq=is_seq,
                modality=modality,
                probe_config=probe_config,
                device=device,
                writer=writer,
                apply_background_subtraction=apply_background_subtraction,
                apply_image_normalization=apply_image_normalization,
                lr_scheduler_config=lr_scheduler_config,
                finetune=finetune,
            )
    finally:
        writer.close()

    return probe_results


def _run_classification(
    *, encoder, embed_dim, is_seq, modality, probe_config, device, writer,
    apply_background_subtraction, normalize_images, lr_scheduler_config,
    finetune=False,
):
    classification_cfg = probe_config.probe.classification
    data_configs = OmegaConf.to_container(classification_cfg.data_configs, resolve=True) \
        if getattr(classification_cfg, 'data_configs', None) is not None else {}
    data_config_path = data_configs.get(modality) or classification_cfg.get('data_config')
    if not data_config_path or not os.path.exists(data_config_path):
        raise FileNotFoundError(
            f"No classification data config for modality '{modality}'. "
            f"Set probe.classification.data_configs.{modality} in probe.yaml."
        )

    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    data_config.setdefault('data', {})
    data_config['data']['batch_size'] = probe_config.probe.batch_size
    data_config['data']['num_workers'] = probe_config.probe.num_workers
    data_config['data']['dataset_split_type'] = 'supervised'
    data_config['data']['train_data_percentage'] = probe_config.probe.probe_data_percentage
    # Baselines handle their own normalization; let caller override if needed.
    data_config['data']['apply_background_subtraction'] = apply_background_subtraction
    data_config['data']['normalize_images'] = normalize_images
    data_config['data']['normalize_tactile'] = False

    train_loader, val_loader, test_loader, train_dataset, _, _ = \
        create_dataloaders_from_config(OmegaConf.create(data_config['data']))

    num_classes = get_num_classes_from_dataset(train_dataset)
    print(f"Classification: num_classes={num_classes}")

    def get_batch_data_fn(batch):
        for key in ('images', 'tactile', 'tactile_img'):
            if key in batch:
                return batch[key].to(device, non_blocking=True)
        raise ValueError(f"Unexpected batch keys {list(batch.keys())}")

    results = run_probe_mlp(
        encoder=encoder,
        shared_trunk=None,
        device=device,
        probe_train_loader=train_loader,
        probe_val_loader=val_loader,
        probe_test_loader=test_loader,
        task_type='classification',
        num_classes=num_classes,
        embed_dim=embed_dim,
        encoder_output_is_sequence=is_seq,
        probe_total_steps=probe_config.probe.probe_total_steps,
        eval_steps=getattr(probe_config.probe, 'eval_steps', None),
        probe_val_interval=getattr(probe_config.probe, 'probe_val_interval', 50),
        probe_lr=float(probe_config.probe.probe_lr),
        probe_seeds=list(probe_config.probe.probe_seeds),
        writer=writer,
        modality=modality,
        get_batch_data_fn=get_batch_data_fn,
        lr_scheduler_config=lr_scheduler_config,
        probe_patience=int(probe_config.probe.get('probe_patience', 5)),
        probe_weight_decay=float(probe_config.probe.get('probe_weight_decay', 1e-4)),
        finetune=finetune,
    )

    if results:
        print(f"\nClassification result: test_acc = "
              f"{results['test_accuracy_mean']:.2f}% ± {results['test_accuracy_std']:.2f}%")
    return results


def _run_force(
    *, encoder, embed_dim, is_seq, modality, probe_config, device, writer,
    apply_background_subtraction, apply_image_normalization, lr_scheduler_config,
    finetune=False,
):
    force_cfg = probe_config.probe.force

    data_roots = OmegaConf.to_container(force_cfg.data_roots, resolve=True)
    config_dir = force_cfg.config_dir
    data_root = data_roots.get(modality, f'data/{modality}_force_4probe_50each/processed')
    random_seed = int(force_cfg.get('data_split_seed') or probe_config.probe.probe_seeds[0])

    common = dict(
        data_root=data_root,
        modality=modality,
        batch_size=int(probe_config.probe.batch_size),
        config_dir=config_dir,
        apply_background_subtraction=apply_background_subtraction,
        apply_image_normalization=apply_image_normalization,
        apply_ref_force_subtraction=True,
        force_clip_min=-20.0,
        force_clip_max=20.0,
        mode_filter=force_cfg.get('mode_filter'),
        compute_friction_mu=force_cfg.get('compute_friction_mu'),
        friction_mu_eps=float(force_cfg.get('friction_mu_eps', 1e-3)),
        labeled_data_root=force_cfg.get('labeled_data_root'),
        load_sliding_labels=force_cfg.get('load_sliding_labels'),
        strict_labeled=bool(force_cfg.get('strict_labeled', False)),
        num_workers=int(probe_config.probe.num_workers),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        timeout=30,
        seed=random_seed,
    )
    train_loader = create_force_dataloaders_4probe(split='train', **common)
    val_loader = create_force_dataloaders_4probe(split='val', **common)
    test_loader = create_force_dataloaders_4probe(split='test', **common)

    force_mean, force_std = load_force_stats_per_dim_4probe(modality, config_dir=config_dir, dims=3)

    def get_batch_data_fn(batch):
        if 'tactile_img' in batch:
            return batch['tactile_img'].to(device, non_blocking=True)
        raise ValueError(f"Force baseline expects 'tactile_img' in batch; got {list(batch.keys())}")

    def get_target_fn(batch):
        return batch['6d_force'][:, :3].to(device, non_blocking=True)

    results = run_probe_mlp(
        encoder=encoder,
        shared_trunk=None,
        device=device,
        probe_train_loader=train_loader,
        probe_val_loader=val_loader,
        probe_test_loader=test_loader,
        task_type='force',
        output_dim=3,
        embed_dim=embed_dim,
        encoder_output_is_sequence=is_seq,
        probe_total_steps=probe_config.probe.probe_total_steps,
        eval_steps=getattr(probe_config.probe, 'eval_steps', None),
        probe_val_interval=getattr(probe_config.probe, 'probe_val_interval', 50),
        probe_lr=float(probe_config.probe.probe_lr),
        probe_seeds=list(probe_config.probe.probe_seeds),
        writer=writer,
        modality=modality,
        get_batch_data_fn=get_batch_data_fn,
        get_target_fn=get_target_fn,
        force_single_sensor=True,
        normalize_targets=True,
        target_mean=force_mean,
        target_std=force_std,
        lr_scheduler_config=lr_scheduler_config,
        probe_patience=int(probe_config.probe.get('probe_patience', 5)),
        probe_weight_decay=float(probe_config.probe.get('probe_weight_decay', 1e-4)),
        finetune=finetune,
    )

    if results:
        print(f"\nForce result: MAE = {results['test_mae_mean']:.4f} ± {results['test_mae_std']:.4f}  "
              f"RMSE = {results['test_rmse_mean']:.4f} ± {results['test_rmse_std']:.4f}")
    return results


def _run_sliding(
    *, encoder, embed_dim, is_seq, modality, probe_config, device, writer,
    apply_background_subtraction, apply_image_normalization, lr_scheduler_config,
    finetune=False,
):
    """
    Sliding bracket probe (gsmini only). Mirrors `_run_force` but builds the
    dataloader with mode_filter='sliding' + load_sliding_labels=True, calls
    run_probe_mlp with task_type='sliding', and applies inverse-frequency class
    weights to CE to handle the ~88/11/2 class imbalance.
    """
    from utils.sliding_labels import compute_sliding_class_weights

    force_cfg = probe_config.probe.force
    data_roots = OmegaConf.to_container(force_cfg.data_roots, resolve=True)
    config_dir = force_cfg.config_dir
    data_root = data_roots.get(modality, f'data/{modality}_force_4probe_50each/processed')
    random_seed = int(force_cfg.get('data_split_seed') or probe_config.probe.probe_seeds[0])

    # Sliding task can only train on episodes that have a matching `.labeled.npz`;
    # static episodes would surface as all -1 labels and CE with ignore_index=-1 → NaN.
    cfg_mode_filter = force_cfg.get('mode_filter')
    if cfg_mode_filter and cfg_mode_filter != 'sliding':
        print(f"[sliding] WARNING: probe.force.mode_filter={cfg_mode_filter} ignored; "
              "forcing mode_filter='sliding' for sliding task.")
    mode_filter = 'sliding'

    common = dict(
        data_root=data_root,
        modality=modality,
        batch_size=int(probe_config.probe.batch_size),
        config_dir=config_dir,
        apply_background_subtraction=apply_background_subtraction,
        apply_image_normalization=apply_image_normalization,
        apply_ref_force_subtraction=True,
        force_clip_min=-20.0,
        force_clip_max=20.0,
        mode_filter=mode_filter,
        compute_friction_mu=force_cfg.get('compute_friction_mu'),
        friction_mu_eps=float(force_cfg.get('friction_mu_eps', 1e-3)),
        labeled_data_root=force_cfg.get('labeled_data_root'),
        load_sliding_labels=True,
        strict_labeled=bool(force_cfg.get('strict_labeled', False)),
        num_workers=int(probe_config.probe.num_workers),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        timeout=30,
        seed=random_seed,
    )
    train_loader = create_force_dataloaders_4probe(split='train', **common)
    val_loader = create_force_dataloaders_4probe(split='val', **common)
    test_loader = create_force_dataloaders_4probe(split='test', **common)

    num_classes = int(force_cfg.get('num_sliding_classes', 3))
    print(f"Sliding classes: {num_classes}")

    # Inverse-frequency class weights from cached `.labeled.npz`.
    labeled_root = force_cfg.get('labeled_data_root') or os.path.join(
        os.path.dirname(data_root), 'sliding_labeled'
    )
    try:
        class_weights = compute_sliding_class_weights(labeled_root, num_classes=num_classes)
        print(f"Sliding class weights (from {labeled_root}): {class_weights.tolist()}")
    except Exception as e:
        print(f"Warning: could not compute class weights ({e}); using uniform weights")
        class_weights = None

    def get_batch_data_fn(batch):
        if 'tactile_img' in batch:
            return batch['tactile_img'].to(device, non_blocking=True)
        raise ValueError(f"Sliding baseline expects 'tactile_img' in batch; got {list(batch.keys())}")

    def get_target_fn(batch):
        if 'sliding_label' not in batch:
            raise ValueError("Sliding baseline expects 'sliding_label' in batch (load_sliding_labels=True).")
        return batch['sliding_label'].to(device, non_blocking=True).long()

    results = run_probe_mlp(
        encoder=encoder,
        shared_trunk=None,
        device=device,
        probe_train_loader=train_loader,
        probe_val_loader=val_loader,
        probe_test_loader=test_loader,
        task_type='sliding',
        num_classes=num_classes,
        embed_dim=embed_dim,
        encoder_output_is_sequence=is_seq,
        probe_total_steps=probe_config.probe.probe_total_steps,
        eval_steps=getattr(probe_config.probe, 'eval_steps', None),
        probe_val_interval=getattr(probe_config.probe, 'probe_val_interval', 50),
        probe_lr=float(probe_config.probe.probe_lr),
        probe_seeds=list(probe_config.probe.probe_seeds),
        writer=writer,
        modality=modality,
        get_batch_data_fn=get_batch_data_fn,
        get_target_fn=get_target_fn,
        lr_scheduler_config=lr_scheduler_config,
        class_weights=class_weights,
        probe_patience=int(probe_config.probe.get('probe_patience', 5)),
        probe_weight_decay=float(probe_config.probe.get('probe_weight_decay', 1e-4)),
        finetune=finetune,
    )

    if results:
        line = f"\nSliding result: acc = {results['test_accuracy_mean']:.2f}% ± {results['test_accuracy_std']:.2f}%"
        if 'test_macro_f1_mean' in results:
            line += f"  | macro-F1 = {results['test_macro_f1_mean']:.2f}% ± {results['test_macro_f1_std']:.2f}%"
        print(line)
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Run probe evaluation with a baseline encoder (SITR / T3).")
    parser.add_argument('--backbone', type=str, required=True, choices=['sitr', 't3', 'anytouch'])
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'force', 'sliding'])
    parser.add_argument('--modality', type=str, default='gsmini', choices=list(IMAGE_MODALITIES))
    parser.add_argument('--probe_config', type=str, default='config/algo/probe.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Override probe.probe_seeds.')
    parser.add_argument('--probe_data_percentage', type=float, default=None)
    parser.add_argument('--use_cls_token_only', action='store_true', default=True)
    parser.add_argument('--no_cls_token_only', dest='use_cls_token_only', action='store_false')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the baseline encoder+trunk alongside the probe head. '
                             'Default: frozen.')
    args = parser.parse_args()

    if not os.path.exists(args.probe_config):
        raise FileNotFoundError(args.probe_config)
    probe_config = OmegaConf.create(yaml.safe_load(open(args.probe_config, 'r')))

    if args.seeds is not None:
        probe_config.probe.probe_seeds = args.seeds
    if args.probe_data_percentage is not None:
        probe_config.probe.probe_data_percentage = args.probe_data_percentage

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    run_baseline_probe(
        backbone=args.backbone,
        task_type=args.task,
        modality=args.modality,
        probe_config=probe_config,
        device=device,
        use_cls_token_only=args.use_cls_token_only,
        finetune=args.finetune,
    )


if __name__ == '__main__':
    _main()
