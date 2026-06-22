"""
CaFormer-S18 backbone with SurgeNet / GastroNet weight loading.

Weight-loading strategy (tried in order):
  1. Direct key match
  2. Strip common prefixes (backbone., model., teacher., ...)
  3. Remap original MetaFormer repo keys → timm key names
  4. Fallback: timm ImageNet-22k pretrained caformer_s18
"""

import torch
import torch.nn as nn
import timm


class CaFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = timm.create_model(
            'caformer_s18',
            pretrained=False,
            num_classes=0,
            global_pool='',
        )
        self.feature_dim = 512

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, 512, H, W] spatial feature map (H=W=7 for 224px input)."""
        feats = self._model.forward_features(x)
        if feats.ndim == 4 and feats.shape[1] != self.feature_dim:
            feats = feats.permute(0, 3, 1, 2).contiguous()
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(x).mean(dim=[2, 3])


# ── Key-name utilities ────────────────────────────────────────────────────────

def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _remap_metaformer_to_timm(sd: dict) -> dict:
    """
    Map original MetaFormer repo key names → timm caformer_s18 key names.

    Original repo layout:
      downsample_layers.0.*     stem patch-embedding
      downsample_layers.N.*     inter-stage downsampling (N=1,2,3)
      network.N.K.*             stage N, block K
      norm_pred.*               final norm

    timm layout:
      stem.*
      stages.N.downsample.*     (N = original_N - 1)
      stages.N.blocks.K.*
      norm.*
    """
    new_sd = {}
    for k, v in sd.items():
        # Stem
        if k.startswith('downsample_layers.0.'):
            nk = k.replace('downsample_layers.0.', 'stem.').replace('post_norm', 'norm')
        # Inter-stage downsampling
        # downsample_layers.N (N=1,2,3) lives at the START of stage N in timm → stages.N
        elif k.startswith('downsample_layers.'):
            parts = k.split('.', 2)
            n    = int(parts[1])
            rest = (parts[2] if len(parts) > 2 else '').replace('pre_norm', 'norm')
            nk   = f'stages.{n}.downsample.{rest}'
        # Blocks
        elif k.startswith('network.'):
            parts = k.split('.', 3)
            n, b  = parts[1], parts[2]
            rest  = parts[3] if len(parts) > 3 else ''
            nk    = f'stages.{n}.blocks.{b}.{rest}' if rest else f'stages.{n}.blocks.{b}'
        # Final norm
        elif k.startswith('norm_pred.'):
            nk = k.replace('norm_pred.', 'norm.')
        else:
            nk = k
        new_sd[nk] = v
    return new_sd


def _missing_non_head(result) -> int:
    return len([k for k in result.missing_keys if 'head' not in k])


# ── Weight loading ────────────────────────────────────────────────────────────

def _try_load(model_part: nn.Module, sd: dict, label: str) -> bool:
    """Attempt load; return True on success, False on any error."""
    try:
        result = model_part.load_state_dict(sd, strict=False)
        if _missing_non_head(result) <= 3:
            print(f'Loaded SurgeNet weights ({label}, '
                  f'{_missing_non_head(result)} non-head keys missing).')
            return True
    except RuntimeError:
        pass
    return False


def _load_imagenet_fallback(model: CaFormerBackbone) -> None:
    print('Falling back to timm ImageNet-22k pretrained caformer_s18 ...')
    pretrained = timm.create_model(
        'caformer_s18.sail_in22k_ft_in1k',
        pretrained=True,
        num_classes=0,
        global_pool='',
    )
    model._model.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained
    print('Loaded timm ImageNet pretrained caformer_s18 weights (fallback).')


def load_surgenet_weights(model: CaFormerBackbone, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict):
        sd = (
            ckpt.get('state_dict')
            or ckpt.get('model')
            or ckpt.get('model_state_dict')
            or ckpt.get('teacher')
            or ckpt
        )
    else:
        sd = ckpt

    # 1 — direct load
    if _try_load(model._model, sd, 'exact match'):
        return

    # 2 — strip common prefixes (with and without MetaFormer remap)
    for prefix in ('backbone.', 'model.', 'encoder.', 'module.', 'base_model.', 'teacher.'):
        if any(k.startswith(prefix) for k in sd):
            stripped = _strip_prefix(sd, prefix)
            if _try_load(model._model, stripped, f'prefix="{prefix}"'):
                return
            remapped = _remap_metaformer_to_timm(stripped)
            if _try_load(model._model, remapped, f'prefix="{prefix}" + MetaFormer remap'):
                return

    # 3 — MetaFormer original repo → timm remap on raw sd
    remapped = _remap_metaformer_to_timm(sd)
    if _try_load(model._model, remapped, 'MetaFormer→timm remap'):
        return

    # 4 — fallback: timm ImageNet-22k pretrained
    print('SurgeNet key mapping exhausted all strategies.')
    _load_imagenet_fallback(model)


def build_backbone(local_backbone_path: str | None = None) -> CaFormerBackbone:
    backbone = CaFormerBackbone()
    if local_backbone_path:
        load_surgenet_weights(backbone, local_backbone_path)
    else:
        print('No backbone path provided — using timm ImageNet pretrained caformer_s18.')
        pretrained = timm.create_model(
            'caformer_s18.sail_in22k_ft_in1k',
            pretrained=True,
            num_classes=0,
            global_pool='',
        )
        backbone._model.load_state_dict(pretrained.state_dict(), strict=True)
        del pretrained
    return backbone
