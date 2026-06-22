"""
CaFormer-S18 backbone with SurgeNet / GastroNet weight loading.

Wraps timm's caformer_s18 and exposes:
  - forward_spatial(x) → [B, 512, 7, 7]   (for FCDD spatial loss)
  - forward(x)         → [B, 512]          (global average pooled)

Weight loading handles the common key-prefix variants found in teacher
checkpoints from knowledge-distillation pipelines.
"""

import torch
import torch.nn as nn
import timm


class CaFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # num_classes=0, global_pool='' → forward_features returns spatial tokens
        self._model = timm.create_model(
            'caformer_s18',
            pretrained=False,
            num_classes=0,
            global_pool='',
        )
        self.feature_dim = 512

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, 512, H, W] spatial feature map (H=W=7 for 224px input)."""
        feats = self._model.forward_features(x)   # [B, H, W, C] or [B, C, H, W]
        if feats.shape[1] != self.feature_dim:
            # timm returns [B, H, W, C] for some MetaFormer variants — permute
            feats = feats.permute(0, 3, 1, 2).contiguous()
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, 512] globally pooled features."""
        feats = self.forward_spatial(x)           # [B, C, H, W]
        return feats.mean(dim=[2, 3])             # [B, C]


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_surgenet_weights(model: CaFormerBackbone, ckpt_path: str) -> None:
    """Load SurgeNet teacher checkpoint into the backbone, tolerating key-prefix variants."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Checkpoints can store state dict under different keys
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

    # Try loading as-is first
    result = model._model.load_state_dict(sd, strict=False)
    if len(result.missing_keys) == 0:
        print('Loaded SurgeNet weights (exact match).')
        return

    # Try common prefixes
    for prefix in ('backbone.', 'model.', 'encoder.', 'module.', 'base_model.'):
        if any(k.startswith(prefix) for k in sd):
            stripped = _strip_prefix(sd, prefix)
            result = model._model.load_state_dict(stripped, strict=False)
            if len(result.missing_keys) == 0:
                print(f'Loaded SurgeNet weights (stripped prefix "{prefix}").')
                return
            # Partial load is still acceptable — backbone layers are what matter
            print(
                f'Partial load (prefix "{prefix}"): '
                f'{len(result.missing_keys)} missing, '
                f'{len(result.unexpected_keys)} unexpected.'
            )
            return

    print(
        f'Warning: could not cleanly match SurgeNet keys. '
        f'Missing: {result.missing_keys[:5]}  Unexpected: {result.unexpected_keys[:5]}'
    )


def build_backbone(local_backbone_path: str | None = None) -> CaFormerBackbone:
    backbone = CaFormerBackbone()
    if local_backbone_path:
        load_surgenet_weights(backbone, local_backbone_path)
    return backbone
