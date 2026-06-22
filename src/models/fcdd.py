"""
FCDD model: CaFormer-S18 backbone + 1×1 projection head.

The projection head maps the 512-channel spatial feature map to rep_dim
channels. The FCDD hypersphere loss operates in this projected space.

  backbone  : CaFormerBackbone → [B, 512, 7, 7]
  proj_head : Conv2d(512, rep_dim, 1) → [B, rep_dim, 7, 7]

Anomaly score per image = mean spatial distance to the hypersphere center,
then bilinearly upsampled to the original resolution for heatmap output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metaformer import CaFormerBackbone


class FCDDModel(nn.Module):
    def __init__(self, backbone: CaFormerBackbone, rep_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.proj_head = nn.Sequential(
            nn.Conv2d(backbone.feature_dim, rep_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(rep_dim),
            nn.ReLU(inplace=True),
        )
        self.rep_dim = rep_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected spatial features [B, rep_dim, H, W]."""
        spatial = self.backbone.forward_spatial(x)    # [B, 512, H, W]
        return self.proj_head(spatial)                # [B, rep_dim, H, W]

    def anomaly_map(
        self,
        x: torch.Tensor,
        center: torch.Tensor,
        output_size: int | None = None,
    ) -> torch.Tensor:
        """
        Return per-pixel anomaly scores [B, H', W'] (optionally upsampled).
        center: [rep_dim] or [rep_dim, 1, 1]
        """
        feats = self.forward(x)                       # [B, rep_dim, H, W]
        c = center.view(1, -1, 1, 1)
        dist = torch.sum((feats - c) ** 2, dim=1)     # [B, H, W]
        if output_size is not None:
            dist = F.interpolate(
                dist.unsqueeze(1),
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        return dist


def build_model(cfg: dict) -> FCDDModel:
    from src.metaformer import build_backbone
    backbone = build_backbone(cfg['model'].get('local_backbone_path'))
    return FCDDModel(backbone, rep_dim=cfg['model']['rep_dim'])
