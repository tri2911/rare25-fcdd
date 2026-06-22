"""
FCDD / Deep SVDD hypersphere loss.

  loss = mean_over_batch( mean_over_spatial( ||proj(x) - c||^2 ) )

center initialisation
---------------------
Call `init_center` once before training starts: it runs a forward pass on
all normal training data and sets c = mean(projections).  The center is then
frozen for the rest of training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def fcdd_loss(output: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    output : [B, rep_dim, H, W]
    center : [rep_dim] or [rep_dim, 1, 1]
    returns: scalar loss
    """
    c = center.view(1, -1, 1, 1).to(output.device)
    dist = torch.sum((output - c) ** 2, dim=1)    # [B, H, W]
    return dist.mean()


@torch.no_grad()
def init_center(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 0.1,
) -> torch.Tensor:
    """
    Compute the hypersphere center as the mean of all projected features
    over the normal training set.

    eps: nudge values too close to zero away (prevents degenerate collapse
    where all features map to the center trivially).
    """
    model.eval()
    sum_feats = None
    n = 0

    for images, labels in loader:
        # Only use normal samples (label == 0) for center initialisation
        mask = labels == 0
        if mask.sum() == 0:
            continue
        imgs_normal = images[mask].to(device)
        feats = model(imgs_normal)                 # [B, rep_dim, H, W]
        feats_pooled = feats.mean(dim=[2, 3])      # [B, rep_dim]

        if sum_feats is None:
            sum_feats = feats_pooled.sum(dim=0)
        else:
            sum_feats += feats_pooled.sum(dim=0)
        n += feats_pooled.shape[0]

    if n == 0:
        raise RuntimeError('No normal (label=0) samples found for center init.')

    center = sum_feats / n

    # Nudge near-zero dimensions
    center[(center.abs() < eps) & (center >= 0)] = eps
    center[(center.abs() < eps) & (center < 0)]  = -eps

    model.train()
    return center.cpu()
