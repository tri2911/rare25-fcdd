"""
Evaluation: AUROC and PPV (precision at a given recall threshold).

Usage:
    python src/evaluate.py \
        --config config.yaml \
        --drive_root $DRIVE_ROOT \
        --checkpoint /path/to/epoch_0050.pth \
        --split test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve


def compute_anomaly_scores(
    model,
    loader: DataLoader,
    center: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) arrays over all samples in loader."""
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)                          # [B, rep_dim, H, W]
            c = center.view(1, -1, 1, 1).to(device)
            dist = torch.sum((feats - c) ** 2, dim=1)     # [B, H, W]
            scores = dist.mean(dim=[1, 2]).cpu().numpy()   # [B]
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def compute_auroc(
    model,
    loader: DataLoader,
    center: torch.Tensor,
    device: torch.device,
) -> float:
    scores, labels = compute_anomaly_scores(model, loader, center, device)
    return float(roc_auc_score(labels, scores))


def compute_ppv_at_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    target_recall: float = 0.95,
) -> float:
    """PPV (precision) at the threshold that achieves ≥ target_recall."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    # Find the highest precision where recall >= target_recall
    mask = recall >= target_recall
    if not mask.any():
        return float('nan')
    return float(precision[mask].max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--drive_root', default=None)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.drive_root:
        placeholder = '/content/drive/MyDrive/rare25-project'
        def _replace(obj):
            if isinstance(obj, str):
                return obj.replace(placeholder, args.drive_root)
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            return obj
        cfg.update(_replace(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.dataset import RARE25Dataset
    from src.models.fcdd import build_model

    ds = RARE25Dataset(
        os.path.join(cfg['data']['splits_dir'], f'{args.split}.csv'),
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    center = ckpt['center'].to(device)

    scores, labels = compute_anomaly_scores(model, loader, center, device)
    auroc = roc_auc_score(labels, scores)
    ppv   = compute_ppv_at_recall(scores, labels, target_recall=0.95)

    print(f'Split : {args.split}')
    print(f'AUROC : {auroc:.4f}')
    print(f'PPV@0.95 recall : {ppv:.4f}')

    # Save report
    report_dir = cfg['outputs']['report_dir']
    os.makedirs(report_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    report_path = os.path.join(report_dir, f'{ckpt_name}_{args.split}_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'checkpoint: {args.checkpoint}\n')
        f.write(f'split:      {args.split}\n')
        f.write(f'AUROC:      {auroc:.4f}\n')
        f.write(f'PPV@0.95:   {ppv:.4f}\n')
    print(f'Report saved to {report_path}')


if __name__ == '__main__':
    main()
