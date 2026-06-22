"""
Anomaly heatmap generation.

Overlays the FCDD distance map (upsampled to 224×224) as a colour heatmap
onto the original image and saves the result to outputs/heatmaps/.

Usage:
    python src/visualise.py \
        --config config.yaml \
        --drive_root $DRIVE_ROOT \
        --checkpoint /path/to/epoch_0050.pth \
        --split test \
        --n 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import RARE25Dataset, IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """[C, H, W] float → uint8 HWC numpy image."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def make_heatmap_overlay(img_np: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    """Overlay a normalised score_map on img_np as a jet heatmap."""
    score_norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    heat = cv2.applyColorMap((score_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.55, heat, 0.45, 0)
    return overlay


@torch.no_grad()
def save_heatmaps(
    model,
    loader: DataLoader,
    center: torch.Tensor,
    device: torch.device,
    heatmap_dir: str,
    max_images: int,
    image_size: int,
) -> None:
    os.makedirs(heatmap_dir, exist_ok=True)
    model.eval()
    saved = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        dist_maps = model.anomaly_map(images, center, output_size=image_size)  # [B, H, W]

        for i in range(images.shape[0]):
            if saved >= max_images:
                return

            img_np    = denormalize(images[i])
            score_map = dist_maps[i].cpu().numpy()
            overlay   = make_heatmap_overlay(img_np, score_map)
            lbl       = int(labels[i].item())

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np);    axes[0].set_title('Input');        axes[0].axis('off')
            axes[1].imshow(score_map, cmap='jet'); axes[1].set_title('Anomaly Score'); axes[1].axis('off')
            axes[2].imshow(overlay);   axes[2].set_title(f'Overlay (label={lbl})'); axes[2].axis('off')
            plt.tight_layout()

            out_path = os.path.join(heatmap_dir, f'batch{batch_idx:04d}_img{i:02d}_lbl{lbl}.png')
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close()
            saved += 1

    print(f'Saved {saved} heatmaps to {heatmap_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--drive_root', default=None)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n',          type=int, default=50, help='Max heatmaps to save')
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

    from src.models.fcdd import build_model

    ds = RARE25Dataset(
        os.path.join(cfg['data']['splits_dir'], f'{args.split}.csv'),
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    center = ckpt['center'].to(device)

    save_heatmaps(
        model, loader, center, device,
        heatmap_dir=cfg['outputs']['heatmap_dir'],
        max_images=args.n,
        image_size=cfg['data']['image_size'],
    )


if __name__ == '__main__':
    main()
