"""
Training entry point.

Usage (from repo root in Colab):
    python src/train.py --config config.yaml --drive_root $DRIVE_ROOT
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import RARE25Dataset
from src.models.fcdd import build_model
from src.losses import fcdd_loss, init_center


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='config.yaml')
    p.add_argument('--drive_root', default=None,
                   help='Override all Drive paths in config if provided.')
    return p.parse_args()


def override_paths(cfg: dict, drive_root: str) -> None:
    """Replace /content/drive/MyDrive/rare25-project with drive_root."""
    placeholder = '/content/drive/MyDrive/rare25-project'

    def _replace(obj):
        if isinstance(obj, str):
            return obj.replace(placeholder, drive_root)
        if isinstance(obj, dict):
            return {k: _replace(v) for k, v in obj.items()}
        return obj

    cfg.update(_replace(cfg))


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.drive_root:
        override_paths(cfg, args.drive_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Datasets ────────────────────────────────────────────────────────────
    splits_dir = cfg['data']['splits_dir']
    image_size = cfg['data']['image_size']
    batch_size = cfg['training']['batch_size']
    num_workers = cfg['training']['num_workers']

    train_ds = RARE25Dataset(
        os.path.join(splits_dir, 'train.csv'),
        image_size=image_size,
        augment=True,
    )
    val_ds = RARE25Dataset(
        os.path.join(splits_dir, 'val.csv'),
        image_size=image_size,
        augment=False,
    )

    # FCDD trains on normal samples only; keep all for center init
    normal_indices = [i for i, lbl in enumerate(train_ds.labels) if lbl == 0]
    normal_ds = Subset(train_ds, normal_indices)

    train_loader = DataLoader(
        normal_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    # Full train loader (incl. anomalies) used only for center initialisation
    full_train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    start_epoch = 1
    center = None

    # ── Resume ───────────────────────────────────────────────────────────────
    resume_path = cfg['training'].get('resume')
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        center = ckpt['center'].to(device)
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}: {resume_path}')

    # ── Center initialisation ────────────────────────────────────────────────
    if center is None:
        print('Initialising hypersphere center ...')
        center = init_center(model, full_train_loader, device).to(device)
        print(f'Center norm: {center.norm().item():.4f}')

    # ── Training loop ────────────────────────────────────────────────────────
    ckpt_dir = cfg['outputs']['checkpoint_dir']
    epochs   = cfg['training']['epochs']

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = fcdd_loss(output, center)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validation AUROC ─────────────────────────────────────────────────
        from src.evaluate import compute_auroc
        val_auroc = compute_auroc(model, val_loader, center, device)

        print(
            f'Epoch {epoch:3d}/{epochs} | '
            f'loss {avg_loss:.4f} | val AUROC {val_auroc:.4f}'
        )

        # Save checkpoint every epoch
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pth')
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'center': center.cpu(),
            'val_auroc': val_auroc,
        }, ckpt_path)

    print('Training complete.')


if __name__ == '__main__':
    main()
