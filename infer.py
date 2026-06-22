"""
Inference on a single image or a directory of images.

Usage:
    python infer.py \
        --config config.yaml \
        --checkpoint /path/to/epoch_0050.pth \
        --input /path/to/image_or_dir \
        [--save_heatmap]
"""

import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(path: str, image_size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(path).convert('RGB')
    return tf(img).unsqueeze(0)   # [1, C, H, W]


@torch.no_grad()
def infer_image(
    model,
    tensor: torch.Tensor,
    center: torch.Tensor,
    device: torch.device,
    image_size: int,
) -> tuple[float, np.ndarray]:
    tensor = tensor.to(device)
    dist_map = model.anomaly_map(tensor, center, output_size=image_size)  # [1, H, W]
    score = float(dist_map.mean().item())
    return score, dist_map[0].cpu().numpy()


def save_heatmap_image(
    img_path: str,
    score_map: np.ndarray,
    out_path: str,
    image_size: int,
) -> None:
    import matplotlib.pyplot as plt
    import cv2
    from src.visualise import make_heatmap_overlay

    img = np.array(Image.open(img_path).convert('RGB').resize((image_size, image_size)))
    overlay = make_heatmap_overlay(img, score_map)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img);     axes[0].set_title('Input');   axes[0].axis('off')
    axes[1].imshow(overlay); axes[1].set_title('Anomaly'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def collect_paths(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    paths = []
    for fpath in sorted(glob.glob(os.path.join(input_path, '**', '*'), recursive=True)):
        if os.path.splitext(fpath)[1].lower() in IMAGE_EXTS:
            paths.append(fpath)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      default='config.yaml')
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--input',       required=True, help='Image or directory')
    parser.add_argument('--save_heatmap', action='store_true')
    parser.add_argument('--heatmap_dir', default='outputs/heatmaps/infer')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = cfg['data']['image_size']

    from src.models.fcdd import build_model
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    center = ckpt['center'].to(device)

    paths = collect_paths(args.input)
    if not paths:
        print(f'No images found at {args.input}')
        return

    if args.save_heatmap:
        os.makedirs(args.heatmap_dir, exist_ok=True)

    print(f'{"Image":<60} {"Score":>10}  {"Prediction":>12}')
    print('-' * 86)

    for img_path in paths:
        tensor = load_image(img_path, image_size)
        score, score_map = infer_image(model, tensor, center, device, image_size)

        # Threshold: determined on val set; placeholder here is the median
        # anomaly/normal boundary — replace with a calibrated value after eval
        prediction = 'ANOMALY' if score > 0.0 else 'NORMAL'
        print(f'{os.path.basename(img_path):<60} {score:>10.4f}  {prediction:>12}')

        if args.save_heatmap:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.heatmap_dir, f'{stem}_heatmap.png')
            save_heatmap_image(img_path, score_map, out_path, image_size)


if __name__ == '__main__':
    main()
