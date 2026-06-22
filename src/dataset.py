"""PyTorch Dataset for loading RARE25 images from CSV split files."""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RARE25Dataset(Dataset):
    """Load images from a CSV with columns: filepath, label."""

    def __init__(self, csv_path: str, image_size: int = 224, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        self.transform = build_transform(image_size, augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['filepath']).convert('RGB')
        except Exception:
            # Return a black image so one bad file doesn't kill the whole run
            img = Image.new('RGB', (224, 224), 0)
        return self.transform(img), int(row['label'])

    @property
    def labels(self) -> list[int]:
        return self.df['label'].tolist()
