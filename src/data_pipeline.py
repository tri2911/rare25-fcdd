"""
Scan raw RARE25 images, print label distribution, and write stratified
train/val/test CSVs to splits_dir.

Usage:
    python src/data_pipeline.py \
        --raw_dir  /path/to/data/raw \
        --splits_dir /path/to/data/splits \
        --val_frac 0.15 --test_frac 0.15 --seed 42
"""

import argparse
import os
import glob
import json
import csv
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

LABEL_MAP = {
    # folder-name → integer label
    'normal': 0, 'healthy': 0, 'ndbe': 0, '0': 0,
    'cancer': 1, 'lesion': 1, 'abnormal': 1, 'polyp': 1, 'neo': 1, '1': 1,
}


def _collect_from_class_dirs(raw_dir: str) -> list[dict]:
    """
    Handles two layouts:
      flat   : raw_dir/<class_name>/<images>
      nested : raw_dir/<center_name>/<class_name>/<images>  (RARE25 format)
    """
    records = []

    for entry in sorted(os.listdir(raw_dir)):
        entry_path = os.path.join(raw_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Is this entry itself a class folder?
        if entry.lower() in LABEL_MAP:
            label = LABEL_MAP[entry.lower()]
            for fpath in sorted(glob.glob(os.path.join(entry_path, '**', '*'), recursive=True)):
                if os.path.splitext(fpath)[1].lower() in IMAGE_EXTS:
                    records.append({'filepath': fpath, 'label': label})
        else:
            # Treat as a center/group folder — look one level deeper for class dirs
            for class_name in sorted(os.listdir(entry_path)):
                class_dir = os.path.join(entry_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                label = LABEL_MAP.get(class_name.lower())
                if label is None:
                    print(f'  [SKIP] Unknown class folder: {entry}/{class_name!r}')
                    continue
                for fpath in sorted(glob.glob(os.path.join(class_dir, '**', '*'), recursive=True)):
                    if os.path.splitext(fpath)[1].lower() in IMAGE_EXTS:
                        records.append({'filepath': fpath, 'label': label})

    return records


def _collect_from_metadata(raw_dir: str) -> list[dict]:
    """HuggingFace datasets with metadata.jsonl or metadata.csv"""
    records = []

    # Try metadata.jsonl first
    meta_jsonl = os.path.join(raw_dir, 'metadata.jsonl')
    meta_csv   = os.path.join(raw_dir, 'metadata.csv')

    if os.path.exists(meta_jsonl):
        with open(meta_jsonl) as f:
            for line in f:
                row = json.loads(line)
                fname = row.get('file_name') or row.get('image')
                raw_label = str(row.get('label', row.get('class', ''))).lower()
                label = LABEL_MAP.get(raw_label)
                if fname and label is not None:
                    records.append({'filepath': os.path.join(raw_dir, fname), 'label': label})
        return records

    if os.path.exists(meta_csv):
        df = pd.read_csv(meta_csv)
        for _, row in df.iterrows():
            fname = row.get('file_name') or row.get('image')
            raw_label = str(row.get('label', row.get('class', ''))).lower()
            label = LABEL_MAP.get(raw_label)
            if fname and label is not None:
                records.append({'filepath': os.path.join(raw_dir, fname), 'label': label})
        return records

    return records


def collect_records(raw_dir: str) -> list[dict]:
    """Auto-detect dataset layout and return list of {filepath, label} dicts."""
    # Try class-directory layout first
    records = _collect_from_class_dirs(raw_dir)
    if records:
        return records

    # Fall back to metadata file
    records = _collect_from_metadata(raw_dir)
    if records:
        return records

    raise RuntimeError(
        f'Could not find images in {raw_dir}. '
        'Expected either class subdirectories (normal/, cancer/) '
        'or a metadata.jsonl / metadata.csv file.'
    )


def make_splits(
    records: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    labels = [r['label'] for r in records]

    train_val, test = train_test_split(
        records, test_size=test_frac, stratify=labels, random_state=seed
    )
    val_frac_adjusted = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_frac_adjusted,
        stratify=[r['label'] for r in train_val],
        random_state=seed,
    )
    return train, val, test


def save_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'label'])
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir',    required=True)
    parser.add_argument('--splits_dir', required=True)
    parser.add_argument('--val_frac',   type=float, default=0.15)
    parser.add_argument('--test_frac',  type=float, default=0.15)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    print(f'Scanning {args.raw_dir} ...')
    records = collect_records(args.raw_dir)
    print(f'Found {len(records)} images')

    dist = Counter(r['label'] for r in records)
    print(f'  Label 0 (normal) : {dist[0]}')
    print(f'  Label 1 (cancer) : {dist[1]}')

    train, val, test = make_splits(records, args.val_frac, args.test_frac, args.seed)
    print(f'\nSplit sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}')

    save_csv(train, os.path.join(args.splits_dir, 'train.csv'))
    save_csv(val,   os.path.join(args.splits_dir, 'val.csv'))
    save_csv(test,  os.path.join(args.splits_dir, 'test.csv'))
    print(f'Splits saved to {args.splits_dir}')


if __name__ == '__main__':
    main()
