"""
Dataset & DataLoader utilities
==============================
Expects this folder layout:

    data/
    ├── train/
    │   ├── benign/       ← class 0
    │   └── malignant/    ← class 1
    ├── val/
    │   ├── benign/
    │   └── malignant/
    └── test/
        ├── benign/
        └── malignant/

Any common image format (jpg, png, tif …) is accepted.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# ── Constants ─────────────────────────────────────────────────────
IMAGE_SIZE   = 224          # pixels (H = W)
MEAN         = [0.485, 0.456, 0.406]   # ImageNet stats — good default
STD          = [0.229, 0.224, 0.225]
CLASS_NAMES  = ["benign", "malignant"]


# ── Transforms ────────────────────────────────────────────────────
def get_train_transform() -> transforms.Compose:
    """Aggressive augmentation to reduce over-fitting on small datasets."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation / test / inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ── Dataset helpers ───────────────────────────────────────────────
def build_datasets(data_root: str) -> dict:
    """
    Return a dict with 'train', 'val', 'test' ImageFolder datasets.
    Missing splits are silently skipped.
    """
    root = Path(data_root)
    splits = {}
    transform_map = {
        "train": get_train_transform(),
        "val":   get_val_transform(),
        "test":  get_val_transform(),
    }
    for split, tfm in transform_map.items():
        split_path = root / split
        if split_path.is_dir():
            splits[split] = datasets.ImageFolder(str(split_path), transform=tfm)
    return splits


def make_balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Return a WeightedRandomSampler that up-samples the minority class,
    so each mini-batch is approximately class-balanced.
    """
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets).float()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    balance_train: bool = True,
    pin_memory: bool = True,
) -> Tuple[dict, dict]:
    """
    Build DataLoader objects for every available split.

    Returns
    -------
    loaders  : dict  {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    datasets : dict  same keys → ImageFolder objects
    """
    ds = build_datasets(data_root)
    loaders: dict = {}

    for split, dataset in ds.items():
        shuffle  = (split == "train") and not balance_train
        sampler  = make_balanced_sampler(dataset) if (split == "train" and balance_train) else None

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(f"[Dataset] {split:5s}  {len(dataset):>6,} images   "
              f"classes={dataset.class_to_idx}")

    return loaders, ds


# ── Single-image inference helper ─────────────────────────────────
def load_single_image(image_path: str) -> torch.Tensor:
    """
    Load one image from disk and return a (1, 3, H, W) tensor
    ready for model inference.
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    tfm = get_val_transform()
    return tfm(img).unsqueeze(0)          # add batch dim


# ── Demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    loaders, datasets_ = build_loaders(data_root, batch_size=8, num_workers=0)
    for split, loader in loaders.items():
        imgs, labels = next(iter(loader))
        print(f"  {split}: batch shape={imgs.shape}  labels={labels.tolist()}")
