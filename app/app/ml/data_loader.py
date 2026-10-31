"""Dataset loader for spinal-cord MRI segmentation + classification.

Expected directory layout::

    data/
      Tumor/
        images/  *.png
        masks/   *.png
      MS/
        images/  *.png
        masks/   *.png
      Injury/
        ...
      Normal/
        ...

Each ``masks/<file>.png`` is a binary spinal-cord mask aligned with the same
filename in ``images/``. Class is inferred from the parent folder name.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

CLASS_NAMES: List[str] = ["Tumor", "MS", "Injury", "Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class SpinalCordMRIDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 128):
        self.root = Path(root)
        self.image_size = image_size
        self.samples: List[Tuple[Path, Path, int]] = []
        
        self.class_names = []
        if self.root.exists():
            self.class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir() and (d / "images").exists()])
        
        if not self.class_names:
            self.class_names = ["Tumor", "MS", "Injury", "Normal"]
            
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            img_dir = self.root / cls / "images"
            mask_dir = self.root / cls / "masks"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    # If no mask is provided, create a blank — model will still learn classification
                    mask_path = None  # type: ignore
                self.samples.append((img_path, mask_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, cls_idx = self.samples[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0  # Normalize 0..1
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)

        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return img_t, mask_t, torch.tensor(cls_idx, dtype=torch.long)


def make_dataloaders(root: str, image_size: int = 128, batch_size: int = 16,
                     train_ratio: float = 0.70, val_ratio: float = 0.15,
                     num_workers: int = 2, seed: int = 42):
    full = SpinalCordMRIDataset(root, image_size=image_size)
    if len(full) == 0:
        raise RuntimeError(f"No samples found under {root}")
    n = len(full)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=g)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
