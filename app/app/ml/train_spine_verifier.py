"""Training script for Spinal Cord MRI Verifier and OOD Rejection Network.

Trains SpinalCordVerifierNet to accurately discriminate between:
  - Valid Spinal Cord MRI scans (Sagittal & Axial)
  - Non-Spinal MRI scans (Brain MRI, Knee MRI, Thoracic/Abdominal, Non-Medical)

Saves the trained model checkpoint to:
  app/app/ml/checkpoints/spine_verifier.pt
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

from spine_verifier import SpinalCordVerifierNet


# ---------------------------------------------------------------------------
# Synthetic Non-Spinal Data Generators for Training
# ---------------------------------------------------------------------------
def generate_brain_mri_sample(size: int = 256) -> np.ndarray:
    """Synthesize a realistic Brain MRI axial slice (skull, cortex, ventricles)."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2 + random.randint(-4, 4), size // 2 + random.randint(-4, 4)
    rx, ry = random.randint(75, 95), random.randint(85, 105)

    # 1. Outer skull bone (bright ring)
    cv2.ellipse(img, (cx, cy), (rx + 4, ry + 4), 0, 0, 360, random.randint(180, 240), 3)

    # 2. Brain parenchyma (oval grey matter / white matter)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, random.randint(120, 160), -1)

    # 3. Interhemispheric fissure (vertical division)
    cv2.line(img, (cx, cy - ry + 10), (cx, cy + ry - 10), random.randint(30, 60), 2)

    # 4. Bilateral Lateral Ventricles (butterfly/C-shaped CSF cavities)
    v_offset = random.randint(12, 18)
    v_w, v_h = random.randint(8, 14), random.randint(20, 32)
    # Left ventricle
    cv2.ellipse(img, (cx - v_offset, cy - 5), (v_w, v_h), 15, 0, 360, random.randint(20, 50), -1)
    # Right ventricle
    cv2.ellipse(img, (cx + v_offset, cy - 5), (v_w, v_h), -15, 0, 360, random.randint(20, 50), -1)

    # 5. Cortical gyri / sulcal convolutions
    noise = np.random.normal(0, 12, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generate_knee_mri_sample(size: int = 256) -> np.ndarray:
    """Synthesize a realistic Knee / Joint MRI slice (femur, tibia, joint space)."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2 + random.randint(-5, 5)

    # Femoral condyles (upper bone)
    femur_y = random.randint(60, 90)
    cv2.ellipse(img, (cx - 35, femur_y), (30, 45), 0, 0, 360, random.randint(130, 170), -1)
    cv2.ellipse(img, (cx + 35, femur_y), (30, 45), 0, 0, 360, random.randint(130, 170), -1)

    # Tibial plateau (lower bone)
    tibia_y = random.randint(165, 195)
    cv2.ellipse(img, (cx, tibia_y), (70, 35), 0, 0, 360, random.randint(120, 160), -1)

    # Joint space (dark transverse band with meniscus wedges)
    cv2.line(img, (cx - 75, (femur_y + tibia_y) // 2), (cx + 75, (femur_y + tibia_y) // 2), 20, 6)

    # Add noise & texture
    noise = np.random.normal(0, 10, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generate_other_non_spinal_sample(size: int = 256) -> np.ndarray:
    """Synthesize other non-spinal medical patterns (chest cavity or random textures)."""
    img = np.zeros((size, size), dtype=np.uint8)
    mode = random.choice(["chest", "texture", "blank"])
    if mode == "chest":
        # Large rounded thorax with bilateral dark lung cavities
        cv2.ellipse(img, (size // 2, size // 2), (105, 90), 0, 0, 360, 140, -1)
        cv2.ellipse(img, (size // 2 - 45, size // 2), (32, 55), 0, 0, 360, 25, -1)
        cv2.ellipse(img, (size // 2 + 45, size // 2), (32, 55), 0, 0, 360, 25, -1)
    elif mode == "texture":
        # Natural texture pattern
        img = np.random.randint(40, 220, (size, size), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)
    else:
        # Uniform or low dynamic range noise
        img = np.random.randint(0, 45, (size, size), dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------
class VerifierDataset(Dataset):
    """Multi-class verification dataset for Spinal Cord MRI validation."""

    def __init__(self, data_root: Path, samples_per_class: int = 150, img_size: int = 256):
        self.img_size = img_size
        self.samples: List[Tuple[np.ndarray, int, int]] = []  # (image, is_spine, modality_idx)

        # 1. Load real Spinal Cord MRI images
        spinal_imgs = []
        if data_root.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in data_root.rglob(ext):
                    if "images" in str(p):
                        spinal_imgs.append(p)

        print(f"[DATA] Found {len(spinal_imgs)} existing spinal cord MRI images.")

        # Modalities:
        # 0: Spinal Cord (Sagittal)
        # 1: Spinal Cord (Axial)
        # 2: Brain MRI
        # 3: Knee / Joint MRI
        # 4: Other Non-Spinal

        # Generate / augment Sagittal Spinal scans (Modality 0)
        target_count = max(samples_per_class, len(spinal_imgs))
        for i in range(target_count):
            if spinal_imgs:
                p = spinal_imgs[i % len(spinal_imgs)]
                gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    gray = cv2.resize(gray, (img_size, img_size))
                    # Augment
                    if random.random() > 0.5:
                        gray = cv2.flip(gray, 1)
                    if random.random() > 0.5:
                        alpha = random.uniform(0.85, 1.25)
                        beta = random.randint(-15, 15)
                        gray = np.clip(gray * alpha + beta, 0, 255).astype(np.uint8)
                    self.samples.append((gray, 1, 0))
                    continue

            # Fallback synthesis if real images not found
            fallback = np.zeros((img_size, img_size), dtype=np.uint8)
            cx = img_size // 2 + random.randint(-10, 10)
            cv2.ellipse(fallback, (cx, img_size // 2), (20, 110), 0, 0, 360, 160, -1)
            self.samples.append((fallback, 1, 0))

        # Axial Spinal scans (Modality 1)
        for _ in range(samples_per_class // 2):
            img = np.zeros((img_size, img_size), dtype=np.uint8)
            cx, cy = img_size // 2 + random.randint(-6, 6), img_size // 2 + random.randint(-6, 6)
            # Vertebral body
            cv2.ellipse(img, (cx, cy - 20), (55, 35), 0, 0, 360, 130, -1)
            # Spinal canal (thecal sac)
            cv2.ellipse(img, (cx, cy + 18), (18, 14), 0, 0, 360, 200, -1)
            # Central cord inside CSF
            cv2.ellipse(img, (cx, cy + 18), (10, 8), 0, 0, 360, 110, -1)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            self.samples.append((img, 1, 1))

        # Brain MRI scans (Modality 2) - Non-spinal
        for _ in range(samples_per_class):
            brain = generate_brain_mri_sample(img_size)
            self.samples.append((brain, 0, 2))

        # Knee / Joint MRI scans (Modality 3) - Non-spinal
        for _ in range(samples_per_class):
            knee = generate_knee_mri_sample(img_size)
            self.samples.append((knee, 0, 3))

        # Other Non-Spinal Scans (Modality 4) - Non-spinal
        for _ in range(samples_per_class // 2):
            other = generate_other_non_spinal_sample(img_size)
            self.samples.append((other, 0, 4))

        random.shuffle(self.samples)
        print(f"[DATA] Total verification dataset size: {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, is_spine, mod_idx = self.samples[idx]
        norm = (img.astype(np.float32) / 255.0)
        t_img = torch.from_numpy(norm).unsqueeze(0)  # (1, H, W)
        t_bin = torch.tensor([float(is_spine)], dtype=torch.float32)
        t_mod = torch.tensor(mod_idx, dtype=torch.long)
        return t_img, t_bin, t_mod


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_verifier(
    data_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: str = "checkpoints/spine_verifier.pt",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Training SpinalCordVerifierNet on: {device}")

    root = Path(data_dir)
    dataset = VerifierDataset(root, samples_per_class=120, img_size=256)

    train_len = int(0.85 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SpinalCordVerifierNet(in_channels=1, num_classes=5).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_bin = 0
        total_samples = 0

        for imgs, bin_labels, mod_labels in train_loader:
            imgs = imgs.to(device)
            bin_labels = bin_labels.to(device)
            mod_labels = mod_labels.to(device)

            optimizer.zero_grad()
            bin_logits, mod_logits = model(imgs)

            loss_bin = bce_loss_fn(bin_logits, bin_labels)
            loss_mod = ce_loss_fn(mod_logits, mod_labels)
            total_loss = loss_bin + 0.8 * loss_mod

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * imgs.size(0)
            preds_bin = (torch.sigmoid(bin_logits) > 0.5).float()
            correct_bin += (preds_bin == bin_labels).sum().item()
            total_samples += imgs.size(0)

        train_acc = correct_bin / max(total_samples, 1)

        # Validation
        model.eval()
        val_correct_bin = 0
        val_correct_mod = 0
        val_total = 0

        with torch.no_grad():
            for imgs, bin_labels, mod_labels in val_loader:
                imgs = imgs.to(device)
                bin_labels = bin_labels.to(device)
                mod_labels = mod_labels.to(device)

                bin_logits, mod_logits = model(imgs)
                preds_bin = (torch.sigmoid(bin_logits) > 0.5).float()
                preds_mod = mod_logits.argmax(dim=1)

                val_correct_bin += (preds_bin == bin_labels).sum().item()
                val_correct_mod += (preds_mod == mod_labels).sum().item()
                val_total += imgs.size(0)

        val_bin_acc = val_correct_bin / max(val_total, 1)
        val_mod_acc = val_correct_mod / max(val_total, 1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss / max(total_samples, 1):.4f} | "
            f"Train Acc: {train_acc * 100:.1f}% | "
            f"Val Spine Acc: {val_bin_acc * 100:.1f}% | "
            f"Val Modality Acc: {val_mod_acc * 100:.1f}%"
        )

        if val_bin_acc >= best_val_acc:
            best_val_acc = val_bin_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_acc": val_bin_acc,
                    "modalities": SpinalCordVerifierNet.MODALITIES,
                },
                save_p,
            )
            print(f"  [SAVED] Checkpoint saved to {save_p} (Val Acc: {val_bin_acc * 100:.1f}%)")

    print(f"\n[DONE] Finished training. Best verification accuracy: {best_val_acc * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spinal Cord MRI Verifier")
    parser.add_argument("--data", default="data", help="Path to spinal cord dataset root")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save", default="checkpoints/spine_verifier.pt", help="Checkpoint save path")
    args = parser.parse_args()

    train_verifier(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save,
    )
