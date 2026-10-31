"""Evaluate trained model on a directory of images, produce predictions + metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import make_dataloaders, CLASS_NAMES
from model import UNetWithClassifier
from utils import dice_coefficient, iou_score, plot_confusion_matrix, visualise_prediction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--out", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader = make_dataloaders(args.data, image_size=args.image_size, batch_size=1)

    model = UNetWithClassifier(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = [], []
    dices, ious = [], []

    with torch.no_grad():
        for i, (x, m, y) in enumerate(test_loader):
            x, m, y = x.to(device), m.to(device), y.to(device)
            seg, cls = model(x)
            pred_cls = int(cls.argmax(dim=1).item())
            confidence = float(torch.softmax(cls, dim=1).max().item()) * 100.0

            d = dice_coefficient(seg, m); i_ = iou_score(seg, m)
            dices.append(d); ious.append(i_)
            y_true.append(int(y.item())); y_pred.append(pred_cls)

            pred_mask = (torch.sigmoid(seg)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            visualise_prediction(
                x[0].cpu().numpy(), m[0].cpu().numpy(), pred_mask,
                pred_cls, CLASS_NAMES, save_path=str(out_dir / f"pred_{i:03d}.png"),
            )
            print(f"[{i}] pred={CLASS_NAMES[pred_cls]} ({confidence:.1f}%) Dice={d:.3f} IoU={i_:.3f}")

    print("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plot_confusion_matrix(np.array(cm), CLASS_NAMES, save_path=str(out_dir / "confusion_matrix.png"))
    print(f"Mean Dice = {np.mean(dices):.3f}  Mean IoU = {np.mean(ious):.3f}")


if __name__ == "__main__":
    main()
