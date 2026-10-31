"""Training script for U-Net + classification head."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from data_loader import make_dataloaders, CLASS_NAMES
from model import UNetWithClassifier
from utils import CombinedLoss, dice_coefficient, iou_score, classification_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset root")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--save", default="checkpoints/best.pt")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    dices, ious, accs = [], [], []
    with torch.no_grad():
        for x, m, y in loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            seg, cls = model(x)
            dices.append(dice_coefficient(seg, m))
            ious.append(iou_score(seg, m))
            accs.append(classification_accuracy(cls, y))
    return (sum(dices) / len(dices), sum(ious) / len(ious), sum(accs) / len(accs))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training on {device}")

    train_loader, val_loader, test_loader = make_dataloaders(
        args.data, image_size=args.image_size, batch_size=args.batch_size
    )

    model = UNetWithClassifier(num_classes=len(CLASS_NAMES)).to(device)
    optim = Adam(model.parameters(), lr=args.lr)
    loss_fn = CombinedLoss()

    best_dice = 0.0
    Path(os.path.dirname(args.save) or ".").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, m, y in bar:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optim.zero_grad()
            seg, cls = model(x)
            loss = loss_fn(seg, cls, m, y)
            loss.backward()
            optim.step()
            running += float(loss)
            bar.set_postfix(loss=f"{running / max(1, bar.n):.4f}")

        dice, iou, acc = evaluate(model, val_loader, device)
        print(f"[VAL] Epoch {epoch}: Dice={dice:.3f} IoU={iou:.3f} Acc={acc:.3f}")
        if dice > best_dice:
            best_dice = dice
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES}, args.save)
            print(f"  -> saved checkpoint: {args.save}")

    print("[INFO] Final test evaluation:")
    dice, iou, acc = evaluate(model, test_loader, device)
    print(f"  Dice={dice:.3f}  IoU={iou:.3f}  Acc={acc:.3f}")


if __name__ == "__main__":
    main()
