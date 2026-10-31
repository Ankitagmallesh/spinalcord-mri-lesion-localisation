"""Utility functions: losses, metrics, visualisation helpers."""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------- Losses -----------------------------------
class DiceLoss(nn.Module):
    """Soft-Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B,1,H,W) logits ; target: (B,1,H,W) 0/1
        pred = torch.sigmoid(pred)
        pred_f = pred.contiguous().view(pred.size(0), -1)
        targ_f = target.contiguous().view(target.size(0), -1)
        intersection = (pred_f * targ_f).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (pred_f.sum(dim=1) + targ_f.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Dice (segmentation) + Categorical Cross-Entropy (classification)."""

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.seg_w = seg_weight
        self.cls_w = cls_weight

    def forward(self, seg_logits, cls_logits, seg_target, cls_target):
        return self.seg_w * self.dice(seg_logits, seg_target) + self.cls_w * self.ce(cls_logits, cls_target)


# ----------------------------- Metrics ----------------------------------
@torch.no_grad()
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum()
    return float((2.0 * inter + eps) / (pred.sum() + target.sum() + eps))


@torch.no_grad()
def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter + eps) / (union + eps))


@torch.no_grad()
def classification_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == target).float().mean())


# ----------------------------- Visualisation ----------------------------
def visualise_prediction(image, gt_mask, pred_mask, pred_class, class_names, save_path=None):
    """Plot original / GT mask / predicted mask side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.squeeze(), cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(gt_mask.squeeze(), cmap="gray"); axes[1].set_title("GT Mask"); axes[1].axis("off")
    axes[2].imshow(pred_mask.squeeze(), cmap="gray")
    axes[2].set_title(f"Pred Mask\\nClass: {class_names[pred_class]}"); axes[2].axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names, save_path: str = "confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Calculate row sums for percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # avoid division by zero
    cm_perc = cm / row_sums
    
    im = ax.imshow(cm_perc, cmap="Blues", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    # Add text annotations with count only
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_perc[i, j] > 0.5 else "black"
            text = f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
            
    ax.set_xlabel("Predicted Label", fontweight='bold')
    ax.set_ylabel("Actual Label", fontweight='bold')
    ax.set_title("Confusion Matrix", fontweight='bold', pad=15)
    
    # Add a colorbar for the percentages
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Recall (%)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
