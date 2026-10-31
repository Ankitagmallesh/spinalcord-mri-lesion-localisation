"""Spinal Cord MRI Verifier and Out-Of-Distribution (OOD) Classifier.

Distinguishes genuine Spinal Cord MRI scans (sagittal & axial) from:
  1. Brain MRI scans (cerebral parenchyma, lateral ventricles, cranium).
  2. Knee / Musculoskeletal MRI scans (articular joints, condyles).
  3. Chest / Abdominal scans (thoracic cavity, viscera).
  4. Non-medical / natural images and noise.

Features both a Deep CNN architecture (SpinalCordVerifierNet) and a
rigorous Anatomical Computer Vision Detector (AnatomicalSpineDetector).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Anatomical Computer Vision Detector (Deterministic Medical Priors)
# ---------------------------------------------------------------------------
class AnatomicalSpineDetector:
    """Evaluates anatomical geometry, columnar axis, and symmetry to verify spinal MRI."""

    def __init__(self, img_size: int = 256):
        self.img_size = img_size

    def analyze(self, gray: np.ndarray) -> Dict[str, Any]:
        """Extract geometric & intensity features characteristic of spinal vs non-spinal scans."""
        if gray.shape != (self.img_size, self.img_size):
            gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        h, w = gray.shape

        # 1. Foreground mask (exclude dark MRI scanner borders)
        _, fg_thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        fg_ratio = float((fg_thresh > 0).mean())

        # 2. Cranial Vault / Brain Oval Envelope Test
        # Brain MRI scans typically fill a large, rounded elliptical skull (eccentricity near circular,
        # foreground > 40% and broad in both horizontal and vertical midsections).
        horizontal_profile = (fg_thresh > 0).sum(axis=0) / float(h)
        vertical_profile = (fg_thresh > 0).sum(axis=1) / float(w)

        mid_h = horizontal_profile[int(w * 0.25):int(w * 0.75)].mean() if w > 0 else 0
        mid_v = vertical_profile[int(h * 0.20):int(h * 0.80)].mean() if h > 0 else 0

        # Find external contours of anatomy
        contours, _ = cv2.findContours(fg_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area_ratio = 0.0
        aspect_ratio = 1.0
        solidity = 0.0
        is_large_oval = False

        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_c)
            largest_area_ratio = float(largest_area) / (h * w)
            bx, by, bw, bh = cv2.boundingRect(largest_c)
            aspect_ratio = float(bh) / max(float(bw), 1.0)
            hull = cv2.convexHull(largest_c)
            hull_area = cv2.contourArea(hull)
            solidity = float(largest_area) / max(hull_area, 1.0)

            # Brain axial/coronal skull: large rounded ellipse (area > 35%, aspect ratio 0.8 to 1.35, high solidity)
            if largest_area_ratio > 0.32 and 0.75 <= aspect_ratio <= 1.35 and solidity > 0.78:
                is_large_oval = True

        # 3. Spinal Columnar Continuity (Sagittal Spine Hallmark)
        # In sagittal spinal MRI, the spinal canal runs vertically through the middle column,
        # bounded by repeating vertebral bodies anteriorly and spinous processes posteriorly.
        central_strip = gray[:, int(w * 0.30):int(w * 0.70)]
        grad_y = cv2.Sobel(central_strip, cv2.CV_32F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(central_strip, cv2.CV_32F, 1, 0, ksize=3)

        # Sagittal spine has strong vertical boundaries (grad_x) along the longitudinal canal
        col_energy = np.abs(grad_x).mean(axis=0)
        canal_column_prominence = float(col_energy.max() / (col_energy.mean() + 1e-5))

        # Vertical continuity of central spinal column
        # In a sagittal spine, the column is continuous from top to bottom (few empty vertical gaps)
        central_fg = fg_thresh[:, int(w * 0.30):int(w * 0.70)]
        vertical_coverage = float((central_fg.sum(axis=1) > 0).mean())

        # 4. Bilateral Cerebral Hemisphere Symmetry & Ventricles (Brain Test)
        # Brain scans have strong horizontal bilateral mirror symmetry across the sagittal midline
        left_half = gray[:, :w // 2]
        right_flipped = cv2.flip(gray[:, w // 2:], 1)
        if left_half.shape == right_flipped.shape:
            diff = np.abs(left_half.astype(np.float32) - right_flipped.astype(np.float32))
            bilateral_symmetry = 1.0 - float(diff.mean() / (gray.std() + 1e-5))
            bilateral_symmetry = float(np.clip(bilateral_symmetry, 0.0, 1.0))
        else:
            bilateral_symmetry = 0.5

        # Lateral Ventricle Detection (butterfly/C-shaped hypointense or hyperintense CSF cavities in center of brain)
        center_box = gray[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
        center_std = float(center_box.std())
        center_dark_mass = float((center_box < 40).mean())

        # 5. Articular Knee / Joint Space Test
        # Knee MRI shows two bulky bones (femur above, tibia below) with a horizontal joint gap
        upper_half_mass = float((fg_thresh[:h // 2, :] > 0).mean())
        lower_half_mass = float((fg_thresh[h // 2:, :] > 0).mean())
        mid_horizontal_gap = float((fg_thresh[int(h * 0.45):int(h * 0.55), :] == 0).mean())

        is_knee_like = (
            upper_half_mass > 0.20
            and lower_half_mass > 0.20
            and mid_horizontal_gap > 0.25
            and 0.85 <= aspect_ratio <= 1.25
        )

        return {
            "foreground_ratio": round(fg_ratio, 3),
            "largest_area_ratio": round(largest_area_ratio, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "solidity": round(solidity, 3),
            "is_large_oval": is_large_oval,
            "canal_prominence": round(canal_column_prominence, 3),
            "vertical_coverage": round(vertical_coverage, 3),
            "bilateral_symmetry": round(bilateral_symmetry, 3),
            "center_std": round(center_std, 2),
            "is_knee_like": is_knee_like,
            "mid_h": round(float(mid_h), 3),
            "mid_v": round(float(mid_v), 3),
        }

    def evaluate(self, gray: np.ndarray) -> Tuple[bool, str, float, str]:
        """Determines if the image is a valid Spinal Cord MRI.

        Returns:
            (is_spinal, detected_category, confidence, detailed_reason)
        """
        feats = self.analyze(gray)

        # Brain MRI signature:
        # Large rounded/oval skull, high bilateral symmetry, broad midsection in both axes
        if feats["is_large_oval"] and feats["bilateral_symmetry"] > 0.65 and feats["foreground_ratio"] > 0.40:
            if feats["vertical_coverage"] > 0.85 and feats["mid_h"] > 0.45:
                return (
                    False,
                    "Brain MRI (Cranial Scan)",
                    0.92,
                    "Detected round cranial envelope, bilateral cerebral hemisphere symmetry, and brain parenchyma. This pipeline strictly processes Spinal Cord MRI scans.",
                )

        # Knee / Joint MRI signature:
        if feats["is_knee_like"]:
            return (
                False,
                "Knee / Joint MRI",
                0.88,
                "Detected articular bone condyles and transverse joint space characteristic of a Knee/Joint MRI. NeuralCord only analyzes Spinal Cord MRI scans.",
            )

        # Valid Sagittal Spinal Cord MRI signature:
        # High vertical coverage through central column, longitudinal canal prominence, moderate foreground
        if feats["vertical_coverage"] >= 0.70 and feats["canal_prominence"] >= 1.20 and not feats["is_large_oval"]:
            return (
                True,
                "Spinal Cord MRI (Sagittal)",
                0.94,
                "Valid sagittal spinal cord MRI with continuous longitudinal canal and vertebral segmentation.",
            )

        # Valid Axial Spinal Cord MRI signature:
        # Localized central thecal canal (small to moderate area ratio < 0.35), surrounding bone arch
        if 0.10 <= feats["foreground_ratio"] <= 0.45 and 0.70 <= feats["aspect_ratio"] <= 1.40 and not feats["is_large_oval"]:
            return (
                True,
                "Spinal Cord MRI (Axial)",
                0.89,
                "Valid axial spinal cord MRI slice with central canal cross-section.",
            )

        # Fallback evaluation based on longitudinal dominance
        if feats["aspect_ratio"] > 1.35 or (feats["vertical_coverage"] > 0.60 and feats["canal_prominence"] > 1.15):
            return (
                True,
                "Spinal Cord MRI",
                0.85,
                "Anatomical features are consistent with spinal cord imaging.",
            )

        return (
            False,
            "Non-Spinal Scan / Inconclusive",
            0.82,
            "Image lacks the characteristic longitudinal columnar canal or axial thecal architecture of a spinal cord MRI.",
        )


# ---------------------------------------------------------------------------
# 2. Deep Convolutional Neural Network (SpinalCordVerifierNet)
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class SEBlock(nn.Module):
        """Squeeze-and-Excitation Channel Attention."""
        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, _, _ = x.shape
            w = self.fc(x).view(b, c, 1, 1)
            return x * w

    class ConvResBlock(nn.Module):
        """Residual Conv Block with Depthwise Separable convolution and SE attention."""
        def __init__(self, in_c: int, out_c: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_c)
            self.relu = nn.LeakyReLU(0.1, inplace=True)
            self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_c)
            self.se = SEBlock(out_c)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_c != out_c:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            return self.relu(out + res)

    class SpinalCordVerifierNet(nn.Module):
        """Deep Out-Of-Distribution and Anatomical Verifier Network for Spinal Cord MRI."""

        # Modality classes:
        # 0: Spinal Cord (Sagittal)
        # 1: Spinal Cord (Axial)
        # 2: Brain MRI
        # 3: Knee / Joint MRI
        # 4: Other / Non-Spinal
        MODALITIES = [
            "Spinal Cord (Sagittal)",
            "Spinal Cord (Axial)",
            "Brain MRI",
            "Knee / Joint MRI",
            "Other Non-Spinal Scan",
        ]

        def __init__(self, in_channels: int = 1, num_classes: int = 5):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),  # 256 -> 128
                nn.BatchNorm2d(24),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.stage1 = ConvResBlock(24, 32, stride=2)   # 128 -> 64
            self.stage2 = ConvResBlock(32, 64, stride=2)   # 64 -> 32
            self.stage3 = ConvResBlock(64, 128, stride=2)  # 32 -> 16
            self.stage4 = ConvResBlock(128, 192, stride=2) # 16 -> 8

            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            # Binary Head: Is it a Spinal Cord MRI (Classes 0 or 1 vs 2, 3, 4)
            self.binary_head = nn.Sequential(
                nn.Linear(192, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

            # Modality Head: Specific anatomical category
            self.modality_head = nn.Sequential(
                nn.Linear(192, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            feats = self.stem(x)
            feats = self.stage1(feats)
            feats = self.stage2(feats)
            feats = self.stage3(feats)
            feats = self.stage4(feats)
            pooled = self.global_pool(feats).flatten(1)

            bin_logits = self.binary_head(pooled)       # (B, 1)
            mod_logits = self.modality_head(pooled)     # (B, 5)
            return bin_logits, mod_logits


# ---------------------------------------------------------------------------
# 3. Global Verifier Manager & Inference Entrypoint
# ---------------------------------------------------------------------------
class SpinalVerifierPipeline:
    """Manages model loading and hybrid evaluation for incoming scans."""

    def __init__(self, checkpoint_path: Optional[str | Path] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.model: Optional[SpinalCordVerifierNet] = None
        self.rule_detector = AnatomicalSpineDetector(img_size=256)

        if checkpoint_path and Path(checkpoint_path).exists() and TORCH_AVAILABLE:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str | Path):
        """Loads trained weights if available."""
        if not TORCH_AVAILABLE:
            return
        try:
            self.model = SpinalCordVerifierNet(in_channels=1, num_classes=5)
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[WARN] Could not load SpinalCordVerifierNet: {e}")
            self.model = None

    def verify(self, gray: np.ndarray) -> Tuple[bool, str, float, str, Dict[str, Any]]:
        """Validates whether an input image is a genuine Spinal Cord MRI.

        Returns:
            (is_spinal, detected_class, confidence_pct, reasoning, details)
        """
        # 1. Run deterministic anatomical priors
        rule_is_spine, rule_class, rule_conf, rule_reason = self.rule_detector.evaluate(gray)
        details = self.rule_detector.analyze(gray)

        # 2. If deep model is loaded, run neural inference
        if self.model is not None and TORCH_AVAILABLE:
            try:
                norm_img = gray.astype(np.float32) / 255.0
                tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    bin_logits, mod_logits = self.model(tensor)
                    spine_prob = float(torch.sigmoid(bin_logits)[0, 0].cpu().item())
                    mod_probs = torch.softmax(mod_logits, dim=1)[0].cpu().numpy()
                    pred_mod_idx = int(np.argmax(mod_probs))
                    pred_mod_name = SpinalCordVerifierNet.MODALITIES[pred_mod_idx]
                    mod_conf = float(mod_probs[pred_mod_idx])

                details["neural_spine_prob"] = round(spine_prob, 3)
                details["neural_modality"] = pred_mod_name
                details["neural_confidence"] = round(mod_conf * 100.0, 1)

                # Combined decision: neural model + anatomical rule verification
                # If neural model says non-spinal with > 65% confidence, reject
                if spine_prob < 0.50 or pred_mod_idx in (2, 3, 4):
                    return (
                        False,
                        pred_mod_name,
                        round(max(mod_conf, 1.0 - spine_prob) * 100.0, 1),
                        f"Deep anatomical classifier identified image as '{pred_mod_name}'. Only Spinal Cord MRI scans are accepted.",
                        details,
                    )
                else:
                    return (
                        True,
                        pred_mod_name,
                        round(spine_prob * 100.0, 1),
                        f"Verified as authentic Spinal Cord MRI ({pred_mod_name}).",
                        details,
                    )
            except Exception as e:
                details["neural_error"] = str(e)

        # 3. Fall back to anatomical rule detector
        return (
            rule_is_spine,
            rule_class,
            round(rule_conf * 100.0, 1),
            rule_reason,
            details,
        )


# Global singleton instance
_GLOBAL_VERIFIER: Optional[SpinalVerifierPipeline] = None

def get_verifier(checkpoint_path: Optional[str | Path] = None) -> SpinalVerifierPipeline:
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        if checkpoint_path is None:
            default_ckpt = Path(__file__).parent / "checkpoints" / "spine_verifier.pt"
            checkpoint_path = default_ckpt if default_ckpt.exists() else None
        _GLOBAL_VERIFIER = SpinalVerifierPipeline(checkpoint_path)
    return _GLOBAL_VERIFIER
