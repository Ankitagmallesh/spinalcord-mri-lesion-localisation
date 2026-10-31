import sys
import asyncio
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "ml"))

from server import (
    app,
    combine_classifications,
    detect_lesions,
    preprocess_image,
    validate_mri,
    get_model_comparison,
    get_training_history,
)
from model import UNetWithClassifier, AttentionGate, AttentionUNetWithClassifier, NeuroCordResAttnNet


def test_prefers_rule_based_ms_when_model_is_unconfident():
    model_probs = {"Tumor": 62.0, "MS": 20.0, "Injury": 10.0, "Normal": 8.0}
    rule_probs = {"Tumor": 15.0, "MS": 70.0, "Injury": 10.0, "Normal": 5.0}

    label, confidence, probs = combine_classifications(
        model_probs=model_probs,
        rule_probs=rule_probs,
        model_confidence=62.0,
    )

    assert label == "MS"
    assert confidence >= 50.0
    assert probs["MS"] > probs["Tumor"]


def test_standard_unet_forward():
    """Verify Standard U-Net produces correct segmentation and classification logits."""
    model = UNetWithClassifier(in_channels=1, num_classes=4, base=16)
    x = torch.randn(2, 1, 128, 128)
    seg_logits, cls_logits = model(x)
    assert seg_logits.shape == (2, 1, 128, 128)
    assert cls_logits.shape == (2, 4)


def test_attention_gate():
    """Verify Attention Gate outputs attention weights alpha between 0 and 1."""
    ag = AttentionGate(f_g=64, f_l=64, f_int=32)
    g = torch.randn(2, 64, 16, 16)
    x = torch.randn(2, 64, 16, 16)
    out, alpha = ag(g, x)
    assert out.shape == (2, 64, 16, 16)
    assert alpha.shape == (2, 1, 16, 16)
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()


def test_attention_unet_forward():
    """Verify Attention U-Net executes with attention maps."""
    model = AttentionUNetWithClassifier(in_channels=1, num_classes=4, base=16)
    x = torch.randn(1, 1, 128, 128)
    seg_logits, cls_logits, attn_maps = model(x, return_attention=True)
    assert seg_logits.shape == (1, 1, 128, 128)
    assert cls_logits.shape == (1, 4)
    assert "ag4" in attn_maps and "ag1" in attn_maps


def test_proposed_neurocord_model_forward():
    """Verify Proposed NeuroCord-ResAttnNet executes with Residual-SE and ASPP bottleneck."""
    model = NeuroCordResAttnNet(in_channels=1, num_classes=4, base=16)
    x = torch.randn(1, 1, 128, 128)
    seg_logits, cls_logits = model(x)
    assert seg_logits.shape == (1, 1, 128, 128)
    assert cls_logits.shape == (1, 4)


def test_models_comparison_api():
    """Verify /api/models/comparison returns 3-way model benchmarks."""
    response = asyncio.run(get_model_comparison())
    data = response.model_dump()
    assert "models" in data
    assert len(data["models"]) == 3
    model_ids = [m["id"] for m in data["models"]]
    assert "unet" in model_ids
    assert "attention_unet" in model_ids
    assert "neurocord_resattn" in model_ids
    assert data["recommended_model"] == "NeuroCord-ResAttnNet (Proposed)"


def test_unet_segmentation_lesion_localization_pipeline():
    """Verify U-Net segmentation mask flows correctly into lesion localization."""
    # Synthetic grayscale cord image with bright lesion
    img = np.zeros((128, 128), dtype=np.uint8)
    img[:, 50:78] = 140  # Cord region
    img[60:70, 58:70] = 230  # Hyperintense lesion inside cord

    # Simulated U-Net cord mask
    cord_mask = np.zeros((128, 128), dtype=np.uint8)
    cord_mask[:, 50:78] = 255

    lesion_mask, lesion_count, total_area, std_i, largest, mean_comp = detect_lesions(img, cord_mask)
    assert lesion_count >= 1
    assert total_area > 0.0
    assert lesion_mask.sum() > 0


def test_gradcam_generation():
    """Verify Grad-CAM produces 2D saliency heatmap and overlay."""
    from xai import GradCAM, generate_gradcam_overlay
    model = UNetWithClassifier(in_channels=1, num_classes=4, base=16)
    x = torch.randn(1, 1, 128, 128)
    
    gradcam = GradCAM(model)
    cam = gradcam.generate(x, target_class_idx=0)
    gradcam.remove_hooks()

    assert cam.shape == (128, 128)
    assert cam.min() >= 0.0 and cam.max() <= 1.0

    gray = np.full((128, 128), 120, dtype=np.uint8)
    overlay, stats = generate_gradcam_overlay(gray, cam)
    assert overlay.shape == (128, 128, 3)
    assert "hotspot" in stats
    assert "peak_activation" in stats


def test_shap_attribution_computation():
    """Verify SHAP attribution produces feature importance weights and visual overlay."""
    from xai import compute_shap_attribution
    gray = np.full((128, 128), 100, dtype=np.uint8)
    cord_mask = np.zeros((128, 128), dtype=np.uint8)
    cord_mask[:, 40:88] = 255
    lesion_mask = np.zeros((128, 128), dtype=np.uint8)
    lesion_mask[50:70, 50:70] = 255

    shap_vis, features = compute_shap_attribution(
        gray_img=gray,
        cord_mask=cord_mask,
        lesion_mask=lesion_mask,
        predicted_class="Tumor",
        confidence=95.0,
    )
    assert shap_vis.shape == (128, 128, 3)
    assert "Focal Hyperintensity / Lesion" in features
    assert features["Focal Hyperintensity / Lesion"] > 0.30


def test_training_history_api():
    """Verify /api/training/history returns 30 epoch convergence curves."""
    data = asyncio.run(get_training_history())
    assert "epochs" in data
    assert len(data["epochs"]) >= 25
    first_epoch = data["epochs"][0]
    last_epoch = data["epochs"][-1]
    assert "train_loss" in first_epoch and "val_loss" in first_epoch
    assert "val_dice" in first_epoch and "val_iou" in first_epoch
    # Verify training loss decreased and validation Dice improved over epochs
    assert last_epoch["train_loss"] < first_epoch["train_loss"]
    assert last_epoch["val_dice"] > first_epoch["val_dice"]


if __name__ == "__main__":
    tests = [
        test_prefers_rule_based_ms_when_model_is_unconfident,
        test_standard_unet_forward,
        test_attention_gate,
        test_attention_unet_forward,
        test_proposed_neurocord_model_forward,
        test_models_comparison_api,
        test_training_history_api,
        test_unet_segmentation_lesion_localization_pipeline,
        test_gradcam_generation,
        test_shap_attribution_computation,
    ]
    print(f"\n[RUNNING {len(tests)} BACKEND TESTS]")
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__} -> {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[TEST RESULT] {passed}/{len(tests)} tests passed successfully!\n")
    if passed != len(tests):
        sys.exit(1)
