import os
import cv2
import numpy as np
from pathlib import Path

CLASSES = ["Tumor", "MS", "Injury", "Normal"]
NUM_SAMPLES = 40
IMG_SIZE = 128

def generate_data(root_dir: str):
    root = Path(root_dir)
    for c in CLASSES:
        (root / c / "images").mkdir(parents=True, exist_ok=True)
        (root / c / "masks").mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    for c in CLASSES:
        for i in range(NUM_SAMPLES):
            # Create a dark background
            img = np.random.randint(0, 30, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

            # Draw a simulated spinal cord
            cx, cy = np.random.randint(50, 78), np.random.randint(20, 108)
            rx, ry = np.random.randint(15, 25), np.random.randint(40, 60)

            # Give Injury some deformities
            if c == "Injury":
                if np.random.rand() > 0.5:
                    cv2.ellipse(img, (cx, cy - 20), (rx, int(ry/2)), 0, 0, 360, 150, -1)
                    cv2.ellipse(mask, (cx, cy - 20), (rx, int(ry/2)), 0, 0, 360, 255, -1)
                    cv2.ellipse(img, (cx+10, cy + 20), (rx-5, int(ry/2)), 0, 0, 360, 150, -1)
                    cv2.ellipse(mask, (cx+10, cy + 20), (rx-5, int(ry/2)), 0, 0, 360, 255, -1)
                else:
                    cv2.ellipse(img, (cx, cy), (rx-5, ry), 0, 0, 360, 150, -1)
                    cv2.ellipse(mask, (cx, cy), (rx-5, ry), 0, 0, 360, 255, -1)
            else:
                cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 150, -1)
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

            # Add lesions
            if c == "Tumor":
                # One large bright spot
                cv2.ellipse(img, (cx, cy), (rx-5, int(ry/2)), 0, 0, 360, 220, -1)
            elif c == "MS":
                # Multiple small bright spots
                for _ in range(np.random.randint(3, 6)):
                    lx, ly = np.random.randint(cx-rx+5, cx+rx-5), np.random.randint(cy-ry+10, cy+ry-10)
                    cv2.circle(img, (lx, ly), np.random.randint(4, 6), 240, -1)
            
            # Smooth out a bit
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Save
            filename = f"{c}_{i:03d}.png"
            cv2.imwrite(str(root / c / "images" / filename), img)
            cv2.imwrite(str(root / c / "masks" / filename), mask)
            
    print(f"Generated {NUM_SAMPLES} samples for each of the {len(CLASSES)} classes in {root_dir}")

if __name__ == "__main__":
    generate_data("data")
