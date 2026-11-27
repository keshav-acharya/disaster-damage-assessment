"""
Utility script to generate sample satellite-like test images for demonstration.

Usage:
    cd disaster_assessment
    python generate_test_image.py

This creates sample_images/before_sample.png and sample_images/after_sample.png
which can be uploaded to the Streamlit UI or used in test scripts.
"""

import os
import random

import numpy as np

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Pillow not found. Install with: pip install Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def generate_satellite_image(
    width: int = 512,
    height: int = 512,
    damaged: bool = False,
) -> np.ndarray:
    """
    Generate a synthetic satellite-like image showing buildings on a grid.

    Parameters
    ----------
    width, height : image dimensions in pixels
    damaged       : if True, adds visual damage effects to some buildings

    Returns
    -------
    np.ndarray of shape (H, W, 3), dtype uint8, RGB
    """
    # Background: dark greenish-grey (ground)
    img = np.full((height, width, 3), fill_value=(80, 90, 70), dtype=np.uint8)

    # Add road grid
    for x in range(0, width, 80):
        img[:, max(0, x-2):x+3] = (120, 115, 100)
    for y in range(0, height, 80):
        img[max(0, y-2):y+3, :] = (120, 115, 100)

    # Draw buildings
    buildings = []
    for row in range(1, height // 80):
        for col in range(1, width // 80):
            bx = col * 80 + 10
            by = row * 80 + 10
            bw = random.randint(25, 45)
            bh = random.randint(25, 45)
            if bx + bw < width and by + bh < height:
                buildings.append((bx, by, bw, bh))

    for (bx, by, bw, bh) in buildings:
        if damaged and random.random() < 0.4:
            # Damaged building: rubble colour
            color = (
                random.randint(100, 150),
                random.randint(80, 110),
                random.randint(60, 90),
            )
            img[by : by + bh, bx : bx + bw] = color
            # Scatter debris
            for _ in range(5):
                dx = random.randint(-10, bw + 10)
                dy = random.randint(-10, bh + 10)
                px, py = bx + dx, by + dy
                if 0 <= px < width and 0 <= py < height:
                    img[py, px] = (160, 140, 110)
        else:
            # Intact building: flat roof (grey)
            roof_color = (
                random.randint(140, 200),
                random.randint(140, 200),
                random.randint(140, 200),
            )
            img[by : by + bh, bx : bx + bw] = roof_color

    return img


def save_image(arr: np.ndarray, path: str):
    """Save an RGB numpy array as a PNG file."""
    if PIL_AVAILABLE:
        Image.fromarray(arr).save(path)
    elif CV2_AVAILABLE:
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        raise RuntimeError("Neither Pillow nor OpenCV is available. Cannot save image.")


if __name__ == "__main__":
    out_dir = "sample_images"
    os.makedirs(out_dir, exist_ok=True)

    before_path = os.path.join(out_dir, "before_sample.png")
    after_path = os.path.join(out_dir, "after_sample.png")

    print("Generating sample satellite images ...")

    before_img = generate_satellite_image(damaged=False)
    save_image(before_img, before_path)
    print(f"  ✅ Saved: {before_path}")

    after_img = generate_satellite_image(damaged=True)
    save_image(after_img, after_path)
    print(f"  ✅ Saved: {after_path}")

    print("\nYou can now:")
    print("  1. Upload these to the Streamlit UI")
    print("  2. Use them in the test script (tests/test_api.py)")
    print(f"     – before: {os.path.abspath(before_path)}")
    print(f"     – after:  {os.path.abspath(after_path)}")
