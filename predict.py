import cv2
import numpy as np

# Standard image size agreed upon in the data pipeline
IMG_SIZE = 512

# Damage class codes (0 = background, 1-4 = damage levels)
DAMAGE_CLASSES = {
    1: "no_damage",
    2: "minor",
    3: "major",
    4: "destroyed"
}

def preprocess_image(image_path):
    """Loads an image, converts BGR to RGB, and resizes to 512x512."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}. Check the file path.")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return img_resized

def overlay_damage_mask(post_img_rgb, mask, alpha=0.45):
    """Paints Green, Yellow, Orange, Red on damaged areas."""
    colors = {
        1: (0, 255, 0),     # Green: No damage
        2: (255, 255, 0),   # Yellow: Minor
        3: (255, 165, 0),   # Orange: Major
        4: (255, 0, 0)      # Red: Destroyed
    }
    overlay = post_img_rgb.copy()
    for cls_id, color in colors.items():
        overlay[mask == cls_id] = color
    return cv2.addWeighted(overlay, alpha, post_img_rgb, 1 - alpha, 0)

def count_buildings(mask):
    """Counts how many distinct buildings fall into each damage category."""
    counts = {"no_damage": 0, "minor": 0, "major": 0, "destroyed": 0}
    
    for cls_id, label in DAMAGE_CLASSES.items():
        binary_mask = (mask == cls_id).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary_mask)
        counts[label] = max(0, num_labels - 1)  # Subtract 1 for background
        
    return counts

def _generate_mock_mask():
    """Creates a fake 512x512 mask with colored boxes until the real AI models are ready."""
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (120, 120), 1, -1)    # Green (No damage)
    cv2.rectangle(mask, (150, 80), (220, 150), 2, -1)   # Yellow (Minor)
    cv2.rectangle(mask, (260, 200), (340, 280), 3, -1)  # Orange (Major)
    cv2.rectangle(mask, (380, 350), (460, 430), 4, -1)  # Red (Destroyed)
    return mask

def predict(pre_image_path, post_image_path, use_mock=True):
    """
    Main pipeline function called by the FastAPI backend.
    Takes before/after image paths and returns the colored overlay + building counts.
    """
    pre_img = preprocess_image(pre_image_path)
    post_img = preprocess_image(post_image_path)

    if use_mock:
        damage_mask = _generate_mock_mask()
    else:
        # TODO: Plug in Pahal & Garima's real models here once trained
        raise NotImplementedError("Real AI models are not plugged in yet!")

    overlay_img = overlay_damage_mask(post_img, damage_mask)
    counts = count_buildings(damage_mask)

    return {
        "overlay_image": overlay_img,
        "mask": damage_mask,
        "counts": counts,
        "total_buildings": sum(counts.values())
    }