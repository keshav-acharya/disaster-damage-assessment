"""
Image preprocessing utilities.

Responsibilities:
  - Validate uploaded image bytes (format + decodability).
  - Resize and normalise images for model input.
  - Verify before/after image pair compatibility.
  - Convert numpy arrays to PyTorch tensors.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_image(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate that *file_bytes* represent a supported, decodable image.

    Returns
    -------
    (True, "OK")                 – image is valid
    (False, "<reason>")          – image is invalid; reason explains why
    """
    # 1. Check file extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in settings.SUPPORTED_FORMATS:
        return False, (
            f"Unsupported file extension '.{ext}'. "
            f"Allowed: {settings.SUPPORTED_FORMATS}"
        )

    # 2. Check size
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, (
            f"File is too large ({len(file_bytes) / 1_048_576:.1f} MB). "
            f"Maximum allowed: {settings.MAX_IMAGE_SIZE_MB} MB."
        )

    # 3. Try to decode
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False, "Could not decode image. The file may be corrupted."

    return True, "OK"


def preprocess_image(
    file_bytes: bytes,
    target_size: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Decode *file_bytes* and return a normalised float32 RGB array.

    Parameters
    ----------
    file_bytes  : raw bytes of the uploaded image file
    target_size : (height, width) to resize to; defaults to settings.IMAGE_SIZE

    Returns
    -------
    np.ndarray of shape (H, W, 3), dtype float32, values in [0, 1]

    Raises
    ------
    ValueError  if the bytes cannot be decoded as an image
    """
    if target_size is None:
        target_size = settings.IMAGE_SIZE  # (height, width)

    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Failed to decode image bytes.")

    # Resize  (cv2.resize takes (width, height))
    img_bgr = cv2.resize(
        img_bgr,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Normalise to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0

    logger.debug("Preprocessed image: shape=%s", img_float.shape)
    return img_float


def validate_image_pair(
    before_array: np.ndarray,
    after_array: np.ndarray,
) -> Tuple[bool, str]:
    """
    Confirm that the before/after image pair has matching spatial dimensions.

    Returns
    -------
    (True, "OK")  or  (False, "<mismatch description>")
    """
    if before_array is None or after_array is None:
        return False, "One or both images could not be loaded."

    if before_array.shape[:2] != after_array.shape[:2]:
        return False, (
            f"Image dimensions do not match: "
            f"before={before_array.shape[:2]}, after={after_array.shape[:2]}. "
            "Both images must be the same size after preprocessing."
        )

    return True, "OK"


def prepare_model_input(image_array: np.ndarray) -> torch.Tensor:
    """
    Convert a preprocessed numpy image array to a batched PyTorch tensor.

    Parameters
    ----------
    image_array : np.ndarray of shape (H, W, C), float32, values in [0, 1]

    Returns
    -------
    torch.Tensor of shape (1, C, H, W)
    """
    # (H, W, C)  →  (C, H, W)
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    # Add batch dimension: (1, C, H, W)
    tensor = tensor.unsqueeze(0)
    logger.debug("Model input tensor shape: %s", tuple(tensor.shape))
    return tensor
