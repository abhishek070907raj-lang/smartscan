"""
utils.py — Shared utility functions for the SmartScan pipeline.

This module provides image loading, EXIF orientation correction,
resizing helpers, and other common routines used across all modules.

Concept (plain-language):
    When you take a photo with a phone, the camera often stores the image
    in landscape orientation and records a rotation flag inside the EXIF
    metadata. If we don't read and apply that flag, the image will appear
    sideways or upside-down. We fix this first.

    Large images (e.g. 4000×6000 from a phone) are slow to process with
    edge detection and contour finding. We shrink them for processing but
    keep the scale factor so that the final crop is done on the full-resolution
    image (better quality).

Java comparison:
    In Java you'd use BufferedImage + ImageIO + AffineTransformOp. Here,
    PIL handles the EXIF rotation in one call (exif_transpose), and OpenCV
    does the resizing with cv2.resize(). NumPy arrays replace int[][] pixel
    buffers.
"""

import io
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """Load an image from raw bytes (e.g. from st.file_uploader) into a
    BGR NumPy array, with EXIF orientation automatically applied.

    Steps:
        1. Open with PIL so we can call exif_transpose (fixes phone rotation).
        2. Convert to RGB (PIL default) then to BGR (OpenCV default).
        3. Return as a uint8 NumPy array, shape (H, W, 3).

    Args:
        raw_bytes: The raw file content (JPEG, PNG, etc.).

    Returns:
        A BGR image as a NumPy array.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        # PIL opens the image and reads EXIF metadata
        pil_image = Image.open(io.BytesIO(raw_bytes))

        # exif_transpose rotates/flips the pixel data according to the
        # EXIF Orientation tag, then removes the tag so we don't double-apply.
        # This is the equivalent of reading the EXIF orientation in Java and
        # then using AffineTransformOp to rotate the BufferedImage.
        pil_image = ImageOps.exif_transpose(pil_image)

        # Convert palette or RGBA images to RGB so we always have 3 channels
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # PIL stores pixels as RGB; OpenCV expects BGR, so we swap channels.
        # np.array() creates a contiguous uint8 array from the PIL image.
        rgb_array = np.array(pil_image, dtype=np.uint8)

        # cv2.cvtColor converts colour spaces. COLOR_RGB2BGR swaps R↔B.
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


def resize_for_processing(
    image: np.ndarray,
    max_dimension: int = 1024,
) -> Tuple[np.ndarray, float]:
    """Resize an image so its longest side is at most `max_dimension` pixels.
    Returns the resized image and the scale factor used.

    Why:
        Edge detection, Gaussian blur, and contour finding are O(n) in the
        number of pixels. A 4000×6000 image has 24 million pixels; resizing
        to 1024 max gives roughly 0.5 million — ~50× faster, and the
        detected corners can be mapped back to full resolution by dividing
        by scale_factor.

    Args:
        image: Input BGR image (any size).
        max_dimension: Target maximum for width or height.

    Returns:
        (resized_image, scale_factor) where scale_factor = resized / original.
        If the image is already small enough, returns (image.copy(), 1.0).
    """
    h, w = image.shape[:2]
    # Pick the larger dimension
    longest = max(h, w)

    if longest <= max_dimension:
        # Already small enough — return a copy so caller can modify safely
        return image.copy(), 1.0

    # scale_factor < 1.0 means we are shrinking
    scale_factor = max_dimension / longest

    # cv2.resize takes (width, height) — note the reversed order vs NumPy
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # INTER_AREA is best for shrinking (anti-aliased, avoids moiré patterns).
    # In Java you'd use Image.getScaledInstance(w, h, Image.SCALE_AREA_AVERAGING).
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale_factor


def scale_points(
    points: np.ndarray, scale_factor: float
) -> np.ndarray:
    """Map corner coordinates from the resized image back to the original.

    If we found a document quadrilateral on a resized image (scale_factor
    = 0.25), the actual pixel coordinates are 4× larger on the original.

    Args:
        points: Array of shape (4, 2) — the four corners on the resized image.
        scale_factor: The value returned by resize_for_processing().

    Returns:
        The corners mapped to original-image coordinates, dtype float32.
    """
    if scale_factor == 1.0:
        return points.astype(np.float32)
    return (points / scale_factor).astype(np.float32)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (OpenCV) image to RGB (for display in Streamlit/PIL).

    Streamlit's st.image() expects RGB. OpenCV loads/processes in BGR.
    This is a one-liner but used in many places, so we centralise it.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Clip and cast an image array to uint8 (0-255).

    After arithmetic operations (division, subtraction for shadow removal)
    pixel values may be floats or exceed [0, 255]. This clamps and converts.
    """
    return np.clip(image, 0, 255).astype(np.uint8)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 90, 180, or 270 degrees clockwise (lossless).

    Args:
        image: BGR image array.
        angle: One of 90, 180, 270 (clockwise). Other values return a copy.

    Returns:
        Rotated image.
    """
    if angle == 90:
        # cv2.ROTATE_90_CLOCKWISE = rotate right
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        # 270 clockwise = 90 counter-clockwise
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image.copy()
