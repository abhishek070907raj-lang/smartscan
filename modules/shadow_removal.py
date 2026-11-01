"""
shadow_removal.py — Shadow and non-uniform lighting compensation.

Implements morphological background estimation based on Retinex illumination theory.
Rather than applying an arbitrary linear brightness gain (which saturates paper),
this module estimates the spatially-varying illumination field L(x, y) and divides it out.
"""

from typing import Tuple
import cv2
import numpy as np
from modules.utils import ensure_uint8


def estimate_background_channel(
    channel: np.ndarray,
    dilate_kernel_size: int = 15,
    blur_kernel_size: int = 21,
) -> np.ndarray:
    """Estimate the ambient illumination background for a single image channel.

    Principle:
    1. Morphological Dilation replaces each pixel with the maximum value in its neighborhood.
       Because text ink is dark (low intensity) and paper is bright (high intensity),
       dilation expands the paper brightness over the dark text, erasing the characters.
    2. Median Blur smooths out high-frequency texture and gradient artifacts,
       yielding a smooth, continuous illumination surface.

    Args:
        channel: Single-channel uint8 array (shape H, W).
        dilate_kernel_size: Size of the rectangular structuring element for dilation.
        blur_kernel_size: Size of the median blur filter kernel (must be odd).

    Returns:
        Estimated background illumination field as uint8 array (shape H, W).
    """
    # Structuring element for morphological dilation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size)
    )

    # 1. Dilate to erase dark foreground text
    dilated = cv2.dilate(channel, kernel)

    # 2. Median blur to smooth out residual step transitions
    # Must be an odd positive integer
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    bg_estimate = cv2.medianBlur(dilated, blur_kernel_size)

    return bg_estimate


def remove_shadows(
    image: np.ndarray,
    dilate_kernel_size: int = 15,
    blur_kernel_size: int = 21,
) -> np.ndarray:
    """Remove shadows and correct uneven lighting from a BGR document image.

    Applies illumination division independently across each colour channel:
        Normalized(x, y) = (Image(x, y) / Background(x, y)) * 255

    Args:
        image: BGR document image (uint8).
        dilate_kernel_size: Structuring element size for dilation (default 15).
        blur_kernel_size: Kernel size for median smoothing (default 21).

    Returns:
        Shadow-removed BGR image with uniform white background.
    """
    # Split BGR channels
    channels = cv2.split(image)
    normalized_channels = []

    for ch in channels:
        # Step 1: Estimate channel background illumination
        bg = estimate_background_channel(
            ch,
            dilate_kernel_size=dilate_kernel_size,
            blur_kernel_size=blur_kernel_size,
        )

        # Step 2: Division normalization in float32 space
        # Add epsilon 1.0 to prevent division-by-zero on pure black pixels
        ch_float = ch.astype(np.float32)
        bg_float = bg.astype(np.float32) + 1.0

        # Scale to 255.0 and clip to [0, 255]
        norm = (ch_float / bg_float) * 255.0
        norm_uint8 = ensure_uint8(norm)
        normalized_channels.append(norm_uint8)

    # Merge normalized channels back into BGR image
    return cv2.merge(normalized_channels)


def remove_shadows_grayscale(
    gray: np.ndarray,
    dilate_kernel_size: int = 15,
    blur_kernel_size: int = 21,
) -> np.ndarray:
    """Remove shadows from a single-channel grayscale document."""
    bg = estimate_background_channel(gray, dilate_kernel_size, blur_kernel_size)
    g_float = gray.astype(np.float32)
    bg_float = bg.astype(np.float32) + 1.0
    norm = (g_float / bg_float) * 255.0
    return ensure_uint8(norm)
