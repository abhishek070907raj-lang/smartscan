"""
image_enhancement.py — Document image enhancement algorithms.

Provides multiple document enhancement modes:
- Original: Unmodified warped image
- Shadow-Free: Morphological illumination compensation
- Color: Illumination-corrected color with Gray-World white balance
- Grayscale: Normalized single-channel reproduction
- Black & White: High-contrast binarization (Adaptive Gaussian or Otsu) with morphological speckle removal
- Enhanced: CLAHE (Contrast-Limited Adaptive Histogram Equalization) on Luminance + Unsharp Masking

Also includes Global Histogram Equalization to demonstrate the technical contrast with CLAHE.
"""

from typing import Tuple
import cv2
import numpy as np
from modules.shadow_removal import remove_shadows, remove_shadows_grayscale
from modules.utils import ensure_uint8


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel 8-bit grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def auto_white_balance(image: np.ndarray) -> np.ndarray:
    """Apply Gray-World algorithm to correct ambient color cast.

    Assumes the average scene reflectance is neutral gray:
    Each channel is scaled by (gray_mean / channel_mean).
    """
    b, g, r = cv2.split(image)
    b_mean = np.mean(b) + 1e-5
    g_mean = np.mean(g) + 1e-5
    r_mean = np.mean(r) + 1e-5
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    b_bal = np.clip(b.astype(np.float32) * (gray_mean / b_mean), 0, 255).astype(np.uint8)
    g_bal = np.clip(g.astype(np.float32) * (gray_mean / g_mean), 0, 255).astype(np.uint8)
    r_bal = np.clip(r.astype(np.float32) * (gray_mean / r_mean), 0, 255).astype(np.uint8)

    return cv2.merge([b_bal, g_bal, r_bal])


def apply_global_hist_eq(image: np.ndarray) -> np.ndarray:
    """Apply classical Global Histogram Equalization.

    For color images, equalization is performed exclusively on the
    Luminance (L) channel in CIELAB space to avoid hue shifting.

    Note: Global HE flattens the cumulative distribution function (CDF)
    globally, which often over-amplifies background noise.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)

    # Convert BGR to CIELAB (L: Luminance 0-255, A: green-red, B: blue-yellow)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Why CLAHE is superior to Global Histogram Equalization:
    1. Operates on local tiles (e.g. 8x8) to adapt to local document contrast.
    2. 'clip_limit' clips the histogram height, redistributing excess counts
       uniformly across all bins, strictly preventing noise amplification.
    3. Bilinear interpolation blends tile boundaries seamlessly.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        return clahe.apply(image)

    # Apply only to Luminance channel in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Enhance text edge definition using a 3x3 Laplacian unsharp kernel.

    Kernel = [[ 0, -1,  0],
              [-1,  5, -1],
              [ 0, -1,  0]]
    """
    kernel = np.array([[0, -1, 0], [-1, 4 + strength, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return ensure_uint8(sharpened)


def binarize_document(
    image: np.ndarray,
    method: str = "adaptive",
    block_size: int = 17,
    c_offset: int = 9,
    clean_specks: bool = True,
) -> np.ndarray:
    """Convert image into clean binary Black & White (0 or 255).

    Args:
        image: Grayscale or BGR image.
        method: 'adaptive' (Gaussian adaptive threshold) or 'otsu' (Otsu's global binarization).
        block_size: Neighborhood size for adaptive threshold (odd integer, e.g. 17).
        c_offset: Constant subtracted from weighted mean in adaptive threshold.
        clean_specks: If True, applies morphological opening (2x2) to clean salt-and-pepper noise.

    Returns:
        Single-channel uint8 binary image (0 = text ink, 255 = clean white paper).
    """
    gray = to_grayscale(image)

    # Ensure block size is odd and >= 3
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    if method == "otsu":
        # Otsu thresholding finds the threshold t minimizing intra-class variance
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Gaussian weighted adaptive thresholding handles localized gradients
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_offset,
        )

    # Clean isolated speckles using morphological opening
    if clean_specks:
        speckle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, speckle_kernel)

    return binary


def enhance_document(
    image: np.ndarray,
    mode: str = "Enhanced",
    bw_method: str = "adaptive",
) -> np.ndarray:
    """Master enhancement function supporting all scanner modes.

    Supported modes:
    - 'Original': Unmodified input
    - 'Shadow-Free': Background-divided illumination compensation
    - 'Color': Shadow-free color with Gray-World white balancing
    - 'Grayscale': Shadow-free converted to crisp 8-bit grayscale
    - 'Black & White': Crisp 1-bit binary with noise speckle elimination
    - 'Enhanced': Shadow-free + CLAHE on Luminance + Laplacian edge sharpening
    """
    mode_clean = mode.strip().lower()

    if mode_clean == "original":
        return image.copy()

    # Base: Eliminate uneven lighting and shadows
    shadow_free = remove_shadows(image)

    if mode_clean in ("shadow-free", "shadow_free"):
        return shadow_free

    if mode_clean == "color":
        return auto_white_balance(shadow_free)

    if mode_clean == "grayscale":
        return to_grayscale(shadow_free)

    if mode_clean in ("black & white", "b&w", "bw", "black_and_white"):
        gray_sf = to_grayscale(shadow_free)
        return binarize_document(gray_sf, method=bw_method)

    if mode_clean == "enhanced":
        # Apply CLAHE to enhance contrast locally, then sharpen ink strokes
        clahe_img = apply_clahe(shadow_free, clip_limit=2.0)
        return sharpen_image(clahe_img, strength=1.0)

    # Default fallback: return shadow_free
    return shadow_free
