"""
perspective_correction.py — Perspective transformation and document unwarping.

Maps an arbitrary 4-point quadrilateral document region in an image
to a top-down, flat rectangular scanner-like view using homography.
"""

from typing import Tuple
import cv2
import numpy as np
from modules.document_detection import order_points


def compute_output_dimensions(rect: np.ndarray) -> Tuple[int, int]:
    """Compute the true physical width and height for the warped document
    from the Euclidean distances between the 4 ordered corners.

    Args:
        rect: Ordered (4, 2) corner points [TL, TR, BR, BL].

    Returns:
        (max_width, max_height) in integer pixels.
    """
    (tl, tr, br, bl) = rect

    # Width: max of top edge length and bottom edge length
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    # Height: max of right edge length and left edge length
    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = int(max(height_right, height_left))

    # Guard against zero or degenerate dimensions
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)

    return max_width, max_height


def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
    margin_trim_percent: float = 0.0,
) -> np.ndarray:
    """Extract and unwarp a 4-point quadrangular region into a flat rectangle.

    Steps:
    1. Order corner points: TL, TR, BR, BL.
    2. Compute output width and height from corner Euclidean distances.
    3. Construct destination rectangle points: [0,0], [W-1, 0], [W-1, H-1], [0, H-1].
    4. Compute 3x3 perspective transform matrix using cv2.getPerspectiveTransform.
    5. Warp image with cv2.warpPerspective using bicubic interpolation.
    6. Optionally trim a small inner margin (e.g. 1-2%) to ensure no desk edges remain.

    Args:
        image: Original or full-resolution BGR image.
        pts: (4, 2) array of corner coordinates.
        margin_trim_percent: Percentage (0.0 to 5.0) of inner margin to trim.
                             Useful to cut off dark desk boundary remnants.

    Returns:
        Flat, upright BGR document image.
    """
    # 1. Ensure corners are in canonical order
    rect = order_points(pts)

    # 2. Determine target dimensions
    max_w, max_h = compute_output_dimensions(rect)

    # 3. Define target planar coordinates
    dst = np.array(
        [
            [0.0, 0.0],
            [float(max_w - 1), 0.0],
            [float(max_w - 1), float(max_h - 1)],
            [0.0, float(max_h - 1)],
        ],
        dtype=np.float32,
    )

    # 4. Compute the 3x3 projective transformation matrix M
    # In projective geometry, M maps each source corner to its destination corner
    M = cv2.getPerspectiveTransform(rect, dst)

    # 5. Apply the perspective transformation
    # INTER_CUBIC produces smooth text edges and avoids pixelation during resampling
    warped = cv2.warpPerspective(
        image, M, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    # 6. Optional margin trim (cuts off 1-2% of outer border to eliminate edge artifacts)
    if margin_trim_percent > 0.0:
        trim_x = int(max_w * (margin_trim_percent / 100.0))
        trim_y = int(max_h * (margin_trim_percent / 100.0))
        # Ensure we don't trim excessively
        if (max_w - 2 * trim_x) > 50 and (max_h - 2 * trim_y) > 50:
            warped = warped[trim_y : max_h - trim_y, trim_x : max_w - trim_x]

    return warped
