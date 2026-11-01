"""
document_detection.py — Automatic document boundary detection using OpenCV.

Detects the 4 corners of a paper document in an image using:
1. Grayscale conversion and Gaussian smoothing
2. Canny edge detection, then dilation + closing so broken edges join up
3. Contour finding, convex hull, and Ramer-Douglas-Peucker (approxPolyDP)
   simplification tried at several tolerances
4. Sanity checks (area, corner angles, not the whole image frame)
5. Multi-stage fallback chain (fixed Canny -> median Canny -> Otsu paper mask -> full image)
6. Canonical 4-point corner sorting (TL, TR, BR, BL)

Public names (order_points, find_quadrilateral_contour, detect_document_corners)
and the returned method strings are unchanged, so app.py keeps working.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points into [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    - Top-Left has the smallest (x + y); Bottom-Right the largest (x + y).
    - Top-Right has the smallest (y - x); Bottom-Left the largest (y - x).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _corner_angles_ok(quad: np.ndarray, lo: float = 40.0, hi: float = 140.0) -> bool:
    """A real page photographed at an angle has corner angles near 90 degrees.
    Reject thin slivers and bow-ties whose angles are far from that."""
    q = order_points(quad)
    for i in range(4):
        a = q[(i - 1) % 4] - q[i]
        b = q[(i + 1) % 4] - q[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        cos = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        angle = np.degrees(np.arccos(cos))
        if angle < lo or angle > hi:
            return False
    return True


def find_quadrilateral_contour(
    binary_edges: np.ndarray,
    min_area_ratio: float = 0.10,
    max_area_ratio: float = 0.97,
) -> Optional[np.ndarray]:
    """Find the largest plausible 4-sided shape in a binary edge/mask image.

    Why this is more tolerant than a single approxPolyDP(0.02) call:
    a real page outline is rarely a perfect polygon in an edge map (small
    bumps, rounded corners). We take the convex hull first, then try several
    simplification tolerances until the hull collapses to exactly 4 points.

    Args:
        binary_edges: Single-channel uint8 binary image.
        min_area_ratio: Smallest accepted quad area as a fraction of the image.
        max_area_ratio: Largest accepted fraction. A quad covering ~the whole
            frame is usually the image border, not the page.

    Returns:
        Ordered (4, 2) float32 corners, or None.
    """
    h, w = binary_edges.shape[:2]
    total_area = float(h * w)
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    contours, _ = cv2.findContours(binary_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:10]:
        if cv2.contourArea(c) < min_area:
            break

        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)

        for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            quad_area = cv2.contourArea(quad)
            if quad_area < min_area or quad_area > max_area:
                continue
            if not _corner_angles_ok(quad):
                continue
            return order_points(quad)

        # Last resort for this contour: if the hull is nearly a rectangle
        # (just with extra vertices), use its minimum-area bounding rectangle.
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype(np.float32)
        box_area = cv2.contourArea(box)
        if (
            box_area > 0
            and cv2.contourArea(hull) / box_area > 0.90
            and min_area <= box_area <= max_area
        ):
            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            return order_points(box)

    return None


def _join_edges(edges: np.ndarray) -> np.ndarray:
    """Make the page outline one closed loop.

    Dilation thickens thin edges so small gaps overlap; closing then fills
    the remaining gaps. Without this the outline stays 'open' and its
    contour area is tiny, so the page is never found.
    """
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    joined = cv2.dilate(edges, k3, iterations=1)
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, k9)
    return joined


def _erase_text(gray: np.ndarray, size: int = 31) -> np.ndarray:
    """Remove dark text/ink from a grayscale page photo.

    Grayscale closing (dilate then erode) replaces thin dark strokes with the
    brightness of the paper around them, while the large paper/desk boundary
    survives. After this, the only big brightness step left is paper vs desk.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)


def _paper_mask_edges(blur: np.ndarray) -> np.ndarray:
    """Fallback: segment the paper region with Otsu's threshold.

    Text is erased first (see _erase_text), otherwise Otsu would separate ink
    from paper instead of paper from desk. Both polarities are tried (paper
    can be darker than the desk) and the best blob that is not the whole
    frame is kept.
    """
    flat = _erase_text(blur)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    h, w = blur.shape[:2]
    best, best_area = None, 0.0
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(flat, 0, 255, flag + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        area = max(cv2.contourArea(c) for c in cnts)
        if area < 0.97 * h * w and area > best_area:
            best, best_area = mask, area
    if best is None:
        return np.zeros_like(blur)
    return best


def detect_document_corners(
    image: np.ndarray,
    return_stages: bool = False,
) -> Tuple[np.ndarray, str, Optional[Dict[str, np.ndarray]]]:
    """Detect the 4 boundary corners of a document with multi-stage fallback.

    Stages tried in order (method string in brackets):
    1. Canny 75/200                             ['primary_contour']
    2. Median-based / low-threshold Canny       ['fallback_dynamic_canny']
    3. Otsu paper-region mask                   ['fallback_adaptive_thresh']
    4. Whole image as the document              ['fallback_full_image']

    Returns:
        (corners, detection_method, stages_dict_or_None) with corners ordered
        [TL, TR, BR, BL]. stages has 'gray', 'blur', 'edges', 'contour_overlay'.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    stages: Optional[Dict[str, np.ndarray]] = {} if return_stages else None

    # --- Stage 1: fixed-threshold Canny ---
    edges_primary = _join_edges(cv2.Canny(blur, 75, 200))
    corners = find_quadrilateral_contour(edges_primary)
    method = "primary_contour"
    chosen_edges = edges_primary

    # --- Stage 2: other Canny settings ---
    if corners is None:
        # (a) thresholds derived from the image's median brightness
        med = float(np.median(blur))
        lower = int(max(0, 0.66 * med))
        upper = int(min(255, 1.33 * med))
        candidates = [cv2.Canny(blur, lower, upper)]
        # (b) low thresholds on a text-free image: catches pale paper on a
        #     pale desk, where the edge is faint but text no longer interferes
        candidates.append(cv2.Canny(cv2.GaussianBlur(_erase_text(blur), (5, 5), 0), 15, 45))
        for cand in candidates:
            edges_dynamic = _join_edges(cand)
            corners = find_quadrilateral_contour(edges_dynamic)
            if corners is not None:
                method = "fallback_dynamic_canny"
                chosen_edges = edges_dynamic
                break

    # --- Stage 3: Otsu paper mask ---
    if corners is None:
        mask = _paper_mask_edges(blur)
        corners = find_quadrilateral_contour(mask)
        if corners is not None:
            method = "fallback_adaptive_thresh"
            chosen_edges = mask

    # --- Stage 4: whole image ---
    if corners is None:
        method = "fallback_full_image"
        corners = np.array(
            [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
            dtype=np.float32,
        )
        chosen_edges = edges_primary

    if return_stages and stages is not None:
        stages["gray"] = gray
        stages["blur"] = blur
        stages["edges"] = chosen_edges

        overlay = image.copy()
        poly_pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=3)

        corner_names = ["TL", "TR", "BR", "BL"]
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 255, 0)]
        for idx, (pt, name) in enumerate(zip(corners, corner_names)):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (cx, cy), 8, colors[idx], -1)
            cv2.putText(overlay, name, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        stages["contour_overlay"] = overlay

    return corners, method, stages
