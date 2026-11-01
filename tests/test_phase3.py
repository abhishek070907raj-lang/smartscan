"""
test_phase3.py — Test suite for shadow removal and image enhancement.

Tests all enhancement modes:
- Original
- Shadow-Free
- Color (Auto White Balance)
- Grayscale
- Black & White (Adaptive Gaussian & Otsu)
- Enhanced (CLAHE + Laplacian Sharpening)
- Comparison: Global Histogram Equalization vs CLAHE
"""

import glob
import os
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from modules.shadow_removal import remove_shadows
from modules.image_enhancement import (
    enhance_document,
    apply_clahe,
    apply_global_hist_eq,
    binarize_document,
    to_grayscale,
)


def run_phase3_tests(
    warped_base: str = "output/phase2_stages",
    output_base: str = "output/phase3_enhancements",
) -> List[Dict[str, Any]]:
    os.makedirs(output_base, exist_ok=True)
    warped_images = sorted(glob.glob(os.path.join(warped_base, "*", "05_warped.jpg")))

    if not warped_images:
        raise FileNotFoundError(f"No warped images found in {warped_base}. Run test_phase2.py first.")

    modes = ["Original", "Shadow-Free", "Color", "Grayscale", "Black & White", "Enhanced"]
    results = []

    print("=" * 95)
    print(f"{'Image Stem':<25} | {'Mode':<18} | {'Output Shape':<14} | {'Time (ms)':<10}")
    print("=" * 95)

    for wpath in warped_images:
        stem = os.path.basename(os.path.dirname(wpath))
        out_dir = os.path.join(output_base, stem)
        os.makedirs(out_dir, exist_ok=True)

        warped = cv2.imread(wpath)
        if warped is None:
            continue

        # 1. Test each standard enhancement mode
        for mode in modes:
            t0 = time.perf_counter()
            enhanced = enhance_document(warped, mode=mode)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            mode_slug = mode.lower().replace(" ", "_").replace("&", "and")
            cv2.imwrite(os.path.join(out_dir, f"{mode_slug}.jpg"), enhanced)

            shape_str = f"{enhanced.shape[1]}x{enhanced.shape[0]}"
            if len(enhanced.shape) == 3:
                shape_str += f"x{enhanced.shape[2]}"
            else:
                shape_str += "x1"

            print(f"{stem:<25} | {mode:<18} | {shape_str:<14} | {elapsed_ms:>8.2f} ms")

            results.append({
                "stem": stem,
                "mode": mode,
                "time_ms": elapsed_ms,
            })

        # 2. Test B&W Otsu vs Adaptive comparison
        gray_sf = to_grayscale(remove_shadows(warped))
        bw_otsu = binarize_document(gray_sf, method="otsu")
        cv2.imwrite(os.path.join(out_dir, "bw_otsu.jpg"), bw_otsu)

        # 3. Test Global Histogram Equalization vs CLAHE comparison
        hist_global = apply_global_hist_eq(warped)
        hist_clahe = apply_clahe(warped, clip_limit=2.0)
        cv2.imwrite(os.path.join(out_dir, "comparison_global_hist_eq.jpg"), hist_global)
        cv2.imwrite(os.path.join(out_dir, "comparison_clahe.jpg"), hist_clahe)

    print("=" * 95)
    print(f"All enhancement modes saved to: {output_base}")
    return results


if __name__ == "__main__":
    run_phase3_tests()
