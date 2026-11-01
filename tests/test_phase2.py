"""
test_phase2.py — Test script for document detection and perspective correction.

Evaluates all sample images, saves intermediate stages to output/phase2_stages/,
and logs execution timing and detection fallback paths.
"""

import glob
import os
import sys
import time
from typing import Dict, Any, List

# Ensure parent project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from modules.document_detection import detect_document_corners
from modules.perspective_correction import four_point_transform
from modules.utils import resize_for_processing, scale_points


def run_phase2_tests(
    samples_dir: str = "sample_images",
    output_base: str = "output/phase2_stages",
) -> List[Dict[str, Any]]:
    os.makedirs(output_base, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(samples_dir, "*.jpg")))

    if not image_paths:
        raise FileNotFoundError(f"No test images found in {samples_dir}")

    results = []

    print("=" * 85)
    print(f"{'Image Name':<30} | {'Method':<25} | {'Warped Size':<12} | {'Time (ms)':<10}")
    print("=" * 85)

    for path in image_paths:
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]
        out_dir = os.path.join(output_base, stem)
        os.makedirs(out_dir, exist_ok=True)

        original_bgr = cv2.imread(path)
        if original_bgr is None:
            print(f"Failed to load {path}")
            continue

        t0 = time.perf_counter()

        # Step 1: Resize for fast processing while maintaining aspect ratio
        resized, scale_factor = resize_for_processing(original_bgr, max_dimension=1024)

        # Step 2: Detect corners on the resized image (with stages captured)
        corners_resized, method, stages = detect_document_corners(
            resized, return_stages=True
        )

        # Step 3: Scale corner coordinates back to the original full-resolution space
        corners_full = scale_points(corners_resized, scale_factor)

        # Step 4: Perform 4-point perspective warp on the full-resolution image
        warped = four_point_transform(original_bgr, corners_full, margin_trim_percent=1.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Save intermediate images
        if stages:
            cv2.imwrite(os.path.join(out_dir, "01_gray.jpg"), stages["gray"])
            cv2.imwrite(os.path.join(out_dir, "02_blur.jpg"), stages["blur"])
            cv2.imwrite(os.path.join(out_dir, "03_edges.jpg"), stages["edges"])
            cv2.imwrite(os.path.join(out_dir, "04_contour.jpg"), stages["contour_overlay"])
        cv2.imwrite(os.path.join(out_dir, "05_warped.jpg"), warped)

        wh_str = f"{warped.shape[1]}x{warped.shape[0]}"
        print(f"{filename:<30} | {method:<25} | {wh_str:<12} | {elapsed_ms:>8.2f} ms")

        results.append({
            "image": filename,
            "method": method,
            "corners": corners_full.tolist(),
            "warped_w": warped.shape[1],
            "warped_h": warped.shape[0],
            "time_ms": elapsed_ms,
        })

    print("=" * 85)
    return results


if __name__ == "__main__":
    run_phase2_tests()
