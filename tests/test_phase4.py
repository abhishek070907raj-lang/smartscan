"""
test_phase4.py — Test suite for OCR text extraction and multi-page PDF compilation.

Evaluates:
1. Tesseract auto-configuration and status reporting
2. OCR extraction fidelity, confidence scores, and paragraph structure
3. PDF compilation (native aspect ratio vs ISO A4 fit) for multi-page documents
"""

import glob
import os
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from modules.ocr import configure_tesseract_path, extract_text_and_confidence, preprocess_for_ocr
from modules.pdf_generator import create_multipage_pdf
from modules.image_enhancement import enhance_document


def run_phase4_tests(
    enhancements_base: str = "output/phase3_enhancements",
    output_base: str = "output/phase4_results",
) -> Dict[str, Any]:
    os.makedirs(output_base, exist_ok=True)
    pdf_out_dir = os.path.join(output_base, "pdf")
    ocr_out_dir = os.path.join(output_base, "ocr")
    os.makedirs(pdf_out_dir, exist_ok=True)
    os.makedirs(ocr_out_dir, exist_ok=True)

    print("=" * 85)
    print("PHASE 4: OCR & PDF GENERATION TEST SUITE")
    print("=" * 85)

    # 1. Tesseract Configuration Check
    is_avail, tesseract_msg = configure_tesseract_path()
    print(f"[*] OCR Availability: {is_avail}")
    print(f"[*] Status: {tesseract_msg}")
    print("-" * 85)

    # 2. OCR Testing on key scenarios
    test_stems = [
        "01_clear_plain",
        "03_dark_shadow",
        "06_multipage_page1",
        "06_multipage_page2",
        "06_multipage_page3",
    ]

    print(f"{'Document Stem':<25} | {'Mode':<12} | {'Words':<8} | {'Confidence':<12} | {'Time (ms)':<10}")
    print("-" * 85)

    ocr_results = []
    for stem in test_stems:
        # Test OCR on both Original (raw) and Enhanced versions to show improvement
        for mode in ["original", "enhanced"]:
            img_path = os.path.join(enhancements_base, stem, f"{mode}.jpg")
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            t0 = time.perf_counter()
            res = extract_text_and_confidence(img, preprocess=(mode == "original"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            words = res["word_count"]
            conf = f"{res['mean_confidence']:.1f}%"
            print(f"{stem:<25} | {mode:<12} | {words:<8} | {conf:<12} | {elapsed_ms:>8.2f} ms")

            # Save extracted text to txt file
            txt_path = os.path.join(ocr_out_dir, f"{stem}_{mode}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Document: {stem} (Mode: {mode}) ===\n")
                f.write(f"Confidence: {conf} | Words: {words} | Time: {elapsed_ms:.1f}ms\n\n")
                f.write(res["text"])

            ocr_results.append({
                "stem": stem,
                "mode": mode,
                "words": words,
                "confidence": res["mean_confidence"],
                "time_ms": elapsed_ms,
            })

    print("-" * 85)

    # 3. Multi-page PDF Compilation
    print("\n[*] Testing Multi-Page PDF Generation...")
    multipage_stems = ["06_multipage_page1", "06_multipage_page2", "06_multipage_page3"]
    pages_enhanced = []

    for m_stem in multipage_stems:
        p_path = os.path.join(enhancements_base, m_stem, "enhanced.jpg")
        img = cv2.imread(p_path)
        if img is not None:
            pages_enhanced.append(img)

    if len(pages_enhanced) == 3:
        # A. Compile with natural image dimensions
        pdf_natural_path = os.path.join(pdf_out_dir, "multipage_natural_aspect.pdf")
        bytes_natural = create_multipage_pdf(
            pages_enhanced,
            output_path=pdf_natural_path,
            fit_a4=False,
        )
        print(f" -> Created Natural Sized PDF: {pdf_natural_path} ({len(bytes_natural) / 1024:.1f} KB)")

        # B. Compile with A4 standard layout fit
        pdf_a4_path = os.path.join(pdf_out_dir, "multipage_iso_a4.pdf")
        bytes_a4 = create_multipage_pdf(
            pages_enhanced,
            output_path=pdf_a4_path,
            fit_a4=True,
        )
        print(f" -> Created ISO A4 Fit PDF:    {pdf_a4_path} ({len(bytes_a4) / 1024:.1f} KB)")
    else:
        print(" [!] Missing pages for multi-page PDF generation.")

    print("=" * 85)
    return {"ocr": ocr_results, "status": "success"}


if __name__ == "__main__":
    run_phase4_tests()
