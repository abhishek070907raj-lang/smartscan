"""
ocr.py — Optical Character Recognition (OCR) integration via Tesseract.

Provides:
- Graceful detection of Tesseract executable (PATH + standard paths)
- Preprocessing tailored for OCR (high-contrast binarization / CLAHE)
- Text extraction preserving natural line and paragraph hierarchy
- Word count and per-page mean recognition confidence calculations
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pytesseract
from modules.image_enhancement import apply_clahe, binarize_document, to_grayscale
from modules.shadow_removal import remove_shadows


def configure_tesseract_path() -> Tuple[bool, str]:
    """Verify and configure Tesseract OCR path.

    Checks:
    1. System PATH
    2. Standard Windows installation paths

    Returns:
        (is_available, status_or_version_message)
    """
    # 1. Check if already found in PATH or configured
    if shutil.which("tesseract") is not None:
        try:
            ver = pytesseract.get_tesseract_version()
            return True, f"Tesseract v{ver} available on PATH"
        except Exception:
            pass

    # 2. Check standard Windows default directories
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\Tesseract-OCR\tesseract.exe",
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                ver = pytesseract.get_tesseract_version()
                return True, f"Tesseract v{ver} loaded from {path}"
            except Exception as e:
                return False, f"Tesseract found at {path} but failed to run: {e}"

    return False, "Tesseract not found. Please install from https://github.com/UB-Mannheim/tesseract/wiki"


def is_ocr_available() -> bool:
    """Quick boolean check if OCR is operational."""
    available, _ = configure_tesseract_path()
    return available


def preprocess_for_ocr(image: np.ndarray, binarize: bool = False) -> np.ndarray:
    """Preprocess document image to optimize Tesseract character segmentation.

    1. Removes lighting gradients / shadows
    2. Converts to Grayscale
    3. Either applies CLAHE (keeps anti-aliased font edges) or Adaptive Binarization
    """
    shadow_free = remove_shadows(image)
    gray = to_grayscale(shadow_free)

    if binarize:
        # 1-bit high contrast binarization
        return binarize_document(gray, method="adaptive", block_size=15, c_offset=8)

    # Grayscale with CLAHE preserves subpixel font details for modern LSTM Tesseract
    return apply_clahe(gray, clip_limit=2.0)


def extract_text_and_confidence(
    image: np.ndarray,
    lang: str = "eng",
    preprocess: bool = True,
) -> Dict[str, Any]:
    """Extract text from an image with recognition confidence scores and layout structure.

    Args:
        image: BGR or grayscale document image.
        lang: Tesseract language code (default 'eng').
        preprocess: Whether to apply shadow removal & CLAHE before recognition.

    Returns:
        Dictionary containing:
        - 'text': Extracted text with preserved line and paragraph breaks
        - 'mean_confidence': Average confidence (0.0 to 100.0)
        - 'word_count': Total recognized words
        - 'success': Boolean indicator
        - 'message': Status information
    """
    available, status_msg = configure_tesseract_path()
    if not available:
        return {
            "text": "",
            "mean_confidence": 0.0,
            "word_count": 0,
            "success": False,
            "message": status_msg,
        }

    proc_img = preprocess_for_ocr(image) if preprocess else image

    try:
        # image_to_data returns hierarchical layout dictionary:
        # block_num, par_num, line_num, word_num, left, top, width, height, conf, text
        data = pytesseract.image_to_data(
            proc_img,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config="--psm 3",  # PSM 3 = Fully automatic page segmentation
        )

        n_boxes = len(data["text"])
        lines: List[str] = []
        current_line: List[str] = []
        valid_confidences: List[float] = []

        last_block = -1
        last_par = -1
        last_line = -1

        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])

            block_num = data["block_num"][i]
            par_num = data["par_num"][i]
            line_num = data["line_num"][i]

            # Detect line or paragraph changes
            if (
                current_line
                and (line_num != last_line or par_num != last_par or block_num != last_block)
            ):
                lines.append(" ".join(current_line))
                current_line = []
                # If paragraph or block changed, add a blank line break
                if par_num != last_par or block_num != last_block:
                    lines.append("")

            last_block = block_num
            last_par = par_num
            last_line = line_num

            if word:
                current_line.append(word)
                if conf >= 0:
                    valid_confidences.append(conf)

        if current_line:
            lines.append(" ".join(current_line))

        full_text = "\n".join(lines).strip()
        mean_conf = (
            float(np.mean(valid_confidences)) if valid_confidences else 0.0
        )
        word_count = len(valid_confidences)

        return {
            "text": full_text,
            "mean_confidence": round(mean_conf, 2),
            "word_count": word_count,
            "success": True,
            "message": "Text extracted successfully" if word_count > 0 else "No text detected on page",
        }

    except Exception as exc:
        return {
            "text": "",
            "mean_confidence": 0.0,
            "word_count": 0,
            "success": False,
            "message": f"OCR processing failed: {exc}",
        }
