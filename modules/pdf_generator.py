"""
pdf_generator.py — Multi-page PDF generation from processed document images.

Features:
- Lossless/high-quality PDF compilation using img2pdf
- Respects user-defined page ordering
- Preserves native document aspect ratio without distortion
- Optional ISO A4 page fitting (210 x 297 mm) with automatic centering
- Supports Color, Grayscale, and 1-bit Black & White page modes
"""

import io
from typing import List, Optional, Tuple
import cv2
import img2pdf
import numpy as np
from PIL import Image
from modules.utils import rotate_image, bgr_to_rgb


def encode_image_for_pdf(
    image: np.ndarray,
    rotation: int = 0,
    jpeg_quality: int = 92,
) -> bytes:
    """Prepare and encode a single page into in-memory image bytes.

    Args:
        image: BGR or Grayscale uint8 NumPy array.
        rotation: Clockwise rotation angle (0, 90, 180, 270 degrees).
        jpeg_quality: Quality parameter for JPEG compression (1-100).

    Returns:
        Encoded image bytes ready for img2pdf.
    """
    # 1. Apply user-selected rotation if any
    if rotation in (90, 180, 270):
        image = rotate_image(image, rotation)

    # 2. Check if image is 1-bit Black & White binary (contains only 0 and 255)
    is_binary = len(image.shape) == 2 and np.array_equal(np.unique(image), np.array([0, 255]))

    if is_binary:
        # Binary images are encoded as PNG (lossless, clean typography)
        pil_img = Image.fromarray(image, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    # 3. Handle Color vs Grayscale encoding
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode="L")
    else:
        # OpenCV is BGR; PIL needs RGB
        rgb = bgr_to_rgb(image)
        pil_img = Image.fromarray(rgb, mode="RGB")

    buf = io.BytesIO()
    # JPEG encoding preserves gradients and minimizes PDF file footprint
    pil_img.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
    return buf.getvalue()


def create_multipage_pdf(
    pages: List[np.ndarray],
    rotations: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    fit_a4: bool = False,
    jpeg_quality: int = 92,
) -> bytes:
    """Compile multiple document page images into a unified PDF document.

    Args:
        pages: List of page images (BGR or Grayscale NumPy arrays) in desired output sequence.
        rotations: Optional list of rotation angles matching the length of `pages`.
        output_path: Optional file path to write the PDF to disk.
        fit_a4: If True, centers images onto standard A4 pages (210 x 297 mm).
                If False, sizes each page to the image's exact natural aspect ratio.
        jpeg_quality: JPEG compression quality (default 92).

    Returns:
        Binary bytes of the generated PDF document.
    """
    if not pages:
        raise ValueError("Cannot create PDF with an empty page list.")

    if rotations is None:
        rotations = [0] * len(pages)
    elif len(rotations) != len(pages):
        raise ValueError("The length of 'rotations' must match the length of 'pages'.")

    # Encode each page into individual image byte streams
    image_bytes_list: List[bytes] = []
    for img, rot in zip(pages, rotations):
        encoded = encode_image_for_pdf(img, rotation=rot, jpeg_quality=jpeg_quality)
        image_bytes_list.append(encoded)

    # Configure page layout
    if fit_a4:
        # Standard A4 dimensions in typographic points (1 pt = 1/72 inch)
        a4_width_pt = img2pdf.mm_to_pt(210)
        a4_height_pt = img2pdf.mm_to_pt(297)
        layout = img2pdf.get_layout_fun(
            pagesize=(a4_width_pt, a4_height_pt),
            fit=img2pdf.FitMode.into,
            auto_orient=True,
        )
    else:
        # Use image's natural dimensions and aspect ratio
        layout = img2pdf.default_layout_fun

    # Compile images directly into standard PDF structure
    pdf_bytes = img2pdf.convert(image_bytes_list, layout_fun=layout)

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
