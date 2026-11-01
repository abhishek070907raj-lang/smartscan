"""
generate_samples.py — Generates realistic synthetic document test images.

Covers all 7 required scenarios:
1. Clear document on plain background
2. Document at an angle (perspective distortion)
3. Document with a dark, sharp shadow (simulating phone/hand cast)
4. Document with uneven, gradient lighting
5. Document on a busy, complex textured background
6. Multi-page document set (Pages 1, 2, and 3)
7. Low-quality, blurred image with noise
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_base_document(
    title: str,
    body_paragraphs: List[str],
    width: int = 700,
    height: int = 950,
) -> np.ndarray:
    """Generate a clean white document image with text rendered via PIL.

    Args:
        title: Header string for the document.
        body_paragraphs: List of text blocks to render.
        width: Pixel width of paper sheet.
        height: Pixel height of paper sheet.

    Returns:
        BGR numpy array representing the rendered page.
    """
    # Create white canvas (255) in RGB
    pil_img = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(pil_img)

    # Fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 26)
        body_font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw header bar and title
    draw.rectangle([40, 40, width - 40, 44], fill=(46, 134, 171))
    draw.text((40, 55), title, fill=(20, 30, 45), font=title_font)
    draw.text((40, 95), "SmartScan Image Processing Evaluation Document", fill=(100, 110, 120), font=small_font)
    draw.line([40, 115, width - 40, 115], fill=(210, 215, 220), width=2)

    # Render body paragraphs with line-wrapping
    curr_y = 135
    for paragraph in body_paragraphs:
        words = paragraph.split(" ")
        current_line: List[str] = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            line_w = bbox[2] - bbox[0]
            if line_w <= (width - 80):
                current_line.append(word)
            else:
                draw.text((40, curr_y), " ".join(current_line), fill=(30, 30, 30), font=body_font)
                curr_y += 24
                current_line = [word]
        if current_line:
            draw.text((40, curr_y), " ".join(current_line), fill=(30, 30, 30), font=body_font)
            curr_y += 35

    # Footer
    draw.line([40, height - 50, width - 40, height - 50], fill=(210, 215, 220), width=1)
    draw.text((40, height - 40), "CONFIDENTIAL & PROPRIETARY — FOR ACADEMIC TESTING ONLY", fill=(140, 140, 140), font=small_font)
    draw.text((width - 90, height - 40), "Page 1/1", fill=(140, 140, 140), font=small_font)

    # Convert RGB to OpenCV BGR
    rgb_arr = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def warp_into_canvas(
    doc: np.ndarray,
    canvas_shape: Tuple[int, int] = (1000, 1200),
    dst_corners: np.ndarray = None,
    bg_color: Tuple[int, int, int] = (45, 52, 54),
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a document into a desk canvas using a 4-point perspective transform.

    Args:
        doc: BGR document image.
        canvas_shape: (height, width) of the scene canvas.
        dst_corners: Target (x, y) coordinates of the 4 document corners on canvas.
                     Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        bg_color: BGR colour of the plain background desk.

    Returns:
        (canvas_image, dst_corners)
    """
    ch, cw = canvas_shape
    dh, dw = doc.shape[:2]

    # Source corners (the flat rectangle)
    src_corners = np.array(
        [[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]],
        dtype=np.float32,
    )

    if dst_corners is None:
        # Default centered placement with moderate perspective tilt
        dst_corners = np.array(
            [[220, 100], [960, 140], [890, 890], [180, 830]],
            dtype=np.float32,
        )

    # Compute 3x3 homography matrix
    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # Create solid background
    canvas = np.full((ch, cw, 3), bg_color, dtype=np.uint8)

    # Mask for document polygon
    mask = np.zeros((ch, cw), dtype=np.uint8)
    poly_pts = dst_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly_pts], 255)

    # Warp document
    warped_doc = cv2.warpPerspective(doc, matrix, (cw, ch), flags=cv2.INTER_LINEAR)

    # Composite document over background
    inv_mask = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(canvas, canvas, mask=inv_mask)
    fg_part = cv2.bitwise_and(warped_doc, warped_doc, mask=mask)
    combined = cv2.add(bg_part, fg_part)

    return combined, dst_corners


def generate_all_samples(output_dir: str = "sample_images") -> List[str]:
    """Generate all 7 scenario images and save them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    standard_paragraphs = [
        "Computer vision systems frequently rely on document scanning pipelines to digitize printed media. Real-world capture introduces geometric perspective distortion, uneven ambient lighting, and sensor noise.",
        "The automated pipeline detects the paper boundary using Canny edge filtering followed by polygonal contour approximation. Once four corner vertices are identified, an inverse homography perspective warp projects the document back to a frontal, flat plane.",
        "Subsequent stages implement morphological background estimation to eliminate shadows without washing out ink strokes. Finally, adaptive thresholding or Otsu binarization yields optimal contrast for OCR analysis."
    ]

    # 1. Clear Document on Plain Background (straight, high contrast)
    doc1 = create_base_document("System Design Specification", standard_paragraphs)
    dst1 = np.array([[250, 70], [950, 70], [950, 920], [250, 920]], dtype=np.float32)
    s1, _ = warp_into_canvas(doc1, (1000, 1200), dst1, bg_color=(35, 38, 42))
    p1 = os.path.join(output_dir, "01_clear_plain.jpg")
    cv2.imwrite(p1, s1)
    generated_files.append(p1)

    # 2. Document at an Angle (prominent perspective tilt)
    doc2 = create_base_document("Perspective Geometry Analysis", standard_paragraphs)
    dst2 = np.array([[180, 140], [1020, 80], [910, 920], [120, 840]], dtype=np.float32)
    s2, _ = warp_into_canvas(doc2, (1000, 1200), dst2, bg_color=(50, 45, 40))
    p2 = os.path.join(output_dir, "02_document_angled.jpg")
    cv2.imwrite(p2, s2)
    generated_files.append(p2)

    # 3. Dark Harsh Shadow (simulating hand or phone holding shadow across page)
    doc3 = create_base_document("Shadow Attenuation Testing", standard_paragraphs)
    dst3 = np.array([[210, 90], [970, 120], [910, 910], [170, 850]], dtype=np.float32)
    s3, _ = warp_into_canvas(doc3, (1000, 1200), dst3, bg_color=(45, 45, 50))
    # Cast shadow across the bottom-left diagonal
    shadow_mask = np.ones((1000, 1200), dtype=np.float32)
    shadow_poly = np.array([[0, 400], [750, 1000], [0, 1000]], dtype=np.int32)
    poly_mask = np.zeros((1000, 1200), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [shadow_poly], 255)
    # Smooth shadow boundary with Gaussian blur
    poly_mask_blurred = cv2.GaussianBlur(poly_mask, (101, 101), 30) / 255.0
    # Shadow factor: drop luminance to ~35%
    shadow_multiplier = 1.0 - (0.65 * poly_mask_blurred)
    s3_shadowed = (s3.astype(np.float32) * shadow_multiplier[:, :, np.newaxis])
    s3_final = np.clip(s3_shadowed, 0, 255).astype(np.uint8)
    p3 = os.path.join(output_dir, "03_dark_shadow.jpg")
    cv2.imwrite(p3, s3_final)
    generated_files.append(p3)

    # 4. Uneven Lighting (radial / gradient spotlight)
    doc4 = create_base_document("Photometric Variation Test", standard_paragraphs)
    dst4 = np.array([[220, 100], [960, 110], [900, 900], [180, 860]], dtype=np.float32)
    s4, _ = warp_into_canvas(doc4, (1000, 1200), dst4, bg_color=(40, 42, 45))
    # Create 2D gradient light map: bright at top-right (1.3x), dim at bottom-left (0.4x)
    y_coords, x_coords = np.ogrid[:1000, :1200]
    light_map = 0.4 + 0.9 * ((x_coords / 1200.0) * 0.6 + ((1000 - y_coords) / 1000.0) * 0.4)
    s4_lit = s4.astype(np.float32) * light_map[:, :, np.newaxis]
    s4_final = np.clip(s4_lit, 0, 255).astype(np.uint8)
    p4 = os.path.join(output_dir, "04_uneven_lighting.jpg")
    cv2.imwrite(p4, s4_final)
    generated_files.append(p4)

    # 5. Complex Background (busy texture, wood grain lines, and clutter objects)
    doc5 = create_base_document("Edge Segmentation Benchmark", standard_paragraphs)
    dst5 = np.array([[240, 120], [940, 100], [890, 890], [200, 870]], dtype=np.float32)
    # Synthesize textured background
    ch, cw = 1000, 1200
    base_bg = np.full((ch, cw, 3), (70, 80, 90), dtype=np.uint8)
    # Add alternating stripes/grids (tablecloth / wood plank pattern)
    for row in range(0, ch, 40):
        cv2.line(base_bg, (0, row), (cw, row), (55, 65, 75), 2)
    for col in range(0, cw, 50):
        cv2.line(base_bg, (col, 0), (col, ch), (60, 70, 80), 2)
    # Add random pen / ruler object near document edge
    cv2.line(base_bg, (80, 200), (120, 750), (20, 140, 220), 8)   # Blue pen
    cv2.circle(base_bg, (1080, 250), 65, (40, 40, 40), 12)        # Coffee mug mark
    # Composite document over complex background
    dh, dw = doc5.shape[:2]
    src_corners = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_corners, dst5)
    warped_doc = cv2.warpPerspective(doc5, matrix, (cw, ch), flags=cv2.INTER_LINEAR)
    mask = np.zeros((ch, cw), dtype=np.uint8)
    poly_pts = dst5.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly_pts], 255)
    inv_mask = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(base_bg, base_bg, mask=inv_mask)
    fg_part = cv2.bitwise_and(warped_doc, warped_doc, mask=mask)
    s5_final = cv2.add(bg_part, fg_part)
    p5 = os.path.join(output_dir, "05_complex_background.jpg")
    cv2.imwrite(p5, s5_final)
    generated_files.append(p5)

    # 6. Multi-page set of 3+ pages (Report Page 1, Page 2, Page 3)
    p6_texts = [
        ("Final Project Report — Part 1: Overview", [
            "Introduction: The Smart Document Scanner project presents a real-time computer vision workflow capable of converting unconstrained mobile phone captures into high-resolution scanner quality outputs.",
            "Motivation: Handheld captures suffer from perspective keystone distortion, non-uniform ambient illumination, and motion blur. Traditional flatbed scanners eliminate these physical anomalies through controlled optical geometry.",
            "Objectives: 1. Accurate quadrangular paper contour localization. 2. Inverse projective homography mapping. 3. Morphological illumination compensation. 4. High-accuracy OCR transcription."
        ]),
        ("Final Project Report — Part 2: Methodology", [
            "Mathematical Foundation: An arbitrary planar surface photographed under projective perspective transforms planar coordinates via a 3x3 homography matrix H with 8 degrees of freedom.",
            "Edge Processing: The Canny edge detector applies Gaussian smoothing followed by Sobel gradient computations, non-maximum suppression, and hysteresis thresholding.",
            "Illumination Correction: Background estimation employs morphological dilation using a circular structuring element followed by median blur filtering."
        ]),
        ("Final Project Report — Part 3: Evaluation", [
            "Experimental Results: The pipeline was evaluated against synthetic benchmark scenarios representing various environmental distortions.",
            "Optical Character Recognition: Evaluated using Tesseract OCR v5. Confidence scores and word recognition rates demonstrate superior fidelity on shadow-removed and CLAHE-enhanced outputs.",
            "Conclusion: The system delivers competitive performance relative to commercial mobile scanning solutions."
        ])
    ]

    for idx, (p_title, p_paras) in enumerate(p6_texts, start=1):
        doc_mp = create_base_document(p_title, p_paras)
        dst_mp = np.array([[230, 80], [960, 110], [910, 910], [180, 850]], dtype=np.float32)
        s_mp, _ = warp_into_canvas(doc_mp, (1000, 1200), dst_mp, bg_color=(38, 40, 44))
        fpath = os.path.join(output_dir, f"06_multipage_page{idx}.jpg")
        cv2.imwrite(fpath, s_mp)
        generated_files.append(fpath)

    # 7. Low-quality or Blurred Image
    doc7 = create_base_document("Degraded Optical Sample", standard_paragraphs)
    dst7 = np.array([[220, 90], [970, 120], [920, 900], [170, 860]], dtype=np.float32)
    s7, _ = warp_into_canvas(doc7, (1000, 1200), dst7, bg_color=(42, 45, 48))
    # Apply severe Gaussian blur to simulate camera focus failure
    s7_blurred = cv2.GaussianBlur(s7, (13, 13), 5.0)
    # Add Gaussian sensor noise
    noise = np.random.normal(0, 12, s7_blurred.shape).astype(np.float32)
    s7_noisy = np.clip(s7_blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    p7 = os.path.join(output_dir, "07_low_quality_blurred.jpg")
    cv2.imwrite(p7, s7_noisy)
    generated_files.append(p7)

    return generated_files


if __name__ == "__main__":
    files = generate_all_samples()
    print(f"Successfully generated {len(files)} test images:")
    for f in files:
        print(f" - {f}")
