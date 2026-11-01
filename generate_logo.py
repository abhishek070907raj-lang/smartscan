"""Generate a clean SmartScan logo icon matching the teal (#0F766E) design system."""
from PIL import Image, ImageDraw
import os

def generate_logo(output_path: str, size: int = 256) -> None:
    """Create a minimalist document-scanner icon.

    Draws a centered rounded-corner rectangle (the 'document') with a scan-line
    beam across it and subtle text lines, inside a circular brand-colored badge.
    Contains no text inside or outside the badge so nothing is cut off.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Background circular badge — brand teal (#0F766E = 15, 118, 110)
    margin = 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(15, 118, 110, 255),
    )

    # Subtle inner border for polish
    draw.ellipse(
        [margin + 4, margin + 4, size - margin - 4, size - margin - 4],
        outline=(255, 255, 255, 40),
        width=2,
    )

    # 2. Centered White Document Sheet
    doc_w = size * 0.44
    doc_h = size * 0.58
    doc_left = (size - doc_w) / 2.0
    doc_right = doc_left + doc_w
    doc_top = (size - doc_h) / 2.0
    doc_bot = doc_top + doc_h

    # Shadow behind document
    draw.rounded_rectangle(
        [doc_left + 2, doc_top + 3, doc_right + 3, doc_bot + 4],
        radius=10,
        fill=(10, 70, 65, 140),
    )

    # Crisp white document
    draw.rounded_rectangle(
        [doc_left, doc_top, doc_right, doc_bot],
        radius=10,
        fill=(255, 255, 255, 255),
    )

    # Header accent line on document
    draw.rectangle(
        [doc_left + 12, doc_top + 12, doc_right - 12, doc_top + 16],
        fill=(15, 118, 110, 180),
    )

    # 3. Document text lines
    line_color = (203, 213, 225, 220)  # Slate 300
    line_y_fractions = [0.26, 0.35, 0.58, 0.67, 0.76]
    for i, frac in enumerate(line_y_fractions):
        ly = doc_top + doc_h * frac
        indent_right = 16 if (i % 2 == 1) else 10
        draw.rounded_rectangle(
            [doc_left + 12, ly, doc_right - indent_right, ly + 4],
            radius=2,
            fill=line_color,
        )

    # 4. Laser Scan Beam — bright glowing emerald line across the document
    scan_y = doc_top + doc_h * 0.46
    scan_h = size * 0.038
    # Soft glow behind the beam
    draw.rectangle(
        [doc_left - 4, scan_y - 2, doc_right + 4, scan_y + scan_h + 2],
        fill=(20, 184, 166, 90),
    )
    # Bright solid laser line
    draw.rectangle(
        [doc_left, scan_y, doc_right, scan_y + scan_h],
        fill=(20, 184, 166, 255),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Teal logo icon saved to {output_path}")


if __name__ == "__main__":
    generate_logo("assets/logo.png")
