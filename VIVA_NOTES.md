# SmartScan — College Viva & Examination Questions & Answers
**Course:** Image Processing and Applications (IPA) — 3rd Year B.Tech IT  
**Project:** Smart Document Scanner and Image Enhancement System

This document contains the comprehensive viva examination preparation notes and Q&A covering every computer vision algorithm implemented in SmartScan.

---

## 1. EXIF Orientation & Downscaling

### Q: Why do we downscale the image during detection if we want a high-resolution final document?
**Answer:**
Image operations like Gaussian blur and Canny gradient convolution are proportional to the pixel count O(N). A 24-megapixel smartphone photo (4000 x 6000) contains 24 million pixels and takes significantly longer to process. Downscaling the image so its maximum dimension is 1024 pixels reduces the pixel count by approximately 98% (down to roughly 0.5 million pixels), allowing the detection to run in under 25 milliseconds. The detected corner coordinates (x', y') are then mapped back to the original full-resolution space by dividing by the scale factor: (x, y) = (x' / scale_factor, y' / scale_factor). The perspective warp and subsequent enhancement are then executed directly on the full-resolution uncompressed photo.

### Q: How does EXIF orientation correction work?
**Answer:**
Smartphone cameras store raw pixel sensors in a fixed orientation and record a tag in the EXIF metadata (Tag 274: Orientation) indicating the phone's physical rotation at the moment of capture (e.g. Tag 6 means rotated 90 degrees clockwise). If not transposed before processing, computer vision algorithms operate on a sideways or upside-down matrix. In our pipeline, `PIL.ImageOps.exif_transpose` inspects this orientation tag, performs the necessary lossless transpose/flip, and strips the tag to ensure an upright image.

---

## 2. Grayscale & Noise Filtering

### Q: Why do we convert color images to grayscale before edge detection?
**Answer:**
Color channels (Red, Green, Blue) increase computation and memory consumption by 300% without providing additional geometric boundary information. Paper-to-table contrast is fundamentally a luminance discontinuity. We convert to grayscale using standard ITU-R BT.601 luminance weights: Y = 0.299 * R + 0.587 * G + 0.114 * B. Green receives the largest weight (0.587) because the human visual system's photopic spectral sensitivity is most sensitive to green wavelengths (~555 nm).

### Q: Why is Gaussian blur preferred over a simple box (mean) filter?
**Answer:**
A simple box filter averages all pixels in a window with uniform weights, causing directional artifacts and blurring sharp edges indiscriminately. A Gaussian filter applies distance-decaying weights according to a 2D Gaussian distribution G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2)). This provides isotropic (rotationally symmetric) smoothing that attenuates high-frequency sensor noise and text micro-edges without angular bias.

---

## 3. Canny Edge Detection & Morphological Closing

### Q: What are the 4 fundamental stages of the Canny edge detector?
**Answer:**
1. **Gaussian Smoothing:** Low-pass filtering to suppress sensor noise.
2. **Gradient Computation:** Calculating spatial derivatives Gx and Gy using 3x3 Sobel kernels to determine gradient magnitude G = sqrt(Gx^2 + Gy^2) and direction theta = arctan(Gy / Gx).
3. **Non-Maximum Suppression (NMS):** Thinning thick gradient bands into 1-pixel-wide ridges by suppressing any pixel that is not a local maximum along its gradient direction vector.
4. **Hysteresis Thresholding:** Dual-thresholding using T_high and T_low. Pixels with gradient > T_high are marked as strong edges; pixels < T_low are discarded; pixels between T_low and T_high are preserved only if connected to a strong edge via 8-connectivity.

### Q: Why do you follow Canny edge detection with a morphological closing?
**Answer:**
Lighting gradients or subtle paper shadows can cause local contrast along the paper boundary to drop below the threshold, resulting in tiny 1-to-2 pixel breaks in the edge loop. Morphological Closing (dilation followed by erosion: (A (+) B) (-) B) bridges these small gaps, forming a continuous closed boundary contour necessary for contour extraction.

---

## 4. Contour Ranking & Polygonal Approximation

### Q: How does the Ramer-Douglas-Peucker (approxPolyDP) algorithm work?
**Answer:**
The Ramer-Douglas-Peucker (RDP) algorithm is a recursive curve-simplification algorithm. It draws a line between the endpoints of a contour, identifies the point on the curve with maximum perpendicular distance d_max to that line, and checks if d_max exceeds a threshold epsilon. In our pipeline, epsilon = 0.02 * perimeter (2% of the arc length). If the resulting simplified polygon has exactly 4 vertices and is convex (cv2.isContourConvex), it is identified as the document's boundary.

### Q: What happens if Canny misses the document boundary?
**Answer:**
Our system uses a 3-tier fallback chain:
1. **Dynamic Median Canny:** Calculates lower and upper thresholds dynamically based on the image's median luminance: T_low = 0.66 * median, T_high = 1.33 * median.
2. **Adaptive Gaussian Thresholding:** Uses localized thresholding across a 15x15 window to separate foreground paper from darker backgrounds under severe lighting gradients.
3. **Bounding Canvas Fallback:** If all automated methods fail, the system selects the full image frame [0, 0] to [W-1, H-1] with a warning, allowing the user to adjust the 4 corners manually using the interactive coordinate sliders.

---

## 5. Corner Ordering & Perspective Transformation

### Q: How do you mathematically order the 4 corners into [Top-Left, Top-Right, Bottom-Right, Bottom-Left]?
**Answer:**
Using vector coordinate sums and differences:
- **Sum (x + y):** The Top-Left corner is closest to the origin (0, 0), so it has the minimum (x + y). The Bottom-Right corner is farthest from the origin, so it has the maximum (x + y).
- **Difference (y - x):** The Top-Right corner has a large x and small y, giving the minimum (y - x). The Bottom-Left corner has a small x and large y, giving the maximum (y - x).

### Q: How does homography transform the perspective?
**Answer:**
A planar surface viewed in 3D projective space maps to another plane via a 3x3 homography matrix H with 8 degrees of freedom. With 4 point correspondences between the detected quadrilateral corners and the target flat rectangle corners [0, 0], [W-1, 0], [W-1, H-1], [0, H-1], `cv2.getPerspectiveTransform` sets up and solves a linear system of 8 equations. `cv2.warpPerspective` then uses inverse mapping with bicubic interpolation: for each destination pixel (x', y'), it samples the source image at H^-1(x', y'), eliminating trapezoidal keystone distortion.

---

## 6. Retinex Illumination & Shadow Removal

### Q: Why does simple brightness adjustment fail on shadowed documents?
**Answer:**
According to Retinex theory, an image I(x, y) is the product of surface reflectance R(x, y) (the ink and paper) and ambient illumination L(x, y) (the lighting and shadows): I(x, y) = R(x, y) * L(x, y).
A linear brightness filter adds a constant value (I + beta). This causes unshadowed white paper to saturate and clip at 255 (washing out text), while the shadowed area remains darker than the rest of the sheet.
Our algorithm estimates the ambient illumination L(x, y) by applying morphological dilation with a 15x15 kernel (which replaces dark text ink with surrounding white paper) followed by a 21x21 median blur to smooth out gradients. We then divide the original image by this estimated illumination: R(x, y) = (I(x, y) / L(x, y)) * 255. This neutralizes shadows and produces a uniform white paper surface across the entire page.

---

## 7. CLAHE vs. Global Histogram Equalization

### Q: Why is CLAHE superior to Global Histogram Equalization for document enhancement?
**Answer:**
Global Histogram Equalization flattens the cumulative distribution function across the entire image. For documents, this over-amplifies background noise, turning faint paper texture and sensor grain into dark splotches.
Contrast Limited Adaptive Histogram Equalization (CLAHE) divides the image into small contextual tiles (8x8). It computes histograms locally within each tile and clips counts that exceed a threshold (clipLimit = 2.0), redistributing the excess uniformly across all bins to prevent noise amplification. It then blends tile boundaries using bilinear interpolation.
Furthermore, applying CLAHE exclusively to the Luminance (L) channel in CIELAB color space enhances character contrast without altering the A and B chrominance channels, preventing hue shifts.

---

## 8. Document Binarization (Adaptive vs. Otsu)

### Q: When does Otsu's thresholding fail, and why is Adaptive thresholding preferred?
**Answer:**
Otsu's method calculates a single global threshold that minimizes intra-class variance. It assumes a bimodal intensity histogram with a clear valley between foreground text and background paper. When an image contains residual lighting gradients, uneven illumination, or localized shadows, a single global threshold causes one side of the page to become completely black while text on the other side is washed out.
Adaptive Gaussian thresholding calculates a localized threshold for every individual pixel based on a Gaussian-weighted mean of its 17x17 neighborhood minus a constant offset C. This adapts to regional illumination variations across the document.

---

## 9. Tesseract OCR & Layout Analysis

### Q: How does Tesseract extract text while preserving layout?
**Answer:**
Tesseract v5 uses a deep Long Short-Term Memory (LSTM) recurrent neural network. Instead of using raw `image_to_string()`, our pipeline calls `image_to_data(output_type=Output.DICT, config='--psm 3')` (Page Segmentation Mode 3 = fully automatic page segmentation). This returns structural bounding boxes with hierarchy identifiers: `block_num`, `par_num`, `line_num`, `word_num`, and per-word confidence `conf`. By detecting transitions in line and paragraph numbers, we preserve natural paragraph spacing and line breaks.

### Q: How is the page confidence score calculated?
**Answer:**
Tesseract assigns a recognition confidence score (0 to 100) to each recognized character/word, while non-word layout structural nodes carry a confidence of -1. We filter out all entries with conf < 0 and compute the arithmetic mean of all recognized word confidences: mean_conf = sum(confs) / len(confs).

---

## 10. PDF Compilation & Aspect Preservation

### Q: How does lossless PDF compilation work without re-compression?
**Answer:**
Many PDF libraries decode images into raw bitmaps and re-encode them as lossy JPEGs, degrading text edges. `img2pdf` embeds the existing JPEG or PNG compressed byte streams directly into the PDF structure as native image XObjects (using DCTDecode for JPEG and FlateDecode for PNG). In standard mode, each PDF page dimension matches the document's physical aspect ratio. In ISO A4 mode, it converts 210 x 297 mm into typographic points (595.28 x 841.89 pt at 72 DPI) and centers the image proportionally with zero distortion.
