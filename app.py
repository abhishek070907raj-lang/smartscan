"""
app.py — SmartScan: Intelligent Document Scanner & Image Enhancement System.

A soft-glass Streamlit workspace for document capture, perspective correction,
Retinex shadow removal, enhancement, OCR and PDF export.
"""

import base64
import html
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.document_detection import detect_document_corners
from modules.icons import icon
from modules.image_enhancement import (
    apply_clahe,
    apply_global_hist_eq,
    binarize_document,
    enhance_document,
    to_grayscale,
)
from modules.ocr import configure_tesseract_path, extract_text_and_confidence
from modules.pdf_generator import create_multipage_pdf
from modules.perspective_correction import four_point_transform
from modules.utils import (
    bgr_to_rgb,
    load_image_from_bytes,
    resize_for_processing,
    rotate_image,
    scale_points,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartScan — Intelligent Document Scanner",
    page_icon=os.path.join(os.path.dirname(__file__), "assets", "logo.png"),
    layout="wide",
    initial_sidebar_state="auto",
)

NAV_LABELS = [
    "Home",
    "Scan & Upload",
    "Detection & Warp",
    "Enhancement",
    "Multi-Page",
    "OCR",
    "Export",
    "About",
]

MODES = ["Enhanced", "Color", "Shadow-Free", "Grayscale", "Black & White", "Original"]


def load_master_css() -> None:
    """Load the single master stylesheet (assets/style.css)."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_data = f.read()
        st.markdown(f"<style>{css_data}</style>", unsafe_allow_html=True)


def get_logo_base64() -> str:
    """Return the brand logo as a base64 data URI."""
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


load_master_css()
LOGO_B64 = get_logo_base64()


def icon_tile(name: str, tint: str = "blue", size: str = "md") -> str:
    """Return a rounded icon tile with an inline SVG icon."""
    icon_sizes = {"sm": 16, "md": 20, "lg": 26}
    return (
        f'<span class="icon-tile icon-tile-{size} tint-{tint}">'
        f'{icon(name, size=icon_sizes.get(size, 20), color="currentColor")}'
        f"</span>"
    )


# -----------------------------------------------------------------------------
# Session State & Document Helpers
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "pages": [],
        "current_page_idx": 0,
        "nav_target": "Home",
        "nav_request": "",
        "flash": "",
        "show_camera": False,
        "uploader_key": 0,
        "camera_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def add_image_page(raw_bgr: np.ndarray, name: str) -> None:
    """Process a newly uploaded or captured raw image and append it to the session."""
    resized, scale_factor = resize_for_processing(raw_bgr, max_dimension=1024)
    corners_res, method, stages = detect_document_corners(resized, return_stages=True)

    page_dict = {
        "id": f"page_{len(st.session_state.pages) + 1}_{int(time.time() * 1000)}",
        "name": name,
        "created_at": time.time(),
        "raw_image": raw_bgr,
        "resized_image": resized,
        "scale_factor": scale_factor,
        "auto_corners": corners_res.copy(),
        "corners": corners_res.copy(),
        "detection_method": method,
        "stages": stages,
        "margin_trim": 1.0,
        "rotation": 0,
        "enhancement_mode": "Enhanced",
        "bw_method": "adaptive",
        "processed_image": None,
        "ocr_text": "",
        "ocr_confidence": 0.0,
        "ocr_words": 0,
    }
    st.session_state.pages.append(page_dict)
    st.session_state.current_page_idx = len(st.session_state.pages) - 1


def load_benchmark_samples() -> int:
    """Load all pre-packaged benchmark sample images."""
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    loaded = 0
    if os.path.exists(sample_dir):
        files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".jpg")])
        for fname in files:
            fpath = os.path.join(sample_dir, fname)
            bgr = cv2.imread(fpath)
            if bgr is not None:
                add_image_page(bgr, fname)
                loaded += 1
    return loaded


def get_current_page() -> Optional[Dict[str, Any]]:
    if not st.session_state.pages:
        return None
    idx = max(0, min(st.session_state.current_page_idx, len(st.session_state.pages) - 1))
    return st.session_state.pages[idx]


def compute_processed_page(page: Dict[str, Any]) -> np.ndarray:
    """Render the final warped, enhanced, and rotated document image."""
    corners_full = scale_points(page["corners"], page["scale_factor"])
    warped = four_point_transform(
        page["raw_image"],
        corners_full,
        margin_trim_percent=page.get("margin_trim", 1.0),
    )
    enhanced = enhance_document(
        warped,
        mode=page.get("enhancement_mode", "Enhanced"),
        bw_method=page.get("bw_method", "adaptive"),
    )
    rotation = page.get("rotation", 0)
    if rotation != 0:
        enhanced = rotate_image(enhanced, rotation)

    page["processed_image"] = enhanced
    return enhanced


def warped_preview(page: Dict[str, Any]) -> np.ndarray:
    """Return the unenhanced warped page, rotated to match the result panel."""
    corners_full = scale_points(page["corners"], page["scale_factor"])
    warped = four_point_transform(
        page["raw_image"],
        corners_full,
        margin_trim_percent=page.get("margin_trim", 1.0),
    )
    rotation = page.get("rotation", 0)
    return rotate_image(warped, rotation) if rotation else warped


# -----------------------------------------------------------------------------
# Presentation Helpers
# -----------------------------------------------------------------------------
def go_to(page: str) -> None:
    """Queue a navigation change for the next rerun (widget-safe)."""
    st.session_state.nav_request = page
    st.rerun()


def format_relative_time(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "Just now"
    delta = max(0, int(time.time() - timestamp))
    if delta < 60:
        return "Just now"
    if delta < 3600:
        return f"{delta // 60} min ago"
    if delta < 86400:
        return f"{delta // 3600} hr ago"
    return time.strftime("%d %b", time.localtime(timestamp))


def page_status(page: Dict[str, Any]) -> tuple[str, str]:
    method = page.get("detection_method", "")
    if method == "primary_contour":
        return "Auto-detected", "success"
    if method == "fallback_full_image":
        return "Manual check", "danger"
    return "Fallback used", "warning"


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
            <div class="eyebrow"><span class="eyebrow-dot"></span>{html.escape(eyebrow)}</div>
            <h1 class="page-title">{html.escape(title)}</h1>
            <p class="page-subtitle">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, note: str = "") -> None:
    note_html = f'<span class="section-note">{html.escape(note)}</span>' if note else ""
    st.markdown(
        f'<div class="section-head"><h2 class="section-title-lg">{html.escape(title)}</h2>{note_html}</div>',
        unsafe_allow_html=True,
    )


def show_flash() -> None:
    message = st.session_state.get("flash")
    if message:
        st.markdown(
            f'<div class="flash">{icon("check-circle", size=17, color="currentColor")}'
            f"<span>{html.escape(message)}</span></div>",
            unsafe_allow_html=True,
        )
        st.session_state.flash = ""


def render_empty_state(
    section_slug: str,
    title: str,
    message: str,
    icon_name: str = "file-text",
    tint: str = "blue",
) -> None:
    with st.container(border=True, key=f"empty_{section_slug}"):
        st.markdown(
            f"""
            <div class="empty-state">
                {icon_tile(icon_name, tint, "lg")}
                <h3>{html.escape(title)}</h3>
                <p>{html.escape(message)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _pad_l, c1, c2, _pad_r = st.columns([0.7, 1.45, 1.1, 0.7])
        with c1:
            if st.button(
                "Go to Scan & Upload",
                type="primary",
                use_container_width=True,
                key=f"empty_scan_{section_slug}",
            ):
                go_to("Scan & Upload")
        with c2:
            if st.button(
                "Load samples",
                use_container_width=True,
                key=f"empty_samples_{section_slug}",
            ):
                with st.spinner("Preparing sample documents…"):
                    count = load_benchmark_samples()
                st.session_state.flash = f"Loaded {count} benchmark samples."
                go_to("Detection & Warp")


def page_selector(section_slug: str) -> Optional[Dict[str, Any]]:
    pages = st.session_state.pages
    if not pages:
        return None
    names = [f"Page {i + 1}: {p['name']}" for i, p in enumerate(pages)]
    selected = st.selectbox(
        "Active document page",
        range(len(names)),
        index=min(st.session_state.current_page_idx, len(names) - 1),
        format_func=lambda i: names[i],
        key=f"{section_slug}_page_{len(pages)}",
    )
    st.session_state.current_page_idx = selected
    return pages[selected]


def render_recent_page_row(page: Dict[str, Any]) -> None:
    status, tone = page_status(page)
    st.markdown(
        f"""
        <div class="recent-row">
            {icon_tile("file-text", "blue", "sm")}
            <div class="recent-main">
                <div class="recent-name">{html.escape(page["name"])}</div>
                <div class="recent-meta">{html.escape(status)} · {format_relative_time(page.get("created_at"))}</div>
            </div>
            <span class="status-dot status-{tone}" title="{html.escape(status)}"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_method_row(
    key: str,
    title: str,
    description: str,
    icon_name: str,
    tint: str,
) -> bool:
    with st.container(border=True, key=f"method_{key}"):
        c1, c2 = st.columns([6, 1])
        with c1:
            st.markdown(
                f"""
                <div class="method-copy">
                    {icon_tile(icon_name, tint, "sm")}
                    <div>
                        <div class="method-title">{html.escape(title)}</div>
                        <p class="method-desc">{html.escape(description)}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            clicked = st.button(":material/chevron_right:", key=f"method_{key}_btn", help=title)
    return clicked


def render_compare_panel(
    key: str,
    kicker: str,
    title: str,
    image_bgr: np.ndarray,
    meta: str,
) -> None:
    with st.container(border=True, key=key):
        st.markdown(
            f"""
            <div class="panel-head">
                <span class="panel-kicker">{html.escape(kicker)}</span>
                <h3>{html.escape(title)}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)
        st.markdown(f'<div class="panel-meta">{html.escape(meta)}</div>', unsafe_allow_html=True)


def clear_pdf_cache() -> None:
    st.session_state["cached_pdf"] = None


def render_page_card(
    index: int,
    page: Dict[str, Any],
    key_prefix: str,
    show_mode: bool = True,
    compact: bool = False,
) -> None:
    pages = st.session_state.pages
    processed = compute_processed_page(page)
    rotation = page.get("rotation", 0)
    with st.container(border=True, key=f"{key_prefix}_card_{page['id']}"):
        thumb_col, info_col = st.columns([1.05, 2.2] if not compact else [0.9, 2.4])
        with thumb_col:
            st.markdown(f'<div class="thumb-kicker">Page {index + 1}</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(processed), use_container_width=True)
        with info_col:
            st.markdown(
                f"""
                <div class="page-card-head">
                    <span class="page-number">Page {index + 1}</span>
                    <h3>{html.escape(page["name"])}</h3>
                </div>
                <div class="meta-chips">
                    <span class="meta-chip">{processed.shape[1]} × {processed.shape[0]} px</span>
                    <span class="meta-chip">Rotation {rotation}°</span>
                    <span class="meta-chip">{html.escape(page.get("enhancement_mode", "Enhanced"))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if show_mode:
                mode = st.selectbox(
                    "Enhancement mode",
                    MODES,
                    index=MODES.index(page.get("enhancement_mode", "Enhanced")),
                    key=f"{key_prefix}_mode_{page['id']}",
                    label_visibility="collapsed",
                )
                if mode != page.get("enhancement_mode"):
                    page["enhancement_mode"] = mode
                    clear_pdf_cache()
                    st.rerun()

            a1, a2, a3, a4, a5 = st.columns(5)
            if a1.button(
                "↑",
                key=f"{key_prefix}_up_{page['id']}",
                disabled=index == 0,
                help="Move page up",
            ):
                pages[index - 1], pages[index] = pages[index], pages[index - 1]
                clear_pdf_cache()
                st.rerun()

            if a2.button(
                "↓",
                key=f"{key_prefix}_down_{page['id']}",
                disabled=index == len(pages) - 1,
                help="Move page down",
            ):
                pages[index + 1], pages[index] = pages[index], pages[index + 1]
                clear_pdf_cache()
                st.rerun()

            if a3.button("↺", key=f"{key_prefix}_rotl_{page['id']}", help="Rotate left"):
                page["rotation"] = (rotation + 270) % 360
                clear_pdf_cache()
                st.rerun()

            if a4.button("↻", key=f"{key_prefix}_rotr_{page['id']}", help="Rotate right"):
                page["rotation"] = (rotation + 90) % 360
                clear_pdf_cache()
                st.rerun()

            if a5.button("✕", key=f"{key_prefix}_del_{page['id']}", help="Delete page"):
                pages.pop(index)
                if st.session_state.current_page_idx >= len(pages):
                    st.session_state.current_page_idx = max(0, len(pages) - 1)
                clear_pdf_cache()
                st.rerun()


# -----------------------------------------------------------------------------
# Background Canvas
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="mesh-bg" aria-hidden="true">
        <span class="mesh-blob blob-a"></span>
        <span class="mesh-blob blob-b"></span>
        <span class="mesh-blob blob-c"></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <img class="brand-logo" src="data:image/png;base64,{LOGO_B64}" alt="SmartScan logo">
            <div>
                <div class="brand-name">SmartScan</div>
                <div class="brand-sub">Document studio</div>
            </div>
        </div>
        <div class="nav-caption">Workspace</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.nav_target not in NAV_LABELS:
        st.session_state.nav_target = NAV_LABELS[0]

    # Apply any queued navigation before the radio widget is instantiated.
    requested_nav = st.session_state.get("nav_request", "")
    if requested_nav in NAV_LABELS:
        st.session_state.nav_target = requested_nav
        st.session_state.nav_radio = requested_nav
        st.session_state.nav_request = ""

    selected_nav = st.radio(
        "Navigation",
        NAV_LABELS,
        index=NAV_LABELS.index(st.session_state.nav_target),
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.nav_target = selected_nav

    ocr_ok, _ = configure_tesseract_path()
    page_count = len(st.session_state.pages)
    ocr_chip = (
        f'<span class="status-chip chip-success">{icon("check-circle", size=13, color="currentColor")}<span>OCR active</span></span>'
        if ocr_ok
        else f'<span class="status-chip chip-warning">{icon("alert-triangle", size=13, color="currentColor")}<span>OCR off</span></span>'
    )
    st.markdown(
        f"""
        <div class="status-chips">
            {ocr_chip}
            <span class="status-chip">{icon("file-text", size=13, color="currentColor")}<span>{page_count} page{"s" if page_count != 1 else ""}</span></span>
        </div>
        <div class="sidebar-rule"></div>
        """,
        unsafe_allow_html=True,
    )

    if page_count > 0:
        if st.button("Clear all pages", use_container_width=True, key="sidebar_clear"):
            st.session_state.pages = []
            st.session_state.current_page_idx = 0
            st.session_state["cached_pdf"] = None
            st.rerun()

    st.markdown(
        f"""
        <div class="sidebar-foot">
            {icon("shield", size=14, color="currentColor")}
            <span>Private by design · processed locally</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------------
show_flash()

# =============================================================================
# HOME
# =============================================================================
if selected_nav == "Home":
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{icon("sparkles", size=14, color="currentColor")} Your document workspace</div>
            <h1 class="hero-title">Ready to scan <span class="hero-light">your documents?</span></h1>
            <p class="hero-sub">Turn phone photos into clean, flat, readable pages. Detection, shadow removal, OCR and PDF export all happen on this device.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="quick_actions_card"):
        section_title("Quick Actions", "Everything stays on this device")

        action_row_1 = st.columns(2)
        with action_row_1[0]:
            with st.container(border=True, key="quick_scan"):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f"""
                        <div class="action-copy">
                            {icon_tile("scan", "blue")}
                            <div class="action-text">
                                <div class="action-title">Scan Document</div>
                                <p class="action-desc">Capture a new page with your webcam.</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button(":material/north_east:", key="quick_scan_btn", help="Open Scan & Upload"):
                        go_to("Scan & Upload")

        with action_row_1[1]:
            with st.container(border=True, key="quick_upload"):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f"""
                        <div class="action-copy">
                            {icon_tile("upload", "purple")}
                            <div class="action-text">
                                <div class="action-title">Upload Images</div>
                                <p class="action-desc">Bring in JPG, JPEG or PNG files.</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button(":material/north_east:", key="quick_upload_btn", help="Open Scan & Upload"):
                        go_to("Scan & Upload")

        action_row_2 = st.columns(2)
        with action_row_2[0]:
            with st.container(border=True, key="quick_samples"):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f"""
                        <div class="action-copy">
                            {icon_tile("folder", "green")}
                            <div class="action-text">
                                <div class="action-title">Load Sample Images</div>
                                <p class="action-desc">Explore the nine benchmark scenarios.</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button(":material/north_east:", key="quick_samples_btn", help="Load sample images"):
                        with st.spinner("Preparing sample documents…"):
                            count = load_benchmark_samples()
                        st.session_state.flash = f"Loaded {count} benchmark samples."
                        st.rerun()

        with action_row_2[1]:
            with st.container(border=True, key="quick_export"):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f"""
                        <div class="action-copy">
                            {icon_tile("download", "orange")}
                            <div class="action-text">
                                <div class="action-title">Export PDF</div>
                                <p class="action-desc">Compile pages into one clean document.</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button(":material/north_east:", key="quick_export_btn", help="Open Export"):
                        go_to("Export")

    section_title("How it works", "Six small steps from camera roll to clean scan")
    steps = [
        ("scan", "blue", "Capture & scale", "EXIF is corrected and the page is resized for fast boundary detection."),
        ("filter", "purple", "Edge filtering", "Grayscale, Gaussian smoothing and Canny edges isolate the paper perimeter."),
        ("crop", "green", "Quad detection", "The largest four-sided contour is simplified and ordered TL, TR, BR, BL."),
        ("compass", "orange", "Perspective warp", "A homography flattens the page into a straight, upright rectangle."),
        ("sun", "green", "Shadow removal", "Retinex background estimation divides out uneven light and hand shadows."),
        ("download", "blue", "OCR & export", "Tesseract reads the page and the result is packaged as a PDF or image."),
    ]
    step_html = "".join(
        f'<div class="step-card delay-{i}">'
        f'<div class="step-top">{icon_tile(icon_name, tint, "sm")}'
        f'<span class="step-number">0{i + 1}</span></div>'
        f'<div class="step-title">{html.escape(title)}</div>'
        f'<p class="step-desc">{html.escape(desc)}</p>'
        f"</div>"
        for i, (icon_name, tint, title, desc) in enumerate(steps)
    )
    st.markdown(f'<div class="steps-grid">{step_html}</div>', unsafe_allow_html=True)

    with st.container(border=True, key="recent_card"):
        head_col, link_col = st.columns([5, 1])
        with head_col:
            st.markdown(
                '<div class="section-head"><div><div class="section-kicker">This session</div>'
                '<h2 class="section-title-lg">Recent Pages</h2></div></div>',
                unsafe_allow_html=True,
            )
        with link_col:
            if st.session_state.pages:
                if st.button("See more →", key="recent_see_more"):
                    go_to("Multi-Page")

        if not st.session_state.pages:
            st.markdown(
                f"""
                <div class="recent-empty">
                    {icon_tile("file-plus", "purple", "lg")}
                    <div>
                        <h3>No pages yet</h3>
                        <p>Scan a page or load the sample set to see it appear here.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns([1.2, 1, 2])
            with c1:
                if st.button("Scan your first page", type="primary", use_container_width=True, key="home_empty_scan"):
                    go_to("Scan & Upload")
            with c2:
                if st.button("Load samples", use_container_width=True, key="home_empty_samples"):
                    with st.spinner("Preparing sample documents…"):
                        count = load_benchmark_samples()
                    st.session_state.flash = f"Loaded {count} benchmark samples."
                    st.rerun()
        else:
            for page in list(reversed(st.session_state.pages))[:5]:
                render_recent_page_row(page)

# =============================================================================
# SCAN & UPLOAD
# =============================================================================
elif selected_nav == "Scan & Upload":
    page_header(
        "Capture",
        "Scan & Upload",
        "Add pages from your computer or webcam. Every image is orientation-corrected and analysed automatically.",
    )

    with st.container(border=True, key="scan_card"):
        st.markdown(
            f"""
            <div class="upload-hero">
                {icon_tile("upload-cloud", "purple", "lg")}
                <h3>Choose your files</h3>
                <p>Drop images here or browse</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Choose your files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
            help="Phone camera photos are automatically rotated using embedded EXIF metadata.",
        )

        st.markdown(
            """
            <div class="format-pills">
                <span class="format-pill">JPG</span>
                <span class="format-pill">JPEG</span>
                <span class="format-pill">PNG</span>
            </div>
            <div class="divider-label"><span>Or use another method</span></div>
            """,
            unsafe_allow_html=True,
        )

        if render_method_row(
            "device",
            "Import from device",
            "Use the dashed upload area above to add image files.",
            "folder",
            "blue",
        ):
            st.toast("Drop files into the dashed area above, or click Browse files.")

        if render_method_row(
            "camera",
            "Capture with camera",
            "Snap a page with your webcam and add it to this document.",
            "camera",
            "orange",
        ):
            st.session_state.show_camera = not st.session_state.show_camera
            st.rerun()

        camera_pic = None
        if st.session_state.show_camera:
            camera_pic = st.camera_input(
                "Camera capture",
                key=f"camera_{st.session_state.camera_key}",
                help="Hold the page flat and fill most of the frame.",
            )

        st.markdown('<div class="scan-action-gap"></div>', unsafe_allow_html=True)

        if st.button(
            "Add to document",
            type="primary",
            use_container_width=True,
            key="scan_add",
        ):
            added = 0
            if uploaded_files:
                with st.spinner("Detecting document edges…"):
                    for uploaded_file in uploaded_files:
                        raw_bytes = uploaded_file.read()
                        try:
                            bgr = load_image_from_bytes(raw_bytes)
                            add_image_page(bgr, uploaded_file.name)
                            added += 1
                        except Exception as exc:
                            st.error(f"Could not load {uploaded_file.name}: {exc}")
                st.session_state.uploader_key += 1
            elif camera_pic is not None:
                with st.spinner("Detecting document edges…"):
                    bgr = load_image_from_bytes(camera_pic.read())
                    add_image_page(bgr, f"Camera_Scan_{int(time.time())}.jpg")
                    added = 1
                st.session_state.camera_key += 1
                st.session_state.show_camera = False
            else:
                st.warning("Choose an image file or capture a page first.")

            if added:
                st.session_state.flash = f"Added {added} page{'s' if added != 1 else ''} to this document."
                st.rerun()

    if st.session_state.pages:
        with st.container(border=True, key="scan_session"):
            head_col, action_col = st.columns([4, 1])
            with head_col:
                st.markdown(
                    f'<div class="section-head"><div><div class="section-kicker">In this session</div>'
                    f'<h2 class="section-title-lg">{len(st.session_state.pages)} page{"s" if len(st.session_state.pages) != 1 else ""} ready</h2></div></div>',
                    unsafe_allow_html=True,
                )
            with action_col:
                if st.button("Review detection", type="primary", use_container_width=True, key="scan_review"):
                    go_to("Detection & Warp")
            for idx, page in enumerate(st.session_state.pages):
                render_recent_page_row(page)
    else:
        render_empty_state(
            "scan",
            "Nothing uploaded yet",
            "Add your first document image above, or load the bundled benchmark set to explore the full pipeline.",
            "upload-cloud",
            "purple",
        )

# =============================================================================
# DETECTION & WARP
# =============================================================================
elif selected_nav == "Detection & Warp":
    page_header(
        "Geometry",
        "Boundary Detection & Warp",
        "Review the detected paper perimeter, fine-tune corners, and compare the original capture with the rectified page.",
    )

    cur_page = page_selector("detect")
    if cur_page is None:
        render_empty_state(
            "detection",
            "No pages to inspect",
            "Upload a document to see its detected boundary and perspective-corrected result side by side.",
            "crop",
            "blue",
        )
    else:
        status, tone = page_status(cur_page)
        st.markdown(
            f"""
            <div class="inline-meta">
                <span class="badge badge-{tone}">{icon("check", size=13, color="currentColor")}{html.escape(status)}</span>
                <span class="meta-chip">{html.escape(cur_page["name"])}</span>
                <span class="meta-chip">{cur_page["resized_image"].shape[1]} × {cur_page["resized_image"].shape[0]} px working copy</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Streamlit runs top-to-bottom, but the corner sliders are drawn further
        # down the page. Read their latest values from session_state first, so the
        # preview above them reflects the current slider positions immediately
        # instead of lagging one interaction behind.
        _pid = cur_page["id"]
        _ck = [f"{p}_{_pid}" for p in ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y")]
        if all(k in st.session_state for k in _ck):
            cur_page["corners"] = np.array(
                [
                    [st.session_state[_ck[0]], st.session_state[_ck[1]]],
                    [st.session_state[_ck[2]], st.session_state[_ck[3]]],
                    [st.session_state[_ck[4]], st.session_state[_ck[5]]],
                    [st.session_state[_ck[6]], st.session_state[_ck[7]]],
                ],
                dtype=np.float32,
            )
        if f"margin_{_pid}" in st.session_state:
            cur_page["margin_trim"] = float(st.session_state[f"margin_{_pid}"])

        overlay = cur_page["resized_image"].copy()
        poly_pts = cur_page["corners"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [poly_pts], isClosed=True, color=(15, 118, 110), thickness=3)
        for pt, lbl in zip(cur_page["corners"], ["TL", "TR", "BR", "BL"]):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (cx, cy), 7, (91, 111, 214), -1)
            cv2.circle(overlay, (cx, cy), 9, (255, 255, 255), 2)
            cv2.putText(
                overlay,
                lbl,
                (cx + 10, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        corners_full = scale_points(cur_page["corners"], cur_page["scale_factor"])
        warped = four_point_transform(
            cur_page["raw_image"],
            corners_full,
            margin_trim_percent=cur_page.get("margin_trim", 1.0),
        )

        with st.container(key="detect_compare"):
            left, right = st.columns(2, gap="large")
            with left:
                render_compare_panel(
                    "detect_original",
                    "Input",
                    "Original capture",
                    overlay,
                    f"{overlay.shape[1]} × {overlay.shape[0]} px · detected perimeter",
                )
            with right:
                render_compare_panel(
                    "detect_processed",
                    "Output",
                    "Rectified document",
                    warped,
                    f"{warped.shape[1]} × {warped.shape[0]} px · perspective corrected",
                )

        with st.expander("Fine-tune corners and crop", expanded=False):
            st.caption("Adjust the coordinate sliders only if the automatic boundary needs a nudge.")
            rh, rw = cur_page["resized_image"].shape[:2]
            page_id = cur_page["id"]

            c_tl1, c_tl2 = st.columns(2)
            tl_x = c_tl1.slider("Top-left X", 0, rw, int(cur_page["corners"][0][0]), key=f"tl_x_{page_id}")
            tl_y = c_tl2.slider("Top-left Y", 0, rh, int(cur_page["corners"][0][1]), key=f"tl_y_{page_id}")

            c_tr1, c_tr2 = st.columns(2)
            tr_x = c_tr1.slider("Top-right X", 0, rw, int(cur_page["corners"][1][0]), key=f"tr_x_{page_id}")
            tr_y = c_tr2.slider("Top-right Y", 0, rh, int(cur_page["corners"][1][1]), key=f"tr_y_{page_id}")

            c_br1, c_br2 = st.columns(2)
            br_x = c_br1.slider("Bottom-right X", 0, rw, int(cur_page["corners"][2][0]), key=f"br_x_{page_id}")
            br_y = c_br2.slider("Bottom-right Y", 0, rh, int(cur_page["corners"][2][1]), key=f"br_y_{page_id}")

            c_bl1, c_bl2 = st.columns(2)
            bl_x = c_bl1.slider("Bottom-left X", 0, rw, int(cur_page["corners"][3][0]), key=f"bl_x_{page_id}")
            bl_y = c_bl2.slider("Bottom-left Y", 0, rh, int(cur_page["corners"][3][1]), key=f"bl_y_{page_id}")

            cur_page["corners"] = np.array(
                [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]],
                dtype=np.float32,
            )

            cur_page["margin_trim"] = st.slider(
                "Inner margin trim (%)",
                0.0,
                5.0,
                float(cur_page.get("margin_trim", 1.0)),
                0.5,
                key=f"margin_{page_id}",
                help="Trims a thin inner border so desk edges do not appear in the scan.",
            )

            if st.button("Reset to auto-detected corners", key=f"reset_corners_{page_id}"):
                cur_page["corners"] = cur_page["auto_corners"].copy()
                cur_page["margin_trim"] = 1.0
                for prefix in ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y", "margin"):
                    st.session_state.pop(f"{prefix}_{page_id}", None)
                st.rerun()

        with st.expander("View processing stages", expanded=False):
            stages = cur_page.get("stages", {})
            if stages:
                tabs = st.tabs(["Grayscale", "Blur", "Edges", "Contour"])
                with tabs[0]:
                    st.image(stages["gray"], caption="1. Grayscale conversion", use_container_width=True)
                with tabs[1]:
                    st.image(stages["blur"], caption="2. Gaussian smoothing (5×5)", use_container_width=True)
                with tabs[2]:
                    st.image(stages["edges"], caption="3. Edge map (Canny + dilation/closing; Otsu paper mask if a fallback was used)", use_container_width=True)
                with tabs[3]:
                    st.image(
                        bgr_to_rgb(stages["contour_overlay"]),
                        caption="4. Polygonal approximation",
                        use_container_width=True,
                    )

# =============================================================================
# ENHANCEMENT
# =============================================================================
elif selected_nav == "Enhancement":
    page_header(
        "Finish",
        "Image Enhancement",
        "Compare the plain rectified page with the finished scan, then choose a look for every document.",
    )

    cur_page = page_selector("enhance")
    if cur_page is None:
        render_empty_state(
            "enhancement",
            "No page to enhance",
            "Add a document first, then compare shadow-free, colorful, grayscale and black-and-white presets.",
            "sun",
            "orange",
        )
    else:
        warped_raw = warped_preview(cur_page)
        processed = compute_processed_page(cur_page)

        with st.container(key="enhance_compare"):
            left, right = st.columns(2, gap="large")
            with left:
                render_compare_panel(
                    "enhance_original",
                    "Before",
                    "Rectified original",
                    warped_raw,
                    "Unenhanced perspective correction",
                )
            with right:
                render_compare_panel(
                    "enhance_processed",
                    "After",
                    "Enhanced result",
                    processed,
                    f"{cur_page.get('enhancement_mode', 'Enhanced')} preset",
                )

        with st.container(border=True, key="enhance_controls"):
            section_title("Enhancement preset", "Pick a look for this page")
            current_mode = cur_page.get("enhancement_mode", "Enhanced")
            new_mode = st.radio(
                "Enhancement preset",
                MODES,
                index=MODES.index(current_mode) if current_mode in MODES else 0,
                horizontal=True,
                label_visibility="collapsed",
            )
            cur_page["enhancement_mode"] = new_mode

            bw_method = cur_page.get("bw_method", "adaptive")
            if new_mode == "Black & White":
                bw_method = st.radio(
                    "Binarization method",
                    ["adaptive", "otsu"],
                    index=0 if bw_method == "adaptive" else 1,
                    horizontal=True,
                )
                cur_page["bw_method"] = bw_method

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Apply to all pages", type="primary", use_container_width=True, key="apply_all_modes"):
                    for page in st.session_state.pages:
                        page["enhancement_mode"] = new_mode
                        page["bw_method"] = bw_method
                    clear_pdf_cache()
                    st.session_state.flash = f"Applied the {new_mode} preset to all pages."
                    st.rerun()
            with c2:
                if st.button("Continue to OCR", use_container_width=True, key="enhance_to_ocr"):
                    go_to("OCR")

        with st.expander("Technical comparison: global histogram equalization vs. CLAHE"):
            st.markdown(
                "Global equalization uses one cumulative histogram for the whole page and often "
                "amplifies paper grain. CLAHE works on local tiles with a clip limit, keeping the "
                "background clean while lifting character strokes."
            )
            comp1, comp2, comp3 = st.columns(3)
            with comp1:
                st.image(bgr_to_rgb(warped_raw), caption="1. Rectified input", use_container_width=True)
            with comp2:
                st.image(
                    bgr_to_rgb(apply_global_hist_eq(warped_raw)),
                    caption="2. Global equalization · noisier",
                    use_container_width=True,
                )
            with comp3:
                st.image(
                    bgr_to_rgb(apply_clahe(warped_raw, clip_limit=2.0)),
                    caption="3. CLAHE · local and crisp",
                    use_container_width=True,
                )

# =============================================================================
# MULTI-PAGE
# =============================================================================
elif selected_nav == "Multi-Page":
    page_header(
        "Organise",
        "Multi-Page Document",
        "Reorder, rotate and restyle every page before compiling the final PDF.",
    )

    if not st.session_state.pages:
        render_empty_state(
            "multi",
            "Your document is empty",
            "Add a few pages to reorder the sequence, rotate orientations, and assign an enhancement preset to each one.",
            "layers",
            "purple",
        )
    else:
        top_left, top_right = st.columns([3, 1.2])
        with top_left:
            section_title(
                f"{len(st.session_state.pages)} page{'s' if len(st.session_state.pages) != 1 else ''} in this document",
                "Use the arrow, rotate and delete buttons on each card",
            )
        with top_right:
            if st.button(
                "Generate PDF",
                type="primary",
                use_container_width=True,
                key="multi_generate_pdf",
            ):
                with st.spinner("Compiling multi-page PDF…"):
                    processed_pages = [compute_processed_page(page) for page in st.session_state.pages]
                    st.session_state["cached_pdf"] = create_multipage_pdf(
                        processed_pages,
                        rotations=[0] * len(processed_pages),
                        fit_a4=True,
                        jpeg_quality=92,
                    )
                st.session_state.flash = "Multi-page PDF generated."
                st.rerun()

        if st.session_state.get("cached_pdf"):
            st.download_button(
                "Download compiled PDF",
                data=st.session_state["cached_pdf"],
                file_name="SmartScan_Document.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="multi_download_pdf",
            )

        for idx, page in enumerate(st.session_state.pages):
            render_page_card(idx, page, "multi", show_mode=True)

# =============================================================================
# OCR
# =============================================================================
elif selected_nav == "OCR":
    page_header(
        "Read",
        "Optical Character Recognition",
        "Extract editable text, review recognition confidence, and download a clean transcript.",
    )

    ocr_available, ocr_message = configure_tesseract_path()
    if not ocr_available:
        st.error(f"OCR is unavailable: {ocr_message}")
        st.info("Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki, then restart the app.")
    elif not st.session_state.pages:
        render_empty_state(
            "ocr",
            "No page to read",
            "Add a document to extract text with Tesseract, review confidence, and download the transcript.",
            "file-text",
            "green",
        )
    else:
        cur_page = page_selector("ocr")
        scope = st.radio(
            "OCR scope",
            ["Current page", "All pages"],
            horizontal=True,
            key="ocr_scope",
        )

        if st.button("Run OCR extraction", type="primary", key="run_ocr"):
            with st.spinner("Reading text and measuring confidence…"):
                if scope == "Current page":
                    assert cur_page is not None
                    processed = compute_processed_page(cur_page)
                    result = extract_text_and_confidence(processed, preprocess=False)
                    cur_page["ocr_text"] = result["text"]
                    cur_page["ocr_confidence"] = result["mean_confidence"]
                    cur_page["ocr_words"] = result["word_count"]
                else:
                    for page in st.session_state.pages:
                        processed = compute_processed_page(page)
                        result = extract_text_and_confidence(processed, preprocess=False)
                        page["ocr_text"] = result["text"]
                        page["ocr_confidence"] = result["mean_confidence"]
                        page["ocr_words"] = result["word_count"]
            st.session_state.flash = "OCR extraction completed."

        if scope == "Current page" and cur_page is not None:
            confidence = float(cur_page.get("ocr_confidence", 0.0))
            words = int(cur_page.get("ocr_words", 0))
            with st.container(border=True, key="ocr_result"):
                section_title("Analysis result", "Extracted text is editable before download")
                st.markdown(
                    f"""
                    <div class="ocr-stats">
                        <div class="stat-card">
                            <span class="stat-label">Confidence</span>
                            <span class="stat-value">{confidence:.1f}%</span>
                            <span class="confidence-bar"><span style="width:{min(confidence, 100):.1f}%"></span></span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-label">Words</span>
                            <span class="stat-value">{words}</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-label">Page</span>
                            <span class="stat-value stat-value-sm">{html.escape(cur_page["name"])}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ocr_text = cur_page.get("ocr_text", "")
                if not ocr_text:
                    st.info("No text yet. Run OCR extraction to read this page.")
                else:
                    edited_text = st.text_area(
                        "Extracted text",
                        ocr_text,
                        height=340,
                        key=f"ocr_text_{cur_page['id']}_{int(confidence)}_{words}",
                    )
                    cur_page["ocr_text"] = edited_text
                    st.download_button(
                        "Download TXT",
                        data=edited_text,
                        file_name=f"{os.path.splitext(cur_page['name'])[0]}_ocr.txt",
                        mime="text/plain",
                        key="download_active_ocr",
                    )
        else:
            total_words = sum(int(page.get("ocr_words", 0)) for page in st.session_state.pages)
            confidences = [
                float(page.get("ocr_confidence", 0.0))
                for page in st.session_state.pages
                if page.get("ocr_words", 0)
            ]
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            with st.container(border=True, key="ocr_all_result"):
                section_title("Document analysis", "Combined transcript for every page")
                st.markdown(
                    f"""
                    <div class="ocr-stats">
                        <div class="stat-card">
                            <span class="stat-label">Pages</span>
                            <span class="stat-value">{len(st.session_state.pages)}</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-label">Words</span>
                            <span class="stat-value">{total_words}</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-label">Mean confidence</span>
                            <span class="stat-value">{mean_confidence:.1f}%</span>
                            <span class="confidence-bar"><span style="width:{min(mean_confidence, 100):.1f}%"></span></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                blocks = []
                for i, page in enumerate(st.session_state.pages):
                    blocks.append(
                        f"--- PAGE {i + 1}: {page['name']} "
                        f"(confidence {page.get('ocr_confidence', 0.0):.1f}%) ---\n"
                        f"{page.get('ocr_text', '')}"
                    )
                combined = "\n\n".join(blocks).strip()
                if total_words == 0:
                    st.info("No text yet. Run OCR extraction to read the document.")
                else:
                    st.text_area("Combined transcript", combined, height=420, key="ocr_all_text")
                    st.download_button(
                        "Download complete TXT",
                        data=combined,
                        file_name="complete_scanned_document.txt",
                        mime="text/plain",
                        key="download_all_ocr",
                    )

# =============================================================================
# EXPORT
# =============================================================================
elif selected_nav == "Export":
    page_header(
        "Deliver",
        "Export & Download",
        "Compile a polished multi-page PDF or download individual pages as PNG and JPG.",
    )

    if not st.session_state.pages:
        render_empty_state(
            "export",
            "Nothing to export yet",
            "Process at least one page, then compile an A4 PDF or download individual high-resolution scans.",
            "download",
            "orange",
        )
    else:
        section_title(
            f"{len(st.session_state.pages)} page{'s' if len(st.session_state.pages) != 1 else ''} ready for export",
            "Rotate or reorder pages before generating the PDF",
        )
        for idx, page in enumerate(st.session_state.pages):
            render_page_card(idx, page, "export", show_mode=False, compact=True)

        with st.container(border=True, key="pdf_settings"):
            section_title("PDF settings", "A4 fitting is enabled by default")
            c1, c2 = st.columns([1.2, 1])
            with c1:
                fit_a4 = st.checkbox(
                    "Fit onto ISO A4 pages (210 × 297 mm)",
                    value=True,
                    key="export_fit_a4",
                )
            with c2:
                pdf_quality = st.slider(
                    "JPEG quality",
                    70,
                    100,
                    92,
                    key="export_pdf_quality",
                )

            if st.button(
                "Generate PDF",
                type="primary",
                use_container_width=True,
                key="export_generate_pdf",
            ):
                with st.spinner("Compiling multi-page PDF…"):
                    processed_pages = [compute_processed_page(page) for page in st.session_state.pages]
                    st.session_state["cached_pdf"] = create_multipage_pdf(
                        processed_pages,
                        rotations=[0] * len(processed_pages),
                        fit_a4=fit_a4,
                        jpeg_quality=pdf_quality,
                    )
                st.session_state.flash = "PDF generated successfully."
                st.rerun()

            if st.session_state.get("cached_pdf"):
                st.download_button(
                    "Download compiled PDF",
                    data=st.session_state["cached_pdf"],
                    file_name="SmartScan_Document.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="export_download_pdf",
                )

        active_page = page_selector("export_active")
        if active_page is not None:
            with st.container(border=True, key="single_page_export"):
                section_title("Single page export", "Download the active page as a lossless PNG or high-quality JPG")
                active_processed = compute_processed_page(active_page)
                st.markdown(
                    f'<div class="meta-chips"><span class="meta-chip">{html.escape(active_page["name"])}</span>'
                    f'<span class="meta-chip">{active_processed.shape[1]} × {active_processed.shape[0]} px</span></div>',
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    ok_png, png_buf = cv2.imencode(".png", active_processed)
                    if ok_png:
                        st.download_button(
                            "Download PNG",
                            data=png_buf.tobytes(),
                            file_name=f"{os.path.splitext(active_page['name'])[0]}_scanned.png",
                            mime="image/png",
                            use_container_width=True,
                            key="download_png",
                        )
                with c2:
                    ok_jpg, jpg_buf = cv2.imencode(
                        ".jpg", active_processed, [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    if ok_jpg:
                        st.download_button(
                            "Download JPG",
                            data=jpg_buf.tobytes(),
                            file_name=f"{os.path.splitext(active_page['name'])[0]}_scanned.jpg",
                            mime="image/jpeg",
                            use_container_width=True,
                            key="download_jpg",
                        )

# =============================================================================
# ABOUT
# =============================================================================
elif selected_nav == "About":
    page_header(
        "Reference",
        "About SmartScan",
        "A computer-vision document workspace built as a B.Tech Image Processing & Applications project.",
    )

    with st.container(border=True, key="about_overview"):
        st.markdown(
            f"""
            <div class="about-hero">
                {icon_tile("scan", "blue", "lg")}
                <div>
                    <h3>From camera roll to clean scan</h3>
                    <p>SmartScan combines geometric rectification, Retinex illumination correction, local contrast enhancement, OCR and PDF packaging in one local-first workspace. Photos are processed on this machine and never sent to a third-party service.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("Core algorithms", "What happens between upload and export")
    algorithms = [
        ("image", "blue", "EXIF & scaling", "Reads the phone's orientation tag, transposes the pixels, then downscales for fast boundary detection while keeping full resolution for the final warp."),
        ("filter", "purple", "Canny edges", "Gaussian smoothing, Sobel gradients, non-maximum suppression and hysteresis thresholding produce a clean paper perimeter."),
        ("crop", "green", "RDP quadrilateral", "The largest contour is simplified with Ramer–Douglas–Peucker until four convex corners remain, then ordered TL, TR, BR, BL."),
        ("compass", "orange", "Homography warp", "An eight-parameter projective transform maps the photographed quadrilateral onto a flat rectangle."),
        ("sun", "green", "Retinex shadows", "Morphological dilation and median filtering estimate the lighting field; division removes shadows and uneven illumination."),
        ("sliders", "purple", "CLAHE & sharpening", "Local histogram equalization on the LAB luminance channel lifts text contrast, followed by unsharp masking."),
        ("file-text", "blue", "Tesseract OCR", "Automatic page segmentation and LSTM recognition return structured text, word counts and mean confidence."),
        ("download", "orange", "PDF assembly", "Pages are compiled with optional ISO A4 fitting, preserving page order and the chosen compression quality."),
    ]
    algorithm_html = "".join(
        f'<div class="algo-card">{icon_tile(icon_name, tint, "sm")}'
        f'<div><div class="algo-title">{html.escape(title)}</div>'
        f'<p class="algo-desc">{html.escape(desc)}</p></div></div>'
        for icon_name, tint, title, desc in algorithms
    )
    st.markdown(f'<div class="algo-grid">{algorithm_html}</div>', unsafe_allow_html=True)

    with st.container(border=True, key="about_stack"):
        section_title("Installed tech stack", "Live versions from this environment")
        import PIL
        import img2pdf
        import pytesseract

        try:
            tess_version = str(pytesseract.get_tesseract_version())
        except Exception:
            tess_version = "Not installed"

        stack = [
            ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
            ("OpenCV", cv2.__version__),
            ("NumPy", np.__version__),
            ("Streamlit", st.__version__),
            ("Pillow", PIL.__version__),
            ("Tesseract", tess_version),
            ("img2pdf", getattr(img2pdf, "__version__", "0.6.1")),
        ]
        stack_html = "".join(
            f'<div class="stack-row"><span>{html.escape(name)}</span><code>{html.escape(version)}</code></div>'
            for name, version in stack
        )
        st.markdown(f'<div class="stack-list">{stack_html}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True, key="about_limits"):
            section_title("Limitations", "Where the math reaches its edge")
            st.markdown(
                """
                - **Curved pages:** a single homography assumes a flat plane, so book spines and deep folds need non-linear dewarping.
                - **Low-contrast boundaries:** white paper on a white desk can defeat edge detection and fall back to manual corners.
                - **Defocus blur:** lost high-frequency text detail cannot be fully recovered by sharpening.
                - **OCR language:** the default engine is configured for English text.
                """
            )
    with c2:
        with st.container(border=True, key="about_future"):
            section_title("Future scope", "Natural next steps")
            st.markdown(
                """
                - **Learned segmentation:** a lightweight U-Net or document-corner model for busy backgrounds and curled pages.
                - **Multi-language OCR:** language packs and automatic script detection.
                - **Batch folders:** watch a folder and process a whole stack unattended.
                - **Mobile capture:** a PWA companion with on-device edge guidance and offline export.
                - **Searchable PDFs:** embed OCR text as an invisible PDF layer.
                """
            )

st.markdown(
    '<div class="app-footnote">SmartScan · B.Tech IPA project · local-first document imaging</div>',
    unsafe_allow_html=True,
)
