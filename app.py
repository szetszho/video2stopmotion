"""
Video to Stop-Motion Composite — Sports Analysis Tool

Simple 3-step workflow:
  1. Upload video
  2. Trim to the action
  3. Generate composite

Advanced controls are tucked away for power users.
Core engine uses the panoramic pipeline (best quality for moving cameras).

Usage:
    python app.py
"""

import tempfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image

from video_processor import (
    VideoProcessor,
    estimate_background,
    segment_foreground,
    segment_foreground_ai,
    composite_stop_motion,
    auto_select_keyframes,
    auto_select_keyframes_moving,
    PanoramicPipeline,
    _AI_AVAILABLE,
)

# AI status detection
_ai_status = {"available": False, "gpu": False}
try:
    from ai_models import check_ai_status
    _s = check_ai_status()
    _ai_status["available"] = _s.get("onnxruntime_installed", False)
    _ai_status["gpu"] = _s.get("gpu_available", False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_processor: VideoProcessor | None = None
_all_section_frames: list[tuple[int, np.ndarray]] = []
_panoramic_pipeline: PanoramicPipeline | None = None


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Step 1 — Load video
# ---------------------------------------------------------------------------

def load_video(video_file):
    global _processor, _all_section_frames, _panoramic_pipeline
    _all_section_frames = []
    _panoramic_pipeline = None

    if video_file is None:
        return None, "No video uploaded.", 0, 0, 0, 0

    path = video_file if isinstance(video_file, str) else video_file.name if hasattr(video_file, 'name') else video_file
    _processor = VideoProcessor(path)

    mid = _processor.get_frame(_processor.total_frames // 2)
    preview = _bgr_to_rgb(mid) if mid is not None else None

    info = (
        f"**{_processor.width}x{_processor.height}** | "
        f"{_processor.fps:.0f} fps | "
        f"{_processor.duration:.1f}s | "
        f"{_processor.total_frames} frames"
    )
    return (preview, info,
            0, round(_processor.duration, 2),
            0, min(3.0, round(_processor.duration, 2)))


# ---------------------------------------------------------------------------
# Step 2 — Extract section
# ---------------------------------------------------------------------------

def extract_section(start_time, end_time):
    global _all_section_frames, _panoramic_pipeline
    _panoramic_pipeline = None

    if _processor is None:
        return None, "Upload a video first."
    if end_time <= start_time:
        return None, "End time must be after start time."

    _all_section_frames = _processor.get_frames_in_range(start_time, end_time)
    if not _all_section_frames:
        return None, "No frames in that range."

    # Show thumbnails
    step = max(1, len(_all_section_frames) // 12)
    thumbs = _all_section_frames[::step][:12]
    gallery = []
    for idx, frame in thumbs:
        t = idx / _processor.fps
        gallery.append((_bgr_to_rgb(frame), f"{t:.2f}s"))

    info = f"**{len(_all_section_frames)}** frames ({end_time - start_time:.1f}s)"
    return gallery, info


# ---------------------------------------------------------------------------
# Step 3 — Generate composite (one-click)
# ---------------------------------------------------------------------------

def generate(num_keyframes, seg_threshold, morph_size,
             dilate_iter, erode_iter, feather_radius,
             min_object_pct, opacity, enable_shadow,
             backdrop_density, alignment_method,
             seg_method, ai_model):
    """Build panorama + segment + composite in one step."""
    global _panoramic_pipeline

    if _processor is None or not _all_section_frames:
        return None, None, "Upload a video and select a section first."

    frames_bgr = [f for _, f in _all_section_frames]
    min_area = float(min_object_pct) / 100.0
    use_ai = (seg_method == "AI Model") and _AI_AVAILABLE
    n_kf = int(num_keyframes)

    # --- Build panoramic pipeline (align + stitch) ---
    _panoramic_pipeline = PanoramicPipeline(frames_bgr)
    _panoramic_pipeline.align_frames(alignment_method=alignment_method)
    _panoramic_pipeline.build_panorama(
        method="median", backdrop_density=int(backdrop_density))

    pipe = _panoramic_pipeline
    pano_preview = _bgr_to_rgb(pipe.panorama)

    # --- Select keyframes ---
    thresh_auto = max(10, 60 - 30)
    selected_ids = auto_select_keyframes_moving(
        _all_section_frames, pipe.panorama,
        pipe.cumulative_H, pipe.offset_H,
        pipe.canvas_w, pipe.canvas_h,
        num_keyframes=n_kf, threshold=thresh_auto,
    )
    id_to_li = {fid: li for li, (fid, _) in enumerate(_all_section_frames)}
    kf_indices = [id_to_li[fid] for fid in selected_ids if fid in id_to_li]

    if not kf_indices:
        # Fallback to uniform
        total = len(frames_bgr)
        if total <= n_kf:
            kf_indices = list(range(total))
        else:
            kf_indices = [int(i * (total - 1) / (n_kf - 1)) for i in range(n_kf)]

    # --- Generate composite ---
    result_bgr = pipe.generate(
        kf_indices,
        threshold=int(seg_threshold),
        morph_size=int(morph_size),
        dilate_iterations=int(dilate_iter),
        erode_iterations=int(erode_iter),
        feather_radius=int(feather_radius),
        opacity=opacity,
        shadow=enable_shadow,
        use_ai=use_ai,
        ai_model=ai_model if use_ai else "u2netp",
    )

    seg_label = f"AI ({ai_model})" if use_ai else "Background subtraction"
    result_rgb = _bgr_to_rgb(result_bgr)
    info = (
        f"**{len(kf_indices)} subjects** on "
        f"**{pipe.canvas_w}x{pipe.canvas_h}** canvas | "
        f"Segmentation: {seg_label}"
    )
    return result_rgb, pano_preview, info


def save_image(image):
    if image is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="stopmotion_")
    Image.fromarray(image).save(tmp.name, quality=100)
    return tmp.name


# ---------------------------------------------------------------------------
# Example image
# ---------------------------------------------------------------------------

def _make_example():
    w, h = 1600, 500
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        r = y / h
        bg[y, :] = [max(0, int(220 - 80 * r)), max(0, int(190 - 90 * r)),
                     max(0, int(90 - 40 * r))]
    slope = np.array([[0, h], [0, int(h * .72)], [int(w * .3), int(h * .78)],
                       [int(w * .7), int(h * .82)], [w, int(h * .75)], [w, h]], np.int32)
    cv2.fillPoly(bg, [slope], (120, 160, 80))
    result = bg.copy()
    color = (20, 100, 240)
    for i in range(9):
        t = i / 8
        cx = int(80 + t * (w - 160))
        cy = int(h * .55 - 100 * np.sin(t * np.pi * .8))
        rad = np.radians(t * 90 - 30)
        c, s = np.cos(rad), np.sin(rad)
        def rot(dx, dy):
            return (int(cx + dx * c - dy * s), int(cy + dx * s + dy * c))
        cv2.line(result, rot(0, -18), rot(0, 18), color, 5, cv2.LINE_AA)
        cv2.circle(result, rot(0, -28), 9, color, -1, cv2.LINE_AA)
        for a in [(-25, -5), (22, -12)]:
            cv2.line(result, rot(0, -18), rot(*a), color, 3, cv2.LINE_AA)
        for leg in [(-8, 38), (8, 38)]:
            cv2.line(result, rot(0, 18), rot(*leg), color, 4, cv2.LINE_AA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


EXAMPLE = _make_example()


# ---------------------------------------------------------------------------
# Gradio UI — simplified 3-step layout
# ---------------------------------------------------------------------------

CSS = """
.main-title { text-align: center; }
.step-hdr {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 6px 14px; border-radius: 6px; margin: 4px 0;
}
.info-bar { font-size: 0.9em; color: #444; }
"""

with gr.Blocks(title="Stop-Motion Composite", css=CSS,
               theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# Stop-Motion Composite\n"
        "Upload a sports video, pick the action, and generate a multi-exposure image.",
        elem_classes=["main-title"],
    )

    with gr.Row(equal_height=False):
        # ── LEFT: Controls ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=340):

            # Step 1: Upload
            gr.Markdown("### 1. Upload Video", elem_classes=["step-hdr"])
            video_input = gr.Video(label="Video", include_audio=False)
            load_btn = gr.Button("Load", variant="primary", size="sm")
            video_info = gr.Markdown("", elem_classes=["info-bar"])

            # Step 2: Trim
            gr.Markdown("### 2. Select the Action", elem_classes=["step-hdr"])
            with gr.Row():
                start_time = gr.Slider(0, 10, value=0, step=0.05,
                                       label="Start (s)")
                end_time = gr.Slider(0, 10, value=2, step=0.05,
                                     label="End (s)")
            extract_btn = gr.Button("Extract", variant="primary", size="sm")
            section_info = gr.Markdown("", elem_classes=["info-bar"])

            # Step 3: Generate
            gr.Markdown("### 3. Generate", elem_classes=["step-hdr"])
            num_keyframes = gr.Slider(
                3, 15, value=7, step=1, label="Number of Poses",
                info="How many subject snapshots in the final image.",
            )
            generate_btn = gr.Button("Generate Composite",
                                      variant="primary", size="lg")
            composite_info = gr.Markdown("", elem_classes=["info-bar"])

            # Advanced (collapsed)
            with gr.Accordion("Advanced Settings", open=False):
                gr.Markdown("**Alignment**")
                alignment_method = gr.Radio(
                    ["orb", "flow"], value="orb", label="Method",
                    info="ORB = fast. Flow = better on snow/water/sky.",
                )
                backdrop_density = gr.Slider(
                    3, 30, value=12, step=1, label="Backdrop Frames",
                    info="Frames used for panorama. Lower = faster.",
                )

                gr.Markdown("**Segmentation**")
                seg_method = gr.Radio(
                    ["Classical (background subtraction)", "AI Model"],
                    value="Classical (background subtraction)",
                    label="Method",
                )
                ai_model = gr.Dropdown(
                    choices=["u2netp", "u2net", "isnet-general", "rmbg-1.4"],
                    value="u2netp", label="AI Model",
                    info="Only used when AI Model is selected above.",
                )
                seg_threshold = gr.Slider(
                    10, 80, value=35, step=1, label="Threshold",
                    info="Lower = more sensitive. For classical mode.",
                )
                morph_size = gr.Slider(3, 15, value=7, step=2,
                                       label="Cleanup Kernel")
                with gr.Row():
                    dilate_iter = gr.Slider(0, 8, value=0, step=1,
                                            label="Dilate")
                    erode_iter = gr.Slider(0, 8, value=0, step=1,
                                           label="Erode")
                feather_radius = gr.Slider(0, 20, value=3, step=1,
                                           label="Edge Feather")
                min_object_pct = gr.Slider(0.01, 5.0, value=0.1, step=0.01,
                                           label="Min Object %")

                gr.Markdown("**Compositing**")
                opacity = gr.Slider(0.3, 1.0, value=1.0, step=0.05,
                                    label="Opacity")
                enable_shadow = gr.Checkbox(value=True, label="Drop Shadow")

        # ── RIGHT: Output ───────────────────────────────────────────
        with gr.Column(scale=2):
            preview_image = gr.Image(label="Video Preview",
                                     interactive=False, height=200)
            section_gallery = gr.Gallery(label="Extracted Frames",
                                         columns=6, height=160)
            panorama_preview = gr.Image(label="Panoramic Background",
                                        interactive=False, height=200)
            composite_output = gr.Image(label="Result",
                                        interactive=False)
            with gr.Row():
                save_btn = gr.Button("Save PNG", variant="secondary")
                save_output = gr.File(label="Download")

    # ── Example ───────────────────────────────────────────────────────
    with gr.Accordion("Example Output", open=False):
        gr.Image(value=EXAMPLE, label="Example composite",
                 interactive=False, height=280)

    # ── Event wiring ──────────────────────────────────────────────────

    load_btn.click(
        fn=load_video, inputs=[video_input],
        outputs=[preview_image, video_info,
                 start_time, start_time, end_time, end_time],
    )

    extract_btn.click(
        fn=extract_section, inputs=[start_time, end_time],
        outputs=[section_gallery, section_info],
    )

    generate_btn.click(
        fn=generate,
        inputs=[
            num_keyframes, seg_threshold, morph_size,
            dilate_iter, erode_iter, feather_radius,
            min_object_pct, opacity, enable_shadow,
            backdrop_density, alignment_method,
            seg_method, ai_model,
        ],
        outputs=[composite_output, panorama_preview, composite_info],
    )

    save_btn.click(fn=save_image, inputs=[composite_output],
                   outputs=[save_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
