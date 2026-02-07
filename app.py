"""
Video to Stop-Motion Effect — Sports Analysis Tool

A Gradio-based application that converts video clips into stop-motion
composite images showing body movement across time, ideal for sports analysis.

Supports both static and moving camera workflows:
  - Static camera: median background + foreground segmentation
  - Moving camera: homography alignment, panoramic stitching, expanded canvas

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
    DirectCompositePipeline,
    _AI_AVAILABLE,
)

# AI status detection
_ai_status = {"available": False, "providers": [], "models": []}
try:
    from ai_models import check_ai_status
    _ai_status_full = check_ai_status()
    _ai_status["available"] = _ai_status_full.get("onnxruntime_installed", False)
    _ai_status["providers"] = _ai_status_full.get("providers", [])
    _ai_status["models"] = _ai_status_full.get("segmentation_models", [])
    _ai_status["gpu"] = _ai_status_full.get("gpu_available", False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# State helpers
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
    global _processor
    if video_file is None:
        return None, "No video loaded.", 0, 0, 0, 0

    path = video_file if isinstance(video_file, str) else video_file.name if hasattr(video_file, 'name') else video_file
    _processor = VideoProcessor(path)

    mid_frame = _processor.get_frame(_processor.total_frames // 2)
    preview = _bgr_to_rgb(mid_frame) if mid_frame is not None else None

    info = (
        f"**Video loaded**\n"
        f"- Resolution: {_processor.width} x {_processor.height}\n"
        f"- FPS: {_processor.fps:.1f}\n"
        f"- Duration: {_processor.duration:.2f}s\n"
        f"- Total frames: {_processor.total_frames}"
    )
    return (preview, info, 0, round(_processor.duration, 2),
            min(2.0, _processor.duration), round(_processor.duration, 2))


# ---------------------------------------------------------------------------
# Step 2 — Extract section
# ---------------------------------------------------------------------------

def extract_section(start_time, end_time):
    global _all_section_frames, _panoramic_pipeline
    _panoramic_pipeline = None

    if _processor is None:
        return None, "Load a video first."
    if end_time <= start_time:
        return None, "End time must be after start time."

    _all_section_frames = _processor.get_frames_in_range(start_time, end_time)
    if not _all_section_frames:
        return None, "No frames extracted."

    step = max(1, len(_all_section_frames) // 20)
    thumbs = _all_section_frames[::step][:20]
    gallery = []
    for idx, frame in thumbs:
        t = idx / _processor.fps
        gallery.append((_bgr_to_rgb(frame), f"Frame {idx} ({t:.2f}s)"))

    info = f"Extracted **{len(_all_section_frames)}** frames ({start_time:.2f}s – {end_time:.2f}s)"
    return gallery, info


# ---------------------------------------------------------------------------
# Step 3 — Align & stitch panorama
# ---------------------------------------------------------------------------

def align_and_stitch(bg_method, backdrop_density, alignment_method):
    global _panoramic_pipeline
    if not _all_section_frames:
        return None, "Extract a video section first."

    frames_bgr = [f for _, f in _all_section_frames]
    _panoramic_pipeline = PanoramicPipeline(frames_bgr)
    _panoramic_pipeline.align_frames(alignment_method=alignment_method)
    _panoramic_pipeline.build_panorama(method=bg_method,
                                       backdrop_density=int(backdrop_density))

    pano = _panoramic_pipeline.get_panorama_preview()
    if pano is None:
        return None, "Panorama build failed."

    align_label = "Dense Optical Flow" if alignment_method == "flow" else "ORB Features"
    pano_rgb = _bgr_to_rgb(pano)
    info = (
        f"**Panorama built**\n"
        f"- Alignment: **{align_label}**\n"
        f"- Canvas size: {_panoramic_pipeline.canvas_w} x {_panoramic_pipeline.canvas_h}\n"
        f"- Frames used for backdrop: **{min(int(backdrop_density), len(frames_bgr))}** "
        f"of {len(frames_bgr)} total\n"
        f"- Background expanded from {_processor.width}x{_processor.height} "
        f"→ {_panoramic_pipeline.canvas_w}x{_panoramic_pipeline.canvas_h}"
    )
    return pano_rgb, info


# ---------------------------------------------------------------------------
# Step 4 — Pick keyframes
# ---------------------------------------------------------------------------

def pick_keyframes_auto(num_keyframes, sensitivity, camera_mode):
    if _processor is None or not _all_section_frames:
        return None, "Extract a video section first."

    if camera_mode == "Moving Camera" and _panoramic_pipeline is not None:
        pipe = _panoramic_pipeline
        threshold = max(10, 60 - int(sensitivity))
        selected_indices = auto_select_keyframes_moving(
            _all_section_frames, pipe.panorama,
            pipe.cumulative_H, pipe.offset_H,
            pipe.canvas_w, pipe.canvas_h,
            num_keyframes=int(num_keyframes), threshold=threshold,
        )
    else:
        frames_bgr = [f for _, f in _all_section_frames]
        background = estimate_background(frames_bgr, method="median")
        threshold = max(10, 60 - int(sensitivity))
        selected_indices = auto_select_keyframes(
            _all_section_frames, background,
            num_keyframes=int(num_keyframes), threshold=threshold,
        )

    gallery = []
    for idx in selected_indices:
        frame = _processor.get_frame(idx)
        if frame is not None:
            t = idx / _processor.fps
            gallery.append((_bgr_to_rgb(frame), f"Frame {idx} ({t:.2f}s)"))

    indices_str = ", ".join(str(i) for i in selected_indices)
    info = f"Auto-selected **{len(selected_indices)}** keyframes: [{indices_str}]"
    return gallery, info


def pick_keyframes_uniform(num_keyframes):
    if not _all_section_frames:
        return None, "Extract a video section first."

    n = int(num_keyframes)
    total = len(_all_section_frames)
    if total <= n:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]

    gallery = []
    for i in indices:
        idx, frame = _all_section_frames[i]
        t = idx / _processor.fps
        gallery.append((_bgr_to_rgb(frame), f"Frame {idx} ({t:.2f}s)"))

    info = f"Uniformly selected **{len(indices)}** keyframes"
    return gallery, info


# ---------------------------------------------------------------------------
# Step 5 — Generate composite
# ---------------------------------------------------------------------------

def _select_keyframe_indices(num_keyframes, selection_mode, sensitivity,
                              camera_mode):
    """Pick keyframe list-indices from the extracted section frames."""
    frames_bgr = [f for _, f in _all_section_frames]
    total = len(frames_bgr)
    n = int(num_keyframes)

    if selection_mode == "Uniform spacing":
        if total <= n:
            return list(range(total))
        return [int(i * (total - 1) / (n - 1)) for i in range(n)]

    # Auto (movement-based)
    if camera_mode == "Moving Camera" and _panoramic_pipeline is not None:
        pipe = _panoramic_pipeline
        thresh = max(10, 60 - int(sensitivity))
        selected_ids = auto_select_keyframes_moving(
            _all_section_frames, pipe.panorama,
            pipe.cumulative_H, pipe.offset_H,
            pipe.canvas_w, pipe.canvas_h,
            num_keyframes=n, threshold=thresh,
        )
        id_to_li = {fid: li for li, (fid, _) in enumerate(_all_section_frames)}
        return [id_to_li[fid] for fid in selected_ids if fid in id_to_li]

    # Static or no panorama available — use static auto-select
    background = estimate_background(frames_bgr, method="median")
    thresh = max(10, 60 - int(sensitivity))
    selected_ids = auto_select_keyframes(
        _all_section_frames, background, num_keyframes=n, threshold=thresh,
    )
    id_to_li = {fid: li for li, (fid, _) in enumerate(_all_section_frames)}
    return [id_to_li.get(fid, 0) for fid in selected_ids]


def generate_composite(
    num_keyframes, selection_mode, sensitivity, camera_mode,
    pipeline_mode, bg_method, backdrop_density, alignment_method,
    seg_method, ai_model,
    seg_threshold, morph_size, dilate_iter, erode_iter, feather_radius,
    min_object_pct, opacity, enable_shadow,
):
    if _processor is None or not _all_section_frames:
        return None, "Extract a video section first."

    frames_bgr = [f for _, f in _all_section_frames]
    min_area = float(min_object_pct) / 100.0
    use_ai_seg = (seg_method == "AI Model") and _AI_AVAILABLE

    # ── Direct pipeline (new lightweight approach) ────────────────────
    if pipeline_mode in ("Direct (keyframes only)", "Opacity Blend (no segmentation)"):
        kf_indices = _select_keyframe_indices(
            num_keyframes, selection_mode, sensitivity, camera_mode)
        keyframes = [frames_bgr[i] for i in kf_indices]

        if not keyframes:
            return None, "No keyframes could be extracted."

        pipe = DirectCompositePipeline()

        if pipeline_mode == "Opacity Blend (no segmentation)":
            result_bgr = pipe.generate_opacity_blend(
                keyframes,
                alignment_method=alignment_method,
                opacity=opacity,
            )
            result_rgb = _bgr_to_rgb(result_bgr)
            info = (
                f"**Multi-exposure blend** with **{len(keyframes)}** frames.\n"
                f"- Canvas: {pipe.canvas_w} x {pipe.canvas_h}\n"
                f"- Alignment: **{alignment_method}**\n"
                f"- No segmentation — pure opacity layering"
            )
            return result_rgb, info

        seg_mode_str = "ai" if use_ai_seg else "classical"
        result_bgr = pipe.generate(
            keyframes,
            alignment_method=alignment_method,
            seg_method=seg_mode_str,
            ai_model=ai_model,
            backdrop_method=bg_method,
            threshold=int(seg_threshold),
            morph_size=int(morph_size),
            dilate_iterations=int(dilate_iter),
            erode_iterations=int(erode_iter),
            feather_radius=int(feather_radius),
            min_contour_area=min_area,
            opacity=opacity,
            shadow=enable_shadow,
        )

        seg_label = f"AI ({ai_model})" if use_ai_seg else "Classical"
        result_rgb = _bgr_to_rgb(result_bgr)
        info = (
            f"**Direct composite** with **{len(keyframes)}** subjects.\n"
            f"- Canvas: {pipe.canvas_w} x {pipe.canvas_h}\n"
            f"- Alignment: **{alignment_method}** (on {len(keyframes)} frames only)\n"
            f"- Segmentation: **{seg_label}** (on original frames)"
        )
        return result_rgb, info

    # ── Panoramic pipeline (legacy — processes all frames) ────────────
    if camera_mode == "Moving Camera":
        global _panoramic_pipeline
        if _panoramic_pipeline is None:
            _panoramic_pipeline = PanoramicPipeline(frames_bgr)
            _panoramic_pipeline.align_frames(alignment_method=alignment_method)
            _panoramic_pipeline.build_panorama(
                method=bg_method,
                backdrop_density=int(backdrop_density),
            )

        pipe = _panoramic_pipeline
        kf_indices = _select_keyframe_indices(
            num_keyframes, selection_mode, sensitivity, camera_mode)

        result_bgr = pipe.generate(
            kf_indices,
            threshold=int(seg_threshold),
            morph_size=int(morph_size),
            dilate_iterations=int(dilate_iter),
            erode_iterations=int(erode_iter),
            feather_radius=int(feather_radius),
            opacity=opacity,
            shadow=enable_shadow,
            use_ai=use_ai_seg,
            ai_model=ai_model,
        )

        seg_label = f"AI ({ai_model})" if use_ai_seg else "Classical"
        result_rgb = _bgr_to_rgb(result_bgr)
        info = (
            f"**Panoramic composite** with **{len(kf_indices)}** subjects.\n"
            f"- Canvas: {pipe.canvas_w} x {pipe.canvas_h}\n"
            f"- Segmentation: **{seg_label}**\n"
            f"- Aligned all {len(frames_bgr)} frames, backdrop from {min(int(backdrop_density), len(frames_bgr))}"
        )
        return result_rgb, info

    # ── Static camera path ────────────────────────────────────────────
    background = estimate_background(frames_bgr, method=bg_method)
    kf_indices = _select_keyframe_indices(
        num_keyframes, selection_mode, sensitivity, camera_mode)

    keyframe_bgr = []
    keyframe_masks = []
    for li in kf_indices:
        frame = frames_bgr[li]
        if use_ai_seg:
            mask = segment_foreground_ai(
                frame, model_name=ai_model,
                morph_size=int(morph_size),
                dilate_iterations=int(dilate_iter),
                erode_iterations=int(erode_iter),
                feather_radius=int(feather_radius),
                min_contour_area=min_area,
            )
        else:
            mask = segment_foreground(
                frame, background,
                threshold=int(seg_threshold),
                morph_size=int(morph_size),
                min_contour_area=min_area,
                dilate_iterations=int(dilate_iter),
                erode_iterations=int(erode_iter),
                feather_radius=int(feather_radius),
            )
        keyframe_bgr.append(frame)
        keyframe_masks.append(mask)

    if not keyframe_bgr:
        return None, "No keyframes could be processed."

    result_bgr = composite_stop_motion(
        background, keyframe_bgr, keyframe_masks,
        opacity=opacity, shadow=enable_shadow,
    )
    seg_label = f"AI ({ai_model})" if use_ai_seg else "Classical"
    result_rgb = _bgr_to_rgb(result_bgr)
    info = (
        f"Composite with **{len(keyframe_bgr)}** subjects on clean background.\n"
        f"- Segmentation: **{seg_label}**"
    )
    return result_rgb, info


def save_image(image):
    if image is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="stopmotion_")
    Image.fromarray(image).save(tmp.name, quality=100)
    return tmp.name


# ---------------------------------------------------------------------------
# Example image (inline)
# ---------------------------------------------------------------------------

def generate_example_image():
    w, h = 1600, 600
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        r = y / h
        bg[y, :] = [max(0, int(220 - 80 * r)), max(0, int(190 - 90 * r)), max(0, int(90 - 40 * r))]
    slope = np.array([[0, h], [0, int(h * .72)], [int(w * .3), int(h * .78)],
                       [int(w * .7), int(h * .82)], [w, int(h * .75)], [w, h]], np.int32)
    cv2.fillPoly(bg, [slope], (120, 160, 80))
    result = bg.copy()
    num = 9
    color = (20, 100, 240)
    for i in range(num):
        t = i / (num - 1)
        cx = int(80 + t * (w - 160))
        cy = int(h * .65 - 120 * np.sin(t * np.pi * .8))
        rad = np.radians(t * 90 - 30)
        c, s = np.cos(rad), np.sin(rad)
        def rot(dx, dy):
            return (int(cx + dx * c - dy * s), int(cy + dx * s + dy * c))
        cv2.line(result, rot(0, -18), rot(0, 18), color, 5, cv2.LINE_AA)
        cv2.circle(result, rot(0, -28), 9, color, -1, cv2.LINE_AA)
        for a in [(-25, -5), (22, -12)]:
            cv2.line(result, rot(0, -18), rot(*a), color, 3, cv2.LINE_AA)
        for l in [(-8, 38), (8, 38)]:
            cv2.line(result, rot(0, 18), rot(*l), color, 4, cv2.LINE_AA)
    banner = result.copy()
    cv2.rectangle(banner, (0, 0), (w, 55), (60, 40, 20), -1)
    cv2.addWeighted(banner, .6, result, .4, 0, result)
    cv2.putText(result, "PANORAMIC STOP-MOTION  |  Moving Camera", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2, cv2.LINE_AA)
    return _bgr_to_rgb(result)


EXAMPLE_IMAGE = generate_example_image()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.main-title { text-align: center; margin-bottom: 0; }
.step-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 8px 16px; border-radius: 8px; margin-bottom: 8px;
}
.guide-text { font-size: 0.88em; color: #555; margin-top: -4px; }
"""

with gr.Blocks(title="Video → Stop-Motion | Sports Analysis", css=CSS, theme=gr.themes.Soft()) as demo:
    # Build AI status message for header
    _ai_info_parts = []
    if _ai_status["available"]:
        gpu_provs = [p for p in _ai_status["providers"] if "CPU" not in p]
        if gpu_provs:
            _ai_info_parts.append(f"GPU: {gpu_provs[0]}")
        else:
            _ai_info_parts.append("CPU only")
        _ai_info_parts.append(f"{len(_ai_status['models'])} AI models available")
        _ai_badge = " | ".join(_ai_info_parts)
        _ai_header = f"  \n**AI Enhanced** — {_ai_badge}"
    else:
        _ai_header = (
            "  \n*AI models not installed. Install `onnxruntime-gpu` (NVIDIA) or "
            "`onnxruntime` (CPU/Apple) + `huggingface-hub` for AI segmentation.*"
        )

    gr.Markdown(
        "# Video → Stop-Motion Effect\n"
        "### Sports Analysis Tool — Visualize body movement across time\n"
        "Upload a video, select a section, pick keyframes, and generate a composite image "
        "showing the subject at multiple points in time. **Supports both static and moving cameras** — "
        "moving camera footage is stitched into an expanded panoramic background."
        + _ai_header,
        elem_classes=["main-title"],
    )

    with gr.Tabs():
        # ── Example ───────────────────────────────────────────────────
        with gr.TabItem("Example Output"):
            gr.Markdown("### What the output looks like")
            gr.Image(value=EXAMPLE_IMAGE, label="Panoramic stop-motion composite", interactive=False)
            gr.Markdown(
                "**Direct pipeline (recommended):**\n"
                "1. Pick keyframes from the video section\n"
                "2. Align only those keyframes to each other (fast — processes 7 frames, not 150)\n"
                "3. Segment the subject on each **original, un-warped frame** (best quality)\n"
                "4. Place subjects at their aligned positions on an expanded canvas\n\n"
                "**Opacity Blend**: Same as Direct but skips segmentation — layers frames with "
                "transparency for a classic multi-exposure / chronophotography look.\n\n"
                "**Panoramic (legacy)**: Aligns ALL frames, builds full panorama backdrop, "
                "then segments via background subtraction. Slower but shows panorama preview.\n\n"
                "**Static camera** uses pixel-wise median to estimate the background."
            )

        # ── Main workflow ─────────────────────────────────────────────
        with gr.TabItem("Create Stop-Motion"):

            # Camera mode + pipeline mode
            gr.Markdown("## Camera & Pipeline Mode", elem_classes=["step-header"])
            with gr.Row():
                camera_mode = gr.Radio(
                    ["Moving Camera", "Static Camera"],
                    value="Moving Camera",
                    label="Camera Mode",
                    info="Moving Camera: camera pans/tracks the subject. "
                         "Static Camera: camera on a tripod.",
                )
                pipeline_mode = gr.Radio(
                    [
                        "Direct (keyframes only)",
                        "Opacity Blend (no segmentation)",
                        "Panoramic (legacy)",
                    ],
                    value="Direct (keyframes only)",
                    label="Pipeline Mode",
                    info="**Direct**: aligns only keyframes, segments on originals — fast & clean. "
                         "**Opacity Blend**: layers keyframes with transparency, no segmentation — fastest. "
                         "**Panoramic**: aligns ALL frames, builds full panorama — slower but keeps panorama preview.",
                )

            # Step 1
            gr.Markdown("## Step 1 — Import Video", elem_classes=["step-header"])
            gr.Markdown(
                "Upload a video clip. For best results use footage where the **subject moves across the frame** "
                "(e.g. a jump, sprint, swing). Clips of 1–5 seconds work best.",
                elem_classes=["guide-text"],
            )
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video", include_audio=False)
                    load_btn = gr.Button("Load Video", variant="primary")
                with gr.Column(scale=1):
                    preview_image = gr.Image(label="Video Preview", interactive=False)
                    video_info = gr.Markdown("No video loaded.")

            # Step 2
            gr.Markdown("## Step 2 — Select Video Section", elem_classes=["step-header"])
            gr.Markdown(
                "Choose the start and end times to isolate the action you want to capture. "
                "Keep it tight — just the movement itself (takeoff to landing, first step to finish, etc.).",
                elem_classes=["guide-text"],
            )
            with gr.Row():
                start_time = gr.Slider(0, 10, value=0, step=0.05, label="Start Time (seconds)")
                end_time = gr.Slider(0, 10, value=2, step=0.05, label="End Time (seconds)")
            extract_btn = gr.Button("Extract Section", variant="primary")
            section_info = gr.Markdown("")
            section_gallery = gr.Gallery(label="Section Frames", columns=5, height=250)

            # Step 3: Panorama
            gr.Markdown("## Step 3 — Align & Stitch Panorama", elem_classes=["step-header"])
            gr.Markdown(
                "*(Moving Camera only — skip for Static Camera)*\n\n"
                "Aligns all frames to a common coordinate system and builds a wide panoramic backdrop. "
                "The panorama only needs to be **good enough as an anchor** for placing subjects — "
                "not pixel-perfect. Use a low **Backdrop Density** for speed, or increase it if "
                "the background looks patchy.",
                elem_classes=["guide-text"],
            )
            with gr.Row():
                bg_method = gr.Radio(
                    ["median", "overlay"], value="median",
                    label="Background Blend",
                    info="median: averages overlapping pixels, removes the subject from the backdrop. "
                         "overlay: just paints frames on top, fastest but subject may ghost through.",
                )
                backdrop_density = gr.Slider(
                    3, 30, value=12, step=1,
                    label="Backdrop Density",
                    info="How many evenly-spaced frames to use for building the panorama. "
                         "Lower = faster & less memory (good for 5+ second clips). "
                         "Higher = smoother backdrop. 8–15 is usually enough.",
                )
                alignment_method = gr.Radio(
                    ["orb", "flow"], value="orb",
                    label="Alignment Method",
                    info="ORB: fast feature-based alignment (works well with textured backgrounds). "
                         "Flow: dense optical flow — more robust on uniform/textureless backgrounds "
                         "(snow, water, sky) but slightly slower.",
                    interactive=True,
                )
            stitch_btn = gr.Button("Align & Build Panorama", variant="primary")
            panorama_info = gr.Markdown("")
            panorama_preview = gr.Image(label="Panoramic Background Preview", interactive=False)

            # Step 4: Keyframes
            gr.Markdown("## Step 4 — Select Keyframes", elem_classes=["step-header"])
            gr.Markdown(
                "Choose how many subject poses to include and how they're picked.\n\n"
                "- **Auto** analyses the subject's position and picks frames that are maximally "
                "spread across the image — best for even visual spacing.\n"
                "- **Uniform** picks frames at equal time intervals — best when you want "
                "consistent temporal spacing.",
                elem_classes=["guide-text"],
            )
            with gr.Row():
                with gr.Column():
                    num_keyframes = gr.Slider(
                        3, 15, value=7, step=1,
                        label="Number of Keyframes",
                        info="How many subject poses appear in the final image. "
                             "More = denser trajectory, fewer = cleaner image.",
                    )
                    selection_mode = gr.Radio(
                        ["Auto (movement-based)", "Uniform spacing"],
                        value="Auto (movement-based)",
                        label="Selection Mode",
                    )
                    sensitivity = gr.Slider(
                        10, 50, value=30, step=1,
                        label="Detection Sensitivity",
                        info="How aggressively the auto-selector detects the subject. "
                             "Increase if the subject is faint or similar to the background. "
                             "Decrease if noise is being picked up.",
                    )
                with gr.Column():
                    keyframe_btn = gr.Button("Preview Keyframes", variant="secondary")
                    keyframe_info = gr.Markdown("")
            keyframe_gallery = gr.Gallery(label="Selected Keyframes", columns=4, height=250)

            # Step 5: Generate
            gr.Markdown("## Step 5 — Generate Stop-Motion Image", elem_classes=["step-header"])
            gr.Markdown(
                "Fine-tune segmentation and compositing. The controls below affect how cleanly "
                "the subject is cut out and blended onto the background.",
                elem_classes=["guide-text"],
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Segmentation Method")
                    seg_method = gr.Radio(
                        ["Classical (background subtraction)", "AI Model"],
                        value="Classical (background subtraction)",
                        label="Segmentation Method",
                        info="Classical: uses background subtraction (fast, no downloads). "
                             "AI Model: uses a neural network for single-image segmentation — "
                             "produces cleaner masks, especially for hair, equipment, and complex poses. "
                             "Requires onnxruntime + huggingface-hub.",
                        interactive=True,
                    )
                    ai_model_choices = ["u2netp", "u2net", "isnet-general", "rmbg-1.4"]
                    ai_model = gr.Dropdown(
                        choices=ai_model_choices,
                        value="u2netp",
                        label="AI Model",
                        info="u2netp: lightweight (4.7 MB), fast. "
                             "u2net: full model (176 MB), better quality. "
                             "isnet-general: IS-Net (176 MB), good for people. "
                             "rmbg-1.4: RMBG (176 MB), production quality.",
                        interactive=True,
                        visible=True,
                    )

                    gr.Markdown("### Segmentation Controls")
                    gr.Markdown(
                        "These controls determine how the subject is separated from the background. "
                        "Adjust them if the subject has missing parts or if background leaks through.",
                        elem_classes=["guide-text"],
                    )
                    seg_threshold = gr.Slider(
                        10, 80, value=35, step=1,
                        label="Segmentation Threshold",
                        info="Pixel-difference cutoff for detecting the subject. "
                             "LOWER = picks up fainter edges (try 15–25 for subjects that blend with the background). "
                             "HIGHER = stricter, removes background noise (try 40–60 if too much junk is detected).",
                    )
                    morph_size = gr.Slider(
                        3, 15, value=7, step=2,
                        label="Morph Kernel Size",
                        info="Size of the cleanup kernel. Larger values fill gaps in the mask "
                             "(good if the subject has holes) but can merge nearby objects. "
                             "Start at 7; increase to 11–15 for distant/small subjects.",
                    )
                    dilate_iter = gr.Slider(
                        0, 8, value=0, step=1,
                        label="Dilate (expand mask)",
                        info="Grow the subject outline outward. Use 1–3 to recover clipped "
                             "edges (hair, equipment, limbs). 0 = no extra dilation.",
                    )
                    erode_iter = gr.Slider(
                        0, 8, value=0, step=1,
                        label="Erode (shrink mask)",
                        info="Shrink the subject outline inward. Use 1–3 to remove thin "
                             "background halo around the subject. 0 = no extra erosion.",
                    )
                    feather_radius = gr.Slider(
                        0, 20, value=3, step=1,
                        label="Edge Feathering",
                        info="Softens the edge of the cut-out. 0 = hard pixel edge (may look jagged). "
                             "3–5 = natural soft blend. 10–20 = very soft/dreamy edge.",
                    )
                    min_object_pct = gr.Slider(
                        0.01, 5.0, value=0.1, step=0.01,
                        label="Min Object Size (%)",
                        info="Discard detected blobs smaller than this percentage of the image area. "
                             "Increase to 0.5–2.0 if small noise patches appear. "
                             "Decrease to 0.01–0.05 for small/distant subjects.",
                    )

                    gr.Markdown("### Compositing Controls")
                    gr.Markdown(
                        "These affect how the cut-out subjects are placed onto the background.",
                        elem_classes=["guide-text"],
                    )
                    opacity = gr.Slider(
                        0.3, 1.0, value=1.0, step=0.05,
                        label="Subject Opacity",
                        info="1.0 = fully opaque subjects. Lower values make subjects semi-transparent, "
                             "useful for showing the background through overlapping poses.",
                    )
                    enable_shadow = gr.Checkbox(
                        value=True, label="Add Drop Shadow",
                        info="Adds a subtle shadow behind each subject for depth and separation.",
                    )

                    generate_btn = gr.Button("Generate Composite", variant="primary", size="lg")
                    composite_info = gr.Markdown("")

                with gr.Column(scale=3):
                    composite_output = gr.Image(label="Stop-Motion Composite", interactive=False)
                    save_btn = gr.Button("Save Full Resolution", variant="secondary")
                    save_output = gr.File(label="Download")

    # ── Event wiring ──────────────────────────────────────────────────

    load_btn.click(
        fn=load_video,
        inputs=[video_input],
        outputs=[preview_image, video_info, start_time, start_time, end_time, end_time],
    )

    extract_btn.click(
        fn=extract_section,
        inputs=[start_time, end_time],
        outputs=[section_gallery, section_info],
    )

    stitch_btn.click(
        fn=align_and_stitch,
        inputs=[bg_method, backdrop_density, alignment_method],
        outputs=[panorama_preview, panorama_info],
    )

    def _preview_keyframes(num_kf, mode, sens, cam):
        if mode == "Auto (movement-based)":
            return pick_keyframes_auto(num_kf, sens, cam)
        return pick_keyframes_uniform(num_kf)

    keyframe_btn.click(
        fn=_preview_keyframes,
        inputs=[num_keyframes, selection_mode, sensitivity, camera_mode],
        outputs=[keyframe_gallery, keyframe_info],
    )

    generate_btn.click(
        fn=generate_composite,
        inputs=[
            num_keyframes, selection_mode, sensitivity, camera_mode,
            pipeline_mode, bg_method, backdrop_density, alignment_method,
            seg_method, ai_model,
            seg_threshold, morph_size, dilate_iter, erode_iter, feather_radius,
            min_object_pct, opacity, enable_shadow,
        ],
        outputs=[composite_output, composite_info],
    )

    save_btn.click(
        fn=save_image,
        inputs=[composite_output],
        outputs=[save_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
