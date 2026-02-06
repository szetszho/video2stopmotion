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

import os
import tempfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image

from video_processor import (
    VideoProcessor,
    estimate_background,
    segment_foreground,
    composite_stop_motion,
    auto_select_keyframes,
    auto_select_keyframes_moving,
    PanoramicPipeline,
)

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
    """Load video and return metadata + preview frame."""
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

    return (
        preview,
        info,
        0,
        round(_processor.duration, 2),
        min(2.0, _processor.duration),
        round(_processor.duration, 2),
    )


# ---------------------------------------------------------------------------
# Step 2 — Select section & extract frames
# ---------------------------------------------------------------------------

def extract_section(start_time, end_time):
    """Extract frames from the selected time range and show thumbnails."""
    global _all_section_frames, _panoramic_pipeline
    _panoramic_pipeline = None  # reset on new section

    if _processor is None:
        return None, "Load a video first."
    if end_time <= start_time:
        return None, "End time must be after start time."

    _all_section_frames = _processor.get_frames_in_range(start_time, end_time)
    if not _all_section_frames:
        return None, "No frames extracted."

    step = max(1, len(_all_section_frames) // 20)
    thumbs = _all_section_frames[::step][:20]
    gallery_images = []
    for idx, frame in thumbs:
        rgb = _bgr_to_rgb(frame)
        t = idx / _processor.fps
        gallery_images.append((rgb, f"Frame {idx} ({t:.2f}s)"))

    info = f"Extracted **{len(_all_section_frames)}** frames ({start_time:.2f}s – {end_time:.2f}s)"
    return gallery_images, info


# ---------------------------------------------------------------------------
# Step 3 — Align & build panorama (moving camera)
# ---------------------------------------------------------------------------

def align_and_stitch(bg_method):
    """For moving camera: align all section frames and build panoramic background."""
    global _panoramic_pipeline
    if not _all_section_frames:
        return None, "Extract a video section first."

    frames_bgr = [f for _, f in _all_section_frames]
    _panoramic_pipeline = PanoramicPipeline(frames_bgr)
    _panoramic_pipeline.align_frames()
    _panoramic_pipeline.build_panorama(method=bg_method)

    pano = _panoramic_pipeline.get_panorama_preview()
    if pano is None:
        return None, "Panorama build failed."

    pano_rgb = _bgr_to_rgb(pano)
    info = (
        f"**Panorama built**\n"
        f"- Canvas size: {_panoramic_pipeline.canvas_w} x {_panoramic_pipeline.canvas_h}\n"
        f"- Frames aligned: {len(frames_bgr)}\n"
        f"- Background expanded from {_processor.width}x{_processor.height} "
        f"to {_panoramic_pipeline.canvas_w}x{_panoramic_pipeline.canvas_h}"
    )
    return pano_rgb, info


# ---------------------------------------------------------------------------
# Step 4 — Pick keyframes
# ---------------------------------------------------------------------------

def pick_keyframes_auto(num_keyframes, sensitivity, camera_mode):
    """Automatically select keyframes based on subject movement."""
    if _processor is None or not _all_section_frames:
        return None, "Extract a video section first."

    if camera_mode == "Moving Camera" and _panoramic_pipeline is not None:
        pipe = _panoramic_pipeline
        threshold = max(10, 60 - int(sensitivity))
        selected_indices = auto_select_keyframes_moving(
            _all_section_frames, pipe.panorama,
            pipe.cumulative_H, pipe.offset_H,
            pipe.canvas_w, pipe.canvas_h,
            num_keyframes=int(num_keyframes),
            threshold=threshold,
        )
    else:
        frames_bgr = [f for _, f in _all_section_frames]
        background = estimate_background(frames_bgr, method="median")
        threshold = max(10, 60 - int(sensitivity))
        selected_indices = auto_select_keyframes(
            _all_section_frames, background,
            num_keyframes=int(num_keyframes),
            threshold=threshold,
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
    """Select keyframes at uniform intervals."""
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

def generate_composite(
    num_keyframes,
    selection_mode,
    sensitivity,
    camera_mode,
    bg_method,
    seg_threshold,
    morph_size,
    opacity,
    enable_shadow,
):
    """Generate the final stop-motion composite image."""
    if _processor is None or not _all_section_frames:
        return None, "Extract a video section first."

    frames_bgr = [f for _, f in _all_section_frames]

    # ── Moving camera path ────────────────────────────────────────────
    if camera_mode == "Moving Camera":
        global _panoramic_pipeline
        # Build pipeline if not already done
        if _panoramic_pipeline is None:
            _panoramic_pipeline = PanoramicPipeline(frames_bgr)
            _panoramic_pipeline.align_frames()
            _panoramic_pipeline.build_panorama(method=bg_method)

        pipe = _panoramic_pipeline

        # Select keyframe list-indices (positions within section frames)
        if selection_mode == "Auto (movement-based)":
            thresh_auto = max(10, 60 - int(sensitivity))
            selected_frame_ids = auto_select_keyframes_moving(
                _all_section_frames, pipe.panorama,
                pipe.cumulative_H, pipe.offset_H,
                pipe.canvas_w, pipe.canvas_h,
                num_keyframes=int(num_keyframes),
                threshold=thresh_auto,
            )
            # Convert frame IDs to list indices
            id_to_li = {fid: li for li, (fid, _) in enumerate(_all_section_frames)}
            kf_list_indices = [id_to_li[fid] for fid in selected_frame_ids if fid in id_to_li]
        else:
            total = len(frames_bgr)
            n = int(num_keyframes)
            if total <= n:
                kf_list_indices = list(range(total))
            else:
                kf_list_indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]

        result_bgr = pipe.generate(
            kf_list_indices,
            threshold=int(seg_threshold),
            morph_size=int(morph_size),
            opacity=opacity,
            shadow=enable_shadow,
        )

        result_rgb = _bgr_to_rgb(result_bgr)
        info = (
            f"**Panoramic composite** generated with **{len(kf_list_indices)}** subjects.\n"
            f"- Canvas: {pipe.canvas_w} x {pipe.canvas_h}\n"
            f"- Background expanded from original {_processor.width}x{_processor.height}"
        )
        return result_rgb, info

    # ── Static camera path ────────────────────────────────────────────
    background = estimate_background(frames_bgr, method=bg_method)

    if selection_mode == "Auto (movement-based)":
        threshold_auto = max(10, 60 - int(sensitivity))
        selected_indices = auto_select_keyframes(
            _all_section_frames, background,
            num_keyframes=int(num_keyframes),
            threshold=threshold_auto,
        )
    else:
        total = len(frames_bgr)
        n = int(num_keyframes)
        if total <= n:
            idxs = list(range(total))
        else:
            idxs = [int(i * (total - 1) / (n - 1)) for i in range(n)]
        selected_indices = [_all_section_frames[i][0] for i in idxs]

    keyframe_bgr = []
    keyframe_masks = []
    for idx in selected_indices:
        frame = _processor.get_frame(idx)
        if frame is not None:
            mask = segment_foreground(
                frame, background,
                threshold=int(seg_threshold),
                morph_size=int(morph_size),
            )
            keyframe_bgr.append(frame)
            keyframe_masks.append(mask)

    if not keyframe_bgr:
        return None, "No keyframes could be processed."

    result_bgr = composite_stop_motion(
        background, keyframe_bgr, keyframe_masks,
        opacity=opacity, shadow=enable_shadow,
    )
    result_rgb = _bgr_to_rgb(result_bgr)
    info = f"Composite generated with **{len(keyframe_bgr)}** subjects on clean background."
    return result_rgb, info


def save_image(image):
    """Save the composite image to a temporary file and return the path."""
    if image is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="stopmotion_")
    Image.fromarray(image).save(tmp.name, quality=100)
    return tmp.name


# ---------------------------------------------------------------------------
# Example image generator (synthetic panoramic example)
# ---------------------------------------------------------------------------

def generate_example_image():
    """Generate a synthetic example of a panoramic stop-motion output."""
    w, h = 1600, 600
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Wide sky gradient
    for y in range(h):
        ratio = y / h
        b = int(220 - 80 * ratio)
        g = int(190 - 90 * ratio)
        r = int(90 - 40 * ratio)
        bg[y, :] = [max(0, b), max(0, g), max(0, r)]

    # Ground
    slope_pts = np.array([
        [0, h], [0, int(h * 0.72)],
        [int(w * 0.3), int(h * 0.78)],
        [int(w * 0.7), int(h * 0.82)],
        [w, int(h * 0.75)], [w, h],
    ], np.int32)
    cv2.fillPoly(bg, [slope_pts], (120, 160, 80))

    # Panoramic expansion indicator arrows
    cv2.arrowedLine(bg, (20, h // 2), (60, h // 2), (200, 200, 255), 2, tipLength=0.5)
    cv2.arrowedLine(bg, (w - 20, h // 2), (w - 60, h // 2), (200, 200, 255), 2, tipLength=0.5)

    result = bg.copy()

    # Draw athlete poses across wide panoramic canvas
    num_poses = 9
    color = (20, 100, 240)  # Orange BGR
    for i in range(num_poses):
        t = i / (num_poses - 1)
        cx = int(80 + t * (w - 160))
        cy = int(h * 0.65 - 120 * np.sin(t * np.pi * 0.8))

        rad = np.radians(t * 90 - 30)
        cos_r, sin_r = np.cos(rad), np.sin(rad)

        def rot(dx, dy):
            rx = int(cx + dx * cos_r - dy * sin_r)
            ry = int(cy + dx * sin_r + dy * cos_r)
            return (rx, ry)

        # Shadow
        sy = int(h * 0.75 + (1 - np.sin(t * np.pi * 0.8)) * 15)
        overlay = result.copy()
        cv2.ellipse(overlay, (cx, sy), (18, 5), 0, 0, 360, (80, 120, 50), -1)
        cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)

        # Body
        p_shoulder = rot(0, -18)
        p_hip = rot(0, 18)
        cv2.line(result, p_shoulder, p_hip, color, 5, cv2.LINE_AA)
        p_head = rot(0, -28)
        cv2.circle(result, p_head, 9, color, -1, cv2.LINE_AA)
        for adx, ady in [(-25, -5), (22, -12)]:
            arm_end = rot(adx, ady)
            cv2.line(result, p_shoulder, arm_end, color, 3, cv2.LINE_AA)
        for ldx, ldy in [(-8, 38), (8, 38)]:
            leg_end = rot(ldx, ldy)
            cv2.line(result, p_hip, leg_end, color, 4, cv2.LINE_AA)

        # Time label
        label = f"t={t * 3.0:.1f}s"
        cv2.putText(result, label, (cx - 18, cy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Trajectory arc
    for i in range(300):
        t = i / 299
        px = int(80 + t * (w - 160))
        py = int(h * 0.65 - 120 * np.sin(t * np.pi * 0.8))
        if i % 6 < 3:
            cv2.circle(result, (px, py), 1, (200, 210, 255), -1, cv2.LINE_AA)

    # Banner
    banner = result.copy()
    cv2.rectangle(banner, (0, 0), (w, 60), (60, 40, 20), -1)
    cv2.addWeighted(banner, 0.6, result, 0.4, 0, result)

    cv2.putText(result, "PANORAMIC STOP-MOTION  |  Moving Camera", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"{num_poses} keyframes stitched onto expanded panoramic background",
                (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1, cv2.LINE_AA)

    # Frame boundary indicators
    frame_w = _approx_frame_w(w, num_poses)
    for i in range(1, 4):
        x = int(w * i / 4)
        cv2.line(result, (x, 62), (x, 72), (180, 180, 200), 1, cv2.LINE_AA)

    return _bgr_to_rgb(result)


def _approx_frame_w(canvas_w, n):
    return canvas_w // max(1, n - 1)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_IMAGE = generate_example_image()

CSS = """
.main-title { text-align: center; margin-bottom: 0; }
.step-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
}
"""

with gr.Blocks(title="Video → Stop-Motion | Sports Analysis", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Video → Stop-Motion Effect\n"
        "### Sports Analysis Tool — Visualize body movement across time\n"
        "Upload a video, select a section, pick keyframes, and generate a composite image "
        "showing the subject at multiple points in time. **Supports both static and moving cameras** — "
        "moving camera footage is stitched into an expanded panoramic background.",
        elem_classes=["main-title"],
    )

    with gr.Tabs():
        # ── Tab: Example ──────────────────────────────────────────────
        with gr.TabItem("Example Output"):
            gr.Markdown("### What the output looks like")
            gr.Markdown(
                "The tool extracts keyframes from a video clip and composites each "
                "frame's subject onto a clean background. For **moving camera** footage, "
                "frames are aligned via homography and stitched into a wide panoramic canvas."
            )
            gr.Image(value=EXAMPLE_IMAGE, label="Example: Panoramic stop-motion composite (moving camera)", interactive=False)
            gr.Markdown(
                "**How it works (moving camera):**\n"
                "1. Feature points (ORB) are detected and matched between consecutive frames\n"
                "2. Homographies align all frames to a common reference coordinate system\n"
                "3. Aligned frames are stitched into an **expanded panoramic background** (wider than any single frame)\n"
                "4. The moving subject is segmented by diffing each warped keyframe against the panorama\n"
                "5. All segmented subjects are composited onto the panoramic background\n\n"
                "**Static camera** mode uses the simpler median-background approach."
            )

        # ── Tab: Main workflow ────────────────────────────────────────
        with gr.TabItem("Create Stop-Motion"):

            # Camera mode
            gr.Markdown("## Camera Mode", elem_classes=["step-header"])
            camera_mode = gr.Radio(
                ["Moving Camera", "Static Camera"],
                value="Moving Camera",
                label="Camera Mode",
                info="Moving Camera: pans/follows subject → panoramic stitching. "
                     "Static Camera: tripod/fixed → median background.",
            )

            # Step 1: Load video
            gr.Markdown("## Step 1 — Import Video", elem_classes=["step-header"])
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video", include_audio=False)
                    load_btn = gr.Button("Load Video", variant="primary")
                with gr.Column(scale=1):
                    preview_image = gr.Image(label="Video Preview", interactive=False)
                    video_info = gr.Markdown("No video loaded.")

            # Step 2: Select section
            gr.Markdown("## Step 2 — Select Video Section", elem_classes=["step-header"])
            with gr.Row():
                start_time = gr.Slider(0, 10, value=0, step=0.05, label="Start Time (seconds)")
                end_time = gr.Slider(0, 10, value=2, step=0.05, label="End Time (seconds)")
            extract_btn = gr.Button("Extract Section", variant="primary")
            section_info = gr.Markdown("")
            section_gallery = gr.Gallery(label="Section Frames", columns=5, height=250)

            # Step 3: Align & stitch (moving camera)
            gr.Markdown("## Step 3 — Align & Stitch Panorama", elem_classes=["step-header"])
            gr.Markdown(
                "*(Moving Camera only)* Aligns all frames using feature matching and "
                "builds an expanded panoramic background. Skip this step for static camera."
            )
            with gr.Row():
                bg_method = gr.Radio(["median", "overlay"], value="median", label="Background Blend Method",
                                     info="median: cleaner (removes subject). overlay: faster.")
                stitch_btn = gr.Button("Align & Build Panorama", variant="primary")
            panorama_info = gr.Markdown("")
            panorama_preview = gr.Image(label="Panoramic Background Preview", interactive=False)

            # Step 4: Keyframe selection
            gr.Markdown("## Step 4 — Select Keyframes", elem_classes=["step-header"])
            with gr.Row():
                with gr.Column():
                    num_keyframes = gr.Slider(3, 15, value=7, step=1, label="Number of Keyframes")
                    selection_mode = gr.Radio(
                        ["Auto (movement-based)", "Uniform spacing"],
                        value="Auto (movement-based)",
                        label="Selection Mode",
                    )
                    sensitivity = gr.Slider(10, 50, value=30, step=1,
                                            label="Detection Sensitivity (higher = more sensitive)")
                with gr.Column():
                    keyframe_btn = gr.Button("Preview Keyframes", variant="secondary")
                    keyframe_info = gr.Markdown("")
            keyframe_gallery = gr.Gallery(label="Selected Keyframes", columns=4, height=250)

            # Step 5: Generate composite
            gr.Markdown("## Step 5 — Generate Stop-Motion Image", elem_classes=["step-header"])
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Fine-tune Parameters")
                    seg_threshold = gr.Slider(10, 80, value=35, step=1, label="Segmentation Threshold")
                    morph_size = gr.Slider(3, 15, value=7, step=2, label="Morph Kernel Size")
                    opacity = gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="Subject Opacity")
                    enable_shadow = gr.Checkbox(value=True, label="Add Drop Shadow")
                    generate_btn = gr.Button("Generate Composite", variant="primary", size="lg")
                    composite_info = gr.Markdown("")
                with gr.Column(scale=2):
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
        inputs=[bg_method],
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
            bg_method, seg_threshold, morph_size, opacity, enable_shadow,
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
