"""
Video to Stop-Motion Effect — Sports Analysis Tool

A Gradio-based application that converts video clips into stop-motion
composite images showing body movement across time, ideal for sports analysis.

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
)

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_processor: VideoProcessor | None = None
_all_section_frames: list[tuple[int, np.ndarray]] = []


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


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
        preview,                        # preview_image
        info,                           # video_info
        0,                              # start_time slider value
        round(_processor.duration, 2),  # start_time maximum
        min(2.0, _processor.duration),  # end_time slider value
        round(_processor.duration, 2),  # end_time maximum
    )


# ---------------------------------------------------------------------------
# Step 2 — Select section & extract frames
# ---------------------------------------------------------------------------

def extract_section(start_time, end_time):
    """Extract frames from the selected time range and show thumbnails."""
    global _all_section_frames
    if _processor is None:
        return None, "Load a video first."

    if end_time <= start_time:
        return None, "End time must be after start time."

    _all_section_frames = _processor.get_frames_in_range(start_time, end_time)

    if not _all_section_frames:
        return None, "No frames extracted."

    # Build a thumbnail gallery (up to 20 evenly spaced frames)
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
# Step 3 — Pick keyframes (auto or manual)
# ---------------------------------------------------------------------------

def pick_keyframes_auto(num_keyframes, sensitivity):
    """Automatically select keyframes based on subject movement."""
    if _processor is None or not _all_section_frames:
        return None, "Extract a video section first."

    frames_bgr = [f for _, f in _all_section_frames]
    background = estimate_background(frames_bgr, method="median")
    threshold = max(10, 60 - int(sensitivity))

    selected_indices = auto_select_keyframes(
        _all_section_frames, background,
        num_keyframes=int(num_keyframes),
        threshold=threshold,
    )

    # Show selected keyframes as gallery
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
# Step 4 — Generate composite
# ---------------------------------------------------------------------------

def generate_composite(
    num_keyframes,
    selection_mode,
    sensitivity,
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

    # 1. Estimate background
    background = estimate_background(frames_bgr, method=bg_method)

    # 2. Select keyframes
    if selection_mode == "Auto (movement-based)":
        threshold_auto = max(10, 60 - int(sensitivity))
        selected_indices = auto_select_keyframes(
            _all_section_frames, background,
            num_keyframes=int(num_keyframes),
            threshold=threshold_auto,
        )
    else:
        total = len(_all_section_frames)
        n = int(num_keyframes)
        if total <= n:
            idxs = list(range(total))
        else:
            idxs = [int(i * (total - 1) / (n - 1)) for i in range(n)]
        selected_indices = [_all_section_frames[i][0] for i in idxs]

    # 3. Segment foreground for each keyframe
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

    # 4. Composite
    result_bgr = composite_stop_motion(
        background, keyframe_bgr, keyframe_masks,
        opacity=opacity,
        shadow=enable_shadow,
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
# Example image generator (creates a synthetic example)
# ---------------------------------------------------------------------------

def generate_example_image():
    """Generate a synthetic example showing what the output looks like."""
    w, h = 1200, 675
    # Sky-blue gradient background
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        bg[y, :] = [int(200 - 100 * ratio), int(220 - 60 * ratio), int(255 - 30 * ratio)]

    # Draw a ground/snow slope
    pts = np.array([[0, h], [0, int(h * 0.7)], [int(w * 0.4), int(h * 0.85)], [w, int(h * 0.95)], [w, h]], np.int32)
    cv2.fillPoly(bg, [pts], (240, 240, 250))

    result = bg.copy()

    # Draw multiple "athlete" silhouettes in an arc path (simulating a jump)
    num_poses = 7
    colors = [
        (40, 80, 220),   # red-ish (BGR)
        (30, 100, 230),
        (20, 120, 240),
        (10, 90, 250),
        (30, 70, 230),
        (40, 80, 220),
        (50, 90, 200),
    ]

    for i in range(num_poses):
        t = i / (num_poses - 1)
        # Parabolic arc
        cx = int(200 + t * (w - 400))
        cy = int(h * 0.8 - 300 * np.sin(t * np.pi))

        # Body (simplified stick figure with filled shapes)
        color = colors[i % len(colors)]

        # Shadow
        shadow_y = int(h * 0.85 + (1 - np.sin(t * np.pi)) * 30)
        cv2.ellipse(result, (cx, shadow_y), (25, 8), 0, 0, 360, (180, 180, 190), -1)

        # Rotation angle based on trajectory
        angle = -180 * t

        # Torso
        torso_len = 50
        rad = np.radians(angle)
        tx = int(cx + torso_len * 0.3 * np.sin(rad))
        ty = int(cy + torso_len * 0.3 * np.cos(rad))
        cv2.line(result, (cx, cy), (tx, ty), color, 6)

        # Head
        hx = int(cx - 15 * np.sin(rad))
        hy = int(cy - 15 * np.cos(rad))
        cv2.circle(result, (hx, hy), 12, color, -1)

        # Arms
        arm_angle1 = angle + 45 + i * 20
        arm_angle2 = angle - 45 - i * 15
        for arm_a in [arm_angle1, arm_angle2]:
            arad = np.radians(arm_a)
            ax = int(cx + 35 * np.sin(arad))
            ay = int(cy + 35 * np.cos(arad))
            cv2.line(result, (cx, cy), (ax, ay), color, 4)

        # Legs
        leg_angle1 = angle + 160 + i * 10
        leg_angle2 = angle + 200 - i * 10
        for leg_a in [leg_angle1, leg_angle2]:
            lrad = np.radians(leg_a)
            lx = int(tx + 40 * np.sin(lrad))
            ly = int(ty + 40 * np.cos(lrad))
            cv2.line(result, (tx, ty), (lx, ly), color, 5)

    # Add labels
    cv2.putText(result, "Stop-Motion Sports Analysis", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "7 keyframes composited on clean background", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # Draw trajectory arc (dotted)
    for i in range(100):
        t = i / 99
        px = int(200 + t * (w - 400))
        py = int(h * 0.8 - 300 * np.sin(t * np.pi))
        if i % 4 < 2:
            cv2.circle(result, (px, py), 1, (200, 200, 255), -1)

    return _bgr_to_rgb(result)


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
        "showing the subject at multiple points in time on a clean background.",
        elem_classes=["main-title"],
    )

    with gr.Tabs():
        # ── Tab: Example ──────────────────────────────────────────────
        with gr.TabItem("Example Output"):
            gr.Markdown("### What the output looks like")
            gr.Markdown(
                "The tool extracts keyframes from a video clip and composites each "
                "frame's subject onto a single clean background — similar to the "
                "multi-exposure photography technique used in sports analysis."
            )
            gr.Image(value=EXAMPLE_IMAGE, label="Example: Stop-motion composite of an athlete jumping", interactive=False)
            gr.Markdown(
                "**How it works:**\n"
                "1. The background is estimated from all frames (median pixel values)\n"
                "2. The moving subject is segmented from the background in each keyframe\n"
                "3. All segmented subjects are composited onto the clean background\n"
                "4. The result shows the full trajectory of movement in one image"
            )

        # ── Tab: Main workflow ────────────────────────────────────────
        with gr.TabItem("Create Stop-Motion"):

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

            # Step 3: Keyframe selection
            gr.Markdown("## Step 3 — Select Keyframes", elem_classes=["step-header"])
            with gr.Row():
                with gr.Column():
                    num_keyframes = gr.Slider(3, 15, value=7, step=1, label="Number of Keyframes")
                    selection_mode = gr.Radio(
                        ["Auto (movement-based)", "Uniform spacing"],
                        value="Auto (movement-based)",
                        label="Selection Mode",
                    )
                    sensitivity = gr.Slider(10, 50, value=30, step=1, label="Detection Sensitivity (higher = more sensitive)")
                with gr.Column():
                    keyframe_btn = gr.Button("Preview Keyframes", variant="secondary")
                    keyframe_info = gr.Markdown("")
            keyframe_gallery = gr.Gallery(label="Selected Keyframes", columns=4, height=250)

            # Step 4: Generate composite
            gr.Markdown("## Step 4 — Generate Stop-Motion Image", elem_classes=["step-header"])
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Fine-tune Parameters")
                    bg_method = gr.Radio(["median", "first"], value="median", label="Background Method")
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

    def _preview_keyframes(num_kf, mode, sens):
        if mode == "Auto (movement-based)":
            return pick_keyframes_auto(num_kf, sens)
        return pick_keyframes_uniform(num_kf)

    keyframe_btn.click(
        fn=_preview_keyframes,
        inputs=[num_keyframes, selection_mode, sensitivity],
        outputs=[keyframe_gallery, keyframe_info],
    )

    generate_btn.click(
        fn=generate_composite,
        inputs=[
            num_keyframes, selection_mode, sensitivity,
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
