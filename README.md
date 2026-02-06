# Video → Stop-Motion Effect | Sports Analysis Tool

Transform video clips into stop-motion composite images that show a subject's full movement trajectory in a single frame — ideal for sports analysis, coaching, and biomechanics study.

![Example Output](example_output.png)

## How It Works

1. **Import a video** of an athlete performing a movement (jump, swing, sprint, etc.)
2. **Select a time section** — a couple of seconds covering the key action
3. **Pick keyframes** — automatically (based on movement detection) or uniformly spaced
4. **Generate composite** — the tool estimates a clean background, segments the subject from each keyframe, and layers all subjects onto one image

The result is a single image showing the athlete at multiple points in time on a crisp, clean background — similar to multi-exposure sports photography.

## Installation

```bash
# Clone the repository
git clone https://github.com/szetszho/video2stopmotion.git
cd video2stopmotion

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- Gradio (for the web UI)
- scikit-image

## Usage

### Launch the Web UI

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### Workflow

| Step | Action |
|------|--------|
| 1 | Upload a video file (MP4, AVI, MOV, etc.) |
| 2 | Set start/end times to select the action section |
| 3 | Choose number of keyframes and selection mode |
| 4 | Adjust parameters and generate the composite |
| 5 | Preview and download the result |

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Number of Keyframes** | How many subject poses to include (3–15) | 7 |
| **Selection Mode** | `Auto` (movement-based) or `Uniform` spacing | Auto |
| **Detection Sensitivity** | Higher = detects subtler movements | 30 |
| **Background Method** | `median` (best) or `first` frame | median |
| **Segmentation Threshold** | Lower = more sensitive foreground detection | 35 |
| **Morph Kernel Size** | Cleanup kernel for mask edges | 7 |
| **Subject Opacity** | Transparency of each subject layer | 1.0 |
| **Drop Shadow** | Adds shadow behind each subject | On |

## Tips for Best Results

- Use a **static camera** (tripod) for cleanest background estimation
- Keep the **subject moving** across the frame (not just in place)
- A section of **1–3 seconds** usually works best
- Higher contrast between subject and background gives better segmentation
- Adjust the segmentation threshold if the subject isn't cleanly extracted

## Architecture

```
video2stopmotion/
├── app.py                # Gradio web UI
├── video_processor.py    # Core processing engine
├── requirements.txt      # Python dependencies
├── example_output.png    # Example composite image
└── README.md
```

### Core Modules (`video_processor.py`)

- `VideoProcessor` — Video loading, frame extraction, metadata
- `estimate_background()` — Pixel-wise median background estimation
- `segment_foreground()` — Background subtraction + morphological cleanup
- `composite_stop_motion()` — Layer subjects onto clean background with shadows
- `auto_select_keyframes()` — Movement-based keyframe selection using centroid tracking

## License

See [LICENSE](LICENSE) for details.
