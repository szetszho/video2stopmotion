# Video → Stop-Motion Composite | Sports Analysis Tool

Transform video clips into stop-motion composite images that show a subject's full movement trajectory in a single frame — ideal for sports analysis, coaching, and biomechanics study.

Moving camera footage is automatically aligned and stitched into an expanded panoramic background.

![Example Output](example_output.png)

## Prerequisites

### FFmpeg

OpenCV uses FFmpeg under the hood to decode video files. Install it before running the app.

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
```
# Option 1: Chocolatey
choco install ffmpeg

# Option 2: Scoop
scoop install ffmpeg

# Option 3: Manual
# Download from https://www.gyan.dev/ffmpeg/builds/
# Extract and add the bin/ folder to your system PATH
```

**Verify installation:**
```bash
ffmpeg -version
```

### Python

Python 3.10 or newer is required.

## Installation

```bash
# Clone the repository
git clone https://github.com/szetszho/video2stopmotion.git
cd video2stopmotion

# Install Python dependencies
pip install -r requirements.txt
```

### Optional: AI-enhanced segmentation

For higher quality subject extraction using neural networks:

```bash
# NVIDIA GPU (Windows / Linux)
pip install onnxruntime-gpu huggingface-hub

# Apple Silicon (macOS) or CPU-only
pip install onnxruntime huggingface-hub
```

## Usage

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### 3-Step Workflow

| Step | What to do |
|------|------------|
| **1. Upload** | Load a video (MP4, AVI, MOV). 1080p recommended. |
| **2. Trim** | Set start/end to isolate the action (1–5 seconds). |
| **3. Generate** | Pick the number of poses and click **Generate Composite**. |

The app automatically aligns frames, builds a panoramic background, selects keyframes, segments the subject, and composites everything.

### Advanced Settings

Click the **Advanced Settings** accordion to fine-tune:

| Setting | What it does | Default |
|---------|-------------|---------|
| **Alignment** | `ORB` (fast) or `Flow` (better on snow/water/sky) | ORB |
| **Backdrop Frames** | How many frames build the panorama (lower = faster) | 12 |
| **Segmentation** | `Classical` (background subtraction) or `AI Model` | Classical |
| **AI Model** | `u2netp` (fast, 5 MB) / `isnet-general` (people, 176 MB) / `rmbg-1.4` (best, 176 MB) | u2netp |
| **Threshold** | Lower = more sensitive foreground detection | 35 |
| **Dilate / Erode** | Grow or shrink the subject mask | 0 |
| **Edge Feather** | Soft blend on cut-out edges | 3 |
| **Opacity** | Subject transparency (lower = see-through) | 1.0 |
| **Drop Shadow** | Adds depth behind each subject | On |

## Image Preprocessing

Frames are automatically preprocessed before stitching and segmentation:

- **Bilateral denoising** — removes H.264/JPEG compression block artifacts that cause spotty masks
- **Exposure normalization** — compensates for auto-exposure drift between frames
- **CLAHE contrast enhancement** — boosts feature detection on low-contrast backgrounds (snow, walls, sky)
- **Max-channel diff** — catches colour differences that average out in grayscale (e.g. red jersey on green grass)

## AI Models

When `onnxruntime` is installed, the app auto-detects GPU support:

| Platform | Provider | Performance |
|----------|----------|-------------|
| NVIDIA GPU | TensorRT → CUDA | Fastest |
| Apple Silicon | CoreML (Neural Engine) | Fast |
| CPU | CPU fallback | Slower but always works |

AI segmentation models are downloaded automatically on first use (~5–176 MB) and cached in `~/.cache/video2stopmotion/models/`.

## Tips

- **1080p** is the sweet spot — enough detail, fast processing
- **1–3 second** clips work best; 5 seconds is fine but slower
- If the background is uniform (snow, sky), switch alignment to **Flow** in Advanced Settings
- If segmentation looks spotty, try lowering the **Threshold** to 20–25
- Use **Dilate** +1–2 to recover clipped edges (hair, equipment)
- Use **Erode** +1–2 to remove background halo around the subject

## Architecture

```
video2stopmotion/
├── app.py                # Gradio web UI (3-step workflow)
├── video_processor.py    # Core processing engine (pure OpenCV/NumPy)
├── ai_models.py          # AI model backends (ONNX Runtime, optional)
├── generate_example.py   # Generates synthetic example images
├── requirements.txt      # Python dependencies
├── example_output.png    # Example composite
└── README.md
```

The core engine (`video_processor.py`) has zero UI dependencies — it's pure OpenCV + NumPy, designed to be wrapped in any frontend (Gradio, native app, CLI).

## License

See [LICENSE](LICENSE) for details.
