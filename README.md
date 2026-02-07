# Video → Stop-Motion Effect | Sports Analysis Tool

Transform video clips into stop-motion composite images that show a subject's full movement trajectory in a single frame — ideal for sports analysis, coaching, and biomechanics study.

**Supports both static and moving cameras** — moving camera footage is automatically aligned and stitched into an expanded panoramic background.

**Optional AI enhancement** — plug in neural-network segmentation (U2-Net, IS-Net, RMBG) and dense optical flow for higher quality on difficult footage.

![Example Output](example_output.png)

## How It Works

### Moving Camera (default)

1. **Import a video** of an athlete performing a movement (filmed with a panning/tracking camera)
2. **Select a time section** — a couple of seconds covering the key action
3. **Align & stitch** — frames are aligned via ORB features or dense optical flow, then stitched into a wide panoramic background
4. **Pick keyframes** — automatically (based on movement detection in panorama coordinates) or uniformly spaced
5. **Generate composite** — the subject is segmented (classical or AI) from each keyframe and composited onto the expanded panoramic canvas

The result is an image **wider than any single video frame**, showing the full trajectory across the scene.

### Static Camera

For tripod/fixed camera footage, uses pixel-wise median to estimate a clean background, then segments and composites the subject at each keyframe.

## Installation

```bash
# Clone the repository
git clone https://github.com/szetszho/video2stopmotion.git
cd video2stopmotion

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install AI acceleration — pick ONE:
pip install onnxruntime-gpu>=1.17.0 huggingface-hub   # NVIDIA GPU
pip install onnxruntime>=1.17.0 huggingface-hub        # CPU / Apple Silicon
```

### Dependencies

**Core (required):**
- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- Gradio (for the web UI)
- scikit-image

**AI features (optional):**
- `onnxruntime-gpu` (NVIDIA) or `onnxruntime` (CPU/Apple)
- `huggingface-hub` (for automatic model download)

## Usage

### Launch the Web UI

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### Workflow

| Step | Action |
|------|--------|
| 1 | Choose camera mode (Moving or Static) |
| 2 | Upload a video file (MP4, AVI, MOV, etc.) |
| 3 | Set start/end times to select the action section |
| 4 | *(Moving Camera)* Click "Align & Build Panorama" to stitch the background |
| 5 | Choose keyframes (auto or uniform) and preview |
| 6 | Choose segmentation method (Classical or AI) and adjust parameters |
| 7 | Generate the composite, preview, and download |

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Camera Mode** | `Moving Camera` (panoramic stitching) or `Static Camera` (median bg) | Moving |
| **Alignment Method** | `ORB` (fast, feature-based) or `Flow` (dense optical flow, robust on textureless bg) | ORB |
| **Segmentation Method** | `Classical` (background subtraction) or `AI Model` (neural network) | Classical |
| **AI Model** | `u2netp` (fast), `u2net` (quality), `isnet-general` (people), `rmbg-1.4` (production) | u2netp |
| **Number of Keyframes** | How many subject poses to include (3–15) | 7 |
| **Selection Mode** | `Auto` (movement-based) or `Uniform` spacing | Auto |
| **Detection Sensitivity** | Higher = detects subtler movements | 30 |
| **Background Method** | `median` (removes subject) or `overlay` (faster) | median |
| **Backdrop Density** | Frames used for panorama (3–30). Lower = faster | 12 |
| **Segmentation Threshold** | Lower = more sensitive foreground detection (classical only) | 35 |
| **Morph Kernel Size** | Cleanup kernel for mask edges | 7 |
| **Dilate / Erode** | Grow or shrink the segmentation mask | 0 |
| **Edge Feathering** | Soft blend radius for cut-out edges | 3 |
| **Min Object Size** | Discard blobs smaller than this % of image | 0.1% |
| **Subject Opacity** | Transparency of each subject layer | 1.0 |
| **Drop Shadow** | Adds shadow behind each subject | On |

## AI Models

When `onnxruntime` is installed, the app auto-detects GPU support and offers AI-powered features:

### AI Segmentation

Uses pre-trained neural networks for single-image background removal — no reference background needed. Produces much cleaner alpha mattes than classical background subtraction, especially for:
- Hair, equipment, and fine details
- Subjects with similar colors to the background
- Complex poses and overlapping limbs

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| **u2netp** | 4.7 MB | Fast | Quick previews, lightweight usage |
| **u2net** | 176 MB | Medium | General purpose, good quality |
| **isnet-general** | 176 MB | Medium | People and athletes |
| **rmbg-1.4** | 176 MB | Medium | Production quality output |

Models are downloaded automatically on first use and cached in `~/.cache/video2stopmotion/models/`.

### Dense Optical Flow Alignment

Uses OpenCV's DIS optical flow for dense frame-to-frame correspondence — more robust than ORB features on:
- Uniform/textureless backgrounds (snow, water, sky)
- Scenes with few distinct features
- Very smooth camera pans

### GPU Acceleration

The app automatically selects the best available execution provider:

| Platform | Provider | Performance |
|----------|----------|-------------|
| NVIDIA GPU | TensorRT → CUDA | Fastest |
| Apple Silicon | CoreML | Fast (Neural Engine) |
| CPU | CPU | Slower but always works |

## Tips for Best Results

- **Moving camera**: Ensure there's enough texture in the background for feature matching (buildings, trees, terrain). If the background is uniform (snow, sky), switch to **Flow** alignment.
- **AI segmentation**: Try `u2netp` first for speed, switch to `isnet-general` or `rmbg-1.4` if the mask quality isn't good enough.
- Keep the **subject moving** across the frame (not just in place)
- A section of **1–3 seconds** usually works best
- Higher contrast between subject and background gives better segmentation
- Adjust the segmentation threshold if the subject isn't cleanly extracted
- For **very long pans**, the affine alignment keeps the panorama geometrically stable

## Architecture

```
video2stopmotion/
├── app.py                # Gradio web UI (both camera modes)
├── video_processor.py    # Core processing engine
├── ai_models.py          # AI model backends (ONNX Runtime)
├── generate_example.py   # Generates synthetic example images
├── requirements.txt      # Python dependencies
├── example_output.png    # Example panoramic composite
└── README.md
```

### Core Modules

**`video_processor.py`** — Processing engine:
- `VideoProcessor` — Video loading, frame extraction, metadata
- `PanoramicPipeline` — End-to-end moving camera workflow (align → stitch → segment → composite)
- `compute_direct_homographies()` — ORB-based affine alignment (avoids chain drift)
- `compute_direct_homographies_flow()` — Dense optical flow alignment
- `segment_foreground()` — Classical background subtraction (static camera)
- `segment_foreground_ai()` — AI neural network segmentation
- `segment_foreground_moving()` — Panoramic background subtraction (moving camera)
- `build_panoramic_background()` — Stitch aligned frames into expanded panorama
- `composite_stop_motion()` — Layer subjects onto clean background with drop shadows

**`ai_models.py`** — AI backends:
- `AISegmenter` — ONNX-based foreground segmentation (U2-Net, IS-Net, RMBG)
- `OpticalFlowAligner` — Dense optical flow frame alignment
- `check_ai_status()` — Detect available GPU providers and models

## License

See [LICENSE](LICENSE) for details.
