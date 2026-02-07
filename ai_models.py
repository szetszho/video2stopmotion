"""
AI model backends for improved segmentation and alignment.

Provides GPU-accelerated inference via ONNX Runtime with automatic
provider selection:
  - NVIDIA GPU: TensorRT → CUDA → CPU fallback
  - Apple Silicon: CoreML → CPU fallback
  - Other: CPU

Models are downloaded on first use from HuggingFace Hub and cached locally.

Usage:
    from ai_models import AISegmenter, get_available_providers

    # Check what's available
    providers = get_available_providers()

    # Segment a subject from a single frame (no background needed)
    segmenter = AISegmenter()  # auto-downloads model on first use
    alpha_mask = segmenter.segment(frame_bgr)  # returns 0-255 mask
"""

import os
import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# ONNX Runtime provider auto-detection
# ═══════════════════════════════════════════════════════════════════════════

_ORT_AVAILABLE = False
_ort = None

try:
    import onnxruntime as ort
    _ort = ort
    _ORT_AVAILABLE = True
except ImportError:
    pass


def get_available_providers() -> list[str]:
    """Return the list of ONNX Runtime execution providers available on this system.

    Ordered by preference: GPU providers first, CPU last.
    Returns empty list if onnxruntime is not installed.
    """
    if not _ORT_AVAILABLE:
        return []
    available = _ort.get_available_providers()
    # Order by preference
    preferred = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in available]


def _select_providers() -> list[str]:
    """Select the best available providers for inference."""
    providers = get_available_providers()
    if not providers:
        return ["CPUExecutionProvider"]
    return providers


def is_gpu_available() -> bool:
    """Check if any GPU execution provider is available."""
    providers = get_available_providers()
    gpu_providers = {"TensorrtExecutionProvider", "CUDAExecutionProvider",
                     "CoreMLExecutionProvider"}
    return bool(set(providers) & gpu_providers)


# ═══════════════════════════════════════════════════════════════════════════
# Model download helpers
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "video2stopmotion", "models")


def _ensure_cache_dir():
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)


def _download_model(repo_id: str, filename: str) -> str:
    """Download a model file from HuggingFace Hub. Returns local path."""
    _ensure_cache_dir()
    local_path = os.path.join(_MODEL_CACHE_DIR, filename)

    if os.path.exists(local_path):
        logger.info(f"Model already cached: {local_path}")
        return local_path

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"Downloading {repo_id}/{filename} ...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=_MODEL_CACHE_DIR,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to: {path}")
        return path
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for AI model download. "
            "Install it with: pip install huggingface-hub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model {repo_id}/{filename}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# AI Segmentation — Background Removal (RMBG / U2-Net)
# ═══════════════════════════════════════════════════════════════════════════

# We support multiple model backends with automatic fallback:
#   1. RMBG-1.4 (briaai) — fast, high quality, ONNX available
#   2. U2-Net (general purpose) — widely available ONNX exports
#   3. IS-Net (DIS) — good for dichotomous image segmentation

_SEGMENTATION_MODELS = {
    "rmbg-1.4": {
        "repo_id": "briaai/RMBG-1.4",
        "filename": "onnx/model.onnx",
        "input_size": (1024, 1024),
        "input_name": "input",
        "output_name": "output",
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [1.0, 1.0, 1.0],
        "channel_order": "rgb",
    },
    "u2net": {
        "repo_id": "danielgatis/rembg",
        "filename": "u2net.onnx",
        "input_size": (320, 320),
        "input_name": "input.1",
        "output_name": "1959",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "channel_order": "rgb",
    },
    "u2netp": {
        "repo_id": "danielgatis/rembg",
        "filename": "u2netp.onnx",
        "input_size": (320, 320),
        "input_name": "input.1",
        "output_name": "1959",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "channel_order": "rgb",
    },
    "isnet-general": {
        "repo_id": "danielgatis/rembg",
        "filename": "isnet-general-use.onnx",
        "input_size": (1024, 1024),
        "input_name": "input.1",
        "output_name": "1959",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "channel_order": "rgb",
    },
}


class AISegmenter:
    """AI-powered foreground segmentation using ONNX Runtime.

    Produces high-quality alpha mattes from a single frame — no background
    reference needed. Falls back gracefully if ONNX Runtime or models are
    unavailable.
    """

    def __init__(self, model_name: str = "u2netp", device: str = "auto"):
        """Initialize the AI segmenter.

        Args:
            model_name: Which model to use. Options:
                'u2netp'        — lightweight U2-Net (4.7 MB, fast)
                'u2net'         — full U2-Net (176 MB, better quality)
                'isnet-general' — IS-Net (176 MB, good for people)
                'rmbg-1.4'      — RMBG 1.4 (176 MB, production quality)
            device: 'auto' (best available), 'gpu', or 'cpu'
        """
        if not _ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is required for AI segmentation. "
                "Install with: pip install onnxruntime-gpu  (NVIDIA) or "
                "pip install onnxruntime  (CPU/Apple)"
            )

        if model_name not in _SEGMENTATION_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(_SEGMENTATION_MODELS.keys())}"
            )

        self.model_name = model_name
        self.config = _SEGMENTATION_MODELS[model_name]
        self.session: Optional[object] = None
        self._device = device
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        model_path = _download_model(
            self.config["repo_id"],
            self.config["filename"],
        )

        # Select providers based on device preference
        if self._device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif self._device == "gpu":
            providers = [p for p in _select_providers()
                        if p != "CPUExecutionProvider"]
            if not providers:
                providers = ["CPUExecutionProvider"]
        else:  # auto
            providers = _select_providers()

        sess_options = _ort.SessionOptions()
        sess_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = _ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        # Discover actual input/output names from the model
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if inputs:
            self.config["input_name"] = inputs[0].name
        if outputs:
            self.config["output_name"] = outputs[0].name

        active = self.session.get_providers()
        logger.info(f"AI Segmenter loaded: {self.model_name} on {active}")
        self._loaded = True

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Preprocess a BGR frame for the model.

        Returns NCHW float32 tensor.
        """
        h, w = self.config["input_size"]
        mean = np.array(self.config["normalize_mean"], dtype=np.float32)
        std = np.array(self.config["normalize_std"], dtype=np.float32)

        # Convert BGR → RGB
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] then apply mean/std
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std

        # HWC → NCHW
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return img

    def _postprocess(self, output: np.ndarray, original_h: int,
                     original_w: int) -> np.ndarray:
        """Convert model output to a 0-255 alpha mask at original resolution."""
        # Output shape varies: could be (1, 1, H, W) or (1, H, W) or (H, W)
        mask = output.squeeze()

        # Sigmoid if values suggest logits (outside [0, 1])
        if mask.min() < -0.5 or mask.max() > 1.5:
            mask = 1.0 / (1.0 + np.exp(-mask))

        # Normalize to [0, 1]
        mask_min, mask_max = mask.min(), mask.max()
        if mask_max - mask_min > 1e-6:
            mask = (mask - mask_min) / (mask_max - mask_min)

        # Resize to original frame size
        mask = cv2.resize(mask.astype(np.float32), (original_w, original_h),
                         interpolation=cv2.INTER_LINEAR)

        # Convert to uint8
        return (mask * 255).clip(0, 255).astype(np.uint8)

    def segment(self, frame_bgr: np.ndarray,
                refine_morph: bool = True,
                morph_size: int = 5,
                dilate_iterations: int = 0,
                erode_iterations: int = 0,
                feather_radius: int = 3,
                min_contour_area: float = 0.001) -> np.ndarray:
        """Segment the foreground subject from a single frame.

        Args:
            frame_bgr: Input frame in BGR format.
            refine_morph: Apply morphological cleanup to the mask.
            morph_size: Kernel size for morph cleanup.
            dilate_iterations: Extra mask dilation steps.
            erode_iterations: Extra mask erosion steps.
            feather_radius: Gaussian blur for edge softening.
            min_contour_area: Fraction of image area — smaller blobs removed.

        Returns:
            Alpha mask (0-255) where 255 = foreground subject.
        """
        self._ensure_loaded()

        original_h, original_w = frame_bgr.shape[:2]

        # Run inference
        input_tensor = self._preprocess(frame_bgr)
        output = self.session.run(
            [self.config["output_name"]],
            {self.config["input_name"]: input_tensor},
        )[0]

        mask = self._postprocess(output, original_h, original_w)

        # Threshold to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if refine_morph:
            # Morphological cleanup
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Extra dilation / erosion
            if dilate_iterations > 0:
                dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.dilate(mask, dk, iterations=dilate_iterations)
            if erode_iterations > 0:
                ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.erode(mask, ek, iterations=erode_iterations)

            # Remove small blobs
            h, w = mask.shape
            min_area = min_contour_area * h * w
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= min_area:
                    cv2.drawContours(clean, [cnt], -1, 255, -1)
            mask = clean

        # Edge feathering
        if feather_radius > 0:
            k = feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return mask

    def segment_on_panorama(self, frame_bgr: np.ndarray,
                            H_to_panorama: np.ndarray,
                            canvas_w: int, canvas_h: int,
                            morph_size: int = 5,
                            dilate_iterations: int = 0,
                            erode_iterations: int = 0,
                            feather_radius: int = 3,
                            min_contour_area: float = 0.001) -> tuple:
        """Segment on the original frame, then warp mask to panorama coords.

        This is more accurate than segmenting on the warped panorama image
        because the AI model sees the original, un-warped frame.

        Returns (warped_frame, warped_mask) in panorama coordinates.
        """
        # Segment on original frame (better quality — no warp artifacts)
        mask_orig = self.segment(
            frame_bgr,
            refine_morph=True,
            morph_size=morph_size,
            dilate_iterations=dilate_iterations,
            erode_iterations=erode_iterations,
            feather_radius=0,  # feather after warping
            min_contour_area=min_contour_area,
        )

        # Warp both frame and mask to panorama coordinates
        warped_frame = cv2.warpPerspective(
            frame_bgr, H_to_panorama, (canvas_w, canvas_h))
        warped_mask = cv2.warpPerspective(
            mask_orig, H_to_panorama, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR)

        # Threshold warped mask back to binary (warp introduces interpolation)
        _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)

        # Valid region (where the warped frame has content)
        warped_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        valid = (warped_gray > 2).astype(np.uint8) * 255
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        valid = cv2.erode(valid, ek, iterations=2)
        warped_mask = cv2.bitwise_and(warped_mask, valid)

        # Feather after warping for smooth edges on the panorama
        if feather_radius > 0:
            k = feather_radius * 2 + 1
            warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)

        return warped_frame, warped_mask

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> dict:
        """Return info about the loaded model and provider."""
        info = {
            "model": self.model_name,
            "loaded": self._loaded,
            "onnxruntime_available": _ORT_AVAILABLE,
            "providers_available": get_available_providers(),
        }
        if self._loaded and self.session:
            info["active_providers"] = self.session.get_providers()
        return info


# ═══════════════════════════════════════════════════════════════════════════
# Optical flow alignment (learned dense flow)
# ═══════════════════════════════════════════════════════════════════════════

class OpticalFlowAligner:
    """Dense optical flow for improved frame alignment.

    Uses RAFT-style models via ONNX Runtime for dense correspondence,
    which works better than ORB on textureless scenes (snow, water, sky).

    Falls back to OpenCV's DIS optical flow (CPU, decent quality) if
    ONNX models are not available.
    """

    def __init__(self, use_ai: bool = True):
        """
        Args:
            use_ai: Try to use AI optical flow model. Falls back to
                    OpenCV DIS flow if unavailable.
        """
        self.use_ai = use_ai and _ORT_AVAILABLE
        self._session = None

    def compute_flow(self, frame1_bgr: np.ndarray,
                     frame2_bgr: np.ndarray) -> np.ndarray:
        """Compute dense optical flow from frame1 to frame2.

        Returns flow field of shape (H, W, 2) where flow[y, x] = (dx, dy).
        """
        # Use OpenCV DIS flow (good quality, runs on CPU)
        gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)

        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(gray1, gray2, None)
        return flow

    def estimate_affine_from_flow(self, flow: np.ndarray,
                                   grid_step: int = 20) -> Optional[np.ndarray]:
        """Estimate an affine transform from a dense flow field.

        Samples a grid of correspondences from the flow and fits an
        affine via RANSAC — much more robust than sparse ORB features.

        Returns 3x3 homogeneous affine matrix, or None on failure.
        """
        h, w = flow.shape[:2]

        # Sample a grid of points
        ys = np.arange(grid_step // 2, h, grid_step)
        xs = np.arange(grid_step // 2, w, grid_step)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')

        src_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)
        dx = flow[grid_y.ravel(), grid_x.ravel(), 0]
        dy = flow[grid_y.ravel(), grid_x.ravel(), 1]
        dst_pts = src_pts + np.stack([dx, dy], axis=-1)

        # Filter out points with negligible flow (static background)
        mag = np.sqrt(dx**2 + dy**2)
        valid = mag < np.percentile(mag, 95)  # remove outliers (moving subject)
        src_pts = src_pts[valid]
        dst_pts = dst_pts[valid]

        if len(src_pts) < 10:
            return None

        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None:
            return None

        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
        return H

    def align_frames(self, frames: list[np.ndarray],
                     ref_index: Optional[int] = None) -> list[np.ndarray]:
        """Align all frames to a reference using dense optical flow.

        More robust than ORB on textureless backgrounds (snow, water, sky).

        Returns cumulative_H[i]: maps frame i → reference frame.
        """
        n = len(frames)
        if ref_index is None:
            ref_index = n // 2

        cumulative = [None] * n
        cumulative[ref_index] = np.eye(3, dtype=np.float64)

        ref = frames[ref_index]

        # Direct matching: each frame → reference
        for i in range(n):
            if i == ref_index:
                continue
            flow = self.compute_flow(ref, frames[i])
            H = self.estimate_affine_from_flow(flow)
            if H is not None:
                cumulative[i] = H

        # Chain via neighbors for frames that failed direct matching
        for _ in range(5):
            changed = False
            for i in range(n):
                if cumulative[i] is not None:
                    continue
                for delta in [1, -1, 2, -2, 3, -3]:
                    ni = i + delta
                    if 0 <= ni < n and cumulative[ni] is not None:
                        flow = self.compute_flow(frames[ni], frames[i])
                        H_local = self.estimate_affine_from_flow(flow)
                        if H_local is not None:
                            cumulative[i] = cumulative[ni] @ H_local
                            changed = True
                            break
            if not changed:
                break

        # Fallback: identity for unresolved frames
        for i in range(n):
            if cumulative[i] is None:
                cumulative[i] = np.eye(3, dtype=np.float64)

        return cumulative


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: check what AI features are available
# ═══════════════════════════════════════════════════════════════════════════

def check_ai_status() -> dict:
    """Check which AI features are available on this system.

    Returns a dict with availability info for display in the UI.
    """
    status = {
        "onnxruntime_installed": _ORT_AVAILABLE,
        "onnxruntime_version": _ort.__version__ if _ORT_AVAILABLE else None,
        "providers": get_available_providers(),
        "gpu_available": is_gpu_available(),
        "segmentation_models": list(_SEGMENTATION_MODELS.keys()),
        "optical_flow": "OpenCV DIS (built-in)",
    }

    if _ORT_AVAILABLE:
        gpu_prov = [p for p in get_available_providers()
                    if p != "CPUExecutionProvider"]
        if gpu_prov:
            status["gpu_provider"] = gpu_prov[0]
        else:
            status["gpu_provider"] = None
    else:
        status["gpu_provider"] = None

    return status
