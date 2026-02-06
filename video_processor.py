"""
Core video processing engine for stop-motion effect generation.

Handles video loading, frame extraction, background estimation,
foreground segmentation, and composite image creation.
"""

import cv2
import numpy as np
from typing import Optional


class VideoProcessor:
    """Loads a video and extracts frames from a specified time range."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a single frame by index (returns BGR image)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frames_in_range(self, start_sec: float, end_sec: float) -> list[tuple[int, np.ndarray]]:
        """Extract all frames between start_sec and end_sec.
        Returns list of (frame_index, frame_bgr)."""
        start_frame = max(0, int(start_sec * self.fps))
        end_frame = min(self.total_frames - 1, int(end_sec * self.fps))
        frames = []
        for idx in range(start_frame, end_frame + 1):
            frame = self.get_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
        return frames

    def get_thumbnail_strip(self, start_sec: float, end_sec: float, num_thumbs: int = 10) -> list[tuple[int, np.ndarray]]:
        """Get evenly spaced thumbnail frames for UI preview."""
        start_frame = max(0, int(start_sec * self.fps))
        end_frame = min(self.total_frames - 1, int(end_sec * self.fps))
        if end_frame <= start_frame:
            return []
        step = max(1, (end_frame - start_frame) // num_thumbs)
        thumbs = []
        for idx in range(start_frame, end_frame + 1, step):
            frame = self.get_frame(idx)
            if frame is not None:
                thumbs.append((idx, frame))
        return thumbs[:num_thumbs]

    def close(self):
        self.cap.release()

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass


def estimate_background(frames: list[np.ndarray], method: str = "median") -> np.ndarray:
    """Estimate a clean background from a set of frames.

    Methods:
        'median' - pixel-wise median (best for mostly static camera with moving subject)
        'first'  - use the first frame as background
    """
    if method == "first":
        return frames[0].copy()

    # Median approach: stack frames and take per-pixel median
    # Subsample if too many frames to save memory
    max_frames = 60
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        subset = [frames[i] for i in indices]
    else:
        subset = frames

    stack = np.stack(subset, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    return background


def segment_foreground(frame: np.ndarray, background: np.ndarray,
                       threshold: int = 40, morph_size: int = 7,
                       min_contour_area: float = 0.001) -> np.ndarray:
    """Segment the foreground subject from the background.

    Returns an alpha mask (0-255) where 255 = foreground.
    Uses background subtraction with morphological cleanup.
    """
    # Compute absolute difference in each channel
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)

    # Threshold
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small holes: find contours and fill
    h, w = mask.shape
    min_area = min_contour_area * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filled_mask, [cnt], -1, 255, -1)

    # Slight feathering for smoother edges
    filled_mask = cv2.GaussianBlur(filled_mask, (5, 5), 0)

    return filled_mask


def composite_stop_motion(background: np.ndarray,
                          frames: list[np.ndarray],
                          masks: list[np.ndarray],
                          opacity: float = 1.0,
                          shadow: bool = True) -> np.ndarray:
    """Composite foreground subjects onto a clean background.

    Each frame's foreground (defined by its mask) is layered onto the background
    in order, creating the stop-motion multi-exposure effect.
    """
    result = background.copy().astype(np.float64)

    for frame, mask in zip(frames, masks):
        # Normalize mask to [0, 1]
        alpha = (mask.astype(np.float64) / 255.0) * opacity

        if shadow:
            # Add a subtle drop shadow behind each subject
            shadow_mask = cv2.GaussianBlur(mask, (21, 21), 10)
            shadow_alpha = (shadow_mask.astype(np.float64) / 255.0) * 0.15
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - shadow_alpha) + 0 * shadow_alpha

        # Blend foreground onto result
        alpha_3 = np.stack([alpha] * 3, axis=-1)
        result = result * (1 - alpha_3) + frame.astype(np.float64) * alpha_3

    return np.clip(result, 0, 255).astype(np.uint8)


def auto_select_keyframes(frames: list[tuple[int, np.ndarray]],
                          background: np.ndarray,
                          num_keyframes: int = 7,
                          threshold: int = 40) -> list[int]:
    """Automatically select keyframes based on subject movement.

    Picks frames where the foreground centroid has moved significantly,
    ensuring good spatial distribution of the subject across the image.
    """
    if len(frames) <= num_keyframes:
        return [idx for idx, _ in frames]

    # Compute foreground centroid for each frame
    centroids = []
    for idx, frame in frames:
        mask = segment_foreground(frame, background, threshold=threshold)
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroids.append((idx, cx, cy, mask))
        else:
            centroids.append((idx, -1, -1, mask))

    # Filter out frames with no detected foreground
    valid = [(idx, cx, cy) for idx, cx, cy, _ in centroids if cx >= 0]
    if len(valid) <= num_keyframes:
        return [idx for idx, _, _ in valid]

    # Always include first and last valid frames
    selected = [valid[0]]
    remaining = valid[1:-1]
    selected.append(valid[-1])

    # Greedily pick frames that maximize minimum distance to already selected
    while len(selected) < num_keyframes and remaining:
        best_score = -1
        best_idx = 0
        for i, (idx, cx, cy) in enumerate(remaining):
            min_dist = min(
                np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
                for _, sx, sy in selected
            )
            if min_dist > best_score:
                best_score = min_dist
                best_idx = i
        selected.append(remaining.pop(best_idx))

    # Sort by frame index to maintain temporal order
    selected.sort(key=lambda x: x[0])
    return [idx for idx, _, _ in selected]
