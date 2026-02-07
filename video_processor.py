"""
Core video processing engine for stop-motion effect generation.

Supports both static and moving camera workflows:
  - Static camera: median background estimation + foreground segmentation
  - Moving camera: homography-based frame alignment, panoramic background
    stitching, and foreground extraction on the expanded canvas

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


# ═══════════════════════════════════════════════════════════════════════════
# Static camera helpers
# ═══════════════════════════════════════════════════════════════════════════

def estimate_background(frames: list[np.ndarray], method: str = "median") -> np.ndarray:
    """Estimate a clean background from a set of frames (static camera).

    Methods:
        'median' - pixel-wise median (best for mostly static camera with moving subject)
        'first'  - use the first frame as background
    """
    if method == "first":
        return frames[0].copy()

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
                       min_contour_area: float = 0.001,
                       dilate_iterations: int = 0,
                       erode_iterations: int = 0,
                       feather_radius: int = 3) -> np.ndarray:
    """Segment the foreground subject from the background.

    Returns an alpha mask (0-255) where 255 = foreground.

    Refinement controls:
        threshold:        Pixel-difference cutoff. Lower → picks up fainter
                          edges (good for similar-colour subjects). Higher →
                          stricter, removes background noise.
        morph_size:       Kernel for morphological close/open.  Larger values
                          bridge bigger gaps in the mask but can merge nearby
                          objects.
        min_contour_area: Fraction of image area.  Blobs smaller than this
                          are discarded as noise.
        dilate_iterations: Grow the mask outward by this many steps.  Use to
                          recover clipped edges (hair, equipment).
        erode_iterations:  Shrink the mask inward.  Use to remove thin
                          background leaking around the subject.
        feather_radius:   Gaussian blur radius applied to the final mask
                          (0 = hard edge, higher = softer blend).
    """
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Extra dilation / erosion for fine-tuning
    if dilate_iterations > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, dk, iterations=dilate_iterations)
    if erode_iterations > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, ek, iterations=erode_iterations)

    # Keep only large contours
    h, w = mask.shape
    min_area = min_contour_area * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filled_mask, [cnt], -1, 255, -1)

    # Edge feathering
    if feather_radius > 0:
        k = feather_radius * 2 + 1  # must be odd
        filled_mask = cv2.GaussianBlur(filled_mask, (k, k), 0)

    return filled_mask


# ═══════════════════════════════════════════════════════════════════════════
# Moving camera — Homography & Panoramic Stitching
# ═══════════════════════════════════════════════════════════════════════════

def _detect_and_match(img1_gray: np.ndarray, img2_gray: np.ndarray,
                      max_features: int = 3000, match_ratio: float = 0.75):
    """Detect ORB features and match between two grayscale images.
    Returns matched keypoint pairs as (pts1, pts2)."""
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < match_ratio * n.distance:
                good.append(m)

    if len(good) < 6:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def _estimate_affine(pts_src: np.ndarray, pts_dst: np.ndarray) -> Optional[np.ndarray]:
    """Estimate a rigid/affine transform (translation + rotation + scale)
    using RANSAC.  Returns a 3x3 homogeneous matrix or None."""
    if len(pts_src) < 4:
        return None
    M, inliers = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
    if M is None:
        return None
    # Convert 2x3 affine to 3x3 homogeneous
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H


def compute_direct_homographies(frames: list[np.ndarray],
                                ref_index: int,
                                max_features: int = 3000,
                                use_affine: bool = True) -> list[np.ndarray]:
    """Compute a transform from every frame directly to the reference frame.

    For nearby frames: match directly against the reference.
    For distant frames: chain via intermediate anchors to handle
    large viewpoint changes, but keep chains short to limit drift.

    Args:
        use_affine: If True, estimate affine (translation+rotation+scale)
                    instead of full homography. More stable for panning cameras.

    Returns cumulative_H[i]: maps frame i → reference frame.
    """
    n = len(frames)
    cumulative = [None] * n
    cumulative[ref_index] = np.eye(3, dtype=np.float64)

    ref_gray = cv2.cvtColor(frames[ref_index], cv2.COLOR_BGR2GRAY)

    def match_to_ref(frame_gray):
        pts_ref, pts_frame = _detect_and_match(ref_gray, frame_gray,
                                               max_features=max_features)
        if pts_ref is None:
            return None
        if use_affine:
            return _estimate_affine(pts_frame, pts_ref)
        H, _ = cv2.findHomography(pts_frame, pts_ref, cv2.RANSAC, 3.0)
        return H

    # Try direct matching for every frame
    for i in range(n):
        if i == ref_index:
            continue
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        H = match_to_ref(g)
        if H is not None:
            cumulative[i] = H

    # For frames that failed direct matching, chain via nearest resolved neighbor
    max_chain = 5  # max pairwise hops
    for _ in range(max_chain):
        changed = False
        for i in range(n):
            if cumulative[i] is not None:
                continue
            # Try matching against nearest resolved frame
            for delta in [1, -1, 2, -2, 3, -3]:
                ni = i + delta
                if 0 <= ni < n and cumulative[ni] is not None:
                    g_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                    g_ni = cv2.cvtColor(frames[ni], cv2.COLOR_BGR2GRAY)
                    pts_ni, pts_i = _detect_and_match(g_ni, g_i,
                                                      max_features=max_features)
                    if pts_ni is not None:
                        if use_affine:
                            H_local = _estimate_affine(pts_i, pts_ni)
                        else:
                            H_local, _ = cv2.findHomography(pts_i, pts_ni,
                                                            cv2.RANSAC, 3.0)
                        if H_local is not None:
                            cumulative[i] = cumulative[ni] @ H_local
                            changed = True
                            break
        if not changed:
            break

    # Any remaining unresolved frames get identity
    for i in range(n):
        if cumulative[i] is None:
            cumulative[i] = np.eye(3, dtype=np.float64)

    return cumulative


def compute_panorama_bounds(frames: list[np.ndarray],
                            cumulative_H: list[np.ndarray]) -> tuple:
    """Compute the bounding box of the panoramic canvas.

    Returns (min_x, min_y, max_x, max_y, canvas_w, canvas_h, offset_H)
    where offset_H is a translation to shift everything into positive coords.
    """
    h, w = frames[0].shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    all_corners = []
    for H in cumulative_H:
        warped = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped)

    all_corners = np.concatenate(all_corners, axis=0)
    min_x = int(np.floor(all_corners[:, 0, 0].min()))
    max_x = int(np.ceil(all_corners[:, 0, 0].max()))
    min_y = int(np.floor(all_corners[:, 0, 1].min()))
    max_y = int(np.ceil(all_corners[:, 0, 1].max()))

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    # Translation matrix to shift into positive coordinates
    offset_H = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1],
    ], dtype=np.float64)

    return min_x, min_y, max_x, max_y, canvas_w, canvas_h, offset_H


def build_panoramic_background(frames: list[np.ndarray],
                               cumulative_H: list[np.ndarray],
                               method: str = "median",
                               backdrop_density: int = 12,
                               max_canvas_dim: int = 6000) -> tuple:
    """Build an expanded panoramic background from aligned frames.

    The panorama serves as a backdrop for anchoring overlay subjects and
    expanding the canvas — it does NOT need to be pixel-perfect.  Only a
    sparse subset of frames is used, keeping memory bounded even for long
    clips (5-10+ seconds / hundreds of frames).

    Args:
        frames:         All section frames (BGR).
        cumulative_H:   Per-frame homographies (frame → reference).
        method:         'median' averages overlapping pixels (removes the
                        moving subject) or 'overlay' paints later frames on
                        top (fastest).
        backdrop_density: How many evenly-spaced frames to use for the
                        backdrop.  5 is minimal / fast, 20 is high quality.
                        Default 12 balances quality and memory.
        max_canvas_dim: Hard cap on the wider canvas dimension (pixels).
                        Prevents runaway memory on very long pans.

    Returns (panorama_bgr, canvas_w, canvas_h, offset_H).
    """
    _, _, _, _, canvas_w, canvas_h, offset_H = compute_panorama_bounds(
        frames, cumulative_H
    )

    # Scale canvas down if it exceeds the hard cap
    if canvas_w > max_canvas_dim or canvas_h > max_canvas_dim:
        scale = max_canvas_dim / max(canvas_w, canvas_h)
        scale_H = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
                           dtype=np.float64)
        offset_H = scale_H @ offset_H
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)

    # ── Subsample frames for the backdrop ──────────────────────────────
    # Only use `backdrop_density` evenly-spaced frames.  This is the key
    # optimisation: even a 5-second clip at 30 fps (150 frames) only
    # warps ~12 frames instead of 150 — a 12× memory and speed saving.
    n_bg = max(3, min(backdrop_density, len(frames)))
    if len(frames) > n_bg:
        indices = np.linspace(0, len(frames) - 1, n_bg, dtype=int).tolist()
    else:
        indices = list(range(len(frames)))

    if method == "median" and len(indices) >= 3:
        # Median blending: stack warped frames and take per-pixel median.
        # NaN-masked so only pixels with actual coverage contribute.
        warped_stack = []
        mask_stack = []
        for i in indices:
            H_total = offset_H @ cumulative_H[i]
            warped = cv2.warpPerspective(frames[i], H_total,
                                         (canvas_w, canvas_h))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            valid = (gray > 0).astype(np.uint8)
            warped_stack.append(warped)
            mask_stack.append(valid)

        stack = np.stack(warped_stack, axis=0).astype(np.float32)
        masks = np.stack(mask_stack, axis=0)

        for c in range(3):
            stack[:, :, :, c] = np.where(masks > 0,
                                         stack[:, :, :, c], np.nan)

        with np.errstate(invalid='ignore'):
            panorama = np.nanmedian(stack, axis=0)

        panorama = np.nan_to_num(panorama, nan=0.0).astype(np.uint8)
    else:
        # Simple overlay: later frames overwrite earlier ones (fastest)
        panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i in indices:
            H_total = offset_H @ cumulative_H[i]
            warped = cv2.warpPerspective(frames[i], H_total,
                                         (canvas_w, canvas_h))
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
            panorama[mask] = warped[mask]

    return panorama, canvas_w, canvas_h, offset_H


def segment_foreground_moving(frame: np.ndarray, panorama: np.ndarray,
                              H_to_panorama: np.ndarray,
                              canvas_w: int, canvas_h: int,
                              threshold: int = 35, morph_size: int = 7,
                              min_contour_area: float = 0.002,
                              dilate_iterations: int = 0,
                              erode_iterations: int = 0,
                              feather_radius: int = 3) -> tuple:
    """Segment the moving subject on the panoramic canvas.

    Warps the frame into panorama coordinates, computes difference
    against the panoramic background, and extracts the foreground.

    Returns (warped_frame, foreground_mask) in panorama coordinates.

    See segment_foreground() for parameter descriptions — they behave
    identically here but operate on the warped panoramic canvas.
    """
    warped = cv2.warpPerspective(frame, H_to_panorama, (canvas_w, canvas_h))

    # Valid region mask (where the warped frame has content)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    valid_mask = (warped_gray > 2).astype(np.uint8) * 255

    # Erode valid mask to avoid edge artifacts from warping
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    valid_mask = cv2.erode(valid_mask, erode_k, iterations=2)

    # Difference against panoramic background
    diff = cv2.absdiff(warped, panorama)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_diff, (7, 7), 0)

    _, fg_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.bitwise_and(fg_mask, valid_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Extra dilation / erosion
    if dilate_iterations > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.dilate(fg_mask, dk, iterations=dilate_iterations)
    if erode_iterations > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.erode(fg_mask, ek, iterations=erode_iterations)

    # Keep only large contours
    total_area = canvas_h * canvas_w
    min_area = min_contour_area * total_area
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(fg_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # Edge feathering
    if feather_radius > 0:
        k = feather_radius * 2 + 1
        clean_mask = cv2.GaussianBlur(clean_mask, (k, k), 0)

    return warped, clean_mask


# ═══════════════════════════════════════════════════════════════════════════
# Composite generation (works for both static and moving camera)
# ═══════════════════════════════════════════════════════════════════════════

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
        alpha = (mask.astype(np.float64) / 255.0) * opacity

        if shadow:
            shadow_mask = cv2.GaussianBlur(mask, (21, 21), 10)
            shadow_alpha = (shadow_mask.astype(np.float64) / 255.0) * 0.15
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - shadow_alpha) + 0 * shadow_alpha

        alpha_3 = np.stack([alpha] * 3, axis=-1)
        result = result * (1 - alpha_3) + frame.astype(np.float64) * alpha_3

    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Auto keyframe selection
# ═══════════════════════════════════════════════════════════════════════════

def auto_select_keyframes(frames: list[tuple[int, np.ndarray]],
                          background: np.ndarray,
                          num_keyframes: int = 7,
                          threshold: int = 40) -> list[int]:
    """Automatically select keyframes based on subject movement (static camera).

    Picks frames where the foreground centroid has moved significantly,
    ensuring good spatial distribution of the subject across the image.
    """
    if len(frames) <= num_keyframes:
        return [idx for idx, _ in frames]

    centroids = []
    for idx, frame in frames:
        mask = segment_foreground(frame, background, threshold=threshold)
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroids.append((idx, cx, cy))
        else:
            centroids.append((idx, -1, -1))

    valid = [(idx, cx, cy) for idx, cx, cy in centroids if cx >= 0]
    if len(valid) <= num_keyframes:
        return [idx for idx, _, _ in valid]

    selected = [valid[0]]
    remaining = valid[1:-1]
    selected.append(valid[-1])

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

    selected.sort(key=lambda x: x[0])
    return [idx for idx, _, _ in selected]


def auto_select_keyframes_moving(frames: list[tuple[int, np.ndarray]],
                                 panorama: np.ndarray,
                                 cumulative_H: list[np.ndarray],
                                 offset_H: np.ndarray,
                                 canvas_w: int, canvas_h: int,
                                 num_keyframes: int = 7,
                                 threshold: int = 35) -> list[int]:
    """Auto-select keyframes for moving camera based on subject position
    on the panoramic canvas.

    Computes the centroid of the foreground in panorama coordinates for
    each frame, then selects frames with maximum spatial spread.
    """
    if len(frames) <= num_keyframes:
        return [idx for idx, _ in frames]

    # Map frame list indices to cumulative_H indices
    # frames is a subset; we need to figure out the mapping
    first_idx = frames[0][0]

    centroids = []
    for list_i, (idx, frame) in enumerate(frames):
        frame_offset = list_i  # position within the frames list
        if frame_offset < len(cumulative_H):
            H_total = offset_H @ cumulative_H[frame_offset]
        else:
            H_total = offset_H @ cumulative_H[-1]

        _, mask = segment_foreground_moving(
            frame, panorama, H_total, canvas_w, canvas_h,
            threshold=threshold,
        )
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroids.append((idx, cx, cy))
        else:
            centroids.append((idx, -1, -1))

    valid = [(idx, cx, cy) for idx, cx, cy in centroids if cx >= 0]
    if len(valid) <= num_keyframes:
        return [idx for idx, _, _ in valid]

    # Greedy selection maximizing spatial spread
    selected = [valid[0]]
    remaining = list(valid[1:-1])
    selected.append(valid[-1])

    while len(selected) < num_keyframes and remaining:
        best_score = -1
        best_i = 0
        for i, (idx, cx, cy) in enumerate(remaining):
            min_dist = min(
                np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
                for _, sx, sy in selected
            )
            if min_dist > best_score:
                best_score = min_dist
                best_i = i
        selected.append(remaining.pop(best_i))

    selected.sort(key=lambda x: x[0])
    return [idx for idx, _, _ in selected]


# ═══════════════════════════════════════════════════════════════════════════
# High-level pipeline for moving camera
# ═══════════════════════════════════════════════════════════════════════════

class PanoramicPipeline:
    """End-to-end pipeline for moving camera stop-motion generation.

    Usage:
        pipe = PanoramicPipeline(frames_bgr)
        pipe.align_frames()
        pipe.build_panorama()
        result = pipe.generate(keyframe_indices, ...)
    """

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.pairwise_H: list[np.ndarray] = []
        self.cumulative_H: list[np.ndarray] = []
        self.offset_H: np.ndarray = np.eye(3)
        self.panorama: Optional[np.ndarray] = None
        self.canvas_w: int = 0
        self.canvas_h: int = 0

    def align_frames(self, ref_index: Optional[int] = None,
                     max_features: int = 3000, use_affine: bool = True):
        """Compute transforms to align all frames to a reference frame.
        By default, the middle frame is used as reference for stability.

        Args:
            use_affine: Use affine (translation+rotation+scale) instead of
                        full homography. Better for panning/tracking cameras.
        """
        if ref_index is None:
            ref_index = len(self.frames) // 2

        self.cumulative_H = compute_direct_homographies(
            self.frames, ref_index=ref_index,
            max_features=max_features, use_affine=use_affine,
        )

    def build_panorama(self, method: str = "median",
                       backdrop_density: int = 12):
        """Build the expanded panoramic background.

        Args:
            backdrop_density: Number of evenly-spaced frames to use.
                Lower = faster & less memory, higher = smoother backdrop.
        """
        self.panorama, self.canvas_w, self.canvas_h, self.offset_H = \
            build_panoramic_background(
                self.frames, self.cumulative_H, method=method,
                backdrop_density=backdrop_density,
            )

    def get_panorama_preview(self) -> Optional[np.ndarray]:
        """Return the panoramic background for preview."""
        return self.panorama

    def segment_keyframe(self, frame_list_index: int,
                         threshold: int = 35, morph_size: int = 7,
                         dilate_iterations: int = 0,
                         erode_iterations: int = 0,
                         feather_radius: int = 3) -> tuple:
        """Segment the subject from a keyframe in panorama coords.
        Returns (warped_frame, mask) on the panoramic canvas."""
        if frame_list_index >= len(self.cumulative_H):
            frame_list_index = len(self.cumulative_H) - 1

        H_total = self.offset_H @ self.cumulative_H[frame_list_index]
        return segment_foreground_moving(
            self.frames[frame_list_index], self.panorama,
            H_total, self.canvas_w, self.canvas_h,
            threshold=threshold, morph_size=morph_size,
            dilate_iterations=dilate_iterations,
            erode_iterations=erode_iterations,
            feather_radius=feather_radius,
        )

    def generate(self, keyframe_list_indices: list[int],
                 threshold: int = 35, morph_size: int = 7,
                 dilate_iterations: int = 0,
                 erode_iterations: int = 0,
                 feather_radius: int = 3,
                 opacity: float = 1.0, shadow: bool = True) -> np.ndarray:
        """Generate the final panoramic stop-motion composite."""
        warped_frames = []
        masks = []

        for li in keyframe_list_indices:
            warped, mask = self.segment_keyframe(
                li, threshold, morph_size,
                dilate_iterations=dilate_iterations,
                erode_iterations=erode_iterations,
                feather_radius=feather_radius,
            )
            warped_frames.append(warped)
            masks.append(mask)

        return composite_stop_motion(
            self.panorama, warped_frames, masks,
            opacity=opacity, shadow=shadow,
        )
