"""Generate example output images for the README.

Creates two examples:
  1. Panoramic (moving camera) — wide canvas with stitched background
  2. Static camera — single-frame background
"""

import cv2
import numpy as np


def draw_athlete(img, cx, cy, rotation_deg, scale=1.0, color=(20, 100, 240)):
    """Draw a simplified athlete figure at given position and rotation."""
    rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rad), np.sin(rad)

    def rot(dx, dy):
        rx = int(cx + (dx * cos_r - dy * sin_r) * scale)
        ry = int(cy + (dx * sin_r + dy * cos_r) * scale)
        return (rx, ry)

    # Torso
    p_shoulder = rot(0, -20)
    p_hip = rot(0, 20)
    cv2.line(img, p_shoulder, p_hip, color, max(3, int(5 * scale)), cv2.LINE_AA)

    # Head
    p_head = rot(0, -32)
    cv2.circle(img, p_head, max(6, int(10 * scale)), color, -1, cv2.LINE_AA)
    cv2.circle(img, p_head, max(6, int(10 * scale)), (30, 120, 255), 1, cv2.LINE_AA)

    # Arms
    p_larm = rot(-28, -8)
    p_rarm = rot(25, -15)
    cv2.line(img, p_shoulder, p_larm, color, max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.line(img, p_shoulder, p_rarm, color, max(2, int(4 * scale)), cv2.LINE_AA)

    # Ski poles
    p_lpole = rot(-38, 10)
    p_rpole = rot(35, 5)
    cv2.line(img, p_larm, p_lpole, (100, 100, 100), max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.line(img, p_rarm, p_rpole, (100, 100, 100), max(1, int(2 * scale)), cv2.LINE_AA)

    # Legs
    p_lknee = rot(-10, 35)
    p_rknee = rot(10, 35)
    p_lfoot = rot(-18, 48)
    p_rfoot = rot(18, 48)
    cv2.line(img, p_hip, p_lknee, color, max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.line(img, p_hip, p_rknee, color, max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.line(img, p_lknee, p_lfoot, color, max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.line(img, p_rknee, p_rfoot, color, max(2, int(4 * scale)), cv2.LINE_AA)

    # Skis
    p_lski_f = rot(-30, 52)
    p_lski_b = rot(-5, 52)
    p_rski_f = rot(5, 52)
    p_rski_b = rot(30, 52)
    cv2.line(img, p_lski_f, p_lski_b, (50, 50, 50), max(2, int(3 * scale)), cv2.LINE_AA)
    cv2.line(img, p_rski_f, p_rski_b, (50, 50, 50), max(2, int(3 * scale)), cv2.LINE_AA)


def generate_panoramic_example():
    """Generate a wide panoramic example (moving camera result)."""
    # Extra-wide canvas to show the panoramic expansion
    w, h = 2200, 700

    # --- Sky gradient ---
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        b = int(230 - 70 * ratio)
        g = int(195 - 85 * ratio)
        r = int(95 - 45 * ratio)
        bg[y, :] = [max(0, b), max(0, g), max(0, r)]

    # --- Snow slope (fills bottom third) ---
    slope_pts = np.array([
        [0, h], [0, int(h * 0.65)],
        [int(w * 0.12), int(h * 0.68)],
        [int(w * 0.3), int(h * 0.75)],
        [int(w * 0.5), int(h * 0.80)],
        [int(w * 0.7), int(h * 0.82)],
        [int(w * 0.85), int(h * 0.78)],
        [w, int(h * 0.74)], [w, h],
    ], np.int32)
    cv2.fillPoly(bg, [slope_pts], (248, 245, 240))

    # Snow texture
    snow_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(snow_mask, [slope_pts], 255)
    noise = np.random.randint(-5, 5, (h, w, 3), dtype=np.int16)
    bg_noisy = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bg = np.where(snow_mask[:, :, None] > 0, bg_noisy, bg)

    # --- Distant mountains ---
    mt_pts = np.array([
        [int(w * 0.45), int(h * 0.50)],
        [int(w * 0.52), int(h * 0.38)],
        [int(w * 0.58), int(h * 0.42)],
        [int(w * 0.65), int(h * 0.33)],
        [int(w * 0.72), int(h * 0.38)],
        [int(w * 0.80), int(h * 0.40)],
        [int(w * 0.88), int(h * 0.35)],
        [int(w * 0.95), int(h * 0.42)],
        [w, int(h * 0.48)],
        [w, int(h * 0.82)],
        [int(w * 0.45), int(h * 0.80)],
    ], np.int32)
    overlay = bg.copy()
    cv2.fillPoly(overlay, [mt_pts], (210, 195, 170))
    cv2.addWeighted(overlay, 0.35, bg, 0.65, 0, bg)

    # --- Snow ramp on the left ---
    ramp_pts = np.array([
        [int(w * 0.04), int(h * 0.66)],
        [int(w * 0.08), int(h * 0.48)],
        [int(w * 0.11), int(h * 0.44)],
        [int(w * 0.14), int(h * 0.50)],
        [int(w * 0.17), int(h * 0.66)],
    ], np.int32)
    cv2.fillPoly(bg, [ramp_pts], (252, 250, 248))
    cv2.polylines(bg, [ramp_pts], False, (230, 228, 225), 2, cv2.LINE_AA)

    result = bg.copy()

    # --- Original camera frame indicator (dashed rectangle) ---
    # Shows approximately what a single camera frame would have captured
    frame_x1, frame_y1 = int(w * 0.30), int(h * 0.05)
    frame_x2, frame_y2 = int(w * 0.70), int(h * 0.95)
    for i in range(frame_y1, frame_y2, 12):
        cv2.line(result, (frame_x1, i), (frame_x1, min(i + 6, frame_y2)), (180, 180, 220), 1, cv2.LINE_AA)
        cv2.line(result, (frame_x2, i), (frame_x2, min(i + 6, frame_y2)), (180, 180, 220), 1, cv2.LINE_AA)
    for i in range(frame_x1, frame_x2, 12):
        cv2.line(result, (i, frame_y1), (min(i + 6, frame_x2), frame_y1), (180, 180, 220), 1, cv2.LINE_AA)
        cv2.line(result, (i, frame_y2), (min(i + 6, frame_x2), frame_y2), (180, 180, 220), 1, cv2.LINE_AA)
    cv2.putText(result, "single frame FOV", (frame_x1 + 5, frame_y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 220), 1, cv2.LINE_AA)

    # --- Athlete poses across the full panoramic width ---
    num_poses = 9
    launch_x, launch_y = int(w * 0.10), int(h * 0.48)
    land_x, land_y = int(w * 0.90), int(h * 0.75)

    positions = []
    for i in range(num_poses):
        t = i / (num_poses - 1)
        x = int(launch_x + t * (land_x - launch_x))
        y = int(launch_y + t * (land_y - launch_y) - 240 * np.sin(t * np.pi))
        rotation = t * 360 * 1.5
        positions.append((x, y, rotation, t))

    # Shadows
    for x, y, _, t in positions:
        shadow_y = int(h * 0.68 + t * (h * 0.14))
        shadow_size = max(10, int(28 - 14 * np.sin(t * np.pi)))
        overlay = result.copy()
        cv2.ellipse(overlay, (x, shadow_y), (shadow_size, 6), 0, 0, 360, (180, 180, 190), -1)
        cv2.addWeighted(overlay, 0.12, result, 0.88, 0, result)

    # Trajectory arc
    for i in range(400):
        t = i / 399
        px = int(launch_x + t * (land_x - launch_x))
        py = int(launch_y + t * (land_y - launch_y) - 240 * np.sin(t * np.pi))
        if i % 6 < 3:
            cv2.circle(result, (px, py), 1, (200, 210, 255), -1, cv2.LINE_AA)

    # Draw athletes
    for x, y, rot_deg, t in positions:
        draw_athlete(result, x, y, rot_deg, scale=1.2, color=(20, 100, 240))

    # Time labels
    for i, (x, y, _, t) in enumerate(positions):
        label = f"t={t * 2.5:.1f}s"
        cv2.putText(result, label, (x - 18, y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Banner ---
    banner = result.copy()
    cv2.rectangle(banner, (0, 0), (w, 65), (60, 40, 20), -1)
    cv2.addWeighted(banner, 0.6, result, 0.4, 0, result)

    cv2.putText(result, "PANORAMIC STOP-MOTION  |  Moving Camera  |  Expanded Background",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result,
                f"{num_poses} keyframes | Homography-aligned | Stitched panoramic canvas {w}x{h}",
                (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 210), 1, cv2.LINE_AA)

    # Expansion arrows at the edges
    arrow_y = h // 2
    cv2.arrowedLine(result, (50, arrow_y), (15, arrow_y), (220, 220, 255), 2, cv2.LINE_AA, tipLength=0.4)
    cv2.arrowedLine(result, (w - 50, arrow_y), (w - 15, arrow_y), (220, 220, 255), 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(result, "expanded", (5, arrow_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 240), 1, cv2.LINE_AA)
    cv2.putText(result, "expanded", (w - 65, arrow_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 240), 1, cv2.LINE_AA)

    return result


if __name__ == "__main__":
    img = generate_panoramic_example()
    cv2.imwrite("example_output.png", img)
    print(f"Generated example_output.png ({img.shape[1]}x{img.shape[0]})")
