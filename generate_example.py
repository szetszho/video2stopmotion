"""Generate the example output image for the README."""

import cv2
import numpy as np


def generate_example_image():
    """Generate a high-quality synthetic example of the stop-motion effect."""
    w, h = 1600, 900

    # --- Background: blue sky gradient + snow slope ---
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        # Deep blue sky at top, lighter near horizon
        b = int(220 - 60 * ratio)
        g = int(180 - 80 * ratio)
        r = int(80 - 30 * ratio)
        bg[y, :] = [max(0, b), max(0, g), max(0, r)]

    # Snow slope (white/light gray polygon)
    slope_pts = np.array([
        [0, h],
        [0, int(h * 0.68)],
        [int(w * 0.15), int(h * 0.72)],
        [int(w * 0.35), int(h * 0.80)],
        [int(w * 0.5), int(h * 0.85)],
        [w, int(h * 0.92)],
        [w, h],
    ], np.int32)
    cv2.fillPoly(bg, [slope_pts], (248, 245, 240))

    # Snow texture (subtle noise on the slope)
    snow_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(snow_mask, [slope_pts], 255)
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    bg_noisy = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bg = np.where(snow_mask[:, :, None] > 0, bg_noisy, bg)

    # Distant mountains silhouette
    mt_pts = np.array([
        [int(w * 0.55), int(h * 0.50)],
        [int(w * 0.62), int(h * 0.38)],
        [int(w * 0.70), int(h * 0.42)],
        [int(w * 0.78), int(h * 0.35)],
        [int(w * 0.85), int(h * 0.40)],
        [int(w * 0.92), int(h * 0.45)],
        [w, int(h * 0.50)],
        [w, int(h * 0.92)],
        [int(w * 0.55), int(h * 0.87)],
    ], np.int32)
    mountain_color = (210, 195, 170)
    overlay = bg.copy()
    cv2.fillPoly(overlay, [mt_pts], mountain_color)
    cv2.addWeighted(overlay, 0.4, bg, 0.6, 0, bg)

    # Snow ramp/kicker
    ramp_pts = np.array([
        [int(w * 0.08), int(h * 0.70)],
        [int(w * 0.14), int(h * 0.55)],
        [int(w * 0.18), int(h * 0.52)],
        [int(w * 0.22), int(h * 0.58)],
        [int(w * 0.26), int(h * 0.72)],
    ], np.int32)
    cv2.fillPoly(bg, [ramp_pts], (252, 250, 248))
    # Ramp edge highlight
    cv2.polylines(bg, [ramp_pts], False, (230, 228, 225), 2, cv2.LINE_AA)

    result = bg.copy()

    # --- Draw athlete silhouettes along a parabolic arc ---
    num_poses = 8
    body_color = (20, 100, 240)  # Orange in BGR

    # Trajectory: launch from ramp, arc through the air
    launch_x, launch_y = int(w * 0.20), int(h * 0.55)
    land_x, land_y = int(w * 0.82), int(h * 0.88)

    positions = []
    for i in range(num_poses):
        t = i / (num_poses - 1)
        # Parabolic arc
        x = int(launch_x + t * (land_x - launch_x))
        y = int(launch_y + t * (land_y - launch_y) - 280 * np.sin(t * np.pi))
        # Rotation: athlete rotates during jump
        rotation = t * 360 * 1.5  # 1.5 full rotations
        positions.append((x, y, rotation, t))

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
        # Helmet highlight
        cv2.circle(img, p_head, max(6, int(10 * scale)), (30, 120, 255), 1, cv2.LINE_AA)

        # Arms (slightly different angles for dynamism)
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

    # Draw shadow on the ground for each pose
    for x, y, rot_deg, t in positions:
        # Project shadow onto slope
        shadow_y = int(h * 0.72 + t * (h * 0.20))
        shadow_x = x
        shadow_size = max(10, int(30 - 15 * np.sin(t * np.pi)))
        shadow_alpha = 0.12
        overlay = result.copy()
        cv2.ellipse(overlay, (shadow_x, shadow_y), (shadow_size, 6), 0, 0, 360, (120, 120, 130), -1)
        cv2.addWeighted(overlay, shadow_alpha, result, 1 - shadow_alpha, 0, result)

    # Draw trajectory arc (dashed)
    for i in range(200):
        t = i / 199
        px = int(launch_x + t * (land_x - launch_x))
        py = int(launch_y + t * (land_y - launch_y) - 280 * np.sin(t * np.pi))
        if i % 6 < 3:
            cv2.circle(result, (px, py), 1, (200, 210, 255), -1, cv2.LINE_AA)

    # Draw each athlete pose
    for x, y, rot_deg, t in positions:
        draw_athlete(result, x, y, rot_deg, scale=1.1, color=(20, 100, 240))

    # --- Labels ---
    # Semi-transparent banner at top
    banner = result.copy()
    cv2.rectangle(banner, (0, 0), (w, 70), (60, 40, 20), -1)
    cv2.addWeighted(banner, 0.6, result, 0.4, 0, result)

    cv2.putText(result, "STOP-MOTION SPORTS ANALYSIS", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"{num_poses} keyframes | Median background | Auto-segmentation",
                (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1, cv2.LINE_AA)

    # Time labels under each pose
    for i, (x, y, _, t) in enumerate(positions):
        label = f"t={t * 2.0:.1f}s"
        label_y = y + 70
        cv2.putText(result, label, (x - 20, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return result


if __name__ == "__main__":
    img = generate_example_image()
    cv2.imwrite("example_output.png", img)
    print("Generated example_output.png")
