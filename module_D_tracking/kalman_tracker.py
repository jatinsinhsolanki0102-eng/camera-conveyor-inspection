"""
Module D — Kalman Filter Tracking
===================================
Implements a Kalman Filter FROM SCRATCH for box tracking on a real conveyor
video. Measurements are produced by the Module A segmentation foreground
detector (box centroid of the tracked object), so the whole chain is:

    video -> Module A segmentation -> box centroid measurement
          -> Kalman filter (predict / measure / update)
          -> plots + annotated overlay video

State vector: x = [x, y, vx, vy] (position + velocity)
Motion model: Constant velocity (CV)

The filter implements:
  Prediction: x_pred = F * x, P_pred = F * P * F^T + Q
  Update:     K = P * H^T * (H * P * H^T + R)^{-1}
              x = x_pred + K * (z - H * x_pred)
              P = (I - K * H) * P_pred

A 3-5 frame measurement dropout is injected to demonstrate tracking through
occlusion (during dropout only the prediction step runs).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_project_root():
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / 'module_A_segmentation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =============================================================================
# KALMAN FILTER FROM SCRATCH
# =============================================================================
class KalmanFilterScratch:
    """
    Kalman Filter implemented from scratch.

    State vector: x = [x, y, vx, vy]^T

    State transition (Constant Velocity Model):
        x_{k+1} = x_k + vx_k * dt
        y_{k+1} = y_k + vy_k * dt
        vx_{k+1} = vx_k
        vy_{k+1} = vy_k

    In matrix form: x_{k+1} = F * x_k + w_k

    F = [1  0  dt  0 ]    Process noise w ~ N(0, Q)
        [0  1  0   dt]
        [0  0  1   0 ]
        [0  0  0   1 ]

    Measurement: z = H * x + v_k
    (We observe only position, not velocity)

    H = [1 0 0 0]    Measurement noise v ~ N(0, R)
        [0 1 0 0]
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=25.0):
        """
        Parameters:
        -----------
        dt : time step (frame duration in seconds)
        process_noise : std dev of process noise (how much velocity can change)
        measurement_noise : variance of measurement noise (sensor accuracy)
        """
        self.dt = dt

        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float64)

        # State covariance matrix (initial uncertainty)
        self.P = np.eye(4, dtype=np.float64) * 100.0

        # State transition matrix (constant velocity model)
        # x_new = F @ x_old
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float64)

        # Measurement matrix (we observe position only)
        # z = H @ x
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Process noise covariance Q
        # Models uncertainty in constant-velocity assumption.
        # Higher Q = filter trusts measurements more, reacts faster.
        q = process_noise
        self.Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ], dtype=np.float64) * q

        # Measurement noise covariance R
        # Models sensor (segmentation) uncertainty.
        r = measurement_noise
        self.R = np.eye(2, dtype=np.float64) * r

        # Identity matrix
        self.I = np.eye(4, dtype=np.float64)

    def predict(self):
        """
        Prediction step:

        x_pred = F @ x
        P_pred = F @ P @ F^T + Q

        This propagates the state forward using the motion model
        and increases uncertainty (P) due to process noise.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """
        Measurement update step:

        Innovation: y = z - H @ x_pred
        Innovation covariance: S = H @ P @ H^T + R
        Kalman gain: K = P @ H^T @ S^{-1}

        Corrected state: x = x_pred + K @ y
        Corrected covariance: P = (I - K @ H) @ P_pred

        The Kalman gain K determines how much to trust the measurement:
        - If K is large (measurement trusted more): x moves toward z
        - If K is small (model trusted more): x stays near prediction
        """
        z = np.array(z, dtype=np.float64).reshape(2, 1)

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Corrected state estimate
        self.x = self.x + K @ y

        # Corrected covariance estimate
        self.P = (self.I - K @ self.H) @ self.P

        return self.x.copy()

    def get_position(self):
        """Return current position estimate [x, y]."""
        return self.x[:2].flatten()

    def get_velocity(self):
        """Return current velocity estimate [vx, vy]."""
        return self.x[2:].flatten()

    def predict_only(self):
        """Prediction without measurement update (dropout/occlusion)."""
        return self.predict()


# =============================================================================
# Measurements from Module A segmentation
# =============================================================================
def extract_box_measurements(video_path, frame_limit=None, gate_px=400.0):
    """
    Run the Module A foreground detector on each video frame and extract the
    centroid of the tracked box as a (x, y) measurement.

    A simple nearest-neighbour association chains the same box across frames
    (the contour centroid closest to the previous measurement is kept), so
    that other passing boxes do not cause jumps.

    Returns:
        measurements : (N, 2) array of centroids, np.nan when no box was found
    """
    from module_A_segmentation.segmentation import detect_foreground_mask

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    measurements = []
    previous = None
    frame_idx = 0
    bg = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        mask = detect_foreground_mask(frame, bg if bg is not None else None)
        mask = mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if bg is None:
            bg = gray.astype(np.float32)
        else:
            bg = bg * 0.90 + gray.astype(np.float32) * 0.10

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cents = []
        for c in contours:
            if cv2.contourArea(c) < 100:      # drop specks of noise
                continue
            M = cv2.moments(c)
            if M['m00'] < 1e-3:
                continue
            cents.append((M['m10'] / M['m00'], M['m01'] / M['m00']))
        cents = np.array(cents).reshape(-1, 2)

        if len(cents) == 0:
            measurements.append(np.full(2, np.nan))
        elif previous is None:
            # First object seen: take the largest blob, then lock onto it.
            area = [cv2.contourArea(c) for c in contours
                    if cv2.contourArea(c) >= 100]
            chosen = int(np.argmax(area)) if area else 0
            chosen = min(chosen, len(cents) - 1)
            measurements.append(cents[chosen].copy())
            previous = cents[chosen].copy()
        else:
            # Nearest contour centroid to the previous measurement (gate) —
            # keeps the SAME box across frames so other boxes do not jump in.
            dists = np.hypot(*(cents - previous).T)
            chosen = int(np.argmin(dists))
            if dists[chosen] > gate_px:      # too far -> treat as missing
                measurements.append(np.full(2, np.nan))
            else:
                measurements.append(cents[chosen].copy())
                previous = cents[chosen].copy()

        frame_idx += 1
        if frame_limit is not None and frame_idx >= frame_limit:
            break

    cap.release()
    return np.array(measurements)


def extract_box_measurements_first_largest(video_path, frame_limit=None):
    """Fallback measurement source: always the largest foreground blob."""
    from module_A_segmentation.segmentation import detect_foreground_mask

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    measurements = []
    frame_idx = 0
    bg = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = detect_foreground_mask(frame, bg if bg is not None else None)
        mask = mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        if bg is None:
            bg = gray.astype(np.float32)
        else:
            bg = bg * 0.90 + gray.astype(np.float32) * 0.10
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 100:
                M = cv2.moments(largest)
                measurements.append((M['m10'] / M['m00'], M['m01'] / M['m00']))
            else:
                measurements.append((np.nan, np.nan))
        else:
            measurements.append((np.nan, np.nan))
        frame_idx += 1
        if frame_limit is not None and frame_idx >= frame_limit:
            break
    cap.release()
    return np.array(measurements)


def generate_box_trajectory(n_frames=30, speed_px=5.0, noise_std=8.0):
    """Synthetic fallback trajectory (used only when no video is available)."""
    np.random.seed(42)
    true_positions = []
    measurements = []
    x, y = 50.0, 200.0
    vx, vy = speed_px, 1.0
    for _ in range(n_frames):
        x += vx
        y += vy
        true_positions.append((x, y))
        mx = x + np.random.randn() * noise_std
        my = y + np.random.randn() * noise_std
        measurements.append((mx, my))
    return np.array(true_positions), np.array(measurements)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Kalman tracking from a video')
    parser.add_argument('--video', default=str(
        PROJECT_ROOT / 'data' / 'video' / 'conveyor.mp4'),
        help='Path to the conveyor video.')
    parser.add_argument('--frames', type=int, default=None,
        help='Limit the number of frames to process (default: all).')
    parser.add_argument('--dropout-start', type=int, default=None,
        help='Frame index where the measurement dropout starts '
             '(default: middle of the video).')
    parser.add_argument('--dropout-len', type=int, default=5,
        help='Number of consecutive dropped (occluded) frames (default 5).')
    parser.add_argument('--synthetic', action='store_true',
        help='Force the synthetic trajectory demo instead of the video.')
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_dir = project_root / 'outputs' / 'tracking'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KALMAN FILTER TRACKING — FROM SCRATCH")
    print("=" * 60)

    video_path = Path(args.video)

    if args.synthetic or not video_path.exists():
        if not args.synthetic:
            print(f"\nWARNING: Video {video_path} not found. Falling back to a")
            print("synthetic constant-velocity trajectory (Module D normally")
            print("uses Module A segmentation on the real video).")
            print("Place real footage at data/video/conveyor.mp4.")
        synthetic = True
    else:
        synthetic = False

    dropout_len = max(1, min(args.dropout_len, 6))

    # ------------------------------------------------------------------ #
    # 1. Get measurements (real: Module A segmentation; synthetic: noise) #
    # ------------------------------------------------------------------ #
    if synthetic:
        n_frames = args.frames or 25
        true_positions, measurements = generate_box_trajectory(
            n_frames=n_frames
        )
    else:
        n_frames = args.frames
        measurements = extract_box_measurements(video_path, frame_limit=n_frames)
        if np.isnan(measurements).all():
            print("\nWARNING: segmentation found no box; retrying with the")
            print("largest-blob fallback...")
            measurements = extract_box_measurements_first_largest(
                video_path, frame_limit=args.frames)
        n_frames = len(measurements)
        true_positions = None  # no ground truth on real footage

    if args.dropout_start is None:
        dropout_start = max(0, n_frames // 2 - dropout_len // 2)
    else:
        dropout_start = args.dropout_start
    dropout_frames = set(range(dropout_start, dropout_start + dropout_len))

    # ------------------------------------------------------------------ #
    # 2. Run the Kalman filter                                             #
    # ------------------------------------------------------------------ #
    kf = KalmanFilterScratch(
        dt=1.0, process_noise=1.0, measurement_noise=25.0
    )

    # Initialise with the first available measurement
    first = np.where(~np.isnan(measurements[:, 0]))[0]
    if len(first) == 0:
        print("ERROR: no measurements at all.")
        sys.exit(1)
    kf.x[0, 0] = measurements[first[0], 0]
    kf.x[1, 0] = measurements[first[0], 1]

    predictions = []
    corrections = []
    p_trace = []

    print(f"\nTracking {n_frames} frames; measurement dropout at frames "
          f"{dropout_start}-{dropout_start + dropout_len - 1}\n")

    for frame_idx in range(n_frames):
        # PREDICT (always)
        pred = kf.predict()

        z = measurements[frame_idx]
        if frame_idx in dropout_frames or np.isnan(z).all():
            # No measurement available — prediction only
            correction = pred.copy()
            status = "DROPOUT (prediction only)"
        else:
            correction = kf.update(z)
            status = "UPDATE"

        predictions.append(pred[:2].flatten().copy())
        corrections.append(correction[:2].flatten().copy())
        p_trace.append(np.trace(kf.P))

        # Print the first 10 frames explicitly + all dropout frames
        if frame_idx < 10 or frame_idx in dropout_frames:
            raws = (f"{z[0]:7.1f}" if not np.isnan(z[0]) else "    nan")
            rawy = (f"{z[1]:7.1f}" if not np.isnan(z[1]) else "    nan")
            print(f"  Frame {frame_idx + 1:3d}: "
                  f"Predicted=({pred[0,0]:7.1f}, {pred[1,0]:7.1f}) | "
                  f"Raw=({raws}, {rawy}) | "
                  f"Corrected=({correction[0,0]:7.1f}, {correction[1,0]:7.1f}) | "
                  f"{status}")

    predictions = np.array(predictions)
    corrections = np.array(corrections)
    valid_meas = np.arange(n_frames)
    meas_xy = np.where(np.isnan(measurements), np.nan, measurements)

    # ------------------------------------------------------------------ #
    # 3. Visualization: x/y vs frame + 2D trajectory path                 #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    frames = np.arange(1, n_frames + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, ax in enumerate(axes):
        ax.scatter(frames, measurements[valid_meas, i], c='red',
                   s=28, marker='x', label='Raw measurements (Module A)')
        ax.plot(frames, corrections[:, i], 'b-', linewidth=2,
                label='Kalman corrected')
        ax.plot(frames, predictions[:, i], 'k--', linewidth=1,
                label='Prediction (no meas.)')
        if true_positions is not None:
            ax.plot(frames, true_positions[:, i], 'g-', linewidth=2,
                    label='True position', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel(['X position (px)', 'Y position (px)'][i])
        ax.set_title(['X-axis tracking', 'Y-axis tracking'][i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvspan(dropout_start + 1, dropout_start + dropout_len,
                   alpha=0.2, color='yellow', label='Dropout')

    plt.suptitle('Kalman Filter Tracking (From Scratch)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'kalman_trajectory.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved: kalman_trajectory.png")

    # 2D trajectory overlay
    fig, ax = plt.subplots(figsize=(12, 8))
    if true_positions is not None:
        ax.plot(true_positions[:, 0], true_positions[:, 1], 'g-',
                linewidth=2, label='True path', alpha=0.7)
    ax.scatter(measurements[valid_meas, 0], measurements[valid_meas, 1],
               c='red', s=40, marker='x', label='Raw detections', alpha=0.7)
    ax.plot(corrections[:, 0], corrections[:, 1], 'b-', linewidth=2,
            label='Kalman filtered')
    d0 = dropout_start
    ax.plot(predictions[d0:d0 + dropout_len, 0],
            predictions[d0:d0 + dropout_len, 1], 'k--', linewidth=2,
            label='Prediction (dropout)')
    ax.set_aspect('equal')
    ax.set_title('Kalman Filter Tracking Overlay', fontsize=14)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'kalman_overlay.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved: kalman_overlay.png")

    # Trace of the covariance P (uncertainty) — shows it growing in dropout
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frames, p_trace, 'm-', linewidth=2)
    ax.axvspan(dropout_start + 1, dropout_start + dropout_len, alpha=0.2,
               color='yellow', label='Dropout (no measurements)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('trace(P) — state uncertainty')
    ax.set_title('Uncertainty growth during measurement dropout', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'kalman_uncertainty.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved: kalman_uncertainty.png")

    # ------------------------------------------------------------------ #
    # 4. Annotated overlay video                                          #
    # ------------------------------------------------------------------ #
    if not synthetic:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_dir / 'kalman_overlay.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if not np.isnan(measurements[i, 0]):
                cv2.drawMarker(frame, (int(measurements[i, 0]),
                                       int(measurements[i, 1])),
                               (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.circle(frame, (int(corrections[i, 0]), int(corrections[i, 1])),
                       5, (255, 0, 0), -1)
            cv2.circle(frame, (int(predictions[i, 0]), int(predictions[i, 1])),
                       2, (0, 0, 0), -1)
            cv2.putText(frame, f'raw=red filtered=blue pred=black',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        1, cv2.LINE_AA)
            writer.write(frame)
        cap.release()
        writer.release()
        print("  Saved: kalman_overlay.mp4")

    # ------------------------------------------------------------------ #
    # 5. Analysis                                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if synthetic:
        mc = (np.sqrt(np.sum((measurements[valid_meas] -
                              true_positions[valid_meas]) ** 2, axis=1)))
        kc = np.sqrt(np.sum((corrections - true_positions) ** 2, axis=1))
        print(f"\n  Measurement RMSE: {np.sqrt(np.mean(mc ** 2)):.2f} pixels")
        print(f"  Kalman RMSE:      {np.sqrt(np.mean(kc ** 2)):.2f} pixels")
        print(f"  Reduction:        "
              f"{(1 - np.sqrt(np.mean(kc ** 2)) / np.sqrt(np.mean(mc ** 2))) * 100:.1f}%")
    else:
        # No ground truth: report the innovation (measurement - filter) RMS,
        # i.e. how tightly the smooth path tracks the noisy detections, and
        # the velocity estimates during the run.
        innov = np.where(np.isnan(meas_xy), 0.0, meas_xy - corrections)
        non_nan = ~np.isnan(measurements[:, 0])
        innov_rms = float(np.sqrt(np.mean(
            np.sum(innov[non_nan] ** 2, axis=1))))
        print(f"\n  Innovation RMS (|z - filtered| over frames with a "
              f"measurement): {innov_rms:.2f} px")
        print(f"  Final KF velocity (px/frame): "
              f"{kf.get_velocity()[0]:.2f}, {kf.get_velocity()[1]:.2f}")
        print(f"  trace(P) before dropout: {p_trace[dropout_start - 1]:.1f} "
              f"-> after dropout: {p_trace[dropout_start + dropout_len - 1]:.1f}")
        print("  (P grows while the model extrapolates through the occlusion)")

    print("""
Kalman Filter Analysis:
----------------------

The Kalman filter from scratch:
1. Uses Module A segmentation detections as measurements (real video).
2. Smooths noisy detections (reduces centroid jitter between frames).
3. Maintains the track through the injected N-frame dropout using the
   constant-velocity model alone (prediction only).
4. Produces velocity estimates from position-only measurements.

During dropout:
  - Only the PREDICT step runs (no update), so the position is extrapolated
    at the learned velocity while uncertainty trace(P) grows.
  - When measurements resume, the filter corrects within a few frames.

This is the mechanism by which a conveyor inspection system keeps tracking
a box as it passes behind an obstacle or a segmentation failure window.
""")

    print(f"\nModule D outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()