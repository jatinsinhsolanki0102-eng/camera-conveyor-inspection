"""
Module C — Optical Flow
========================
Implements:
1. Lucas-Kanade optical flow FROM SCRATCH (mathematical implementation)
2. OpenCV sparse optical flow for comparison
3. Affine flow model with RANSAC
4. Belt speed estimation in cm/s

The Lucas-Kanade method solves the brightness constancy equation:
  Ix*u + Iy*v + It = 0
in a local window using least squares.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow cross-module imports (largest_box_region pulls in Module A's
# foreground detector). When this script is run as
#   python module_C_optical_flow/optical_flow.py
# the interpreter puts module_C_optical_flow on sys.path, not the repo root.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_project_root():
    return Path(__file__).parent.parent


def load_consecutive_frames(video_path, frame1_idx=0, frame2_idx=1):
    """Load two consecutive frames from a video."""
    cap = cv2.VideoCapture(str(video_path))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
    ret1, frame1 = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
    ret2, frame2 = cap.read()

    cap.release()

    if not ret1 or not ret2:
        raise ValueError(f"Could not read frames {frame1_idx} and {frame2_idx}")

    return frame1, frame2


def largest_box_region(frame):
    """
    Find the largest foreground blob (the biggest/closest box) in a frame.

    Reuses Module A's foreground detector so the flow and RANSAC stages work on
    the SAME box region the segmentation module identified, as the spec asks.
    Returns ((x, y, w, h), (cx, cy)) of that blob, or None.
    """
    from module_A_segmentation.segmentation import detect_foreground_mask
    fg = (detect_foreground_mask(frame) > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    return (x, y, w, h), (x + w // 2, y + h // 2)


def compute_image_gradients(gray):
    """
    Compute image gradients Ix, Iy, It using Sobel operators.

    The brightness constancy assumption states:
      I(x, y, t) = I(x+u, y+v, t+1)

    Taking the Taylor expansion:
      Ix*u + Iy*v + It ≈ 0

    Where:
      Ix = dI/dx (spatial gradient in x)
      Iy = dI/dy (spatial gradient in y)
      It = dI/dt (temporal gradient)
    """
    # Spatial gradients using Sobel.
    # NOTE: OpenCV does NOT normalize the Sobel kernel for any output depth;
    # the 3x3 kernel sums its weights to 8, so the raw response is 8x the true
    # derivative (verified: a 12.5 px^-1 edge returns 400). Dividing by 8 makes
    # Ix/Iy unit derivatives so they match It (a plain intensity difference) in
    # the brightness-constancy least-squares system.
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    return Ix, Iy


def compute_temporal_gradient(gray1, gray2):
    """
    Compute temporal gradient It = I(x,y,t+1) - I(x,y,t).

    Simple frame differencing for the temporal derivative.
    """
    It = gray2.astype(np.float64) - gray1.astype(np.float64)
    return It


# =============================================================================
# LUCAS-KANADE FROM SCRATCH
# =============================================================================
def lucas_kanade_from_scratch(gray1, gray2, window_size=15, step=4):
    """
    Lucas-Kanade optical flow implemented from scratch.

    ALGORITHM:
    ==========
    The brightness constancy constraint for a pixel (x, y) over time dt:

      Ix * u + Iy * v + It = 0

    where (u, v) is the optical flow velocity vector.

    For a local window W of size n x n pixels (n = window_size),
    we have n^2 equations. This is an overdetermined system:

      [Ix1 Iy1] [u]   [-It1]
      [Ix2 Iy2] [v] = [-It2]
      [... ...]       [...]
      [IxN IyN]       [-ItN]

    In matrix form: A * d = b

    We solve using the least-squares solution:
      d = (A^T A)^{-1} A^T b

    The solution exists only if A^T A is invertible (non-singular).
    A^T A = sum[Ix^2  Ix*Iy]
            sum[Ix*Iy  Iy^2]

    This is the structure tensor. Its eigenvalues determine
    whether the flow is well-defined:
      - Two large eigenvalues: corner (good)
      - One large, one small: edge (aperture problem)
      - Both small: textureless (no flow)

    Parameters:
    -----------
    gray1 : first grayscale frame (float64)
    gray2 : second grayscale frame (float64)
    window_size : local window size (odd integer)
    step : sampling step (compute flow every `step` pixels)

    Returns:
    --------
    flow : optical flow field (H, W, 2)
    valid_mask : boolean mask of valid flow vectors
    """
    h, w = gray1.shape
    half_w = window_size // 2

    # Compute image gradients
    Ix, Iy = compute_image_gradients(gray1)
    It = compute_temporal_gradient(gray1, gray2)

    # Initialize flow field
    flow = np.zeros((h, w, 2), dtype=np.float64)
    valid_mask = np.zeros((h, w), dtype=bool)

    # Process each pixel (with step for efficiency)
    for y in range(half_w, h - half_w, step):
        for x in range(half_w, w - half_w, step):
            # Extract local window gradients
            Ix_window = Ix[y - half_w:y + half_w + 1,
                           x - half_w:x + half_w + 1].flatten()
            Iy_window = Iy[y - half_w:y + half_w + 1,
                           x - half_w:x + half_w + 1].flatten()
            It_window = It[y - half_w:y + half_w + 1,
                           x - half_w:x + half_w + 1].flatten()

            # Build the matrix A and vector b
            # A = [Ix1 Iy1; Ix2 Iy2; ...; IxN IyN]
            # b = [-It1; -It2; ...; -ItN]
            A = np.column_stack([Ix_window, Iy_window])
            b = -It_window

            # Compute A^T A (structure tensor / second moment matrix)
            ATA = A.T @ A

            # Check eigenvalues for reliability. An absolute gate (1e-6) is far
            # below every real value and so never rejects anything; a RELATIVE
            # test is what captures the aperture problem: on an edge or a
            # textureless surface the smaller eigenvalue is a tiny fraction of
            # the larger one, so the normal-direction component is unobservable.
            eigenvalues = np.linalg.eigvalsh(ATA)
            min_eigenvalue = eigenvalues[0]
            max_eigenvalue = eigenvalues[1]

            # Only compute flow if the structure tensor is well-conditioned
            # (both eigenvalues significant — not textureless or edge)
            if min_eigenvalue > 1.0 and min_eigenvalue > 1e-3 * max_eigenvalue:
                # Least squares solution: d = (A^T A)^{-1} A^T b
                ATb = A.T @ b
                try:
                    d = np.linalg.solve(ATA, ATb)
                    # A still-nearly-singular window can pass the relative gate
                    # and return an absurd (hundreds of px) motion. Cap at a
                    # physically plausible inter-frame displacement: nothing on
                    # a belt moves hundreds of pixels between two frames.
                    if np.hypot(d[0], d[1]) > 15.0:
                        continue
                    flow[y, x, 0] = d[0]  # u (horizontal)
                    flow[y, x, 1] = d[1]  # v (vertical)
                    valid_mask[y, x] = True
                except np.linalg.LinAlgError:
                    pass  # Singular matrix — skip this pixel

    return flow, valid_mask


# =============================================================================
# OPENCV SPARSE OPTICAL FLOW (for comparison)
# =============================================================================
def opencv_sparse_flow(gray1, gray2):
    """
    OpenCV's implementation of Lucas-Kanade optical flow.

    Uses cv2.goodFeaturesToTrack() for feature selection and
    cv2.calcOpticalFlowPyrLK() for pyramidal Lucas-Kanade tracking.

    This is a fast, optimized implementation using image pyramids
    for handling large displacements.
    """
    # Detect good features to track (Shi-Tomasi corners)
    features = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=7
    )

    if features is None or len(features) == 0:
        return np.array([]), np.array([]), np.array([])

    # Calculate optical flow using pyramidal Lucas-Kanade
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, features, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # Filter valid tracks
    good_mask = status.flatten() == 1
    good_old = features[good_mask].reshape(-1, 2)
    good_new = next_pts[good_mask].reshape(-1, 2)

    return good_old, good_new, good_mask


# =============================================================================
# AFFINE FLOW + RANSAC
# =============================================================================
def fit_affine_flow(points_old, points_new):
    """
    Fit a 6-parameter affine flow model to flow vectors.

    Affine model:
      u = a1*x + a2*y + a3
      v = a4*x + a5*y + a6

    We solve two independent least-squares problems:
      [x y 1] [a1]   [u]
      [...]   [a2] = [...]
              [a3]

    Returns: (a1, a2, a3, a4, a5, a6) — the 6 affine parameters
    """
    if len(points_old) < 3:
        return None

    # Flow vectors
    dx = points_new[:, 0] - points_old[:, 0]
    dy = points_new[:, 1] - points_old[:, 1]

    # Design matrix: [x, y, 1]
    A = np.column_stack([
        points_old[:, 0],
        points_old[:, 1],
        np.ones(len(points_old))
    ])

    # Solve for u-flow parameters: u = a1*x + a2*y + a3
    params_u, _, _, _ = np.linalg.lstsq(A, dx, rcond=None)

    # Solve for v-flow parameters: v = a4*x + a5*y + a6
    params_v, _, _, _ = np.linalg.lstsq(A, dy, rcond=None)

    return np.concatenate([params_u, params_v])


def ransac_affine_flow(points_old, points_new, num_iterations=500, threshold=2.0, seed=None):
    """
    RANSAC to robustly fit affine flow model, rejecting outliers.

    RANSAC Algorithm:
    1. Randomly select minimum number of points (3 for affine)
    2. Fit model to these points
    3. Count inliers (points fitting model within threshold)
    4. Repeat for N iterations
    5. Refit using all inliers from best model

    This is essential because optical flow can have outliers due to:
    - Occlusion
    - Textureless regions
    - Motion boundaries
    """
    if len(points_old) < 3:
        return None, np.array([], dtype=bool), np.array([], dtype=bool)

    best_inliers = np.array([], dtype=bool)
    best_params = None

    n_points = len(points_old)
    rng = np.random.default_rng(seed)

    for _ in range(num_iterations):
        # Step 1: Random sample of 3 points
        indices = rng.choice(n_points, 3, replace=False)
        sample_old = points_old[indices]
        sample_new = points_new[indices]

        # Step 2: Fit affine model to sample
        params = fit_affine_flow(sample_old, sample_new)
        if params is None:
            continue

        # Step 3: Compute residuals for all points
        dx_pred = params[0] * points_old[:, 0] + params[1] * points_old[:, 1] + params[2]
        dy_pred = params[3] * points_old[:, 0] + params[4] * points_old[:, 1] + params[5]

        dx_actual = points_new[:, 0] - points_old[:, 0]
        dy_actual = points_new[:, 1] - points_old[:, 1]

        errors = np.sqrt((dx_pred - dx_actual)**2 + (dy_pred - dy_actual)**2)

        # Step 4: Count inliers
        inlier_mask = errors < threshold
        n_inliers = np.sum(inlier_mask)

        if n_inliers > np.sum(best_inliers):
            best_inliers = inlier_mask
            best_params = params

    # Step 5: Refit using all inliers
    if np.sum(best_inliers) >= 3:
        best_params = fit_affine_flow(
            points_old[best_inliers],
            points_new[best_inliers]
        )

    outlier_mask = ~best_inliers
    return best_params, best_inliers, outlier_mask


def visualize_flow_arrows(frame, flow, valid_mask, step=8, output_path=None):
    """Visualize optical flow as quiver arrows on the image."""
    h, w = frame.shape[:2]
    vis = frame.copy()

    # Take the actually-computed pixels (valid_mask is only True where flow was
    # solved) and subsample by `step`. Sampling a fresh grid with np.mgrid and
    # then only keeping points that coincide with computed pixels is why the
    # old code drew zero arrows: it sampled even coordinates while the scratch
    # LK computed odd ones, so the intersection was empty.
    ys, xs = np.where(valid_mask)
    order = np.argsort(ys * w + xs)
    ys, xs = ys[order], xs[order]
    send = slice(0, len(ys), step)

    for yi, xi in zip(ys[send], xs[send]):
        u_val = flow[yi, xi, 0]
        v_val = flow[yi, xi, 1]
        magnitude = np.sqrt(u_val**2 + v_val**2)

        if magnitude < 0.3:  # Skip very small flows
            continue

        # Arrow end point
        end_x = int(xi + u_val * 4)  # Scale for visibility
        end_y = int(yi + v_val * 4)

        # Color by magnitude
        color_val = min(255, int(magnitude * 50))
        color = (0, color_val, 255 - color_val)

        cv2.arrowedLine(vis, (int(xi), int(yi)), (int(end_x), int(end_y)),
                        color, 1, tipLength=0.3)

    if output_path:
        cv2.imwrite(str(output_path), vis)

    return vis


def visualize_flow_field(frame1, flow, valid_mask, output_path=None):
    """Create a color-coded optical flow visualization."""
    h, w = frame1.shape[:2]

    # Convert flow to HSV
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 1] = 255

    # Compute magnitude and angle
    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])

    # Map to HSV
    hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[:, :, 2] = np.clip(magnitude * 20, 0, 255).astype(np.uint8)

    # Convert to RGB
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if output_path:
        plt.figure(figsize=(10, 8))
        plt.imshow(flow_rgb)
        plt.title('Optical Flow (Color-coded)')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return flow_rgb


# =============================================================================
# BELT SPEED ESTIMATION
# =============================================================================
def load_pixel_to_cm(project_root):
    """
    Real-world scale (cm per pixel) on the belt plane from Module B calibration.

    The board->image homography stored by calibrate.py maps checkerboard
    coordinates (in units of squares) to pixels. If the checkerboard square is
    `square_size_cm`, then walking one square along the board's X axis moves
    `norm(H[:, 0])` pixels in the image, so:

        pixels per cm along the plane  =  norm(H[:, 0]) / square_size_cm
        cm per pixel                   =  square_size_cm / norm(H[:, 0])

    The belt is flush with the board plane, so this scale applies to belt
    velocities. Returns (scale, source) or (None, reason).
    """
    calib_file = project_root / 'outputs' / 'calibration' / 'calibration.npz'
    if not calib_file.exists():
        return None, "no calibration.npz produced (run module_B_calibration/calibrate.py)"

    data = np.load(calib_file)
    req = ['K', 'H_board_to_image', 'square_size_cm']
    if not all(k in data for k in req):
        missing = [k for k in req if k not in data]
        return None, f"calibration.npz missing {missing}"

    H = data['H_board_to_image']
    square_cm = float(data['square_size_cm'])
    norm_x = float(np.linalg.norm(H[:, 0]))
    if norm_x < 1e-9:
        return None, "degenerate stored homography"
    cm_per_px = square_cm / norm_x
    return cm_per_px, f"calibration homography ({square_cm} cm squares)"


def estimate_belt_speed(affine_params, centroid, fps, pixel_to_cm,
                        pixel_to_cm_source=None):
    """
    Belt speed in cm/s from the RANSAC-fitted affine flow model.

    The affine model describes box motion: at the box centroid (cx, cy) the
    velocity per frame is the affine transform of the position, u = a1*x + a2*y
    + a3 (and likewise for v). Because the RANSAC fit uses only inlier flow
    vectors from ONE box region, the median-frame displacement at the centroid
    is the belt's real motion, free of background/parallax outliers.

    speed = |(du, dv) per frame| * (pixels/frame to cm/frame) * fps
    """
    cx, cy = centroid
    du = affine_params[0] * cx + affine_params[1] * cy + affine_params[2]
    dv = affine_params[3] * cx + affine_params[4] * cy + affine_params[5]
    disp_px_per_frame = float(np.hypot(du, dv))

    digits = 2
    if pixel_to_cm is None:
        # Nothing measured the scene scale yet: report a physical assuming the
        # default 0.05 cm/px (README's rough floor-scale) but say so loudly.
        pixel_to_cm = 0.05
        pixel_to_cm_source = ("ASSUMED 0.05 cm/px (no calibration; run "
                              "module_B_calibration/calibrate.py for a real scale)")
        digits = 3

    speed_cm_per_sec = disp_px_per_frame * fps * pixel_to_cm
    if pixel_to_cm_source:
        print(f"  Scale: {pixel_to_cm_source}")

    return speed_cm_per_sec, disp_px_per_frame


# =============================================================================
# Main
# =============================================================================
def main():
    project_root = get_project_root()
    output_dir = project_root / 'outputs' / 'optical_flow'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find video
    video_path = project_root / 'data' / 'video' / 'conveyor.mp4'
    if not video_path.exists():
        alt_paths = list((project_root / 'data' / 'video').glob('*.mp4')) + \
                    list((project_root / 'data' / 'video').glob('*.avi')) + \
                    list((project_root / 'data' / 'video').glob('*.mov'))
        if alt_paths:
            video_path = alt_paths[0]
        else:
            print("ERROR: No video found.")
            print(f"Please place your video at: {project_root / 'data' / 'video' / 'conveyor.mp4'}")
            sys.exit(1)

    print(f"Using video: {video_path}")

    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {total_frames} frames, {fps:.1f} FPS")

    # Load two consecutive frames
    print("\nLoading frames...")
    frame1, frame2 = load_consecutive_frames(video_path, 10, 11)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # =========================================================================
    # Part 1: Lucas-Kanade from scratch
    # =========================================================================
    print("\n" + "=" * 60)
    print("Part 1: Lucas-Kanade FROM SCRATCH")
    print("=" * 60)

    flow_scratch, valid_scratch = lucas_kanade_from_scratch(
        gray1, gray2, window_size=15, step=4
    )

    n_valid = np.sum(valid_scratch)
    total_pixels = valid_scratch.size
    print(f"  Computed flow for {n_valid}/{total_pixels} pixels")

    # Visualize
    visualize_flow_arrows(
        frame1, flow_scratch, valid_scratch,
        step=10,
        output_path=output_dir / 'lk_scratch_arrows.png'
    )
    visualize_flow_field(
        frame1, flow_scratch, valid_scratch,
        output_path=output_dir / 'lk_scratch_field.png'
    )

    print("  Saved: lk_scratch_arrows.png, lk_scratch_field.png")

    # =========================================================================
    # Part 2: OpenCV sparse optical flow
    # =========================================================================
    print("\n" + "=" * 60)
    print("Part 2: OpenCV Sparse Optical Flow")
    print("=" * 60)

    pts_old, pts_new, good_mask = opencv_sparse_flow(
        gray1.astype(np.uint8), gray2.astype(np.uint8)
    )

    print(f"  Tracked {len(pts_old)} features")

    # Visualize sparse flow
    vis_sparse = frame1.copy()
    for old, new in zip(pts_old, pts_new):
        old_pt = (int(old[0]), int(old[1]))
        new_pt = (int(new[0]), int(new[1]))
        cv2.circle(vis_sparse, old_pt, 3, (0, 255, 0), -1)
        cv2.arrowedLine(vis_sparse, old_pt, new_pt, (0, 0, 255), 1, tipLength=0.3)

    cv2.imwrite(str(output_dir / 'opencv_sparse_flow.png'), vis_sparse)
    print("  Saved: opencv_sparse_flow.png")

    # =========================================================================
    # Part 3: Comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("Part 3: Comparison — Hand-built vs OpenCV")
    print("=" * 60)

    # Compare flow vectors at common feature locations
    if len(pts_old) > 0:
        # Sample some points from our scratch flow
        comparison_points = []
        for i in range(min(20, len(pts_old))):
            x, y = int(pts_old[i][0]), int(pts_old[i][1])
            if valid_scratch[y, x]:
                scratch_flow = flow_scratch[y, x]
                opencv_flow = pts_new[i] - pts_old[i]
                comparison_points.append({
                    'pos': (x, y),
                    'scratch': scratch_flow,
                    'opencv': opencv_flow
                })

        print(f"\n  Flow comparison at {len(comparison_points)} sample points:")
        print(f"  {'Position':>12} {'Scratch (u,v)':>20} {'OpenCV (u,v)':>20} {'Diff':>10}")
        print("  " + "-" * 65)

        for pt in comparison_points:
            diff = np.linalg.norm(pt['scratch'] - pt['opencv'])
            print(f"  ({pt['pos'][0]:4d},{pt['pos'][1]:4d})  "
                  f"({pt['scratch'][0]:8.2f},{pt['scratch'][1]:8.2f})  "
                  f"({pt['opencv'][0]:8.2f},{pt['opencv'][1]:8.2f})  "
                  f"{diff:8.2f}")

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(cv2.cvtColor(vis_sparse, cv2.COLOR_BGR2RGB))
    axes[0].set_title('OpenCV Sparse LK', fontsize=12)
    axes[0].axis('off')

    # The arrows image was already written by visualize_flow_arrows above;
    # load it back from disk (the old code passed a pathlib.Path straight
    # into cv2.cvtColor, which crashes on every version of OpenCV).
    scratch_path = output_dir / 'lk_scratch_arrows.png'
    scratch_bgr = cv2.imread(str(scratch_path))
    if scratch_bgr is None:
        scratch_bgr = frame1
    scratch_vis = cv2.cvtColor(scratch_bgr, cv2.COLOR_BGR2RGB)

    axes[1].imshow(scratch_vis)
    axes[1].set_title('Hand-built Lucas-Kanade', fontsize=12)
    axes[1].axis('off')

    plt.suptitle('Optical Flow Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("""
  Analysis:
  ---------
  The hand-built Lucas-Kanade and OpenCV's implementation may differ because:
  1. OpenCV uses image pyramids for handling large displacements
  2. OpenCV uses sub-pixel refinement and iterative refinement
  3. Feature selection (Shi-Tomasi corners) concentrates flow on texture-rich areas

  On textureless box surfaces, both methods struggle (aperture problem):
  - Only the gradient-normal component of flow is observable
  - Flow along edges is ambiguous
  - This is a fundamental limitation of local brightness constancy methods.
""")

    # =========================================================================
    # Part 4: Affine Flow + RANSAC (on ONE box region)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Part 4: Affine Flow Model + RANSAC")
    print("=" * 60)

    box = largest_box_region(frame1)
    if box is None:
        print("  No foreground box detected in frame — skipping affine fit.")
        params_ransac = None
        box_centroid = None
    else:
        bbox, box_centroid = box
        bx, by, bw, bh = bbox
        print(f"  Box region: bbox=({bx},{by},{bw}x{bh}) centroid={box_centroid}")

        # Spec: the affine model + RANSAC is fit on ONE box region's flow
        # vectors, not the whole frame (which mixes the belt background and
        # motion boundaries into the fit). Restrict the sparse flow features
        # to the detected box bounding box.
        if len(pts_old) >= 3:
            inside = (
                (pts_old[:, 0] >= bx) & (pts_old[:, 0] <= bx + bw) &
                (pts_old[:, 1] >= by) & (pts_old[:, 1] <= by + bh)
            )
            box_old, box_new = pts_old[inside], pts_new[inside]
            print(f"  Sparse features inside box region: {len(box_old)}")
        else:
            box_old, box_new = pts_old, pts_new

        if len(box_old) >= 3:
            # Fit affine flow
            affine_params = fit_affine_flow(box_old, box_new)
            if affine_params is not None:
                print(f"\n  Affine flow parameters (box region):")
                print(f"    u = {affine_params[0]:.4f}*x + {affine_params[1]:.4f}*y + {affine_params[2]:.4f}")
                print(f"    v = {affine_params[3]:.4f}*x + {affine_params[4]:.4f}*y + {affine_params[5]:.4f}")

            # RANSAC
            params_ransac, inliers, outliers = ransac_affine_flow(
                box_old, box_new, num_iterations=500, threshold=2.0, seed=7
            )

            n_inliers = np.sum(inliers)
            n_outliers = np.sum(outliers)
            print(f"\n  RANSAC results (box region):")
            print(f"    Inliers:  {n_inliers}")
            print(f"    Outliers: {n_outliers}")

            if params_ransac is not None:
                print(f"\n  RANSAC affine parameters:")
                print(f"    u = {params_ransac[0]:.4f}*x + {params_ransac[1]:.4f}*y + {params_ransac[2]:.4f}")
                print(f"    v = {params_ransac[3]:.4f}*x + {params_ransac[4]:.4f}*y + {params_ransac[5]:.4f}")

            # Visualize inliers/outliers
            vis_ransac = frame1.copy()
            for i, (old, new) in enumerate(zip(box_old, box_new)):
                old_pt = (int(old[0]), int(old[1]))
                new_pt = (int(new[0]), int(new[1]))
                if inliers[i]:
                    cv2.circle(vis_ransac, old_pt, 4, (0, 255, 0), -1)
                    cv2.arrowedLine(vis_ransac, old_pt, new_pt, (0, 255, 0), 1, tipLength=0.3)
                else:
                    cv2.circle(vis_ransac, old_pt, 4, (0, 0, 255), -1)
                    cv2.arrowedLine(vis_ransac, old_pt, new_pt, (0, 0, 255), 1, tipLength=0.3)

            # Draw the box region boundary so the ROI is visible in the figure
            cv2.rectangle(vis_ransac, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
            cv2.imwrite(str(output_dir / 'ransac_affine_flow.png'), vis_ransac)
            print("  Saved: ransac_affine_flow.png")
        else:
            params_ransac = None
            print(f"  Too few features in the box region ({len(box_old)}), skipping affine fit.")

    # =========================================================================
    # Part 5: Belt Speed Estimation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Part 5: Belt Speed Estimation")
    print("=" * 60)

    # Scale from Module B calibration (belt plane homography + known square size)
    pixel_to_cm, scale_source = load_pixel_to_cm(project_root)
    if pixel_to_cm is not None:
        print(f"  Using camera calibration for scale: {scale_source}")
    else:
        print(f"  Calibration not available ({scale_source}).")

    if params_ransac is not None:
        belt_speed, disp_px = estimate_belt_speed(
            params_ransac, box_centroid, fps,
            pixel_to_cm=pixel_to_cm,
            pixel_to_cm_source=scale_source if pixel_to_cm is not None else None
        )
        print(f"\n  Box-centre displacement: {disp_px:.2f} px/frame")
        print(f"  Estimated belt speed: {belt_speed:.2f} cm/s")
    else:
        print("\n  Cannot estimate belt speed (no RANSAC affine model of a box).")

    print("\nModule C optical flow complete.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
