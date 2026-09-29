"""
Module B — Box Measurement with 4 Camera Models
=================================================
Estimates the on-plane real-world size of an unknown box using a reference
box of known size, through four increasingly exact camera models:
  1. Orthographic        – one global px/cm scale from the reference box
  2. Weak perspective    – a per-observer scale from the local linear
                           (Jacobian) map of the belt plane at the target
  3. Affine              – the plane's 2x2 linear map anchored at the
                           reference (uniform for both boxes, includes
                           rotation/shear)
  4. Full perspective    – exact inversion of the belt-plane homography at
                           the measured image corners

All measurements are performed on UNDISTORTED images using the plane
homography H recovered in calibrate.py (H maps belt-plane cm to pixels),
so the whole pipeline is calibration-aware. The reference and unknown box
must lie on the same plane as the calibrated checkerboard (the belt).
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def get_project_root():
    return Path(__file__).parent.parent


# Allow importing the shared foreground mask from Module A (runs from the
# project root, or from inside module_B_calibration).
PROJECT_ROOT = get_project_root()
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / 'module_A_segmentation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_calibration(calib_path):
    """Load calibration parameters (incl. plane homography) from file."""
    calib_path = Path(calib_path)
    if not calib_path.exists():
        print(f"ERROR: Calibration file not found: {calib_path}")
        print("Run calibrate.py first to generate calibration data.")
        sys.exit(1)
    data = np.load(calib_path)
    out = {
        'K': data['K'],
        'dist': data['dist'],
        'image_size': tuple(data['image_size']),
    }
    if 'H_board_to_image' in data:
        # H maps board-plane SQUARE units -> pixels; convert to cm -> pixels
        s = float(data['square_size_cm'])
        H_sq = data['H_board_to_image']
        out['H_cm'] = H_sq @ np.diag([1.0 / s, 1.0 / s, 1.0])
        out['square_size_cm'] = s
    return out


def homography_for_undistorted(H_raw, K, dist, new_K):
    """
    Remap the plane homography so it holds in the UNDISTORTED image space.

    Samples 4 belt-plane points, projects them to the distorted image with
    H_raw, undistorts those pixels with cv2.undistortPoints (using new_K as
    the rectified intrinsic), and fits the plane->undistorted homography.
    """
    pts_cm = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], np.float32)
    ones = np.ones((len(pts_cm), 1), np.float32)
    distorted_pts = H_raw @ np.hstack([pts_cm, ones]).T
    distorted_2d = (distorted_pts[:2] / distorted_pts[2]).T.astype(np.float32)
    undistorted_2d = cv2.undistortPoints(distorted_2d, K, dist, None, new_K)
    H_u, _ = cv2.findHomography(pts_cm, undistorted_2d.reshape(-1, 2))
    return H_u


def homography_jacobian(H, pt_cm):
    """
    Local 2x2 Jacobian d(pixel)/d(cm) of the plane homography at a point on
    the belt plane. This is the weak-perspective / affine linearization and
    differs across the image when the plane is tilted.
    """
    eps = 1e-3
    p = np.array([pt_cm[0], pt_cm[1], 1.0])
    base = H @ p
    base_u = base[:2] / base[2]

    J = np.zeros((2, 2))
    for i in range(2):
        pd = p.copy()
        pd[i] += eps
        q = H @ pd
        q_u = q[:2] / q[2]
        J[:, i] = (q_u - base_u) / eps
    return J


def measure_box_pixel(image, roi=None):
    """
    Measure box pixel size using the Module A shared foreground detector.

    Returns: (width_px, height_px, contour, bounding_rect)
    """
    if roi is not None:
        x, y, w, h = roi
        image = image[y:y + h, x:x + w]

    try:
        from module_A_segmentation.segmentation import detect_foreground_mask
        mask = detect_foreground_mask(image)
    except Exception:
        # Fallback: saturation/edge detection if Module A is unavailable
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, mask = cv2.threshold(hsv[:, :, 1], 40, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No objects detected in image")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return w, h, largest, (x, y, w, h)


def measure_reference_and_target(reference_img, target_img):
    """Measure both reference and target boxes in pixels."""
    ref_w, ref_h, ref_contour, ref_bbox = measure_box_pixel(reference_img)
    tgt_w, tgt_h, tgt_contour, tgt_bbox = measure_box_pixel(target_img)

    return {
        'reference': {'width_px': ref_w, 'height_px': ref_h, 'bbox': ref_bbox},
        'target': {'width_px': tgt_w, 'height_px': tgt_h, 'bbox': tgt_bbox},
    }


# =============================================================================
# Camera Model 1: Orthographic
# =============================================================================
def orthographic_measurement(ref_px, target_px, ref_real, H_cm, ref_center_cm=None):
    """
    Orthographic: a single global scale, s = ref_real / ref_px, estimated
    from the known reference and applied everywhere. No plane geometry, no
    principal point, no depth variation. Valid when the belt plane is
    essentially parallel to the image and both boxes are near the optical
    axis (negligible perspective foreshortening).
    """
    sx = ref_real[0] / ref_px['width_px']
    sy = ref_real[1] / ref_px['height_px']
    return target_px['width_px'] * sx, target_px['height_px'] * sy


# =============================================================================
# Camera Model 2: Weak Perspective
# =============================================================================
def weak_perspective_measurement(ref_px, target_px, ref_real, H_cm, ref_center_cm=None):
    """
    Weak perspective: models the whole scene as a scaled orthographic
    projection, but lets the scale vary per object by linearising the true
    plane homography AT THE TARGET's own location. This captures that a
    tilted belt foreshortens objects differently at different image
    positions. The object itself is still treated as flat (no variation of
    the scale across its own extent).
    """
    Hinv = np.linalg.inv(H_cm)
    # Target centre in belt-plane cm, via exact homography inversion
    bx, by, bw, bh = target_px['bbox']
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    hc = Hinv @ np.array([cx, cy, 1.0])
    center_cm = hc[:2] / hc[2]

    J = homography_jacobian(H_cm, center_cm)          # px per cm at target
    w_px, h_px = target_px['width_px'], target_px['height_px']
    cm = np.linalg.inv(J) @ np.array([w_px, h_px])
    return float(cm[0]), float(cm[1])


# =============================================================================
# Camera Model 3: Affine
# =============================================================================
def affine_measurement(ref_px, target_px, ref_real, H_cm, ref_center_cm):
    """
    Affine: a single global 2x2 linear map (rotation + non-uniform scale +
    shear) applied to both boxes. This is the plane homography's Jacobian
    anchored at the REFERENCE box: an exact weak-perspective model around
    the reference, extrapolated to the target without the homography's
    projective (depth-varying) correction.
    """
    Hinv = np.linalg.inv(H_cm)
    hc = Hinv @ np.array([ref_center_cm[0], ref_center_cm[1], 1.0])
    center_cm = hc[:2] / hc[2] if ref_center_cm is None else ref_center_cm

    J = homography_jacobian(H_cm, center_cm)
    w_px, h_px = target_px['width_px'], target_px['height_px']
    cm = np.linalg.inv(J) @ np.array([w_px, h_px])
    return float(cm[0]), float(cm[1])


# =============================================================================
# Camera Model 4: Full Perspective
# =============================================================================
def perspective_measurement(ref_px, target_px, ref_real, H_cm, ref_center_cm=None):
    """
    Full perspective: inverts the belt-plane homography EXACTLY at the two
    image corners that delimit each measured dimension. The scale is allowed
    to change continuously across the box, which is the correct treatment of
    perspective foreshortening for objects lying on the belt plane.
    """
    Hinv = np.linalg.inv(H_cm)

    def to_cm(u, v):
        q = Hinv @ np.array([u, v, 1.0])
        return q[:2] / q[2]

    bx, by, bw, bh = target_px['bbox']
    # Corner pixel EDGES (-0.5/+0.5) delimit the physical object boundary.
    # Width taken along the bottom edge, height along the left edge.
    p_l = to_cm(bx - 0.5, by + bh - 0.5)
    p_r = to_cm(bx + bw - 0.5, by + bh - 0.5)
    p_t = to_cm(bx - 0.5, by - 0.5)
    p_b = to_cm(bx - 0.5, by + bh - 0.5)
    width_cm = float(np.hypot(*(p_r - p_l)))
    height_cm = float(np.hypot(*(p_b - p_t)))
    return width_cm, height_cm


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure a box with 4 camera models')
    parser.add_argument(
        '--reference-width-cm', type=float, default=20.0,
        help='Known real-world width of the reference box (cm).')
    parser.add_argument(
        '--reference-height-cm', type=float, default=15.0,
        help='Known real-world height of the reference box (cm).')
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_dir = project_root / 'outputs' / 'calibration'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration (K, dist, belt-plane homography)
    calib_file = output_dir / 'calibration.npz'
    calibration = load_calibration(str(calib_file))
    K = calibration['K']
    dist = calibration['dist']
    H_raw = calibration['H_cm']

    # Data directories
    boxes_dir = project_root / 'data' / 'boxes'
    ref_dir = boxes_dir / 'reference'
    target_dir = boxes_dir / 'target'

    def _images(d):
        return (list(d.glob('*.png')) + list(d.glob('*.jpg')) +
                list(d.glob('*.jpeg'))) if d.exists() else []

    ref_images = _images(ref_dir)
    target_images = _images(target_dir)

    if not ref_images:
        print("ERROR: No reference box image found.")
        print("Please place a reference box image (KNOWN real-world size) at:")
        print(f"  {boxes_dir / 'reference' / 'reference_box.png'}")
        sys.exit(1)
    if not target_images:
        print("ERROR: No target box images found.")
        print(f"Please place a target box image at: {boxes_dir / 'target'}\\target_box.png")
        sys.exit(1)

    # Known reference dimensions (cm)
    ref_real = (args.reference_width_cm, args.reference_height_cm)

    # ------------------------------------------------------------------ #
    # Undistort both images and remap the homography to undistorted space  #
    # ------------------------------------------------------------------ #
    print("Undistorting images and remapping the belt-plane homography...")
    ref_img_raw = cv2.imread(str(ref_images[0]))
    target_img_raw = cv2.imread(str(target_images[0]))
    h1, w1 = ref_img_raw.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w1, h1), 1, (w1, h1))
    ref_img = cv2.undistort(ref_img_raw, K, dist, None, new_K)
    target_img = cv2.undistort(target_img_raw, K, dist, None, new_K)
    H_cm = homography_for_undistorted(H_raw, K, dist, new_K)

    print(f"Reference box: {ref_real[0]} x {ref_real[1]} cm")
    print(f"Reference image: {ref_images[0].name}")
    print(f"Target image: {target_images[0].name}")

    # Measure pixel sizes on undistorted images
    measurements = measure_reference_and_target(ref_img, target_img)
    ref_px = measurements['reference']
    target_px = measurements['target']

    print(f"\nPixel measurements (undistorted):")
    print(f"  Reference: {ref_px['width_px']:.1f} x {ref_px['height_px']:.1f} px")
    print(f"  Target: {target_px['width_px']:.1f} x {target_px['height_px']:.1f} px")

    # Sanity check: what does the homography predict for the reference box?
    Hinv = np.linalg.inv(H_cm)
    rbx, rby, rbw, rbh = ref_px['bbox']
    ref_c = Hinv @ np.array([
        rbx + rbw / 2.0, rby + rbh / 2.0, 1.0
    ])
    ref_center_cm = ref_c[:2] / ref_c[2]
    pred = perspective_measurement(ref_px, ref_px, ref_real, H_cm, ref_center_cm)
    print(f"\n  Calibration cross-check: homography predicts the reference as "
          f"{pred[0]:.1f} x {pred[1]:.1f} cm (given {ref_real[0]} x "
          f"{ref_real[1]} cm)")

    # Apply all 4 camera models
    print("\n" + "=" * 62)
    print("BOX MEASUREMENT RESULTS")
    print("=" * 62)
    print(f"\n{'Camera Model':<22} {'Width':>12} {'Height':>12}")
    print("-" * 50)

    models = [
        ("Orthographic", orthographic_measurement(
            ref_px, target_px, ref_real, H_cm, ref_center_cm)),
        ("Weak Perspective", weak_perspective_measurement(
            ref_px, target_px, ref_real, H_cm, ref_center_cm)),
        ("Affine", affine_measurement(
            ref_px, target_px, ref_real, H_cm, ref_center_cm)),
        ("Full Perspective", perspective_measurement(
            ref_px, target_px, ref_real, H_cm, ref_center_cm)),
    ]

    for name, (w, h) in models:
        print(f"{name:<22} {w:>8.2f} cm {h:>8.2f} cm")

    print("-" * 50)
    print("""
Camera Model Comparison:
------------------------
Orthographic: one constant px->cm scale, taken from the reference box. Ignores
  perspective completely. Fine for small objects near the image centre on a
  fronto-parallel belt; otherwise the least accurate.

Weak Perspective: scale refined at the TARGET position by linearising the belt
  plane homography there. Correctly absorbs the slow change of scale across
  the image but still treats each object as flat.

Affine: a single 2x2 linear map (rotation + non-uniform scale) anchored at the
  reference box. Better than orthographic when the belt is slightly tilted,
  worse than weak perspective if the target sits far from the reference.

Full Perspective: exact inversion of the belt-plane homography at the corners
  of each measured edge. Lets the scale vary across the box, so this is the
  most physically correct model for objects lying on the belt; differences
  among the four are the size of the perspective distortion in your setup.

All measurements reported are on the belt plane (footprint). For a camera
looking straight down on a level belt the four models nearly agree, which is
itself a good sanity check on the calibration.
                  """)

    # Annotated output image for the demo
    overlay = np.hstack([ref_img, target_img])
    cv2.imwrite(str(output_dir / 'boxes_measured.png'), overlay)
    print("Saved: outputs/calibration/boxes_measured.png (undistorted ref + target)")

    print("\nModule B measurement complete.")


if __name__ == '__main__':
    main()