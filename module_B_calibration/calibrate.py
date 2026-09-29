"""
Module B — Camera Calibration
===============================
Performs checkerboard-based camera calibration:
1. Detects checkerboard corners in calibration images
2. Computes intrinsic matrix K and distortion coefficients
3. Undistorts images
4. Recovers extrinsic parameters (R, t, projection matrix P)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_project_root():
    return Path(__file__).parent.parent


def find_checkerboard_corners(images, board_size=(9, 6), square_size_cm=2.5):
    """
    Detect checkerboard corners in a set of images.

    Uses cv2.findChessboardCorners with sub-pixel refinement.

    board_size: (inner_cols, inner_rows) — number of inner corners
    square_size_cm: edge length of one square; object points are built in
        real units (cm) so that tvecs come out in cm directly.
    """
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    valid_images = []

    # 3D object points for a flat checkerboard at z=0, scaled to real units
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:cols, 0:rows
    ].T.reshape(-1, 2) * square_size_cm

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, board_size, flags)

        if ret:
            # Sub-pixel corner refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners_refined)
            valid_images.append(img_path)
            print(f"  Found corners in: {img_path.name}")
        else:
            print(f"  No corners found in: {img_path.name}")

    return obj_points, img_points, valid_images


def calibrate_camera(obj_points, img_points, image_size):
    """
    Perform camera calibration using cv2.calibrateCamera.

    Returns:
        K: Intrinsic matrix (3x3)
        dist: Distortion coefficients
        rvecs: Rotation vectors for each image
        tvecs: Translation vectors for each image
    """
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    if not ret:
        raise RuntimeError("Camera calibration failed!")

    return K, dist, rvecs, tvecs


def compute_reprojection_error(obj_points, img_points, rvecs, tvecs, K, dist):
    """Compute the mean per-point RMS reprojection error, in pixels."""
    total_sq = 0.0
    total_pts = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], K, dist
        )
        # cv2.norm(..., NORM_L2) returns sqrt(sum of squared distances over
        # ALL points); the per-point RMS is sqrt(sum/N). Dividing by the point
        # count instead understates the true per-point error by sqrt(N), which
        # is the classic calibration-metric bug (N = 54 -> ~7.3x off).
        sq = float(cv2.norm(img_points[i] - projected, cv2.NORM_L2) ** 2)
        total_sq += sq
        total_pts += projected.shape[0]
    return float(np.sqrt(total_sq / max(total_pts, 1)))


def rq_decomposition(M):
    """
    Hand-rolled decomposition M = K @ R where K is upper triangular and
    R orthogonal — exactly the K[R|t] split needed for P[:, :3] = K R.

    Key identity: M_{3x3} = K R  <=>  M^{-1} = R^T K^{-1}, where R^T is
    orthogonal and K^{-1} is upper triangular. So an ordinary QR of the
    inverse M^{-1} = Q @ R_gives Q = R^T and R_gives = K^{-1} directly:

        K = inv(R_gives)   (upper triangular)
        R = Q^T            (orthogonal)

    This is verified in-code against the returned product.
    """
    Minv = np.linalg.inv(M)
    Q, R = np.linalg.qr(Minv)          # Minv = Q @ R (R upper triangular)
    K = np.linalg.inv(R)               # K upper triangular, M = K @ Q^T
    Rmat = Q.T

    # Canonical intrinsic form: the diagonal of K must be positive (fx, fy > 0
    # and K[2,2] = 1). Flip the sign of the i-th COLUMN of K and the i-th ROW
    # of Rmat together — this pairs S @ S = I, so M = K @ Rmat is unchanged —
    # and, because det(M) > 0 for a real camera (det(K) > 0, det(R) = 1),
    # forcing diag(K) > 0 automatically makes R a proper rotation (det +1).
    signs = np.sign(np.diag(K))
    if np.any(signs < 0):
        S = np.diag(signs)
        K = K @ S
        Rmat = S @ Rmat
    # Verify the product reproduces the input
    residual = float(np.abs(K @ Rmat - M).max())
    if residual > 1e-6:
        raise ValueError(f"RQ decomposition failed: max residual {residual:.2e}")
    return K, Rmat, residual


def decompose_projection_matrix_manual(P):
    """
    Manual decomposition of the 3x4 projection matrix P = K[R|t].

    Because the image point is homogeneous, the overall scale of P is free,
    so we note:
      1. M = P[:, :3] = K R  ->  K, R via rq_decomposition(M)
      2. t = P[:, 3]          (but must be rescaled with K)
      3. Normalize K so K[2,2] = 1 and rescale t by the same factor.

    Returns (K, R, t, residual) recovered purely from P. No OpenCV call is
    used — the intrinsics and extrinsics are pulled out of a bare P by hand.
    """
    P = np.asarray(P, dtype=np.float64)
    if abs(P[2, 3]) < 1e-12:
        raise ValueError("P[2,3] == 0; P is affine, not a perspective projection")

    M = P[:, :3]
    p4 = P[:, 3].reshape(3, 1)          # homogeneous image of the world origin
    K, R, residual = rq_decomposition(M)

    # Translation from P = K[R|t]: solve K t = p4.
    t = np.linalg.solve(K, p4)

    # Enforce the canonical intrinsic form K[2,2] = 1. This rescales the left
    # 3x3 block by s, i.e. P -> sP, which is the same camera (homogeneous).
    # Because t solves K t = p4, it must NOT be rescaled when K is.
    s = 1.0 / K[2, 2]
    K = K * s

    # Final verification that the decomposition reproduces P. Because the
    # image is homogeneous, P and any s*P describe the same camera, so
    # compare P_rebuilt against the *rescaled* input.
    P_rebuilt = K @ np.hstack([R, t])
    P_scaled = s * P
    residual_final = float(np.abs(P_rebuilt - P_scaled).max())
    if residual_final > 1e-5:
        raise ValueError(f"Decomposition failed to reproduce P (max err {residual_final:.2e})")

    return K, R, t, residual_final


def undistort_image(img, K, dist):
    """Remove lens distortion from an image."""
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, dist, None, new_K)
    return undistorted, new_K, roi


def draw_checkerboard_corners(img, board_size, corners_found):
    """Visualize detected corners on the checkerboard image."""
    vis = img.copy()
    cv2.drawChessboardCorners(vis, board_size, corners_found, True)
    return vis


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Checkerboard camera calibration')
    parser.add_argument(
        '--square-size-cm', type=float, default=2.5,
        help='Real-world edge length of one checkerboard square in cm'
             ' (needed to convert the calibration to physical units).')
    parser.add_argument(
        '--checkerboard', type=str, default='9x6',
        help='Checkerboard inner-corner count, e.g. 9x6 (cols x rows).')
    args = parser.parse_args()

    try:
        board_cols, board_rows = map(int, args.checkerboard.split('x'))
    except ValueError:
        parser.error("--checkerboard must look like 9x6")
    board_size = (board_cols, board_rows)

    square_size_cm = args.square_size_cm

    project_root = get_project_root()
    calib_dir = project_root / 'data' / 'calibration'
    output_dir = project_root / 'outputs' / 'calibration'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    calibration_images = []
    for ext in image_extensions:
        calibration_images.extend(calib_dir.glob(ext))

    if not calibration_images:
        print("ERROR: No checkerboard images found.")
        print("Please place 10–15 checkerboard photographs in:")
        print(f"  {calib_dir}")
        print(f"\nExpected checkerboard: {board_size[0]}x{board_size[1]} inner corners")
        print(f"Assumed square size: {square_size_cm} cm (pass --square-size-cm to change)")
        print("Print a standard checkerboard and photograph it from")
        print("different angles and distances.")
        sys.exit(1)

    print(f"Found {len(calibration_images)} calibration images")

    # Detect corners
    print("\nStep 1: Detecting checkerboard corners...")
    obj_points, img_points, valid_images = find_checkerboard_corners(
        calibration_images, board_size
    )

    if len(valid_images) < 3:
        print(f"\nERROR: Only {len(valid_images)} valid images found.")
        print("Need at least 3 images with detected corners.")
        print("Tips:")
        print("  - Use a well-printed checkerboard (no curling)")
        print("  - Ensure good, even lighting")
        print("  - Capture from varied angles")
        sys.exit(1)

    print(f"\nSuccessfully detected corners in {len(valid_images)} images")

    # Get image size from first valid image
    first_img = cv2.imread(str(valid_images[0]))
    image_size = (first_img.shape[1], first_img.shape[0])

    # Calibrate
    print("\nStep 2: Calibrating camera...")
    K, dist, rvecs, tvecs = calibrate_camera(obj_points, img_points, image_size)

    # Reprojection error
    repro_error = compute_reprojection_error(
        obj_points, img_points, rvecs, tvecs, K, dist
    )
    print(f"\nCalibration Results:")
    print(f"  Reprojection error: {repro_error:.4f} pixels")

    # Intrinsic matrix
    print(f"\n  Intrinsic Matrix K:")
    print(f"    fx={K[0,0]:.4f}  fy={K[1,1]:.4f}  cx={K[0,2]:.4f}  cy={K[1,2]:.4f}")

    # =========================================================================
    # Step 3: Undistortion comparison
    # =========================================================================
    print("\nStep 3: Creating undistortion comparison...")
    test_img = cv2.imread(str(valid_images[0]))
    undistorted, new_K, roi = undistort_image(test_img, K, dist)

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (Distorted)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Undistorted', fontsize=12)
    axes[1].axis('off')

    plt.suptitle('Camera Distortion Correction', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'undistortion_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: undistortion_comparison.png")

    # Save undistorted image
    cv2.imwrite(str(output_dir / 'undistorted.png'), undistorted)

    print(f"\n  Distortion coefficients:")
    dist_labels = ['k1', 'k2', 'p1', 'p2', 'k3']
    for i, (label, val) in enumerate(zip(dist_labels, dist[0])):
        print(f"    {label} = {val:.6f}")

    # =========================================================================
    # Step 4: Extrinsic parameters for one image (belt-plane image)
    # =========================================================================
    print("\nStep 4: Extrinsic parameters...")
    # Use the last valid image
    last_idx = len(valid_images) - 1
    R_vec = rvecs[last_idx]
    t_vec = tvecs[last_idx].reshape(3, 1)

    # Convert rotation vector to matrix
    R_mat, _ = cv2.Rodrigues(R_vec)

    print(f"\n  Extrinsic parameters for image: {valid_images[last_idx].name}")
    print(f"  Rotation vector: {R_vec.flatten()}")
    print(f"  Translation vector (cm): {t_vec.flatten()}")

    print("\n  Camera pose relative to the checkerboard/belt plane")
    print("  (this is the 'rotation + translation of the camera relative to")
    print("  the belt' the assignment asks for — the world-to-camera extrinsics")
    print("  describe the plane in camera coordinates, so we invert them):")
    R_board_to_camera = R_mat.T          # board axes expressed in camera frame
    t_board_to_camera = -R_board_to_camera @ t_vec
    print(f"    R (camera -> board plane):")
    for row in R_board_to_camera:
        print(f"      [{row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f}]")
    print(f"    t (camera -> board origin, cm): {t_board_to_camera.flatten()}")

    # Construct projection matrix P = K[R|t]
    R_t = np.hstack([R_mat, t_vec])
    P = K @ R_t

    print(f"\n  Projection Matrix P = K[R|t]:")
    for row in P:
        print(f"    [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}]")

    # -------------------------------------------------------------------------
    # Step 4b: MANUAL decomposition of P = K[R|t] (assignment item)
    # We pull K, R, t back out of the bare projection matrix by hand
    # (QR of P[:, :3]'s inverse, then sign/scale normalization) instead of
    # calling cv2.decomposeProjectionMatrix, to demonstrate the theory.
    # -------------------------------------------------------------------------
    K_manual, R_manual, t_manual, _ = decompose_projection_matrix_manual(P)
    t_manual = t_manual.flatten()

    print("\n  Manual decomposition of P = K[R|t] (recovered only from P):")
    print(f"    fx={K_manual[0,0]:.4f}  fy={K_manual[1,1]:.4f}  "
          f"cx={K_manual[0,2]:.4f}  cy={K_manual[1,2]:.4f}")
    print(f"    t (cm) = {np.round(t_manual, 4)}")
    print(f"    det(R) = {np.linalg.det(R_manual):.6f}")

    # Compare with the values from cv2.calibrateCamera / Rodrigues
    print("\n  Comparison with the direct calibration values:")
    print(f"    fx: manual {K_manual[0,0]:.4f}  vs cv2 {K[0,0]:.4f}")
    print(f"    fy: manual {K_manual[1,1]:.4f}  vs cv2 {K[1,1]:.4f}")
    print(f"    cx: manual {K_manual[0,2]:.4f}  vs cv2 {K[0,2]:.4f}")
    print(f"    cy: manual {K_manual[1,2]:.4f}  vs cv2 {K[1,2]:.4f}")
    r_recovered, _ = cv2.Rodrigues(R_manual)
    print(f"    Rvec: manual {np.round(r_recovered.flatten(), 6)}  "
          f"vs cv2 {np.round(R_vec.flatten(), 6)}")

    # -------------------------------------------------------------------------
    # Homography of the belt/checkerboard plane (used by Module C for the
    # pixel<->cm scale and by measure_box.py). H maps board-plane points
    # (in SQUARE units, origin at the first inner corner) to pixels.
    # -------------------------------------------------------------------------
    r1 = R_mat[:, 0]
    r2 = R_mat[:, 1]
    H_board_to_image = K @ np.hstack([r1.reshape(3, 1), r2.reshape(3, 1), t_vec])

    # Save calibration (now with the extras the other modules need)
    calib_file = output_dir / 'calibration.npz'
    np.savez(
        calib_file,
        K=K,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=image_size,
        reprojection_error=repro_error,
        board_size=board_size,
        square_size_cm=square_size_cm,
        H_board_to_image=H_board_to_image,
        R_board_to_camera=R_board_to_camera,
        t_board_to_camera=t_board_to_camera,
        K_manual=K_manual,
        R_manual=R_manual,
        t_manual=t_manual,
    )
    print(f"\nCalibration saved to: {calib_file}")
    print("  (includes H_board_to_image + square_size_cm + camera-to-board pose)")

    print("\nCamera calibration module complete.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
