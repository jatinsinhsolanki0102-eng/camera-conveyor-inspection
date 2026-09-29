# Camera-Based Conveyor Inspection & Tracking System

A classical computer vision system for inspecting and tracking boxes on a conveyor belt. Uses no deep learning — only classical CV algorithms (optical flow, Kalman filtering, PCA, segmentation).

## Overview

The system processes a video of boxes moving on a conveyor belt and produces:
- Segmentation masks (4 methods compared)
- Real-world size measurements (4 camera models)
- Belt speed estimation (cm/s)
- Box tracking with Kalman filter (handles occlusion)
- Box type recognition (alignment + PCA + Hu moments)

## Setup

```bash
pip install -r requirements.txt
```

## Web UI (recommended for running with your own video)

Launch the interactive page and upload your video from the browser:

```bash
python webui.py
# then open http://127.0.0.1:5000
```

The page accepts:
- **Conveyor video** (required) — a steady clip of boxes on a belt.
- **Checkerboard photos** (optional) — 10–15 photos of a 9x6-inner-corner
  checkerboard on the belt plane; if given, Module B runs first and the
  report then shows real-world sizes and belt speed in cm/s.
- **Box crops per class** (optional) — ~10 tight crop images each for
  `box_type_A/B/C`; if skipped, synthetic glyph templates are used.
- Two modes:
  - *Pipeline report (fast)* — runs the integrated pipeline
    (frame → A segment → D Kalman → B cm → C speed → E type) and prints the
    per-box report with an overlay figure.
  - *Full demo (modules A–E + pipeline)* — also runs the standalone module
    scripts to produce the segmentation comparison grid, optical-flow and
    RANSAC visualizations, Kalman tracking overlay video, and the
    recognition accuracy table.

Every run starts from a **clean `outputs/` folder**, so the report and
figures always belong to the video you just uploaded; previous runs are
archived automatically in `outputs_archive/`.

## Data Setup

Place your files in these directories:

```
data/
  video/
    conveyor.mp4          # Your conveyor belt video (phone camera is fine)
  calibration/
    checker_01.jpg        # 10-15 checkerboard photos
    checker_02.jpg        # Print a 9x6 inner corner checkerboard
    ...
  boxes/
    box_type_A/
      crop_001.png        # ~10 cropped box images per type
      crop_002.png
      ...
    box_type_B/
      crop_001.png
      ...
    box_type_C/
      crop_001.png
      ...
```

## Running Individual Modules

```bash
# Module A — Segmentation (4 methods + comparison grid on 5 frames)
python module_A_segmentation/segmentation.py

# Module B — Camera Calibration
#   --square-size-cm: edge length of one checkerboard square (default 2.5)
python module_B_calibration/calibrate.py --square-size-cm 2.5
python module_B_calibration/measure_box.py \
    --reference-width-cm 20 --reference-height-cm 15

# Module C — Optical Flow (LK from scratch + belt speed)
python module_C_optical_flow/optical_flow.py

# Module D — Kalman Filter Tracking (uses Module A detections on the video)
#   --dropout-start/--dropout-len: inject occlusion to demonstrate recovery
python module_D_tracking/kalman_tracker.py --dropout-len 5

# Module E — Object Recognition (alignment + PCA + Hu + rotation invariance)
python module_E_recognition/recognize.py
```

## Running Full Pipeline

```bash
python pipeline.py
# or with custom video / a shorter run / occlusion injection:
python pipeline.py --video path/to/video.mp4 --frames 30 --dropout-len 5
```

## Module Descriptions

**Module A — Segmentation:** Compares Active Contours, Watershed, Graph-Based, and Mean Shift segmentation on 5 representative frames. Produces a comparison grid and per-method masks. `detect_foreground_mask` is the shared detector reused by Modules C, D and the pipeline.

**Module B — Camera Calibration:** Checkerboard-based calibration (K, distortion coefficients, RMS reprojection error), undistortion before/after, the camera's rotation + translation relative to the belt/checkerboard plane, a MANUAL decomposition of the projection matrix P = K[R|t] (QR-of-inverse RQ decomposition, verified against cv2), and the belt-plane homography stored for the other modules. `measure_box.py` measures an unknown box with 4 camera models (Orthographic, Weak Perspective, Affine, Full Perspective) on undistorted images, where the Full Perspective result comes from exact inversion of the belt-plane homography.

**Module C — Optical Flow:** Implements Lucas-Kanade from scratch (Ix*u + Iy*v + It = 0 with Shi-Tomasi conditioning), compares against OpenCV's pyramidal LK, fits an affine flow model to one tracked box with RANSAC, and estimates belt speed in cm/s using Module B's calibrated scale (falls back to an assumed scale with a loud warning).

**Module D — Kalman Filter:** Constant-velocity Kalman filter from scratch. Measurements come from Module A segmentation (box centroid) on the real video; it prints predict → measure → update for the first frames, extrapolates through an injected 3-5 frame measurement dropout while trace(P) grows, and plots raw vs filtered trajectories plus an overlay video.

**Module E — Object Recognition:** Alignment-based (ORB + RANSAC homography), PCA/Eigenspace ("Eigenboxes", top-3 eigenvectors), and Hu moments (binary-silhouette invariants). Prints a 3-method accuracy table, confusion matrices, and a rotation-invariance comparison showing that Hu stays accurate at any angle while PCA degrades.

## Outputs

All results are saved in `outputs/`:
- `outputs/segmentation/` — Comparison grids and masks
- `outputs/calibration/` — Calibration file, undistortion, box measurements
- `outputs/optical_flow/` — Flow arrows, field, RANSAC/affine, belt speed
- `outputs/tracking/` — Kalman trajectory + uncertainty plots, overlay video
- `outputs/recognition/` — Eigenvectors, accuracy plots, confusion matrices
- `outputs/inspection_results.txt` — Full-pipeline per-box report

## Demo Checklist (10-minute recording)

1. **0-1 min:** Project overview, show video
2. **1-2 min:** Module A segmentation comparison grid
3. **2-3 min:** Camera calibration + undistortion
4. **3-5 min:** Optical flow — hand-built LK, OpenCV comparison, RANSAC, belt speed
5. **5-7 min:** Kalman tracking — prediction, correction, dropout handling
6. **7-9 min:** Recognition — alignment, PCA eigenvectors, accuracy table
7. **9-10 min:** Run full pipeline, show final results

## Requirements

- Python 3.10+
- OpenCV, NumPy, scikit-image, scikit-learn, Matplotlib
- No deep learning frameworks required
