"""
Pipeline — Integrated Conveyor Inspection System
=================================================
Chains the five real assignment modules into one end-to-end inspection run:

    video
      -> Module B  : pixel<->cm scale from the calibrated belt plane (if any)
      -> Module A  : foreground segmentation per frame (box detections)
      -> Module D  : Kalman filter (from scratch) tracked per box
      -> Module C  : optical-flow belt speed from tracked features
      -> Module E  : Hu-moment box-type recognition of each box's crop
      -> inspection report (outputs/inspection_results.txt)

Run:
    python pipeline.py [--video PATH] [--frames N]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent
for _p in (str(PROJECT_ROOT),
           str(PROJECT_ROOT / 'module_A_segmentation'),
           str(PROJECT_ROOT / 'module_B_calibration'),
           str(PROJECT_ROOT / 'module_C_optical_flow'),
           str(PROJECT_ROOT / 'module_D_tracking'),
           str(PROJECT_ROOT / 'module_E_recognition')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def find_video_file(video_arg, data_dir):
    """Resolve the video path (CLI arg or the first file in data/video)."""
    if video_arg:
        p = Path(video_arg)
        if p.exists():
            return p
        print(f"WARNING: video {video_arg} not found; falling back to "
              f"data/video/.")
    for ext in ['*.mp4', '*.avi', '*.mov']:
        try:
            found = sorted(data_dir.glob(ext))
        except OSError:
            continue
        if found:
            return found[0]
    return data_dir / 'conveyor.mp4'


def main():
    parser = argparse.ArgumentParser(description='Conveyor Inspection Pipeline')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to process (default: min(total, 50)).')
    parser.add_argument('--start-frame', type=int, default=None,
                        help='Skip to this absolute frame index before processing.')
    parser.add_argument('--dropout-len', type=int, default=0,
                        help='Inject an artificial measurement dropout of this '
                             'many frames to demonstrate occlusion handling.')
    args = parser.parse_args()

    video_path = find_video_file(args.video, PROJECT_ROOT / 'data' / 'video')
    output_dir = PROJECT_ROOT / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Module B: calibration / pixel<->cm scale                            #
    # ------------------------------------------------------------------ #
    calibration = None
    calib_file = output_dir / 'calibration' / 'calibration.npz'
    if calib_file.exists():
        try:
            from module_B_calibration.measure_box import load_calibration
            calibration = load_calibration(str(calib_file))
            print("  Calibration loaded: belt-plane scale from Module B.")
        except Exception as exc:
            print(f"  WARNING: calibration present but unusable ({exc}); "
                  f"scale assumed.")
    else:
        print("  Calibration not found — assuming 0.05 cm/px (run Module B "
              "for a real scale).")

    def px_to_cm_factor():
        """cm per px at image centre, or the assumed constant."""
        if calibration is not None and 'H_cm' in calibration:
            H = calibration['H_cm']
            # Local scale of the homography near the optical axis
            scale = np.linalg.norm(H[:2, 0]) + np.linalg.norm(H[:2, 1])
            return 2.0 / scale
        return 0.05

    # ------------------------------------------------------------------ #
    # Open video                                                          #
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = args.frames if args.frames is not None else min(total, 50)
    start_frame = args.start_frame if args.start_frame is not None else 0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(start_frame, max(0, total - 1)))

    print("=" * 60)
    print("CONVEYOR INSPECTION PIPELINE")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print(f"  Resolution: {w_vid}x{h_vid}, FPS {fps:.1f}, total {total}")
    print(f"  Processing: {n_frames} frames from index {start_frame}")

    # ------------------------------------------------------------------ #
    # Import the real module functions                                    #
    # ------------------------------------------------------------------ #
    from module_A_segmentation.segmentation import detect_foreground_mask
    from module_D_tracking.kalman_tracker import KalmanFilterScratch
    from module_C_optical_flow.optical_flow import (
        opencv_sparse_flow, load_pixel_to_cm,
        estimate_belt_speed as _speed_est)
    from module_E_recognition.recognize import (
        compute_hu_moments, load_box_crops)

    # Module E templates: all crops per class (mean Hu distance decision)
    box_images, box_labels = load_box_crops(PROJECT_ROOT / 'data' / 'boxes')
    templates_by_label = {}
    for lbl in set(box_labels):
        templates_by_label[lbl] = [im for im, L in zip(box_images, box_labels)
                                   if L == lbl]

    # Module C: pixel<->cm scale (uses Module B calibration if available)
    pixel_to_cm, scale_source = load_pixel_to_cm(PROJECT_ROOT)
    if pixel_to_cm is None:
        print(f"  Belt scale source: {scale_source}")
        print("  -> ASSUMED 0.05 cm/px for the report.")
        pixel_to_cm_const = 0.05
    else:
        print(f"  Belt scale source: {scale_source}")
        print(f"  -> {pixel_to_cm:.4f} cm/px.")

    # ------------------------------------------------------------------ #
    # Main loop: detect -> associate -> Kalman -> flow speed              #
    # ------------------------------------------------------------------ #
    tracks = {}            # track_id -> dict(state dict, bbox history, ...)
    tid = 0
    prev_gray = None
    prev_features = None
    flow_speeds_px = []    # px/frame from tracked feature displacement
    dropout = set()
    if args.dropout_len and args.dropout_len > 0:
        start = start_frame + n_frames // 2 - args.dropout_len // 2
        dropout = set(range(start, start + min(args.dropout_len, 6)))

    kernel = np.ones((5, 5), np.uint8)
    bg = None

    for rel in range(n_frames):
        fi = start_frame + rel
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Module C: belt speed through tracked feature displacement ---
        if prev_gray is not None:
            good_old, good_new, _ = opencv_sparse_flow(prev_gray, gray)
            if len(good_old):
                dx = good_new[:, 0] - good_old[:, 0]
                dy = good_new[:, 1] - good_old[:, 1]
                flow_speeds_px.append((float(np.median(dx)),
                                       float(np.median(dy))))
        prev_gray = gray.copy()

        # ---- Module A: segmentation (fast shared foreground model) ------
        mask = detect_foreground_mask(frame, bg if bg is not None else None)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                kernel, iterations=2)
        # Refresh the empty-belt background model (slow EMA so a box that
        # pauses briefly stays foreground; the belt itself is stationary).
        if bg is None:
            bg = gray.astype(np.float32)
        else:
            bg = bg * 0.90 + gray.astype(np.float32) * 0.10
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for c in contours:
            if cv2.contourArea(c) < w_vid * h_vid * 0.001:
                continue
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M['m00'] < 1e-3:
                continue
            blobs.append({'bbox': (x, y, w, h),
                          'centroid': (M['m10'] / M['m00'],
                                       M['m01'] / M['m00']),
                          'area': cv2.contourArea(c)})

        # ---- Module D: associate blobs to tracks, Kalman predict/update --
        assigned = set()
        for blob in blobs:
            cx, cy = blob['centroid']
            best_t, best_d = None, 60.0
            for t_id, t in tracks.items():
                tx, ty = t['kf'].get_position()
                d = np.hypot(cx - tx, cy - ty)
                if d < best_d:
                    best_d = d
                    best_t = t_id
            if best_t is not None:
                kf = tracks[best_t]['kf']
                kf.predict()
                if fi not in dropout:
                    kf.update((cx, cy))
                tracks[best_t]['last'] = fi
                tracks[best_t]['bbox'] = blob['bbox']
                tracks[best_t]['centroids'].append(blob['centroid'])
                assigned.add(best_t)
            else:
                kf = KalmanFilterScratch(dt=1.0, process_noise=1.0,
                                         measurement_noise=25.0)
                kf.x[0, 0], kf.x[1, 0] = cx, cy
                kf.predict()
                tracks[tid] = {'kf': kf, 'first': fi, 'last': fi,
                               'bbox': blob['bbox'],
                               'centroids': [(cx, cy)]}
                assigned.add(tid)
                tid += 1

        # ---- predict-only step for tracks that missed this frame --------
        for t_id, t in tracks.items():
            if t_id not in assigned and fi > t['last'] + 0:
                t['kf'].predict()

    cap.release()

    # Base image for the per-run overlay (last processed frame)
    base = frame.copy() if frame is not None else None

    # ------------------------------------------------------------------ #
    # Module C: belt speed (px/frame and cm/s).                           #
    # The whole-frame median of sparse flow is biased by the static       #
    # background (most features sit on it). Instead this uses the ON-BOX   #
    # motion measured by the Module A->D tracker: the median per-frame     #
    # centroid displacement across all tracks, keeping only deltas in a    #
    # plausible range (a stationary blob gives ~0; a blob that vanished    #
    # and reappeared elsewhere gives a jump > 40 px/frame).                #
    # ------------------------------------------------------------------ #
    deltas_all = []
    for t in tracks.values():
        for a, b in zip(t['centroids'], t['centroids'][1:]):
            deltas_all.append((b[0] - a[0], b[1] - a[1]))
    deltas_all = np.array(deltas_all) if deltas_all else np.empty((0, 2))

    dx_med = dy_med = 0.0
    if len(deltas_all) >= 3:
        mag = np.linalg.norm(deltas_all, axis=1)
        keep = deltas_all[(mag >= 0.3) & (mag <= 40.0)]
        if len(keep) >= 3:
            d = np.median(keep, axis=0)
            dx_med, dy_med = float(d[0]), float(d[1])
    if abs(dx_med) + abs(dy_med) < 1e-6 and flow_speeds_px:
        dx_med, dy_med = np.median(np.array(flow_speeds_px), axis=0)

    if abs(dx_med) + abs(dy_med) > 1e-6:
        affine_params = (0.0, 0.0, float(dx_med), 0.0, 0.0, float(dy_med))
        speed_cm_s, disp_px = _speed_est(
            affine_params, (w_vid / 2, h_vid / 2), fps,
            pixel_to_cm if pixel_to_cm is not None else 0.05,
            scale_source if pixel_to_cm is not None else
            "ASSUMED 0.05 cm/px (no calibration)")
        avg_px_frame = disp_px
    else:
        avg_px_frame = 0.0
        speed_cm_s = 0.0

    # ---------------------------------------------------------------- #
    # Module E + sizes: per-track inspection report                     #
    # ---------------------------------------------------------------- #
    scale = px_to_cm_factor()

    # Stage 1: candidate boxes = tracks that lived a meaningful span.
    # A track of only a few frames is normally a mask flicker, not a box.
    cands = []
    for i, (t_id, t) in enumerate(tracks.items()):
        span = t['last'] - t['first'] + 1
        if span < 5:
            continue
        cands.append({'id': i + 1, 't': t, 'bbox': t['bbox'], 'span': span})

    # Stage 2: merge fragments. Segmentation often splits one object into
    # two touching blobs (e.g. a thin edge highlight); two overlapping
    # boxes are treated as the same physical box - keep the larger one.
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        xx = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        yy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = xx * yy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    accepted = []
    for c in sorted(cands, key=lambda c: c['bbox'][2] * c['bbox'][3],
                    reverse=True):
        if any(_iou(c['bbox'], a['bbox']) >= 0.15 for a in accepted):
            continue
        accepted.append(c)
    accepted.sort(key=lambda c: c['id'])
    cands = accepted

    print("\n" + "=" * 60)
    print("BOX INSPECTION RESULTS")
    print("=" * 60)

    results = []
    for i, c in enumerate(cands):
        t = c['t']
        span = c['span']
        x, y, w, h = t['bbox']
        w_cm, h_cm = w * scale, h * scale

        # Module E: classify the box crop with Hu moments
        frame = None
        cap2 = cv2.VideoCapture(str(video_path))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, t['first'])
        ok, frame = cap2.read()
        cap2.release()
        btype = "Unknown"
        if ok and templates_by_label:
            crop = frame[y:y + h, x:x + w]
            if crop.size:
                hu_q = compute_hu_moments(crop)
                dists = {lbl: np.mean(
                    [np.abs(hu_q - compute_hu_moments(tm)).sum()
                     for tm in tpls]) for lbl, tpls in
                    templates_by_label.items()}
                btype = min(dists, key=dists.get)

        if base is not None:
            cv2.rectangle(base, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(base, f"Box{i + 1:02d} {btype} ({w_cm:.1f}x{h_cm:.1f}cm)",
                        (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        print(f"\n  Box ID: {i + 1:02d}")
        print(f"  Type: {btype}")
        print(f"  Real-world width:  {w_cm:.2f} cm")
        print(f"  Real-world height: {h_cm:.2f} cm")
        print(f"  Estimated belt speed: {speed_cm_s:.2f} cm/s "
              f"(~{avg_px_frame:.2f} px/frame)")
        print(f"  Frames tracked: {t['first'] + 1} to {t['last'] + 1} "
              f"({span} frames)")

        results.append({'id': i + 1, 'type': btype, 'width_cm': w_cm,
                        'height_cm': h_cm, 'belt_speed': speed_cm_s,
                        'frames': span})

    print("\n" + "=" * 60)
    print(f"Pipeline complete. {len(results)} boxes inspected.")
    print("=" * 60)

    # Save report
    report = output_dir / 'inspection_results.txt'
    with open(report, 'w') as f:
        f.write("CONVEYOR INSPECTION RESULTS\n")
        f.write("=" * 40 + "\n\n")
        for r in results:
            f.write(f"Box ID: {r['id']:02d}\n")
            f.write(f"Type: {r['type']}\n")
            f.write(f"Width: {r['width_cm']:.2f} cm\n")
            f.write(f"Height: {r['height_cm']:.2f} cm\n")
            f.write(f"Belt Speed: {r['belt_speed']:.2f} cm/s\n")
            f.write(f"Frames: {r['frames']}\n")
            f.write("-" * 30 + "\n")

    # Save a per-run overlay figure (detected boxes + type + size in cm)
    if base is not None:
        overlay_dir = output_dir / 'pipeline'
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / 'inspection_overlay.png'
        cv2.imwrite(str(overlay_path), base)
        print(f"Overlay saved to: {overlay_path}")

    print(f"\nResults saved to: {report}")


if __name__ == '__main__':
    main()