"""
Module A — Segmentation
========================
Implements 4 classical segmentation methods on conveyor belt frames:
1. Active Contours (Snakes)
2. Watershed
3. Graph-Based (Felzenszwalb)
4. Mean Shift Clustering

Produces a comparison grid for 5 representative frames.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from skimage.segmentation import active_contour, felzenszwalb, watershed
from skimage.filters import gaussian
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from scipy import ndimage
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_frame(video_path, frame_idx):
    """Load a specific frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def get_representative_frames(video_path, num_frames=5):
    """Select representative frames spread across the video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        step = total_frames // (num_frames + 1)
        indices = [int(step * (i + 1)) for i in range(num_frames)]

    frames = []
    for idx in indices:
        frame = load_frame(video_path, idx)
        frames.append((idx, frame))
    return frames


# =============================================================================
# SHARED PREPROCESSING: foreground (box) mask
# =============================================================================
def detect_foreground_mask(frame, background=None):
    """
    Extract a binary foreground mask of the boxes, 255 = box, 0 = belt/background.

    Boxes are normally lighter than the belt, so the primary attempt is an
    adaptive threshold on the locally-brighter class (C=9, roughly 2 sigma of
    sensor noise for 8-bit data). Two fallbacks follow, because a single fixed
    threshold silently returns an all-foreground or all-background image on
    scenes that do not match its assumption -- and an empty mask makes every
    downstream stage (distance transform, watershed) return nothing at all
    while still looking like a successful run:

      1. adaptive threshold, locally-brighter class
      2. Otsu global threshold, polarity chosen by which side looks like
         a plausible amount of foreground
      3. intensity percentile cut

    If a ``background`` grayscale image is supplied (a running or static
    background model of the empty belt), a *motion* candidate is added as well:
    the difference between the current frame and the background, thresholded
    with Otsu and an adaptive threshold. This finds boxes that move on the
    belt even when they are darker than it or the light is dim/uniform, which
    the brightness-only candidates miss.

    A result is rejected if it is essentially empty (<1% of pixels) or
    essentially everything (>70%), since neither can describe boxes on a belt.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def clean(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def n_regions(mask):
        return cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1

    lo, hi = np.percentile(blurred, [5, 95])
    span = float(hi - lo)

    raw = [
        # Adaptive threshold on the locally-brighter class.
        cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 9
        ),
    ]
    # Otsu, both polarities. NOTE: cv2.threshold returns (retval, dst) --
    # the actual mask is the SECOND value.
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw.append(otsu.astype(np.uint8))
    raw.append(cv2.bitwise_not(otsu))
    # Percentile cuts at two aggressiveness levels.
    if span > 0:
        for frac in (0.25, 0.50):
            raw.append(np.where(
                blurred > lo + frac * span, 255, 0).astype(np.uint8))

    # Motion candidates: the difference from the empty-belt background.
    if background is not None:
        bg8 = (np.clip(background, 0, 255)).astype(np.uint8)
        if bg8.shape != blurred.shape:
            bg8 = cv2.resize(bg8, (blurred.shape[1], blurred.shape[0]))
        diff = cv2.absdiff(blurred, bg8)
        _, md_otsu = cv2.threshold(diff, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw.append(md_otsu.astype(np.uint8))
        raw.append(cv2.adaptiveThreshold(
            diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5))

    cleaned = [clean(m) for m in raw]

    # Boxes occupy a minority of a conveyor frame, so accept candidates whose
    # coverage is small enough to be believable. Among those, prefer the one
    # that resolved the most separate objects: boxes are physically distinct,
    # so a candidate that lumps the belt and the boxes into one blob is wrong
    # even when its coverage looks reasonable.
    plausible = [m for m in cleaned if 0.005 <= float((m > 0).mean()) <= 0.60]
    if plausible:
        return max(plausible, key=lambda m: (n_regions(m), -float((m > 0).mean())))

    # Nothing believable: under-segment rather than claim the whole frame is
    # box. Returning the "everything" candidate here is what made an empty or
    # inverted mask propagate silently through the distance transform and
    # watershed, producing an empty result that still looked like a success.
    return min(cleaned, key=lambda m: float((m > 0).mean()))


# =============================================================================
# METHOD 1: Active Contours (Snakes)
# =============================================================================
def segment_active_contour(frame):
    """
    Active Contour segmentation.

    The snake minimizes the energy functional:
      E = integral( alpha * |dC/ds|^2 + beta * |d^2C/ds|^2 + E_ext(C) ) ds

    where:
      - alpha controls continuity (tension)
      - beta controls smoothness (rigidity)
      - E_ext is the image energy (typically negative gradient of smoothed image)

    The contour iteratively moves toward edges by computing:
      C_new = C_old + gamma * (alpha * C'' + beta * C'''' + grad(E_ext))

    We initialize the snake as a circle near the center of the frame
    and let it converge toward the strongest box boundary.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian smoothing to create the external energy field
    smoothed = gaussian(gray, sigma=3, preserve_range=False)

    h, w = gray.shape

    # Initialize snake as an ellipse near the center
    center_x, center_y = w // 2, h // 2
    radius_x, radius_y = w // 4, h // 4
    s = np.linspace(0, 2 * np.pi, 400)
    init_x = center_x + radius_x * np.cos(s)
    init_y = center_y + radius_y * np.sin(s)
    init = np.array([init_x, init_y]).T

    # Run active contour
    snake = active_contour(
        smoothed,
        init,
        alpha=0.015,   # Higher = more rigid
        beta=0.1,       # Higher = smoother
        gamma=0.001,
        # skimage renamed max_iterations -> max_num_iter in 0.20; the old name is gone.
        max_num_iter=2500,
        convergence=1e-6
    )

    # Draw result
    result = frame.copy()
    for i in range(len(snake) - 1):
        pt1 = (int(snake[i, 1]), int(snake[i, 0]))
        pt2 = (int(snake[i + 1, 1]), int(snake[i + 1, 0]))
        cv2.line(result, pt1, pt2, (0, 255, 0), 2)

    # Create mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    pts = snake.astype(np.int32)
    cv2.fillPoly(mask, [pts[:, ::-1]], 255)

    return result, mask


# =============================================================================
# METHOD 2: Watershed
# =============================================================================
def segment_watershed(frame):
    """
    Watershed segmentation using distance transform markers.

    Algorithm:
      1. Convert to grayscale, apply Gaussian blur
      2. Threshold to get binary mask
      3. Distance transform: d(x) = min_dist(x, boundary)
      4. Find markers via peak_local_max on distance transform
      5. Apply watershed on gradient image with markers

    The distance transform helps separate touching objects by finding
    local maxima (object centers) as watershed seeds.
    """
    # Which method handles touching/overlapping boxes best, and why:
    # Watershed wins on this case. Two boxes in contact share a silhouette, so
    # every thresholding method (and therefore every region-growing or
    # graph-based method) sees one blob, but the distance transform of that blob
    # has two separate interior maxima - one per box - and each peak becomes its
    # own marker. The watershed then floods the gradient outward from those
    # markers until the fronts collide along the contact seam, so it cuts
    # exactly where the boxes touch. Active contours and Felzenszwalb both
    # inherit the merged blob from thresholding and have no mechanism to place
    # a boundary inside it, and Mean Shift separates by colour/position only, so
    # two same-coloured boxes in contact stay a single cluster.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thresh = detect_foreground_mask(frame)

    # Distance transform — key for separating touching objects
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # Marker selection. A box's far interior is a flat plateau in the distance
    # transform, and peak_local_max reports every cell of a plateau as a
    # separate peak, so peak counting over-segments a single box. Instead we
    # threshold the distance map and take the connected components of the
    # resulting "deep interior" band: each box contributes exactly one
    # component (its whole floor), while two touching boxes still form two
    # lobes whose connecting ridge is shallower than the band threshold.
    # We scan a few band depths and keep the one that separates the most
    # plausible box interiors.
    fg_dists = dist_transform[thresh > 0]
    max_dist = float(fg_dists.max()) if fg_dists.size else 0.0
    fg_count = int((thresh > 0).sum())
    min_area = max(int(fg_count * 0.004), 25)
    band_slices = []   # boolean masks of each deep-interior component
    best_count = 0
    for frac in (0.50, 0.60, 0.70, 0.80, 0.90):
        band = dist_transform >= max(max_dist * frac, 2.0)
        n, labels = cv2.connectedComponents(band.astype(np.uint8))
        comp_masks = []
        for lab in range(1, n):
            mask = labels == lab
            if int(mask.sum()) < min_area:
                continue
            comp_masks.append(mask)
        if len(comp_masks) >= best_count:
            best_count = len(comp_masks)
            band_slices = comp_masks
        if len(band_slices) >= 12:
            break
    seeds = []
    for m in band_slices:
        ys, xs = np.where(m)
        k = np.argmax(dist_transform[ys, xs])
        seeds.append((int(ys[k]), int(xs[k])))

    if not seeds:
        # Empty foreground or a blob too small to hold a deep band: fall back
        # to connected-component labels so the function still returns a mask.
        comp_labels = ndimage.label(thresh > 0)[0]
        markers = np.ones((h, w), dtype=np.int32)
        markers[comp_labels > 0] = comp_labels[comp_labels > 0] + 1
        do_watershed = False
    else:
        # Marker-controlled watershed (OpenCV tutorial pattern). Instead of
        # single-point seeds - which cannot flood a flat box floor - label the
        # entire deep-interior regions produced above, leave the shallow box
        # edges unlabelled (0) so the algorithm carves them, and give the
        # background label 1 so the flood cannot escape the silhouettes.
        marker_img = np.zeros((h, w), dtype=np.int32)
        for idx, (r, c) in enumerate(seeds):
            rr, cc = np.where(band_slices[idx])
            marker_img[rr, cc] = idx + 2
        markers = np.where(thresh > 0, 0, 1)
        markers[marker_img > 1] = marker_img[marker_img > 1]
        do_watershed = True

    if do_watershed:
        # Watershed the INVERTED distance transform. Box interiors are flat in
        # the raw frame, so flooding the frame from a seed goes nowhere. The
        # inverted distance image is 0 at the box center and rises monotonically
        # toward the boundary, so the labeled interiors are basin minima and
        # the unlabelled edge band becomes the ridge the algorithm carves.
        inv_dist = (max_dist - dist_transform).astype(np.uint8)
        inv_dist = cv2.cvtColor(inv_dist, cv2.COLOR_GRAY2BGR)
        markers = markers.astype(np.int32)
        cv2.watershed(inv_dist, markers)

    # Color the result
    result = frame.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 255, 0), (255, 128, 0), (0, 128, 255)
    ]

    mask = np.zeros(gray.shape, dtype=np.uint8)
    unique_labels = np.unique(markers)
    for i, label in enumerate(unique_labels):
        if label <= 1:
            continue
        result[markers == label] = colors[i % len(colors)]
        mask[markers == label] = 255

    # Blend for better visualization
    result = cv2.addWeighted(frame, 0.6, result, 0.4, 0)

    # Outline each region. cv2.drawContours needs an (N,1,2) int32 array;
    # np.where() returns a tuple, so stack it along a new last axis.
    for label in unique_labels:
        if label <= 1:
            continue
        boundary = np.stack(np.where(markers == label), axis=-1).astype(np.int32)
        cv2.drawContours(result, [boundary.reshape(-1, 1, 2)], -1, (0, 0, 0), 1)

    return result, mask


# =============================================================================
# METHOD 3: Graph-Based Segmentation (Felzenszwalb)
# =============================================================================
def segment_graph(frame):
    """
    Felzenszwalb graph-based segmentation.

    Treats the image as a graph where:
      - Each pixel is a node
      - Edges connect neighboring pixels with weight = difference in feature space
      - Merges regions based on: w(e) <= min(MST(Ri), MST(Rj)) + k

    k controls the threshold for preferring larger components.
    sigma controls the pre-smoothing.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Felzenszwalb segmentation
    segments = felzenszwalb(
        rgb,
        sigma=0.8,       # Pre-smoothing sigma
        scale=100,        # Threshold for merging
        min_size=50       # Minimum component size
    )

    # Create colored visualization
    result_rgb = label2rgb(segments, rgb, kind='avg')

    # Create mask from the largest segments (likely the boxes)
    unique, counts = np.unique(segments, return_counts=True)
    # Get top segments by area
    sorted_idx = np.argsort(-counts)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    # Keep segments that are at least 2% of image area
    total_pixels = segments.size
    for idx in sorted_idx:
        if counts[idx] > total_pixels * 0.02:
            mask[segments == unique[idx]] = 255

    # label2rgb already returns uint8 in 0..255, so this is a plain colour swap.
    result_bgr = cv2.cvtColor(result_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Draw segment boundaries
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(segments, mode='thick')
    result_bgr[boundaries] = [0, 0, 255]

    return result_bgr, mask


# =============================================================================
# METHOD 4: Mean Shift Clustering
# =============================================================================
def segment_mean_shift(frame):
    """
    Mean Shift pixel clustering segmentation.

    Algorithm:
      1. Create feature vector for each pixel: [x, y, r, g, b] (spatial + color)
      2. For each point, compute the mean shift vector:
         m(x) = sum(K(||x - xi||) * xi) / sum(K(||x - xi||)) - x
      3. Shift point by m(x) and iterate until convergence
      4. Points converging to the same mode form a segment

    The bandwidth parameter controls the kernel size — larger bandwidth
    produces fewer, larger segments.
    """
    h, w = frame.shape[:2]

    # Downsample for computational efficiency
    scale = 2
    small = cv2.resize(frame, (w // scale, h // scale))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Create feature vectors: [x_scaled, y_scaled, r, g, b]
    # Position is mapped onto the same 0..255 range as the colour channels.
    # Without this the spatial axes span a few units while colour spans 255,
    # so estimate_bandwidth is set entirely by colour and the "color+position"
    # clustering degenerates to colour-only (two same-coloured boxes at opposite
    # ends of the belt get merged into one segment).
    sh, sw = small.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw]
    features = np.column_stack([
        xx.ravel() * (255.0 / max(sw - 1, 1)),
        yy.ravel() * (255.0 / max(sh - 1, 1)),
        rgb_small[:, :, 0].ravel(),
        rgb_small[:, :, 1].ravel(),
        rgb_small[:, :, 2].ravel()
    ]).astype(np.float32)

    # Estimate bandwidth automatically
    bandwidth = estimate_bandwidth(features, quantile=0.15, n_samples=2000)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 40.0
    # Cap the kernel so it spans at most ~15% of the frame spatially. The
    # estimator is a single joint radius across all 5 axes, so an uncapped
    # value lets the colour axes drag the radius past the image extent and
    # collapse every pixel into one cluster.
    bandwidth = float(np.clip(bandwidth, 15.0, 40.0))

    # Mean Shift clustering
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=10)
    labels = ms.fit_predict(features)

    # Reshape back to image
    label_map = labels.reshape(small.shape[:2])

    # Upscale labels to original size
    label_map_full = cv2.resize(
        label_map.astype(np.float32), (w, h),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    # Color each segment with its mean color
    result = np.zeros_like(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)

    unique_labels = np.unique(label_map_full)
    total_pixels = label_map_full.size

    for label_id in unique_labels:
        region = (label_map_full == label_id)
        region_size = np.sum(region)

        # Color with mean color of the region
        mean_color = np.mean(frame[region], axis=0).astype(np.uint8)
        result[region] = mean_color

        # Mask: keep regions larger than 1% of image (potential boxes)
        if region_size > total_pixels * 0.01:
            mask[region] = 255

    return result, mask


# =============================================================================
# Visualization: Comparison Grid
# =============================================================================
def create_comparison_grid(frames, methods_results, output_path):
    """
    Create a comparison grid:
    Frame N: Original | Active Contour | Watershed | Graph | Mean Shift
    """
    method_names = ['Original', 'Active Contour', 'Watershed', 'Graph-Based', 'Mean Shift']
    n_frames = len(frames)
    n_methods = len(method_names)

    fig, axes = plt.subplots(n_frames, n_methods, figsize=(20, 4 * n_frames))
    if n_frames == 1:
        axes = axes.reshape(1, -1)

    for i, (frame_idx, frame) in enumerate(frames):
        # Original
        axes[i, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Frame {frame_idx}\nOriginal', fontsize=10)
        axes[i, 0].axis('off')

        # Each method
        for j, (method_name, results) in enumerate(methods_results.items()):
            result, mask = results[i]
            if result.shape[2] == 3:
                display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            else:
                display = result
            axes[i, j + 1].imshow(display)
            axes[i, j + 1].set_title(f'Frame {frame_idx}\n{method_name}', fontsize=10)
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    project_root = get_project_root()
    video_path = project_root / 'data' / 'video' / 'conveyor.mp4'

    # Check for video
    if not video_path.exists():
        # Try common alternatives
        alt_paths = list((project_root / 'data' / 'video').glob('*.mp4')) + \
                    list((project_root / 'data' / 'video').glob('*.avi')) + \
                    list((project_root / 'data' / 'video').glob('*.mov'))
        if alt_paths:
            video_path = alt_paths[0]
        else:
            print("ERROR: No video found.")
            print("Please place your video at:")
            print(f"  {project_root / 'data' / 'video' / 'conveyor.mp4'}")
            sys.exit(1)

    print(f"Using video: {video_path}")

    # Extract representative frames
    print("Extracting representative frames...")
    frames = get_representative_frames(video_path, num_frames=5)
    print(f"  Selected frames: {[idx for idx, _ in frames]}")

    # Save extracted frames
    frames_dir = project_root / 'data' / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in frames:
        cv2.imwrite(str(frames_dir / f'frame_{idx:04d}.png'), frame)

    # Run all 4 segmentation methods on all 5 frames
    methods = {
        'Active Contour': segment_active_contour,
        'Watershed': segment_watershed,
        'Graph-Based': segment_graph,
        'Mean Shift': segment_mean_shift,
    }

    all_results = {}
    failures = {}
    for method_name, method_func in methods.items():
        print(f"\nRunning {method_name} segmentation...")
        results = []
        for idx, frame in frames:
            try:
                result, mask = method_func(frame)
                results.append((idx, result, mask))
                print(f"  Frame {idx}: OK")
            except Exception as e:
                # Record the failure loudly. Swallowing it here would put an
                # unprocessed frame into the comparison grid, making a broken
                # method look like a working one.
                print(f"  Frame {idx}: FAILED - {type(e).__name__}: {e}")
                failures.setdefault(method_name, []).append((idx, str(e)))
                results.append((idx, frame, np.zeros(frame.shape[:2], dtype=np.uint8)))
        all_results[method_name] = results

    if failures:
        print("\n" + "!" * 60)
        print("SOME METHODS FAILED - the grid below is NOT a valid comparison")
        for method_name, errs in failures.items():
            print(f"  {method_name}: failed on frames {[e[0] for e in errs]}")
            for idx, msg in errs:
                print(f"      frame {idx}: {msg}")
        print("!" * 60)

    # Create comparison grid
    output_dir = project_root / 'outputs' / 'segmentation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for grid visualization
    grid_data = {}
    for method_name, results in all_results.items():
        grid_data[method_name] = [(r[1], r[2]) for r in results]

    create_comparison_grid(frames, grid_data, output_dir / 'comparison_grid.png')

    # Save individual segmentation masks
    for method_name, results in all_results.items():
        method_dir = output_dir / method_name.lower().replace(' ', '_')
        method_dir.mkdir(parents=True, exist_ok=True)
        for idx, result, mask in results:
            cv2.imwrite(str(method_dir / f'frame_{idx:04d}_result.png'), result)
            cv2.imwrite(str(method_dir / f'frame_{idx:04d}_mask.png'), mask)

    # Print analysis. Region counts are measured from the actual masks rather
    # than asserted, so the conclusion follows from the run instead of being
    # boilerplate that would print even if every method had failed.
    print("\n" + "=" * 60)
    print("SEGMENTATION ANALYSIS")
    print("=" * 60)
    print(f"{'Method':<18}{'mean regions/frame':>20}{'mean coverage':>16}")
    region_stats = {}
    for method_name, results in all_results.items():
        counts, covs = [], []
        for idx, result, mask in results:
            n_regions, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
            counts.append(n_regions - 1)
            covs.append((mask > 0).mean() * 100.0)
        region_stats[method_name] = (float(np.mean(counts)), float(np.mean(covs)))
        print(f"{method_name:<18}{np.mean(counts):>20.2f}{np.mean(covs):>15.1f}%")

    if failures:
        print("\nNOTE: the statistics above include frames where the method crashed")
        print("and the mask was left empty, so they understate those methods.")

    print("\nSegmentation module complete. Outputs saved to:", output_dir)


if __name__ == '__main__':
    main()
