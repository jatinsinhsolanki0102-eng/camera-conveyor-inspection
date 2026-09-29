"""
Module E — Object Recognition
==============================
Identifies box types using classical computer vision:
1. Alignment-based recognition (template matching + homography)
2. PCA/Eigenspace recognition (Eigenboxes)
3. Hu Moments (rotation/scale invariant feature)
4. Accuracy comparison table

No deep learning — uses geometric and appearance-based methods.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist


def get_project_root():
    return Path(__file__).parent.parent


# =============================================================================
# DATA LOADING
# =============================================================================
def load_box_crops(boxes_dir, target_size=(100, 100)):
    """
    Load box crop images from directory structure:
      boxes_dir/
        box_type_A/
          crop_001.png
          ...
        box_type_B/
          ...
        box_type_C/
          ...

    Returns:
        images: list of (label, resized_image) tuples
        labels: list of label strings
    """
    images = []
    labels = []

    box_types = sorted([d for d in boxes_dir.iterdir() if d.is_dir()])

    if not box_types:
        return [], []

    for box_type_dir in box_types:
        label = box_type_dir.name
        image_files = list(box_type_dir.glob('*.png')) + \
                     list(box_type_dir.glob('*.jpg')) + \
                     list(box_type_dir.glob('*.jpeg'))

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            resized = cv2.resize(img, target_size)
            images.append(resized)
            labels.append(label)

    return images, labels


# =============================================================================
# METHOD 1: ALIGNMENT-BASED RECOGNITION
# =============================================================================
def compute_alignment_error(query, template):
    """
    Compute alignment error between query and template images.

    Steps:
    1. Convert to grayscale
    2. Detect ORB features
    3. Match features between query and template
    4. Estimate homography using RANSAC
    5. Compute alignment residual (reprojection error)

    Lower error = better match.
    """
    gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray_query, None)
    kp2, des2 = orb.detectAndCompute(gray_template, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return float('inf'), 0

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 4:
        return float('inf'), len(matches)

    # Sort matches by distance
    matches = sorted(matches, key=lambda m: m.distance)

    # Extract matched point positions
    pts_query = np.array([kp1[m.queryIdx].pt for m in matches])
    pts_template = np.array([kp2[m.trainIdx].pt for m in matches])

    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(pts_query, pts_template, cv2.RANSAC, 5.0)

    if H is None:
        return float('inf'), len(matches)

    # Compute reprojection error (alignment residual)
    pts_query_h = np.column_stack([pts_query, np.ones(len(pts_query))])
    projected = (H @ pts_query_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.sqrt(np.sum((projected - pts_template)**2, axis=1))
    mean_error = np.mean(errors)

    return mean_error, len(matches)


def alignment_classifier(query, templates):
    """
    Classify query image by finding the template with lowest alignment error.

    Returns: (predicted_label, best_error, all_errors)
    """
    best_label = None
    best_error = float('inf')
    all_errors = {}

    for label, template in templates:
        error, n_matches = compute_alignment_error(query, template)
        all_errors[label] = error

        if error < best_error:
            best_error = error
            best_label = label

    return best_label, best_error, all_errors


# =============================================================================
# METHOD 2: PCA / EIGENSPACE RECOGNITION
# =============================================================================
class EigenboxClassifier:
    """
    PCA-based box recognition (Eigenfaces-like method for boxes).

    Algorithm:
    1. Flatten all training images to vectors
    2. Compute mean face
    3. Subtract mean to get centered vectors
    4. Compute PCA (eigenvectors of covariance matrix)
    5. Project training images into eigenspace
    6. For query: project into eigenspace, find nearest training sample

    The eigenvectors represent the principal modes of variation
    in the training set (shape, lighting, texture).
    """

    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.training_data = None  # (flattened, label) pairs
        self.training_labels = []
        self.mean_face = None

    def fit(self, images, labels):
        """
        Fit PCA model on training images.

        images: list of BGR images (same size)
        labels: list of label strings
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 training images")

        target_size = images[0].shape[:2]

        # Flatten images
        flat = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (target_size[1], target_size[0]))
            flat.append(resized.flatten().astype(np.float64))

        X = np.array(flat)

        # Compute mean face
        self.mean_face = np.mean(X, axis=0)

        # Center data
        X_centered = X - self.mean_face

        # Fit PCA
        self.pca = PCA(n_components=min(self.n_components, len(images) - 1))
        self.X_projected = self.pca.fit_transform(X_centered)

        self.training_labels = labels
        self.training_images = images

        print(f"  PCA fitted: {self.pca.n_components_} components, "
              f"explained variance: {sum(self.pca.explained_variance_ratio_):.1%}")

    def predict(self, query_img):
        """
        Predict box type for a query image.

        Returns: (predicted_label, distances_to_classes)
        """
        gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.mean_face.shape[1] // 1 if len(self.mean_face.shape) > 1 else int(np.sqrt(len(self.mean_face))),
                                  self.mean_face.shape[0] if len(self.mean_face.shape) > 1 else int(np.sqrt(len(self.mean_face)))))
        flat = gray.flatten().astype(np.float64)

        # Project query into eigenspace
        query_centered = flat - self.mean_face
        query_projected = self.pca.transform(query_centered.reshape(1, -1))

        # Find nearest training sample
        distances = cdist(query_projected, self.X_projected, metric='euclidean')[0]

        # Compute per-class distances (average distance to all samples of each class)
        unique_labels = list(set(self.training_labels))
        class_distances = {}
        for label in unique_labels:
            class_mask = [i for i, l in enumerate(self.training_labels) if l == label]
            class_distances[label] = np.mean(distances[class_mask])

        predicted_label = min(class_distances, key=class_distances.get)
        return predicted_label, class_distances

    def get_eigenvectors(self, n=3):
        """Return top n eigenvectors as images."""
        eigenfaces = []
        for i in range(min(n, self.pca.n_components_)):
            eigenface = self.pca.components_[i]
            # Reshape to image
            h = int(np.sqrt(len(eigenface)))
            w = len(eigenface) // h
            eigenface_img = eigenface[:h*w].reshape(h, w)
            eigenfaces.append(eigenface_img)
        return eigenfaces


# =============================================================================
# METHOD 3: HU MOMENTS (Invariant Feature)
# =============================================================================
def compute_hu_moments(image):
    """
    Compute Hu Moments — 7 rotation, scale, and translation invariant features.

    Hu (1962) derived 7 moment invariants from central moments:
      I1 = n20*n02 - n11^2
      I2 = (n30*n03 - n12*n21)^2 + ...
      ... (7 equations)

    These are invariant to:
      - Translation (central moments)
      - Scale (normalized moments)
      - Rotation (moment combinations)

    Useful as a fast descriptor for shape comparison.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Hu moment invariants are defined on binary silhouettes, NOT on raw
    # grayscale levels: otherwise every change of cardboard brightness (or
    # illumination) rescales the intensities and the moments stop being
    # shape-based. Binarise the crop first so only the printed shape counts.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure the printed mark is the foreground (white) shape; take whichever
    # polarity makes the better-balanced silhouette
    fg = cv2.countNonZero(binary)
    if fg > binary.size * 0.7:
        binary = 255 - binary
    moments = cv2.moments(binary.astype(np.float64) / 255.0)
    hu = cv2.HuMoments(moments).flatten()

    # Log-transform for better numerical properties
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu_log


def compare_hu_moments(query, reference):
    """
    Compare two images using Hu Moments distance.

    Lower distance = more similar shapes.
    """
    hu_query = compute_hu_moments(query)
    hu_ref = compute_hu_moments(reference)

    # Euclidean distance in Hu moment space
    distance = np.sqrt(np.sum((hu_query - hu_ref)**2))
    return distance


def hu_moment_classifier(query, templates_by_label):
    """
    Classify using Hu Moments — find most similar template.

    Returns: (predicted_label, all_distances)
    """
    all_distances = {}
    for label, templates in templates_by_label.items():
        distances = [compare_hu_moments(query, t) for t in templates]
        all_distances[label] = np.mean(distances)

    predicted_label = min(all_distances, key=all_distances.get)
    return predicted_label, all_distances


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_eigenvectors(classifier, output_path):
    """Visualize top 3 eigenvectors (eigenboxes)."""
    eigenfaces = classifier.get_eigenvectors(n=3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, eigenface in enumerate(eigenfaces):
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f'Eigenvector {i+1}')
        axes[i].axis('off')

    plt.suptitle('Top 3 PCA Eigenvectors (Eigenboxes)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_hu_moments_invariance(output_path):
    """
    Show that Hu moments are invariant to translation, rotation, and scale.
    """
    # Create a simple test shape
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 70), 255, -1)

    # Original
    hu_orig = compute_hu_moments(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    # Translated
    M_translate = np.float32([[1, 0, 30], [0, 1, 20]])
    translated = cv2.warpAffine(img, M_translate, (150, 150))
    hu_trans = compute_hu_moments(cv2.cvtColor(translated, cv2.COLOR_GRAY2BGR))

    # Rotated
    center = (50, 50)
    M_rotate = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, M_rotate, (100, 100))
    hu_rot = compute_hu_moments(cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR))

    # Scaled
    scaled = cv2.resize(img, None, fx=1.5, fy=1.5)
    hu_scaled = compute_hu_moments(cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR))

    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(translated, cmap='gray')
    axes[0, 1].set_title('Translated (+30, +20)')
    axes[0, 2].imshow(rotated, cmap='gray')
    axes[0, 2].set_title('Rotated (45°)')
    axes[0, 3].imshow(scaled, cmap='gray')
    axes[0, 3].set_title('Scaled (1.5x)')

    # Show Hu moments
    for i, (name, hu) in enumerate([
        ('Original', hu_orig), ('Translated', hu_trans),
        ('Rotated', hu_rot), ('Scaled', hu_scaled)
    ]):
        axes[1, i].bar(range(7), hu)
        axes[1, i].set_title(f'{name}\nHu[0]={hu[0]:.2f}')
        axes[1, i].set_xlabel('Moment index')

    plt.suptitle('Hu Moment Invariance Demonstration', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# ACCURACY EVALUATION
# =============================================================================
def evaluate_classifiers(images, labels, test_size=0.3):
    """
    Split data into train/test and evaluate all three classifiers
    (alignment, PCA/eigenspace, Hu moments).

    Returns a dict with accuracies and the fitted artifacts needed for the
    rotation-invariance evaluation that follows in main().
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42
    )

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # --- PCA / Eigenspace Classifier ---
    print("\n  Training PCA classifier...")
    pca_clf = EigenboxClassifier(n_components=min(20, len(X_train) - 1))
    pca_clf.fit(X_train, y_train)

    pca_predictions = []
    for img in X_test:
        pred, _ = pca_clf.predict(img)
        pca_predictions.append(pred)
    pca_accuracy = accuracy_score(y_test, pca_predictions) * 100

    # --- Alignment Classifier ---
    print("  Training Alignment classifier...")
    # One template per class: the first training image of that class
    templates_by_class = {}
    for label in set(y_train):
        class_images = [img for img, l in zip(X_train, y_train) if l == label]
        templates_by_class[label] = class_images[0]

    alignment_predictions = []
    for img in X_test:
        templates = [(label, tmpl) for label, tmpl in templates_by_class.items()]
        pred, _, _ = alignment_classifier(img, templates)
        alignment_predictions.append(pred)
    # A None prediction means the method found too few features to decide;
    # count it honestly as a miss.
    alignment_predictions = [p if p is not None else 'UNKNOWN'
                             for p in alignment_predictions]
    alignment_accuracy = accuracy_score(y_test, alignment_predictions) * 100

    # --- Hu Moments Classifier ---
    print("  Training Hu-moments classifier...")
    # Needs ALL training images per class (uses the mean Hu distance)
    templates_by_label = {}
    for label in set(y_train):
        templates_by_label[label] = [
            img for img, l in zip(X_train, y_train) if l == label
        ]

    hu_predictions = []
    for img in X_test:
        pred, _ = hu_moment_classifier(img, templates_by_label)
        hu_predictions.append(pred)
    hu_accuracy = accuracy_score(y_test, hu_predictions) * 100

    return {
        'X_test': X_test, 'y_test': y_test,
        'x_train': X_train, 'y_train': y_train,
        'pca_accuracy': pca_accuracy, 'alignment_accuracy': alignment_accuracy,
        'hu_accuracy': hu_accuracy,
        'pca_predictions': pca_predictions,
        'align_predictions': alignment_predictions,
        'hu_predictions': hu_predictions,
        'pca_clf': pca_clf,
        'templates_by_class': templates_by_class,
        'templates_by_label': templates_by_label,
    }


def evaluate_rotation_invariance(test_images, test_labels, templates_by_class,
                                 templates_by_label, pca_clf,
                                 angles=(15, 30, 60, 90, 135, 180)):
    """
    Rotate every test image by each angle and re-classify with all three
    methods. Hu moments should stay accurate across rotations (true
    rotation invariance), while PCA degrades sharply (it expects aligned
    crops) and alignment stays good (its homography absorbs rotation).

    Returns a dict: method name -> list of accuracies (one per angle).
    """
    from collections import defaultdict
    acc = defaultdict(list)

    templates = [(label, tmpl) for label, tmpl in templates_by_class.items()]

    print("\n  Rotation invariance test (all test images rotated then"
          "\n  re-classified, accuracy vs the ORIGINAL labels):")
    print(f"    {'Angle':>7} | {'Alignment':>10} {'PCA':>10} {'Hu':>10}")

    for angle in angles:
        rot_images = []
        for img in test_images:
            h, w = img.shape[:2]
            # Pad first so the rotated object never leaves the canvas:
            # filling with reflected PIXELS would add mirrored glyph parts
            # and corrupt the silhouette (esp. for Hu moments / PCA).
            pad = 90
            mid = [int(np.median(img[:, :, k])) for k in range(3)]
            canvas = cv2.copyMakeBorder(
                img, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=tuple(mid))
            ch, cw = canvas.shape[:2]
            M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
            canvas = cv2.warpAffine(canvas, M, (cw, ch))
            rot_images.append(canvas[pad:pad + h, pad:pad + w])

        pca_preds, align_preds, hu_preds = [], [], []
        for r_img in rot_images:
            pred, _ = pca_clf.predict(r_img)
            pca_preds.append(pred)
            pred, _, _ = alignment_classifier(r_img, templates)
            align_preds.append(pred if pred is not None else 'UNKNOWN')
            pred, _ = hu_moment_classifier(r_img, templates_by_label)
            hu_preds.append(pred)

        a_pca = accuracy_score(test_labels, pca_preds) * 100
        a_align = accuracy_score(test_labels, align_preds) * 100
        a_hu = accuracy_score(test_labels, hu_preds) * 100
        acc['Alignment'].append(a_align)
        acc['PCA/Eigenspace'].append(a_pca)
        acc['Hu moments'].append(a_hu)
        print(f"    {angle:>5}deg | {a_align:>9.1f}% {a_pca:>9.1f}% "
              f"{a_hu:>9.1f}%")

    return acc


# =============================================================================
# Main
# =============================================================================
def main():
    project_root = get_project_root()
    output_dir = project_root / 'outputs' / 'recognition'
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes_dir = project_root / 'data' / 'boxes'

    print("=" * 60)
    print("MODULE E — OBJECT RECOGNITION")
    print("=" * 60)

    # Load data
    print("\nLoading box crops...")
    images, labels = load_box_crops(boxes_dir)

    if not images:
        print("\nERROR: No box crop images found.")
        print("Please organize your box samples as:")
        print(f"  {boxes_dir / 'box_type_A'}/")
        print(f"    crop_001.png")
        print(f"    crop_002.png")
        print(f"    ... (approximately 10 per type)")
        print(f"  {boxes_dir / 'box_type_B'}/")
        print(f"  {boxes_dir / 'box_type_C'}/")
        print("\nEach image should be a cropped view of the box face.")
        sys.exit(1)

    unique_labels = list(set(labels))
    print(f"  Found {len(images)} images across {len(unique_labels)} classes: {unique_labels}")

    for label in unique_labels:
        count = labels.count(label)
        print(f"    {label}: {count} samples")

    # =========================================================================
    # Evaluate classifiers
    # =========================================================================
    print("\n" + "=" * 60)
    print("CLASSIFIER EVALUATION")
    print("=" * 60)

    res = evaluate_classifiers(images, labels)
    pca_clf = res['pca_clf']
    y_test = res['y_test']
    pca_preds = res['pca_predictions']
    align_preds = res['align_predictions']
    hu_preds = res['hu_predictions']
    pca_acc = res['pca_accuracy']
    align_acc = res['alignment_accuracy']
    hu_acc = res['hu_accuracy']

    # Print accuracy table
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)
    print(f"\n{'Method':<30} {'Accuracy':>10}")
    print("-" * 45)
    print(f"{'Alignment-based':<30} {align_acc:>9.1f}%")
    print(f"{'PCA/Eigenspace':<30} {pca_acc:>9.1f}%")
    print(f"{'Hu moments':<30} {hu_acc:>9.1f}%")
    print("-" * 45)

    # Rotation-invariance comparison across all three methods
    print("\n" + "=" * 60)
    print("ROTATION INVARIANCE (all three descriptors)")
    print("=" * 60)
    rot_acc = evaluate_rotation_invariance(
        res['X_test'], y_test,
        res['templates_by_class'], res['templates_by_label'], pca_clf)

    # Plot rotation accuracies
    fig, ax = plt.subplots(figsize=(9, 6))
    angles = [15, 30, 60, 90, 135, 180]
    for method, accs in rot_acc.items():
        ax.plot(angles, accs, 'o-', linewidth=2, label=method)
    ax.set_xlabel('Rotation angle (degrees)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Rotation invariance comparison', fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(angles)
    plt.tight_layout()
    plt.savefig(output_dir / 'rotation_invariance.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved: rotation_invariance.png")

    # =========================================================================
    # Hu Moments invariance demonstration
    # =========================================================================
    print("\n" + "=" * 60)
    print("HU MOMENTS INVARIANCE")
    print("=" * 60)

    visualize_hu_moments_invariance(output_path=output_dir / 'hu_moments_invariance.png')

    # =========================================================================
    # Visualize eigenvectors
    # =========================================================================
    print("\n" + "=" * 60)
    print("PCA EIGENVECTORS")
    print("=" * 60)

    visualize_eigenvectors(pca_clf, output_path=output_dir / 'eigenvectors.png')

    # =========================================================================
    # Confusion matrices (PCA + Hu)
    # =========================================================================
    if len(unique_labels) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for j, (title, preds) in enumerate([
                ('PCA/Eigenspace', pca_preds),
                ('Hu Moments', hu_preds)]):
            cm = confusion_matrix(y_test, preds, labels=unique_labels)
            ax = axes[j]
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f'{title} Confusion Matrix', fontsize=14)
            plt.colorbar(im, ax=ax)
            tick_marks = np.arange(len(unique_labels))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(unique_labels, rotation=45)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(unique_labels)
            for i in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    ax.text(k, i, format(cm[i, k], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, k] > cm.max() / 2 else "black")
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150,
                    bbox_inches='tight')
        plt.close()
        print("  Saved: confusion_matrix.png")

    # =========================================================================
    # Analysis comments
    # =========================================================================
    print("""
Classifier Comparison:
---------------------

Alignment-based:
  + Handles rotation and scale changes well (via homography)
  + Requires fewer training samples (one template per class)
  - Sensitive to appearance changes (lighting, color)
  - Requires sufficient texture/features for matching
  - Slower (feature detection + matching per query)

PCA/Eigenspace:
  + Captures global appearance variations
  + Fast prediction (simple distance computation)
  + Robust to minor appearance changes
  - Sensitive to rotation and scale (must be normalized)
  - Requires more training samples for good coverage

Hu moments:
  + Rotation, scale and translation invariant by construction
  + Very fast and cheap (pure moments, no training)
  - Captures only coarse shape statistics, not texture/appearance
  - Two visually different textured boxes with the same silhouette are
    indistinguishable to Hu features alone

Rotation invariance (from rotation_invariance.png):
  Hu moments hold ~constant accuracy at every angle, because they are
  rotation invariants by definition — this is the cheapest way to keep
  recognising a box that lies sideways on the belt.
  The alignment method can withstand rotation when enough keypoints are
  found, because its homography absorbs the rotation before comparing
  templates (it degrades on low-texture faces).
  PCA drops sharply as soon as a crop is rotated, because the eigenspace
  was built from aligned training crops.
""")

    print(f"\nModule E outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
