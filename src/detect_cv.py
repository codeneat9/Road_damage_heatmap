"""
STEP 1-3: Computer Vision Image Processing Module
=================================================

Purpose:
- Load road damage images from dataset
- Preprocess images (grayscale, Gaussian Blur)
- Detect edges for candidate damage regions
- Find contours from edges for candidate damage regions

No ML models used - pure OpenCV image processing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# Configuration
DATASET_IMAGES_DIR = Path("dataset/images")
CV_OUTPUT_DIR = Path("data/cv_output")
BLUR_KERNEL_SIZE = (5, 5)  # Gaussian blur kernel
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150
MIN_CONTOUR_AREA = 500


def ensure_output_dir(output_dir: Path = CV_OUTPUT_DIR) -> None:
    """Create the CV output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def save_processed_outputs(
    image_path: Path,
    original_image: np.ndarray,
    preprocessed_image: np.ndarray,
    edge_image: np.ndarray,
    annotated_image: np.ndarray,
    output_dir: Path = CV_OUTPUT_DIR,
) -> dict[str, Path]:
    """
    STEP 8: Save the processed output images to disk.

    Saves a small set of useful artifacts for inspection:
    - original copy
    - preprocessed grayscale/blurred image
    - edge map
    - annotated bounding-box image
    """
    ensure_output_dir(output_dir)

    stem = image_path.stem
    saved_paths: dict[str, Path] = {}

    original_out = output_dir / f"{stem}_original.jpg"
    preprocessed_out = output_dir / f"{stem}_preprocessed.png"
    edges_out = output_dir / f"{stem}_edges.png"
    annotated_out = output_dir / f"{stem}_annotated.jpg"

    cv2.imwrite(str(original_out), original_image)
    cv2.imwrite(str(preprocessed_out), preprocessed_image)
    cv2.imwrite(str(edges_out), edge_image)
    cv2.imwrite(str(annotated_out), annotated_image)

    saved_paths["original"] = original_out
    saved_paths["preprocessed"] = preprocessed_out
    saved_paths["edges"] = edges_out
    saved_paths["annotated"] = annotated_out

    print(f"  -> STEP 8 save complete | Output dir: {output_dir}")
    print(f"     Saved: {annotated_out.name}")
    return saved_paths


def load_image(image_path: str | Path) -> Optional[np.ndarray]:
    """
    STEP 1: Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array, or None if loading fails
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Failed to load image: {image_path}")
            return None
        print(f"[OK] Loaded: {image_path} | Shape: {image.shape}")
        return image
    except Exception as e:
        print(f"[ERROR] Error loading {image_path}: {e}")
        return None


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    STEP 2A: Convert color image to grayscale.
    
    Why grayscale?
    - Reduces data (3 channels -> 1 channel)
    - Simplifies edge/contour detection
    - Faster processing
    
    Args:
        image: Color image (BGR format from OpenCV)
        
    Returns:
        Grayscale image
    """
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"  -> Converted to grayscale | Shape: {gray.shape}")
    return gray


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = BLUR_KERNEL_SIZE) -> np.ndarray:
    """
    STEP 2B: Apply Gaussian Blur to remove noise.
    
    Why blur?
    - Reduces noise and small variations
    - Smooths image for cleaner edge detection
    - Prevents false positives in contour detection
    
    Parameters:
    - kernel_size: Must be odd (e.g., 5x5, 7x7)
    - Larger kernel = more blur = smoother edges
    
    Args:
        image: Grayscale image
        kernel_size: Blur kernel size (default 5x5)
        
    Returns:
        Blurred image
    """
    if image is None:
        return None
    
    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=0)
    print(f"  -> Applied Gaussian Blur (kernel: {kernel_size}) | Shape: {blurred.shape}")
    return blurred


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    STEP 3: Complete preprocessing pipeline.
    
    Chain together:
    1. Grayscale conversion
    2. Gaussian Blur
    
    Args:
        image: Original color image
        
    Returns:
        Preprocessed (grayscale + blurred) image
    """
    if image is None:
        return None
    
    print("\n  [PIPELINE] Preprocessing pipeline:")
    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    print(f"  -> Preprocessing complete!\n")
    return blurred


def detect_edges(image: np.ndarray, threshold1: int = CANNY_THRESHOLD_1, threshold2: int = CANNY_THRESHOLD_2) -> np.ndarray:
    """
    STEP 4: Detect edges using Canny edge detection.

    Why Canny?
    - Finds strong intensity transitions in the image
    - Highlights road cracks, pothole boundaries, and damaged region outlines
    - Provides a clean edge map for contour detection in the next step

    Args:
        image: Preprocessed grayscale image
        threshold1: Lower hysteresis threshold
        threshold2: Upper hysteresis threshold

    Returns:
        Binary edge image
    """
    if image is None:
        return None

    edges = cv2.Canny(image, threshold1, threshold2)
    edge_pixels = int(np.count_nonzero(edges))
    print(f"  -> Canny edge detection complete | Thresholds: ({threshold1}, {threshold2}) | Edge pixels: {edge_pixels}")
    return edges


def find_contours(edge_image: np.ndarray) -> list[np.ndarray]:
    """
    STEP 5: Find contours from the binary edge image.

    Why contours?
    - Contours convert edges into closed region candidates
    - These regions can later be filtered and boxed as possible damage areas
    - Useful for visual validation of the damage detector

    Args:
        edge_image: Binary edge image from Canny detection

    Returns:
        List of detected contours
    """
    if edge_image is None:
        return []

    contours_info = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    print(f"  -> Contour detection complete | Found contours: {len(contours)}")
    return contours


def filter_contours(contours: list[np.ndarray], min_area: int = MIN_CONTOUR_AREA) -> list[np.ndarray]:
    """
    STEP 6: Filter out small noisy contours.

    Why filter?
    - Tiny contours are usually noise, texture, or compression artifacts
    - We keep only meaningful candidate regions that may correspond to damage

    Args:
        contours: Raw contours from contour detection
        min_area: Minimum contour area to keep

    Returns:
        Filtered list of contours
    """
    if not contours:
        return []

    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered.append(contour)

    print(
        f"  -> Noise filtering complete | Min area: {min_area} | Kept: {len(filtered)} / {len(contours)} contours"
    )
    return filtered


def draw_bounding_boxes(image: np.ndarray, contours: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """
    STEP 7: Draw bounding boxes around filtered contours.

    Why bounding boxes?
    - Gives a clear visual highlight of possible damage regions
    - Useful for validation and later comparison with annotations
    - Keeps the module simple and explainable without any ML model

    Args:
        image: Original color image
        contours: Filtered contours from STEP 6

    Returns:
        Tuple of (annotated_image, boxes_drawn_count)
    """
    if image is None:
        return None, 0

    annotated = image.copy()
    box_count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            "Possible Damage",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        box_count += 1

    print(f"  -> Bounding box drawing complete | Boxes drawn: {box_count}")
    return annotated, box_count


def process_image(image_path: str | Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], list[np.ndarray], list[np.ndarray], Optional[np.ndarray], int, dict[str, Path]]:
    """
    STEP 1-7 COMBINED: Load, preprocess, detect edges, find contours, filter noise, and draw boxes for a single image.
    
    This is the main entry point for CV processing.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (original_image, preprocessed_image, edge_image, contours, filtered_contours, annotated_image, box_count, saved_paths), or (None, None, None, [], [], None, 0, {}) on failure
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Step 1: Load
    original = load_image(image_path)
    if original is None:
        return None, None, None, [], [], None, 0, {}
    
    # Step 2-3: Preprocess
    preprocessed = preprocess_image(original)
    if preprocessed is None:
        return None, None, None, [], [], None, 0, {}

    # Step 4: Edge detection
    edges = detect_edges(preprocessed)
    if edges is None:
        return None, None, None, [], [], None, 0, {}

    # Step 5: Contour detection
    contours = find_contours(edges)
    
    # Step 6: Remove small noisy contours
    filtered_contours = filter_contours(contours)

    # Step 7: Draw bounding boxes on original image
    annotated_image, box_count = draw_bounding_boxes(original, filtered_contours)

    # Step 8: Save outputs to disk for later review
    saved_paths = save_processed_outputs(
        image_path=Path(image_path),
        original_image=original,
        preprocessed_image=preprocessed,
        edge_image=edges,
        annotated_image=annotated_image if annotated_image is not None else original,
    )
    
    return original, preprocessed, edges, contours, filtered_contours, annotated_image, box_count, saved_paths


def batch_process_images(
    image_dir: Path = DATASET_IMAGES_DIR,
    max_images: Optional[int] = None,
) -> list[Tuple[Path, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], Optional[np.ndarray], int, dict[str, Path]]]:
    """
    STEP 1-3 BATCH: Process multiple images from a directory.
    
    Args:
        image_dir: Directory containing images
        max_images: Limit number of images (None = all)
        
    Returns:
        List of tuples: (image_path, original_image, preprocessed_image, edge_image, contours, filtered_contours, annotated_image, box_count, saved_paths)
    """
    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        return []
    
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"[WARN] No images found in {image_dir}")
        return []
    
    # Limit if specified
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n[INFO] Found {len(image_files)} images in {image_dir}\n")
    
    results = []
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]")
        original, preprocessed, edges, contours, filtered_contours, annotated_image, box_count, saved_paths = process_image(img_path)
        
        if original is not None and preprocessed is not None and edges is not None:
            results.append((img_path, original, preprocessed, edges, contours, filtered_contours, annotated_image, box_count, saved_paths))
    
    print(f"\n{'='*60}")
    print(f"[OK] Successfully processed: {len(results)}/{len(image_files)} images")
    print(f"{'='*60}\n")
    
    return results


def main() -> None:
    """
    DEMO: Process first 3 images from dataset.
    
    Shows the preprocessing pipeline in action.
    """
    print("\n" + "="*60)
    print("STEP 1-8: CV IMAGE LOADING, DETECTION, AND SAVE OUTPUT")
    print("="*60)
    print("\nThis demo loads road damage images, detects candidate damage regions, and saves outputs.")
    print("No ML model is used - this is pure OpenCV image processing.\n")
    
    # Process up to 3 images
    results = batch_process_images(max_images=3)
    
    if results:
        print("\n[SUCCESS] EDGE + CONTOUR BOXING + SAVE COMPLETE")
        print(f"   Processed {len(results)} images successfully")
        print("\nResults saved to data/cv_output/ - ready for dashboard toggle integration")
    else:
        print("\n[ERROR] No images were processed. Check dataset/images/ directory.")


if __name__ == "__main__":
    main()
