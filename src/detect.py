from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2

from parse_xml import parse_annotations


def draw_boxes(image, boxes: List[Dict[str, int]]):
    """Draw annotation bounding boxes on an image."""
    for box in boxes:
        cv2.rectangle(
            image,
            (box["xmin"], box["ymin"]),
            (box["xmax"], box["ymax"]),
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            image,
            "damage",
            (box["xmin"], max(15, box["ymin"] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def find_image_path(images_dir: Path, image_name: str) -> Path:
    """Resolve the image path from annotation filename with fallback extensions."""
    candidate = images_dir / image_name
    if candidate.exists():
        return candidate

    stem = Path(image_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        fallback = images_dir / f"{stem}{ext}"
        if fallback.exists():
            return fallback

    return candidate


def visualize_detections(
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
) -> tuple[int, int, int]:
    """
    Create annotated images by drawing parsed bounding boxes.

    Returns tuple:
    - processed_xml_count
    - saved_image_count
    - missing_image_count
    """
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raw_fallback = images_dir / "raw"
        if raw_fallback.exists():
            images_dir = raw_fallback
        else:
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_xml = 0
    saved_images = 0
    missing_images = 0

    parsed_annotations = parse_annotations(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
    )

    for image_name, boxes in parsed_annotations.items():
        processed_xml += 1

        image_path = find_image_path(images_dir, image_name)
        if not image_path.exists():
            missing_images += 1
            print(f"[WARN] Image not found. Expected: {image_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not open image: {image_path.name}")
            continue

        annotated = draw_boxes(image, boxes)
        out_path = output_dir / image_path.name
        success = cv2.imwrite(str(out_path), annotated)
        if success:
            saved_images += 1
        else:
            print(f"[WARN] Failed to save output image: {out_path.name}")

    return processed_xml, saved_images, missing_images


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STEP 2: Draw and save road damage detections"
    )
    parser.add_argument(
        "--annotations-dir",
        default="dataset/annotations",
        help="Path to Pascal VOC XML annotations directory",
    )
    parser.add_argument(
        "--images-dir",
        default="dataset/images",
        help="Path to image directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/detections",
        help="Path where annotated images will be saved",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    annotations_dir = Path(args.annotations_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    processed_xml, saved_images, missing_images = visualize_detections(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        output_dir=output_dir,
    )

    print(f"Processed XML files: {processed_xml}")
    print(f"Annotated images saved: {saved_images}")
    print(f"Missing images: {missing_images}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
