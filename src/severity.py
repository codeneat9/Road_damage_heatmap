from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2

from parse_xml import parse_annotations


def image_area(image_path: Path) -> int:
    """Return image area (width * height)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to open image: {image_path}")

    height, width = image.shape[:2]
    return width * height


def bbox_area(box: Dict[str, int]) -> int:
    """Compute area of one bounding box."""
    return max(0, box["xmax"] - box["xmin"]) * max(0, box["ymax"] - box["ymin"])


def resolve_image_path(image_roots: List[Path], image_name: str) -> Path:
    """Resolve image path from one of the available image roots."""
    for root in image_roots:
        direct = root / image_name
        if direct.exists():
            return direct

        stem = Path(image_name).stem
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    # Return first root candidate for clear warning messages.
    return image_roots[0] / image_name


def compute_severity_scores(
    annotations_dir: Path,
    images_dir: Path,
) -> List[Dict[str, float | str]]:
    """
    Compute severity for each image:
    severity = total_damage_area / image_area
    """
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_roots = [images_dir]
    raw_dir = images_dir / "raw"
    if raw_dir.exists():
        image_roots.append(raw_dir)

    parsed = parse_annotations(annotations_dir=annotations_dir, images_dir=None)

    rows: List[Dict[str, float | str]] = []

    for image_name, boxes in sorted(parsed.items()):
        img_path = resolve_image_path(image_roots, image_name)
        if not img_path.exists():
            print(f"[WARN] Skipping missing image: {image_name}")
            continue

        try:
            total_img_area = image_area(img_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        if total_img_area <= 0:
            print(f"[WARN] Invalid image area for: {image_name}")
            continue

        total_damage_area = sum(bbox_area(box) for box in boxes)
        severity = total_damage_area / total_img_area

        rows.append(
            {
                "image_name": image_name,
                "severity": round(severity, 6),
            }
        )

    return rows


def save_scores_csv(rows: List[Dict[str, float | str]], output_csv: Path) -> None:
    """Save severity rows to CSV with columns [image_name, severity]."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "severity"])
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3: Compute damage severity per image from XML annotations"
    )
    parser.add_argument("--annotations-dir", default="dataset/annotations")
    parser.add_argument("--images-dir", default="dataset/images")
    parser.add_argument("--output-csv", default="data/scored.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    rows = compute_severity_scores(
        annotations_dir=Path(args.annotations_dir),
        images_dir=Path(args.images_dir),
    )

    save_scores_csv(rows, Path(args.output_csv))

    print(f"Images scored: {len(rows)}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
