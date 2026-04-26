from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ImageRecord:
    xml_name: str
    image_name: str
    image_path: str
    image_stem: str
    image_hash: str
    width: int
    height: int
    box_count: int
    primary_label: str
    labels: str
    is_duplicate: int
    canonical_image_path: str


DEFAULT_LABEL_MAP = {
    "crack": "crack",
    "damage": "damage",
    "pothole": "pothole",
    "pothole_water": "pothole_water",
    "pothole_water_m": "pothole_water",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_xml_metadata(xml_path: Path) -> Tuple[str, int, int, List[str], int]:
    """Extract filename, image size, object labels, and valid box count from Pascal VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.findtext("filename")
    if not image_name:
        image_name = f"{xml_path.stem}.jpg"

    width = 0
    height = 0
    size_node = root.find("size")
    if size_node is not None:
        width_text = size_node.findtext("width", default="0")
        height_text = size_node.findtext("height", default="0")
        width = int(float(width_text)) if width_text else 0
        height = int(float(height_text)) if height_text else 0

    labels: List[str] = []
    box_count = 0

    for obj in root.findall("object"):
        label = obj.findtext("name", default="damage").strip().lower() or "damage"
        labels.append(label)

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = bndbox.findtext("xmin")
        ymin = bndbox.findtext("ymin")
        xmax = bndbox.findtext("xmax")
        ymax = bndbox.findtext("ymax")

        if not all([xmin, ymin, xmax, ymax]):
            continue

        ixmin = int(float(xmin))
        iymin = int(float(ymin))
        ixmax = int(float(xmax))
        iymax = int(float(ymax))

        if ixmax <= ixmin or iymax <= iymin:
            continue

        box_count += 1

    return image_name, width, height, labels, box_count


def find_image_by_name_or_stem(images_index: Dict[str, Path], stem_index: Dict[str, Path], image_name: str) -> Path | None:
    name_key = image_name.lower()
    if name_key in images_index:
        return images_index[name_key]

    stem_key = Path(image_name).stem.lower()
    return stem_index.get(stem_key)


def build_image_indexes(images_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    images_index: Dict[str, Path] = {}
    stem_index: Dict[str, Path] = {}

    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        images_index.setdefault(image_path.name.lower(), image_path)
        stem_index.setdefault(image_path.stem.lower(), image_path)

    return images_index, stem_index


def normalize_primary_label(labels: List[str], label_map: Dict[str, str]) -> str:
    if not labels:
        return "unknown"

    freq: Dict[str, int] = {}
    for label in labels:
        normalized = label_map.get(label, label)
        freq[normalized] = freq.get(normalized, 0) + 1

    # Choose the most frequent label in the image.
    return sorted(freq.items(), key=lambda item: (-item[1], item[0]))[0][0]


def split_records(records: List[ImageRecord], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[ImageRecord]]:
    """Create deterministic, per-class splits using deduplicated records only."""
    from random import Random

    by_label: Dict[str, List[ImageRecord]] = {}
    for rec in records:
        by_label.setdefault(rec.primary_label, []).append(rec)

    rng = Random(seed)
    train: List[ImageRecord] = []
    val: List[ImageRecord] = []
    test: List[ImageRecord] = []

    for _, group in sorted(by_label.items()):
        items = list(group)
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if n >= 3 and n_train == 0:
            n_train = 1
        if n >= 4 and n_val == 0:
            n_val = 1

        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    return {"train": train, "val": val, "test": test}


def write_csv(path: Path, records: List[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "xml_name",
        "image_name",
        "image_path",
        "image_stem",
        "image_hash",
        "width",
        "height",
        "box_count",
        "primary_label",
        "labels",
        "is_duplicate",
        "canonical_image_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "xml_name": rec.xml_name,
                "image_name": rec.image_name,
                "image_path": rec.image_path,
                "image_stem": rec.image_stem,
                "image_hash": rec.image_hash,
                "width": rec.width,
                "height": rec.height,
                "box_count": rec.box_count,
                "primary_label": rec.primary_label,
                "labels": rec.labels,
                "is_duplicate": rec.is_duplicate,
                "canonical_image_path": rec.canonical_image_path,
            })


def prepare_dataset(
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, object]:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    if not images_dir.exists():
        fallback = images_dir / "raw"
        if fallback.exists():
            images_dir = fallback
        else:
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be > 0 and train_ratio + val_ratio < 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = dict(DEFAULT_LABEL_MAP)
    images_index, stem_index = build_image_indexes(images_dir)

    all_records: List[ImageRecord] = []
    missing_images = 0
    parse_errors = 0

    hash_to_canonical: Dict[str, str] = {}

    for xml_path in sorted(annotations_dir.glob("*.xml")):
        try:
            image_name, width, height, labels, box_count = parse_xml_metadata(xml_path)
        except Exception:
            parse_errors += 1
            continue

        image_path = find_image_by_name_or_stem(images_index, stem_index, image_name)
        if image_path is None:
            missing_images += 1
            continue

        image_hash = sha256_file(image_path)
        canonical_path = hash_to_canonical.setdefault(image_hash, str(image_path))
        is_duplicate = 0 if canonical_path == str(image_path) else 1

        normalized_labels = [label_map.get(label, label) for label in labels]
        primary_label = normalize_primary_label(labels, label_map)

        all_records.append(
            ImageRecord(
                xml_name=xml_path.name,
                image_name=image_name,
                image_path=str(image_path),
                image_stem=image_path.stem,
                image_hash=image_hash,
                width=width,
                height=height,
                box_count=box_count,
                primary_label=primary_label,
                labels="|".join(normalized_labels),
                is_duplicate=is_duplicate,
                canonical_image_path=canonical_path,
            )
        )

    deduplicated: List[ImageRecord] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0

    for rec in all_records:
        if rec.image_hash in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(rec.image_hash)
        deduplicated.append(rec)

    splits = split_records(
        records=deduplicated,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    write_csv(output_dir / "dataset_manifest_all.csv", all_records)
    write_csv(output_dir / "dataset_manifest_deduplicated.csv", deduplicated)
    write_csv(output_dir / "train.csv", splits["train"])
    write_csv(output_dir / "val.csv", splits["val"])
    write_csv(output_dir / "test.csv", splits["test"])

    quality_report = {
        "annotations_dir": str(annotations_dir),
        "images_dir": str(images_dir),
        "total_xml": len(list(annotations_dir.glob("*.xml"))),
        "records_with_images": len(all_records),
        "parse_errors": parse_errors,
        "missing_images": missing_images,
        "duplicate_records_removed": duplicate_count,
        "unique_records": len(deduplicated),
        "split_counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "split_ratios_requested": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1 - train_ratio - val_ratio, 4),
        },
        "labels_after_normalization": sorted({rec.primary_label for rec in deduplicated}),
    }

    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(quality_report, handle, indent=2)

    return quality_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and clean dataset for training")
    parser.add_argument("--annotations-dir", default="dataset/annotations")
    parser.add_argument("--images-dir", default="dataset/images")
    parser.add_argument("--output-dir", default="data/prepared")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    report = prepare_dataset(
        annotations_dir=Path(args.annotations_dir),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("Dataset preparation complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
