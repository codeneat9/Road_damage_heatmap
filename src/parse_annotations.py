from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET


@dataclass
class BoundingBox:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class AnnotationRecord:
    image_name: str
    xml_name: str
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def _safe_int(element: ET.Element | None, tag: str) -> int:
    """Read an integer child field from an XML element with clear errors."""
    if element is None:
        raise ValueError(f"Missing required element: {tag}")

    value_text = element.findtext(tag)
    if value_text is None:
        raise ValueError(f"Missing required field: {tag}")

    try:
        return int(float(value_text))
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {tag}: {value_text}") from exc


def parse_pascal_voc_xml(xml_path: Path) -> Dict[str, object]:
    """
    Parse a Pascal VOC XML file and return image metadata with all damage boxes.

    Returns a dictionary with keys:
    - image_name: image filename from XML or guessed from XML filename
    - boxes: List[BoundingBox]
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.findtext("filename")
    if not image_name:
        image_name = f"{xml_path.stem}.jpg"

    boxes: List[BoundingBox] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="damage").strip() or "damage"
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = _safe_int(bndbox, "xmin")
        ymin = _safe_int(bndbox, "ymin")
        xmax = _safe_int(bndbox, "xmax")
        ymax = _safe_int(bndbox, "ymax")

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append(
            BoundingBox(
                label=label,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return {"image_name": image_name, "boxes": boxes}


def parse_all_annotations(annotations_dir: Path) -> Dict[str, List[BoundingBox]]:
    """Parse all XML annotation files and return image -> bounding boxes mapping."""
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    results: Dict[str, List[BoundingBox]] = {}
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        parsed = parse_pascal_voc_xml(xml_path)
        results[str(parsed["image_name"])] = parsed["boxes"]  # type: ignore[index]

    return results


def flatten_annotations(annotations_dir: Path) -> List[AnnotationRecord]:
    """Flatten all parsed annotations into row-based records for CSV storage."""
    rows: List[AnnotationRecord] = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        parsed = parse_pascal_voc_xml(xml_path)
        image_name = str(parsed["image_name"])
        boxes = parsed["boxes"]

        for box in boxes:  # type: ignore[assignment]
            rows.append(
                AnnotationRecord(
                    image_name=image_name,
                    xml_name=xml_path.name,
                    label=box.label,
                    xmin=box.xmin,
                    ymin=box.ymin,
                    xmax=box.xmax,
                    ymax=box.ymax,
                )
            )

    return rows


def save_records_to_csv(rows: List[AnnotationRecord], output_csv: Path) -> None:
    """Save flattened annotation rows to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "xml_name",
                "label",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STEP 1: Parse Road Damage Pascal VOC XML annotations"
    )
    parser.add_argument(
        "--annotations-dir",
        default="dataset/annotations",
        help="Path to Pascal VOC XML annotations directory",
    )
    parser.add_argument(
        "--output-csv",
        default="data/parsed_annotations.csv",
        help="Where to save parsed annotation rows",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    annotations_dir = Path(args.annotations_dir)
    output_csv = Path(args.output_csv)

    rows = flatten_annotations(annotations_dir)
    save_records_to_csv(rows, output_csv)

    print(f"Parsed XML files from: {annotations_dir}")
    print(f"Total bounding boxes: {len(rows)}")
    print(f"Saved parsed rows to: {output_csv}")


if __name__ == "__main__":
    main()
