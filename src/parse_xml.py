from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def _read_int(parent: ET.Element, tag: str, xml_path: Path) -> int:
    """Read an integer value from a child tag with clear error context."""
    text_value = parent.findtext(tag)
    if text_value is None:
        raise ValueError(f"Missing '{tag}' in {xml_path.name}")

    try:
        return int(float(text_value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value for '{tag}' in {xml_path.name}: {text_value}"
        ) from exc


def parse_xml_file(xml_path: Path) -> tuple[str, List[BoundingBox]]:
    """
    Parse one Pascal VOC XML file and return:
    - image name
    - list of bounding boxes
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.findtext("filename")
    if not image_name:
        # Fallback when filename is missing in XML
        image_name = f"{xml_path.stem}.jpg"

    boxes: List[BoundingBox] = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = _read_int(bndbox, "xmin", xml_path)
        ymin = _read_int(bndbox, "ymin", xml_path)
        xmax = _read_int(bndbox, "xmax", xml_path)
        ymax = _read_int(bndbox, "ymax", xml_path)

        # Skip invalid/inverted boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append(BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    return image_name, boxes


def parse_annotations(
    annotations_dir: Path,
    images_dir: Path | None = None,
) -> Dict[str, List[Dict[str, int]]]:
    """
    Parse all Pascal VOC XML files in a directory.

    Returns dictionary in required format:
    {
        image_name: [
            {"xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...},
            ...
        ]
    }
    """
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    parsed: Dict[str, List[Dict[str, int]]] = {}

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        print(f"Warning: No XML files found in {annotations_dir}")

    for xml_path in xml_files:
        try:
            image_name, boxes = parse_xml_file(xml_path)
        except ET.ParseError as exc:
            print(f"Warning: Could not parse {xml_path.name}: {exc}")
            continue
        except ValueError as exc:
            print(f"Warning: Skipping invalid data in {xml_path.name}: {exc}")
            continue

        if images_dir is not None:
            image_path = images_dir / image_name
            if not image_path.exists():
                print(
                    f"Warning: Image '{image_name}' referenced by {xml_path.name} "
                    "was not found in images directory"
                )

        parsed[image_name] = [
            {
                "xmin": box.xmin,
                "ymin": box.ymin,
                "xmax": box.xmax,
                "ymax": box.ymax,
            }
            for box in boxes
        ]

    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 1: Parse Pascal VOC XML annotations")
    parser.add_argument(
        "--annotations-dir",
        default="dataset/annotations",
        help="Directory containing Pascal VOC XML files",
    )
    parser.add_argument(
        "--images-dir",
        default="dataset/images",
        help="Directory containing dataset images",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    annotations_dir = Path(args.annotations_dir)
    images_dir = Path(args.images_dir)

    parsed = parse_annotations(annotations_dir=annotations_dir, images_dir=images_dir)

    total_boxes = sum(len(boxes) for boxes in parsed.values())
    print(f"Parsed image entries: {len(parsed)}")
    print(f"Total bounding boxes: {total_boxes}")


if __name__ == "__main__":
    main()
