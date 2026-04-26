from __future__ import annotations

import argparse
import hashlib
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import cv2
from kaggle.api.kaggle_api_extended import KaggleApi


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["crack", "damage", "pothole", "pothole_water", "pothole_water_m"]


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_existing_hashes(images_root: Path) -> set[str]:
    hashes: set[str] = set()
    for img in images_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
            hashes.add(file_hash(img))
    return hashes


def with_retries(action_name: str, fn, retries: int = 5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait_seconds = min(2**attempt, 20)
            print(f"[WARN] {action_name} failed (attempt {attempt}/{retries}): {exc}")
            if attempt < retries:
                time.sleep(wait_seconds)
    raise RuntimeError(f"{action_name} failed after {retries} retries") from last_exc


def find_next_numeric_stem(annotations_dir: Path) -> int:
    max_id = 0
    for xml_path in annotations_dir.glob("*.xml"):
        try:
            number = int(xml_path.stem)
        except ValueError:
            continue
        max_id = max(max_id, number)
    return max_id + 1


def yolo_to_voc_boxes(label_path: Path, img_w: int, img_h: int) -> List[Tuple[str, int, int, int, int]]:
    boxes: List[Tuple[str, int, int, int, int]] = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "damage"

        xmin = int(round((xc - bw / 2.0) * img_w))
        ymin = int(round((yc - bh / 2.0) * img_h))
        xmax = int(round((xc + bw / 2.0) * img_w))
        ymax = int(round((yc + bh / 2.0) * img_h))

        xmin = max(0, min(xmin, img_w - 1))
        ymin = max(0, min(ymin, img_h - 1))
        xmax = max(0, min(xmax, img_w - 1))
        ymax = max(0, min(ymax, img_h - 1))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append((cls_name, xmin, ymin, xmax, ymax))

    return boxes


def write_voc_xml(xml_path: Path, filename: str, img_w: int, img_h: int, boxes: List[Tuple[str, int, int, int, int]]) -> None:
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"

    name = ET.SubElement(annotation, "filename")
    name.text = filename

    size = ET.SubElement(annotation, "size")
    w_node = ET.SubElement(size, "width")
    w_node.text = str(img_w)
    h_node = ET.SubElement(size, "height")
    h_node.text = str(img_h)
    d_node = ET.SubElement(size, "depth")
    d_node.text = "3"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for cls_name, xmin, ymin, xmax, ymax in boxes:
        obj = ET.SubElement(annotation, "object")
        obj_name = ET.SubElement(obj, "name")
        obj_name.text = cls_name

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def sync_dataset(
    dataset_ref: str,
    annotations_dir: Path,
    images_dir: Path,
    max_new: int,
) -> Dict[str, int | str]:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)

    # Keep new images in raw folder to stay compatible with current project layout.
    image_output_dir = images_dir / "raw"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Listing files from Kaggle dataset: {dataset_ref}")

    existing_hashes = collect_existing_hashes(images_dir)
    image_files_seen: set[str] = set()

    temp_dir = Path("data/tmp_kaggle_sync")
    temp_dir.mkdir(parents=True, exist_ok=True)

    copied_pairs = 0
    skipped_duplicate_images = 0
    skipped_missing_image_match = 0
    skipped_invalid_labels = 0
    attempted_pairs = 0
    pages_scanned = 0
    remote_label_files_seen = 0

    next_id = find_next_numeric_stem(annotations_dir)

    page_token = None

    while copied_pairs < max_new:
        page = with_retries(
            "dataset_list_files",
            lambda: api.dataset_list_files(dataset_ref, page_size=100, page_token=page_token),
        )
        files = page.files or []
        if not files:
            break

        pages_scanned += 1

        for remote_file in files:
            rel_path = remote_file.name.replace("\\", "/")
            ext = Path(rel_path).suffix.lower()

            if ext in IMAGE_EXTENSIONS:
                image_files_seen.add(rel_path)
                continue

            if not (rel_path.lower().endswith(".txt") and "/labels/" in rel_path):
                continue

            remote_label_files_seen += 1

            img_base = rel_path.replace("/labels/", "/images/")
            stem = Path(img_base).stem

            img_rel = None
            for image_ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = str(Path(img_base).with_suffix(image_ext)).replace("\\", "/")
                if candidate in image_files_seen:
                    img_rel = candidate
                    break

            if img_rel is None:
                skipped_missing_image_match += 1
                continue

            attempted_pairs += 1

            local_img = temp_dir / f"{stem}{Path(img_rel).suffix.lower()}"
            local_label = temp_dir / f"{stem}.txt"

            with_retries(
                "dataset_download_file(image)",
                lambda: api.dataset_download_file(
                    dataset_ref, img_rel, path=str(temp_dir), force=True, quiet=True
                ),
            )
            with_retries(
                "dataset_download_file(label)",
                lambda: api.dataset_download_file(
                    dataset_ref, rel_path, path=str(temp_dir), force=True, quiet=True
                ),
            )

            if not local_img.exists() or not local_label.exists():
                skipped_missing_image_match += 1
                continue

            img_hash = file_hash(local_img)
            if img_hash in existing_hashes:
                skipped_duplicate_images += 1
                local_img.unlink(missing_ok=True)
                local_label.unlink(missing_ok=True)
                continue

            image = cv2.imread(str(local_img))
            if image is None:
                skipped_invalid_labels += 1
                local_img.unlink(missing_ok=True)
                local_label.unlink(missing_ok=True)
                continue

            img_h, img_w = image.shape[:2]

            try:
                boxes = yolo_to_voc_boxes(local_label, img_w=img_w, img_h=img_h)
            except Exception:
                skipped_invalid_labels += 1
                local_img.unlink(missing_ok=True)
                local_label.unlink(missing_ok=True)
                continue

            new_stem = str(next_id)
            next_id += 1

            dst_img = image_output_dir / f"{new_stem}{local_img.suffix.lower()}"
            dst_xml = annotations_dir / f"{new_stem}.xml"

            shutil.copy2(local_img, dst_img)
            write_voc_xml(dst_xml, filename=dst_img.name, img_w=img_w, img_h=img_h, boxes=boxes)

            local_img.unlink(missing_ok=True)
            local_label.unlink(missing_ok=True)

            existing_hashes.add(img_hash)
            copied_pairs += 1

            if copied_pairs >= max_new:
                break

        page_token = getattr(page, "next_page_token", None)
        if not page_token:
            break

    return {
        "dataset_ref": dataset_ref,
        "pages_scanned": pages_scanned,
        "remote_label_files_seen": remote_label_files_seen,
        "attempted_pairs": attempted_pairs,
        "copied_pairs": copied_pairs,
        "skipped_duplicate_images": skipped_duplicate_images,
        "skipped_missing_image_match": skipped_missing_image_match,
        "skipped_invalid_labels": skipped_invalid_labels,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Kaggle dataset and merge new image/XML pairs into workspace dataset"
    )
    parser.add_argument(
        "--dataset-ref",
        default="aliabdelmenam/rdd-2022",
        help="Kaggle dataset reference in owner/dataset format",
    )
    parser.add_argument("--annotations-dir", default="dataset/annotations")
    parser.add_argument("--images-dir", default="dataset/images")
    parser.add_argument(
        "--max-new",
        type=int,
        default=300,
        help="Maximum number of new image/XML pairs to import",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    result = sync_dataset(
        dataset_ref=args.dataset_ref,
        annotations_dir=Path(args.annotations_dir),
        images_dir=Path(args.images_dir),
        max_new=args.max_new,
    )

    print("Sync finished")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
