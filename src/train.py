from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import os
import xml.etree.ElementTree as ET

import cv2
from dotenv import load_dotenv
import yaml


@dataclass
class VocObject:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class VocRecord:
    image_name: str
    width: int
    height: int
    objects: List[VocObject]


def _safe_int(parent: ET.Element | None, tag: str) -> int:
    if parent is None:
        raise ValueError(f"Missing parent element for {tag}")
    text = parent.findtext(tag)
    if text is None:
        raise ValueError(f"Missing required field: {tag}")
    return int(float(text))


def parse_voc_xml(xml_path: Path) -> VocRecord:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.findtext("filename")
    if not image_name:
        image_name = f"{xml_path.stem}.jpg"

    size = root.find("size")
    width = _safe_int(size, "width") if size is not None and size.findtext("width") else 0
    height = _safe_int(size, "height") if size is not None and size.findtext("height") else 0

    objects: List[VocObject] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="damage").strip() or "damage"
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = _safe_int(bnd, "xmin")
        ymin = _safe_int(bnd, "ymin")
        xmax = _safe_int(bnd, "xmax")
        ymax = _safe_int(bnd, "ymax")

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append(
            VocObject(
                label=label,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return VocRecord(
        image_name=image_name,
        width=width,
        height=height,
        objects=objects,
    )


def find_image_path(images_dir: Path, image_name: str) -> Path | None:
    direct = images_dir / image_name
    if direct.exists():
        return direct

    stem = Path(image_name).stem
    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    for ext in exts:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    for ext in exts:
        matches = list(images_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    return None


def to_yolo_line(
    class_id: int,
    box: VocObject,
    width: int,
    height: int,
) -> str:
    xmin = max(0, min(box.xmin, width - 1))
    xmax = max(0, min(box.xmax, width - 1))
    ymin = max(0, min(box.ymin, height - 1))
    ymax = max(0, min(box.ymax, height - 1))

    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    x_center = xmin + bw / 2.0
    y_center = ymin + bh / 2.0

    return (
        f"{class_id} "
        f"{x_center / width:.6f} "
        f"{y_center / height:.6f} "
        f"{bw / width:.6f} "
        f"{bh / height:.6f}"
    )


def _ensure_dimensions(record: VocRecord, image_path: Path) -> Tuple[int, int]:
    if record.width > 0 and record.height > 0:
        return record.width, record.height

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w = image.shape[:2]
    return w, h


def _copy_samples_to_split(
    split_name: str,
    split_samples: List[Tuple[VocRecord, Path]],
    output_dir: Path,
    class_to_id: Dict[str, int],
) -> None:
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for record, image_path in split_samples:
        width, height = _ensure_dimensions(record, image_path)

        out_image = images_out / image_path.name
        shutil.copy2(image_path, out_image)

        label_lines: List[str] = []
        for obj in record.objects:
            class_id = class_to_id[obj.label]
            label_lines.append(to_yolo_line(class_id, obj, width, height))

        label_file = labels_out / f"{Path(image_path.name).stem}.txt"
        label_file.write_text("\n".join(label_lines), encoding="utf-8")


def _collect_samples_from_dirs(
    annotations_dir: Path,
    images_dir: Path,
) -> Tuple[List[Tuple[VocRecord, Path]], int, int]:
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found in: {annotations_dir}")

    samples: List[Tuple[VocRecord, Path]] = []
    skipped_missing = 0
    skipped_empty = 0

    for xml_path in xml_files:
        record = parse_voc_xml(xml_path)
        if not record.objects:
            skipped_empty += 1
            continue

        image_path = find_image_path(images_dir, record.image_name)
        if image_path is None:
            skipped_missing += 1
            continue

        samples.append((record, image_path))

    return samples, skipped_missing, skipped_empty


def prepare_yolo_dataset_with_existing_split(
    dataset_dir: Path,
    output_dir: Path,
) -> Tuple[Path, Dict[str, int]]:
    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    all_labels = set()
    split_samples: Dict[str, List[Tuple[VocRecord, Path]]] = {"train": [], "val": [], "test": []}
    skipped_missing = 0
    skipped_empty = 0
    total_xml = 0

    for src_split, dst_split in split_map.items():
        ann_dir = dataset_dir / src_split / "annotations"
        img_dir = dataset_dir / src_split / "images"
        if not ann_dir.exists() or not img_dir.exists():
            raise FileNotFoundError(
                f"Expected split directories not found: {ann_dir} and {img_dir}"
            )

        samples, split_missing, split_empty = _collect_samples_from_dirs(ann_dir, img_dir)
        split_samples[dst_split] = samples
        skipped_missing += split_missing
        skipped_empty += split_empty
        total_xml += len(list(ann_dir.glob("*.xml")))

        for record, _ in samples:
            for obj in record.objects:
                all_labels.add(obj.label)

    if not split_samples["train"] or not split_samples["val"]:
        raise ValueError("Pre-split dataset must contain non-empty train and valid splits.")

    class_names = sorted(all_labels)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    if output_dir.exists():
        shutil.rmtree(output_dir)

    _copy_samples_to_split("train", split_samples["train"], output_dir, class_to_id)
    _copy_samples_to_split("val", split_samples["val"], output_dir, class_to_id)
    if split_samples["test"]:
        _copy_samples_to_split("test", split_samples["test"], output_dir, class_to_id)

    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    if split_samples["test"]:
        payload["test"] = "images/test"

    data_yaml_path = output_dir / "dataset.yaml"
    data_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    stats = {
        "total_xml": total_xml,
        "usable_samples": len(split_samples["train"]) + len(split_samples["val"]) + len(split_samples["test"]),
        "train_samples": len(split_samples["train"]),
        "val_samples": len(split_samples["val"]),
        "test_samples": len(split_samples["test"]),
        "classes": len(class_names),
        "skipped_missing_images": skipped_missing,
        "skipped_empty_annotations": skipped_empty,
    }
    return data_yaml_path, stats


def has_presplit_rdd_dirs(dataset_dir: Path) -> bool:
    required = [
        dataset_dir / "train" / "annotations",
        dataset_dir / "train" / "images",
        dataset_dir / "valid" / "annotations",
        dataset_dir / "valid" / "images",
    ]
    return all(path.exists() for path in required)


def prepare_yolo_dataset(
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    val_split: float,
    seed: int,
) -> Tuple[Path, Dict[str, int]]:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found in: {annotations_dir}")

    samples, skipped_missing, skipped_empty = _collect_samples_from_dirs(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
    )
    label_set = set()
    for record, _ in samples:
        for obj in record.objects:
            label_set.add(obj.label)

    if not samples:
        raise ValueError("No training samples were prepared. Check your dataset paths.")

    class_names = sorted(label_set)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    rng = random.Random(seed)
    rng.shuffle(samples)

    split_idx = int(len(samples) * (1 - val_split))
    split_idx = min(max(split_idx, 1), len(samples) - 1) if len(samples) > 1 else 1

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] if len(samples) > 1 else samples

    if output_dir.exists():
        shutil.rmtree(output_dir)

    _copy_samples_to_split("train", train_samples, output_dir, class_to_id)
    _copy_samples_to_split("val", val_samples, output_dir, class_to_id)

    data_yaml_path = output_dir / "dataset.yaml"
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    data_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    stats = {
        "total_xml": len(xml_files),
        "usable_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "classes": len(class_names),
        "skipped_missing_images": skipped_missing,
        "skipped_empty_annotations": skipped_empty,
    }
    return data_yaml_path, stats


def maybe_download_rdd_from_roboflow(
    dataset_dir: Path,
    workspace: str,
    project: str,
    version: int,
) -> Path:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY is not set. Put it in your environment or .env file."
        )

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ImportError(
            "roboflow package not installed. Run: pip install roboflow"
        ) from exc

    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    rf_version = rf_project.version(version)
    downloaded = rf_version.download("voc", location=str(dataset_dir))

    return Path(downloaded.location)


def train_yolo(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    project_dir: Path,
    run_name: str,
    use_cpu: bool,
) -> None:
    print("Starting Ultralytics training setup...", flush=True)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics package not installed. Run: pip install ultralytics"
        ) from exc

    print(f"Loading model definition: {model_name}", flush=True)
    model = YOLO(model_name)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "project": str(project_dir),
        "name": run_name,
        "workers": 0,
        "verbose": True,
    }
    if use_cpu:
        train_kwargs["device"] = "cpu"

    print("Launching training loop...", flush=True)
    model.train(**train_kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on RDD (Pascal VOC XML) with safe API token usage"
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root directory")
    parser.add_argument(
        "--annotations-dir",
        default=None,
        help="Override annotations directory (default: <dataset-dir>/annotations)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Override images directory (default: <dataset-dir>/images)",
    )
    parser.add_argument("--output-dir", default="data/yolo", help="YOLO prepared dataset output")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--project-dir", default="models", help="Training output project directory")
    parser.add_argument("--run-name", default="rdd-yolo", help="Training run name")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument(
        "--preserve-rdd-split",
        action="store_true",
        help=(
            "Use official pre-split folders if dataset has train/valid/test with annotations+images"
        ),
    )

    parser.add_argument(
        "--use-roboflow",
        action="store_true",
        help="Download/update dataset from Roboflow using ROBOFLOW_API_KEY",
    )
    parser.add_argument("--rf-workspace", default=None, help="Roboflow workspace slug")
    parser.add_argument("--rf-project", default=None, help="Roboflow project slug")
    parser.add_argument("--rf-version", type=int, default=None, help="Roboflow version number")

    return parser


def main() -> None:
    load_dotenv()
    args = build_arg_parser().parse_args()

    dataset_dir = Path(args.dataset_dir)

    if args.use_roboflow:
        if not args.rf_workspace or not args.rf_project or args.rf_version is None:
            raise ValueError(
                "When --use-roboflow is set, provide --rf-workspace, --rf-project, and --rf-version"
            )
        downloaded_dir = maybe_download_rdd_from_roboflow(
            dataset_dir=dataset_dir,
            workspace=args.rf_workspace,
            project=args.rf_project,
            version=args.rf_version,
        )
        print(f"Downloaded dataset to: {downloaded_dir}")

    output_dir = Path(args.output_dir)
    if args.preserve_rdd_split or has_presplit_rdd_dirs(dataset_dir):
        data_yaml, stats = prepare_yolo_dataset_with_existing_split(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
        print("Detected/selected pre-split RDD dataset (train/valid/test)")
    else:
        annotations_dir = Path(args.annotations_dir) if args.annotations_dir else dataset_dir / "annotations"
        images_dir = Path(args.images_dir) if args.images_dir else dataset_dir / "images"

        data_yaml, stats = prepare_yolo_dataset(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            output_dir=output_dir,
            val_split=args.val_split,
            seed=args.seed,
        )
        print("Using flat dataset directory and random train/val split")

    print("Dataset prepared for YOLO training")
    for k, v in stats.items():
        print(f"- {k}: {v}")
    print(f"- data_yaml: {data_yaml}")

    train_yolo(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project_dir=Path(args.project_dir),
        run_name=args.run_name,
        use_cpu=args.cpu,
    )

    print("Training complete")


if __name__ == "__main__":
    main()
