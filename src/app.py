from __future__ import annotations

import json
import base64
import os
from pathlib import Path
import sys
from typing import List, Tuple

import folium
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from folium.plugins import Fullscreen, HeatMap, MiniMap
from streamlit_folium import st_folium

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from map_utils import quality_to_color
from parse_xml import parse_xml_file
from route_score import load_damage_points, score_route
from routing import geocode_place_name, get_route
from detect_cv import process_image as process_cv_image


DATA_DIR = Path("data")
DATASET_IMAGES_DIR = Path("dataset/images")
ANNOTATIONS_DIR = Path("dataset/annotations")
CV_OUTPUT_DIR = Path("data/cv_output")
SCORED_CSV = DATA_DIR / "scored.csv"
ROUTE_POINTS_JSON = DATA_DIR / "route_points.json"
ROUTE_SCORES_JSON = DATA_DIR / "route_scores.json"
GEO_SCORED_CSV = DATA_DIR / "geo_scored.csv"

# Cache encoded image URIs to keep map building smooth.
IMAGE_DATA_URI_CACHE: dict[str, str] = {}
HOVER_IMAGE_POOL_SIZE = 120
HOVER_MAX_POINTS = 80
EXTREME_RED_PERCENTILE = 85
EXTREME_RED_FALLBACK_PERCENTILE = 75
EXTREME_RED_NORMALIZED_MIN = 0.72


def load_env_files() -> None:
    """Load API keys from common env files if present."""
    load_dotenv(Path(".env"), override=False)
    load_dotenv(Path(".env.local"), override=False)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --brand-ink: #0b1220;
            --brand-muted: #516074;
            --brand-accent: #2563eb;
            --brand-accent-2: #14b8a6;
            --brand-accent-3: #f97316;
            --brand-bg: #eef5ff;
            --card-bg: rgba(255, 255, 255, 0.84);
            --card-border: rgba(148, 163, 184, 0.22);
            --shadow-lg: 0 24px 70px rgba(15, 23, 42, 0.12);
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--brand-ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 12%, rgba(59, 130, 246, 0.18) 0%, rgba(59, 130, 246, 0) 28%),
                radial-gradient(circle at 92% 8%, rgba(20, 184, 166, 0.16) 0%, rgba(20, 184, 166, 0) 26%),
                radial-gradient(circle at 50% 100%, rgba(249, 115, 22, 0.08) 0%, rgba(249, 115, 22, 0) 35%),
                linear-gradient(180deg, #f9fbff 0%, #edf4ff 100%);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(11, 18, 32, 0.04) 0%, rgba(37, 99, 235, 0.03) 100%),
                linear-gradient(180deg, #f7faff 0%, #eef4ff 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.7);
        }

        [data-testid="stSidebar"] .stTextInput > label,
        [data-testid="stSidebar"] .stSlider > label {
            font-size: 0.95rem;
            font-weight: 700;
            color: #243245;
            letter-spacing: 0.01em;
        }

        [data-testid="stSidebar"] .stTextInput > div > div > input {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.3);
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
            padding: 0.6rem 0.85rem;
            font-size: 1rem;
            color: #0f172a;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            font-weight: 700;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 52%, #14b8a6 100%);
            border: 0;
            color: #ffffff;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, #2563eb 0%, #14b8a6 100%);
        }

        .sidebar-shell {
            padding: 14px 14px 12px;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background:
                radial-gradient(circle at 100% 0%, rgba(37, 99, 235, 0.16) 0%, rgba(37, 99, 235, 0) 42%),
                linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(241,246,255,0.92) 100%);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 14px;
        }

        .sidebar-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: #0b1220;
            margin: 0;
            line-height: 1.15;
        }

        .sidebar-subtitle {
            margin-top: 6px;
            color: #516074;
            font-size: 0.87rem;
            line-height: 1.35;
        }

        .sidebar-helper {
            margin-top: 10px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            color: #1d4ed8;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.22);
            background: rgba(255, 255, 255, 0.82);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.25rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 28px;
            padding: 22px 24px 20px;
            background:
                radial-gradient(circle at 14% 20%, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0) 24%),
                radial-gradient(circle at 88% 16%, rgba(20, 184, 166, 0.16) 0%, rgba(20, 184, 166, 0) 22%),
                linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(244,248,255,0.96) 48%, rgba(236,248,255,0.96) 100%);
            box-shadow: var(--shadow-lg);
            margin-bottom: 1rem;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: auto -18px -18px auto;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(37, 99, 235, 0.2) 0%, rgba(37, 99, 235, 0) 70%);
            pointer-events: none;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding: 7px 12px;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.16);
            background: rgba(255, 255, 255, 0.72);
            color: #1d4ed8;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2.35rem, 4vw, 3.55rem);
            font-weight: 700;
            line-height: 0.98;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
            color: var(--brand-ink);
        }

        .subtitle {
            color: var(--brand-muted);
            font-size: 1.04rem;
            max-width: 860px;
            line-height: 1.55;
            margin-bottom: 0;
        }

        .card {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--card-border);
            border-radius: 22px;
            padding: 16px 16px 14px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(245, 250, 255, 0.86) 100%);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            min-height: 122px;
            backdrop-filter: blur(12px);
        }

        .card:after {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(20,184,166,0.9), rgba(249,115,22,0.9));
        }

        .card-title {
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .card-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.85rem, 2.6vw, 2.65rem);
            font-weight: 700;
            color: #0b1220;
            line-height: 1.02;
            letter-spacing: -0.04em;
        }

        .status-chip {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(37, 99, 235, 0.08);
            color: #1d4ed8;
            border: 1px solid rgba(37, 99, 235, 0.14);
        }

        .route-summary {
            margin-top: 1rem;
            margin-bottom: 0.95rem;
            padding: 16px 18px 15px;
            border-radius: 20px;
            border: 1px solid rgba(37, 99, 235, 0.16);
            background:
                radial-gradient(circle at 100% 0%, rgba(20, 184, 166, 0.14) 0%, rgba(20, 184, 166, 0) 32%),
                linear-gradient(180deg, rgba(239, 246, 255, 0.96) 0%, rgba(248, 250, 252, 0.95) 100%);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }

        .route-summary-title {
            font-size: 0.75rem;
            color: #1d4ed8;
            font-weight: 700;
            margin-bottom: 4px;
            letter-spacing: 0.14em;
        }

        .route-summary-main {
            font-size: 1.02rem;
            color: #0f172a;
            font-weight: 600;
            line-height: 1.52;
        }

        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.18rem;
            font-weight: 700;
            color: var(--brand-ink);
            margin: 0.4rem 0 0.3rem;
            letter-spacing: -0.02em;
        }

        .section-subtitle {
            color: var(--brand-muted);
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }

        .surface-panel {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 24px;
            background: rgba(255,255,255,0.78);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
            padding: 16px;
            backdrop-filter: blur(10px);
        }

        .helper-note {
            display: flex;
            gap: 10px;
            align-items: center;
            padding: 14px 16px;
            border-radius: 18px;
            border: 1px solid rgba(37, 99, 235, 0.14);
            background:
                linear-gradient(135deg, rgba(239, 246, 255, 0.96) 0%, rgba(255, 255, 255, 0.94) 100%);
            color: #1d4ed8;
            font-weight: 600;
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.08);
        }

        .helper-note strong {
            font-family: 'Space Grotesk', sans-serif;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 0.85rem 0 1rem;
        }

        @media (max-width: 1100px) {
            .metric-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 640px) {
            .metric-row {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 18px 16px;
            }

            .main-title {
                line-height: 1.05;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def simplify_route_points(route_points: List[Tuple[float, float]], max_points: int = 350) -> List[Tuple[float, float]]:
    """Downsample dense route geometry for stable rendering and scoring."""
    n = len(route_points)
    if n <= max_points:
        return route_points

    indices = []
    for i in range(max_points):
        idx = round(i * (n - 1) / (max_points - 1))
        indices.append(idx)

    # Preserve order while removing duplicates from rounding.
    unique_indices = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    return [route_points[idx] for idx in unique_indices]


def compress_segments(route_segments: List[dict]) -> List[dict]:
    """Merge consecutive segments with identical quality to reduce map draw load."""
    if not route_segments:
        return []

    merged: List[dict] = []
    current = {
        "quality": route_segments[0]["quality"],
        "coords": [route_segments[0]["start"], route_segments[0]["end"]],
        "scores": [float(route_segments[0]["score"])],
    }

    for seg in route_segments[1:]:
        quality = seg["quality"]
        if quality == current["quality"]:
            current["coords"].append(seg["end"])
            current["scores"].append(float(seg["score"]))
        else:
            merged.append(current)
            current = {
                "quality": quality,
                "coords": [seg["start"], seg["end"]],
                "scores": [float(seg["score"])],
            }

    merged.append(current)
    return merged


def filter_damage_points_for_route(
    damage_df: pd.DataFrame,
    route_segments: List[dict],
    margin_deg: float = 0.75,
) -> pd.DataFrame:
    """Keep only damage points that fall near the chosen route corridor."""
    if damage_df.empty or not route_segments:
        return damage_df.iloc[0:0]

    lat_vals = []
    lon_vals = []
    for seg in route_segments:
        lat_vals.extend([float(seg["start"][0]), float(seg["end"][0])])
        lon_vals.extend([float(seg["start"][1]), float(seg["end"][1])])

    min_lat = min(lat_vals) - margin_deg
    max_lat = max(lat_vals) + margin_deg
    min_lon = min(lon_vals) - margin_deg
    max_lon = max(lon_vals) + margin_deg

    route_bbox = damage_df[
        (damage_df["lat"] >= min_lat)
        & (damage_df["lat"] <= max_lat)
        & (damage_df["lon"] >= min_lon)
        & (damage_df["lon"] <= max_lon)
    ]

    return route_bbox.sort_values("severity", ascending=False)


def build_route_heat_data(route_segments: List[dict]) -> List[List[float]]:
    """Build weighted heat points along the route from segment-level risk scores."""
    if not route_segments:
        return []

    raw_scores = [max(0.0, float(seg.get("score", 0.0))) for seg in route_segments]
    min_score = min(raw_scores)
    max_score = max(raw_scores)

    heat_points: List[List[float]] = []
    for seg, score in zip(route_segments, raw_scores):
        s_lat, s_lon = float(seg["start"][0]), float(seg["start"][1])
        e_lat, e_lon = float(seg["end"][0]), float(seg["end"][1])
        mid_lat = (s_lat + e_lat) / 2.0
        mid_lon = (s_lon + e_lon) / 2.0

        if max_score > min_score:
            normalized = (score - min_score) / (max_score - min_score)
        else:
            normalized = 0.6

        weight = 0.2 + (0.8 * normalized)
        heat_points.append([mid_lat, mid_lon, weight])

    return heat_points


def severity_to_color(severity: float) -> str:
    if severity < 1.5:
        return "#facc15"  # yellow
    if severity < 2.5:
        return "#f97316"  # orange
    return "#dc2626"  # red


def list_dataset_images() -> List[Path]:
    if not DATASET_IMAGES_DIR.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(
        [path for path in DATASET_IMAGES_DIR.iterdir() if path.suffix.lower() in image_extensions],
        key=lambda item: item.name,
    )


def attach_damage_image_metadata(damage_df: pd.DataFrame) -> pd.DataFrame:
    """Attach representative image names to damage points using scored.csv ordering."""
    df = damage_df.copy()
    if SCORED_CSV.exists():
        scored_df = pd.read_csv(SCORED_CSV)
        if "image_name" in scored_df.columns and len(scored_df) == len(df):
            df["image_name"] = scored_df["image_name"].astype(str)
        else:
            df["image_name"] = ""
    else:
        df["image_name"] = ""
    return df


def get_preview_image_path(image_name: str) -> Path | None:
    if not image_name:
        return None
    stem = Path(image_name).stem
    candidates = [
        CV_OUTPUT_DIR / f"{stem}_annotated.jpg",
        DATASET_IMAGES_DIR / image_name,
        DATASET_IMAGES_DIR / f"{stem}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_image_comparison_paths(image_name: str) -> tuple[Path | None, Path | None]:
    """Return (original_image, annotated_image) paths for hover comparison."""
    if not image_name:
        return None, None

    stem = Path(image_name).stem
    original_candidates = [
        DATASET_IMAGES_DIR / image_name,
        DATASET_IMAGES_DIR / f"{stem}.jpg",
    ]
    annotated_candidates = [
        CV_OUTPUT_DIR / f"{stem}_annotated.jpg",
    ]

    original_path = next((p for p in original_candidates if p.exists()), None)
    annotated_path = next((p for p in annotated_candidates if p.exists()), None)

    # If CV overlay is missing, generate it on demand.
    if annotated_path is None and original_path is not None:
        try:
            _, _, _, _, _, _, box_count, saved_paths = process_cv_image(original_path)
            candidate = saved_paths.get("annotated") if isinstance(saved_paths, dict) else None
            if box_count > 0 and isinstance(candidate, Path) and candidate.exists():
                annotated_path = candidate
        except Exception:
            annotated_path = None

    # Fallback to XML overlay so the "after" image always has annotation boxes.
    if annotated_path is None and original_path is not None:
        xml_path = ANNOTATIONS_DIR / f"{stem}.xml"
        if xml_path.exists():
            try:
                _, boxes = parse_xml_file(xml_path)
            except Exception:
                boxes = []

            if boxes:
                xml_overlay_path = CV_OUTPUT_DIR / f"{stem}_xml_overlay.jpg"
                if not xml_overlay_path.exists():
                    image = cv2.imread(str(original_path))
                    if image is not None:
                        for box in boxes:
                            cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 2)
                            cv2.putText(
                                image,
                                "XML",
                                (box.xmin, max(0, box.ymin - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                        CV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(xml_overlay_path), image)

                if xml_overlay_path.exists():
                    annotated_path = xml_overlay_path

    return original_path, annotated_path


def encode_image_to_data_uri(image_path: Path, max_width: int = 220) -> str | None:
    cache_key = f"{image_path}|{max_width}"
    cached = IMAGE_DATA_URI_CACHE.get(cache_key)
    if cached is not None:
        return cached

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]
    if width > max_width:
        new_height = max(1, int(height * (max_width / width)))
        image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        return None
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"
    IMAGE_DATA_URI_CACHE[cache_key] = data_uri
    return data_uri


def get_hover_image_pool(damage_df: pd.DataFrame) -> List[str]:
    """Build a stable image pool so different route segments show different evidence images."""
    if "image_name" not in damage_df.columns:
        return []

    names = [str(name) for name in damage_df["image_name"].dropna().unique() if str(name).strip()]
    if not names:
        return []

    # Prefer images that can produce an annotated "after" view.
    with_overlay: List[str] = []
    without_overlay: List[str] = []
    for name in names:
        stem = Path(name).stem
        has_cv = (CV_OUTPUT_DIR / f"{stem}_annotated.jpg").exists()
        has_xml = (ANNOTATIONS_DIR / f"{stem}.xml").exists()
        if has_cv or has_xml:
            with_overlay.append(name)
        else:
            without_overlay.append(name)

    pool = with_overlay + without_overlay
    return pool[:HOVER_IMAGE_POOL_SIZE]


def build_segment_hover_html(
    segment: dict,
    segment_score: float,
    segment_index: int,
    damage_df: pd.DataFrame,
    image_pool: List[str],
) -> str:
    """Build HTML shown when user hovers a route segment."""
    quality = str(segment.get("quality", "moderate")).upper()
    score = float(segment_score)
    start = segment.get("start", [0.0, 0.0])
    end = segment.get("end", [0.0, 0.0])
    mid_lat = (float(start[0]) + float(end[0])) / 2.0
    mid_lon = (float(start[1]) + float(end[1])) / 2.0

    nearby_text = "No catalog evidence available"
    image_html = ""

    if not damage_df.empty:
        candidates = damage_df.copy()
        candidates["dist2"] = (candidates["lat"] - mid_lat) ** 2 + (candidates["lon"] - mid_lon) ** 2
        nearest = candidates.sort_values("dist2", ascending=True).iloc[0]
        severity = float(nearest.get("severity", 0.0))
        nearby_text = f"Nearest matched severity: {severity:.2f}"

        # Spread different images across route segments and ensure before/after are distinct.
        before_uri = None
        after_uri = None
        if image_pool:
            max_checks = min(12, len(image_pool))
            for offset in range(max_checks):
                candidate_name = image_pool[(segment_index + offset * 7) % len(image_pool)]
                original_path, annotated_path = get_image_comparison_paths(candidate_name)
                before_uri = encode_image_to_data_uri(original_path, max_width=100) if original_path is not None else None
                after_uri = encode_image_to_data_uri(annotated_path, max_width=100) if annotated_path is not None else None
                if before_uri and after_uri and before_uri != after_uri:
                    break

        if not (before_uri and after_uri):
            fallback_name = str(nearest.get("image_name", ""))
            original_path, annotated_path = get_image_comparison_paths(fallback_name)
            before_uri = encode_image_to_data_uri(original_path, max_width=100) if original_path is not None else None
            after_uri = encode_image_to_data_uri(annotated_path, max_width=100) if annotated_path is not None else None

        if before_uri and after_uri and before_uri != after_uri:
            before_img = (
                f'<div style="flex:1; min-width:0;">'
                f'<div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-bottom:2px;">Before</div>'
                f'<img src="{before_uri}" style="display:block; width:100%; border-radius:7px; border:1px solid rgba(148,163,184,0.35);" />'
                f'</div>'
            )
            after_img = (
                f'<div style="flex:1; min-width:0;">'
                f'<div style="font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-bottom:2px;">After</div>'
                f'<img src="{after_uri}" style="display:block; width:100%; border-radius:7px; border:1px solid rgba(148,163,184,0.35);" />'
                f'</div>'
            )
            image_html = (
                '<div style="display:flex; gap:6px; margin-top:7px; width:210px;">'
                + before_img
                + after_img
                + '</div>'
            )

    return f"""
        <div style="min-width: 230px; font-family: Manrope, Arial, sans-serif;">
            <div style="font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color:#64748b; font-weight:700;">Route Segment</div>
            <div style="margin-top:3px; font-size: 14px; font-weight:800; color:#0f172a;">Condition: {quality}</div>
            <div style="font-size: 12px; color:#334155; margin-top:2px;">Score: {score:.2f}/100</div>
            <div style="font-size: 12px; color:#334155; margin-top:2px;">{nearby_text}</div>
            {image_html}
            <div style="font-size: 11px; color:#64748b; margin-top:4px;">Hover other segments to inspect local condition and damage evidence.</div>
        </div>
    """


def load_rgb_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_xml_overlay(image_path: Path, image_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    xml_path = ANNOTATIONS_DIR / f"{image_path.stem}.xml"
    if not xml_path.exists():
        return image_rgb, 0

    try:
        _, boxes = parse_xml_file(xml_path)
    except Exception:
        return image_rgb, 0

    overlay = image_rgb.copy()
    for box in boxes:
        cv2.rectangle(overlay, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 2)
        cv2.putText(
            overlay,
            "XML Damage",
            (box.xmin, max(0, box.ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay, len(boxes)


def get_opencv_overlay(image_path: Path) -> tuple[np.ndarray | None, int]:
    saved_annotated = CV_OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
    if saved_annotated.exists():
        saved = cv2.imread(str(saved_annotated))
        if saved is not None:
            return cv2.cvtColor(saved, cv2.COLOR_BGR2RGB), -1

    original, _, _, _, _, annotated, box_count, _ = process_cv_image(image_path)
    if annotated is None:
        return None, 0

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), box_count


def render_image_toggle() -> None:
    images = list_dataset_images()
    if not images:
        st.info("No dataset images found for the comparison toggle.")
        return

    st.markdown('<div class="section-title">Image Validation Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Compare the original road image, OpenCV-detected candidate regions, and XML ground-truth annotations.</div>',
        unsafe_allow_html=True,
    )

    selected_image = st.selectbox(
        "Choose an image",
        options=images,
        format_func=lambda item: item.name,
        key="image_validation_select",
    )

    original_rgb = load_rgb_image(selected_image)
    if original_rgb is None:
        st.error(f"Unable to load {selected_image.name}")
        return

    opencv_image, box_count = get_opencv_overlay(selected_image)
    xml_overlay, xml_count = draw_xml_overlay(selected_image, original_rgb)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
        st.markdown('**Original**')
        st.image(original_rgb, use_container_width=True)
        st.caption(f"{selected_image.name} | raw dataset image")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
        st.markdown('**OpenCV Detected**')
        if opencv_image is not None:
            st.image(opencv_image, use_container_width=True)
            st.caption(f"Heuristic regions found: {box_count if box_count >= 0 else 'loaded from cache'}")
        else:
            st.warning("OpenCV overlay could not be generated for this image.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="surface-panel">', unsafe_allow_html=True)
        st.markdown('**XML Overlay**')
        st.image(xml_overlay, use_container_width=True)
        st.caption(f"XML boxes found: {xml_count}")
        st.markdown('</div>', unsafe_allow_html=True)


def add_map_legend(fmap: folium.Map) -> None:
        legend_html = """
        <div style="
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 9999;
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                box-shadow: 0 8px 22px rgba(2, 6, 23, 0.16);
                padding: 10px 12px;
                min-width: 180px;
                font-family: Manrope, Segoe UI, Arial, sans-serif;
        ">
            <div style="font-size:12px; font-weight:800; color:#0f172a; margin-bottom:8px; letter-spacing:0.02em;">
                Route Risk Intensity
            </div>
            <div style="height: 12px; border-radius: 999px; margin-bottom: 6px;
                                    background: linear-gradient(90deg, #facc15 0%, #f97316 52%, #dc2626 100%);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#475569;">
                <span>Low</span><span>Moderate</span><span>High</span>
            </div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))


def build_dashboard_map(
    route_segments: List[dict],
    damage_df: pd.DataFrame,
    route_quality: str,
    route_score_value: float,
    route_damage_df: pd.DataFrame,
) -> folium.Map:
    lat_vals = []
    lon_vals = []

    for seg in route_segments:
        s = seg["start"]
        e = seg["end"]
        lat_vals.extend([float(s[0]), float(e[0])])
        lon_vals.extend([float(s[1]), float(e[1])])

    if not lat_vals:
        center = [17.3850, 78.4867]
    else:
        center = [sum(lat_vals) / len(lat_vals), sum(lon_vals) / len(lon_vals)]

    fmap = folium.Map(
        location=center,
        zoom_start=8,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer("CartoDB Positron", name="Light", control=True).add_to(fmap)
    folium.TileLayer("CartoDB Voyager", name="Streets", control=True).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=True,
    ).add_to(fmap)

    route_heat_layer = folium.FeatureGroup(name="Route Heat", show=True)
    damage_points_layer = folium.FeatureGroup(name="Damage Points", show=True)
    route_line_layer = folium.FeatureGroup(name="Route Segments", show=True)
    hover_image_pool = get_hover_image_pool(damage_df)

    # Show hover cards only for extreme red heat-zones (highest local segment scores).
    segment_scores = [float(seg.get("score", 0.0)) for seg in route_segments]
    if segment_scores:
        extreme_threshold = float(np.percentile(segment_scores, EXTREME_RED_PERCENTILE))
        fallback_threshold = float(np.percentile(segment_scores, EXTREME_RED_FALLBACK_PERCENTILE))
        min_score = float(min(segment_scores))
        max_score = float(max(segment_scores))
    else:
        extreme_threshold = 0.0
        fallback_threshold = 0.0
        min_score = 0.0
        max_score = 0.0

    score_range = max(max_score - min_score, 1e-9)

    extreme_indices = [
        idx
        for idx, seg in enumerate(route_segments)
        if float(seg.get("score", 0.0)) >= extreme_threshold
        or (
            (float(seg.get("score", 0.0)) - min_score) / score_range >= EXTREME_RED_NORMALIZED_MIN
            and float(seg.get("score", 0.0)) >= fallback_threshold
        )
    ]

    # Always keep some hover targets active even when score distribution is very flat.
    if not extreme_indices and segment_scores:
        extreme_indices = [
            idx
            for idx, seg in enumerate(route_segments)
            if float(seg.get("score", 0.0)) >= fallback_threshold
        ]

    # Keep interaction smooth if too many extreme segments are selected.
    if len(extreme_indices) > HOVER_MAX_POINTS:
        step = max(1, len(extreme_indices) // HOVER_MAX_POINTS)
        extreme_indices = extreme_indices[::step][:HOVER_MAX_POINTS]

    extreme_index_set = set(extreme_indices)

    # Always render route-based risk heat so the selected route has visible severity shading.
    route_heat_data = build_route_heat_data(route_segments)
    if route_heat_data:
        HeatMap(
            route_heat_data,
            min_opacity=0.3,
            radius=24,
            blur=18,
            max_zoom=17,
            gradient={0.2: "yellow", 0.6: "orange", 1.0: "red"},
        ).add_to(route_heat_layer)

    # Overlay real route-near damage points when available.
    if not route_damage_df.empty:
        for row in route_damage_df.head(120).itertuples(index=False):
            severity_val = float(row.severity)
            folium.CircleMarker(
                location=[float(row.lat), float(row.lon)],
                radius=3 + min(severity_val * 1.4, 5),
                color=severity_to_color(severity_val),
                fill=True,
                fill_color=severity_to_color(severity_val),
                fill_opacity=0.78,
                weight=1,
                tooltip=f"Damage severity: {severity_val:.2f}",
            ).add_to(damage_points_layer)

    # Keep segment granularity so hover scores and evidence vary naturally across the route.
    for idx, seg in enumerate(route_segments):
        color = quality_to_color(str(seg["quality"]))
        coords = [seg["start"], seg["end"]]
        local_score = float(seg.get("score", 0.0))
        hover_html = None
        if idx in extreme_index_set:
            hover_html = build_segment_hover_html(seg, local_score, idx, damage_df, hover_image_pool)

        casing_kwargs = {
            "locations": [[c[0], c[1]] for c in coords],
            "color": "#ffffff",
            "weight": 10,
            "opacity": 0.42,
        }
        if hover_html is not None:
            casing_kwargs["tooltip"] = folium.Tooltip(hover_html, sticky=True, max_width=300)
        folium.PolyLine(**casing_kwargs).add_to(route_line_layer)

        line_kwargs = {
            "locations": [[c[0], c[1]] for c in coords],
            "color": color,
            "weight": 6.5,
            "opacity": 0.9,
        }
        if hover_html is not None:
            line_kwargs["tooltip"] = folium.Tooltip(hover_html, sticky=True, max_width=300)

        folium.PolyLine(**line_kwargs).add_to(route_line_layer)

        # Add a dedicated hover hit-target at segment midpoint so cards open reliably.
        if hover_html is not None:
            mid_lat = (float(seg["start"][0]) + float(seg["end"][0])) / 2.0
            mid_lon = (float(seg["start"][1]) + float(seg["end"][1])) / 2.0
            folium.CircleMarker(
                location=[mid_lat, mid_lon],
                radius=10,
                color="#dc2626",
                weight=1,
                fill=True,
                fill_color="#dc2626",
                fill_opacity=0.03,
                tooltip=folium.Tooltip(hover_html, sticky=True, max_width=300),
            ).add_to(route_line_layer)

    first_start = route_segments[0]["start"]
    last_end = route_segments[-1]["end"]

    folium.Marker(
        [first_start[0], first_start[1]],
        tooltip="Start",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(route_line_layer)

    folium.Marker(
        [last_end[0], last_end[1]],
        tooltip="Destination",
        icon=folium.Icon(color="darkred", icon="flag"),
    ).add_to(route_line_layer)

    route_heat_layer.add_to(fmap)
    damage_points_layer.add_to(fmap)
    route_line_layer.add_to(fmap)

    title_html = f"""
        <div style=\"
            margin: 10px 12px;
            display: inline-block;
            padding: 8px 10px;
            border-radius: 10px;
            background: rgba(255,255,255,0.92);
            border: 1px solid #dbeafe;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
            font-family: Manrope, Segoe UI, Arial, sans-serif;
            font-size: 14px;
            color: #0f172a;
            font-weight: 700;
        \">
          Quality: {route_quality.upper()} | Score: {route_score_value:.1f}/100
        </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    add_map_legend(fmap)

    MiniMap(toggle_display=True, position="bottomleft").add_to(fmap)
    Fullscreen(position="topleft", title="Expand", title_cancel="Exit").add_to(fmap)
    folium.LayerControl(position="topright", collapsed=True).add_to(fmap)

    if lat_vals and lon_vals:
        bounds = [[min(lat_vals), min(lon_vals)], [max(lat_vals), max(lon_vals)]]
        fmap.fit_bounds(bounds, padding=(36, 36), max_zoom=14)

    return fmap


def main() -> None:
    load_env_files()

    st.set_page_config(
        page_title="Road Damage Route Quality Dashboard",
        layout="wide",
    )

    inject_custom_css()

    if "route_analysis" not in st.session_state:
        st.session_state.route_analysis = None

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Road Damage Intelligence Lab</div>
            <div class="main-title">Road Damage Route Intelligence</div>
            <div class="subtitle">A production-style dashboard blending route risk scoring, annotation-based heatmaps, and OpenCV visual validation in a single polished workflow.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-shell">
                <div class="sidebar-title">Route Command Panel</div>
                <div class="sidebar-subtitle">Set source, destination, and radius to generate route intelligence and hover-based road insights.</div>
                <div class="sidebar-helper">Interactive Route Controls</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        source_place = st.text_input("Source address", value="", placeholder="Enter source address")
        destination_place = st.text_input(
            "Destination address", value="", placeholder="Enter destination address"
        )

        radius_m = st.slider("Nearby damage radius (meters)", 30, 300, 120, 10)
        generate_clicked = st.button(
            "Generate Route",
            type="primary",
            disabled=not (source_place.strip() and destination_place.strip()),
        )

        key_in_env = bool(os.getenv("OPENROUTESERVICE_API_KEY"))
        st.caption(
            "OpenRouteService API key: "
            + ("found in environment" if key_in_env else "not found (set OPENROUTESERVICE_API_KEY)")
        )

    if st.sidebar.button("Clear Result"):
        st.session_state.route_analysis = None
        st.rerun()

    if not GEO_SCORED_CSV.exists():
        st.error("Missing data/geo_scored.csv. Run Step 4 first.")
        return

    damage_df = pd.read_csv(GEO_SCORED_CSV)
    damage_df = attach_damage_image_metadata(damage_df)

    if generate_clicked:
        try:
            source = geocode_place_name(source_place)
            destination = geocode_place_name(destination_place)
            st.markdown(
                f"<span class='status-chip'>Resolved coordinates: Source ({source[0]:.5f}, {source[1]:.5f}) | Destination ({destination[0]:.5f}, {destination[1]:.5f})</span>",
                unsafe_allow_html=True,
            )
        except ValueError as exc:
            st.error(f"Invalid source/destination input: {exc}")
            return
        except Exception as exc:
            st.error(f"Failed to resolve place names: {exc}")
            return

        with st.spinner("Fetching route and scoring road quality..."):
            try:
                route_points = get_route(source=source, destination=destination)
            except Exception as exc:
                st.error(f"Failed to fetch route from OpenRouteService: {exc}")
                return

            route_points = simplify_route_points(route_points, max_points=320)
            damage_points = load_damage_points(GEO_SCORED_CSV)
            score_payload = score_route(
                route_points=route_points,
                damage_points=damage_points,
                radius_m=float(radius_m),
            )

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ROUTE_POINTS_JSON.write_text(json.dumps(route_points, indent=2), encoding="utf-8")
        ROUTE_SCORES_JSON.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")

        st.session_state.route_analysis = {
            "source_place": source_place,
            "destination_place": destination_place,
            "resolved_source": source,
            "resolved_destination": destination,
            "radius_m": float(radius_m),
            "route_points": route_points,
            "score_payload": score_payload,
        }

    analysis = st.session_state.route_analysis
    if not analysis:
        st.markdown(
            """
            <div class="helper-note">
                <strong>Start here:</strong>
                Enter source and destination addresses, then click Generate Route to build the route heat visualization.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    score_payload = analysis["score_payload"]
    route_points = analysis["route_points"]

    st.markdown(
        f"<span class='status-chip'>Source: {analysis['source_place']} | Destination: {analysis['destination_place']}</span>",
        unsafe_allow_html=True,
    )

    route_damage_df = filter_damage_points_for_route(damage_df, score_payload["segments"])
    st.caption(f"Route-scoped damage points: {len(route_damage_df)}")

    st.markdown(
        f"""
        <div class='route-summary'>
            <div class='route-summary-title'>ROUTE SUMMARY</div>
            <div class='route-summary-main'>
                {analysis['source_place']} → {analysis['destination_place']}<br/>
                Quality: <b>{str(score_payload['route_quality']).upper()}</b> |
                Score: <b>{float(score_payload['route_score']):.1f}/100</b> |
                Segments: <b>{len(score_payload['segments'])}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div class='card'><div class='card-title'>Route Quality</div><div class='card-value'>{str(score_payload['route_quality']).upper()}</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='card'><div class='card-title'>Route Score</div><div class='card-value'>{float(score_payload['route_score']):.1f}<span style='font-size: 1rem; color: #64748b;'>/100</span></div></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='card'><div class='card-title'>Segments</div><div class='card-value'>{len(score_payload['segments'])}</div></div>",
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div class='card'><div class='card-title'>Nearby Damage Hits</div><div class='card-value'>{int(score_payload.get('total_nearby_hits', 0))}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="surface-panel" style="margin: 0.15rem 0 1rem; padding: 14px 16px;">
            <div class="section-title" style="margin-top: 0;">Scoring Explanation</div>
            <div class="section-subtitle" style="margin-bottom: 0;">
                Scores range from 0–100, where higher indicates more road damage along the route. The route heat, segment colors, and summary cards all use the same annotation-based scoring logic for consistency.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Route Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Use the layer controls to inspect route heat, damage markers, and route segments.</div>', unsafe_allow_html=True)

    route_map = build_dashboard_map(
        route_segments=score_payload["segments"],
        damage_df=damage_df,
        route_quality=str(score_payload["route_quality"]),
        route_score_value=float(score_payload["route_score"]),
        route_damage_df=route_damage_df,
    )

    st_folium(route_map, height=860, use_container_width=True, key="route_map")

    with st.expander("Segment-level scores", expanded=False):
        seg_df = pd.DataFrame(score_payload["segments"])
        st.dataframe(
            seg_df[["segment_index", "distance_m", "score", "quality"]],
            use_container_width=True,
            height=320,
        )


if __name__ == "__main__":
    main()
