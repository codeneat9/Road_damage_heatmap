from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import List, Tuple

import folium
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from folium.plugins import HeatMap
from streamlit_folium import st_folium

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from map_utils import quality_to_color
from route_score import load_damage_points, score_route
from routing import geocode_place_name, get_route


DATA_DIR = Path("data")
ROUTE_POINTS_JSON = DATA_DIR / "route_points.json"
ROUTE_SCORES_JSON = DATA_DIR / "route_scores.json"
GEO_SCORED_CSV = DATA_DIR / "geo_scored.csv"


def load_env_files() -> None:
    """Load API keys from common env files if present."""
    load_dotenv(Path(".env"), override=False)
    load_dotenv(Path(".env.local"), override=False)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.06;
            margin-bottom: 0.3rem;
            color: #1f2937;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .card {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 14px 16px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            box-shadow: 0 6px 20px rgba(0,0,0,0.04);
        }
        .card-title {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 4px;
        }
        .card-value {
            font-size: 2.15rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.1;
        }
        .status-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: #eef2ff;
            color: #3730a3;
        }
        .route-summary {
            margin-top: 0.95rem;
            margin-bottom: 0.85rem;
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid #dbeafe;
            background: linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%);
        }
        .route-summary-title {
            font-size: 0.92rem;
            color: #1d4ed8;
            font-weight: 700;
            margin-bottom: 4px;
            letter-spacing: 0.02em;
        }
        .route-summary-main {
            font-size: 1.06rem;
            color: #0f172a;
            font-weight: 600;
            line-height: 1.35;
        }
        .warn-badge {
            display: inline-block;
            margin-top: 0.35rem;
            padding: 6px 10px;
            border-radius: 10px;
            border: 1px solid #fed7aa;
            background: #fff7ed;
            color: #9a3412;
            font-size: 0.84rem;
            font-weight: 600;
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

    fmap = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    # Only render heatmap when there are damage points near the selected route.
    if not route_damage_df.empty:
        heat_df = route_damage_df.head(200)
        heat_data = heat_df[["lat", "lon", "severity"]].values.tolist()
        HeatMap(heat_data, min_opacity=0.35, radius=16, blur=14, max_zoom=17).add_to(fmap)

    merged_segments = compress_segments(route_segments)
    for seg in merged_segments:
        color = quality_to_color(str(seg["quality"]))
        coords = seg["coords"]
        avg_score = sum(seg["scores"]) / max(1, len(seg["scores"]))

        folium.PolyLine(
            locations=[[c[0], c[1]] for c in coords],
            color=color,
            weight=8,
            opacity=0.92,
            tooltip=f"{str(seg['quality']).upper()} | Avg score: {avg_score:.6f}",
        ).add_to(fmap)

    first_start = route_segments[0]["start"]
    last_end = route_segments[-1]["end"]

    folium.Marker(
        [first_start[0], first_start[1]],
        tooltip="Start",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(fmap)

    folium.Marker(
        [last_end[0], last_end[1]],
        tooltip="Destination",
        icon=folium.Icon(color="darkred", icon="flag"),
    ).add_to(fmap)

    title_html = f"""
        <h4 style=\"margin: 8px 10px; font-family: Segoe UI, Arial, sans-serif;\">
      Road Quality: {route_quality.upper()} | Route Score: {route_score_value:.6f}
    </h4>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    bounds = [[min(lat_vals), min(lon_vals)], [max(lat_vals), max(lon_vals)]]
    fmap.fit_bounds(bounds, padding=(20, 20))

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

    st.markdown('<div class="main-title">Road Damage Detection and Route Quality Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Address-driven routing, annotation-based severity scoring, and route risk visualization.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Route Input")
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
        st.info("Enter source and destination addresses, then click Generate Route.")
        return

    score_payload = analysis["score_payload"]
    route_points = analysis["route_points"]

    st.markdown(
        f"<span class='status-chip'>Source: {analysis['source_place']} | Destination: {analysis['destination_place']}</span>",
        unsafe_allow_html=True,
    )

    route_damage_df = filter_damage_points_for_route(damage_df, score_payload["segments"])
    if route_damage_df.empty:
        st.markdown(
            "<span class='warn-badge'>No nearby damage points for this route</span>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Showing {len(route_damage_df)} damage points near the selected route.")

    st.markdown(
        f"""
        <div class='route-summary'>
            <div class='route-summary-title'>ROUTE SUMMARY</div>
            <div class='route-summary-main'>
                {analysis['source_place']} → {analysis['destination_place']}<br/>
                Quality: <b>{str(score_payload['route_quality']).upper()}</b> |
                Score: <b>{float(score_payload['route_score']):.8f}</b> |
                Segments: <b>{len(score_payload['segments'])}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div class='card'><div class='card-title'>Route Quality</div><div class='card-value'>{str(score_payload['route_quality']).upper()}</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='card'><div class='card-title'>Route Score</div><div class='card-value'>{float(score_payload['route_score']):.8f}</div></div>",
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

    st.caption(
        "Higher score means the route passes closer to more damage points. "
        "When no nearby damage exists, the fallback score keeps the route informative instead of zero."
    )

    route_map = build_dashboard_map(
        route_segments=score_payload["segments"],
        damage_df=damage_df,
        route_quality=str(score_payload["route_quality"]),
        route_score_value=float(score_payload["route_score"]),
        route_damage_df=route_damage_df,
    )

    st_folium(route_map, height=820, use_container_width=True, key="route_map")

    with st.expander("Segment-level scores", expanded=False):
        seg_df = pd.DataFrame(score_payload["segments"])
        st.dataframe(
            seg_df[["segment_index", "distance_m", "score", "quality"]],
            use_container_width=True,
            height=320,
        )


if __name__ == "__main__":
    main()
