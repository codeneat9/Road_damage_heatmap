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


def parse_lat_lon(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("Use format: lat,lon")

    lat = float(parts[0])
    lon = float(parts[1])

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError("Latitude/longitude out of range")

    return lat, lon


def build_dashboard_map(
    route_segments: List[dict],
    damage_df: pd.DataFrame,
    route_quality: str,
    route_score_value: float,
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

    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # Base heatmap from known damage severity points.
    heat_data = damage_df[["lat", "lon", "severity"]].values.tolist()
    HeatMap(heat_data, min_opacity=0.35, radius=16, blur=14, max_zoom=17).add_to(fmap)

    for seg in route_segments:
        color = quality_to_color(str(seg["quality"]))
        start = seg["start"]
        end = seg["end"]

        folium.PolyLine(
            locations=[[start[0], start[1]], [end[0], end[1]]],
            color=color,
            weight=7,
            opacity=0.95,
            tooltip=f"Segment {seg['segment_index']} | {seg['quality']} | score={seg['score']:.6f}",
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
    <h4 style=\"margin: 8px 10px; font-family: Arial, sans-serif;\">
      Road Quality: {route_quality.upper()} | Route Score: {route_score_value:.6f}
    </h4>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    return fmap


def main() -> None:
    load_env_files()

    st.set_page_config(
        page_title="Road Damage Route Quality Dashboard",
        layout="wide",
    )

    st.title("Road Damage Detection and Route Quality Dashboard")
    st.caption("Annotation-based pipeline: no ML model inference is used.")

    with st.sidebar:
        st.subheader("Route Input")
        input_mode = st.radio(
            "Input type",
            options=["Coordinates", "Place Names"],
            index=1,
            horizontal=False,
        )

        if input_mode == "Coordinates":
            source_text = st.text_input("Source (lat,lon)", value="17.3850,78.4867")
            destination_text = st.text_input(
                "Destination (lat,lon)", value="17.4250,78.5267"
            )
            source_place = ""
            destination_place = ""
        else:
            source_place = st.text_input("Source place", value="Hyderabad Railway Station")
            destination_place = st.text_input(
                "Destination place", value="Rajiv Gandhi International Airport Hyderabad"
            )
            source_text = ""
            destination_text = ""

        radius_m = st.slider("Nearby damage radius (meters)", 30, 300, 120, 10)
        generate_clicked = st.button("Generate Route", type="primary")

        key_in_env = bool(os.getenv("OPENROUTESERVICE_API_KEY"))
        st.caption(
            "OpenRouteService API key: "
            + ("found in environment" if key_in_env else "not found (set OPENROUTESERVICE_API_KEY)")
        )

    if not GEO_SCORED_CSV.exists():
        st.error("Missing data/geo_scored.csv. Run Step 4 first.")
        return

    damage_df = pd.read_csv(GEO_SCORED_CSV)

    if not generate_clicked:
        st.info("Enter source and destination, then click Generate Route.")
        return

    try:
        if input_mode == "Coordinates":
            source = parse_lat_lon(source_text)
            destination = parse_lat_lon(destination_text)
        else:
            source = geocode_place_name(source_place)
            destination = geocode_place_name(destination_place)
            st.caption(
                f"Resolved coordinates: Source ({source[0]:.5f}, {source[1]:.5f}) | "
                f"Destination ({destination[0]:.5f}, {destination[1]:.5f})"
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

        damage_points = load_damage_points(GEO_SCORED_CSV)
        score_payload = score_route(
            route_points=route_points,
            damage_points=damage_points,
            radius_m=float(radius_m),
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ROUTE_POINTS_JSON.write_text(json.dumps(route_points, indent=2), encoding="utf-8")
    ROUTE_SCORES_JSON.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")

    c1, c2, c3 = st.columns(3)
    c1.metric("Route Quality", str(score_payload["route_quality"]).upper())
    c2.metric("Route Score", f"{float(score_payload['route_score']):.6f}")
    c3.metric("Segments", f"{len(score_payload['segments'])}")

    route_map = build_dashboard_map(
        route_segments=score_payload["segments"],
        damage_df=damage_df,
        route_quality=str(score_payload["route_quality"]),
        route_score_value=float(score_payload["route_score"]),
    )

    st_folium(route_map, width=1200, height=650)

    with st.expander("Segment-level scores"):
        seg_df = pd.DataFrame(score_payload["segments"])
        st.dataframe(seg_df[["segment_index", "distance_m", "score", "quality"]], use_container_width=True)


if __name__ == "__main__":
    main()
