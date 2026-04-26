from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import folium
import pandas as pd


Coordinate = Tuple[float, float]


def quality_to_color(quality: str) -> str:
    """Map segment quality label to Folium polyline color."""
    normalized = (quality or "").strip().lower()
    if normalized == "good":
        return "green"
    if normalized == "moderate":
        return "yellow"
    return "red"


def _center_from_segments(segments: List[Dict[str, object]]) -> Coordinate:
    lat_values: List[float] = []
    lon_values: List[float] = []

    for seg in segments:
        start = seg.get("start", [0.0, 0.0])
        end = seg.get("end", [0.0, 0.0])

        lat_values.extend([float(start[0]), float(end[0])])
        lon_values.extend([float(start[1]), float(end[1])])

    if not lat_values:
        return 17.3850, 78.4867

    return sum(lat_values) / len(lat_values), sum(lon_values) / len(lon_values)


def _add_legend(fmap: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        z-index: 9999;
        background: white;
        border: 2px solid #444;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    ">
      <div style="font-weight: 700; margin-bottom: 6px;">Road Quality</div>
      <div><span style="color: green; font-weight: 700;">&#9632;</span> Good</div>
      <div><span style="color: #b59b00; font-weight: 700;">&#9632;</span> Moderate</div>
      <div><span style="color: red; font-weight: 700;">&#9632;</span> Bad</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))


def generate_route_quality_map(
    route_scores_json: Path,
    output_html: Path,
    damage_csv: Path | None = None,
) -> folium.Map:
    """
    Step 8: Draw a route map with segment colors:
    - green: good
    - yellow: moderate
    - red: bad
    """
    if not route_scores_json.exists():
        raise FileNotFoundError(f"Route score JSON not found: {route_scores_json}")

    payload = json.loads(route_scores_json.read_text(encoding="utf-8"))
    segments: List[Dict[str, object]] = payload.get("segments", [])

    if not segments:
        raise ValueError("No segments found in route score JSON")

    center_lat, center_lon = _center_from_segments(segments)
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Optional: show damage points for context.
    if damage_csv is not None and damage_csv.exists():
        damage_df = pd.read_csv(damage_csv)
        if {"lat", "lon", "severity"}.issubset(damage_df.columns):
            for row in damage_df.itertuples(index=False):
                severity = float(row.severity)
                if severity <= 0:
                    continue
                folium.CircleMarker(
                    location=[float(row.lat), float(row.lon)],
                    radius=max(2, min(8, int(2 + severity * 20))),
                    color="orangered",
                    fill=True,
                    fill_opacity=0.35,
                    weight=1,
                    opacity=0.5,
                ).add_to(fmap)

    for seg in segments:
        start = seg.get("start", [0.0, 0.0])
        end = seg.get("end", [0.0, 0.0])
        quality = str(seg.get("quality", "bad"))
        color = quality_to_color(quality)
        score = float(seg.get("score", 0.0))

        folium.PolyLine(
            locations=[[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]],
            color=color,
            weight=6,
            opacity=0.95,
            tooltip=f"Quality: {quality} | Score: {score:.6f}",
        ).add_to(fmap)

    first_start = segments[0].get("start", [center_lat, center_lon])
    last_end = segments[-1].get("end", [center_lat, center_lon])

    folium.Marker(
        [float(first_start[0]), float(first_start[1])],
        tooltip="Start",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(fmap)

    folium.Marker(
        [float(last_end[0]), float(last_end[1])],
        tooltip="Destination",
        icon=folium.Icon(color="darkred", icon="flag"),
    ).add_to(fmap)

    route_quality = payload.get("route_quality", "unknown")
    route_score = payload.get("route_score", "n/a")
    title_html = f"""
    <h4 style=\"margin: 8px 10px; font-family: Arial, sans-serif;\">
      Route Quality: {route_quality} | Route Score: {route_score}
    </h4>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    _add_legend(fmap)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))

    return fmap


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 8: Visualize route quality with color-coded segments"
    )
    parser.add_argument("--route-scores-json", default="data/route_scores.json")
    parser.add_argument("--output-html", default="data/route_quality_map.html")
    parser.add_argument("--damage-csv", default="data/geo_scored.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    generate_route_quality_map(
        route_scores_json=Path(args.route_scores_json),
        output_html=Path(args.output_html),
        damage_csv=Path(args.damage_csv) if args.damage_csv else None,
    )

    print(f"Saved: {args.output_html}")


if __name__ == "__main__":
    main()
