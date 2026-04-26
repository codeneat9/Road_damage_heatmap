from __future__ import annotations

import argparse
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap


def generate_heatmap(input_csv: Path, output_html: Path) -> folium.Map:
    """
    Step 5: Build heatmap from geo-scored CSV.

    Expected CSV columns:
    - lat
    - lon
    - severity
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = {"lat", "lon", "severity"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input CSV has no rows")

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    heat_data = df[["lat", "lon", "severity"]].values.tolist()
    HeatMap(heat_data, min_opacity=0.4, radius=18, blur=15, max_zoom=17).add_to(fmap)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))

    return fmap


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 5: Generate Folium heatmap")
    parser.add_argument("--input-csv", default="data/geo_scored.csv")
    parser.add_argument("--output-html", default="data/heatmap.html")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    generate_heatmap(
        input_csv=Path(args.input_csv),
        output_html=Path(args.output_html),
    )

    print(f"Heatmap saved: {args.output_html}")


if __name__ == "__main__":
    main()
