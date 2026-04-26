from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_simulated_coordinates(
    scored_csv: Path,
    output_csv: Path,
    base_lat: float = 17.3850,
    base_lon: float = 78.4867,
    step: float = 0.0001,
) -> pd.DataFrame:
    """
    Step 4: Assign simulated route-like coordinates to each image row.

    For row index i:
    lat = base_lat + i * step
    lon = base_lon + i * step
    """
    if not scored_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {scored_csv}")

    df = pd.read_csv(scored_csv)

    required_columns = {"image_name", "severity"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing_columns)}")

    df = df.copy()
    df = df.sort_values("image_name").reset_index(drop=True)

    df["lat"] = [base_lat + i * step for i in range(len(df))]
    df["lon"] = [base_lon + i * step for i in range(len(df))]

    # Required output columns for downstream heatmap pipeline
    out_df = df[["lat", "lon", "severity"]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    return out_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 4: Geo-tag severity records with simulated coordinates"
    )
    parser.add_argument("--input-csv", default="data/scored.csv")
    parser.add_argument("--output-csv", default="data/geo_scored.csv")
    parser.add_argument("--base-lat", type=float, default=17.3850)
    parser.add_argument("--base-lon", type=float, default=78.4867)
    parser.add_argument("--step", type=float, default=0.0001)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    out_df = add_simulated_coordinates(
        scored_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        base_lat=args.base_lat,
        base_lon=args.base_lon,
        step=args.step,
    )

    print(f"Rows geotagged: {len(out_df)}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
