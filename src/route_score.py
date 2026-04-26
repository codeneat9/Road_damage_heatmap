from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


Coordinate = Tuple[float, float]  # (lat, lon)
DamagePoint = Tuple[float, float, float]  # (lat, lon, severity)


def haversine_distance_m(a: Coordinate, b: Coordinate) -> float:
    """Compute great-circle distance between two (lat, lon) points in meters."""
    lat1, lon1 = a
    lat2, lon2 = b

    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def point_risk_score(
    route_point: Coordinate,
    damage_points: Sequence[DamagePoint],
    radius_m: float,
    epsilon_m: float = 5.0,
) -> float:
    """
    Compute local risk at one route point using nearby damage points.

    score += severity / distance
    Only points within radius_m contribute.
    """
    score = 0.0

    for d_lat, d_lon, severity in damage_points:
        distance_m = haversine_distance_m(route_point, (d_lat, d_lon))
        if distance_m > radius_m:
            continue

        score += float(severity) / max(distance_m, epsilon_m)

    return score


def classify_segment(score: float, good_threshold: float, moderate_threshold: float) -> str:
    if score < good_threshold:
        return "good"
    if score < moderate_threshold:
        return "moderate"
    return "bad"


def load_damage_points(geo_csv: Path) -> List[DamagePoint]:
    if not geo_csv.exists():
        raise FileNotFoundError(f"Damage CSV not found: {geo_csv}")

    df = pd.read_csv(geo_csv)
    required = {"lat", "lon", "severity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in damage CSV: {sorted(missing)}")

    return [
        (float(row.lat), float(row.lon), float(row.severity))
        for row in df.itertuples(index=False)
    ]


def load_route_points(route_json: Path) -> List[Coordinate]:
    if not route_json.exists():
        raise FileNotFoundError(f"Route JSON not found: {route_json}")

    raw = json.loads(route_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("Route JSON must be a list with at least two points")

    points: List[Coordinate] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each route point must be [lat, lon]")

        lat = float(item[0])
        lon = float(item[1])
        points.append((lat, lon))

    return points


def score_route(
    route_points: Sequence[Coordinate],
    damage_points: Sequence[DamagePoint],
    radius_m: float = 120.0,
    good_threshold: float = 0.0015,
    moderate_threshold: float = 0.0040,
) -> Dict[str, object]:
    """
    Step 7: Score route quality based on nearby road damage points.

    For each segment i (point i -> i+1):
    - compute risk at endpoints
    - segment score = average of endpoint risks
    """
    if len(route_points) < 2:
        raise ValueError("At least two route points are required")

    segments: List[Dict[str, object]] = []
    segment_scores: List[float] = []

    for i in range(len(route_points) - 1):
        p1 = route_points[i]
        p2 = route_points[i + 1]

        s1 = point_risk_score(p1, damage_points, radius_m=radius_m)
        s2 = point_risk_score(p2, damage_points, radius_m=radius_m)
        seg_score = (s1 + s2) / 2.0
        seg_distance_m = haversine_distance_m(p1, p2)

        segment_scores.append(seg_score)
        segments.append(
            {
                "segment_index": i,
                "start": [p1[0], p1[1]],
                "end": [p2[0], p2[1]],
                "distance_m": round(seg_distance_m, 3),
                "score": round(seg_score, 8),
                "quality": classify_segment(
                    seg_score,
                    good_threshold=good_threshold,
                    moderate_threshold=moderate_threshold,
                ),
            }
        )

    route_score_value = sum(segment_scores) / len(segment_scores)
    route_quality = classify_segment(
        route_score_value,
        good_threshold=good_threshold,
        moderate_threshold=moderate_threshold,
    )

    return {
        "route_points": len(route_points),
        "segments": segments,
        "route_score": round(route_score_value, 8),
        "route_quality": route_quality,
        "thresholds": {
            "good": good_threshold,
            "moderate": moderate_threshold,
            "bad": f">= {moderate_threshold}",
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 7: Compute route quality score")
    parser.add_argument("--route-json", default="data/route_points.json")
    parser.add_argument("--damage-csv", default="data/geo_scored.csv")
    parser.add_argument("--output-json", default="data/route_scores.json")
    parser.add_argument("--radius-m", type=float, default=120.0)
    parser.add_argument("--good-threshold", type=float, default=0.0015)
    parser.add_argument("--moderate-threshold", type=float, default=0.0040)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    route_points = load_route_points(Path(args.route_json))
    damage_points = load_damage_points(Path(args.damage_csv))

    result = score_route(
        route_points=route_points,
        damage_points=damage_points,
        radius_m=args.radius_m,
        good_threshold=args.good_threshold,
        moderate_threshold=args.moderate_threshold,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Segments scored: {len(result['segments'])}")
    print(f"Route score: {result['route_score']}")
    print(f"Route quality: {result['route_quality']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
