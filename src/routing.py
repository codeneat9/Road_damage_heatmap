from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import requests


Coordinate = Tuple[float, float]  # (lat, lon)


def _to_lat_lon(value: str | Sequence[float] | Coordinate) -> Coordinate:
    """Parse a coordinate from 'lat,lon' string or a 2-value sequence."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Coordinate string must be 'lat,lon', got: {value}")
        return float(parts[0]), float(parts[1])

    if len(value) != 2:
        raise ValueError(f"Coordinate sequence must have 2 values, got: {value}")

    return float(value[0]), float(value[1])


def _validate_coordinate(lat: float, lon: float, name: str) -> None:
    if not (-90 <= lat <= 90):
        raise ValueError(f"{name} latitude must be between -90 and 90, got {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"{name} longitude must be between -180 and 180, got {lon}")


def get_route(
    source: str | Sequence[float] | Coordinate,
    destination: str | Sequence[float] | Coordinate,
    api_key: str | None = None,
    profile: str = "driving-car",
    timeout: int = 30,
) -> List[Coordinate]:
    """
    Step 6: Fetch route points using OpenRouteService.

    Inputs:
    - source: (lat, lon) tuple/list or 'lat,lon' string
    - destination: (lat, lon) tuple/list or 'lat,lon' string

    Returns:
    - list of (lat, lon) coordinates along the route
    """
    src_lat, src_lon = _to_lat_lon(source)
    dst_lat, dst_lon = _to_lat_lon(destination)

    _validate_coordinate(src_lat, src_lon, "Source")
    _validate_coordinate(dst_lat, dst_lon, "Destination")

    key = api_key or os.getenv("OPENROUTESERVICE_API_KEY")
    if not key:
        raise ValueError(
            "Missing OpenRouteService API key. Set OPENROUTESERVICE_API_KEY or pass api_key."
        )

    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    headers = {
        "Authorization": key,
        "Content-Type": "application/json",
    }
    payload = {
        # ORS expects [lon, lat]
        "coordinates": [[src_lon, src_lat], [dst_lon, dst_lat]],
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouteService error {response.status_code}: {response.text[:400]}"
        )

    data = response.json()
    features = data.get("features", [])
    if not features:
        raise RuntimeError("No route features returned by OpenRouteService")

    geometry = features[0].get("geometry", {})
    coordinates = geometry.get("coordinates", [])
    if not coordinates:
        raise RuntimeError("No route coordinates returned by OpenRouteService")

    # Convert from [lon, lat] to (lat, lon)
    return [(float(lat), float(lon)) for lon, lat in coordinates]


def geocode_place_name(
    place_name: str,
    api_key: str | None = None,
    country_code: str = "IN",
    focus_lat: float = 17.3850,
    focus_lon: float = 78.4867,
    timeout: int = 30,
) -> Coordinate:
    """Resolve a place name into a (lat, lon) coordinate using OpenRouteService geocoding."""
    query = place_name.strip()
    if not query:
        raise ValueError("Place name cannot be empty")

    key = api_key or os.getenv("OPENROUTESERVICE_API_KEY")
    if not key:
        raise ValueError(
            "Missing OpenRouteService API key. Set OPENROUTESERVICE_API_KEY or pass api_key."
        )

    url = "https://api.openrouteservice.org/geocode/search"
    params = {
        "api_key": key,
        "text": query,
        "size": 1,
        "boundary.country": country_code,
        "focus.point.lat": focus_lat,
        "focus.point.lon": focus_lon,
    }

    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouteService geocoding error {response.status_code}: {response.text[:400]}"
        )

    data = response.json()
    features = data.get("features", [])
    if not features:
        raise RuntimeError(f"No geocoding result found for '{query}'")

    lon, lat = features[0].get("geometry", {}).get("coordinates", [None, None])
    if lat is None or lon is None:
        raise RuntimeError(f"Invalid geocoding response for '{query}'")

    return float(lat), float(lon)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 6: Fetch route with OpenRouteService")
    parser.add_argument(
        "--source",
        required=True,
        help="Source coordinate as 'lat,lon'",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Destination coordinate as 'lat,lon'",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouteService API key (or set OPENROUTESERVICE_API_KEY)",
    )
    parser.add_argument(
        "--profile",
        default="driving-car",
        help="ORS route profile, e.g., driving-car",
    )
    parser.add_argument(
        "--save-json",
        default="data/route_points.json",
        help="Where to save fetched route points",
    )
    return parser


def main() -> None:
    import json

    args = build_arg_parser().parse_args()

    route_points = get_route(
        source=args.source,
        destination=args.destination,
        api_key=args.api_key,
        profile=args.profile,
    )

    out_path = Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(route_points, handle, indent=2)

    print(f"Route points fetched: {len(route_points)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
