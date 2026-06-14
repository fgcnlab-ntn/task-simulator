#!/usr/bin/env python3
"""Convert a WorldPop population-count GeoTIFF into demand-point CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PopulationBin:
    population: float = 0.0
    representative_population: float = 0.0
    representative_lat: float = 0.0
    representative_lon: float = 0.0

    def add(self, lat: float, lon: float, population: float) -> None:
        lat = float(lat)
        lon = float(lon)
        population = float(population)
        self.population += population
        if population > self.representative_population:
            self.representative_population = population
            self.representative_lat = lat
            self.representative_lon = lon

    @property
    def lat(self) -> float:
        return self.representative_lat

    @property
    def lon(self) -> float:
        return self.representative_lon


def bin_key(lat: float, lon: float, aggregate_deg: float) -> tuple[int, int]:
    return (
        math.floor((lat + 90.0) / aggregate_deg),
        math.floor((lon + 180.0) / aggregate_deg),
    )


def aggregate_points(
    points: Iterable[tuple[float, float, float]],
    aggregate_deg: float,
) -> dict[tuple[int, int], PopulationBin]:
    bins: dict[tuple[int, int], PopulationBin] = {}
    for lat, lon, population in points:
        if not math.isfinite(population) or population <= 0.0:
            continue
        cell = bins.setdefault(bin_key(lat, lon, aggregate_deg), PopulationBin())
        cell.add(lat, lon, population)
    return bins


def iter_population_points(path: Path, bbox: tuple[float, float, float, float] | None):
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import xy
        from rasterio.windows import Window, from_bounds
    except ImportError as exc:
        raise SystemExit(
            "WorldPop conversion requires rasterio and numpy. "
            "Install them with: python3 -m pip install -r requirements-worldpop.txt"
        ) from exc

    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one population band, found {dataset.count}")
        if dataset.crs is None or not dataset.crs.is_geographic:
            raise ValueError("WorldPop input must use a geographic latitude/longitude CRS")

        window = None
        if bbox is not None:
            west, south, east, north = bbox
            window = from_bounds(west, south, east, north, dataset.transform)
            window = window.round_offsets().round_lengths()
            full = Window(0, 0, dataset.width, dataset.height)
            try:
                window = window.intersection(full)
            except rasterio.errors.WindowError as exc:
                raise ValueError("bbox does not overlap the input raster bounds") from exc

        for _, block_window in dataset.block_windows(1):
            if window is not None:
                try:
                    block_window = block_window.intersection(window)
                except rasterio.errors.WindowError:
                    continue
            values = dataset.read(1, window=block_window, masked=True)

            array = values.filled(np.nan)
            valid = np.isfinite(array) & (array > 0.0)

            rows, cols = np.nonzero(valid)
            populations = array[rows, cols]

            if len(rows) == 0:
                continue
            global_rows = rows + int(block_window.row_off)
            global_cols = cols + int(block_window.col_off)
            lons, lats = xy(dataset.transform, global_rows, global_cols, offset="center")
            populations = values.data[rows, cols]
            yield from zip(lats, lons, populations)


def write_csv(
    path: Path,
    bins: dict[tuple[int, int], PopulationBin],
    min_population: float,
) -> tuple[int, float]:
    kept = [cell for cell in bins.values() if cell.population >= min_population]
    kept.sort(key=lambda cell: (cell.lat, cell.lon))
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["lat", "lon", "weight"])
        for cell in kept:
            writer.writerow([f"{cell.lat:.8f}", f"{cell.lon:.8f}", f"{cell.population:.6f}"])
    return len(kept), sum(cell.population for cell in kept)


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox is outside valid longitude/latitude bounds")
    return west, south, east, north


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="WorldPop population-count GeoTIFF")
    parser.add_argument("output", type=Path, help="output demand-point CSV")
    parser.add_argument(
        "--aggregate-deg",
        type=float,
        default=0.01,
        help="latitude/longitude aggregation size; 0.01 degrees is roughly 1 km",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        help="optional crop as west,south,east,north",
    )
    parser.add_argument(
        "--min-population",
        type=float,
        default=1.0,
        help="discard aggregated cells below this population",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_deg <= 0.0:
        raise ValueError("--aggregate-deg must be positive")
    if args.min_population < 0.0:
        raise ValueError("--min-population must be non-negative")

    bins = aggregate_points(
        iter_population_points(args.input, args.bbox),
        args.aggregate_deg,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    point_count, output_population = write_csv(args.output, bins, args.min_population)
    input_population = sum(cell.population for cell in bins.values())
    metadata = {
        "source": "WorldPop population-count GeoTIFF",
        "source_file": str(args.input),
        "output_file": str(args.output),
        "aggregate_deg": args.aggregate_deg,
        "bbox": args.bbox,
        "min_population": args.min_population,
        "input_positive_population": input_population,
        "output_population": output_population,
        "discarded_population": input_population - output_population,
        "demand_points": point_count,
        "coordinate": "highest-population source pixel in each aggregated cell",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {point_count} demand points to {args.output}")
    print(f"Preserved population: {output_population:.3f} / {input_population:.3f}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
