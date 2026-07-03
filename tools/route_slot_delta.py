#!/usr/bin/env python3
"""Measure shortest-route changes across time slots.

Sidecar analysis only: imports satmulator's existing orbit/topology helpers but
never changes simulator behavior.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from satmulator.cli import (  # noqa: E402
    DEFAULT_CONFIG,
    load_standalone_json_config,
    parse_utc_datetime,
    validate_args,
    walker_raan_spread_deg,
)
from satmulator.constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM  # noqa: E402
from satmulator.geometry import circular_state, is_sunlit_cylindrical_shadow  # noqa: E402
from satmulator.isl import build_constellation_layout, build_isl_graph, shortest_route  # noqa: E402
from satmulator.models import ISLConfig, SatelliteView  # noqa: E402

RouteNodes = tuple[int, ...] | None
Pair = tuple[int, int]
QUARTER_EARTH_CIRCUMFERENCE_FRACTION = 0.25


def load_args_from_config(config_path: Path) -> argparse.Namespace:
    values = load_standalone_json_config(config_path)
    merged = DEFAULT_CONFIG.copy()
    merged.update(values)
    merged["config"] = config_path
    merged["plot_run"] = None
    merged["tle_file"] = None if merged["tle_file"] is None else Path(merged["tle_file"])
    merged["task_demand_points_file"] = (
        None
        if merged["task_demand_points_file"] is None
        else Path(merged["task_demand_points_file"])
    )
    merged["out"] = Path(merged["out"])
    return argparse.Namespace(**merged)


def make_isl_config(args: argparse.Namespace) -> ISLConfig:
    return ISLConfig(
        rate_bps=args.isl_rate_bps,
        tx_power_w=args.isl_tx_power_w,
        topology=args.isl_topology,
        max_range_km=args.isl_max_range_km,
    )


def circular_views(args: argparse.Namespace, time_s: int) -> list[SatelliteView]:
    sats_per_plane = args.satellites // args.planes
    radius_km = EARTH_RADIUS_KM + args.altitude_km
    inclination_rad = math.radians(args.inclination_deg)
    raan_spread_rad = math.radians(walker_raan_spread_deg(args))
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (radius_km**3))
    sun_unit = (1.0, 0.0, 0.0)

    views: list[SatelliteView] = []
    for plane in range(args.planes):
        raan = raan_spread_rad * plane / args.planes
        plane_phase = 2.0 * math.pi * args.walker_phase * plane / args.satellites
        for slot in range(sats_per_plane):
            sat_id = plane * sats_per_plane + slot
            arg = (
                2.0 * math.pi * slot / sats_per_plane
                + plane_phase
                + mean_motion * time_s
            )
            pos, _vel = circular_state(radius_km, inclination_rad, raan, arg)
            views.append(
                SatelliteView(
                    sat_id=sat_id,
                    x_km=pos[0],
                    y_km=pos[1],
                    z_km=pos[2],
                    sunlit=is_sunlit_cylindrical_shadow(pos, sun_unit),
                    plane=plane,
                    slot=slot,
                )
            )
    return views


def central_angle_rad(a: SatelliteView, b: SatelliteView) -> float:
    dot = a.x_km * b.x_km + a.y_km * b.y_km + a.z_km * b.z_km
    norm_a = math.sqrt(a.x_km * a.x_km + a.y_km * a.y_km + a.z_km * a.z_km)
    norm_b = math.sqrt(b.x_km * b.x_km + b.y_km * b.y_km + b.z_km * b.z_km)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("satellite position vector must be non-zero")
    cosine = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return math.acos(cosine)


def earth_surface_distance_km(a: SatelliteView, b: SatelliteView) -> float:
    return EARTH_RADIUS_KM * central_angle_rad(a, b)


def route_text(route: RouteNodes) -> str:
    return "" if route is None else " ".join(map(str, route))


def hop_count(route: RouteNodes) -> int | None:
    return None if route is None else len(route) - 1


def blank_none(value: object) -> object:
    return "" if value is None else value


def node_jaccard_distance(a: RouteNodes, b: RouteNodes) -> float | None:
    if a is None or b is None:
        return None
    left = set(a)
    right = set(b)
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def symdiff_count(a: RouteNodes, b: RouteNodes) -> int | None:
    if a is None or b is None:
        return None
    return len(set(a) ^ set(b))


def length_delta(a: RouteNodes, b: RouteNodes) -> int | None:
    left = hop_count(a)
    right = hop_count(b)
    if left is None or right is None:
        return None
    return right - left


def parse_lags(value: str) -> tuple[int, ...]:
    lags = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    if not lags or any(lag <= 0 for lag in lags):
        raise argparse.ArgumentTypeError("lags must be positive slot counts, e.g. 1,2,4,8")
    return lags


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=int, help="source satellite id for one fixed pair")
    parser.add_argument("--dest", type=int, help="destination satellite id for one fixed pair")
    parser.add_argument(
        "--random-pairs",
        type=int,
        default=0,
        help="number of random source/destination pairs to add",
    )
    parser.add_argument("--seed", type=int, default=1, help="random-pair seed")
    parser.add_argument(
        "--min-distance-earth-circumference-fraction",
        type=float,
        default=0.0,
        help=(
            "minimum source-dest central-angle distance as a fraction of "
            "Earth circumference, checked at t=0; use 0.25 for >= 1/4"
        ),
    )
    parser.add_argument(
        "--lags",
        type=parse_lags,
        default=(1,),
        help="comma-separated slot gaps to compare; 1 means t vs t+1",
    )
    parser.add_argument("--out", type=Path, help="delta CSV output path; default prints to stdout")
    parser.add_argument(
        "--routes-out",
        type=Path,
        help="optional CSV storing every route's satellite ids for each pair/time",
    )
    return parser.parse_args()


def validate_pair(pair: Pair, satellites: int) -> None:
    source, dest = pair
    if not (0 <= source < satellites) or not (0 <= dest < satellites):
        raise SystemExit(f"source/dest must be in [0, {satellites - 1}]")
    if source == dest:
        raise SystemExit("source and dest must be different for route-delta sampling")


def select_pairs(
    cli: argparse.Namespace,
    satellites: int,
    initial_views: list[SatelliteView],
) -> list[Pair]:
    if cli.min_distance_earth_circumference_fraction < 0.0:
        raise SystemExit("--min-distance-earth-circumference-fraction must be non-negative")

    min_distance_km = (
        cli.min_distance_earth_circumference_fraction
        * 2.0
        * math.pi
        * EARTH_RADIUS_KM
    )
    by_id = {sat.sat_id: sat for sat in initial_views}

    def allowed(pair: Pair) -> bool:
        source, dest = pair
        return earth_surface_distance_km(by_id[source], by_id[dest]) >= min_distance_km

    pairs: list[Pair] = []
    seen: set[Pair] = set()

    if (cli.source is None) != (cli.dest is None):
        raise SystemExit("--source and --dest must be provided together")
    if cli.source is not None and cli.dest is not None:
        pair = (cli.source, cli.dest)
        validate_pair(pair, satellites)
        if not allowed(pair):
            raise SystemExit(
                "fixed source/dest pair does not satisfy "
                "--min-distance-earth-circumference-fraction"
            )
        pairs.append(pair)
        seen.add(pair)

    if cli.random_pairs < 0:
        raise SystemExit("--random-pairs must be non-negative")
    if cli.random_pairs:
        if satellites < 2:
            raise SystemExit("need at least two satellites for random pairs")
        candidates = [
            pair
            for pair in permutations(range(satellites), 2)
            if pair not in seen and allowed(pair)
        ]
        target_count = len(pairs) + cli.random_pairs
        if len(pairs) + len(candidates) < target_count:
            raise SystemExit(
                "not enough source/dest pairs satisfy the distance constraint: "
                f"need {target_count}, available {len(pairs) + len(candidates)}"
            )
        rng = random.Random(cli.seed)
        pairs.extend(rng.sample(candidates, cli.random_pairs))

    if not pairs:
        raise SystemExit("provide --source/--dest, --random-pairs, or both")
    return pairs


def write_csv(path: Path | None, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    cli = parse_cli()
    args = load_args_from_config(cli.config)
    validate_args(args)
    parse_utc_datetime(args.start_utc)  # validate only; route geometry is relative time.

    if args.orbit_model != "circular":
        raise SystemExit("this sidecar currently measures circular orbit configs only")

    times = list(range(0, args.duration_s + 1, args.step_s))
    isl_config = make_isl_config(args)
    initial_views = circular_views(args, 0)
    pairs = select_pairs(cli, args.satellites, initial_views)
    layout = build_constellation_layout(
        initial_views,
        isl_config,
        walker_phase=args.walker_phase,
    )

    routes: dict[tuple[int, int], RouteNodes] = {}
    route_rows: list[dict[str, object]] = []

    for time_index, time_s in enumerate(times):
        views = circular_views(args, time_s)
        graph = build_isl_graph(
            views,
            isl_config,
            layout=layout,
            walker_phase=args.walker_phase,
        )
        for pair_id, (source, dest) in enumerate(pairs):
            route = shortest_route(graph, source, dest)
            nodes = None if route is None else route.nodes
            routes[(pair_id, time_index)] = nodes
            route_rows.append(
                {
                    "pair_id": pair_id,
                    "source": source,
                    "dest": dest,
                    "time_s": time_s,
                    "reachable": nodes is not None,
                    "hop_count": blank_none(hop_count(nodes)),
                    "route": route_text(nodes),
                    "source_dest_surface_distance_km_t0": (
                        f"{earth_surface_distance_km(initial_views[source], initial_views[dest]):.6f}"
                    ),
                }
            )

    delta_rows: list[dict[str, object]] = []
    for pair_id, (source, dest) in enumerate(pairs):
        for lag in cli.lags:
            for time_index, time_s in enumerate(times):
                other_index = time_index + lag
                if other_index >= len(times):
                    continue
                current = routes[(pair_id, time_index)]
                future = routes[(pair_id, other_index)]
                signed_length_delta = length_delta(current, future)
                delta_rows.append(
                    {
                        "pair_id": pair_id,
                        "source": source,
                        "dest": dest,
                        "lag_slots": lag,
                        "lag_s": lag * args.step_s,
                        "time_s": time_s,
                        "compare_time_s": times[other_index],
                        "reachable_t": current is not None,
                        "reachable_compare": future is not None,
                        "hop_count_t": blank_none(hop_count(current)),
                        "hop_count_compare": blank_none(hop_count(future)),
                        "hop_count_delta": blank_none(signed_length_delta),
                        "abs_hop_count_delta": blank_none(
                            None if signed_length_delta is None else abs(signed_length_delta)
                        ),
                        "node_jaccard_distance": blank_none(
                            node_jaccard_distance(current, future)
                        ),
                        "node_symdiff_count": blank_none(symdiff_count(current, future)),
                        "route_t": route_text(current),
                        "route_compare": route_text(future),
                        "source_dest_surface_distance_km_t0": (
                            f"{earth_surface_distance_km(initial_views[source], initial_views[dest]):.6f}"
                        ),
                    }
                )

    write_csv(cli.out, delta_rows)
    if cli.routes_out is not None:
        write_csv(cli.routes_out, route_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
