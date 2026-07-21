#!/usr/bin/env python3
"""Plot one satellite for the standard five methods at one loading level."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plot_output import format_written
from tools.plot_satellite_loading_timeseries import (
    DEFAULT_ROLLING_MINUTES,
    load_satellite_samples,
    plot_satellite_loading,
)

METHODS = (
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix2",
    "method7",
)
LOADING_RE = re.compile(r"^r\d+(?:\.\d+)?$")


def normalize_loading(value: str) -> str:
    loading = value if value.startswith("r") else f"r{value}"
    if LOADING_RE.fullmatch(loading) is None:
        raise ValueError(f"invalid loading level: {value!r}; expected r100 or 100")
    return loading


def method_run_dirs(base_dir: Path, loading: str) -> list[tuple[str, Path]]:
    loading_dir = base_dir / normalize_loading(loading)
    runs = [(method, loading_dir / method) for method in METHODS]
    missing = [str(run_dir) for _, run_dir in runs if not run_dir.is_dir()]
    if missing:
        raise FileNotFoundError("missing method run directories: " + ", ".join(missing))
    return runs


def plot_methods(
    base_dir: Path,
    loading: str,
    satellite_id: int,
    *,
    output_dir: Path | None = None,
    rolling_minutes: float = DEFAULT_ROLLING_MINUTES,
    rolling_steps: int | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[Path]:
    loading = normalize_loading(loading)
    destination = output_dir or base_dir / loading / "satellite-loading-timeseries"
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for method, run_dir in method_run_dirs(base_dir, loading):
        print(f"Reading {run_dir}", flush=True)
        samples, step_s, run = load_satellite_samples(
            run_dir,
            satellite_id,
            start_s=start_s,
            end_s=end_s,
        )
        paths = plot_satellite_loading(
            samples,
            step_s,
            run,
            satellite_id,
            destination / f"{method}_satellite_{satellite_id}_loading_timeseries",
            rolling_minutes=rolling_minutes,
            rolling_steps=rolling_steps,
        )
        written.extend(paths)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one satellite for local-only, nearest-sunlit, greedy-energy, "
            "phoenix2, and method7."
        )
    )
    parser.add_argument("--satellite", type=int, required=True, help="satellite ID")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/final-loading-ratio"),
        help="directory containing r30, r40, ... groups",
    )
    parser.add_argument(
        "--loading",
        default="r100",
        help="loading group such as r100 or 100 (default: r100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="shared output directory for all five methods",
    )
    rolling_group = parser.add_mutually_exclusive_group()
    rolling_group.add_argument(
        "--rolling-minutes",
        type=float,
        default=DEFAULT_ROLLING_MINUTES,
        help="average window in minutes (default: 15)",
    )
    rolling_group.add_argument(
        "--rolling-steps",
        type=int,
        help="override the average window with a number of simulation steps",
    )
    parser.add_argument("--start-s", type=float, help="first logged time in seconds")
    parser.add_argument("--end-s", type=float, help="last logged time in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.satellite < 0:
        raise SystemExit("--satellite must be non-negative")
    if args.rolling_steps is not None and args.rolling_steps <= 0:
        raise SystemExit("--rolling-steps must be positive")
    if args.rolling_minutes <= 0.0:
        raise SystemExit("--rolling-minutes must be positive")

    paths = plot_methods(
        args.base_dir,
        args.loading,
        args.satellite,
        output_dir=args.output_dir,
        rolling_minutes=args.rolling_minutes,
        rolling_steps=args.rolling_steps,
        start_s=args.start_s,
        end_s=args.end_s,
    )
    print(f"Wrote {format_written(paths)}")


if __name__ == "__main__":
    main()
