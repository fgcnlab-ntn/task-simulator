#!/usr/bin/env python3
"""Sweep fixed per-ground-point demand load against battery safety breaches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.cli import load_standalone_json_config, run, validate_args
from satmulator.constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from satmulator.runlog import append_json_line, write_json


DEFAULT_CONFIG = Path("configs/demand_points.json")
DEFAULT_OUTPUT = Path("output/demand_energy_sweep")
DEFAULT_DATA_SIZES_BITS = (1.0e6, 1.0e7, 1.0e8)
DEFAULT_SLOT_INTERVALS_S = (30, 60, 120, 300)


def parse_float_list(value: str) -> list[float]:
    values = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = float(raw)
        if parsed <= 0.0:
            raise argparse.ArgumentTypeError("values must be positive")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_int_list(value: str) -> list[int]:
    values = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic demand-point load sweep.  Each demand point "
            "generates one fixed-size task at every selected time slot, and "
            "the task is executed by its nearest visible satellite."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-sizes-bits",
        type=parse_float_list,
        default=list(DEFAULT_DATA_SIZES_BITS),
        help="comma-separated fixed input sizes to sweep",
    )
    parser.add_argument(
        "--slot-intervals-s",
        type=parse_int_list,
        default=list(DEFAULT_SLOT_INTERVALS_S),
        help="comma-separated task generation intervals to sweep",
    )
    parser.add_argument(
        "--duration-s",
        type=int,
        default=None,
        help="override run duration; defaults to one circular orbit rounded up to the step",
    )
    parser.add_argument(
        "--deadline-s",
        type=float,
        default=None,
        help="task deadline; defaults to a very large value so assigned tasks complete",
    )
    parser.add_argument(
        "--global-total-demand",
        action="store_true",
        help=(
            "treat each data size as the total global input per slot and "
            "split it across demand points by weight"
        ),
    )
    args = parser.parse_args()

    base_values = load_standalone_json_config(args.config)
    base_duration_s = (
        args.duration_s
        if args.duration_s is not None
        else one_circular_orbit_duration_s(base_values)
    )
    if base_duration_s <= 0:
        raise ValueError("duration must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for data_size_bits in args.data_sizes_bits:
        for slot_interval_s in args.slot_intervals_s:
            run_args = scenario_args(
                base_values,
                out=scenario_output_dir(args.out, data_size_bits, slot_interval_s),
                data_size_bits=data_size_bits,
                slot_interval_s=slot_interval_s,
                duration_s=base_duration_s,
                deadline_s=args.deadline_s,
                global_total_demand=args.global_total_demand,
            )
            run_args.out.mkdir(parents=True, exist_ok=True)
            print(
                "running demand energy scenario: "
                f"data_size={data_size_bits:g} bits, "
                f"slot={slot_interval_s}s -> {run_args.out}"
            )
            status = run(run_args)
            if status != 0:
                raise SystemExit(status)
            results.append(scenario_result(run_args.out, data_size_bits, slot_interval_s))

    write_results_csv(args.out / "demand_energy_sweep.csv", results)
    write_results_jsonl(args.out / "demand_energy_sweep.jsonl", results)
    write_json(
        args.out / "demand_energy_sweep_summary.json",
        {
            "schema_version": 1,
            "experiment": "demand_energy_sweep",
            "config": str(args.config),
            "duration_s": base_duration_s,
            "data_sizes_bits": args.data_sizes_bits,
            "slot_intervals_s": args.slot_intervals_s,
            "scenarios": results,
        },
    )
    write_heatmap_svg(args.out / "battery_breach_ratio_heatmap.svg", results)

    print(
        "Demand energy sweep complete: "
        f"{len(results)} scenarios, duration {base_duration_s}s -> {args.out}"
    )
    return 0


def one_circular_orbit_duration_s(values: dict[str, object]) -> int:
    if values["orbit_model"] != "circular":
        return int(values["duration_s"])
    radius_km = EARTH_RADIUS_KM + float(values["altitude_km"])
    period_s = 2.0 * math.pi * math.sqrt(radius_km**3 / EARTH_MU_KM3_S2)
    step_s = int(values["step_s"])
    return int(math.ceil(period_s / step_s) * step_s)


def scenario_args(
    base_values: dict[str, object],
    *,
    out: Path,
    data_size_bits: float,
    slot_interval_s: int,
    duration_s: int,
    deadline_s: float | None,
    global_total_demand: bool = False,
) -> SimpleNamespace:
    values = base_values.copy()
    values.update(
        {
            "run_name": "demand_energy_sweep",
            "run_description": (
                "Deterministic fixed-all demand-point battery breach sweep"
            ),
            "duration_s": duration_s,
            "task_interval_s": slot_interval_s,
            "task_generation_mode": (
                "demand-points-fixed-weighted-all"
                if global_total_demand
                else "demand-points-fixed-all"
            ),
            "task_input_bits": data_size_bits,
            "task_output_bits": 0.0,
            "task_deadline_s": 1.0e12 if deadline_s is None else deadline_s,
            "scheduler": "local",
            "out": out,
            "config": None,
            "plot_run": None,
        }
    )
    values["tle_file"] = None if values["tle_file"] is None else Path(values["tle_file"])
    values["task_demand_points_file"] = (
        None
        if values["task_demand_points_file"] is None
        else Path(values["task_demand_points_file"])
    )
    values["out"] = Path(values["out"])
    run_args = SimpleNamespace(**values)
    validate_args(run_args)
    return run_args


def scenario_output_dir(root: Path, data_size_bits: float, slot_interval_s: int) -> Path:
    data_label = f"{data_size_bits:g}".replace("+", "").replace(".", "p")
    return root / "runs" / f"data_{data_label}_bits" / f"slot_{slot_interval_s}s"


def scenario_result(
    run_dir: Path,
    data_size_bits: float,
    slot_interval_s: int,
) -> dict[str, object]:
    summary = json.loads((run_dir / "summary.json").read_text())
    violations = summary["battery_violations"]
    tasks = summary["tasks"]
    return {
        "schema_version": 1,
        "data_size_bits": data_size_bits,
        "slot_interval_s": slot_interval_s,
        "run_dir": str(run_dir),
        "unique_breached_satellites": violations["unique_breached_satellites"],
        "unique_breached_ratio": violations["unique_breached_ratio"],
        "unique_eclipse_breached_satellites": violations[
            "unique_eclipse_breached_satellites"
        ],
        "unique_eclipse_breached_ratio": violations["unique_eclipse_breached_ratio"],
        "tasks_generated": tasks["generated"],
        "tasks_completed": tasks["completed"],
        "tasks_failed": tasks["failed"],
        "tasks_pending": tasks["pending"],
    }


def write_results_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "data_size_bits",
        "slot_interval_s",
        "unique_breached_satellites",
        "unique_breached_ratio",
        "unique_eclipse_breached_satellites",
        "unique_eclipse_breached_ratio",
        "tasks_generated",
        "tasks_completed",
        "tasks_failed",
        "tasks_pending",
        "run_dir",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})


def write_results_jsonl(path: Path, results: list[dict[str, object]]) -> None:
    with path.open("w") as output:
        for row in results:
            append_json_line(output, row)


def write_heatmap_svg(path: Path, results: list[dict[str, object]]) -> None:
    data_sizes = sorted({float(row["data_size_bits"]) for row in results})
    intervals = sorted({int(row["slot_interval_s"]) for row in results})
    values = {
        (float(row["data_size_bits"]), int(row["slot_interval_s"])): float(
            row["unique_breached_ratio"]
        )
        for row in results
    }
    cell_w = 130
    cell_h = 70
    margin_left = 150
    margin_top = 70
    width = margin_left + cell_w * len(intervals) + 40
    height = margin_top + cell_h * len(data_sizes) + 80

    def color(ratio: float) -> str:
        ratio = max(0.0, min(1.0, ratio))
        red = int(255 * ratio)
        green = int(180 * (1.0 - ratio))
        blue = int(80 * (1.0 - ratio))
        return f"#{red:02x}{green:02x}{blue:02x}"

    cells = []
    for row_index, data_size in enumerate(data_sizes):
        y = margin_top + row_index * cell_h
        cells.append(
            f'<text x="{margin_left - 12}" y="{y + cell_h / 2 + 5:.1f}" '
            f'text-anchor="end">{data_size:g}</text>'
        )
        for col_index, interval in enumerate(intervals):
            x = margin_left + col_index * cell_w
            ratio = values.get((data_size, interval), 0.0)
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{color(ratio)}" stroke="white" />'
            )
            cells.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" '
                f'text-anchor="middle">{ratio:.3f}</text>'
            )
    headers = []
    for col_index, interval in enumerate(intervals):
        x = margin_left + col_index * cell_w + cell_w / 2
        headers.append(f'<text x="{x:.1f}" y="55" text-anchor="middle">{interval}s</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; fill: #222; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .label {{ font-size: 15px; font-weight: 600; }}
  </style>
  <rect width="100%" height="100%" fill="white" />
  <text class="title" x="{width / 2}" y="28" text-anchor="middle">Fixed demand load vs unique battery breach ratio</text>
  {''.join(headers)}
  {''.join(cells)}
  <text class="label" x="{margin_left + cell_w * len(intervals) / 2}" y="{height - 28}" text-anchor="middle">task generation interval</text>
  <text class="label" transform="translate(24 {margin_top + cell_h * len(data_sizes) / 2}) rotate(-90)" text-anchor="middle">input bits per demand point per slot</text>
</svg>
'''
    path.write_text(svg)


if __name__ == "__main__":
    raise SystemExit(main())
