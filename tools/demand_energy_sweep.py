#!/usr/bin/env python3
"""Sweep fixed per-ground-point demand load against battery safety breaches."""

from __future__ import annotations

import argparse
import csv
import html
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
DEFAULT_OUTPUT = Path("experiments/breach_ratio/demand_energy_sweep")
DEFAULT_DATA_SIZES_BITS = (1.0e6, 1.0e7, 1.0e8)
DEFAULT_SLOT_INTERVALS_S = (30, 60, 120, 300)
BREACH_RATIO_AXIS_MIN = 0.0
BREACH_RATIO_AXIS_MAX = 1.0
BREACH_RATIO_TICK_STEP = 0.1


def config_float_values(value: object, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        values = [float(value)]
    elif isinstance(value, list):
        values = [float(item) for item in value]
    else:
        raise ValueError(f"{name} must be a number or a list of numbers")
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} values must be non-negative")
    return values


def constellation_label(values: dict[str, object]) -> str:
    name = str(values.get("run_name", "")).strip()
    if name:
        return name
    if values.get("orbit_model") == "circular":
        return f'{int(values["satellites"])} sats'
    tle_file = values.get("tle_file")
    return Path(tle_file).stem if tle_file else "tle"


def slug(value: object) -> str:
    text = str(value).strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "value"


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
    cpu_powers_w = config_float_values(
        base_values["satellite_cpu_power_w"], "compute.cpu_power_w"
    )
    constellation = constellation_label(base_values)

    results: list[dict[str, object]] = []
    for cpu_power_w in cpu_powers_w:
        for data_size_bits in args.data_sizes_bits:
            for slot_interval_s in args.slot_intervals_s:
                run_args = scenario_args(
                    base_values,
                    out=scenario_output_dir(
                        args.out,
                        data_size_bits,
                        slot_interval_s,
                        cpu_power_w=cpu_power_w,
                        constellation=constellation,
                    ),
                    data_size_bits=data_size_bits,
                    slot_interval_s=slot_interval_s,
                    duration_s=base_duration_s,
                    deadline_s=args.deadline_s,
                    global_total_demand=args.global_total_demand,
                    cpu_power_w=cpu_power_w,
                )
                run_args.out.mkdir(parents=True, exist_ok=True)
                print(
                    "running demand energy scenario: "
                    f"constellation={constellation}, "
                    f"cpu={cpu_power_w:g}W, "
                    f"data_size={data_size_bits:g} bits, "
                    f"slot={slot_interval_s}s -> {run_args.out}"
                )
                status = run(run_args)
                if status != 0:
                    raise SystemExit(status)
                results.append(
                    scenario_result(
                        run_args.out,
                        data_size_bits,
                        slot_interval_s,
                        cpu_power_w=cpu_power_w,
                        constellation=constellation,
                    )
                )

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
            "cpu_powers_w": cpu_powers_w,
            "constellation": constellation,
            "scenarios": results,
        },
    )
    if len(cpu_powers_w) == 1:
        write_heatmap_svg(args.out / "battery_breach_ratio_heatmap.svg", results)
    write_line_outputs(args.out / "battery_breach_ratio_line", results)

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
    cpu_power_w: float | None = None,
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
            "satellite_cpu_power_w": (
                float(values["satellite_cpu_power_w"])
                if cpu_power_w is None
                else cpu_power_w
            ),
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


def scenario_output_dir(
    root: Path,
    data_size_bits: float,
    slot_interval_s: int,
    *,
    cpu_power_w: float | None = None,
    constellation: str | None = None,
) -> Path:
    data_label = f"{data_size_bits:g}".replace("+", "").replace(".", "p")
    parts = [root, Path("runs")]
    if constellation is not None:
        parts.append(Path(slug(constellation)))
    if cpu_power_w is not None:
        cpu_label = f"{cpu_power_w:g}".replace("+", "").replace(".", "p")
        parts.append(Path(f"cpu_{cpu_label}w"))
    parts.extend([Path(f"data_{data_label}_bits"), Path(f"slot_{slot_interval_s}s")])
    path = parts[0]
    for part in parts[1:]:
        path /= part
    return path


def scenario_result(
    run_dir: Path,
    data_size_bits: float,
    slot_interval_s: int,
    *,
    cpu_power_w: float | None = None,
    constellation: str | None = None,
) -> dict[str, object]:
    summary = json.loads((run_dir / "summary.json").read_text())
    violations = summary["battery_violations"]
    tasks = summary["tasks"]
    return {
        "schema_version": 1,
        "constellation": constellation,
        "cpu_power_w": cpu_power_w,
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
        "constellation",
        "cpu_power_w",
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


def format_bits(value: float) -> str:
    bytes_value = value / 8.0
    units = ((1.0e9, "GB"), (1.0e6, "MB"), (1.0e3, "KB"))
    for scale, suffix in units:
        if abs(bytes_value) >= scale:
            scaled = bytes_value / scale
            return f"{scaled:g}{suffix}"
    return f"{bytes_value:g}B"


def line_chart_groups(
    results: list[dict[str, object]],
) -> dict[tuple[str, int], list[dict[str, object]]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in results:
        constellation = str(row.get("constellation") or "constellation")
        slot_interval_s = int(row["slot_interval_s"])
        groups.setdefault((constellation, slot_interval_s), []).append(row)
    return groups


def write_line_outputs(prefix: Path, results: list[dict[str, object]]) -> None:
    write_line_csv(prefix.with_suffix(".csv"), results)
    write_json(
        prefix.with_suffix(".json"),
        {
            "schema_version": 1,
            "chart": "battery_breach_ratio_line",
            "x": "data_size_bits",
            "y": "unique_breached_ratio",
            "series": "cpu_power_w",
            "rows": results,
        },
    )

    groups = line_chart_groups(results)
    if len(groups) == 1:
        write_line_svg(prefix.with_suffix(".svg"), next(iter(groups.values())))
        return

    for (constellation, slot_interval_s), rows in groups.items():
        path = prefix.with_name(
            f"{prefix.name}_{slug(constellation)}_{slot_interval_s}s.svg"
        )
        write_line_svg(path, rows)


def write_line_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "constellation",
        "cpu_power_w",
        "data_size_bits",
        "slot_interval_s",
        "unique_breached_ratio",
        "unique_breached_satellites",
        "unique_eclipse_breached_ratio",
        "unique_eclipse_breached_satellites",
        "tasks_generated",
        "tasks_completed",
        "tasks_failed",
        "tasks_pending",
        "run_dir",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            results,
            key=lambda item: (
                str(item.get("constellation") or ""),
                int(item["slot_interval_s"]),
                float(item.get("cpu_power_w") or 0.0),
                float(item["data_size_bits"]),
            ),
        ):
            writer.writerow({field: row[field] for field in fields})


def write_line_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width = 960
    height = 520
    margin_l = 82
    margin_r = 180
    margin_t = 72
    margin_b = 86
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    data_sizes = sorted({float(row["data_size_bits"]) for row in rows})
    cpu_powers = sorted({float(row.get("cpu_power_w") or 0.0) for row in rows})
    slot_interval_s = int(rows[0]["slot_interval_s"]) if rows else 0
    constellation = (
        str(rows[0].get("constellation") or "constellation")
        if rows
        else "constellation"
    )
    values = {
        (float(row.get("cpu_power_w") or 0.0), float(row["data_size_bits"])): float(
            row["unique_breached_ratio"]
        )
        for row in rows
    }
    y_min = BREACH_RATIO_AXIS_MIN
    y_max = BREACH_RATIO_AXIS_MAX
    y_range = y_max - y_min
    denom_x = max(1, len(data_sizes) - 1)

    def x_pos(index: int) -> float:
        return margin_l + index * plot_w / denom_x

    def y_pos(ratio: float) -> float:
        ratio = max(y_min, min(y_max, ratio))
        return margin_t + plot_h - (ratio - y_min) / y_range * plot_h

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    lines: list[str] = []
    esc_constellation = html.escape(constellation)
    title = f"{esc_constellation}: battery breach ratio by CPU power"
    subtitle = f"input data size per {slot_interval_s}s slot"

    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )
    lines.append(
        "  <style>text { font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif; fill: #222; font-size: 13px; } .title { font-size: 21px; font-weight: 700; } .label { font-size: 15px; font-weight: 600; }</style>\n"
    )
    lines.append('  <rect width="100%" height="100%" fill="white" />\n')
    lines.append(
        f'  <text class="title" x="{width / 2}" y="30" text-anchor="middle">{title}</text>\n'
    )
    lines.append(
        f'  <text x="{width / 2}" y="52" text-anchor="middle" fill="#666">{subtitle}</text>\n'
    )
    lines.append(
        f'  <line x1="{margin_l}" y1="{height - margin_b}" x2="{width - margin_r}" y2="{height - margin_b}" stroke="#999" />\n'
    )
    lines.append(
        f'  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height - margin_b}" stroke="#999" />\n'
    )

    tick_step = BREACH_RATIO_TICK_STEP
    tick_count = int(round((y_max - y_min) / tick_step))
    for tick in range(0, tick_count + 1):
        ratio = y_min + tick * tick_step
        y = y_pos(ratio)
        lines.append(
            f'  <line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#e5e5e5" />\n'
        )
        lines.append(
            f'  <text x="{margin_l - 10}" y="{y + 4:.1f}" text-anchor="end">{ratio:.1f}</text>\n'
        )

    for index, data_size in enumerate(data_sizes):
        x = x_pos(index)
        lines.append(
            f'  <line x1="{x:.1f}" y1="{height - margin_b}" x2="{x:.1f}" y2="{height - margin_b + 5}" stroke="#999" />\n'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="{height - margin_b + 24}" text-anchor="middle">{html.escape(format_bits(data_size))}</text>\n'
        )

    for power_index, cpu_power in enumerate(cpu_powers):
        color = colors[power_index % len(colors)]
        points = []
        for index, data_size in enumerate(data_sizes):
            ratio = values.get((cpu_power, data_size))
            if ratio is None:
                continue
            points.append((x_pos(index), y_pos(ratio), ratio, data_size))
        if not points:
            continue
        path_points = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
        lines.append(
            f'  <polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="2.5" />\n'
        )
        for x, y, ratio, data_size in points:
            title_text = html.escape(
                f"CPU {cpu_power:g}W, {format_bits(data_size)}, breach ratio {ratio:.4f}"
            )
            lines.append(
                f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{title_text}</title></circle>\n'
            )
        legend_y = margin_t + 24 * power_index
        lines.append(
            f'  <line x1="{width - margin_r + 28}" y1="{legend_y}" x2="{width - margin_r + 58}" y2="{legend_y}" stroke="{color}" stroke-width="2.5" />\n'
        )
        lines.append(
            f'  <text x="{width - margin_r + 66}" y="{legend_y + 4}">{cpu_power:g}W</text>\n'
        )

    lines.append(
        f'  <text class="label" x="{margin_l + plot_w / 2}" y="{height - 22}" text-anchor="middle">data size per time slot</text>\n'
    )
    lines.append(
        f'  <text class="label" transform="translate(24 {margin_t + plot_h / 2}) rotate(-90)" text-anchor="middle">breach ratio</text>\n'
    )
    lines.append("</svg>\n")
    path.write_text("".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
