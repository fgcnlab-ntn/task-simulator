#!/usr/bin/env python3
"""Sweep CPU power for one satellite during one eclipse interval."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.cli import effective_run_config, load_standalone_json_config, validate_args
from satmulator.runlog import append_json_line, write_json

DEFAULT_CONFIG = Path("configs/template.json")
DEFAULT_OUTPUT = Path("P_cut")
DEFAULT_ECLIPSE_DURATION_S = 32 * 60
DEFAULT_CPU_POWERS_W = tuple(float(power) for power in range(0, 35, 5))


def parse_cpu_powers(value: str) -> list[float]:
    powers = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        power = float(raw)
        if power < 0:
            raise argparse.ArgumentTypeError("CPU powers must be non-negative")
        powers.append(power)
    if not powers:
        raise argparse.ArgumentTypeError("at least one CPU power is required")
    return powers


def parse_safe_battery_pcts(value: str) -> list[float]:
    pcts = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        pct = float(raw)
        if not 0.0 <= pct <= 100.0:
            raise argparse.ArgumentTypeError(
                "safe battery percentages must be within [0, 100]"
            )
        pcts.append(pct)
    if not pcts:
        raise argparse.ArgumentTypeError(
            "at least one safe battery percentage is required"
        )
    return pcts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one satellite's eclipse energy when the CPU is fully active "
            "for the whole eclipse interval."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--eclipse-duration-s",
        type=int,
        default=DEFAULT_ECLIPSE_DURATION_S,
        help="single-satellite eclipse duration used for the energy calculation",
    )
    parser.add_argument(
        "--cpu-powers-w",
        type=parse_cpu_powers,
        default=list(DEFAULT_CPU_POWERS_W),
        help="comma-separated CPU power sweep in watts",
    )
    parser.add_argument(
        "--safe-battery-pct",
        type=float,
        default=None,
        help="override the minimum safe battery percentage for this experiment",
    )
    parser.add_argument(
        "--safe-battery-pcts",
        type=parse_safe_battery_pcts,
        default=None,
        help="comma-separated minimum safe battery percentages to plot",
    )
    args = parser.parse_args()

    if args.eclipse_duration_s <= 0:
        raise ValueError("eclipse duration must be positive")

    run_args = load_run_args(args)
    run_args.out.mkdir(parents=True, exist_ok=True)

    safe_battery_pcts = (
        args.safe_battery_pcts
        if args.safe_battery_pcts is not None
        else [run_args.battery_min_safe_pct]
    )
    results = energy_sweep(
        cpu_powers_w=args.cpu_powers_w,
        eclipse_duration_s=args.eclipse_duration_s,
        idle_w=run_args.idle_w,
    )
    base_summary = {
        "schema_version": 1,
        "scope": "single_satellite",
        "eclipse_duration_s": args.eclipse_duration_s,
        "idle_power_w": run_args.idle_w,
        "battery_capacity_j": run_args.battery_capacity_j,
        "initial_battery_pct": run_args.battery_initial_pct,
    }

    write_json(run_args.out / "run_config.json", effective_run_config(run_args))
    write_results_csv(run_args.out / "p_cut_results.csv", results)
    write_results_jsonl(run_args.out / "p_cut_results.jsonl", results)
    summaries = []
    for safe_battery_pct in safe_battery_pcts:
        safe_energy_j = energy_to_safe_battery_j(
            run_args,
            safe_battery_pct=safe_battery_pct,
        )
        summary = {
            **base_summary,
            "safe_battery_pct": safe_battery_pct,
            "energy_from_initial_to_safe_battery_j": safe_energy_j,
        }
        summaries.append(summary)
        suffix = safe_pct_suffix(safe_battery_pct)
        write_json(run_args.out / f"p_cut_summary_{suffix}.json", summary)
        write_energy_svg(
            run_args.out / f"p_cut_energy_{suffix}.svg",
            results,
            safe_energy_j,
            safe_battery_pct,
        )
    write_json(
        run_args.out / "p_cut_summary.json",
        {"schema_version": 1, "scenarios": summaries},
    )

    stale_samples = run_args.out / "p_cut_samples.jsonl"
    if stale_samples.exists():
        stale_samples.unlink()
    stale_legacy_plot = run_args.out / "p_cut_energy.svg"
    if stale_legacy_plot.exists():
        stale_legacy_plot.unlink()

    print(
        "P_cut experiment complete: "
        f"one satellite, eclipse duration {args.eclipse_duration_s}s, "
        f"{len(results)} CPU power points, "
        f"{len(safe_battery_pcts)} safe-battery plots -> {run_args.out}"
    )
    return 0


def load_run_args(args: argparse.Namespace) -> SimpleNamespace:
    values = load_standalone_json_config(args.config)
    if args.safe_battery_pct is not None:
        values["battery_min_safe_pct"] = args.safe_battery_pct
    values["task_enable"] = False
    values["out"] = args.out
    values["config"] = args.config
    values["plot_run"] = None
    values["run_name"] = "P_cut"
    values["run_description"] = (
        "Single-satellite CPU-power sweep with full CPU utilization during eclipse"
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


def energy_to_safe_battery_j(
    args: SimpleNamespace,
    *,
    safe_battery_pct: float | None = None,
) -> float:
    safe_pct = (
        args.battery_min_safe_pct
        if safe_battery_pct is None
        else safe_battery_pct
    )
    usable_fraction = (
        args.battery_initial_pct - safe_pct
    ) / 100.0
    return args.battery_capacity_j * usable_fraction


def safe_pct_suffix(safe_battery_pct: float) -> str:
    label = f"{safe_battery_pct:g}".replace(".", "p")
    return f"safe_{label}pct"


def energy_sweep(
    *,
    cpu_powers_w: list[float],
    eclipse_duration_s: float,
    idle_w: float,
) -> list[dict[str, float | int | str]]:
    results = []
    idle_energy_j = idle_w * eclipse_duration_s
    for index, cpu_power_w in enumerate(cpu_powers_w):
        cpu_energy_j = cpu_power_w * eclipse_duration_s
        results.append(
            {
                "schema_version": 1,
                "scope": "single_satellite",
                "index": index,
                "cpu_power_w": cpu_power_w,
                "eclipse_duration_s": eclipse_duration_s,
                "cpu_energy_j": cpu_energy_j,
                "idle_energy_j": idle_energy_j,
                "total_eclipse_energy_j": idle_energy_j + cpu_energy_j,
            }
        )
    return results


def write_results_csv(path: Path, results: list[dict[str, float | int | str]]) -> None:
    fields = [
        "scope",
        "cpu_power_w",
        "cpu_energy_j",
        "idle_energy_j",
        "total_eclipse_energy_j",
        "eclipse_duration_s",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})


def write_results_jsonl(path: Path, results: list[dict[str, float | int | str]]) -> None:
    with path.open("w") as output:
        for row in results:
            append_json_line(output, row)


def write_energy_svg(
    path: Path,
    results: list[dict[str, float | int | str]],
    safe_energy_j: float,
    safe_battery_pct: float,
) -> None:
    width = 900
    height = 560
    margin_left = 90
    margin_right = 30
    margin_top = 50
    margin_bottom = 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_energy_j = max(
        [safe_energy_j, *[float(row["total_eclipse_energy_j"]) for row in results]],
        default=1.0,
    )
    unit_name, unit_divisor = energy_unit(max_energy_j)
    xs = [float(row["cpu_power_w"]) for row in results]
    ys = [float(row["total_eclipse_energy_j"]) / unit_divisor for row in results]
    idle_y = (
        float(results[0]["idle_energy_j"]) / unit_divisor
        if results
        else 0.0
    )
    safe_y = safe_energy_j / unit_divisor
    max_x = max(max(xs) if xs else 1.0, 1.0)
    max_y = max(max([*ys, safe_y]) if ys else safe_y, 1.0)
    x_ticks, max_x = nice_tick_values(max_x, target_count=6)
    y_ticks, max_y = nice_tick_values(max_y, target_count=5)

    def sx(x: float) -> float:
        return margin_left + plot_w * x / max_x

    def sy(y: float) -> float:
        return margin_top + plot_h * (1.0 - y / max_y)

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
    circles = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="#1f77b4" />'
        for x, y in zip(xs, ys)
    )
    idle_line = (
        f'<line x1="{margin_left}" y1="{sy(idle_y):.2f}" '
        f'x2="{margin_left + plot_w}" y2="{sy(idle_y):.2f}" '
        'stroke="#d62728" stroke-width="2" stroke-dasharray="6 4" />'
        f'<text x="{margin_left + plot_w - 6}" y="{sy(idle_y) - 8:.2f}" '
        'text-anchor="end" fill="#d62728">'
        f'idle baseline {idle_y:g} {unit_name}'
        '</text>'
    )
    eclipse_seconds = (
        float(results[0]["eclipse_duration_s"])
        if results
        else 0.0
    )
    safe_power_w = (
        max(0.0, (safe_energy_j - float(results[0]["idle_energy_j"])) / eclipse_seconds)
        if eclipse_seconds > 0 and results
        else 0.0
    )
    safe_marker = (
        f'<line x1="{margin_left}" y1="{sy(safe_y):.2f}" '
        f'x2="{margin_left + plot_w}" y2="{sy(safe_y):.2f}" '
        'stroke="#ff7f0e" stroke-width="2" stroke-dasharray="10 5" />'
        f'<text x="{margin_left + plot_w - 6}" y="{sy(safe_y) - 8:.2f}" '
        'text-anchor="end" fill="#ff7f0e">'
        f'safe battery budget {safe_y:g} {unit_name}; limit ≈ {safe_power_w:.1f} W'
        '</text>'
    )

    x_axis = "\n".join(
        f'<line x1="{sx(t):.2f}" y1="{margin_top + plot_h}" x2="{sx(t):.2f}" y2="{margin_top + plot_h + 6}" stroke="#333" />'
        f'<text x="{sx(t):.2f}" y="{height - 45}" text-anchor="middle">{t:g}</text>'
        for t in x_ticks
    )
    y_axis = "\n".join(
        f'<line x1="{margin_left - 6}" y1="{sy(t):.2f}" x2="{margin_left}" y2="{sy(t):.2f}" stroke="#333" />'
        f'<text x="{margin_left - 10}" y="{sy(t) + 4:.2f}" text-anchor="end">{t:g}</text>'
        f'<line x1="{margin_left}" y1="{sy(t):.2f}" x2="{margin_left + plot_w}" y2="{sy(t):.2f}" stroke="#eee" />'
        for t in y_ticks
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; fill: #222; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .label {{ font-size: 15px; font-weight: 600; }}
  </style>
  <rect width="100%" height="100%" fill="white" />
  <text class="title" x="{width / 2}" y="28" text-anchor="middle">P_cut: CPU power vs one-satellite eclipse energy (safe battery {safe_battery_pct:g}%)</text>
  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ccc" />
  {y_axis}
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333" />
  {x_axis}
  {safe_marker}
  {idle_line}
  <polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="3" />
  {circles}
  <text class="label" x="{margin_left + plot_w / 2}" y="{height - 12}" text-anchor="middle">CPU power (W)</text>
  <text class="label" transform="translate(22 {margin_top + plot_h / 2}) rotate(-90)" text-anchor="middle">Total eclipse energy ({unit_name})</text>
</svg>
'''
    path.write_text(svg)


def energy_unit(max_energy_j: float) -> tuple[str, float]:
    if max_energy_j >= 1_000_000.0:
        return "MJ", 1_000_000.0
    if max_energy_j >= 1_000.0:
        return "kJ", 1_000.0
    return "J", 1.0


def nice_tick_values(
    max_value: float,
    *,
    target_count: int,
) -> tuple[list[float], float]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    rough_step = max_value / target_count
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    axis_max = step * math.ceil(max_value / step)
    ticks = [step * i for i in range(int(round(axis_max / step)) + 1)]
    return ticks, axis_max


if __name__ == "__main__":
    raise SystemExit(main())
