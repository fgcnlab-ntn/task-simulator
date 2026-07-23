#!/usr/bin/env python3
"""Sweep CPU power for one satellite during one eclipse interval."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.cli import effective_run_config, load_standalone_json_config, validate_args
from satmulator.runlog import append_json_line, write_json
from tools.plot_output import save_png_pdf

DEFAULT_CONFIG = Path("configs/base/template.json")
DEFAULT_OUTPUT = Path("experiments/P_cut")
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
        write_energy_plot(
            run_args.out / f"p_cut_energy_{suffix}",
            results,
            safe_energy_j,
            safe_battery_pct,
        )
    write_combined_energy_plot(
        run_args.out / "p_cut_energy_combined",
        results,
        summaries,
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


def _pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "satmulator-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "font.size": 11,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def write_energy_plot(
    path: Path,
    results: list[dict[str, float | int | str]],
    safe_energy_j: float,
    safe_battery_pct: float,
) -> None:
    plt = _pyplot()
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
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=2.5)
    ax.axhline(idle_y, color="#d62728", linestyle="--", linewidth=1.8, label=f"idle baseline {idle_y:g} {unit_name}")
    ax.axhline(safe_y, color="#ff7f0e", linestyle=(0, (8, 4)), linewidth=1.8, label=f"safe budget {safe_y:g} {unit_name}; limit ~= {safe_power_w:.1f} W")
    ax.set_title(f"P_cut: CPU power vs one-satellite eclipse energy (safe battery {safe_battery_pct:g}%)", fontweight="bold")
    ax.set_xlabel("CPU power (W)")
    ax.set_ylabel(f"Total eclipse energy ({unit_name})")
    ax.grid(True, alpha=0.7)
    ax.legend(loc="best")
    save_png_pdf(fig, path)
    plt.close(fig)


def write_combined_energy_plot(
    path: Path,
    results: list[dict[str, float | int | str]],
    summaries: list[dict[str, float | int | str]],
) -> None:
    if not results or not summaries:
        return

    idle_energy_j = float(results[0]["idle_energy_j"])
    eclipse_seconds = float(results[0]["eclipse_duration_s"])
    cutoffs = [
        {
            "safe_battery_pct": float(summary["safe_battery_pct"]),
            "safe_energy_j": float(summary["energy_from_initial_to_safe_battery_j"]),
            "p_cut_w": p_cut_power_w(
                float(summary["energy_from_initial_to_safe_battery_j"]),
                idle_energy_j,
                eclipse_seconds,
            ),
        }
        for summary in summaries
    ]
    max_energy_j = max(
        [
            *[float(row["total_eclipse_energy_j"]) for row in results],
            *[float(cutoff["safe_energy_j"]) for cutoff in cutoffs],
        ],
        default=1.0,
    )
    unit_name, unit_divisor = energy_unit(max_energy_j)

    xs = [float(row["cpu_power_w"]) for row in results]
    ys = [float(row["total_eclipse_energy_j"]) / unit_divisor for row in results]
    max_x = max([*xs, *[float(cutoff["p_cut_w"]) for cutoff in cutoffs], 1.0])

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.8, 6.1))
    line_xs = [0.0, max_x]
    line_ys = [
        idle_energy_j / unit_divisor,
        (idle_energy_j + max_x * eclipse_seconds) / unit_divisor,
    ]
    ax.plot(line_xs, line_ys, color="#1f77b4", linewidth=2.5, label="energy model")
    ax.scatter(xs, ys, color="#1f77b4", s=28, zorder=3)

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for index, cutoff in enumerate(sorted(cutoffs, key=lambda item: float(item["safe_battery_pct"]))):
        color = colors[index % len(colors)]
        safe_pct = float(cutoff["safe_battery_pct"])
        safe_y = float(cutoff["safe_energy_j"]) / unit_divisor
        p_cut_w = float(cutoff["p_cut_w"])
        ax.axhline(safe_y, color=color, linestyle=(0, (8, 4)), linewidth=1.6)
        ax.axvline(p_cut_w, color=color, linestyle=(0, (3, 5)), linewidth=1.4)
        ax.scatter([p_cut_w], [safe_y], color=color, s=34, zorder=4, label=f"{safe_pct:g}% min: P_cut ~= {p_cut_w:.1f} W")

    ax.set_title("P_cut by minimum battery limit", fontweight="bold")
    ax.set_xlabel("CPU power (W)")
    ax.set_ylabel(f"Total eclipse energy ({unit_name})")
    ax.grid(True, alpha=0.7)
    ax.legend(loc="best")
    fig.text(0.5, 0.94, f"One satellite, {eclipse_seconds:g}s eclipse, idle power {idle_energy_j / eclipse_seconds:g} W", ha="center", fontsize=10, color="#666666")
    save_png_pdf(fig, path)
    plt.close(fig)


def p_cut_power_w(
    safe_energy_j: float,
    idle_energy_j: float,
    eclipse_seconds: float,
) -> float:
    if eclipse_seconds <= 0.0:
        raise ValueError("eclipse duration must be positive")
    return max(0.0, (safe_energy_j - idle_energy_j) / eclipse_seconds)


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
