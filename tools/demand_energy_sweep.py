#!/usr/bin/env python3
"""Sweep fixed per-ground-point demand load against battery safety breaches."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.cli import load_standalone_json_config, run, validate_args
from satmulator.constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from satmulator.runlog import append_json_line, write_json
from tools.plot_output import save_png_pdf


DEFAULT_CONFIG = Path("configs/base/demand_points.json")
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
        write_heatmap_plot(args.out / "battery_breach_ratio_heatmap", results)
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
    values.setdefault("objective_alpha", 0.5)
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


def write_heatmap_plot(path: Path, results: list[dict[str, object]]) -> None:
    data_sizes = sorted({float(row["data_size_bits"]) for row in results})
    intervals = sorted({int(row["slot_interval_s"]) for row in results})
    values = {
        (float(row["data_size_bits"]), int(row["slot_interval_s"])): float(
            row["unique_breached_ratio"]
        )
        for row in results
    }
    matrix = [
        [values.get((data_size, interval), 0.0) for interval in intervals]
        for data_size in data_sizes
    ]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6.0, 1.35 * len(intervals)), max(4.0, 0.72 * len(data_sizes))))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
    ax.set_title("Fixed demand load vs unique battery breach ratio", fontweight="bold")
    ax.set_xlabel("task generation interval")
    ax.set_ylabel("input bits per demand point per slot")
    ax.set_xticks(range(len(intervals)))
    ax.set_xticklabels([f"{interval}s" for interval in intervals])
    ax.set_yticks(range(len(data_sizes)))
    ax.set_yticklabels([f"{data_size:g}" for data_size in data_sizes])
    for row_index, row in enumerate(matrix):
        for col_index, ratio in enumerate(row):
            ax.text(col_index, row_index, f"{ratio:.3f}", ha="center", va="center", color="#222222")
    fig.colorbar(image, ax=ax, label="breach ratio")
    save_png_pdf(fig, path)
    plt.close(fig)


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
        write_line_plot(prefix, next(iter(groups.values())))
        return

    for (constellation, slot_interval_s), rows in groups.items():
        path = prefix.with_name(
            f"{prefix.name}_{slug(constellation)}_{slot_interval_s}s"
        )
        write_line_plot(path, rows)


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


def write_line_plot(path: Path, rows: list[dict[str, object]]) -> None:
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
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for power_index, cpu_power in enumerate(cpu_powers):
        xs = []
        ys = []
        for index, data_size in enumerate(data_sizes):
            ratio = values.get((cpu_power, data_size))
            if ratio is None:
                continue
            xs.append(index)
            ys.append(ratio)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=f"{cpu_power:g}W")

    ax.set_title(f"{constellation}: battery breach ratio by CPU power", fontweight="bold")
    ax.text(0.5, 1.01, f"input data size per {slot_interval_s}s slot", transform=ax.transAxes, ha="center", color="#666666", fontsize=10)
    ax.set_xlabel("data size per time slot")
    ax.set_ylabel("breach ratio")
    ax.set_ylim(BREACH_RATIO_AXIS_MIN, BREACH_RATIO_AXIS_MAX)
    ax.set_xticks(range(len(data_sizes)))
    ax.set_xticklabels([format_bits(data_size) for data_size in data_sizes])
    ax.grid(True, alpha=0.7)
    ax.legend(title="CPU power", loc="best")
    save_png_pdf(fig, path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
