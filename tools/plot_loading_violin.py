from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.plot_styles import (
    EDGE_COLOR,
    canonical_method,
    method_style,
    ordered_methods,
)


METHOD_DIRS = {
    "local-only": "local-only",
    "nearest-sunlit": "nearest-sunlit",
    "greedy-energy": "greedy-energy",
    "PHOENIX": "phoenix",
    "Method3": "method3",
}


def _plotting():
    cache_dir = Path(tempfile.gettempdir()) / "satmulator-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.size": 11,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.size": 11,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        },
    )
    return plt, sns


def output_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return "svg"
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpg"
    raise ValueError("output path must end with .svg, .png, .jpg, or .jpeg")


def discover_run_dirs(base_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    for method in ordered_methods():
        run_dir = base_dir / METHOD_DIRS[method]
        if run_dir.is_dir():
            dirs.append(run_dir)
    return dirs


def compute_time_per_completed_task(run_dir: Path) -> float:
    run = json.loads((run_dir / "run.json").read_text())
    config = run["config"]
    task = config["task"]
    fixed_compute_time = task.get("compute_time_s")
    if fixed_compute_time is not None:
        return float(fixed_compute_time)

    choices = task.get("input_bits_choices")
    if isinstance(choices, list) and len(choices) == 1:
        input_bits = float(choices[0])
    else:
        input_bits = float(task["input_bits"])

    weights = task.get("input_bits_weights")
    if isinstance(choices, list) and len(choices) > 1:
        raise ValueError(
            f"{run_dir}: cannot derive exact compute loading from completed "
            "task counts when input_bits_choices has multiple values; use "
            "--metric completed-tasks or --metric task-energy-j"
        )
    if isinstance(weights, list) and len(weights) > 1:
        raise ValueError(
            f"{run_dir}: cannot derive exact compute loading from completed "
            "task counts when input_bits_weights has multiple values; use "
            "--metric completed-tasks or --metric task-energy-j"
        )

    compute = config["compute"]
    return input_bits * float(compute["cycles_per_input_bit"]) / float(
        compute["cpu_frequency_hz"]
    )


def cpu_power_w(run_dir: Path) -> float:
    run = json.loads((run_dir / "run.json").read_text())
    return float(run["config"]["compute"]["cpu_power_w"])


def step_s(run_dir: Path) -> float:
    run = json.loads((run_dir / "run.json").read_text())
    return float(run["config"]["time"]["step_s"])


def task_failure_ratio(run_dir: Path) -> float:
    summary = json.loads((run_dir / "summary.json").read_text())
    tasks = summary["tasks"]
    generated = int(tasks["generated"])
    if generated == 0:
        return 0.0
    return float(tasks["failed"]) / generated


def satellite_count(run_dir: Path) -> int:
    run = json.loads((run_dir / "run.json").read_text())
    return int(run["config"]["orbit"]["satellites"])


def aggregate_state_loading(
    run_dir: Path,
    *,
    metric: str,
) -> tuple[list[float], str]:
    count = satellite_count(run_dir)
    values = [0.0 for _ in range(count)]

    if metric == "completed-compute-s":
        scale = compute_time_per_completed_task(run_dir)
        field = "completed"
        ylabel = "Per-satellite total completed compute time (s)"
    elif metric == "completed-tasks":
        scale = 1.0
        field = "completed"
        ylabel = "Per-satellite completed tasks"
    elif metric == "task-energy-cpu-s":
        scale = 1.0 / cpu_power_w(run_dir)
        field = "task_energy_j"
        ylabel = "Per-satellite task energy, CPU-second equivalent (s)"
    elif metric == "task-energy-j":
        scale = 1.0
        field = "task_energy_j"
        ylabel = "Per-satellite total task energy (J)"
    else:
        raise ValueError(f"unknown metric: {metric}")

    with (run_dir / "states.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            snapshot = json.loads(line)
            for sat in snapshot.get("satellites", []):
                sat_id = int(sat["id"])
                if field == "task_energy_j":
                    values[sat_id] += float(sat["energy_delta_j"]["tasks"]) * scale
                else:
                    values[sat_id] += float(sat["task_counts"][field]) * scale

    return values, ylabel


def aggregate_illumination_relative_loading(run_dir: Path) -> tuple[dict[str, list[float]], str]:
    """Aggregate per-satellite loading ratios split by illumination state.

    New logs record task_load.compute_time_s directly.  Old logs did not split
    compute from transmission, so they fall back to task energy divided by CPU
    power as a CPU-second equivalent.
    The t=0 snapshot is skipped because it has no preceding interval.
    """

    count = satellite_count(run_dir)
    compute_seconds_by_state = {
        "sunlit": [0.0 for _ in range(count)],
        "eclipse": [0.0 for _ in range(count)],
    }
    duration_by_state = {
        "sunlit": [0.0 for _ in range(count)],
        "eclipse": [0.0 for _ in range(count)],
    }
    task_energy_to_cpu_seconds = 1.0 / cpu_power_w(run_dir)
    interval_s = step_s(run_dir)
    exact_compute_load = False

    with (run_dir / "states.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            snapshot = json.loads(line)
            if int(snapshot.get("time_s", 0)) == 0:
                continue
            for sat in snapshot.get("satellites", []):
                sat_id = int(sat["id"])
                state = "sunlit" if bool(sat["sunlit"]) else "eclipse"
                duration_by_state[state][sat_id] += interval_s
                task_load = sat.get("task_load")
                if isinstance(task_load, dict) and "compute_time_s" in task_load:
                    compute_seconds_by_state[state][sat_id] += float(
                        task_load["compute_time_s"]
                    )
                    exact_compute_load = True
                else:
                    compute_seconds_by_state[state][sat_id] += (
                        float(sat["energy_delta_j"]["tasks"])
                        * task_energy_to_cpu_seconds
                    )

    values_by_state: dict[str, list[float]] = {"sunlit": [], "eclipse": []}
    for state in ("sunlit", "eclipse"):
        for load, duration in zip(
            compute_seconds_by_state[state],
            duration_by_state[state],
        ):
            values_by_state[state].append(0.0 if duration == 0.0 else load / duration)

    ylabel = (
        "Task compute load ratio"
        if exact_compute_load
        else "Task-energy equivalent load ratio"
    )
    return values_by_state, ylabel


def cache_path(run_dir: Path, metric: str) -> Path:
    return run_dir / f"loading-{metric}.csv"


def illumination_cache_path(run_dir: Path) -> Path:
    return run_dir / "loading-illumination-compute-ratio.csv"


def load_or_build_loading(
    run_dir: Path,
    *,
    metric: str,
    use_cache: bool,
) -> tuple[list[float], str]:
    path = cache_path(run_dir, metric)
    ylabel = metric_ylabel(run_dir, metric)
    if use_cache and path.exists():
        values: list[float] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values.append(float(row["loading"]))
        return values, ylabel

    values, ylabel = aggregate_state_loading(run_dir, metric=metric)
    if use_cache:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sat_id", "loading"])
            writer.writeheader()
            for sat_id, value in enumerate(values):
                writer.writerow({"sat_id": sat_id, "loading": f"{value:.12g}"})
    return values, ylabel


def load_or_build_illumination_loading(
    run_dir: Path,
    *,
    use_cache: bool,
) -> tuple[dict[str, list[float]], str]:
    path = illumination_cache_path(run_dir)
    ylabel = "Task-energy equivalent load ratio"
    if use_cache and path.exists():
        values_by_state = {"sunlit": [], "eclipse": []}
        exact_compute_load = True
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values_by_state["sunlit"].append(float(row["sunlit"]))
                values_by_state["eclipse"].append(float(row["eclipse"]))
                exact_compute_load = row.get("source", "compute_time_s") == "compute_time_s"
        ylabel = (
            "Task compute load ratio"
            if exact_compute_load
            else "Task-energy equivalent load ratio"
        )
        return values_by_state, ylabel

    values_by_state, ylabel = aggregate_illumination_relative_loading(run_dir)
    if use_cache:
        with path.open("w", newline="") as f:
            source = (
                "compute_time_s"
                if ylabel == "Task compute load ratio"
                else "task_energy_cpu_s"
            )
            writer = csv.DictWriter(
                f,
                fieldnames=["sat_id", "sunlit", "eclipse", "source"],
            )
            writer.writeheader()
            for sat_id, (sunlit, eclipse) in enumerate(
                zip(values_by_state["sunlit"], values_by_state["eclipse"])
            ):
                writer.writerow(
                    {
                        "sat_id": sat_id,
                        "sunlit": f"{sunlit:.12g}",
                        "eclipse": f"{eclipse:.12g}",
                        "source": source,
                    }
                )
    return values_by_state, ylabel


def metric_ylabel(run_dir: Path, metric: str) -> str:
    if metric == "completed-compute-s":
        compute_time_per_completed_task(run_dir)
        return "Per-satellite total completed compute time (s)"
    if metric == "completed-tasks":
        return "Per-satellite completed tasks"
    if metric == "task-energy-cpu-s":
        cpu_power_w(run_dir)
        return "Per-satellite task energy, CPU-second equivalent (s)"
    if metric == "task-energy-j":
        return "Per-satellite total task energy (J)"
    raise ValueError(f"unknown metric: {metric}")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(values: list[float]) -> float:
    return percentile(values, 50.0)


def gini(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(max(0.0, value) for value in values)
    total = sum(ordered)
    if total == 0.0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    n = len(ordered)
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def write_summary_csv(path: Path, series: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "satellites",
                "mean",
                "median",
                "p95",
                "max",
                "gini",
                "failure_ratio",
            ],
        )
        writer.writeheader()
        for item in series:
            values = list(item["values"])
            writer.writerow(
                {
                    "method": item["method"],
                    "satellites": len(values),
                    "mean": f"{sum(values) / len(values):.12g}" if values else "nan",
                    "median": f"{median(values):.12g}",
                    "p95": f"{percentile(values, 95.0):.12g}",
                    "max": f"{max(values):.12g}" if values else "nan",
                    "gini": f"{gini(values):.12g}",
                    "failure_ratio": f"{float(item['failure_ratio']):.12g}",
                }
            )


def write_illumination_summary_csv(path: Path, series: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "illumination",
                "satellites",
                "mean",
                "median",
                "p95",
                "max",
                "gini",
                "failure_ratio",
            ],
        )
        writer.writeheader()
        for item in series:
            values_by_state = item["values_by_state"]
            if not isinstance(values_by_state, dict):
                raise ValueError("illumination series is missing values_by_state")
            for state in ("sunlit", "eclipse"):
                values = list(values_by_state[state])
                writer.writerow(
                    {
                        "method": item["method"],
                        "illumination": state,
                        "satellites": len(values),
                        "mean": f"{sum(values) / len(values):.12g}" if values else "nan",
                        "median": f"{median(values):.12g}",
                        "p95": f"{percentile(values, 95.0):.12g}",
                        "max": f"{max(values):.12g}" if values else "nan",
                        "gini": f"{gini(values):.12g}",
                        "failure_ratio": f"{float(item['failure_ratio']):.12g}",
                    }
                )


def write_violin(
    path: Path,
    series: list[dict[str, object]],
    *,
    ylabel: str,
    title: str,
    scale_width_by_count: bool,
    annotate_failure_rate: bool,
) -> None:
    plt, sns = _plotting()
    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    labels = []
    method_column: list[str] = []
    loading_column: list[float] = []
    palette: dict[str, str] = {}
    for item in series:
        label = str(item["label"])
        if annotate_failure_rate:
            label = f"{label}\nfail {100.0 * float(item['failure_ratio']):.1f}%"
        labels.append(label)
        palette[label] = method_style(str(item["method"])).color
        for value in item["values"]:
            method_column.append(label)
            loading_column.append(value)

    density_norm = "count" if scale_width_by_count else "width"
    sns.violinplot(
        data={"method": method_column, "loading": loading_column},
        x="method",
        y="loading",
        hue="method",
        order=labels,
        hue_order=labels,
        palette=palette,
        density_norm=density_norm,
        cut=0,
        inner=None,
        linewidth=1.0,
        saturation=1.0,
        dodge=False,
        legend=False,
        ax=ax,
    )
    for body, item in zip(ax.collections, series):
        body.set_edgecolor(EDGE_COLOR)
        body.set_alpha(method_style(str(item["method"])).alpha)

    values = [list(item["values"]) for item in series]
    positions = list(range(len(series)))
    medians = [median(group) for group in values]
    maxima = [max(group) for group in values]

    ax.scatter(positions, medians, marker="o", s=28, color="#222222", label="median", zorder=3)

    for xpos, max_value in zip(positions, maxima):
        ax.text(
            xpos,
            max_value,
            f"max {max_value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.94)
    ax.margins(y=0.08)

    fig.savefig(path, format=output_format(path), dpi=300)
    plt.close(fig)


def write_illumination_violin(
    path: Path,
    series: list[dict[str, object]],
    *,
    ylabel: str,
    title: str,
    annotate_failure_rate: bool,
) -> None:
    plt, sns = _plotting()
    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    labels: list[str] = []
    method_column: list[str] = []
    illumination_column: list[str] = []
    loading_column: list[float] = []
    for item in series:
        label = str(item["label"])
        if annotate_failure_rate:
            label = f"{label}\nfail {100.0 * float(item['failure_ratio']):.1f}%"
        labels.append(label)
        values_by_state = item["values_by_state"]
        if not isinstance(values_by_state, dict):
            raise ValueError("illumination series is missing values_by_state")
        for state, display in (("sunlit", "sunlit"), ("eclipse", "eclipse")):
            for value in values_by_state[state]:
                method_column.append(label)
                illumination_column.append(display)
                loading_column.append(value)

    sns.violinplot(
        data={
            "method": method_column,
            "illumination": illumination_column,
            "loading": loading_column,
        },
        x="method",
        y="loading",
        hue="illumination",
        order=labels,
        hue_order=["sunlit", "eclipse"],
        palette={"sunlit": "#FF7F0E", "eclipse": "#1F77B4"},
        split=True,
        density_norm="width",
        cut=0,
        inner=None,
        linewidth=1.0,
        saturation=1.0,
        ax=ax,
    )
    for body in ax.collections:
        body.set_edgecolor(EDGE_COLOR)
        body.set_alpha(0.62)

    offsets = {"sunlit": -0.18, "eclipse": 0.18}
    for index, item in enumerate(series):
        values_by_state = item["values_by_state"]
        if not isinstance(values_by_state, dict):
            raise ValueError("illumination series is missing values_by_state")
        for state in ("sunlit", "eclipse"):
            values = list(values_by_state[state])
            xpos = index + offsets[state]
            ax.scatter(
                [xpos],
                [median(values)],
                marker="o",
                s=24,
                color="#222222",
                zorder=3,
            )

    ax.scatter([], [], marker="o", s=24, color="#222222", label="median")
    ax.set_xticks(list(range(len(series))))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.94, ncols=2)
    ax.margins(y=0.08)

    fig.savefig(path, format=output_format(path), dpi=300)
    plt.close(fig)


def build_series(
    run_dirs: Iterable[Path],
    *,
    labels: list[str] | None,
    metric: str,
    use_cache: bool,
) -> tuple[list[dict[str, object]], str]:
    series: list[dict[str, object]] = []
    ylabel = ""
    for index, run_dir in enumerate(run_dirs):
        if not (run_dir / "states.jsonl").exists():
            raise FileNotFoundError(f"missing states.jsonl: {run_dir}")
        if not (run_dir / "run.json").exists():
            raise FileNotFoundError(f"missing run.json: {run_dir}")
        values, ylabel = load_or_build_loading(run_dir, metric=metric, use_cache=use_cache)
        method = canonical_method(run_dir.name)
        label = labels[index] if labels is not None else method_style(method).label
        series.append(
            {
                "method": method,
                "label": label,
                "values": values,
                "failure_ratio": task_failure_ratio(run_dir),
            }
        )
    return series, ylabel


def build_illumination_series(
    run_dirs: Iterable[Path],
    *,
    labels: list[str] | None,
    use_cache: bool,
) -> tuple[list[dict[str, object]], str]:
    series: list[dict[str, object]] = []
    ylabel = ""
    for index, run_dir in enumerate(run_dirs):
        if not (run_dir / "states.jsonl").exists():
            raise FileNotFoundError(f"missing states.jsonl: {run_dir}")
        if not (run_dir / "run.json").exists():
            raise FileNotFoundError(f"missing run.json: {run_dir}")
        values_by_state, ylabel = load_or_build_illumination_loading(
            run_dir,
            use_cache=use_cache,
        )
        method = canonical_method(run_dir.name)
        label = labels[index] if labels is not None else method_style(method).label
        series.append(
            {
                "method": method,
                "label": label,
                "values_by_state": values_by_state,
                "failure_ratio": task_failure_ratio(run_dir),
            }
        )
    return series, ylabel


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-satellite total loading distributions as one violin per "
            "scheduler method. Loading is aggregated from states.jsonl."
        )
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Directory containing one run subdirectory per method, e.g. output/final-...",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        help="Optional explicit run directories. Defaults to known method subdirectories under base_dir.",
    )
    parser.add_argument("--labels", nargs="*", help="Optional labels matching the run count")
    parser.add_argument(
        "--plot",
        choices=["illumination-relative", "total"],
        default="illumination-relative",
        help=(
            "Plot type. illumination-relative draws split sunlit/eclipse violins "
            "using task-energy-equivalent load divided by time in that state."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=[
            "task-energy-cpu-s",
            "task-energy-j",
            "completed-compute-s",
            "completed-tasks",
        ],
        default="task-energy-cpu-s",
        help="Loading metric to aggregate from states.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output figure path. Defaults to base_dir/loading-violin.png",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Output CSV for method-level statistics. Defaults next to the figure.",
    )
    parser.add_argument(
        "--title",
        help="Figure title",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write per-run loading-<metric>.csv cache files",
    )
    parser.add_argument(
        "--fixed-width",
        action="store_true",
        help="Use equal maximum violin widths instead of scaling width by satellite count",
    )
    parser.add_argument(
        "--no-failure-labels",
        action="store_true",
        help="Do not add each method's task failure rate under the x-axis label",
    )
    args = parser.parse_args()

    run_dirs = args.runs if args.runs is not None else discover_run_dirs(args.base_dir)
    if not run_dirs:
        raise ValueError(f"no run directories found under {args.base_dir}")
    if args.labels is not None and len(args.labels) != len(run_dirs):
        raise ValueError("--labels count must match the number of runs")

    default_name = (
        "loading-illumination-violin.png"
        if args.plot == "illumination-relative"
        else "loading-violin.png"
    )
    out = args.out or (args.base_dir / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or out.with_name(f"{out.stem}-summary.csv")

    if args.plot == "illumination-relative":
        series, ylabel = build_illumination_series(
            run_dirs,
            labels=args.labels,
            use_cache=not args.no_cache,
        )
        write_illumination_violin(
            out,
            series,
            ylabel=ylabel,
            title=args.title
            or "Sunlit/eclipsed relative loading distribution by scheduler",
            annotate_failure_rate=not args.no_failure_labels,
        )
        write_illumination_summary_csv(summary_csv, series)
    else:
        series, ylabel = build_series(
            run_dirs,
            labels=args.labels,
            metric=args.metric,
            use_cache=not args.no_cache,
        )
        write_violin(
            out,
            series,
            ylabel=ylabel,
            title=args.title
            or "Per-satellite total loading distribution by scheduler",
            scale_width_by_count=not args.fixed_width,
            annotate_failure_rate=not args.no_failure_labels,
        )
        write_summary_csv(summary_csv, series)
    print(f"Wrote {out}")
    print(f"Wrote {summary_csv}")
    for run_dir in run_dirs:
        if not args.no_cache:
            path = (
                illumination_cache_path(run_dir)
                if args.plot == "illumination-relative"
                else cache_path(run_dir, args.metric)
            )
            print(f"Cached {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
