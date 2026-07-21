#!/usr/bin/env python3
"""Plot per-satellite average compute loading across workload levels.

Each violin contains one observation per satellite.  An observation is the
satellite's total task execution time divided by the logged simulation time;
illumination state is deliberately ignored.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.plot_styles import EDGE_COLOR, method_style
from tools.plot_output import format_written, save_png_pdf


DEFAULT_METHODS = [
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix2",
    "method7",
]
LOADING_RE = re.compile(r"^r(?P<pct>\d+(?:\.\d+)?)$")
CACHE_NAME = "loading-average-compute-ratio.csv"


def _plotting():
    cache_dir = Path(tempfile.gettempdir()) / "satmulator-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

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


def loading_pct(group_dir: Path) -> float:
    match = LOADING_RE.fullmatch(group_dir.name)
    if match is None:
        raise ValueError(f"cannot parse loading percentage from {group_dir.name!r}")
    return float(match.group("pct"))


def loading_label(value: float) -> str:
    return f"{value:g}%"


def discover_loading_dirs(
    base_dir: Path,
    *,
    selected: set[str] | None = None,
) -> list[Path]:
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base directory does not exist: {base_dir}")
    result: list[Path] = []
    for path in base_dir.iterdir():
        if not path.is_dir() or LOADING_RE.fullmatch(path.name) is None:
            continue
        if selected is not None and path.name not in selected:
            continue
        result.append(path)
    return sorted(result, key=loading_pct)


def aggregate_average_loading(run_dir: Path) -> tuple[list[float], float]:
    """Return one execution-time/runtime ratio per satellite.

    The denominator is the elapsed simulation time covered by states.jsonl,
    rather than wall-clock time spent running the simulator.  The t=0 snapshot
    contributes neither execution time nor duration.
    """

    run_file = run_dir / "run.json"
    states_file = run_dir / "states.jsonl"
    if not run_file.exists():
        raise FileNotFoundError(f"missing run.json: {run_dir}")
    if not states_file.exists():
        raise FileNotFoundError(f"missing states.jsonl: {run_dir}")

    run = json.loads(run_file.read_text())
    satellite_count = int(run["config"]["orbit"]["satellites"])
    execution_s = [0.0 for _ in range(satellite_count)]
    first_time_s: float | None = None
    last_time_s: float | None = None

    with states_file.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            snapshot = json.loads(line)
            time_s = float(snapshot["time_s"])
            if first_time_s is None:
                first_time_s = time_s
            last_time_s = time_s
            if time_s == first_time_s:
                continue
            for satellite in snapshot.get("satellites", []):
                sat_id = int(satellite["id"])
                if not 0 <= sat_id < satellite_count:
                    raise ValueError(
                        f"{states_file}:{line_number}: invalid satellite id {sat_id}"
                    )
                task_load = satellite.get("task_load")
                if not isinstance(task_load, dict) or "compute_time_s" not in task_load:
                    raise ValueError(
                        f"{states_file}:{line_number}: satellite {sat_id} has no "
                        "task_load.compute_time_s"
                    )
                execution_s[sat_id] += float(task_load["compute_time_s"])

    if first_time_s is None or last_time_s is None:
        raise ValueError(f"no snapshots found in {states_file}")
    run_time_s = last_time_s - first_time_s
    if run_time_s <= 0.0:
        raise ValueError(f"non-positive logged run time in {states_file}: {run_time_s:g}")
    return [value / run_time_s for value in execution_s], run_time_s


def cache_path(run_dir: Path) -> Path:
    return run_dir / CACHE_NAME


def load_or_build_average_loading(
    run_dir: Path,
    *,
    use_cache: bool,
) -> tuple[list[float], float]:
    path = cache_path(run_dir)
    if use_cache and path.exists():
        values: list[float] = []
        run_time_s: float | None = None
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                values.append(float(row["average_loading"]))
                row_run_time_s = float(row["run_time_s"])
                if run_time_s is None:
                    run_time_s = row_run_time_s
                elif not math.isclose(run_time_s, row_run_time_s):
                    raise ValueError(f"inconsistent run_time_s values in {path}")
        if not values or run_time_s is None:
            raise ValueError(f"empty loading cache: {path}")
        return values, run_time_s

    values, run_time_s = aggregate_average_loading(run_dir)
    if use_cache:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sat_id", "execution_time_s", "run_time_s", "average_loading"],
            )
            writer.writeheader()
            for sat_id, value in enumerate(values):
                writer.writerow(
                    {
                        "sat_id": sat_id,
                        "execution_time_s": f"{value * run_time_s:.12g}",
                        "run_time_s": f"{run_time_s:.12g}",
                        "average_loading": f"{value:.12g}",
                    }
                )
    return values, run_time_s


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def collect_series(
    loading_dirs: list[Path],
    *,
    methods: list[str],
    use_cache: bool,
    workers: int,
) -> list[dict[str, object]]:
    jobs: list[tuple[Path, float, str, Path]] = []
    for loading_dir in loading_dirs:
        loading = loading_pct(loading_dir)
        for method in methods:
            run_dir = loading_dir / method
            jobs.append((loading_dir, loading, method, run_dir))

    results: list[tuple[list[float], float] | None] = [None] * len(jobs)
    if workers == 1:
        for index, (_, _, _, run_dir) in enumerate(jobs):
            source = (
                cache_path(run_dir)
                if use_cache and cache_path(run_dir).exists()
                else run_dir / "states.jsonl"
            )
            print(f"Reading {source}", file=sys.stderr, flush=True)
            results[index] = load_or_build_average_loading(
                run_dir,
                use_cache=use_cache,
            )
            print(f"Loaded {run_dir}", file=sys.stderr, flush=True)

    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as executor:
            futures: dict[cf.Future[tuple[list[float], float]], int] = {}
            for index, (_, _, _, run_dir) in enumerate(jobs):
                source = (
                    cache_path(run_dir)
                    if use_cache and cache_path(run_dir).exists()
                    else run_dir / "states.jsonl"
                )
                print(f"Reading {source}", file=sys.stderr, flush=True)
                future = executor.submit(
                    load_or_build_average_loading,
                    run_dir,
                    use_cache=use_cache,
                )
                futures[future] = index

            for future in cf.as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                print(f"Loaded {jobs[index][3]}", file=sys.stderr, flush=True)

    series: list[dict[str, object]] = []
    for (_, loading, method, run_dir), result in zip(jobs, results):
        if result is None:
            raise RuntimeError(f"loading aggregation did not finish: {run_dir}")
        values, run_time_s = result
        series.append(
            {
                "loading_pct": loading,
                "loading_label": loading_label(loading),
                "method": method,
                "method_label": method_style(method).label,
                "values": values,
                "run_time_s": run_time_s,
                "run_dir": run_dir,
            }
        )
    return series


def write_summary_csv(path: Path, series: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "loading_pct",
                "method",
                "satellites",
                "run_time_s",
                "mean",
                "median",
                "p95",
                "max",
            ],
        )
        writer.writeheader()
        for item in series:
            values = list(item["values"])
            writer.writerow(
                {
                    "loading_pct": f"{float(item['loading_pct']):.12g}",
                    "method": item["method"],
                    "satellites": len(values),
                    "run_time_s": f"{float(item['run_time_s']):.12g}",
                    "mean": f"{sum(values) / len(values):.12g}",
                    "median": f"{percentile(values, 50.0):.12g}",
                    "p95": f"{percentile(values, 95.0):.12g}",
                    "max": f"{max(values):.12g}",
                }
            )


def write_violin(
    path: Path,
    series: list[dict[str, object]],
    *,
    methods: list[str],
    title: str,
) -> tuple[Path, Path]:
    plt, sns = _plotting()
    from matplotlib.collections import PolyCollection
    from matplotlib.ticker import PercentFormatter

    loading_labels = list(dict.fromkeys(str(item["loading_label"]) for item in series))
    method_labels = [method_style(method).label for method in methods]
    palette = {
        method_style(method).label: method_style(method).color for method in methods
    }
    loading_column: list[str] = []
    method_column: list[str] = []
    value_column: list[float] = []
    for item in series:
        values = list(item["values"])
        loading_column.extend([str(item["loading_label"])] * len(values))
        method_column.extend([str(item["method_label"])] * len(values))
        value_column.extend(values)

    width = max(9.4, 1.15 * len(loading_labels) + 4.0)
    fig, ax = plt.subplots(figsize=(width, 5.8))
    sns.violinplot(
        data={
            "input_loading": loading_column,
            "method": method_column,
            "average_loading": value_column,
        },
        x="input_loading",
        y="average_loading",
        hue="method",
        order=loading_labels,
        hue_order=method_labels,
        palette=palette,
        density_norm="width",
        cut=0,
        inner="quart",
        linewidth=0.9,
        saturation=1.0,
        dodge=True,
        ax=ax,
    )
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_edgecolor(EDGE_COLOR)
            collection.set_alpha(0.65)

    ax.set_xlabel("Input task loading")
    ax.set_ylabel("Average compute loading (execution time / run time)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.7)
    ax.legend(title="Scheduler", ncols=min(len(methods), 5), loc="upper left")
    ax.margins(y=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped violin distributions of per-satellite average compute "
            "loading across input loading levels."
        )
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        type=Path,
        default=Path("output/final-loading-ratio"),
        help="Directory containing r30, r40, ... loading folders.",
    )
    parser.add_argument(
        "--loadings",
        nargs="+",
        help="Optional loading folder names to include, e.g. r30 r50 r70.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method subdirectories to include, in violin/legend order.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output figure path or prefix. Writes .png and .pdf.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Output method/loading statistics CSV. Defaults next to the figure.",
    )
    parser.add_argument(
        "--title",
        default="Average compute loading distribution by scheduler",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=f"Do not read or write per-run {CACHE_NAME} files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of run logs to aggregate in parallel (default: 8).",
    )
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    loading_dirs = discover_loading_dirs(
        args.base_dir,
        selected=set(args.loadings) if args.loadings is not None else None,
    )
    if not loading_dirs:
        raise ValueError(f"no loading directories found under {args.base_dir}")
    series = collect_series(
        loading_dirs,
        methods=args.methods,
        use_cache=not args.no_cache,
        workers=args.workers,
    )
    out = args.out or (args.base_dir / "compare" / "average-loading-violin")
    summary_csv = args.summary_csv or out.with_name(f"{out.stem}-summary.csv")
    written = write_violin(out, series, methods=args.methods, title=args.title)
    write_summary_csv(summary_csv, series)

    print(f"Wrote {format_written(written)}")
    print(f"Wrote {summary_csv}")
    if not args.no_cache:
        for item in series:
            print(f"Cached {cache_path(Path(item['run_dir']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
