from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.plot_styles import line_kwargs, method_style, ordered_methods

RUN_METHODS = [
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "method3",
    "phoenix",
]

CANONICAL_TO_RUN_METHOD = {
    "local-only": "local-only",
    "nearest-sunlit": "nearest-sunlit",
    "greedy-energy": "greedy-energy",
    "PHOENIX": "phoenix",
    "Method3": "method3",
}

_FINAL_RE = re.compile(r"^final-.*-(?P<tasks>\d+)x(?P<size>\d+(?:\.\d+)?)(?P<unit>[KMG]?)")
_UNIT_BITS = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}


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
            "axes.titleweight": "bold",
            "font.size": 11,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def loading_gb_per_min(final_dir: Path) -> float:
    match = _FINAL_RE.match(final_dir.name)
    if not match:
        raise ValueError(f"cannot parse loading from {final_dir.name!r}")
    tasks_per_step = float(match.group("tasks"))
    input_bits = float(match.group("size")) * _UNIT_BITS[match.group("unit")]

    run_files = sorted(final_dir.glob("*/run.json"))
    if not run_files:
        raise ValueError(f"{final_dir}: no method run.json files")
    run = json.loads(run_files[0].read_text())
    interval_s = float(run["config"]["task"]["interval_s"])

    return tasks_per_step * input_bits / 8.0 / 1e9 * (60.0 / interval_s)


def task_failure_ratio(summary: dict) -> float:
    tasks = summary["tasks"]
    generated = int(tasks["generated"])
    return 0.0 if generated == 0 else float(tasks["failed"]) / generated


def collect_rows(base_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for final_dir in sorted(base_dir.glob("final-*")):
        if not final_dir.is_dir():
            continue
        loading = loading_gb_per_min(final_dir)
        for method in RUN_METHODS:
            summary_file = final_dir / method / "summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text())
            objective = summary.get("objective", {})
            battery = summary.get("battery_violations", {})
            rows.append(
                {
                    "run": final_dir.name,
                    "method": method,
                    "loading_gb_per_min": loading,
                    "below_e_safe_ratio": float(
                        objective.get("avg_eclipse_unsafe_ratio", 0.0)
                    ),
                    "unique_below_e_safe_ratio": float(
                        battery.get("unique_breached_ratio", 0.0)
                    ),
                    "task_failure_ratio": task_failure_ratio(summary),
                }
            )
    return sorted(rows, key=lambda row: (float(row["loading_gb_per_min"]), str(row["method"])))


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "method",
        "loading_gb_per_min",
        "below_e_safe_ratio",
        "unique_below_e_safe_ratio",
        "task_failure_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.12g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def plot_metric(
    rows: list[dict[str, float | str]],
    path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.4))

    for canonical_method in ordered_methods(RUN_METHODS):
        method = CANONICAL_TO_RUN_METHOD[canonical_method]
        points = [row for row in rows if row["method"] == method]
        if not points:
            continue
        xs = [float(row["loading_gb_per_min"]) for row in points]
        ys = [100.0 * float(row[metric]) for row in points]
        kwargs = line_kwargs(method)
        ax.plot(
            xs,
            ys,
            linewidth=2,
            markersize=5,
            label=method_style(method).label,
            **kwargs,
        )

    fig.suptitle(title, fontweight="bold", y=0.955)
    ax.set_xlabel("Task loading (GB/min)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="both", alpha=0.75)
    ax.set_xlim(15, 30)
    ax.set_ylim(bottom=0)
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.84)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot final-run task loading against E_safe breaches and task failures."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("output"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/compare"))
    args = parser.parse_args()

    rows = collect_rows(args.base_dir)
    if not rows:
        raise SystemExit(f"no final-* runs found under {args.base_dir}")

    write_csv(rows, args.out_dir / "final-loading-effects.csv")
    for suffix in ("svg", "png"):
        plot_metric(
            rows,
            args.out_dir / f"final-loading-below-e-safe.{suffix}",
            metric="below_e_safe_ratio",
            ylabel="Eclipse breach / eclipse satellites (%)",
            title="Task loading vs eclipse-side battery breaches",
        )
        plot_metric(
            rows,
            args.out_dir / f"final-loading-task-fail-ratio.{suffix}",
            metric="task_failure_ratio",
            ylabel="Task failure ratio (%)",
            title="Task loading vs task failure ratio",
        )


if __name__ == "__main__":
    main()
