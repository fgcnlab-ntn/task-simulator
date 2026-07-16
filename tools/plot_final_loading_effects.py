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

from satmulator.plot_styles import canonical_method, method_style, ordered_methods
from tools.plot_output import save_png_pdf

RUN_METHODS = [
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix",
    "method3",
    "method3mod",
]
METHOD_ORDER_INDEX = {
    method: index for index, method in enumerate(ordered_methods(RUN_METHODS))
}

LOADING_RE = re.compile(r"^r(?P<pct>\d+(?:\.\d+)?)$")


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


def loading_pct(group_dir: Path) -> float:
    match = LOADING_RE.match(group_dir.name)
    if not match:
        raise ValueError(f"cannot parse loading percentage from {group_dir.name!r}")
    return float(match.group("pct"))


def task_failure_ratio(summary: dict) -> float:
    tasks = summary["tasks"]
    generated = int(tasks["generated"])
    return 0.0 if generated == 0 else float(tasks["failed"]) / generated


def method_order_key(method: str) -> int:
    return METHOD_ORDER_INDEX[canonical_method(method)]


def collect_rows(base_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for group_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        if not any((group_dir / method / "run.json").exists() for method in RUN_METHODS):
            continue
        loading = loading_pct(group_dir)
        for method in RUN_METHODS:
            summary_file = group_dir / method / "summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text())
            objective = summary.get("objective", {})
            battery = summary.get("battery_violations", {})
            rows.append(
                {
                    "run": group_dir.name,
                    "method": method,
                    "label": method_style(method).label,
                    "loading_pct": loading,
                    "below_e_safe_ratio": float(
                        objective.get("avg_eclipse_unsafe_ratio", 0.0)
                    ),
                    "unique_below_e_safe_ratio": float(
                        battery.get("unique_breached_ratio", 0.0)
                    ),
                    "task_failure_ratio": task_failure_ratio(summary),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            float(row["loading_pct"]),
            method_order_key(str(row["method"])),
        ),
    )


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "method",
        "label",
        "loading_pct",
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

    methods = sorted(
        {str(row["method"]) for row in rows},
        key=method_order_key,
    )
    for method in methods:
        points = [row for row in rows if row["method"] == method]
        if not points:
            continue
        xs = [float(row["loading_pct"]) for row in points]
        ys = [100.0 * float(row[metric]) for row in points]
        style = method_style(method)
        label = style.label
        ax.plot(
            xs,
            ys,
            linewidth=2,
            markersize=5,
            label=label,
            color=style.color,
            alpha=style.alpha,
            marker=style.marker,
        )

    fig.suptitle(title, fontweight="bold", y=0.955)
    ax.set_xlabel("Task loading (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="both", alpha=0.75)
    x_values = [float(row["loading_pct"]) for row in rows]
    x_min = min(x_values)
    x_max = max(x_values)
    padding = max(1.0, (x_max - x_min) * 0.08)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_xticks(sorted({float(row["loading_pct"]) for row in rows}))
    ax.set_ylim(bottom=0)
    ax.legend(
        ncol=len(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.84)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_png_pdf(fig, path)
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
        raise SystemExit(f"no run groups found under {args.base_dir}")

    write_csv(rows, args.out_dir / "final-loading-effects.csv")
    plot_metric(
        rows,
        args.out_dir / "final-loading-below-e-safe",
        metric="below_e_safe_ratio",
        ylabel="Eclipse breach / eclipse satellites (%)",
        title="Task loading vs eclipse-side battery breaches",
    )
    plot_metric(
        rows,
        args.out_dir / "final-loading-task-fail-ratio",
        metric="task_failure_ratio",
        ylabel="Task failure ratio (%)",
        title="Task loading vs task failure ratio",
    )


if __name__ == "__main__":
    main()
