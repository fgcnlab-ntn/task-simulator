from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.plot_styles import (
    EDGE_COLOR,
    canonical_method,
    method_style,
    ordered_methods,
)
from tools.plot_output import format_written, save_png_pdf


RUN_METHODS = [
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix",
    "method3",
]
STATE_ORDER = ("idle", "transmit", "run")
STATE_LABELS = {
    "idle": "Idle",
    "transmit": "Transmit",
    "run": "Run",
}
STATE_COLORS = {
    "idle": "#9E9E9E",
    "transmit": "#4C78A8",
    "run": "#F58518",
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
            "font.size": 10,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def loading_sort_key(path: Path) -> tuple[float, str]:
    match = LOADING_RE.match(path.name)
    if match:
        return float(match.group("pct")), path.name
    return float("inf"), path.name


def method_order_key(method: str) -> int:
    return ordered_methods(RUN_METHODS).index(canonical_method(method))


def discover_loading_dirs(base_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in base_dir.iterdir()
            if path.is_dir() and LOADING_RE.match(path.name)
        ),
        key=loading_sort_key,
    )


def select_loading_dirs(base_dir: Path, groups: list[str] | None) -> list[Path]:
    group_dirs = discover_loading_dirs(base_dir)
    if not groups:
        return group_dirs

    wanted = set(groups)
    selected = [path for path in group_dirs if path.name in wanted]
    missing = sorted(wanted - {path.name for path in selected})
    if missing:
        raise SystemExit(f"missing run groups under {base_dir}: {', '.join(missing)}")
    return selected


def discover_methods(group_dir: Path) -> list[str]:
    methods = [
        method
        for method in RUN_METHODS
        if (group_dir / method / "states.jsonl").exists()
    ]
    return sorted(methods, key=method_order_key)


def iter_snapshots(states_path: Path) -> Iterable[dict]:
    with states_path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def aggregate_run(run_dir: Path, slot_s: int) -> list[dict[str, float | int]]:
    totals_by_slot: dict[int, dict[str, float]] = defaultdict(
        lambda: {state: 0.0 for state in STATE_ORDER}
    )

    for snapshot in iter_snapshots(run_dir / "states.jsonl"):
        time_s = float(snapshot.get("time_s", 0.0))
        if time_s <= 0.0:
            continue
        slot = int((time_s - 1.0) // slot_s)
        totals = totals_by_slot[slot]
        for sat in snapshot.get("satellites", []):
            energy_delta = sat.get("energy_delta_j", {})
            task_load = sat.get("task_load", {})
            idle_j = float(energy_delta.get("consumed", 0.0))
            transmit_j = float(task_load.get("transmission_energy_j", 0.0))
            run_j = float(task_load.get("compute_energy_j", 0.0))
            if "compute_energy_j" not in task_load:
                task_j = float(energy_delta.get("tasks", 0.0))
                run_j = max(0.0, task_j - transmit_j)
            totals["idle"] += max(0.0, idle_j)
            totals["transmit"] += max(0.0, transmit_j)
            totals["run"] += max(0.0, run_j)

    rows: list[dict[str, float | int]] = []
    for slot in sorted(totals_by_slot):
        totals = totals_by_slot[slot]
        total_j = sum(totals.values())
        row: dict[str, float | int] = {
            "slot": slot,
            "start_s": slot * slot_s,
            "end_s": (slot + 1) * slot_s,
            "total_j": total_j,
        }
        for state in STATE_ORDER:
            row[f"{state}_j"] = totals[state]
            row[f"{state}_ratio"] = 0.0 if total_j == 0.0 else totals[state] / total_j
        rows.append(row)
    return rows


def collect_group_rows(group_dir: Path, slot_s: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method in discover_methods(group_dir):
        print(f"reading {group_dir.name}/{method}", flush=True)
        for row in aggregate_run(group_dir / method, slot_s):
            rows.append(
                {
                    "run": group_dir.name,
                    "method": method,
                    "label": method_style(method).label,
                    **row,
                }
            )
    return rows


def average_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["slot"]))].append(row)

    averaged: list[dict[str, float | int | str]] = []
    for (method, slot), items in sorted(
        grouped.items(),
        key=lambda item: (method_order_key(item[0][0]), item[0][1]),
    ):
        out: dict[str, float | int | str] = {
            "run": "average",
            "method": method,
            "label": method_style(method).label,
            "slot": slot,
            "start_s": float(items[0]["start_s"]),
            "end_s": float(items[0]["end_s"]),
            "total_j": sum(float(item["total_j"]) for item in items) / len(items),
        }
        for state in STATE_ORDER:
            out[f"{state}_j"] = sum(
                float(item[f"{state}_j"]) for item in items
            ) / len(items)
            out[f"{state}_ratio"] = (
                sum(float(item[f"{state}_ratio"]) for item in items) / len(items)
            )
        averaged.append(out)
    return averaged


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "method",
        "label",
        "slot",
        "start_s",
        "end_s",
        "idle_j",
        "transmit_j",
        "run_j",
        "total_j",
        "idle_ratio",
        "transmit_ratio",
        "run_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.12g}" if isinstance(value, float) else value
                    for key, value in row.items()
                    if key in fields
                }
            )


def plot_rows(
    rows: list[dict[str, float | int | str]],
    path: Path,
    *,
    title: str,
    slot_s: int,
) -> tuple[Path, Path]:
    plt = _pyplot()
    from matplotlib.patches import Patch

    methods = sorted({str(row["method"]) for row in rows}, key=method_order_key)
    slots = sorted({int(row["slot"]) for row in rows})
    rows_by_key = {(str(row["method"]), int(row["slot"])): row for row in rows}

    figure_width = max(9.5, min(24.0, 0.42 * len(slots) * max(1, len(methods))))
    fig, ax = plt.subplots(figsize=(figure_width, 5.8))

    group_width = 0.82
    bar_width = group_width / max(1, len(methods))
    x_positions = list(range(len(slots)))

    for method_index, method in enumerate(methods):
        style = method_style(method)
        offset = -group_width / 2.0 + bar_width * (method_index + 0.5)
        bottoms = [0.0 for _ in slots]
        for state in STATE_ORDER:
            heights = [
                100.0
                * float(rows_by_key.get((method, slot), {}).get(f"{state}_ratio", 0.0))
                for slot in slots
            ]
            ax.bar(
                [x + offset for x in x_positions],
                heights,
                bar_width,
                bottom=bottoms,
                color=STATE_COLORS[state],
                edgecolor=EDGE_COLOR,
                linewidth=0.5,
                alpha=style.alpha,
                hatch=style.hatch,
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    slot_hours = slot_s / 3600.0
    if slot_hours >= 1.0:
        labels = [f"{slot * slot_hours:g}" for slot in slots]
        xlabel = "Time slot start (h)"
    else:
        slot_minutes = slot_s / 60.0
        labels = [f"{slot * slot_minutes:g}" for slot in slots]
        xlabel = "Time slot start (min)"

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy consumption ratio (%)")
    ax.set_ylim(0, 100)
    fig.suptitle(title, fontweight="bold", y=0.98)
    ax.grid(True, axis="y", alpha=0.75)

    state_handles = [
        Patch(
            facecolor=STATE_COLORS[state],
            edgecolor=EDGE_COLOR,
            label=STATE_LABELS[state],
        )
        for state in STATE_ORDER
    ]
    method_handles = [
        Patch(
            facecolor="white",
            edgecolor=EDGE_COLOR,
            hatch=method_style(method).hatch,
            label=method_style(method).label,
        )
        for method in methods
    ]
    state_legend = ax.legend(
        handles=state_handles,
        ncol=len(state_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        bbox_transform=fig.transFigure,
    )
    ax.add_artist(state_legend)
    ax.legend(
        handles=method_handles,
        ncol=len(method_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        bbox_transform=fig.transFigure,
    )
    fig.subplots_adjust(top=0.80)

    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stacked bar charts of idle, transmit, and run energy ratios "
            "for final-loading-ratio experiments."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/final-loading-ratio"),
        help="Directory containing r30-r90 run groups.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/final-loading-ratio/compare"),
        help="Directory for generated CSV and figures.",
    )
    parser.add_argument(
        "--slot-minutes",
        type=float,
        default=60.0,
        help="Aggregate snapshots into this many minutes per plotted time slot.",
    )
    parser.add_argument(
        "--per-loading",
        action="store_true",
        help="Also write one plot per loading group, e.g. r30, r40, ...",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Optional subset of loading groups to process, e.g. --groups r30 r40.",
    )
    args = parser.parse_args()

    slot_s = int(round(args.slot_minutes * 60.0))
    if slot_s <= 0:
        raise SystemExit("--slot-minutes must be positive")

    group_dirs = select_loading_dirs(args.base_dir, args.groups)
    if not group_dirs:
        raise SystemExit(f"no rXX run groups found under {args.base_dir}")

    all_rows: list[dict[str, float | int | str]] = []
    written_paths: list[Path] = []
    for group_dir in group_dirs:
        group_rows = collect_group_rows(group_dir, slot_s)
        if not group_rows:
            continue
        all_rows.extend(group_rows)
        if args.per_loading:
            write_csv(args.out_dir / group_dir.name / "energy-state-ratio.csv", group_rows)
            written_paths.extend(
                plot_rows(
                    group_rows,
                    args.out_dir / group_dir.name / "energy-state-ratio",
                    title=f"{group_dir.name}: energy consumption ratio by time slot",
                    slot_s=slot_s,
                )
            )

    if not all_rows:
        raise SystemExit(f"no states.jsonl files found under {args.base_dir}")

    write_csv(args.out_dir / "energy-state-ratio.csv", all_rows)
    averaged = average_rows(all_rows)
    write_csv(args.out_dir / "energy-state-ratio-average.csv", averaged)
    written_paths.extend(
        plot_rows(
            averaged,
            args.out_dir / "energy-state-ratio-average",
            title="Average energy consumption ratio by time slot",
            slot_s=slot_s,
        )
    )
    print(f"wrote {format_written(written_paths)}")


if __name__ == "__main__":
    main()
