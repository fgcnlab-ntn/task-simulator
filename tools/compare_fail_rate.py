from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plot_output import format_written, save_png_pdf


def load_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")
    return json.loads(summary_path.read_text())


def default_method_label(output_dir: Path) -> str:
    name = output_dir.name.replace("_", "-")
    if name == "phoenix2":
        return "phoenix"
    return name


def load_fail_rate(output_dir: Path, label: str | None = None) -> dict:
    summary = load_summary(output_dir)
    tasks = summary.get("tasks", {})

    generated = int(tasks.get("generated", 0))
    completed = int(tasks.get("completed", 0))
    failed = int(tasks.get("failed", 0))
    pending = int(tasks.get("pending", 0))
    deferred = int(tasks.get("deferred", 0))

    if generated > 0:
        fail_rate_pct = 100.0 * failed / generated
        completion_rate_pct = 100.0 * completed / generated
        pending_rate_pct = 100.0 * pending / generated
    else:
        fail_rate_pct = 0.0
        completion_rate_pct = 0.0
        pending_rate_pct = 0.0

    return {
        "method": label if label is not None else default_method_label(output_dir),
        "output_dir": str(output_dir),
        "generated": generated,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "deferred_actions": deferred,
        "completion_rate_pct": completion_rate_pct,
        "fail_rate_pct": fail_rate_pct,
        "pending_rate_pct": pending_rate_pct,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "output_dir",
                "generated",
                "completed",
                "failed",
                "pending",
                "deferred_actions",
                "completion_rate_pct",
                "fail_rate_pct",
                "pending_rate_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


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


def write_plot(path: Path, rows: list[dict]) -> tuple[Path, Path]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    labels = [str(row["method"]) for row in rows]
    values = [float(row["fail_rate_pct"]) for row in rows]
    x = list(range(len(rows)))

    ax.bar(x, values, color="#ffb703", edgecolor="#222222", linewidth=0.8)
    ax.set_title("Task failure rate by scheduler", fontweight="bold")
    ax.set_ylabel("Failed tasks (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(values) * 1.18 if values else 1.0)
    ax.grid(True, axis="y", alpha=0.7)
    for xpos, value, row in zip(x, values, rows):
        ax.text(xpos, value, f"{value:.2f}%\n{row['failed']}/{row['generated']}", ha="center", va="bottom", fontsize=9)
    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare task failure rates across scheduler output folders."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help=(
            "Output directories, e.g. "
            "output/compare/local output/compare/nearest_sunlit output/compare/method3"
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help=(
            "Optional labels for the runs. "
            "The number of labels must match the number of run directories."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/compare"),
        help="Directory for comparison CSV/PNG/PDF files",
    )

    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        raise ValueError("--labels count must match the number of run directories")

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, run_dir in enumerate(args.runs):
        label = args.labels[i] if args.labels is not None else None
        rows.append(load_fail_rate(run_dir, label=label))

    write_csv(args.out / "fail_rate_comparison.csv", rows)
    written = write_plot(args.out / "fail_rate_comparison", rows)

    print(f"Wrote {args.out / 'fail_rate_comparison.csv'}")
    print(f"Wrote {format_written(written)}")

    for row in rows:
        print(
            f"{row['method']}: "
            f"failed={row['failed']}, "
            f"generated={row['generated']}, "
            f"fail_rate={row['fail_rate_pct']:.2f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
