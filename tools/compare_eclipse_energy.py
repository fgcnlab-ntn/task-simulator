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


def scheduler_label(output_dir: Path, summary: dict) -> str:
    run_path = output_dir / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text())
        config = run.get("config", {})
        scheduler = config.get("scheduler", {})
        if isinstance(scheduler, dict):
            name = scheduler.get("name")
            if isinstance(name, str) and name:
                return name

    return output_dir.name


def load_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")
    return json.loads(summary_path.read_text())


def load_eclipse_energy(output_dir: Path) -> dict:
    summary = load_summary(output_dir)

    energy = summary.get("energy", {})
    if not isinstance(energy, dict):
        raise ValueError(f"{output_dir}: summary.json missing energy object")

    eclipse = energy.get("eclipse", {})
    if not isinstance(eclipse, dict):
        raise ValueError(f"{output_dir}: summary.json missing energy.eclipse object")

    return {
        "method": scheduler_label(output_dir, summary),
        "output_dir": str(output_dir),
        "eclipse_idle_j": float(eclipse.get("idle_j", 0.0)),
        "eclipse_task_j": float(eclipse.get("task_j", 0.0)),
        "eclipse_total_j": float(eclipse.get("total_j", 0.0)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "output_dir",
                "eclipse_idle_j",
                "eclipse_task_j",
                "eclipse_total_j",
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


def fmt_j(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MJ"
    if value >= 1_000:
        return f"{value / 1_000:.2f} kJ"
    return f"{value:.2f} J"


def write_plot(path: Path, rows: list[dict], metric: str, title: str) -> tuple[Path, Path]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    labels = [str(row["method"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    x = list(range(len(rows)))

    ax.bar(x, values, color="#8ecae6", edgecolor="#222222", linewidth=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Energy (J)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(values) * 1.18 if values else 1.0)
    ax.grid(True, axis="y", alpha=0.7)
    for xpos, value in zip(x, values):
        ax.text(xpos, value, fmt_j(value), ha="center", va="bottom", fontsize=9)
    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare eclipsed-side energy across scheduler output folders."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Output directories, e.g. output/compare/local output/compare/method1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/compare"),
        help="Directory for comparison CSV/PNG/PDF files",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rows = [load_eclipse_energy(run_dir) for run_dir in args.runs]

    write_csv(args.out / "eclipse_energy_comparison.csv", rows)

    task_written = write_plot(
        args.out / "eclipse_task_energy_comparison",
        rows,
        metric="eclipse_task_j",
        title="Eclipsed-side task energy consumption by scheduler",
    )
    total_written = write_plot(
        args.out / "eclipse_total_energy_comparison",
        rows,
        metric="eclipse_total_j",
        title="Eclipsed-side total energy by scheduler",
    )

    print(f"Wrote {args.out / 'eclipse_energy_comparison.csv'}")
    print(f"Wrote {format_written(task_written)}")
    print(f"Wrote {format_written(total_written)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
