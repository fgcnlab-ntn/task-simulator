from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from satmulator.plot_styles import method_style


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
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def default_method(run_dir: Path) -> str:
    return run_dir.name.replace("_", "-")


def load_fail_rate(run_dir: Path) -> tuple[int, float]:
    summary = json.loads((run_dir / "summary.json").read_text())
    tasks = summary["tasks"]
    generated = int(tasks["generated"])
    completed = int(tasks["completed"])
    pending = int(tasks["pending"])
    failed = int(tasks["failed"])
    if completed + pending + failed != generated:
        raise ValueError(
            f"{run_dir}: completed + pending + failed does not equal generated"
        )
    rate = 0.0 if generated == 0 else 100.0 * failed / generated
    return failed, rate


def output_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return "svg"
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpg"
    raise ValueError("output path must end with .svg, .png, .jpg, or .jpeg")


def write_figure(path: Path, series: list[dict[str, object]]) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 5.1))

    labels = [method_style(str(item["method"])).label for item in series]
    failed = [int(item["failed"]) for item in series]
    rates = [float(item["fail_rate"]) for item in series]
    x = list(range(len(series)))

    for xpos, item, rate in zip(x, series, rates):
        style = method_style(str(item["method"]))
        ax.bar(
            xpos,
            rate,
            0.68,
            facecolor="none",
            edgecolor=style.color,
            hatch=style.hatch,
            linewidth=1.2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Failed tasks (%)")
    ax.set_title("Task Fail Rate")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, axis="y", alpha=0.7)

    for xpos, rate, count in zip(x, rates, failed):
        if count == 0:
            continue
        ax.text(
            xpos,
            rate,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            color="#222222",
            fontsize=8,
            fontweight="bold",
        )

    fig.savefig(path, format=output_format(path), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare task fail rates across runs."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories with summary.json",
    )
    parser.add_argument("--labels", nargs="*", help="Optional method names for each run")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output figure path, ending with .svg, .png, .jpg, or .jpeg",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        raise ValueError("--labels count must match the number of run directories")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    series = []
    for index, run_dir in enumerate(args.runs):
        failed, fail_rate = load_fail_rate(run_dir)
        series.append(
            {
                "method": (
                    args.labels[index]
                    if args.labels is not None
                    else default_method(run_dir)
                ),
                "failed": failed,
                "fail_rate": fail_rate,
            }
        )

    write_figure(args.out, series)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
