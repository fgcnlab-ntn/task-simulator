from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

FIXED_X_MAX = 50.0
X_TICK_STEP = 5.0


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


def load_capacity_j(run_dir: Path) -> float:
    run = json.loads((run_dir / "run.json").read_text())
    return float(run["config"]["battery"]["capacity_j"])


def default_label(run_dir: Path) -> str:
    return run_dir.name.replace("_", "-")


def load_eclipse_dod_values(run_dir: Path) -> list[float]:
    capacity_j = load_capacity_j(run_dir)
    states_path = run_dir / "states.jsonl"
    values: list[float] = []

    with states_path.open() as f:
        for line in f:
            snapshot = json.loads(line)
            for sat in snapshot.get("satellites", []):
                if sat.get("sunlit", True):
                    continue
                battery_j = float(sat["battery_j"])
                dod_pct = 100.0 * (1.0 - battery_j / capacity_j)
                values.append(dod_pct)
    return values


def write_svg(
    path: Path,
    series: list[dict[str, object]],
    *,
    title: str = "CDF of Eclipse Satellite-Time DoD",
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    default_colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#B279A2",
        "#72B7B2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]

    for index, item in enumerate(series):
        values = sorted(item["values"])
        if not values:
            continue
        cdf = [(i + 1) / len(values) for i in range(len(values))]
        ax.step(
            values,
            cdf,
            where="post",
            linewidth=2.2,
            color=str(item.get("color") or default_colors[index % len(default_colors)]),
            linestyle=str(item.get("linestyle") or "-"),
            label=str(item["label"]),
        )

    ax.set_xlabel("Eclipse satellite-time DoD (%)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, FIXED_X_MAX)
    ax.set_xticks([tick for tick in range(0, int(FIXED_X_MAX) + 1, int(X_TICK_STEP))])
    ax.grid(True, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.94)
    fig.savefig(path, format="svg")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare eclipse satellite-time DoD CDFs across multiple runs."
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories with states.jsonl")
    parser.add_argument("--labels", nargs="*", help="Optional labels for each run")
    parser.add_argument("--colors", nargs="*", help="Optional colors for each run")
    parser.add_argument("--linestyles", nargs="*", help="Optional line styles for each run")
    parser.add_argument(
        "--title",
        default="CDF of Eclipse Satellite-Time DoD",
        help="Figure title",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output SVG path for the combined CDF figure",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        raise ValueError("--labels count must match the number of run directories")
    if args.colors is not None and len(args.colors) != len(args.runs):
        raise ValueError("--colors count must match the number of run directories")
    if args.linestyles is not None and len(args.linestyles) != len(args.runs):
        raise ValueError("--linestyles count must match the number of run directories")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    series = []
    for index, run_dir in enumerate(args.runs):
        label = args.labels[index] if args.labels is not None else default_label(run_dir)
        item = {"label": label, "values": load_eclipse_dod_values(run_dir)}
        if args.colors is not None:
            item["color"] = args.colors[index]
        if args.linestyles is not None:
            item["linestyle"] = args.linestyles[index]
        series.append(item)

    write_svg(args.out, series, title=args.title)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
