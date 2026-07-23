#!/usr/bin/env python3
"""Build a P_cut table from existing constellation eclipse-duration summaries."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.runlog import write_json
from tools.plot_output import save_png_pdf


DEFAULT_BATTERY_CONFIG = Path("configs/base/template.json")
DEFAULT_ECLIPSE_ROOT = Path("experiments/eclipse_time")
DEFAULT_OUTPUT = Path("experiments/P_cut")
DEFAULT_SAFE_BATTERY_PCTS = tuple(float(value) for value in range(90, -10, -10))
DEFAULT_CONSTELLATIONS = (
    ("Starlink 1584", "eclipse_time_starlink"),
    ("Kuiper 784", "eclipse_time_kuiper"),
    ("OneWeb 648", "eclipse_time_oneweb"),
    ("Iridium 66", "eclipse_time_iridium"),
)


@dataclass(frozen=True)
class BatteryModel:
    capacity_j: float
    initial_pct: float
    idle_w: float


@dataclass(frozen=True)
class ConstellationDuration:
    label: str
    run_dir: str
    p75_s: float
    p90_s: float
    mean_s: float
    median_s: float
    max_s: float
    intervals: int

    def duration_s(self, stat: str) -> float:
        return {
            "mean": self.mean_s,
            "median": self.median_s,
            "p75": self.p75_s,
            "p90": self.p90_s,
            "max": self.max_s,
        }[stat]


@dataclass(frozen=True)
class PCutCell:
    constellation: str
    safe_battery_pct: float
    eclipse_duration_stat: str
    eclipse_duration_s: float
    p_cut_w: float
    usable_energy_j: float
    idle_energy_j: float


def parse_safe_battery_pcts(value: str) -> list[float]:
    pcts = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        pct = float(raw)
        if not 0.0 <= pct <= 100.0:
            raise argparse.ArgumentTypeError(
                "safe battery percentages must be within [0, 100]"
            )
        pcts.append(pct)
    if not pcts:
        raise argparse.ArgumentTypeError(
            "at least one safe battery percentage is required"
        )
    return pcts


def load_battery_model(path: Path) -> BatteryModel:
    data = json.loads(path.read_text())
    battery = data["battery"]
    return BatteryModel(
        capacity_j=float(battery["capacity_j"]),
        initial_pct=float(battery["initial_pct"]),
        idle_w=float(battery["idle_w"]),
    )


def load_constellation_duration(
    eclipse_root: Path,
    label: str,
    run_dir: str,
) -> ConstellationDuration:
    path = eclipse_root / run_dir / "eclipse_time_summary.json"
    data = json.loads(path.read_text())
    return ConstellationDuration(
        label=label,
        run_dir=run_dir,
        p75_s=float(data["p75_s"]),
        p90_s=float(data["p90_s"]),
        mean_s=float(data["mean_s"]),
        median_s=float(data["median_s"]),
        max_s=float(data["max_s"]),
        intervals=int(data["intervals"]),
    )


def p_cut_power_w(
    *,
    battery: BatteryModel,
    safe_battery_pct: float,
    eclipse_duration_s: float,
    eclipse_duration_stat: str,
) -> PCutCell:
    if eclipse_duration_s <= 0.0:
        raise ValueError("eclipse duration must be positive")
    usable_energy_j = (
        battery.capacity_j * (battery.initial_pct - safe_battery_pct) / 100.0
    )
    idle_energy_j = battery.idle_w * eclipse_duration_s
    p_cut_w = max(0.0, (usable_energy_j - idle_energy_j) / eclipse_duration_s)
    return PCutCell(
        constellation="",
        safe_battery_pct=safe_battery_pct,
        eclipse_duration_stat=eclipse_duration_stat,
        eclipse_duration_s=eclipse_duration_s,
        p_cut_w=p_cut_w,
        usable_energy_j=usable_energy_j,
        idle_energy_j=idle_energy_j,
    )


def build_cells(
    *,
    battery: BatteryModel,
    constellations: list[ConstellationDuration],
    safe_battery_pcts: list[float],
    eclipse_duration_stat: str,
) -> list[PCutCell]:
    cells = []
    for safe_pct in safe_battery_pcts:
        for constellation in constellations:
            base = p_cut_power_w(
                battery=battery,
                safe_battery_pct=safe_pct,
                eclipse_duration_s=constellation.duration_s(eclipse_duration_stat),
                eclipse_duration_stat=eclipse_duration_stat,
            )
            cells.append(
                PCutCell(
                    constellation=constellation.label,
                    safe_battery_pct=base.safe_battery_pct,
                    eclipse_duration_stat=base.eclipse_duration_stat,
                    eclipse_duration_s=base.eclipse_duration_s,
                    p_cut_w=base.p_cut_w,
                    usable_energy_j=base.usable_energy_j,
                    idle_energy_j=base.idle_energy_j,
                )
            )
    return cells


def write_matrix_csv(
    path: Path,
    *,
    constellations: list[ConstellationDuration],
    safe_battery_pcts: list[float],
    cells: list[PCutCell],
) -> None:
    by_key = {
        (cell.safe_battery_pct, cell.constellation): cell
        for cell in cells
    }
    fields = ["E_safe_pct", *[f"{item.label} P_cut_W" for item in constellations]]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for safe_pct in safe_battery_pcts:
            row: dict[str, str | float] = {"E_safe_pct": f"{safe_pct:g}"}
            for constellation in constellations:
                row[f"{constellation.label} P_cut_W"] = (
                    f"{by_key[(safe_pct, constellation.label)].p_cut_w:.2f}"
                )
            writer.writerow(row)


def write_long_csv(path: Path, cells: list[PCutCell]) -> None:
    fields = [
        "constellation",
        "safe_battery_pct",
        "eclipse_duration_stat",
        "eclipse_duration_s",
        "p_cut_w",
        "usable_energy_j",
        "idle_energy_j",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "constellation": cell.constellation,
                    "safe_battery_pct": f"{cell.safe_battery_pct:g}",
                    "eclipse_duration_stat": cell.eclipse_duration_stat,
                    "eclipse_duration_s": f"{cell.eclipse_duration_s:g}",
                    "p_cut_w": f"{cell.p_cut_w:.2f}",
                    "usable_energy_j": f"{cell.usable_energy_j:.2f}",
                    "idle_energy_j": f"{cell.idle_energy_j:.2f}",
                }
            )


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
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def write_heatmap_plot(
    path: Path,
    *,
    constellations: list[ConstellationDuration],
    safe_battery_pcts: list[float],
    cells: list[PCutCell],
    eclipse_duration_stat: str,
) -> None:
    by_key = {
        (cell.safe_battery_pct, cell.constellation): cell
        for cell in cells
    }
    matrix = [
        [by_key[(safe_pct, constellation.label)].p_cut_w for constellation in constellations]
        for safe_pct in safe_battery_pcts
    ]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(constellations)), max(5.0, 0.48 * len(safe_battery_pcts))))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_title("P_cut by constellation and minimum battery", fontweight="bold")
    ax.text(0.5, 1.02, f"Eclipse duration uses {eclipse_duration_stat} from experiments/eclipse_time; cell values are P_cut in W", transform=ax.transAxes, ha="center", fontsize=10, color="#666666")
    ax.set_xticks(range(len(constellations)))
    ax.set_xticklabels([constellation.label for constellation in constellations])
    ax.set_yticks(range(len(safe_battery_pcts)))
    ax.set_yticklabels([f"E_safe {safe_pct:g}%" for safe_pct in safe_battery_pcts])
    ax.set_ylabel("minimum safe battery E_safe")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="#222222")
    for col_index, constellation in enumerate(constellations):
        ax.text(col_index, len(safe_battery_pcts) - 0.1, f"{eclipse_duration_stat} {constellation.duration_s(eclipse_duration_stat) / 60:g} min", ha="center", va="top", fontsize=8, color="#666666")
    fig.colorbar(image, ax=ax, label="P_cut (W)")
    save_png_pdf(fig, path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a P_cut table from constellation eclipse durations."
    )
    parser.add_argument("--battery-config", type=Path, default=DEFAULT_BATTERY_CONFIG)
    parser.add_argument("--eclipse-root", type=Path, default=DEFAULT_ECLIPSE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--eclipse-duration-stat",
        choices=("mean", "median", "p75", "p90", "max"),
        default="p90",
        help="eclipse duration statistic used for P_cut",
    )
    parser.add_argument(
        "--safe-battery-pcts",
        type=parse_safe_battery_pcts,
        default=list(DEFAULT_SAFE_BATTERY_PCTS),
        help="comma-separated minimum safe battery percentages",
    )
    args = parser.parse_args()

    battery = load_battery_model(args.battery_config)
    constellations = [
        load_constellation_duration(args.eclipse_root, label, run_dir)
        for label, run_dir in DEFAULT_CONSTELLATIONS
    ]
    cells = build_cells(
        battery=battery,
        constellations=constellations,
        safe_battery_pcts=args.safe_battery_pcts,
        eclipse_duration_stat=args.eclipse_duration_stat,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    output_prefix = f"p_cut_constellation_{args.eclipse_duration_stat}"
    write_matrix_csv(
        args.out / f"{output_prefix}_table.csv",
        constellations=constellations,
        safe_battery_pcts=args.safe_battery_pcts,
        cells=cells,
    )
    write_long_csv(args.out / f"{output_prefix}_long.csv", cells)
    write_heatmap_plot(
        args.out / f"{output_prefix}_heatmap",
        constellations=constellations,
        safe_battery_pcts=args.safe_battery_pcts,
        cells=cells,
        eclipse_duration_stat=args.eclipse_duration_stat,
    )
    write_json(
        args.out / f"{output_prefix}_table.json",
        {
            "schema_version": 1,
            "eclipse_duration_stat": args.eclipse_duration_stat,
            "battery": {
                "capacity_j": battery.capacity_j,
                "initial_pct": battery.initial_pct,
                "idle_w": battery.idle_w,
            },
            "constellations": [
                {
                    "label": item.label,
                    "run_dir": item.run_dir,
                    "p75_s": item.p75_s,
                    "p90_s": item.p90_s,
                    "mean_s": item.mean_s,
                    "median_s": item.median_s,
                    "max_s": item.max_s,
                    "intervals": item.intervals,
                }
                for item in constellations
            ],
            "safe_battery_pcts": args.safe_battery_pcts,
            "results": [
                {
                    "constellation": cell.constellation,
                    "safe_battery_pct": cell.safe_battery_pct,
                    "eclipse_duration_stat": cell.eclipse_duration_stat,
                    "eclipse_duration_s": cell.eclipse_duration_s,
                    "p_cut_w": cell.p_cut_w,
                    "usable_energy_j": cell.usable_energy_j,
                    "idle_energy_j": cell.idle_energy_j,
                }
                for cell in cells
            ],
        },
    )
    print(
        "wrote P_cut constellation "
        f"{args.eclipse_duration_stat} table to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
