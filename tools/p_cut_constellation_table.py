#!/usr/bin/env python3
"""Build a P_cut table from existing constellation eclipse-duration summaries."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.runlog import write_json


DEFAULT_BATTERY_CONFIG = Path("configs/template.json")
DEFAULT_ECLIPSE_ROOT = Path("experiments/eclipse_time")
DEFAULT_OUTPUT = Path("P_cut")
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


def write_heatmap_svg(
    path: Path,
    *,
    constellations: list[ConstellationDuration],
    safe_battery_pcts: list[float],
    cells: list[PCutCell],
    eclipse_duration_stat: str,
) -> None:
    cell_w = 135
    cell_h = 42
    margin_left = 120
    margin_top = 88
    margin_right = 30
    margin_bottom = 48
    width = margin_left + cell_w * len(constellations) + margin_right
    height = margin_top + cell_h * len(safe_battery_pcts) + margin_bottom

    by_key = {
        (cell.safe_battery_pct, cell.constellation): cell
        for cell in cells
    }
    values = [cell.p_cut_w for cell in cells]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)

    def color(value: float) -> str:
        ratio = (value - min_value) / span
        red = int(255 - 150 * ratio)
        green = int(238 - 80 * ratio)
        blue = int(230 - 190 * ratio)
        return f"#{red:02x}{green:02x}{blue:02x}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #222; font-size: 13px; }',
        ".title { font-size: 20px; font-weight: 700; }",
        ".axis { font-size: 14px; font-weight: 600; }",
        ".note { font-size: 12px; fill: #666; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white" />',
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">P_cut by constellation and minimum battery</text>',
        f'<text class="note" x="{width / 2}" y="50" text-anchor="middle">Eclipse duration uses {html.escape(eclipse_duration_stat)} from experiments/eclipse_time; cell values are P_cut in W</text>',
    ]

    for col, constellation in enumerate(constellations):
        x = margin_left + col * cell_w + cell_w / 2
        parts.append(
            f'<text class="axis" x="{x:.1f}" y="{margin_top - 14}" text-anchor="middle">{html.escape(constellation.label)}</text>'
        )
        parts.append(
            f'<text class="note" x="{x:.1f}" y="{margin_top + len(safe_battery_pcts) * cell_h + 22}" text-anchor="middle">{html.escape(eclipse_duration_stat)} {constellation.duration_s(eclipse_duration_stat) / 60:g} min</text>'
        )

    for row, safe_pct in enumerate(safe_battery_pcts):
        y = margin_top + row * cell_h
        parts.append(
            f'<text class="axis" x="{margin_left - 12}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="end">E_safe {safe_pct:g}%</text>'
        )
        for col, constellation in enumerate(constellations):
            cell = by_key[(safe_pct, constellation.label)]
            x = margin_left + col * cell_w
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color(cell.p_cut_w)}" stroke="#ccc" />'
            )
            parts.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="middle">{cell.p_cut_w:.2f}</text>'
            )

    parts.append(
        f'<text class="axis" transform="translate(24 {margin_top + len(safe_battery_pcts) * cell_h / 2}) rotate(-90)" text-anchor="middle">minimum safe battery E_safe</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


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
    write_heatmap_svg(
        args.out / f"{output_prefix}_heatmap.svg",
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
