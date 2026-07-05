from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def write_svg(path: Path, rows: list[dict], metric: str, title: str) -> None:
    width = 900
    height = 420
    margin_l = 90
    margin_r = 40
    margin_t = 60
    margin_b = 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    values = [row[metric] for row in rows]
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0

    bar_gap = 40
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / max(1, len(rows))

    def y_at(value: float) -> float:
        return margin_t + plot_h * (1.0 - value / max_value)

    def fmt_j(value: float) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} MJ"
        if value >= 1_000:
            return f"{value / 1_000:.2f} kJ"
        return f"{value:.2f} J"

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )
    lines.append('<rect width="100%" height="100%" fill="#0b1020"/>\n')
    lines.append(
        f'<text x="24" y="34" fill="white" font-family="sans-serif" font-size="22">{title}</text>\n'
    )

    x_axis_y = height - margin_b
    lines.append(
        f'<line x1="{margin_l}" y1="{x_axis_y}" x2="{width - margin_r}" y2="{x_axis_y}" stroke="#9fb3c8"/>\n'
    )
    lines.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{x_axis_y}" stroke="#9fb3c8"/>\n'
    )

    lines.append(
        f'<text x="18" y="{margin_t + 5}" fill="#cbd5e1" font-family="sans-serif" font-size="13">{fmt_j(max_value)}</text>\n'
    )
    lines.append(
        f'<text x="38" y="{x_axis_y + 5}" fill="#cbd5e1" font-family="sans-serif" font-size="13">0</text>\n'
    )

    for i, row in enumerate(rows):
        value = row[metric]
        x = margin_l + i * (bar_w + bar_gap)
        y = y_at(value)
        h = x_axis_y - y
        label = row["method"]

        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="#8ecae6"/>\n'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" fill="white" font-family="sans-serif" font-size="13" text-anchor="middle">{fmt_j(value)}</text>\n'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 28}" fill="white" font-family="sans-serif" font-size="14" text-anchor="middle">{label}</text>\n'
        )

    lines.append("</svg>\n")
    path.write_text("".join(lines))


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
        help="Directory for comparison CSV/SVG files",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rows = [load_eclipse_energy(run_dir) for run_dir in args.runs]

    write_csv(args.out / "eclipse_energy_comparison.csv", rows)

    write_svg(
        args.out / "eclipse_task_energy_comparison.svg",
        rows,
        metric="eclipse_task_j",
        title="Eclipsed-side task energy consumption by scheduler",
    )
    write_svg(
        args.out / "eclipse_total_energy_comparison.svg",
        rows,
        metric="eclipse_total_j",
        title="Eclipsed-side total energy by scheduler",
    )

    print(f"Wrote {args.out / 'eclipse_energy_comparison.csv'}")
    print(f"Wrote {args.out / 'eclipse_task_energy_comparison.svg'}")
    print(f"Wrote {args.out / 'eclipse_total_energy_comparison.svg'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
