from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")
    return json.loads(summary_path.read_text())


def load_row(output_dir: Path, label: str) -> dict:
    summary = load_summary(output_dir)
    tasks = summary.get("tasks", {})
    energy = summary.get("energy", {})
    eclipse_energy = energy.get("eclipse", {})

    generated = int(tasks.get("generated", 0))
    completed = int(tasks.get("completed", 0))
    failed = int(tasks.get("failed", 0))
    pending = int(tasks.get("pending", 0))
    deferred = int(tasks.get("deferred", 0))

    eclipse_idle_j = float(eclipse_energy.get("idle_j", 0.0))
    eclipse_task_j = float(eclipse_energy.get("task_j", 0.0))
    eclipse_total_j = float(eclipse_energy.get("total_j", 0.0))

    if generated > 0:
        completion_rate_pct = 100.0 * completed / generated
        fail_rate_pct = 100.0 * failed / generated
        pending_rate_pct = 100.0 * pending / generated
        deferred_actions_per_task = deferred / generated
        eclipse_task_j_per_task = eclipse_task_j / generated
        eclipse_total_j_per_task = eclipse_total_j / generated
    else:
        completion_rate_pct = 0.0
        fail_rate_pct = 0.0
        pending_rate_pct = 0.0
        deferred_actions_per_task = 0.0
        eclipse_task_j_per_task = 0.0
        eclipse_total_j_per_task = 0.0

    return {
        "label": label,
        "output_dir": str(output_dir),
        "generated": generated,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "deferred_actions": deferred,
        "completion_rate_pct": completion_rate_pct,
        "fail_rate_pct": fail_rate_pct,
        "pending_rate_pct": pending_rate_pct,
        "deferred_actions_per_task": deferred_actions_per_task,
        "eclipse_side_idle_energy_j": eclipse_idle_j,
        "eclipse_side_task_energy_j": eclipse_task_j,
        "eclipse_side_total_energy_j": eclipse_total_j,
        "eclipse_side_task_energy_j_per_task": eclipse_task_j_per_task,
        "eclipse_side_total_energy_j_per_task": eclipse_total_j_per_task,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "output_dir",
                "generated",
                "completed",
                "failed",
                "pending",
                "deferred_actions",
                "completion_rate_pct",
                "fail_rate_pct",
                "pending_rate_pct",
                "deferred_actions_per_task",
                "eclipse_side_idle_energy_j",
                "eclipse_side_task_energy_j",
                "eclipse_side_total_energy_j",
                "eclipse_side_task_energy_j_per_task",
                "eclipse_side_total_energy_j_per_task",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt_energy_j(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MJ"
    if value >= 1_000:
        return f"{value / 1_000:.2f} kJ"
    return f"{value:.2f} J"


def write_outcome_svg(path: Path, rows: list[dict]) -> None:
    width = 900
    height = 460
    margin_l = 90
    margin_r = 40
    margin_t = 75
    margin_b = 110
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    bar_gap = 100
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / max(1, len(rows))
    x_axis_y = height - margin_b

    def pct_to_h(pct: float) -> float:
        return plot_h * pct / 100.0

    def fmt_pct(value: float) -> str:
        return f"{value:.2f}%"

    colors = {
        "completed": "#8ecae6",
        "failed": "#fb8500",
        "pending": "#adb5bd",
    }

    lines: list[str] = []

    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )
    lines.append('<rect width="100%" height="100%" fill="#0b1020"/>\n')

    lines.append(
        '<text x="24" y="34" fill="white" '
        'font-family="sans-serif" font-size="22">'
        "Effect of defer decision on task outcomes"
        "</text>\n"
    )

    lines.append(
        f'<line x1="{margin_l}" y1="{x_axis_y}" '
        f'x2="{width - margin_r}" y2="{x_axis_y}" stroke="#9fb3c8"/>\n'
    )
    lines.append(
        f'<line x1="{margin_l}" y1="{margin_t}" '
        f'x2="{margin_l}" y2="{x_axis_y}" stroke="#9fb3c8"/>\n'
    )

    for pct in [0, 25, 50, 75, 100]:
        y = x_axis_y - pct_to_h(pct)
        lines.append(
            f'<line x1="{margin_l - 5}" y1="{y:.1f}" '
            f'x2="{margin_l}" y2="{y:.1f}" stroke="#9fb3c8"/>\n'
        )
        lines.append(
            f'<text x="38" y="{y + 4:.1f}" fill="#cbd5e1" '
            f'font-family="sans-serif" font-size="12">{pct}%</text>\n'
        )

    legend_x = width - 330
    legend_y = 28
    legend_items = [
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("pending", "Pending"),
    ]
    for i, (key, label) in enumerate(legend_items):
        x = legend_x + i * 105
        lines.append(
            f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{colors[key]}"/>\n'
        )
        lines.append(
            f'<text x="{x + 20}" y="{legend_y + 12}" fill="#cbd5e1" '
            f'font-family="sans-serif" font-size="12">{label}</text>\n'
        )

    for i, row in enumerate(rows):
        x = margin_l + i * (bar_w + bar_gap)

        completed_h = pct_to_h(row["completion_rate_pct"])
        failed_h = pct_to_h(row["fail_rate_pct"])
        pending_h = pct_to_h(row["pending_rate_pct"])

        y = x_axis_y

        y -= completed_h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{completed_h:.1f}" '
            f'fill="{colors["completed"]}"/>\n'
        )

        y -= failed_h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{failed_h:.1f}" '
            f'fill="{colors["failed"]}"/>\n'
        )

        y -= pending_h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{pending_h:.1f}" '
            f'fill="{colors["pending"]}"/>\n'
        )

        lines.append(
            f'<rect x="{x:.1f}" y="{margin_t:.1f}" width="{bar_w:.1f}" height="{plot_h:.1f}" '
            f'fill="none" stroke="#e2e8f0" stroke-width="1"/>\n'
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 28}" fill="white" '
            f'font-family="sans-serif" font-size="14" text-anchor="middle">{row["label"]}</text>\n'
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 50}" fill="#cbd5e1" '
            f'font-family="sans-serif" font-size="12" text-anchor="middle">'
            f"fail {row['fail_rate_pct']:.2f}%</text>\n"
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 68}" fill="#cbd5e1" '
            f'font-family="sans-serif" font-size="12" text-anchor="middle">'
            f"defer actions {row['deferred_actions']}</text>\n"
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{margin_t - 10}" fill="white" '
            f'font-family="sans-serif" font-size="13" text-anchor="middle">'
            f"completed {row['completion_rate_pct']:.2f}%</text>\n"
        )

    lines.append("</svg>\n")
    path.write_text("".join(lines))


def write_eclipse_energy_svg(path: Path, rows: list[dict]) -> None:
    width = 900
    height = 420
    margin_l = 90
    margin_r = 40
    margin_t = 75
    margin_b = 95
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    values = [row["eclipse_side_task_energy_j"] for row in rows]
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0

    bar_gap = 100
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / max(1, len(rows))
    x_axis_y = height - margin_b

    def y_at(value: float) -> float:
        return margin_t + plot_h * (1.0 - value / max_value)

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )
    lines.append('<rect width="100%" height="100%" fill="#0b1020"/>\n')

    lines.append(
        '<text x="24" y="34" fill="white" '
        'font-family="sans-serif" font-size="22">'
        "Effect of defer decision on eclipse-side task energy consumption"
        "</text>\n"
    )

    lines.append(
        f'<line x1="{margin_l}" y1="{x_axis_y}" '
        f'x2="{width - margin_r}" y2="{x_axis_y}" stroke="#9fb3c8"/>\n'
    )
    lines.append(
        f'<line x1="{margin_l}" y1="{margin_t}" '
        f'x2="{margin_l}" y2="{x_axis_y}" stroke="#9fb3c8"/>\n'
    )

    lines.append(
        f'<text x="18" y="{margin_t + 5}" fill="#cbd5e1" '
        f'font-family="sans-serif" font-size="13">{fmt_energy_j(max_value)}</text>\n'
    )
    lines.append(
        f'<text x="38" y="{x_axis_y + 5}" fill="#cbd5e1" '
        f'font-family="sans-serif" font-size="13">0</text>\n'
    )

    for i, row in enumerate(rows):
        value = row["eclipse_side_task_energy_j"]
        x = margin_l + i * (bar_w + bar_gap)
        y = y_at(value)
        h = x_axis_y - y

        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
            f'rx="6" fill="#90be6d"/>\n'
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" fill="white" '
            f'font-family="sans-serif" font-size="13" text-anchor="middle">'
            f"{fmt_energy_j(value)}</text>\n"
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 28}" fill="white" '
            f'font-family="sans-serif" font-size="14" text-anchor="middle">{row["label"]}</text>\n'
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 48}" fill="#cbd5e1" '
            f'font-family="sans-serif" font-size="12" text-anchor="middle">'
            f"{fmt_energy_j(row['eclipse_side_task_energy_j_per_task'])} / task</text>\n"
        )

    lines.append("</svg>\n")
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare slack-aware scheduler with and without defer decisions."
    )
    parser.add_argument(
        "--with-defer",
        type=Path,
        required=True,
        help="Output directory of slack-aware run with defer enabled.",
    )
    parser.add_argument(
        "--without-defer",
        type=Path,
        required=True,
        help="Output directory of slack-aware run with defer disabled.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/defer_compare"),
        help="Directory for comparison CSV/SVG files.",
    )

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [
        load_row(args.with_defer, "with defer"),
        load_row(args.without_defer, "without defer"),
    ]

    write_csv(args.out / "defer_effect_comparison.csv", rows)
    write_outcome_svg(args.out / "defer_effect_outcome_rate.svg", rows)
    write_eclipse_energy_svg(args.out / "defer_effect_eclipse_energy.svg", rows)

    print(f"Wrote {args.out / 'defer_effect_comparison.csv'}")
    print(f"Wrote {args.out / 'defer_effect_outcome_rate.svg'}")
    print(f"Wrote {args.out / 'defer_effect_eclipse_energy.svg'}")

    for row in rows:
        print(
            f"{row['label']}: "
            f"completed={row['completed']}/{row['generated']} "
            f"({row['completion_rate_pct']:.2f}%), "
            f"failed={row['failed']}/{row['generated']} "
            f"({row['fail_rate_pct']:.2f}%), "
            f"pending={row['pending']}/{row['generated']} "
            f"({row['pending_rate_pct']:.2f}%), "
            f"deferred_actions={row['deferred_actions']}, "
            f"eclipse_task_energy={fmt_energy_j(row['eclipse_side_task_energy_j'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
