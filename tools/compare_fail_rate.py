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


def default_method_label(output_dir: Path) -> str:
    name = output_dir.name
    return name.replace("_", "-").replace("slack-aware", "slack-aware")


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


def write_svg(path: Path, rows: list[dict]) -> None:
    width = 900
    height = 420
    margin_l = 90
    margin_r = 40
    margin_t = 60
    margin_b = 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    values = [row["fail_rate_pct"] for row in rows]
    max_value = max(values) if values else 1.0

    # Keep the y-axis readable even when failure rates are small.
    if max_value <= 0:
        max_value = 1.0

    bar_gap = 40
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / max(1, len(rows))

    def y_at(value: float) -> float:
        return margin_t + plot_h * (1.0 - value / max_value)

    def fmt_pct(value: float) -> str:
        return f"{value:.2f}%"

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )
    lines.append('<rect width="100%" height="100%" fill="#0b1020"/>\n')

    lines.append(
        '<text x="24" y="34" fill="white" '
        'font-family="sans-serif" font-size="22">'
        "Task failure rate by scheduler"
        "</text>\n"
    )

    x_axis_y = height - margin_b

    # Axes.
    lines.append(
        f'<line x1="{margin_l}" y1="{x_axis_y}" '
        f'x2="{width - margin_r}" y2="{x_axis_y}" '
        f'stroke="#9fb3c8"/>\n'
    )
    lines.append(
        f'<line x1="{margin_l}" y1="{margin_t}" '
        f'x2="{margin_l}" y2="{x_axis_y}" '
        f'stroke="#9fb3c8"/>\n'
    )

    # Y-axis labels.
    lines.append(
        f'<text x="28" y="{margin_t + 5}" fill="#cbd5e1" '
        f'font-family="sans-serif" font-size="13">{fmt_pct(max_value)}</text>\n'
    )
    lines.append(
        f'<text x="58" y="{x_axis_y + 5}" fill="#cbd5e1" '
        f'font-family="sans-serif" font-size="13">0%</text>\n'
    )

    # Bars.
    for i, row in enumerate(rows):
        value = row["fail_rate_pct"]
        x = margin_l + i * (bar_w + bar_gap)
        y = y_at(value)
        h = x_axis_y - y
        label = row["method"]

        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{bar_w:.1f}" height="{h:.1f}" '
            f'rx="6" fill="#ffb703"/>\n'
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" '
            f'fill="white" font-family="sans-serif" font-size="13" '
            f'text-anchor="middle">{fmt_pct(value)}</text>\n'
        )

        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 28}" '
            f'fill="white" font-family="sans-serif" font-size="14" '
            f'text-anchor="middle">{label}</text>\n'
        )

        detail = f"{row['failed']}/{row['generated']}"
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{x_axis_y + 48}" '
            f'fill="#cbd5e1" font-family="sans-serif" font-size="12" '
            f'text-anchor="middle">{detail}</text>\n'
        )

    lines.append("</svg>\n")
    path.write_text("".join(lines))


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
            "output/compare/local output/compare/nearest_sunlit output/compare/slack_aware"
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
        help="Directory for comparison CSV/SVG files",
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
    write_svg(args.out / "fail_rate_comparison.svg", rows)

    print(f"Wrote {args.out / 'fail_rate_comparison.csv'}")
    print(f"Wrote {args.out / 'fail_rate_comparison.svg'}")

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
