from __future__ import annotations

import csv
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

from .constants import EARTH_RADIUS_KM
from .models import SatelliteState, SnapshotContext, TaskRecord


def write_states_csv(path: Path, start: dt.datetime, all_steps: list[list[SatelliteState]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "time_iso",
                "sat_id",
                "name",
                "plane",
                "slot",
                "x_km",
                "y_km",
                "z_km",
                "vx_km_s",
                "vy_km_s",
                "vz_km_s",
                "lat_deg",
                "lon_deg",
                "elevation_km",
                "sunlit",
                "battery_j",
                "battery_pct",
                "harvested_j",
                "consumed_j",
                "safe_battery",
                "generated_tasks",
                "completed_tasks",
                "failed_tasks",
                "task_energy_j",
            ]
        )
        for states in all_steps:
            for s in states:
                writer.writerow(
                    [
                        s.time_s,
                        (start + dt.timedelta(seconds=s.time_s)).isoformat(),
                        s.sat_id,
                        s.name,
                        s.plane,
                        s.slot,
                        f"{s.x_km:.6f}",
                        f"{s.y_km:.6f}",
                        f"{s.z_km:.6f}",
                        f"{s.vx_km_s:.9f}",
                        f"{s.vy_km_s:.9f}",
                        f"{s.vz_km_s:.9f}",
                        "" if s.lat_deg is None else f"{s.lat_deg:.9f}",
                        "" if s.lon_deg is None else f"{s.lon_deg:.9f}",
                        "" if s.elevation_km is None else f"{s.elevation_km:.9f}",
                        int(s.sunlit),
                        f"{s.battery_j:.3f}",
                        f"{s.battery_pct:.6f}",
                        f"{s.harvested_j:.3f}",
                        f"{s.consumed_j:.3f}",
                        int(s.safe_battery),
                        s.generated_tasks,
                        s.completed_tasks,
                        s.failed_tasks,
                        f"{s.task_energy_j:.3f}",
                    ]
                )


def write_summary_csv(
    path: Path,
    all_steps: list[list[SatelliteState]],
    task_records_by_step: list[list[TaskRecord]] | None = None,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "sunlit",
            "eclipse",
            "satellites",
            "min_battery_pct",
            "avg_battery_pct",
            "unsafe_battery",
            "generated_tasks",
            "completed_tasks",
            "failed_tasks",
            "task_energy_j",
        ])
        records_by_step = task_records_by_step or [[] for _ in all_steps]
        generated_by_time = Counter(
            record.created_time_s for records in records_by_step for record in records
        )
        for states, records in zip(all_steps, records_by_step):
            sunlit = sum(1 for s in states if s.sunlit)
            min_battery_pct = min(s.battery_pct for s in states)
            avg_battery_pct = sum(s.battery_pct for s in states) / len(states)
            unsafe_battery = sum(1 for s in states if not s.safe_battery)
            unassigned_failures = sum(
                not record.completed and record.source_sat < 0 for record in records
            )
            generated_tasks = (
                generated_by_time[states[0].time_s]
                if task_records_by_step is not None
                else sum(s.generated_tasks for s in states)
            )
            completed_tasks = sum(s.completed_tasks for s in states)
            failed_tasks = sum(s.failed_tasks for s in states) + unassigned_failures
            task_energy_j = sum(s.task_energy_j for s in states)
            writer.writerow([
                states[0].time_s,
                sunlit,
                len(states) - sunlit,
                len(states),
                f"{min_battery_pct:.6f}",
                f"{avg_battery_pct:.6f}",
                unsafe_battery,
                generated_tasks,
                completed_tasks,
                failed_tasks,
                f"{task_energy_j:.3f}",
            ])


def write_tasks_csv(path: Path, task_records: list[TaskRecord]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "created_time_s",
            "source_sat",
            "target_sat",
            "mode",
            "lat_deg",
            "lon_deg",
            "cpu_cycles",
            "input_bits",
            "output_bits",
            "deadline_s",
            "waiting_time_s",
            "compute_time_s",
            "transmission_time_s",
            "total_time_s",
            "energy_j",
            "source_energy_j",
            "target_energy_j",
            "total_energy_j",
            "completed",
            "failed_reason",
        ])
        for task in task_records:
            writer.writerow([
                task.task_id,
                task.created_time_s,
                task.source_sat,
                task.target_sat,
                task.mode,
                "" if task.lat_deg is None else f"{task.lat_deg:.6f}",
                "" if task.lon_deg is None else f"{task.lon_deg:.6f}",
                f"{task.cpu_cycles:.6f}",
                f"{task.input_bits:.6f}",
                f"{task.output_bits:.6f}",
                f"{task.deadline_s:.6f}",
                f"{task.waiting_time_s:.6f}",
                f"{task.compute_time_s:.6f}",
                f"{task.transmission_time_s:.6f}",
                f"{task.total_time_s:.6f}",
                f"{task.energy_j:.6f}",
                f"{task.source_energy_j:.6f}",
                f"{task.target_energy_j:.6f}",
                f"{task.total_energy_j:.6f}",
                int(task.completed),
                task.failed_reason,
            ])


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#08111f"/>\n'
    )


def write_snapshot_svg(path: Path, states: list[SatelliteState], title: str, context: SnapshotContext) -> None:
    width = 900
    height = 700
    cx = width / 2
    cy = height / 2
    max_abs = max(max(abs(s.x_km), abs(s.y_km)) for s in states) * 1.15
    scale = min(width, height) * 0.44 / max_abs

    def px(x_km: float) -> float:
        return cx + x_km * scale

    def py(y_km: float) -> float:
        return cy - y_km * scale

    lines = [svg_header(width, height)]
    earth_r = EARTH_RADIUS_KM * scale
    lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{earth_r:.1f}" fill="#17365d" stroke="#7fb3ff" stroke-width="2"/>\n')
    lines.append(f'<text x="24" y="36" fill="white" font-family="sans-serif" font-size="22">{title}</text>\n')
    lines.append(f'<text x="24" y="60" fill="#9fb3c8" font-family="sans-serif" font-size="13">{context.projection_label}</text>\n')
    lines.append('<text x="24" y="82" fill="#ffd166" font-family="sans-serif" font-size="14">yellow = sunlit</text>\n')
    lines.append('<text x="24" y="102" fill="#6ea8fe" font-family="sans-serif" font-size="14">blue = eclipse</text>\n')

    if context.sun_xy_unit is not None:
        sx, sy = context.sun_xy_unit
        arrow_len = earth_r * 1.15
        ax1 = cx
        ay1 = cy
        ax2 = cx + sx * arrow_len
        ay2 = cy - sy * arrow_len
        dark_x = cx - sx * earth_r
        dark_y = cy + sy * earth_r
        dark_w = abs(sx) * earth_r * 3.0 + abs(sy) * earth_r * 0.7
        dark_h = abs(sy) * earth_r * 3.0 + abs(sx) * earth_r * 0.7
        lines.append(
            f'<rect x="{dark_x - dark_w / 2:.1f}" y="{dark_y - dark_h / 2:.1f}" '
            f'width="{dark_w:.1f}" height="{dark_h:.1f}" fill="#000000" opacity="0.12"/>\n'
        )
        lines.append(
            f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}" '
            'stroke="#ffd166" stroke-width="4" marker-end="url(#arrow)"/>\n'
        )
        lines.append(
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
            'orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L0,6 L9,3 z" fill="#ffd166"/></marker></defs>\n'
        )
        lines.append(f'<text x="{ax2 + 8:.1f}" y="{ay2:.1f}" fill="#ffd166" font-family="sans-serif" font-size="16">Sun direction</text>\n')

    for s in states:
        color = "#ffd166" if s.sunlit else "#6ea8fe"
        lines.append(
            f'<circle cx="{px(s.x_km):.1f}" cy="{py(s.y_km):.1f}" r="4" fill="{color}">'
            f'<title>{s.name} id {s.sat_id} battery {s.battery_pct:.1f}%</title></circle>\n'
        )

    lines.append("</svg>\n")
    path.write_text("".join(lines))


def write_summary_svg(path: Path, all_steps: list[list[SatelliteState]]) -> None:
    width = 900
    height = 360
    margin_l = 60
    margin_r = 30
    margin_t = 40
    margin_b = 45
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_t = max(states[0].time_s for states in all_steps) or 1
    total = len(all_steps[0])

    def x_at(t: int) -> float:
        return margin_l + (t / max_t) * plot_w

    def y_at(v: int) -> float:
        return margin_t + (1.0 - v / total) * plot_h

    sun_points = []
    eclipse_points = []
    for states in all_steps:
        t = states[0].time_s
        sun = sum(1 for s in states if s.sunlit)
        sun_points.append(f"{x_at(t):.1f},{y_at(sun):.1f}")
        eclipse_points.append(f"{x_at(t):.1f},{y_at(total - sun):.1f}")

    lines = [svg_header(width, height)]
    lines.append('<text x="24" y="28" fill="white" font-family="sans-serif" font-size="22">Sunlit / eclipse count over time</text>\n')
    lines.append(f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<polyline fill="none" stroke="#ffd166" stroke-width="3" points="{" ".join(sun_points)}"/>\n')
    lines.append(f'<polyline fill="none" stroke="#6ea8fe" stroke-width="3" points="{" ".join(eclipse_points)}"/>\n')
    lines.append('<text x="680" y="70" fill="#ffd166" font-family="sans-serif" font-size="15">sunlit</text>\n')
    lines.append('<text x="680" y="92" fill="#6ea8fe" font-family="sans-serif" font-size="15">eclipse</text>\n')
    lines.append(f'<text x="{margin_l}" y="{height-16}" fill="white" font-family="sans-serif" font-size="13">0s</text>\n')
    lines.append(f'<text x="{width-margin_r-70}" y="{height-16}" fill="white" font-family="sans-serif" font-size="13">{max_t}s</text>\n')
    lines.append(f'<text x="12" y="{margin_t+5}" fill="white" font-family="sans-serif" font-size="13">{total}</text>\n')
    lines.append('<text x="24" y="315" fill="#9fb3c8" font-family="sans-serif" font-size="13">time</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines))


def write_battery_svg(path: Path, all_steps: list[list[SatelliteState]]) -> None:
    width = 900
    height = 360
    margin_l = 60
    margin_r = 30
    margin_t = 40
    margin_b = 45
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_t = max(states[0].time_s for states in all_steps) or 1

    def x_at(t: int) -> float:
        return margin_l + (t / max_t) * plot_w

    def y_at(pct: float) -> float:
        return margin_t + (1.0 - pct / 100.0) * plot_h

    min_points = []
    avg_points = []
    for states in all_steps:
        t = states[0].time_s
        min_pct = min(s.battery_pct for s in states)
        avg_pct = sum(s.battery_pct for s in states) / len(states)
        min_points.append(f"{x_at(t):.1f},{y_at(min_pct):.1f}")
        avg_points.append(f"{x_at(t):.1f},{y_at(avg_pct):.1f}")

    lines = [svg_header(width, height)]
    lines.append('<text x="24" y="28" fill="white" font-family="sans-serif" font-size="22">Battery over time</text>\n')
    lines.append(f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<polyline fill="none" stroke="#06d6a0" stroke-width="3" points="{" ".join(avg_points)}"/>\n')
    lines.append(f'<polyline fill="none" stroke="#ef476f" stroke-width="3" points="{" ".join(min_points)}"/>\n')
    lines.append('<text x="680" y="70" fill="#06d6a0" font-family="sans-serif" font-size="15">average battery</text>\n')
    lines.append('<text x="680" y="92" fill="#ef476f" font-family="sans-serif" font-size="15">minimum battery</text>\n')
    lines.append(f'<text x="{margin_l}" y="{height-16}" fill="white" font-family="sans-serif" font-size="13">0s</text>\n')
    lines.append(f'<text x="{width-margin_r-70}" y="{height-16}" fill="white" font-family="sans-serif" font-size="13">{max_t}s</text>\n')
    lines.append(f'<text x="12" y="{margin_t+5}" fill="white" font-family="sans-serif" font-size="13">100%</text>\n')
    lines.append(f'<text x="24" y="{height-margin_b}" fill="white" font-family="sans-serif" font-size="13">0%</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines))


def write_task_svg(
    path: Path,
    all_steps: list[list[SatelliteState]],
    task_records_by_step: list[list[TaskRecord]] | None = None,
) -> None:
    width = 900
    height = 360
    margin_l = 60
    margin_r = 30
    margin_t = 40
    margin_b = 45
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_t = max(states[0].time_s for states in all_steps) or 1

    cumulative_completed = []
    cumulative_failed = []
    done = 0
    failed = 0
    max_count = 1
    records_by_step = task_records_by_step or [[] for _ in all_steps]
    for states, records in zip(all_steps, records_by_step):
        done += sum(s.completed_tasks for s in states)
        failed += sum(s.failed_tasks for s in states) + sum(
            not record.completed and record.source_sat < 0 for record in records
        )
        cumulative_completed.append((states[0].time_s, done))
        cumulative_failed.append((states[0].time_s, failed))
        max_count = max(max_count, done, failed)

    def x_at(t: int) -> float:
        return margin_l + (t / max_t) * plot_w

    def y_at(count: int) -> float:
        return margin_t + (1.0 - count / max_count) * plot_h

    completed_points = [f"{x_at(t):.1f},{y_at(v):.1f}" for t, v in cumulative_completed]
    failed_points = [f"{x_at(t):.1f},{y_at(v):.1f}" for t, v in cumulative_failed]

    lines = [svg_header(width, height)]
    lines.append('<text x="24" y="28" fill="white" font-family="sans-serif" font-size="22">Tasks over time</text>\n')
    lines.append(f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<polyline fill="none" stroke="#06d6a0" stroke-width="3" points="{" ".join(completed_points)}"/>\n')
    lines.append(f'<polyline fill="none" stroke="#ef476f" stroke-width="3" points="{" ".join(failed_points)}"/>\n')
    lines.append('<text x="680" y="70" fill="#06d6a0" font-family="sans-serif" font-size="15">completed</text>\n')
    lines.append('<text x="680" y="92" fill="#ef476f" font-family="sans-serif" font-size="15">failed</text>\n')
    lines.append(f'<text x="{margin_l}" y="{height-16}" fill="white" font-family="sans-serif" font-size="13">0s</text>\n')
    lines.append(f'<text x="{width-margin_r-70}" y="{height-16}" fill="white" font-family="sans-serif" font-size="13">{max_t}s</text>\n')
    lines.append(f'<text x="12" y="{margin_t+5}" fill="white" font-family="sans-serif" font-size="13">{max_count}</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines))


def write_sunlight_timeline_svg(path: Path, all_steps: list[list[SatelliteState]]) -> None:
    cell_w = 10
    cell_h = 8
    margin_l = 70
    margin_t = 44
    margin_r = 20
    margin_b = 34
    steps = len(all_steps)
    sats = len(all_steps[0])
    width = margin_l + steps * cell_w + margin_r
    height = margin_t + sats * cell_h + margin_b
    lines = [svg_header(width, height)]
    lines.append('<text x="20" y="28" fill="white" font-family="sans-serif" font-size="20">Sunlight timeline</text>\n')
    lines.append('<text x="20" y="42" fill="#9fb3c8" font-family="sans-serif" font-size="12">yellow = sunlit, blue = eclipse</text>\n')
    for x_idx, states in enumerate(all_steps):
        for y_idx, s in enumerate(states):
            color = "#ffd166" if s.sunlit else "#315f9f"
            x = margin_l + x_idx * cell_w
            y = margin_t + y_idx * cell_h
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}">'
                f'<title>t={s.time_s}s sat={s.sat_id} {"sunlit" if s.sunlit else "eclipse"}</title></rect>\n'
            )
    for tick in range(0, sats, max(1, sats // 8)):
        y = margin_t + tick * cell_h + cell_h
        lines.append(f'<text x="14" y="{y}" fill="white" font-family="sans-serif" font-size="11">sat {tick}</text>\n')
    lines.append(f'<text x="{margin_l}" y="{height-12}" fill="white" font-family="sans-serif" font-size="11">0s</text>\n')
    lines.append(f'<text x="{max(margin_l, width-90)}" y="{height-12}" fill="white" font-family="sans-serif" font-size="11">{all_steps[-1][0].time_s}s</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines))


def battery_color(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    if pct >= 50:
        t = (pct - 50) / 50
        r = int(255 * (1 - t) + 6 * t)
        g = int(209 * (1 - t) + 214 * t)
        b = int(102 * (1 - t) + 160 * t)
    else:
        t = pct / 50
        r = int(239 * (1 - t) + 255 * t)
        g = int(71 * (1 - t) + 209 * t)
        b = int(111 * (1 - t) + 102 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_battery_timeline_svg(path: Path, all_steps: list[list[SatelliteState]]) -> None:
    cell_w = 10
    cell_h = 8
    margin_l = 70
    margin_t = 44
    margin_r = 20
    margin_b = 34
    steps = len(all_steps)
    sats = len(all_steps[0])
    width = margin_l + steps * cell_w + margin_r
    height = margin_t + sats * cell_h + margin_b
    lines = [svg_header(width, height)]
    lines.append('<text x="20" y="28" fill="white" font-family="sans-serif" font-size="20">Battery timeline</text>\n')
    lines.append('<text x="20" y="42" fill="#9fb3c8" font-family="sans-serif" font-size="12">red = low, yellow = mid, green = high</text>\n')
    for x_idx, states in enumerate(all_steps):
        for y_idx, s in enumerate(states):
            x = margin_l + x_idx * cell_w
            y = margin_t + y_idx * cell_h
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{battery_color(s.battery_pct)}">'
                f'<title>t={s.time_s}s sat={s.sat_id} battery={s.battery_pct:.2f}%</title></rect>\n'
            )
    for tick in range(0, sats, max(1, sats // 8)):
        y = margin_t + tick * cell_h + cell_h
        lines.append(f'<text x="14" y="{y}" fill="white" font-family="sans-serif" font-size="11">sat {tick}</text>\n')
    lines.append(f'<text x="{margin_l}" y="{height-12}" fill="white" font-family="sans-serif" font-size="11">0s</text>\n')
    lines.append(f'<text x="{max(margin_l, width-90)}" y="{height-12}" fill="white" font-family="sans-serif" font-size="11">{all_steps[-1][0].time_s}s</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines))


def write_task_mode_summary_svg(path: Path, task_records: list[TaskRecord]) -> None:
    width = 900
    height = 360
    margin_l = 60
    margin_r = 30
    margin_t = 44
    margin_b = 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    by_time: dict[int, Counter[str]] = defaultdict(Counter)
    for task in task_records:
        key = "failed" if not task.completed else task.mode
        by_time[task.created_time_s][key] += 1
    times = sorted(by_time)
    max_total = max((sum(by_time[t].values()) for t in times), default=1)
    bar_w = max(4, plot_w / max(1, len(times)) * 0.7)
    lines = [svg_header(width, height)]
    lines.append('<text x="24" y="28" fill="white" font-family="sans-serif" font-size="22">Task mode per generation slot</text>\n')
    lines.append(f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    colors = {"local": "#06d6a0", "offload": "#ffd166", "failed": "#ef476f"}
    for idx, t in enumerate(times):
        x = margin_l + (idx + 0.15) * (plot_w / max(1, len(times)))
        y_base = height - margin_b
        for key in ("local", "offload", "failed"):
            count = by_time[t][key]
            if count == 0:
                continue
            h = count / max_total * plot_h
            y_base -= h
            lines.append(
                f'<rect x="{x:.1f}" y="{y_base:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[key]}">'
                f'<title>t={t}s {key}: {count}</title></rect>\n'
            )
    lines.append('<text x="650" y="70" fill="#06d6a0" font-family="sans-serif" font-size="15">local</text>\n')
    lines.append('<text x="650" y="92" fill="#ffd166" font-family="sans-serif" font-size="15">offload</text>\n')
    lines.append('<text x="650" y="114" fill="#ef476f" font-family="sans-serif" font-size="15">failed</text>\n')
    lines.append(f'<text x="12" y="{margin_t+5}" fill="white" font-family="sans-serif" font-size="13">{max_total}</text>\n')
    if times:
        lines.append(f'<text x="{margin_l}" y="{height-16}" fill="white" font-family="sans-serif" font-size="12">{times[0]}s</text>\n')
        lines.append(f'<text x="{width-margin_r-70}" y="{height-16}" fill="white" font-family="sans-serif" font-size="12">{times[-1]}s</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines))


def write_offload_target_histogram_svg(path: Path, task_records: list[TaskRecord]) -> None:
    width = 900
    height = 360
    margin_l = 60
    margin_r = 30
    margin_t = 44
    margin_b = 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    counts = Counter(task.target_sat for task in task_records if task.mode == "offload" and task.completed)
    targets = sorted(counts)
    max_count = max(counts.values(), default=1)
    bar_w = max(2, plot_w / max(1, len(targets)) * 0.75)
    lines = [svg_header(width, height)]
    lines.append('<text x="24" y="28" fill="white" font-family="sans-serif" font-size="22">Offload target concentration</text>\n')
    lines.append(f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#9fb3c8"/>\n')
    for idx, target in enumerate(targets):
        count = counts[target]
        h = count / max_count * plot_h
        x = margin_l + idx * (plot_w / max(1, len(targets)))
        y = height - margin_b - h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#ffd166">'
            f'<title>target sat {target}: {count} offloaded tasks</title></rect>\n'
        )
    if not targets:
        lines.append('<text x="300" y="180" fill="#9fb3c8" font-family="sans-serif" font-size="18">No offloaded tasks</text>\n')
    lines.append(f'<text x="12" y="{margin_t+5}" fill="white" font-family="sans-serif" font-size="13">{max_count}</text>\n')
    lines.append('<text x="24" y="330" fill="#9fb3c8" font-family="sans-serif" font-size="13">target satellite id</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines))
