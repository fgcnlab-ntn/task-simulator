#!/usr/bin/env python3
"""Run and summarize eclipse-duration statistics for the Walker shell."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satmulator.cli import (
    build_configs,
    effective_run_config,
    load_standalone_json_config,
    parse_utc_datetime,
    validate_args,
    walker_raan_spread_deg,
)
from satmulator.orbit import iter_circular_states
from satmulator.runlog import append_json_line, iter_state_steps, write_json
from satmulator.scheduler import create_scheduler
from tools.plot_output import save_png_pdf

DEFAULT_CONFIG = Path("configs/base/template.json")
DEFAULT_OUTPUT = Path("eclipse_time")
DEFAULT_DURATION_S = 43_200


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a no-task eclipse-time experiment and write PNG/PDF plots."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=int, default=DEFAULT_DURATION_S)
    parser.add_argument("--step-s", type=int, default=None)
    args = parser.parse_args()

    values = load_standalone_json_config(args.config)
    values["duration_s"] = args.duration_s
    if args.step_s is not None:
        values["step_s"] = args.step_s
    values["task_enable"] = False
    values["out"] = args.out
    values["config"] = args.config
    values["plot_run"] = None
    values["run_description"] = (
        "No-task eclipse-duration experiment for the configured constellation"
    )
    values["tle_file"] = None if values["tle_file"] is None else Path(values["tle_file"])
    values["task_demand_points_file"] = (
        None
        if values["task_demand_points_file"] is None
        else Path(values["task_demand_points_file"])
    )
    values["out"] = Path(values["out"])

    run_args = SimpleNamespace(**values)
    durations = run_eclipse_experiment(run_args)
    summary = summarize_durations(durations)
    write_summary(args.out / "eclipse_time_summary.json", summary)
    write_bar_plot(args.out / "eclipse_time_bar", summary)
    bins = histogram_bins(durations)
    write_pdf_plot(args.out / "eclipse_time_pdf", bins, summary)
    write_cdf_plot(args.out / "eclipse_time_cdf", durations, summary)
    satellite_sunlit_ratios = satellite_sunlit_ratios_from_csv(
        args.out / "satellite_sunlit_ratios.csv"
    )
    write_satellite_sunlit_ratio_cdf_plot(
        args.out / "satellite_sunlit_ratio_cdf",
        satellite_sunlit_ratios,
        summary=summarize_ratios(satellite_sunlit_ratios),
        label=constellation_label(run_args),
    )
    print_eclipse_summary(summary)
    return 0


def run_eclipse_experiment(args: SimpleNamespace) -> list[float]:
    if args.orbit_model != "circular":
        raise ValueError("eclipse_time experiment currently requires orbit.orbit_model circular")

    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "run_config.json", effective_run_config(args))

    battery, compute_config, task_config, isl_config, scheduler_config = build_configs(args)
    scheduler = create_scheduler(args.scheduler)
    start = parse_utc_datetime(args.start_utc)
    tracker = EclipseIntervalTracker()
    sunlit_counts: dict[int, int] = {}
    sample_counts: dict[int, int] = {}
    satellite_layout: dict[int, tuple[int, int]] = {}

    with (args.out / "eclipse_samples.jsonl").open("w") as samples, (
        args.out / "eclipse_intervals.jsonl"
    ).open("w") as intervals:
        for states, _ in iter_circular_states(
            start=start,
            sun_position_file=args.sun_position_file,
            satellites=args.satellites,
            planes=args.planes,
            altitude_km=args.altitude_km,
            inclination_deg=args.inclination_deg,
            duration_s=args.duration_s,
            step_s=args.step_s,
            battery=battery,
            compute_config=compute_config,
            task_config=task_config,
            isl_config=isl_config,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
            walker_phase=args.walker_phase,
            raan_spread_deg=walker_raan_spread_deg(args),
        ):
            eclipsed = sum(1 for state in states if not state.sunlit)
            for state in states:
                sat_id = int(state.sat_id)
                sample_counts[sat_id] = sample_counts.get(sat_id, 0) + 1
                sunlit_counts[sat_id] = sunlit_counts.get(sat_id, 0) + int(state.sunlit)
                satellite_layout[sat_id] = (int(state.plane), int(state.slot))
            append_json_line(
                samples,
                {
                    "schema_version": 1,
                    "time_s": states[0].time_s,
                    "satellites": len(states),
                    "sunlit": len(states) - eclipsed,
                    "eclipsed": eclipsed,
                },
            )
            for interval in tracker.update(states):
                append_json_line(intervals, {"schema_version": 1, **interval})

    durations = tracker.durations_s
    if not durations:
        raise ValueError("no complete eclipse intervals found")
    write_satellite_sunlit_ratios(
        args.out / "satellite_sunlit_ratios.csv",
        args.out / "satellite_sunlit_ratios.json",
        sunlit_counts=sunlit_counts,
        sample_counts=sample_counts,
        satellite_layout=satellite_layout,
    )
    return durations


class EclipseIntervalTracker:
    def __init__(self) -> None:
        self._seen_first_step = False
        self._in_eclipse: dict[int, bool] = {}
        self._starts: dict[int, int | None] = {}
        self.durations_s: list[float] = []

    def update(self, states) -> list[dict[str, object]]:
        completed = []
        for state in states:
            sat_id = state.sat_id
            eclipsed = not state.sunlit
            if not self._seen_first_step:
                self._in_eclipse[sat_id] = eclipsed
                self._starts[sat_id] = None
                continue

            was_eclipsed = self._in_eclipse.get(sat_id, False)
            if eclipsed and not was_eclipsed:
                self._starts[sat_id] = state.time_s
            elif not eclipsed and was_eclipsed:
                start = self._starts.get(sat_id)
                if start is not None:
                    duration = float(state.time_s - start)
                    self.durations_s.append(duration)
                    completed.append(
                        {
                            "satellite_id": sat_id,
                            "plane": state.plane,
                            "slot": state.slot,
                            "start_s": start,
                            "end_s": state.time_s,
                            "duration_s": duration,
                        }
                    )
                self._starts[sat_id] = None
            self._in_eclipse[sat_id] = eclipsed
        self._seen_first_step = True
        return completed


def eclipse_durations_from_run(output_dir: Path) -> list[float]:
    starts: dict[int, int | None] = {}
    in_eclipse: dict[int, bool] = {}
    durations: list[float] = []

    first = True
    for record in iter_state_steps(output_dir):
        time_s = int(record["time_s"])
        satellites = record["satellites"]
        if not isinstance(satellites, list):
            raise ValueError("state record satellites must be a list")

        for satellite in satellites:
            if not isinstance(satellite, dict):
                raise ValueError("satellite state must be an object")
            sat_id = int(satellite["id"])
            sunlit = bool(satellite["sunlit"])
            eclipsed = not sunlit

            if first:
                in_eclipse[sat_id] = eclipsed
                starts[sat_id] = None if eclipsed else None
                continue

            was_eclipsed = in_eclipse.get(sat_id, False)
            if eclipsed and not was_eclipsed:
                starts[sat_id] = time_s
            elif not eclipsed and was_eclipsed:
                start = starts.get(sat_id)
                if start is not None:
                    durations.append(float(time_s - start))
                starts[sat_id] = None
            in_eclipse[sat_id] = eclipsed
        first = False

    if not durations:
        raise ValueError("no complete eclipse intervals found")
    return durations


def summarize_durations(durations_s: list[float]) -> dict[str, object]:
    ordered = sorted(durations_s)
    count = len(ordered)
    return {
        "intervals": count,
        "min_s": ordered[0],
        "mean_s": statistics.fmean(ordered),
        "p25_s": percentile(ordered, 0.25),
        "median_s": percentile(ordered, 0.50),
        "p75_s": percentile(ordered, 0.75),
        "p10_s": percentile(ordered, 0.10),
        "p90_s": percentile(ordered, 0.90),
        "max_s": ordered[-1],
        "min_min": ordered[0] / 60.0,
        "mean_min": statistics.fmean(ordered) / 60.0,
        "p25_min": percentile(ordered, 0.25) / 60.0,
        "median_min": percentile(ordered, 0.50) / 60.0,
        "p75_min": percentile(ordered, 0.75) / 60.0,
        "p10_min": percentile(ordered, 0.10) / 60.0,
        "p90_min": percentile(ordered, 0.90) / 60.0,
        "max_min": ordered[-1] / 60.0,
    }


def write_satellite_sunlit_ratios(
    csv_path: Path,
    json_path: Path,
    *,
    sunlit_counts: dict[int, int],
    sample_counts: dict[int, int],
    satellite_layout: dict[int, tuple[int, int]],
) -> None:
    rows = []
    for sat_id in sorted(sample_counts):
        samples = sample_counts[sat_id]
        if samples <= 0:
            raise ValueError("satellite sample count must be positive")
        plane, slot = satellite_layout[sat_id]
        sunlit_samples = sunlit_counts.get(sat_id, 0)
        rows.append(
            {
                "satellite_id": sat_id,
                "plane": plane,
                "slot": slot,
                "samples": samples,
                "sunlit_samples": sunlit_samples,
                "sunlit_ratio_pct": 100.0 * sunlit_samples / samples,
            }
        )
    if not rows:
        raise ValueError("no satellite sunlit-ratio rows found")

    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "satellite_id",
                "plane",
                "slot",
                "samples",
                "sunlit_samples",
                "sunlit_ratio_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        json_path,
        {
            "schema_version": 1,
            "satellites": len(rows),
            "ratios": rows,
        },
    )


def satellite_sunlit_ratios_from_csv(path: Path) -> list[float]:
    with path.open(newline="") as stream:
        ratios = [float(row["sunlit_ratio_pct"]) for row in csv.DictReader(stream)]
    if not ratios:
        raise ValueError("no satellite sunlit-ratio rows found")
    return ratios


def summarize_ratios(ratios: list[float]) -> dict[str, float]:
    ordered = sorted(ratios)
    return {
        "count": float(len(ordered)),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def constellation_label(args: SimpleNamespace) -> str:
    name = str(args.run_name).strip()
    if name and name != "eclipse_time":
        return name
    return f"{int(args.satellites)} satellites, {int(args.planes)} planes"


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        raise ValueError("cannot compute percentile of empty data")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def histogram_bins(durations_s: list[float], *, bin_width_s: int = 60) -> list[dict[str, float]]:
    """Return PDF bins aligned to the simulation time resolution.

    The experiment uses a 30 s simulation step by default. A 60 s bin is a
    readable two-sample bucket: it preserves the discrete nature of the data
    without producing a needlessly jagged plot. Bin edges are stored in seconds.
    """
    width = float(bin_width_s)
    lo = math.floor(min(durations_s) / width) * width
    hi = math.ceil(max(durations_s) / width) * width
    if lo == hi:
        return [{"left": lo, "right": lo + width, "density": 1.0 / width, "share": 1.0}]

    bin_count = int(round((hi - lo) / width))
    counts = [0] * bin_count
    for value in durations_s:
        index = min(bin_count - 1, int((value - lo) / width))
        counts[index] += 1

    total = len(durations_s)
    return [
        {
            "left": lo + index * width,
            "right": lo + (index + 1) * width,
            "density": count / total / width,
            "share": count / total,
        }
        for index, count in enumerate(counts)
    ]


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


def write_bar_plot(path: Path, summary: dict[str, object]) -> None:
    plt = _pyplot()
    stats = [
        {
            "label": "",
            "whislo": float(summary["min_s"]),
            "q1": float(summary["p25_s"]),
            "med": float(summary["mean_s"]),
            "q3": float(summary["p75_s"]),
            "whishi": float(summary["max_s"]),
            "fliers": [],
        }
    ]

    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    ax.bxp(
        stats,
        showfliers=False,
        widths=0.36,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "edgecolor": "#222222", "linewidth": 1.7},
        medianprops={"color": "#E45756", "linewidth": 2.4},
        whiskerprops={"color": "#222222", "linewidth": 1.7},
        capprops={"color": "#222222", "linewidth": 1.7},
    )
    ax.set_title("Eclipse duration summary")
    ax.set_ylabel("Eclipse duration t (s)")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.7)
    ax.text(
        0.03,
        0.97,
        f"low={float(summary['min_s']):.0f}s\n"
        f"mean={float(summary['mean_s']):.0f}s\n"
        f"high={float(summary['max_s']):.0f}s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    save_png_pdf(fig, path)
    plt.close(fig)


def write_pdf_plot(path: Path, bins: list[dict[str, float]], summary: dict[str, object]) -> None:
    plt = _pyplot()
    lefts = [bin_["left"] for bin_ in bins]
    widths = [bin_["right"] - bin_["left"] for bin_ in bins]
    centers = [left + width / 2 for left, width in zip(lefts, widths)]
    range_labels = [
        f"{bin_['left']:.0f}–{bin_['right']:.0f}"
        for bin_ in bins
    ]
    shares_pct = [100.0 * bin_["share"] for bin_ in bins]
    bin_width_s = widths[0] if widths else 0.0

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(
        lefts,
        shares_pct,
        width=widths,
        align="edge",
        color="#4C78A8",
        edgecolor="white",
        linewidth=1.0,
    )
    ax.set_xlabel("Eclipse duration range (s)")
    ax.set_ylabel("Share of eclipse intervals (%)")
    ax.set_title("Distribution of eclipse duration")
    ax.set_xticks(centers)
    ax.set_xticklabels(range_labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.7)
    ax.set_xlim(lefts[0], bins[-1]["right"])
    ax.set_ylim(0.0, max(shares_pct) * 1.18 if shares_pct else 1.0)
    ax.text(
        0.02,
        0.96,
        f"bin width={bin_width_s:.0f}s; intervals={int(summary['intervals'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    save_png_pdf(fig, path)
    plt.close(fig)


def write_cdf_plot(path: Path, durations_s: list[float], summary: dict[str, object]) -> None:
    plt = _pyplot()
    values = sorted(durations_s)
    cdf = [(index + 1) / len(values) for index in range(len(values))]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.step(values, cdf, where="post", color="#4C78A8", linewidth=2.2)
    ax.set_xlabel("Eclipse duration t")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF of eclipse duration")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(values[0], values[-1])
    ax.grid(True, alpha=0.7)
    save_png_pdf(fig, path)
    plt.close(fig)


def write_satellite_sunlit_ratio_cdf_plot(
    path: Path,
    sunlit_ratios_pct: list[float],
    *,
    summary: dict[str, float],
    label: str,
) -> None:
    plt = _pyplot()
    values = sorted(sunlit_ratios_pct)
    cdf = [(index + 1) / len(values) for index in range(len(values))]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.step(values, cdf, where="post", color="#1f77b4", linewidth=2.8, label=label)
    ax.set_xlabel("Sunlit Ratio (%)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of per-satellite sunlit ratio")
    ax.set_ylim(0.0, 1.02)
    # Keep the same visual frame as the constellation-comparison plot this
    # figure mirrors.  A single Walker shell has a narrow range, and autoscale
    # would make tiny fluctuations look larger than they are.
    if 60.0 <= values[0] and values[-1] <= 100.0:
        ax.set_xlim(60.0, 100.0)
    else:
        ax.set_xlim(
            max(0.0, math.floor(values[0] - 1.0)),
            min(100.0, math.ceil(values[-1] + 1.0)),
        )
    ax.grid(True, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.94)
    ax.text(
        0.02,
        0.96,
        f"satellites={int(summary['count'])}\n"
        f"min={summary['min']:.1f}%\n"
        f"mean={summary['mean']:.1f}%\n"
        f"max={summary['max']:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    save_png_pdf(fig, path)
    plt.close(fig)


def print_eclipse_summary(summary: dict[str, object]) -> None:
    print("Eclipse-time experiment complete")
    print(f"  intervals: {summary['intervals']}")
    print(f"  low/mean/high: {summary['min_min']:.2f}/{summary['mean_min']:.2f}/{summary['max_min']:.2f} min")


if __name__ == "__main__":
    raise SystemExit(main())
