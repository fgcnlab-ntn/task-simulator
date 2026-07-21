#!/usr/bin/env python3
"""Plot one satellite's compute loading over time for an existing run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plot_output import format_written, save_png_pdf

SUNLIT_COLOR = "#F6C85F"
ECLIPSE_COLOR = "#6EA8FE"
ROLLING_COLOR = "#D1495B"
BATTERY_COLOR = "#2A9D8F"
DEFAULT_ROLLING_MINUTES = 15.0


@dataclass(frozen=True)
class SatelliteLoadingSample:
    time_s: float
    loading: float
    sunlit: bool
    battery_pct: float


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
            "font.size": 10,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def load_run_metadata(run_dir: Path) -> tuple[dict[str, object], float, float]:
    run_path = run_dir / "run.json"
    states_path = run_dir / "states.jsonl"
    if not run_path.exists():
        raise FileNotFoundError(f"missing run.json: {run_dir}")
    if not states_path.exists():
        raise FileNotFoundError(f"missing states.jsonl: {run_dir}")

    run = json.loads(run_path.read_text())
    config = run["config"]
    step_s = float(config["time"]["step_s"])
    capacity_j = float(config["battery"]["capacity_j"])
    if step_s <= 0.0:
        raise ValueError(f"time.step_s must be positive in {run_path}")
    if capacity_j <= 0.0:
        raise ValueError(f"battery.capacity_j must be positive in {run_path}")
    return run, step_s, capacity_j


def load_satellite_samples(
    run_dir: Path,
    satellite_id: int,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[list[SatelliteLoadingSample], float, dict[str, object]]:
    """Load positive-time samples for one satellite.

    A state logged at time ``t`` represents one ``step_s`` loading interval.
    The t=0 snapshot is excluded because it has no preceding interval.
    """

    if satellite_id < 0:
        raise ValueError("satellite id must be non-negative")
    if start_s is not None and end_s is not None and start_s > end_s:
        raise ValueError("start time must not be later than end time")

    run, step_s, capacity_j = load_run_metadata(run_dir)
    samples: list[SatelliteLoadingSample] = []
    seen_satellite = False

    with (run_dir / "states.jsonl").open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            snapshot = json.loads(line)
            time_s = float(snapshot["time_s"])
            if time_s <= 0.0:
                continue
            if start_s is not None and time_s < start_s:
                continue
            if end_s is not None and time_s > end_s:
                break

            satellite = next(
                (
                    item
                    for item in snapshot.get("satellites", [])
                    if int(item["id"]) == satellite_id
                ),
                None,
            )
            if satellite is None:
                continue
            seen_satellite = True

            task_load = satellite.get("task_load")
            if not isinstance(task_load, dict) or "compute_time_s" not in task_load:
                raise ValueError(
                    f"states.jsonl:{line_number}: satellite {satellite_id} has no "
                    "task_load.compute_time_s"
                )
            compute_time_s = float(task_load["compute_time_s"])
            if compute_time_s < 0.0:
                raise ValueError(
                    f"states.jsonl:{line_number}: satellite {satellite_id} has "
                    "negative compute_time_s"
                )

            samples.append(
                SatelliteLoadingSample(
                    time_s=time_s,
                    loading=compute_time_s / step_s,
                    sunlit=bool(satellite["sunlit"]),
                    battery_pct=100.0 * float(satellite["battery_j"]) / capacity_j,
                )
            )

    if not seen_satellite:
        raise ValueError(
            f"satellite {satellite_id} has no samples in the selected time range"
        )
    return samples, step_s, run


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    result: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def resolve_rolling_steps(
    step_s: float,
    *,
    rolling_minutes: float = DEFAULT_ROLLING_MINUTES,
    rolling_steps: int | None = None,
) -> int:
    if rolling_steps is not None:
        if rolling_steps <= 0:
            raise ValueError("rolling window must be positive")
        return rolling_steps
    if rolling_minutes <= 0.0:
        raise ValueError("rolling minutes must be positive")
    return max(1, round(rolling_minutes * 60.0 / step_s))


def illumination_spans(
    samples: list[SatelliteLoadingSample], step_s: float
) -> list[tuple[float, float, bool]]:
    """Merge adjacent equal illumination states into plot spans."""

    if not samples:
        return []
    spans: list[tuple[float, float, bool]] = []
    start = samples[0].time_s - step_s
    end = samples[0].time_s
    state = samples[0].sunlit
    for sample in samples[1:]:
        interval_start = sample.time_s - step_s
        if sample.sunlit == state and math.isclose(interval_start, end):
            end = sample.time_s
            continue
        spans.append((start, end, state))
        start = interval_start
        end = sample.time_s
        state = sample.sunlit
    spans.append((start, end, state))
    return spans


def satellite_label(run: dict[str, object], satellite_id: int) -> str:
    for satellite in run.get("satellites", []):
        if int(satellite.get("id", -1)) == satellite_id:
            name = satellite.get("name")
            if name:
                return f"{name} (ID {satellite_id})"
    return f"Satellite {satellite_id}"


def run_label(run: dict[str, object]) -> str | None:
    config = run.get("config")
    if not isinstance(config, dict):
        return None
    run_config = config.get("run")
    if not isinstance(run_config, dict):
        return None
    name = run_config.get("name")
    return str(name) if name else None


def plot_satellite_loading(
    samples: list[SatelliteLoadingSample],
    step_s: float,
    run: dict[str, object],
    satellite_id: int,
    output: Path,
    *,
    rolling_minutes: float = DEFAULT_ROLLING_MINUTES,
    rolling_steps: int | None = None,
) -> tuple[Path, Path]:
    if not samples:
        raise ValueError("cannot plot an empty satellite series")

    plt = _pyplot()
    from matplotlib.patches import Patch

    times_h = [sample.time_s / 3600.0 for sample in samples]
    interval_centers_h = [(sample.time_s - step_s / 2.0) / 3600.0 for sample in samples]
    loading_pct = [100.0 * sample.loading for sample in samples]
    window_steps = resolve_rolling_steps(
        step_s,
        rolling_minutes=rolling_minutes,
        rolling_steps=rolling_steps,
    )
    smoothed_pct = rolling_mean(loading_pct, window_steps)
    battery_pct = [sample.battery_pct for sample in samples]

    fig, (loading_ax, battery_ax) = plt.subplots(
        2,
        1,
        figsize=(12.0, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08},
    )

    for start, end, sunlit in illumination_spans(samples, step_s):
        color = SUNLIT_COLOR if sunlit else ECLIPSE_COLOR
        for axis in (loading_ax, battery_ax):
            axis.axvspan(start / 3600.0, end / 3600.0, color=color, alpha=0.16, lw=0)

    rolling_line = loading_ax.plot(
        interval_centers_h,
        smoothed_pct,
        color=ROLLING_COLOR,
        linewidth=1.8,
        label=(
            f"{rolling_minutes:g}-minute average loading"
            if rolling_steps is None
            else f"Average loading ({window_steps} steps)"
        ),
    )[0]
    upper = max(100.0, math.ceil(max(smoothed_pct) / 10.0) * 10.0)
    loading_ax.set_ylim(0.0, upper)
    loading_ax.set_ylabel("CPU loading (%)")
    identity = satellite_label(run, satellite_id)
    method = run_label(run)
    title = (
        f"Per-satellite loading over time — {method} — {identity}"
        if method
        else f"Per-satellite loading over time — {identity}"
    )
    fig.suptitle(
        title,
        y=0.98,
        fontsize=15,
        fontweight="bold",
    )
    loading_ax.grid(axis="y")
    fig.legend(
        handles=[
            rolling_line,
            Patch(facecolor=SUNLIT_COLOR, alpha=0.25, label="Sunlit"),
            Patch(facecolor=ECLIPSE_COLOR, alpha=0.25, label="Eclipse"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
    )
    fig.subplots_adjust(top=0.84)

    battery_ax.plot(times_h, battery_pct, color=BATTERY_COLOR, linewidth=1.4)
    battery_ax.set_ylim(0.0, max(100.0, math.ceil(max(battery_pct) / 10.0) * 10.0))
    battery_ax.set_ylabel("Battery (%)")
    battery_ax.set_xlabel("Simulation time (hours)")
    battery_ax.grid(axis="y")

    first_edge_h = (samples[0].time_s - step_s) / 3600.0
    battery_ax.set_xlim(first_edge_h, samples[-1].time_s / 3600.0)
    paths = save_png_pdf(fig, output)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one satellite's CPU loading, illumination, and battery over time."
    )
    parser.add_argument("run_dir", type=Path, help="completed simulator run directory")
    parser.add_argument("--satellite", type=int, required=True, help="satellite ID")
    rolling_group = parser.add_mutually_exclusive_group()
    rolling_group.add_argument(
        "--rolling-minutes",
        type=float,
        default=DEFAULT_ROLLING_MINUTES,
        help="average window in minutes (default: 15)",
    )
    rolling_group.add_argument(
        "--rolling-steps",
        type=int,
        help="override the average window with a number of simulation steps",
    )
    parser.add_argument("--start-s", type=float, help="first logged time in seconds")
    parser.add_argument("--end-s", type=float, help="last logged time in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        help="output prefix; writes both PNG and PDF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rolling_steps is not None and args.rolling_steps <= 0:
        raise SystemExit("--rolling-steps must be positive")
    if args.rolling_minutes <= 0.0:
        raise SystemExit("--rolling-minutes must be positive")
    samples, step_s, run = load_satellite_samples(
        args.run_dir,
        args.satellite,
        start_s=args.start_s,
        end_s=args.end_s,
    )
    output = args.output or (
        args.run_dir / f"satellite_{args.satellite}_loading_timeseries"
    )
    paths = plot_satellite_loading(
        samples,
        step_s,
        run,
        args.satellite,
        output,
        rolling_minutes=args.rolling_minutes,
        rolling_steps=args.rolling_steps,
    )
    print(f"Wrote {format_written(paths)}")


if __name__ == "__main__":
    main()
