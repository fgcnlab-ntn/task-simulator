from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from .models import BatteryConfig, ComputeConfig, ISLConfig, SchedulerConfig, TaskConfig
from .orbit import iter_circular_states
from .runlog import RunLog
from .scheduler import create_scheduler
from .workload import demand_points_provenance, load_demand_points


OPTIONAL_CONFIG_DEFAULTS = {
    "logging_state_steps": "full",
    "logging_summary_start_s": None,
    "logging_summary_duration_s": None,
}


CONFIG_SECTIONS = {
    "run": {
        "name": "run_name",
        "description": "run_description",
    },
    "orbit": {
        "sun_position_file": "sun_position_file",
        "satellites": "satellites",
        "planes": "planes",
        "altitude_km": "altitude_km",
        "inclination_deg": "inclination_deg",
        "walker_phase": "walker_phase",
    },
    "time": {"start_utc": "start_utc", "duration_s": "duration_s", "step_s": "step_s"},
    "battery": {
        "capacity_j": "battery_capacity_j",
        "initial_pct": "battery_initial_pct",
        "min_safe_pct": "battery_min_safe_pct",
        "harvest_w": "harvest_w",
        "idle_w": "idle_w",
    },
    "task": {
        "enabled": "task_enable",
        "interval_s": "task_interval_s",
        "generation_mode": "task_generation_mode",
        "random_seed": "task_random_seed",
        "tasks_per_sat": "tasks_per_sat",
        "tasks_per_step": "tasks_per_step",
        "input_bits": "task_input_bits",
        "output_bits": "task_output_bits",
        "demand_points_file": "task_demand_points_file",
        "min_elevation_deg": "task_min_elevation_deg",
        "deadline_s": "task_deadline_s",
        "deadline_min_s": "task_deadline_min_s",
    },
    "compute": {
        "cycles_per_input_bit": "compute_cycles_per_input_bit",
        "cpu_frequency_hz": "satellite_cpu_frequency_hz",
        "cpu_power_w": "satellite_cpu_power_w",
    },
    "isl": {
        "rate_bps": "isl_rate_bps",
        "tx_power_w": "isl_tx_power_w",
        "max_range_km": "isl_max_range_km",
    },
    "scheduler": {
        "name": "scheduler",
    },
    "output": {"path": "out"},
    "logging": {
        "task_events": "logging_task_events",
        "state_steps": "logging_state_steps",
        "summary_start_s": "logging_summary_start_s",
        "summary_duration_s": "logging_summary_duration_s",
    },
}

OPTIONAL_CONFIG_KEYS = {
    "logging": {"state_steps", "summary_start_s", "summary_duration_s"},
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the minimal satellite orbit simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="complete standalone JSON config file",
    )
    return resolve_config(p.parse_args())


def flatten_config(config: dict) -> dict:
    flat = {}
    for key, value in config.items():
        if key in CONFIG_SECTIONS:
            if not isinstance(value, dict):
                raise ValueError(f"config section {key!r} must be an object")
            mapping = CONFIG_SECTIONS[key]
            for section_key, section_value in value.items():
                target = mapping.get(section_key)
                if target is None:
                    raise ValueError(f"unknown config key: {key}.{section_key}")
                flat[target] = section_value
        else:
            raise ValueError(f"unknown config key: {key}")
    return flat


def load_standalone_json_config(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON config must be an object")

    expected_sections = set(CONFIG_SECTIONS)
    actual_sections = set(data)
    if actual_sections != expected_sections:
        missing = sorted(expected_sections - actual_sections)
        extra = sorted(actual_sections - expected_sections)
        details = []
        if missing:
            details.append(f"missing sections: {', '.join(missing)}")
        if extra:
            details.append(f"unknown sections: {', '.join(extra)}")
        raise ValueError(
            "standalone config must define every section ("
            + "; ".join(details)
            + ")"
        )

    for section, mapping in CONFIG_SECTIONS.items():
        value = data[section]
        if not isinstance(value, dict):
            raise ValueError(f"config section {section!r} must be an object")
        expected_keys = set(mapping)
        optional_keys = OPTIONAL_CONFIG_KEYS.get(section, set())
        actual_keys = set(value)
        required_keys = expected_keys - optional_keys
        if not required_keys <= actual_keys <= expected_keys:
            missing = sorted(required_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if extra:
                details.append(f"unknown keys: {', '.join(extra)}")
            raise ValueError(
                f"standalone config section {section!r} must define every key ("
                + "; ".join(details)
                + ")"
            )

    return flatten_config(data)


def resolve_config(cli_args: argparse.Namespace) -> argparse.Namespace:
    config_path = cli_args.config
    values = OPTIONAL_CONFIG_DEFAULTS.copy()
    values.update(load_standalone_json_config(config_path))
    values["task_demand_points_file"] = (
        None
        if values["task_demand_points_file"] is None
        else Path(values["task_demand_points_file"])
    )
    values["out"] = Path(values["out"])
    return argparse.Namespace(**values)


def parse_utc_datetime(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_s < 0:
        raise ValueError("time.duration_s must be non-negative")
    if args.step_s <= 0:
        raise ValueError("time.step_s must be positive")
    if not 0 <= args.battery_initial_pct <= 100:
        raise ValueError("battery.initial_pct must be within [0, 100]")
    if not 0 <= args.battery_min_safe_pct <= 100:
        raise ValueError("battery.min_safe_pct must be within [0, 100]")
    if args.task_interval_s <= 0:
        raise ValueError("task.interval_s must be positive")
    if args.tasks_per_sat < 0:
        raise ValueError("task.tasks_per_sat must be non-negative")
    if args.tasks_per_step < 0:
        raise ValueError("task.tasks_per_step must be non-negative")
    if args.task_deadline_s <= 0:
        raise ValueError("task.deadline_s must be positive")
    if args.task_deadline_min_s <= 0:
        raise ValueError("task.deadline_min_s must be positive")
    if args.task_deadline_min_s >= args.task_deadline_s:
        raise ValueError(
            "task.deadline_min_s must be less than task.deadline_s"
        )
    if not 0.0 <= args.task_min_elevation_deg <= 90.0:
        raise ValueError("task.min_elevation_deg must be within [0, 90]")
    if args.compute_cycles_per_input_bit <= 0:
        raise ValueError("compute.cycles_per_input_bit must be positive")
    if args.satellite_cpu_frequency_hz <= 0:
        raise ValueError("compute.cpu_frequency_hz must be positive")
    if args.satellite_cpu_power_w < 0:
        raise ValueError("compute.cpu_power_w must be non-negative")
    if args.isl_rate_bps <= 0:
        raise ValueError("isl.rate_bps must be positive")
    if args.isl_tx_power_w < 0:
        raise ValueError("isl.tx_power_w must be non-negative")
    if args.isl_max_range_km is None or args.isl_max_range_km <= 0.0:
        raise ValueError("isl.max_range_km must be positive")
    logging_task_events = getattr(args, "logging_task_events", "full")
    if logging_task_events not in {"full", "lifecycle", "summary", "off"}:
        raise ValueError(
            "logging.task_events must be full, lifecycle, summary, or off"
        )
    logging_state_steps = getattr(args, "logging_state_steps", "full")
    if logging_state_steps not in {"full", "off"}:
        raise ValueError("logging.state_steps must be full or off")
    summary_start_s = getattr(args, "logging_summary_start_s", None)
    summary_duration_s = getattr(args, "logging_summary_duration_s", None)
    if (summary_start_s is None) != (summary_duration_s is None):
        raise ValueError(
            "logging.summary_start_s and logging.summary_duration_s "
            "must be specified together"
        )
    if summary_start_s is not None:
        if (
            not isinstance(summary_start_s, int)
            or isinstance(summary_start_s, bool)
        ):
            raise ValueError("logging.summary_start_s must be an integer")
        if (
            not isinstance(summary_duration_s, int)
            or isinstance(summary_duration_s, bool)
        ):
            raise ValueError("logging.summary_duration_s must be an integer")
        if summary_start_s < 0:
            raise ValueError("logging.summary_start_s must be non-negative")
        if summary_duration_s <= 0:
            raise ValueError("logging.summary_duration_s must be positive")
        if summary_start_s % args.step_s or summary_duration_s % args.step_s:
            raise ValueError("logging summary window must align with time.step_s")
        if summary_start_s + summary_duration_s > args.duration_s:
            raise ValueError("logging summary window exceeds time.duration_s")


def walker_raan_spread_deg(args: argparse.Namespace) -> float:
    """Return the RAAN spread for the built-in Walker constellation presets."""

    return 180.0 if str(args.run_name).lower().startswith("iridium") else 360.0


def build_configs(
    args: argparse.Namespace,
) -> tuple[BatteryConfig, ComputeConfig, TaskConfig, ISLConfig, SchedulerConfig]:
    battery = BatteryConfig(
        capacity_j=args.battery_capacity_j,
        initial_j=args.battery_capacity_j * args.battery_initial_pct / 100.0,
        min_safe_j=args.battery_capacity_j * args.battery_min_safe_pct / 100.0,
        harvest_w=args.harvest_w,
        idle_w=args.idle_w,
    )
    task_config = TaskConfig(
        enabled=args.task_enable,
        interval_s=args.task_interval_s,
        generation_mode=args.task_generation_mode,
        random_seed=args.task_random_seed,
        tasks_per_sat=args.tasks_per_sat,
        tasks_per_step=args.tasks_per_step,
        input_bits=args.task_input_bits,
        output_bits=args.task_output_bits,
        deadline_s=args.task_deadline_s,
        deadline_min_s=args.task_deadline_min_s,
        demand_distribution=load_demand_points(args.task_demand_points_file),
        min_elevation_deg=args.task_min_elevation_deg,
    )
    compute_config = ComputeConfig(
        cycles_per_input_bit=args.compute_cycles_per_input_bit,
        cpu_frequency_hz=args.satellite_cpu_frequency_hz,
        cpu_power_w=args.satellite_cpu_power_w,
    )
    isl_config = ISLConfig(
        rate_bps=args.isl_rate_bps,
        tx_power_w=args.isl_tx_power_w,
        max_range_km=args.isl_max_range_km,
    )
    scheduler_config = SchedulerConfig(name=args.scheduler)
    return battery, compute_config, task_config, isl_config, scheduler_config


def effective_run_config(args: argparse.Namespace) -> dict:
    orbit_config = {
        "sun_position_file": args.sun_position_file,
        "satellites": args.satellites,
        "planes": args.planes,
        "altitude_km": args.altitude_km,
        "inclination_deg": args.inclination_deg,
        "walker_phase": args.walker_phase,
    }
    config = {
        "run": {
            "name": args.run_name,
            "description": args.run_description,
        },
        "orbit": orbit_config,
        "time": {
            "start_utc": args.start_utc,
            "duration_s": args.duration_s,
            "step_s": args.step_s,
        },
        "battery": {
            "capacity_j": args.battery_capacity_j,
            "initial_pct": args.battery_initial_pct,
            "min_safe_pct": args.battery_min_safe_pct,
            "harvest_w": args.harvest_w,
            "idle_w": args.idle_w,
        },
        "task": {
            "enabled": args.task_enable,
            "interval_s": args.task_interval_s,
            "generation_mode": args.task_generation_mode,
            "random_seed": args.task_random_seed,
            "tasks_per_sat": args.tasks_per_sat,
            "tasks_per_step": args.tasks_per_step,
            "input_bits": args.task_input_bits,
            "output_bits": args.task_output_bits,
            "demand_points_file": None
            if args.task_demand_points_file is None
            else str(args.task_demand_points_file),
            "demand_points_provenance": demand_points_provenance(
                args.task_demand_points_file
            ),
            "min_elevation_deg": args.task_min_elevation_deg,
            "deadline_s": args.task_deadline_s,
            "deadline_min_s": args.task_deadline_min_s,
        },
        "compute": {
            "cycles_per_input_bit": args.compute_cycles_per_input_bit,
            "cpu_frequency_hz": args.satellite_cpu_frequency_hz,
            "cpu_power_w": args.satellite_cpu_power_w,
        },
        "isl": {
            "rate_bps": args.isl_rate_bps,
            "tx_power_w": args.isl_tx_power_w,
            "max_range_km": args.isl_max_range_km,
        },
        "scheduler": {
            "name": args.scheduler,
        },
        "output": {
            "path": str(args.out),
        },
        "logging": {
            "task_events": getattr(args, "logging_task_events", "full"),
        },
    }
    if getattr(args, "logging_state_steps", "full") != "full":
        config["logging"]["state_steps"] = args.logging_state_steps
    summary_start_s = getattr(args, "logging_summary_start_s", None)
    summary_duration_s = getattr(args, "logging_summary_duration_s", None)
    if summary_start_s is not None and summary_duration_s is not None:
        config["logging"]["summary_start_s"] = summary_start_s
        config["logging"]["summary_duration_s"] = summary_duration_s
    return config


def run(args: argparse.Namespace) -> int:
    start = parse_utc_datetime(args.start_utc)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    run_config = effective_run_config(args)
    battery, compute_config, task_config, isl_config, scheduler_config = build_configs(args)
    scheduler = create_scheduler(args.scheduler)
    run_log = RunLog(args.out, start, run_config)

    try:
        common = {
            "start": start,
            "duration_s": args.duration_s,
            "step_s": args.step_s,
            "battery": battery,
            "compute_config": compute_config,
            "task_config": task_config,
            "isl_config": isl_config,
            "scheduler": scheduler,
            "scheduler_config": scheduler_config,
            "task_event_sink": run_log.write_task_event,
            "step_sink": run_log.write_step,
        }

        step_iterator = iter_circular_states(
            satellites=args.satellites,
            planes=args.planes,
            altitude_km=args.altitude_km,
            inclination_deg=args.inclination_deg,
            sun_position_file=args.sun_position_file,
            walker_phase=args.walker_phase,
            raan_spread_deg=walker_raan_spread_deg(args),
            **common,
        )

        first = None
        last = None
        steps = 0

        total_steps = args.duration_s // args.step_s + 1
        progress_started = time.monotonic()
        last_progress_print = 0.0
        bar_width = 40

        def fmt_seconds(seconds: float) -> str:
            seconds = int(max(0, seconds))
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            if h:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        for states, _ in step_iterator:
            if first is None:
                first = states
            last = states
            steps += 1

            now = time.monotonic()
            if now - last_progress_print >= 1.0 or steps == total_steps:
                elapsed = now - progress_started
                rate = steps / elapsed if elapsed > 0 else 0.0
                remaining_steps = max(0, total_steps - steps)
                eta = remaining_steps / rate if rate > 0 else 0.0
                pct = steps / total_steps

                filled = int(bar_width * pct)
                bar = "#" * filled + "-" * (bar_width - filled)

                sys.stderr.write(
                    "\r"
                    f"Simulating: [{bar}] "
                    f"{steps}/{total_steps} "
                    f"({pct * 100:5.1f}%) "
                    f"elapsed {fmt_seconds(elapsed)} "
                    f"eta {fmt_seconds(eta)}"
                )
                sys.stderr.flush()
                last_progress_print = now

        sys.stderr.write("\n")
        run_log.complete()

    except BaseException as exc:
        run_log.fail(exc)
        raise

    assert first is not None and last is not None

    summary = json.loads((args.out / "summary.json").read_text())
    task_summary = summary["tasks"]
    battery_violations = summary.get("battery_violations", {})

    print("Minimal orbit simulation complete")
    print(f"  scheduler: {scheduler.name}")
    print(f"  satellites: {len(first)}")
    print(f"  planes: {args.planes}")
    print(f"  steps: {steps}, duration: {args.duration_s}s, step: {args.step_s}s")
    print(
        f"  t=0 sunlit/eclipse: "
        f"{sum(s.sunlit for s in first)}/{len(first) - sum(s.sunlit for s in first)}"
    )
    print(
        f"  final sunlit/eclipse: "
        f"{sum(s.sunlit for s in last)}/{len(last) - sum(s.sunlit for s in last)}"
    )
    print(
        f"  final battery min/avg: "
        f"{min(s.battery_pct for s in last):.2f}%/"
        f"{sum(s.battery_pct for s in last) / len(last):.2f}%"
    )
    print(
        "  tasks completed/deferred/failed/pending: "
        f"{task_summary['completed']}/"
        f"{task_summary.get('deferred', 0)}/"
        f"{task_summary['failed']}/"
        f"{task_summary.get('pending', 0)}"
    )
    print(
        "  battery breaches total/eclipse: "
        f"{battery_violations.get('unique_breached_satellites', 0)}/"
        f"{battery_violations.get('unique_eclipse_breached_satellites', 0)}"
    )
    print(f"  output: {args.out.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    return run(args)
