from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .models import BatteryConfig, ISLConfig, TaskConfig
from .orbit import circular_snapshot_context, iter_circular_states, iter_tle_states, tle_snapshot_context
from .output import (
    write_battery_svg,
    write_battery_timeline_svg,
    write_offload_target_histogram_svg,
    write_snapshot_svg,
    write_states_csv,
    write_summary_csv,
    write_summary_svg,
    write_sunlight_timeline_svg,
    write_task_mode_summary_svg,
    write_task_svg,
    write_tasks_csv,
)
from .scheduler import create_scheduler
from .workload import load_demand_points


DEFAULT_CONFIG = {
    "orbit_model": "circular",
    "tle_file": None,
    "sun_position_file": "de421.bsp",
    "start_utc": "2026-05-22T12:00:00Z",
    "satellites": 66,
    "planes": 6,
    "altitude_km": 550.0,
    "inclination_deg": 53.0,
    "duration_s": 1800,
    "step_s": 30,
    "walker_phase": 1,
    "battery_capacity_j": 100000.0,
    "battery_initial_pct": 80.0,
    "battery_min_safe_pct": 20.0,
    "harvest_w": 80.0,
    "idle_w": 40.0,
    "task_enable": True,
    "scheduler": "local",
    "task_interval_s": 300,
    "task_generation_mode": "satellite-deterministic",
    "task_random_seed": 42,
    "tasks_per_sat": 1,
    "tasks_per_step_choices": [0, 5, 10, 20],
    "tasks_per_step_weights": [0.2, 0.4, 0.2, 0.2],
    "task_cpu_cycles": 1.0e9,
    "task_cpu_cycles_choices": [1.0e8, 1.0e9, 5.0e9],
    "task_cpu_cycles_weights": [0.6, 0.3, 0.1],
    "task_input_bits": 1.0e7,
    "task_input_bits_choices": [1.0e6, 1.0e7, 1.0e8],
    "task_input_bits_weights": [0.6, 0.3, 0.1],
    "task_output_bits": 1.0e6,
    "task_output_bits_choices": [1.0e5, 1.0e6, 1.0e7],
    "task_output_bits_weights": [0.6, 0.3, 0.1],
    "task_demand_points_file": None,
    "task_min_elevation_deg": 30.0,
    "task_deadline_s": 120.0,
    "cpu_rate_cycles_s": 1.0e8,
    "joule_per_cycle": 1.0e-8,
    "isl_forward_rate_bps": 1.0e7,
    "isl_return_rate_bps": 1.0e7,
    "isl_tx_energy_per_bit_j": 1.0e-7,
    "isl_rx_energy_per_bit_j": 5.0e-8,
    "out": "output/minimal_orbit",
}


CONFIG_SECTIONS = {
    "orbit": {
        "orbit_model": "orbit_model",
        "tle_file": "tle_file",
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
        "tasks_per_step_choices": "tasks_per_step_choices",
        "tasks_per_step_weights": "tasks_per_step_weights",
        "cpu_cycles": "task_cpu_cycles",
        "cpu_cycles_choices": "task_cpu_cycles_choices",
        "cpu_cycles_weights": "task_cpu_cycles_weights",
        "input_bits": "task_input_bits",
        "input_bits_choices": "task_input_bits_choices",
        "input_bits_weights": "task_input_bits_weights",
        "output_bits": "task_output_bits",
        "output_bits_choices": "task_output_bits_choices",
        "output_bits_weights": "task_output_bits_weights",
        "demand_points_file": "task_demand_points_file",
        "min_elevation_deg": "task_min_elevation_deg",
        "deadline_s": "task_deadline_s",
        "cpu_rate_cycles_s": "cpu_rate_cycles_s",
        "joule_per_cycle": "joule_per_cycle",
    },
    "isl": {
        "isl_forward_rate_bps": "isl_forward_rate_bps",
        "isl_return_rate_bps": "isl_return_rate_bps",
        "isl_tx_energy_per_bit_j": "isl_tx_energy_per_bit_j",
        "isl_rx_energy_per_bit_j": "isl_rx_energy_per_bit_j",
    },
    "scheduler": {"name": "scheduler"},
    "output": {"path": "out"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the minimal satellite orbit simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, help="JSON config file; CLI flags override it")
    p.add_argument("--orbit-model", dest="orbit_model", choices=("circular", "tle"))
    p.add_argument("--tle-file", type=Path, help="local TLE file for --orbit-model tle")
    p.add_argument("--sun-position-file", dest="sun_position_file", help="local file used by Skyfield for Sun position")
    p.add_argument("--start-utc")
    p.add_argument("--satellites", type=int, help="total satellite count")
    p.add_argument("--planes", type=int, help="orbital plane count; must divide satellites")
    p.add_argument("--altitude-km", type=float)
    p.add_argument("--inclination-deg", type=float)
    p.add_argument("--duration-s", type=int)
    p.add_argument("--step-s", type=int)
    p.add_argument("--walker-phase", type=int)
    p.add_argument("--battery-capacity-j", type=float)
    p.add_argument("--battery-initial-pct", type=float)
    p.add_argument("--battery-min-safe-pct", type=float)
    p.add_argument("--harvest-w", type=float, help="charging power while sunlit")
    p.add_argument("--idle-w", type=float, help="baseline power draw")
    p.add_argument("--task-enable", dest="task_enable", action="store_true", default=None, help="enable deterministic local tasks")
    p.add_argument("--no-task", dest="task_enable", action="store_false", default=None, help="disable task generation and execution")
    p.add_argument("--scheduler", choices=("local", "nearest-sunlit"))
    p.add_argument("--task-interval-s", type=int)
    p.add_argument("--task-generation-mode", choices=("satellite-deterministic", "demand-points"))
    p.add_argument("--task-random-seed", type=int)
    p.add_argument("--tasks-per-sat", type=int)
    p.add_argument("--task-demand-points-file", type=Path)
    p.add_argument("--task-min-elevation-deg", type=float)
    p.add_argument("--task-cpu-cycles", type=float)
    p.add_argument("--task-input-bits", dest="task_input_bits", type=float)
    p.add_argument("--task-output-bits", dest="task_output_bits", type=float)
    p.add_argument("--task-deadline-s", type=float)
    p.add_argument("--cpu-rate-cycles-s", type=float)
    p.add_argument("--joule-per-cycle", type=float)
    p.add_argument("--isl-forward-rate-bps", type=float)
    p.add_argument("--isl-return-rate-bps", type=float)
    p.add_argument("--isl-tx-energy-per-bit-j", type=float)
    p.add_argument("--isl-rx-energy-per-bit-j", type=float)
    p.add_argument("--out", type=Path)
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
        elif key == "run":
            if not isinstance(value, dict):
                raise ValueError("config section 'run' must be an object")
        elif key in DEFAULT_CONFIG:
            flat[key] = value
        else:
            raise ValueError(f"unknown config key: {key}")
    return flat


def load_json_config(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON config must be an object")
    return flatten_config(data)


def resolve_config(cli_args: argparse.Namespace) -> argparse.Namespace:
    cli_values = vars(cli_args).copy()
    config_path = cli_values.pop("config", None)
    values = DEFAULT_CONFIG.copy()
    if config_path is not None:
        values.update(load_json_config(config_path))
    values.update({key: value for key, value in cli_values.items() if value is not None})
    values["config"] = config_path
    values["tle_file"] = None if values["tle_file"] is None else Path(values["tle_file"])
    values["task_demand_points_file"] = (
        None if values["task_demand_points_file"] is None else Path(values["task_demand_points_file"])
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
    if not 0 <= args.battery_initial_pct <= 100:
        raise ValueError("--battery-initial-pct must be within [0, 100]")
    if not 0 <= args.battery_min_safe_pct <= 100:
        raise ValueError("--battery-min-safe-pct must be within [0, 100]")
    if args.task_interval_s <= 0:
        raise ValueError("--task-interval-s must be positive")
    if args.tasks_per_sat < 0:
        raise ValueError("--tasks-per-sat must be non-negative")
    if args.task_cpu_cycles <= 0:
        raise ValueError("--task-cpu-cycles must be positive")
    if args.task_deadline_s <= 0:
        raise ValueError("--task-deadline-s must be positive")
    if not 0.0 <= args.task_min_elevation_deg <= 90.0:
        raise ValueError("--task-min-elevation-deg must be within [0, 90]")
    if args.cpu_rate_cycles_s <= 0:
        raise ValueError("--cpu-rate-cycles-s must be positive")
    if args.joule_per_cycle < 0:
        raise ValueError("--joule-per-cycle must be non-negative")
    if args.isl_forward_rate_bps <= 0 or args.isl_return_rate_bps <= 0:
        raise ValueError("--isl-forward-rate-bps and --isl-return-rate-bps must be positive")
    if args.isl_tx_energy_per_bit_j < 0 or args.isl_rx_energy_per_bit_j < 0:
        raise ValueError("--isl-tx-energy-per-bit-j and --isl-rx-energy-per-bit-j must be non-negative")


def build_configs(args: argparse.Namespace) -> tuple[BatteryConfig, TaskConfig, ISLConfig]:
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
        tasks_per_step_choices=tuple(args.tasks_per_step_choices),
        tasks_per_step_weights=tuple(args.tasks_per_step_weights),
        cpu_cycles=args.task_cpu_cycles,
        cpu_cycles_choices=tuple(args.task_cpu_cycles_choices),
        cpu_cycles_weights=tuple(args.task_cpu_cycles_weights),
        input_bits=args.task_input_bits,
        input_bits_choices=tuple(args.task_input_bits_choices),
        input_bits_weights=tuple(args.task_input_bits_weights),
        output_bits=args.task_output_bits,
        output_bits_choices=tuple(args.task_output_bits_choices),
        output_bits_weights=tuple(args.task_output_bits_weights),
        deadline_s=args.task_deadline_s,
        cpu_rate_cycles_s=args.cpu_rate_cycles_s,
        joule_per_cycle=args.joule_per_cycle,
        demand_points=load_demand_points(args.task_demand_points_file),
        min_elevation_deg=args.task_min_elevation_deg,
    )
    isl_config = ISLConfig(
        isl_forward_rate_bps=args.isl_forward_rate_bps,
        isl_return_rate_bps=args.isl_return_rate_bps,
        isl_tx_energy_per_bit_j=args.isl_tx_energy_per_bit_j,
        isl_rx_energy_per_bit_j=args.isl_rx_energy_per_bit_j,
    )
    return battery, task_config, isl_config


def effective_run_config(args: argparse.Namespace) -> dict:
    return {
        "orbit": {
            "orbit_model": args.orbit_model,
            "tle_file": None if args.tle_file is None else str(args.tle_file),
            "sun_position_file": args.sun_position_file,
            "satellites": args.satellites,
            "planes": args.planes,
            "altitude_km": args.altitude_km,
            "inclination_deg": args.inclination_deg,
            "walker_phase": args.walker_phase,
        },
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
            "tasks_per_step_choices": args.tasks_per_step_choices,
            "tasks_per_step_weights": args.tasks_per_step_weights,
            "cpu_cycles": args.task_cpu_cycles,
            "cpu_cycles_choices": args.task_cpu_cycles_choices,
            "cpu_cycles_weights": args.task_cpu_cycles_weights,
            "input_bits": args.task_input_bits,
            "input_bits_choices": args.task_input_bits_choices,
            "input_bits_weights": args.task_input_bits_weights,
            "output_bits": args.task_output_bits,
            "output_bits_choices": args.task_output_bits_choices,
            "output_bits_weights": args.task_output_bits_weights,
            "demand_points_file": None if args.task_demand_points_file is None else str(args.task_demand_points_file),
            "min_elevation_deg": args.task_min_elevation_deg,
            "deadline_s": args.task_deadline_s,
            "cpu_rate_cycles_s": args.cpu_rate_cycles_s,
            "joule_per_cycle": args.joule_per_cycle,
        },
        "isl": {
            "isl_forward_rate_bps": args.isl_forward_rate_bps,
            "isl_return_rate_bps": args.isl_return_rate_bps,
            "isl_tx_energy_per_bit_j": args.isl_tx_energy_per_bit_j,
            "isl_rx_energy_per_bit_j": args.isl_rx_energy_per_bit_j,
        },
        "scheduler": {
            "name": args.scheduler,
        },
        "output": {
            "path": str(args.out),
        },
    }


def run(args: argparse.Namespace) -> int:
    start = parse_utc_datetime(args.start_utc)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(
        json.dumps(effective_run_config(args), indent=2, sort_keys=True) + "\n"
    )
    battery, task_config, isl_config = build_configs(args)
    scheduler = create_scheduler(args.scheduler)

    if args.orbit_model == "tle":
        if args.tle_file is None:
            raise ValueError("--tle-file is required when --orbit-model tle")
        raw_steps = list(
            iter_tle_states(
                tle_file=args.tle_file,
                sun_position_file=args.sun_position_file,
                start=start,
                duration_s=args.duration_s,
                step_s=args.step_s,
                battery=battery,
                task_config=task_config,
                isl_config=isl_config,
                scheduler=scheduler,
            )
        )
        start_context = tle_snapshot_context(sun_position_file=args.sun_position_file, start=start, time_s=0)
        end_context = tle_snapshot_context(sun_position_file=args.sun_position_file, start=start, time_s=raw_steps[-1][0][0].time_s)
    else:
        raw_steps = list(
            iter_circular_states(
                start=start,
                satellites=args.satellites,
                planes=args.planes,
                altitude_km=args.altitude_km,
                inclination_deg=args.inclination_deg,
                duration_s=args.duration_s,
                step_s=args.step_s,
                battery=battery,
                task_config=task_config,
                isl_config=isl_config,
                scheduler=scheduler,
                walker_phase=args.walker_phase,
            )
        )
        start_context = circular_snapshot_context()
        end_context = circular_snapshot_context()

    task_records = [task for _, tasks in raw_steps for task in tasks]
    all_steps = [states for states, _ in raw_steps]
    task_records_by_step = [tasks for _, tasks in raw_steps]

    write_states_csv(args.out / "states.csv", start, all_steps)
    write_summary_csv(args.out / "summary.csv", all_steps, task_records_by_step)
    write_tasks_csv(args.out / "tasks.csv", task_records)
    write_snapshot_svg(args.out / "snapshot_start.svg", all_steps[0], "Orbit snapshot at t=0s", start_context)  # pdf
    write_snapshot_svg(args.out / "snapshot_end.svg", all_steps[-1], f"Orbit snapshot at t={all_steps[-1][0].time_s}s", end_context)
    write_summary_svg(args.out / "sunlight_summary.svg", all_steps)
    write_battery_svg(args.out / "battery_summary.svg", all_steps)
    write_task_svg(args.out / "task_summary.svg", all_steps, task_records_by_step)
    write_sunlight_timeline_svg(args.out / "sunlight_timeline.svg", all_steps)
    write_battery_timeline_svg(args.out / "battery_timeline.svg", all_steps)
    write_task_mode_summary_svg(args.out / "task_mode_summary.svg", task_records)
    write_offload_target_histogram_svg(args.out / "offload_target_histogram.svg", task_records)

    first = all_steps[0]
    last = all_steps[-1]
    print("Minimal orbit simulation complete")
    print(f"  orbit model: {args.orbit_model}")
    print(f"  scheduler: {scheduler.name}")
    print(f"  satellites: {len(first)}")
    if args.orbit_model == "circular":
        print(f"  planes: {args.planes}")
    print(f"  steps: {len(all_steps)}, duration: {args.duration_s}s, step: {args.step_s}s")
    print(f"  t=0 sunlit/eclipse: {sum(s.sunlit for s in first)}/{len(first) - sum(s.sunlit for s in first)}")
    print(f"  final sunlit/eclipse: {sum(s.sunlit for s in last)}/{len(last) - sum(s.sunlit for s in last)}")
    print(f"  final battery min/avg: {min(s.battery_pct for s in last):.2f}%/{sum(s.battery_pct for s in last) / len(last):.2f}%")
    print(f"  tasks completed/failed: {sum(1 for t in task_records if t.completed)}/{sum(1 for t in task_records if not t.completed)}")
    print(f"  output: {args.out.resolve()}")
    print("  open snapshot_start.svg, snapshot_end.svg, sunlight_summary.svg, battery_summary.svg, or task_summary.svg to see results")
    return 0


def main() -> int:
    return run(parse_args())
