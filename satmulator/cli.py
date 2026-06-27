from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from .models import BatteryConfig, ISLConfig, SchedulerConfig, TaskConfig
from .orbit import iter_circular_states, iter_tle_states
from .plotting import render_run_plots
from .runlog import RunLog
from .scheduler import create_scheduler
from .workload import demand_points_provenance, load_demand_points


DEFAULT_CONFIG = {
    "run_name": "satmulator",
    "run_description": "",
    "orbit_model": "circular",
    "tle_file": None,
    "sun_position_file": "de440s.bsp",
    "start_utc": "2026-05-22T12:00:00Z",
    "satellites": 1584,
    "planes": 72,
    "altitude_km": 550.0,
    "inclination_deg": 53.05,
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
    "isl_topology": "grid",
    "isl_max_range_km": 5000.0,
    "out": "output/minimal_orbit",
    "scheduler_load_max_cycles_per_slot": 4.0e9,
    "scheduler_defer_penalty": 3.0,
    "scheduler_fail_penalty": 1000.0,
    "scheduler_time_weight": 1.0,
    "scheduler_energy_weight": 2.0,
    "scheduler_battery_weight": 5.0,
    "scheduler_load_weight": 0.1,
    "scheduler_eclipse_local_penalty": 2.0,
    "scheduler_low_battery_threshold_pct": 35.0,
}


CONFIG_SECTIONS = {
    "run": {
        "name": "run_name",
        "description": "run_description",
    },
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
        "topology": "isl_topology",
        "max_range_km": "isl_max_range_km",
    },
    "scheduler": {
        "name": "scheduler",
        "load_max_cycles_per_slot": "scheduler_load_max_cycles_per_slot",
        "defer_penalty": "scheduler_defer_penalty",
        "fail_penalty": "scheduler_fail_penalty",
        "time_weight": "scheduler_time_weight",
        "energy_weight": "scheduler_energy_weight",
        "battery_weight": "scheduler_battery_weight",
        "load_weight": "scheduler_load_weight",
        "eclipse_local_penalty": "scheduler_eclipse_local_penalty",
        "low_battery_threshold_pct": "scheduler_low_battery_threshold_pct",
    },
    "output": {"path": "out"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the minimal satellite orbit simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, help="JSON config file; CLI flags override it"
    )
    p.add_argument("--orbit-model", dest="orbit_model", choices=("circular", "tle"))
    p.add_argument("--tle-file", type=Path, help="local TLE file for --orbit-model tle")
    p.add_argument(
        "--sun-position-file",
        dest="sun_position_file",
        help="local file used by Skyfield for Sun position",
    )
    p.add_argument("--start-utc")
    p.add_argument("--satellites", type=int, help="total satellite count")
    p.add_argument(
        "--planes", type=int, help="orbital plane count; must divide satellites"
    )
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
    p.add_argument(
        "--task-enable",
        dest="task_enable",
        action="store_true",
        default=None,
        help="enable deterministic local tasks",
    )
    p.add_argument(
        "--no-task",
        dest="task_enable",
        action="store_false",
        default=None,
        help="disable task generation and execution",
    )
    p.add_argument("--scheduler", choices=("local", "nearest-sunlit", "slack-aware"))
    p.add_argument("--scheduler-load-max-cycles-per-slot", type=float)
    p.add_argument("--scheduler-defer-penalty", type=float)
    p.add_argument("--scheduler-fail-penalty", type=float)
    p.add_argument("--scheduler-time-weight", type=float)
    p.add_argument("--scheduler-energy-weight", type=float)
    p.add_argument("--scheduler-battery-weight", type=float)
    p.add_argument("--scheduler-load-weight", type=float)
    p.add_argument("--scheduler-eclipse-local-penalty", type=float)
    p.add_argument("--scheduler-low-battery-threshold-pct", type=float)
    p.add_argument("--task-interval-s", type=int)
    p.add_argument(
        "--task-generation-mode", choices=("satellite-deterministic", "demand-points")
    )
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
    p.add_argument("--isl-topology", choices=("fully-connected", "grid"))
    p.add_argument("--isl-max-range-km", type=float)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--plot-run",
        type=Path,
        help="regenerate SVG plots from an existing JSON/JSONL run log",
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
    plot_run = cli_values.pop("plot_run", None)
    values = DEFAULT_CONFIG.copy()
    if config_path is not None:
        values.update(load_json_config(config_path))
    values.update(
        {key: value for key, value in cli_values.items() if value is not None}
    )
    values["config"] = config_path
    values["plot_run"] = plot_run
    values["tle_file"] = (
        None if values["tle_file"] is None else Path(values["tle_file"])
    )
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
        raise ValueError(
            "--isl-forward-rate-bps and --isl-return-rate-bps must be positive"
        )
    if args.isl_tx_energy_per_bit_j < 0 or args.isl_rx_energy_per_bit_j < 0:
        raise ValueError(
            "--isl-tx-energy-per-bit-j and --isl-rx-energy-per-bit-j must be non-negative"
        )
    if args.isl_topology not in {"fully-connected", "grid"}:
        raise ValueError("--isl-topology must be fully-connected or grid")
    if args.isl_topology == "grid" and (
        args.isl_max_range_km is None or args.isl_max_range_km <= 0.0
    ):
        raise ValueError("--isl-max-range-km must be positive for grid topology")
    if args.orbit_model == "tle" and args.isl_topology == "grid":
        raise ValueError(
            "grid ISL topology requires plane/slot metadata unavailable in TLE mode; "
            "use --isl-topology fully-connected"
        )
    if args.scheduler_load_max_cycles_per_slot <= 0:
        raise ValueError("--scheduler-load-max-cycles-per-slot must be positive")
    if args.scheduler_fail_penalty < 0 or args.scheduler_defer_penalty < 0:
        raise ValueError("scheduler penalties must be non-negative")
    if not 0 <= args.scheduler_low_battery_threshold_pct <= 100:
        raise ValueError(
            "--scheduler-low-battery-threshold-pct must be within [0, 100]"
        )


def build_configs(
    args: argparse.Namespace,
) -> tuple[BatteryConfig, TaskConfig, ISLConfig, SchedulerConfig]:
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
        demand_distribution=load_demand_points(args.task_demand_points_file),
        min_elevation_deg=args.task_min_elevation_deg,
    )
    isl_config = ISLConfig(
        isl_forward_rate_bps=args.isl_forward_rate_bps,
        isl_return_rate_bps=args.isl_return_rate_bps,
        isl_tx_energy_per_bit_j=args.isl_tx_energy_per_bit_j,
        isl_rx_energy_per_bit_j=args.isl_rx_energy_per_bit_j,
        topology=args.isl_topology,
        max_range_km=args.isl_max_range_km,
    )
    scheduler_config = SchedulerConfig(
        name=args.scheduler,
        load_max_cycles_per_slot=args.scheduler_load_max_cycles_per_slot,
        defer_penalty=args.scheduler_defer_penalty,
        fail_penalty=args.scheduler_fail_penalty,
        time_weight=args.scheduler_time_weight,
        energy_weight=args.scheduler_energy_weight,
        battery_weight=args.scheduler_battery_weight,
        load_weight=args.scheduler_load_weight,
        eclipse_local_penalty=args.scheduler_eclipse_local_penalty,
        low_battery_threshold_pct=args.scheduler_low_battery_threshold_pct,
    )
    return battery, task_config, isl_config, scheduler_config


def effective_run_config(args: argparse.Namespace) -> dict:
    if args.orbit_model == "tle":
        orbit_config = {
            "orbit_model": args.orbit_model,
            "tle_file": None if args.tle_file is None else str(args.tle_file),
            "sun_position_file": args.sun_position_file,
        }
    else:
        orbit_config = {
            "orbit_model": args.orbit_model,
            "satellites": args.satellites,
            "planes": args.planes,
            "altitude_km": args.altitude_km,
            "inclination_deg": args.inclination_deg,
            "walker_phase": args.walker_phase,
        }
    return {
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
            "demand_points_file": None
            if args.task_demand_points_file is None
            else str(args.task_demand_points_file),
            "demand_points_provenance": demand_points_provenance(
                args.task_demand_points_file
            ),
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
            "topology": args.isl_topology,
            "max_range_km": args.isl_max_range_km,
        },
        "scheduler": {
            "name": args.scheduler,
            "load_max_cycles_per_slot": args.scheduler_load_max_cycles_per_slot,
            "defer_penalty": args.scheduler_defer_penalty,
            "fail_penalty": args.scheduler_fail_penalty,
            "time_weight": args.scheduler_time_weight,
            "energy_weight": args.scheduler_energy_weight,
            "battery_weight": args.scheduler_battery_weight,
            "load_weight": args.scheduler_load_weight,
            "eclipse_local_penalty": args.scheduler_eclipse_local_penalty,
            "low_battery_threshold_pct": args.scheduler_low_battery_threshold_pct,
        },
        "output": {
            "path": str(args.out),
        },
    }


def run(args: argparse.Namespace) -> int:
    start = parse_utc_datetime(args.start_utc)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    run_config = effective_run_config(args)
    battery, task_config, isl_config, scheduler_config = build_configs(args)
    scheduler = create_scheduler(args.scheduler)
    run_log = RunLog(args.out, start, run_config)

    try:
        common = {
            "start": start,
            "duration_s": args.duration_s,
            "step_s": args.step_s,
            "battery": battery,
            "task_config": task_config,
            "isl_config": isl_config,
            "scheduler": scheduler,
            "scheduler_config": scheduler_config,
            "task_event_sink": run_log.write_task_event,
            "step_sink": run_log.write_step,
        }

        if args.orbit_model == "tle":
            if args.tle_file is None:
                raise ValueError("--tle-file is required when --orbit-model tle")
            step_iterator = iter_tle_states(
                tle_file=args.tle_file,
                sun_position_file=args.sun_position_file,
                **common,
            )
        else:
            step_iterator = iter_circular_states(
                satellites=args.satellites,
                planes=args.planes,
                altitude_km=args.altitude_km,
                inclination_deg=args.inclination_deg,
                walker_phase=args.walker_phase,
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

    render_run_plots(args.out)
    summary = json.loads((args.out / "summary.json").read_text())
    task_summary = summary["tasks"]

    print("Minimal orbit simulation complete")
    print(f"  orbit model: {args.orbit_model}")
    print(f"  scheduler: {scheduler.name}")
    print(f"  satellites: {len(first)}")
    if args.orbit_model == "circular":
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
    print(f"  output: {args.out.resolve()}")
    print(
        "  open snapshot_start.svg, snapshot_end.svg, sunlight_summary.svg, "
        "battery_summary.svg, or task_summary.svg to see results"
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.plot_run is not None:
        render_run_plots(args.plot_run)
        print(f"Plots regenerated from {args.plot_run.resolve()}")
        return 0
    return run(args)
