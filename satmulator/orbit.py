from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .battery import apply_battery_step, validate_battery_config
from .constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from .geometry import circular_state, is_sunlit_cylindrical_shadow, xy_unit
from .models import (
    Assignment,
    BatteryConfig,
    ISLConfig,
    SatelliteState,
    SatelliteView,
    SnapshotContext,
    Task,
    TaskConfig,
    TaskRecord,
)
from .scheduler import Scheduler
from .runtime import EnvironmentRuntime, SatelliteRuntime


@dataclass
class SatelliteStepStats:
    generated_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    task_energy_j: float = 0.0
    harvested_j: float = 0.0
    consumed_j: float = 0.0


@dataclass(frozen=True)
class AssignmentEnergyCost:
    source_energy_j: float
    target_energy_j: float

    @property
    def total_energy_j(self) -> float:
        return self.source_energy_j + self.target_energy_j


@dataclass(frozen=True)
class AssignmentTimeCost:
    compute_time_s: float
    transmission_time_s: float

    @property
    def total_time_s(self) -> float:
        return self.compute_time_s + self.transmission_time_s


def estimate_assignment_energy(
    *,
    task: Task,
    assignment: Assignment,
    task_config: TaskConfig,
    isl_config: ISLConfig,
) -> AssignmentEnergyCost:
    compute_energy_j = task.cpu_cycles * task_config.joule_per_cycle
    if assignment.mode == "local":
        return AssignmentEnergyCost(
            source_energy_j=compute_energy_j,
            target_energy_j=0.0,
        )
    if assignment.mode == "offload":
        return AssignmentEnergyCost(
            source_energy_j=(
                task.input_bits * isl_config.isl_tx_energy_per_bit_j
                + task.output_bits * isl_config.isl_rx_energy_per_bit_j
            ),
            target_energy_j=(
                task.input_bits * isl_config.isl_rx_energy_per_bit_j
                + compute_energy_j
                + task.output_bits * isl_config.isl_tx_energy_per_bit_j
            ),
        )
    raise ValueError(f"unknown assignment mode: {assignment.mode}")


def estimate_assignment_time(
    *,
    task: Task,
    assignment: Assignment,
    task_config: TaskConfig,
    isl_config: ISLConfig,
) -> AssignmentTimeCost:
    compute_time_s = task.cpu_cycles / task_config.cpu_rate_cycles_s
    if assignment.mode == "local":
        return AssignmentTimeCost(
            compute_time_s=compute_time_s,
            transmission_time_s=0.0,
        )
    if assignment.mode == "offload":
        return AssignmentTimeCost(
            compute_time_s=compute_time_s,
            transmission_time_s=(
                task.input_bits / isl_config.isl_forward_rate_bps
                + task.output_bits / isl_config.isl_return_rate_bps
            ),
        )
    raise ValueError(f"unknown assignment mode: {assignment.mode}")


def generate_step_tasks(env: EnvironmentRuntime, task_config: TaskConfig) -> list[Task]:
    if not task_config.enabled:
        return []
    if env.time_s <= 0 or env.time_s % task_config.interval_s != 0:
        return []

    tasks: list[Task] = []
    for sat in env.satellites:
        for _ in range(task_config.tasks_per_sat):
            tasks.append(
                Task(
                    task_id=env.next_task_id,
                    created_time_s=env.time_s,
                    source_sat=sat.sat_id,
                    cpu_cycles=task_config.cpu_cycles,
                    input_bits=task_config.input_bits,
                    output_bits=task_config.output_bits,
                    deadline_s=task_config.deadline_s,
                )
            )
            env.next_task_id += 1
    return tasks


def assign_step_tasks(
    *,
    scheduler: Scheduler,
    tasks: list[Task],
    satellite_views: list[SatelliteView],
) -> list[Assignment]:
    return [
        scheduler.assign_task(task=task, satellite_views=satellite_views)
        for task in tasks
    ]


def apply_step(
    *,
    env: EnvironmentRuntime,
    step_s: int,
    battery: BatteryConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    tasks: list[Task],
    assignments: list[Assignment],
) -> tuple[list[SatelliteState], list[TaskRecord]]:
    stats_by_sat = {sat.sat_id: SatelliteStepStats() for sat in env.satellites}
    records: list[TaskRecord] = []
    task_by_id = {task.task_id: task for task in tasks}
    battery_before = {sat.sat_id: sat.battery_j for sat in env.satellites}
    idle_energy_by_sat = {
        sat.sat_id: battery.idle_w * step_s if env.time_s > 0 else 0.0
        for sat in env.satellites
    }

    for task in tasks:
        stats_by_sat[task.source_sat].generated_tasks += 1

    for assignment in assignments:
        task = task_by_id[assignment.task_id]
        source_stats = stats_by_sat[assignment.source_sat]
        target_stats = stats_by_sat[assignment.target_sat]
        time_cost = estimate_assignment_time(
            task=task,
            assignment=assignment,
            task_config=task_config,
            isl_config=isl_config,
        )
        cost = estimate_assignment_energy(
            task=task,
            assignment=assignment,
            task_config=task_config,
            isl_config=isl_config,
        )
        failed_reason = ""
        is_completed = True
        source_energy_j = 0.0
        target_energy_j = 0.0

        if time_cost.total_time_s > task.deadline_s:
            is_completed = False
            failed_reason = "deadline"
        else:
            source_required_j = (
                idle_energy_by_sat[assignment.source_sat]
                + source_stats.task_energy_j
                + cost.source_energy_j
            )
            target_required_j = (
                idle_energy_by_sat[assignment.target_sat]
                + target_stats.task_energy_j
                + cost.target_energy_j
            )
            if battery_before[assignment.source_sat] < source_required_j:
                is_completed = False
                failed_reason = "battery"
            elif battery_before[assignment.target_sat] < target_required_j:
                is_completed = False
                failed_reason = "target_battery"
            else:
                source_energy_j = cost.source_energy_j
                target_energy_j = cost.target_energy_j

        if is_completed:
            source_stats.completed_tasks += 1
            source_stats.task_energy_j += source_energy_j
            target_stats.task_energy_j += target_energy_j
            env.completed_tasks.append(task.task_id)
        else:
            source_stats.failed_tasks += 1
            env.failed_tasks.append(task.task_id)

        records.append(
            TaskRecord(
                task_id=assignment.task_id,
                created_time_s=task.created_time_s,
                source_sat=assignment.source_sat,
                target_sat=assignment.target_sat,
                mode=assignment.mode,
                cpu_cycles=task.cpu_cycles,
                input_bits=task.input_bits,
                output_bits=task.output_bits,
                deadline_s=task.deadline_s,
                compute_time_s=time_cost.compute_time_s,
                transmission_time_s=time_cost.transmission_time_s,
                total_time_s=time_cost.total_time_s,
                energy_j=source_energy_j if is_completed else 0.0,
                completed=is_completed,
                failed_reason=failed_reason,
                source_energy_j=source_energy_j if is_completed else 0.0,
                target_energy_j=target_energy_j if is_completed else 0.0,
                total_energy_j=(source_energy_j + target_energy_j) if is_completed else 0.0,
            )
        )

    states: list[SatelliteState] = []
    for sat in env.satellites:
        stats = stats_by_sat[sat.sat_id]
        battery_now, harvested_j, consumed_j = apply_battery_step(
            battery_now=sat.battery_j,
            sunlit=sat.sunlit,
            step_s=step_s,
            battery=battery,
            task_energy_j=stats.task_energy_j,
            update=env.time_s > 0,
        )
        sat.battery_j = battery_now
        stats.harvested_j = harvested_j
        stats.consumed_j = consumed_j
        states.append(
            sat.snapshot(
                time_s=env.time_s,
                harvested_j=stats.harvested_j,
                consumed_j=stats.consumed_j,
                battery=battery,
                generated_tasks=stats.generated_tasks,
                completed_tasks=stats.completed_tasks,
                failed_tasks=stats.failed_tasks,
                task_energy_j=stats.task_energy_j,
            )
        )
    return states, records


def iter_circular_states(
    *,
    satellites: int,
    planes: int,
    altitude_km: float,
    inclination_deg: float,
    duration_s: int,
    step_s: int,
    battery: BatteryConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    scheduler: Scheduler,
    walker_phase: int = 0,
) -> Iterable[tuple[list[SatelliteState], list[TaskRecord]]]:
    if satellites <= 0:
        raise ValueError("satellites must be positive")
    if planes <= 0 or satellites % planes != 0:
        raise ValueError("planes must be positive and divide satellites")
    if step_s <= 0:
        raise ValueError("step must be positive")
    validate_battery_config(battery)

    sats_per_plane = satellites // planes
    radius_km = EARTH_RADIUS_KM + altitude_km
    inclination_rad = math.radians(inclination_deg)
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (radius_km**3))
    env = EnvironmentRuntime(
        satellites=[
            SatelliteRuntime(
                sat_id=sat_id,
                name=f"sat_{sat_id}",
                plane=sat_id // sats_per_plane,
                slot=sat_id % sats_per_plane,
                battery_j=battery.initial_j,
            )
            for sat_id in range(satellites)
        ]
    )

    for time_s in range(0, duration_s + 1, step_s):
        env.time_s = time_s
        for plane in range(planes):
            raan = 2.0 * math.pi * plane / planes
            plane_phase = 2.0 * math.pi * walker_phase * plane / satellites
            for slot in range(sats_per_plane):
                sat_id = plane * sats_per_plane + slot
                arg = 2.0 * math.pi * slot / sats_per_plane + plane_phase + mean_motion * time_s
                pos, vel = circular_state(radius_km, inclination_rad, raan, arg)
                sunlit = is_sunlit_cylindrical_shadow(pos)
                env.satellites[sat_id].update_orbit(pos_km=pos, vel_km_s=vel, sunlit=sunlit)

        tasks = generate_step_tasks(env, task_config)
        assignments = assign_step_tasks(
            scheduler=scheduler,
            tasks=tasks,
            satellite_views=env.views(),
        )
        states, task_records = apply_step(
            env=env,
            step_s=step_s,
            battery=battery,
            task_config=task_config,
            isl_config=isl_config,
            tasks=tasks,
            assignments=assignments,
        )
        yield states, task_records


def load_tle_satellites(tle_file: Path):
    try:
        from skyfield.api import load
        from skyfield.iokit import parse_tle_file
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for --orbit-model tle. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    ts = load.timescale()
    with tle_file.open("rb") as f:
        satellites = list(parse_tle_file(f, ts))
    if not satellites:
        raise ValueError(f"no satellites found in TLE file: {tle_file}")
    return ts, satellites


def iter_tle_states(
    *,
    tle_file: Path,
    sun_position_file: str,
    start: dt.datetime,
    duration_s: int,
    step_s: int,
    battery: BatteryConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    scheduler: Scheduler,
) -> Iterable[tuple[list[SatelliteState], list[TaskRecord]]]:
    if step_s <= 0:
        raise ValueError("step must be positive")
    validate_battery_config(battery)

    try:
        from skyfield.api import load, wgs84
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for --orbit-model tle. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    ts, satellites = load_tle_satellites(tle_file)
    eph = load(sun_position_file)
    env = EnvironmentRuntime(
        satellites=[
            SatelliteRuntime(
                sat_id=sat_id,
                name=sat.name or f"sat_{sat_id}",
                plane=-1,
                slot=sat_id,
                battery_j=battery.initial_j,
            )
            for sat_id, sat in enumerate(satellites)
        ]
    )

    for time_s in range(0, duration_s + 1, step_s):
        env.time_s = time_s
        now = start + dt.timedelta(seconds=time_s)
        t = ts.from_datetime(now)
        for sat_id, sat in enumerate(satellites):
            geocentric = sat.at(t)
            pos = tuple(float(x) for x in geocentric.position.km)
            vel = tuple(float(x) for x in geocentric.velocity.km_per_s)
            subpoint = wgs84.subpoint(geocentric)
            sunlit = bool(geocentric.is_sunlit(eph))
            env.satellites[sat_id].update_orbit(
                pos_km=pos,
                vel_km_s=vel,
                lat_deg=float(subpoint.latitude.degrees),
                lon_deg=float(subpoint.longitude.degrees),
                elevation_km=float(subpoint.elevation.km),
                sunlit=sunlit,
            )

        tasks = generate_step_tasks(env, task_config)
        assignments = assign_step_tasks(
            scheduler=scheduler,
            tasks=tasks,
            satellite_views=env.views(),
        )
        states, task_records = apply_step(
            env=env,
            step_s=step_s,
            battery=battery,
            task_config=task_config,
            isl_config=isl_config,
            tasks=tasks,
            assignments=assignments,
        )
        yield states, task_records


def circular_snapshot_context() -> SnapshotContext:
    return SnapshotContext(
        projection_label="ECI x-y projection; circular orbit model uses fixed +x sun direction",
        sun_xy_unit=(1.0, 0.0),
    )


def tle_snapshot_context(*, sun_position_file: str, start: dt.datetime, time_s: int) -> SnapshotContext:
    try:
        from skyfield.api import load
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for --orbit-model tle. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    ts = load.timescale()
    eph = load(sun_position_file)
    t = ts.from_datetime(start + dt.timedelta(seconds=time_s))
    earth = eph["earth"].at(t)
    sun = eph["sun"].at(t)
    sun_vector = tuple(float(x) for x in (sun.position.km - earth.position.km))
    return SnapshotContext(
        projection_label="ECI x-y projection; sun arrow is the real Sun vector projected into this plane",
        sun_xy_unit=xy_unit(sun_vector),
    )
