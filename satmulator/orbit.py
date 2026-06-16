from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .battery import apply_battery_step, validate_battery_config
from .constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from .geometry import circular_state, is_sunlit_cylindrical_shadow, vector_unit, xy_unit
from .models import (
    Assignment,
    BatteryConfig,
    ISLConfig,
    SatelliteState,
    SatelliteView,
    SnapshotContext,
    SchedulerConfig,
    Task,
    TaskConfig,
    TaskRecord,
)
from .runtime import EnvironmentRuntime, SatelliteRuntime, TaskEventSink
from .scheduler import Scheduler
from .workload import generate_step_tasks, validate_task_config


StepSink = Callable[[list[SatelliteState], SnapshotContext], None]


@dataclass
class SatelliteStepStats:
    generated_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    deferred_tasks: int = 0
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


def assign_step_tasks(
    *,
    scheduler: Scheduler,
    tasks: list[Task],
    satellite_views: list[SatelliteView],
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    scheduler_config: SchedulerConfig,
) -> list[Assignment]:
    if any(task.source_sat is None for task in tasks):
        raise ValueError("cannot assign tasks without a visible source satellite")

    return scheduler.assign_tasks(
        tasks=tasks,
        satellite_views=satellite_views,
        time_s=time_s,
        step_s=step_s,
        battery=battery,
        task_config=task_config,
        isl_config=isl_config,
        scheduler_config=scheduler_config,
    )


def pop_deferred_tasks(env: EnvironmentRuntime) -> tuple[list[Task], list[Task]]:
    ready: list[Task] = []
    expired: list[Task] = []

    for task in env.deferred_tasks:
        if env.time_s - task.created_time_s >= task.deadline_s:
            expired.append(task)
        else:
            ready.append(task)

    env.deferred_tasks.clear()
    return ready, expired


def apply_step(
    *,
    env: EnvironmentRuntime,
    step_s: int,
    battery: BatteryConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    tasks: list[Task],
    assignments: list[Assignment],
    expired_tasks: list[Task] | None = None,
) -> tuple[list[SatelliteState], list[TaskRecord]]:
    stats_by_sat = {sat.sat_id: SatelliteStepStats() for sat in env.satellites}
    records: list[TaskRecord] = []
    task_by_id = {task.task_id: task for task in tasks}
    battery_before = {sat.sat_id: sat.battery_j for sat in env.satellites}

    idle_energy_by_sat = {
        sat.sat_id: battery.idle_w * step_s if env.time_s > 0 else 0.0
        for sat in env.satellites
    }

    def remaining_deadline_s(task: Task) -> float:
        return task.created_time_s + task.deadline_s - env.time_s

    def make_task_record(
        *,
        task: Task,
        source_sat: int,
        target_sat: int,
        mode: str,
        waiting_time_s: float,
        compute_time_s: float,
        transmission_time_s: float,
        total_time_s: float,
        energy_j: float,
        completed: bool,
        failed_reason: str,
        source_energy_j: float,
        target_energy_j: float,
        total_energy_j: float,
        status: str,
        score: float,
    ) -> TaskRecord:
        return TaskRecord(
            task_id=task.task_id,
            created_time_s=task.created_time_s,
            source_sat=source_sat,
            target_sat=target_sat,
            mode=mode,
            lat_deg=task.lat_deg,
            lon_deg=task.lon_deg,
            cpu_cycles=task.cpu_cycles,
            input_bits=task.input_bits,
            output_bits=task.output_bits,
            deadline_s=task.deadline_s,
            waiting_time_s=waiting_time_s,
            compute_time_s=compute_time_s,
            transmission_time_s=transmission_time_s,
            total_time_s=total_time_s,
            energy_j=energy_j,
            completed=completed,
            failed_reason=failed_reason,
            source_energy_j=source_energy_j,
            target_energy_j=target_energy_j,
            total_energy_j=total_energy_j,
            status=status,
            remaining_deadline_s=remaining_deadline_s(task),
            score=score,
        )

    expired_tasks = [] if expired_tasks is None else expired_tasks

    # Only count tasks generated at this time slot.
    # Deferred tasks that re-enter Q(t) should not be counted as newly generated.
    for task in tasks:
        if task.source_sat is not None and task.created_time_s == env.time_s:
            stats_by_sat[task.source_sat].generated_tasks += 1

    # Expired tasks can come from two places:
    # 1. demand-point workload pending tasks with no coverage
    # 2. slack-aware deferred tasks whose deadline has expired
    for task in expired_tasks:
        waiting_time_s = env.time_s - task.created_time_s
        env.failed_tasks.append(task.task_id)
        source_sat = task.source_sat if task.source_sat is not None else -1
        target_sat = source_sat

        if source_sat in stats_by_sat:
            stats_by_sat[source_sat].failed_tasks += 1
            failed_reason = "deadline"
        else:
            failed_reason = "no_coverage"

        env.emit_task_event(
            "task_failed",
            task.task_id,
            source_sat=source_sat,
            reason=failed_reason,
            waiting_time_s=waiting_time_s,
        )
        records.append(
            make_task_record(
                task=task,
                source_sat=source_sat,
                target_sat=target_sat,
                mode="unassigned",
                waiting_time_s=waiting_time_s,
                compute_time_s=0.0,
                transmission_time_s=0.0,
                total_time_s=waiting_time_s,
                energy_j=0.0,
                completed=False,
                failed_reason=failed_reason,
                source_energy_j=0.0,
                target_energy_j=0.0,
                total_energy_j=0.0,
                status="failed",
                score=0.0,
            )
        )

    for assignment in assignments:
        task = task_by_id[assignment.task_id]
        waiting_time_s = env.time_s - task.created_time_s
        env.emit_task_event(
            "task_assigned",
            task.task_id,
            source_sat=assignment.source_sat,
            target_sat=assignment.target_sat,
            route=list(assignment.route.nodes),
            mode=assignment.mode,
            score=assignment.score,
            waiting_time_s=waiting_time_s,
        )
        source_stats = stats_by_sat[assignment.source_sat]
        target_stats = stats_by_sat[assignment.target_sat]

        # Slack-aware defer action.
        # A deferred task does not consume task energy and is not counted as failed.
        if assignment.mode == "defer":
            env.deferred_tasks.append(task)
            source_stats.deferred_tasks += 1
            env.emit_task_event(
                "task_deferred",
                task.task_id,
                source_sat=assignment.source_sat,
                route=list(assignment.route.nodes),
                score=assignment.score,
                waiting_time_s=waiting_time_s,
            )

            records.append(
                make_task_record(
                    task=task,
                    source_sat=assignment.source_sat,
                    target_sat=assignment.target_sat,
                    mode="defer",
                    waiting_time_s=waiting_time_s,
                    compute_time_s=0.0,
                    transmission_time_s=0.0,
                    total_time_s=waiting_time_s,
                    energy_j=0.0,
                    completed=False,
                    failed_reason="",
                    source_energy_j=0.0,
                    target_energy_j=0.0,
                    total_energy_j=0.0,
                    status="deferred",
                    score=assignment.score,
                )
            )
            continue

        # Slack-aware explicit fail action.
        if assignment.mode == "fail":
            source_stats.failed_tasks += 1
            env.failed_tasks.append(task.task_id)
            env.emit_task_event(
                "task_failed",
                task.task_id,
                source_sat=assignment.source_sat,
                target_sat=assignment.target_sat,
                route=list(assignment.route.nodes),
                mode="fail",
                reason=assignment.failed_reason or "scheduler_fail",
                score=assignment.score,
                waiting_time_s=waiting_time_s,
            )

            records.append(
                make_task_record(
                    task=task,
                    source_sat=assignment.source_sat,
                    target_sat=assignment.target_sat,
                    mode="fail",
                    waiting_time_s=waiting_time_s,
                    compute_time_s=0.0,
                    transmission_time_s=0.0,
                    total_time_s=waiting_time_s,
                    energy_j=0.0,
                    completed=False,
                    failed_reason=assignment.failed_reason or "scheduler_fail",
                    source_energy_j=0.0,
                    target_energy_j=0.0,
                    total_energy_j=0.0,
                    status="failed",
                    score=assignment.score,
                )
            )
            continue

        # Existing local / offload execution path.
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

        total_time_s = waiting_time_s + time_cost.total_time_s

        if total_time_s > task.deadline_s:
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
            status = "completed"
            env.emit_task_event(
                "task_completed",
                task.task_id,
                source_sat=assignment.source_sat,
                target_sat=assignment.target_sat,
                route=list(assignment.route.nodes),
                mode=assignment.mode,
                waiting_time_s=waiting_time_s,
                compute_time_s=time_cost.compute_time_s,
                transmission_time_s=time_cost.transmission_time_s,
                total_time_s=total_time_s,
                energy_j={
                    "source": source_energy_j,
                    "target": target_energy_j,
                    "total": source_energy_j + target_energy_j,
                },
                score=assignment.score,
            )
        else:
            source_stats.failed_tasks += 1
            env.failed_tasks.append(task.task_id)
            status = "failed"
            env.emit_task_event(
                "task_failed",
                task.task_id,
                source_sat=assignment.source_sat,
                target_sat=assignment.target_sat,
                route=list(assignment.route.nodes),
                mode=assignment.mode,
                reason=failed_reason,
                score=assignment.score,
                waiting_time_s=waiting_time_s,
                total_time_s=total_time_s,
            )

        records.append(
            make_task_record(
                task=task,
                source_sat=assignment.source_sat,
                target_sat=assignment.target_sat,
                mode=assignment.mode,
                waiting_time_s=waiting_time_s,
                compute_time_s=time_cost.compute_time_s,
                transmission_time_s=time_cost.transmission_time_s,
                total_time_s=total_time_s,
                energy_j=source_energy_j if is_completed else 0.0,
                completed=is_completed,
                failed_reason=failed_reason,
                source_energy_j=source_energy_j if is_completed else 0.0,
                target_energy_j=target_energy_j if is_completed else 0.0,
                total_energy_j=(source_energy_j + target_energy_j)
                if is_completed
                else 0.0,
                status=status,
                score=assignment.score,
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
                deferred_tasks=stats.deferred_tasks,
                task_energy_j=stats.task_energy_j,
            )
        )

    return states, records


def iter_circular_states(
    *,
    start: dt.datetime | None = None,
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
    scheduler_config: SchedulerConfig,
    walker_phase: int = 0,
    task_event_sink: TaskEventSink | None = None,
    step_sink: StepSink | None = None,
) -> Iterable[tuple[list[SatelliteState], list[TaskRecord]]]:
    if satellites <= 0:
        raise ValueError("satellites must be positive")
    if planes <= 0 or satellites % planes != 0:
        raise ValueError("planes must be positive and divide satellites")
    if step_s <= 0:
        raise ValueError("step must be positive")

    validate_battery_config(battery)
    validate_task_config(task_config)

    sats_per_plane = satellites // planes
    radius_km = EARTH_RADIUS_KM + altitude_km
    inclination_rad = math.radians(inclination_deg)
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (radius_km**3))

    env = EnvironmentRuntime(
        rng=random.Random(task_config.random_seed),
        task_event_sink=task_event_sink,
        satellites=[
            SatelliteRuntime(
                sat_id=sat_id,
                name=f"sat_{sat_id}",
                plane=sat_id // sats_per_plane,
                slot=sat_id % sats_per_plane,
                battery_j=battery.initial_j,
            )
            for sat_id in range(satellites)
        ],
    )

    for time_s in range(0, duration_s + 1, step_s):
        env.time_s = time_s
        env.time_utc = None if start is None else start + dt.timedelta(seconds=time_s)

        for plane in range(planes):
            raan = 2.0 * math.pi * plane / planes
            plane_phase = 2.0 * math.pi * walker_phase * plane / satellites

            for slot in range(sats_per_plane):
                sat_id = plane * sats_per_plane + slot
                arg = (
                    2.0 * math.pi * slot / sats_per_plane
                    + plane_phase
                    + mean_motion * time_s
                )
                pos, vel = circular_state(radius_km, inclination_rad, raan, arg)
                sunlit = is_sunlit_cylindrical_shadow(pos)

                env.satellites[sat_id].update_orbit(
                    pos_km=pos,
                    vel_km_s=vel,
                    sunlit=sunlit,
                )

        new_tasks, expired_tasks = generate_step_tasks(env, task_config)
        deferred_tasks, expired_deferred_tasks = pop_deferred_tasks(env)

        tasks = deferred_tasks + new_tasks
        expired_tasks = expired_tasks + expired_deferred_tasks

        assignments = assign_step_tasks(
            scheduler=scheduler,
            tasks=tasks,
            satellite_views=env.views(),
            time_s=env.time_s,
            step_s=step_s,
            battery=battery,
            task_config=task_config,
            isl_config=isl_config,
            scheduler_config=scheduler_config,
        )

        states, task_records = apply_step(
            env=env,
            step_s=step_s,
            battery=battery,
            task_config=task_config,
            isl_config=isl_config,
            tasks=tasks,
            assignments=assignments,
            expired_tasks=expired_tasks,
        )

        if step_sink is not None:
            step_sink(states, circular_snapshot_context())
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
    scheduler_config: SchedulerConfig,
    task_event_sink: TaskEventSink | None = None,
    step_sink: StepSink | None = None,
) -> Iterable[tuple[list[SatelliteState], list[TaskRecord]]]:
    if step_s <= 0:
        raise ValueError("step must be positive")

    validate_battery_config(battery)
    validate_task_config(task_config)

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
        rng=random.Random(task_config.random_seed),
        task_event_sink=task_event_sink,
        satellites=[
            SatelliteRuntime(
                sat_id=sat_id,
                name=sat.name or f"sat_{sat_id}",
                plane=-1,
                slot=sat_id,
                battery_j=battery.initial_j,
            )
            for sat_id, sat in enumerate(satellites)
        ],
    )

    for time_s in range(0, duration_s + 1, step_s):
        env.time_s = time_s
        now = start + dt.timedelta(seconds=time_s)
        env.time_utc = now
        t = ts.from_datetime(now)
        earth = eph["earth"].at(t)
        sun = eph["sun"].at(t)
        sun_vector = tuple(float(x) for x in (sun.position.km - earth.position.km))
        context = snapshot_context_from_sun_vector(sun_vector)

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

        new_tasks, expired_tasks = generate_step_tasks(env, task_config)
        deferred_tasks, expired_deferred_tasks = pop_deferred_tasks(env)

        tasks = deferred_tasks + new_tasks
        expired_tasks = expired_tasks + expired_deferred_tasks

        assignments = assign_step_tasks(
            scheduler=scheduler,
            tasks=tasks,
            satellite_views=env.views(),
            time_s=env.time_s,
            step_s=step_s,
            battery=battery,
            task_config=task_config,
            isl_config=isl_config,
            scheduler_config=scheduler_config,
        )

        states, task_records = apply_step(
            env=env,
            step_s=step_s,
            battery=battery,
            task_config=task_config,
            isl_config=isl_config,
            tasks=tasks,
            assignments=assignments,
            expired_tasks=expired_tasks,
        )

        if step_sink is not None:
            step_sink(states, context)
        yield states, task_records


def circular_snapshot_context() -> SnapshotContext:
    return SnapshotContext(
        projection_label="ECI x-y projection; circular orbit model uses fixed +x sun direction",
        sun_xy_unit=(1.0, 0.0),
        sun_eci_unit=(1.0, 0.0, 0.0),
    )


def snapshot_context_from_sun_vector(
    sun_vector: tuple[float, float, float],
) -> SnapshotContext:
    return SnapshotContext(
        projection_label="ECI x-y projection; sun arrow is the real Sun vector projected into this plane",
        sun_xy_unit=xy_unit(sun_vector),
        sun_eci_unit=vector_unit(sun_vector),
    )


def tle_snapshot_context(
    *,
    sun_position_file: str,
    start: dt.datetime,
    time_s: int,
) -> SnapshotContext:
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

    return snapshot_context_from_sun_vector(sun_vector)
