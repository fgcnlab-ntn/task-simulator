from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from .battery import (
    apply_battery_step,
    validate_battery_config,
)
from .constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from .geometry import circular_state, is_sunlit_cylindrical_shadow, vector_unit, xy_unit
from .isl import build_constellation_layout, build_isl_graph
from .models import (
    Assignment,
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    SatelliteState,
    SatelliteView,
    SnapshotContext,
    SchedulerConfig,
    Task,
    TaskConfig,
    TaskRecord,
)
from .route_cost import compute_cycles, estimate_route_cost
from .runtime import EnvironmentRuntime, RunningTask, SatelliteRuntime, TaskEventSink
from .scheduler import Scheduler
from .workload import generate_step_tasks, validate_task_config


StepSink = Callable[[list[SatelliteState], SnapshotContext], None]


def validate_compute_config(compute: ComputeConfig) -> None:
    if compute.cycles_per_input_bit <= 0:
        raise ValueError("compute cycles per input bit must be positive")
    if compute.cpu_frequency_hz <= 0:
        raise ValueError("CPU frequency must be positive")
    if compute.cpu_power_w < 0:
        raise ValueError("CPU power must be non-negative")


def update_circular_next_sunlit_times(
    *,
    env: EnvironmentRuntime,
    planes: int,
    sats_per_plane: int,
    time_s: int,
    mean_motion: float,
) -> None:
    """Estimate the next illumination transitions from circular plane phase."""

    if sats_per_plane <= 0 or mean_motion <= 0.0:
        return

    slot_duration_s = (2.0 * math.pi / mean_motion) / sats_per_plane

    for plane in range(planes):
        base = plane * sats_per_plane
        sunlit_by_slot = [
            env.satellites[base + slot].sunlit for slot in range(sats_per_plane)
        ]

        def next_delta_with_state(
            slot: int,
            desired_sunlit: bool,
            *,
            start_delta: int = 1,
        ) -> int | None:
            for delta_slots in range(start_delta, 2 * sats_per_plane + 1):
                if (
                    sunlit_by_slot[(slot + delta_slots) % sats_per_plane]
                    == desired_sunlit
                ):
                    return delta_slots
            return None

        for slot, is_sunlit in enumerate(sunlit_by_slot):
            sat = env.satellites[base + slot]
            if is_sunlit:
                next_eclipse_delta = next_delta_with_state(slot, False)
                sat.next_eclipse_time_s = (
                    None
                    if next_eclipse_delta is None
                    else float(time_s)
                    + max(0, next_eclipse_delta - 1) * slot_duration_s
                )
                next_sunlit_delta = (
                    None
                    if next_eclipse_delta is None
                    else next_delta_with_state(
                        slot,
                        True,
                        start_delta=next_eclipse_delta + 1,
                    )
                )
                sat.next_sunlit_time_s = (
                    float(time_s)
                    if next_eclipse_delta is None
                    else (
                        None
                        if next_sunlit_delta is None
                        else float(time_s)
                        + next_sunlit_delta * slot_duration_s
                    )
                )
                continue

            sat.next_eclipse_time_s = float(time_s)
            next_delta_slots = next_delta_with_state(slot, True)

            sat.next_sunlit_time_s = (
                None
                if next_delta_slots is None
                else float(time_s) + next_delta_slots * slot_duration_s
            )


@dataclass
class SatelliteStepStats:
    generated_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    deferred_tasks: int = 0
    task_energy_j: float = 0.0
    task_compute_time_s: float = 0.0
    task_compute_energy_j: float = 0.0
    task_transmission_energy_j: float = 0.0
    harvested_j: float = 0.0
    consumed_j: float = 0.0


def assign_step_tasks(
    *,
    scheduler: Scheduler,
    tasks: list[Task],
    satellite_views: list[SatelliteView],
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    isl_graph,
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
        compute_config=compute_config,
        task_config=task_config,
        isl_config=isl_config,
        isl_graph=isl_graph,
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
    compute_config: ComputeConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    tasks: list[Task],
    assignments: list[Assignment],
    expired_tasks: list[Task] | None = None,
    scheduler_config: SchedulerConfig | None = None,
) -> tuple[list[SatelliteState], list[TaskRecord]]:
    stats_by_sat = {sat.sat_id: SatelliteStepStats() for sat in env.satellites}
    records: list[TaskRecord] = []
    task_by_id = {task.task_id: task for task in tasks}
    cpu_time_left_s = {
        sat.sat_id: step_s
        * (
            1.0
            if scheduler_config is None
            else scheduler_config.cpu_utilization_limit
        )
        for sat in env.satellites
    }
    satellite_by_id = {sat.sat_id: sat for sat in env.satellites}

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
            compute_cycles=compute_cycles(task, compute_config),
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
    # 2. deferred tasks whose deadline has expired
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

        # Scheduler defer action.
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

        # Scheduler explicit fail action.
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

        cost = estimate_route_cost(
            task=task,
            route=assignment.route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        transmission_energy_by_sat = {
            sat_id: energy_j
            for sat_id, energy_j in cost.energy_by_sat.items()
            if sat_id != assignment.target_sat
        }
        satellite_by_id[assignment.target_sat].task_queue.append(
            RunningTask(
                task=task,
                route=assignment.route,
                mode=assignment.mode,
                total_compute_time_s=cost.compute_time_s,
                remaining_compute_time_s=cost.compute_time_s,
                executed_compute_time_s=0.0,
                transmission_time_s=cost.transmission_time_s,
                transmission_energy_by_sat=transmission_energy_by_sat,
                energy_by_sat={},
                score=assignment.score,
            )
        )

    def accumulated_energy(running: RunningTask) -> tuple[float, float, float]:
        source_energy_j = running.energy_by_sat.get(running.source_sat, 0.0)
        target_energy_j = (
            0.0
            if running.target_sat == running.source_sat
            else running.energy_by_sat.get(running.target_sat, 0.0)
        )
        return source_energy_j, target_energy_j, sum(running.energy_by_sat.values())

    def fail_running_task(
        running: RunningTask,
        *,
        waiting_time_s: float,
        compute_time_s: float,
        transmission_time_s: float,
        total_time_s: float,
    ) -> None:
        source_energy_j, target_energy_j, total_energy_j = accumulated_energy(running)
        stats_by_sat[running.source_sat].failed_tasks += 1
        env.failed_tasks.append(running.task.task_id)
        env.emit_task_event(
            "task_failed",
            running.task.task_id,
            source_sat=running.source_sat,
            target_sat=running.target_sat,
            route=list(running.route.nodes),
            mode=running.mode,
            reason="deadline",
            waiting_time_s=waiting_time_s,
            executed_compute_time_s=running.executed_compute_time_s,
            remaining_compute_time_s=running.remaining_compute_time_s,
            transmission_time_s=transmission_time_s,
            total_time_s=total_time_s,
            score=running.score,
        )
        records.append(
            make_task_record(
                task=running.task,
                source_sat=running.source_sat,
                target_sat=running.target_sat,
                mode=running.mode,
                waiting_time_s=waiting_time_s,
                compute_time_s=compute_time_s,
                transmission_time_s=transmission_time_s,
                total_time_s=total_time_s,
                energy_j=source_energy_j,
                completed=False,
                failed_reason="deadline",
                source_energy_j=source_energy_j,
                target_energy_j=target_energy_j,
                total_energy_j=total_energy_j,
                status="failed",
                score=running.score,
            )
        )

    for sat in env.satellites:
        cpu_time_capacity = max(0.0, cpu_time_left_s[sat.sat_id])
        cpu_time_left = cpu_time_capacity
        while sat.task_queue:
            running = sat.task_queue[0]
            task = running.task
            waiting_time_s = env.time_s - task.created_time_s

            if remaining_deadline_s(task) <= 0.0 and running.remaining_compute_time_s > 0.0:
                fail_running_task(
                    running,
                    waiting_time_s=waiting_time_s,
                    compute_time_s=running.executed_compute_time_s,
                    transmission_time_s=0.0,
                    total_time_s=waiting_time_s,
                )
                sat.task_queue.pop(0)
                continue

            if cpu_time_left <= 1.0e-12:
                break

            elapsed_cpu_time_s = cpu_time_capacity - cpu_time_left
            target_cpu_time_s = min(running.remaining_compute_time_s, cpu_time_left)
            cpu_time_left -= target_cpu_time_s
            running.remaining_compute_time_s -= target_cpu_time_s

            transmission_energy_by_sat = dict(running.transmission_energy_by_sat)
            running.transmission_energy_by_sat.clear()
            compute_energy_j = target_cpu_time_s * compute_config.cpu_power_w

            for sat_id, energy_j in transmission_energy_by_sat.items():
                stats_by_sat[sat_id].task_transmission_energy_j += energy_j

            if compute_energy_j:
                stats_by_sat[running.target_sat].task_compute_time_s += (
                    target_cpu_time_s
                )
                stats_by_sat[running.target_sat].task_compute_energy_j += (
                    compute_energy_j
                )

            energy_by_sat = transmission_energy_by_sat
            if compute_energy_j:
                energy_by_sat[running.target_sat] = (
                    energy_by_sat.get(running.target_sat, 0.0) + compute_energy_j
                )
            for sat_id, energy_j in energy_by_sat.items():
                stats_by_sat[sat_id].task_energy_j += energy_j
                running.energy_by_sat[sat_id] = (
                    running.energy_by_sat.get(sat_id, 0.0) + energy_j
                )

            running.executed_compute_time_s += target_cpu_time_s

            if running.remaining_compute_time_s > 1.0e-12:
                break

            finish_time_s = (
                env.time_s
                + elapsed_cpu_time_s
                + target_cpu_time_s
                + running.transmission_time_s
            )
            total_time_s = finish_time_s - task.created_time_s
            if total_time_s > task.deadline_s:
                fail_running_task(
                    running,
                    waiting_time_s=waiting_time_s,
                    compute_time_s=running.executed_compute_time_s,
                    transmission_time_s=running.transmission_time_s,
                    total_time_s=total_time_s,
                )
                sat.task_queue.pop(0)
                continue

            stats_by_sat[running.source_sat].completed_tasks += 1
            env.completed_tasks.append(task.task_id)
            source_energy_j, target_energy_j, total_energy_j = accumulated_energy(running)
            env.emit_task_event(
                "task_completed",
                task.task_id,
                source_sat=running.source_sat,
                target_sat=running.target_sat,
                route=list(running.route.nodes),
                mode=running.mode,
                waiting_time_s=waiting_time_s,
                executed_compute_time_s=running.executed_compute_time_s,
                remaining_compute_time_s=0.0,
                transmission_time_s=running.transmission_time_s,
                total_time_s=total_time_s,
                energy_j={
                    "source": source_energy_j,
                    "target": target_energy_j,
                    "total": total_energy_j,
                },
                energy_by_sat={
                    str(sat_id): energy_j
                    for sat_id, energy_j in running.energy_by_sat.items()
                },
                score=running.score,
            )
            records.append(
                make_task_record(
                    task=task,
                    source_sat=running.source_sat,
                    target_sat=running.target_sat,
                    mode=running.mode,
                    waiting_time_s=waiting_time_s,
                    compute_time_s=running.executed_compute_time_s,
                    transmission_time_s=running.transmission_time_s,
                    total_time_s=total_time_s,
                    energy_j=source_energy_j,
                    completed=True,
                    failed_reason="",
                    source_energy_j=source_energy_j,
                    target_energy_j=target_energy_j,
                    total_energy_j=total_energy_j,
                    status="completed",
                    score=running.score,
                )
            )
            sat.task_queue.pop(0)
        cpu_time_left_s[sat.sat_id] = cpu_time_left

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
                task_compute_time_s=stats.task_compute_time_s,
                task_compute_energy_j=stats.task_compute_energy_j,
                task_transmission_energy_j=stats.task_transmission_energy_j,
            )
        )

    return states, records


def iter_circular_states(
    *,
    start: dt.datetime | None = None,
    sun_position_file: str | None = None,
    satellites: int,
    planes: int,
    altitude_km: float,
    inclination_deg: float,
    duration_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    scheduler: Scheduler,
    scheduler_config: SchedulerConfig,
    walker_phase: int = 0,
    raan_spread_deg: float = 360.0,
    task_event_sink: TaskEventSink | None = None,
    step_sink: StepSink | None = None,
) -> Iterable[tuple[list[SatelliteState], list[TaskRecord]]]:
    if satellites <= 0:
        raise ValueError("satellites must be positive")
    if planes <= 0 or satellites % planes != 0:
        raise ValueError("planes must be positive and divide satellites")
    if step_s <= 0:
        raise ValueError("step must be positive")
    if not 0.0 < raan_spread_deg <= 360.0:
        raise ValueError("RAAN spread must be within (0, 360]")

    validate_battery_config(battery)
    validate_compute_config(compute_config)
    validate_task_config(task_config)

    sats_per_plane = satellites // planes
    radius_km = EARTH_RADIUS_KM + altitude_km
    inclination_rad = math.radians(inclination_deg)
    raan_spread_rad = math.radians(raan_spread_deg)
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (radius_km**3))
    sun_ephemeris = (
        None
        if start is None or sun_position_file is None
        else load_sun_ephemeris(sun_position_file)
    )

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
    constellation_layout = build_constellation_layout(
        env.views(),
        isl_config,
        walker_phase=walker_phase,
    )

    for time_s in range(0, duration_s + 1, step_s):
        env.time_s = time_s
        env.time_utc = None if start is None else start + dt.timedelta(seconds=time_s)
        sun_vector = (
            None
            if env.time_utc is None or sun_ephemeris is None
            else sun_vector_from_ephemeris(sun_ephemeris, env.time_utc)
        )
        sun_unit = (1.0, 0.0, 0.0) if sun_vector is None else vector_unit(sun_vector)
        if sun_unit is None:
            raise ValueError("Sun vector norm must be positive")

        for plane in range(planes):
            raan = raan_spread_rad * plane / planes
            plane_phase = 2.0 * math.pi * walker_phase * plane / satellites

            for slot in range(sats_per_plane):
                sat_id = plane * sats_per_plane + slot
                arg = (
                    2.0 * math.pi * slot / sats_per_plane
                    + plane_phase
                    + mean_motion * time_s
                )
                pos, vel = circular_state(radius_km, inclination_rad, raan, arg)
                sunlit = is_sunlit_cylindrical_shadow(pos, sun_unit)

                env.satellites[sat_id].update_orbit(
                    pos_km=pos,
                    vel_km_s=vel,
                    sunlit=sunlit,
                )

        update_circular_next_sunlit_times(
            env=env,
            planes=planes,
            sats_per_plane=sats_per_plane,
            time_s=time_s,
            mean_motion=mean_motion,
        )

        new_tasks, expired_tasks = generate_step_tasks(env, task_config, compute_config)
        deferred_tasks, expired_deferred_tasks = pop_deferred_tasks(env)

        tasks = deferred_tasks + new_tasks
        expired_tasks = expired_tasks + expired_deferred_tasks

        satellite_views = env.views()
        assignments = assign_step_tasks(
            scheduler=scheduler,
            tasks=tasks,
            satellite_views=satellite_views,
            time_s=env.time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            task_config=task_config,
            isl_config=isl_config,
            isl_graph=build_isl_graph(
                satellite_views,
                isl_config,
                layout=constellation_layout,
            ),
            scheduler_config=scheduler_config,
        )

        states, task_records = apply_step(
            env=env,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            task_config=task_config,
            isl_config=isl_config,
            tasks=tasks,
            assignments=assignments,
            expired_tasks=expired_tasks,
            scheduler_config=scheduler_config,
        )

        if step_sink is not None:
            step_sink(states, circular_snapshot_context(sun_vector))
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
    compute_config: ComputeConfig,
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
    validate_compute_config(compute_config)
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
    constellation_layout = build_constellation_layout(env.views(), isl_config)

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

        new_tasks, expired_tasks = generate_step_tasks(env, task_config, compute_config)
        deferred_tasks, expired_deferred_tasks = pop_deferred_tasks(env)

        tasks = deferred_tasks + new_tasks
        expired_tasks = expired_tasks + expired_deferred_tasks

        satellite_views = env.views()
        assignments = assign_step_tasks(
            scheduler=scheduler,
            tasks=tasks,
            satellite_views=satellite_views,
            time_s=env.time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            task_config=task_config,
            isl_config=isl_config,
            isl_graph=build_isl_graph(
                satellite_views,
                isl_config,
                layout=constellation_layout,
            ),
            scheduler_config=scheduler_config,
        )

        states, task_records = apply_step(
            env=env,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            task_config=task_config,
            isl_config=isl_config,
            tasks=tasks,
            assignments=assignments,
            expired_tasks=expired_tasks,
            scheduler_config=scheduler_config,
        )

        if step_sink is not None:
            step_sink(states, context)
        yield states, task_records


def circular_snapshot_context(
    sun_vector: tuple[float, float, float] | None = None,
) -> SnapshotContext:
    if sun_vector is None:
        return SnapshotContext(
            projection_label="ECI x-y projection; circular orbit model uses fixed +x sun direction",
            sun_xy_unit=(1.0, 0.0),
            sun_eci_unit=(1.0, 0.0, 0.0),
        )
    return SnapshotContext(
        projection_label="ECI x-y projection; circular orbit model uses ephemeris Sun vector",
        sun_xy_unit=xy_unit(sun_vector),
        sun_eci_unit=vector_unit(sun_vector),
    )


@lru_cache(maxsize=4)
def load_sun_ephemeris(sun_position_file: str):
    try:
        from skyfield.api import load
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for ephemeris Sun positions. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return load.timescale(), load(sun_position_file)


def sun_vector_from_ephemeris(
    sun_ephemeris,
    time_utc: dt.datetime,
) -> tuple[float, float, float]:
    ts, eph = sun_ephemeris
    t = ts.from_datetime(time_utc)
    earth = eph["earth"].at(t)
    sun = eph["sun"].at(t)
    return tuple(float(x) for x in (sun.position.km - earth.position.km))


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
