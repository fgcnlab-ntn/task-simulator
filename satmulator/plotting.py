from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .models import SatelliteState, SnapshotContext, TaskRecord
from .output import (
    write_battery_svg,
    write_battery_timeline_svg,
    write_offload_target_histogram_svg,
    write_snapshot_svg,
    write_summary_svg,
    write_sunlight_timeline_svg,
    write_task_mode_summary_svg,
    write_task_svg,
)
from .runlog import JsonObject, iter_state_steps, iter_task_events, load_run
from .task_projection import TaskLifecycle, project_task_lifecycles


def render_run_plots(output_dir: Path) -> None:
    manifest = load_run(output_dir)
    all_steps, contexts = load_state_steps(manifest, iter_state_steps(output_dir))
    task_records = [
        task_record_from_lifecycle(lifecycle)
        for lifecycle in project_task_lifecycles(iter_task_events(output_dir))
        if lifecycle.status in {"completed", "failed", "deferred"}
    ]
    task_records_by_step: dict[int, list[TaskRecord]] = defaultdict(list)
    for record in task_records:
        task_records_by_step[_record_time_s(record)].append(record)

    records_at_steps = [
        task_records_by_step.get(states[0].time_s, [])
        for states in all_steps
    ]
    write_snapshot_svg(
        output_dir / "snapshot_start.svg",
        all_steps[0],
        "Orbit snapshot at t=0s",
        contexts[0],
    )
    write_snapshot_svg(
        output_dir / "snapshot_end.svg",
        all_steps[-1],
        f"Orbit snapshot at t={all_steps[-1][0].time_s}s",
        contexts[-1],
    )
    write_summary_svg(output_dir / "sunlight_summary.svg", all_steps)
    write_battery_svg(output_dir / "battery_summary.svg", all_steps)
    write_task_svg(output_dir / "task_summary.svg", all_steps, records_at_steps)
    write_sunlight_timeline_svg(output_dir / "sunlight_timeline.svg", all_steps)
    write_battery_timeline_svg(output_dir / "battery_timeline.svg", all_steps)
    write_task_mode_summary_svg(output_dir / "task_mode_summary.svg", task_records)
    write_offload_target_histogram_svg(
        output_dir / "offload_target_histogram.svg", task_records
    )


def load_state_steps(
    manifest: JsonObject,
    records: Iterable[JsonObject],
) -> tuple[list[list[SatelliteState]], list[SnapshotContext]]:
    config = _object(manifest.get("config"), "run.json config")
    battery = _object(config.get("battery"), "run.json battery config")
    capacity_j = _number(battery.get("capacity_j"), "battery capacity_j")
    min_safe_pct = _number(battery.get("min_safe_pct"), "battery min_safe_pct")
    catalog = {
        _integer(satellite.get("id"), "satellite catalog id"): satellite
        for satellite in _object_list(manifest.get("satellites"), "satellite catalog")
    }

    all_steps: list[list[SatelliteState]] = []
    contexts: list[SnapshotContext] = []
    for record in records:
        record = _object(record, "state record")
        time_s = _integer(record.get("time_s"), "state time_s")
        states = [
            satellite_state_from_record(
                time_s=time_s,
                record=_object(satellite, "satellite state"),
                catalog=catalog,
                capacity_j=capacity_j,
                min_safe_pct=min_safe_pct,
            )
            for satellite in _object_list(record.get("satellites"), "state satellites")
        ]
        if not states:
            raise ValueError(f"state record at time {time_s} has no satellites")
        all_steps.append(states)
        contexts.append(snapshot_context_from_record(record))

    if not all_steps:
        raise ValueError("states.jsonl contains no state records")
    return all_steps, contexts


def satellite_state_from_record(
    *,
    time_s: int,
    record: JsonObject,
    catalog: dict[int, JsonObject],
    capacity_j: float,
    min_safe_pct: float,
) -> SatelliteState:
    sat_id = _integer(record.get("id"), "satellite id")
    metadata = catalog.get(sat_id)
    if metadata is None:
        raise ValueError(f"satellite {sat_id} is missing from run.json catalog")
    position = _vector(record.get("position_km"), "position_km", 3)
    velocity = _vector(record.get("velocity_km_s"), "velocity_km_s", 3)
    battery_j = _number(record.get("battery_j"), "battery_j")
    energy = _object(record.get("energy_delta_j"), "energy_delta_j")
    task_load_value = record.get("task_load")
    task_load = (
        {}
        if task_load_value is None
        else _object(task_load_value, "task_load")
    )
    counts = _object(record.get("task_counts"), "task_counts")
    geodetic_value = record.get("geodetic")
    geodetic = None if geodetic_value is None else _object(geodetic_value, "geodetic")
    battery_pct = 100.0 * battery_j / capacity_j

    return SatelliteState(
        time_s=time_s,
        sat_id=sat_id,
        name=_string(metadata.get("name"), "satellite name"),
        plane=_integer(metadata.get("plane"), "satellite plane"),
        slot=_integer(metadata.get("slot"), "satellite slot"),
        x_km=position[0],
        y_km=position[1],
        z_km=position[2],
        vx_km_s=velocity[0],
        vy_km_s=velocity[1],
        vz_km_s=velocity[2],
        lat_deg=None if geodetic is None else _number(geodetic.get("lat_deg"), "lat_deg"),
        lon_deg=None if geodetic is None else _number(geodetic.get("lon_deg"), "lon_deg"),
        elevation_km=(
            None if geodetic is None else _number(geodetic.get("elevation_km"), "elevation_km")
        ),
        sunlit=_boolean(record.get("sunlit"), "sunlit"),
        battery_j=battery_j,
        battery_pct=battery_pct,
        harvested_j=_number(energy.get("harvested"), "energy harvested"),
        consumed_j=_number(energy.get("consumed"), "energy consumed"),
        safe_battery=battery_pct >= min_safe_pct,
        generated_tasks=_integer(counts.get("generated"), "generated tasks"),
        completed_tasks=_integer(counts.get("completed"), "completed tasks"),
        failed_tasks=_integer(counts.get("failed"), "failed tasks"),
        task_energy_j=_number(energy.get("tasks"), "task energy"),
        task_compute_time_s=_number(
            task_load.get("compute_time_s", 0.0),
            "task_load compute_time_s",
        ),
        task_compute_energy_j=_number(
            task_load.get("compute_energy_j", 0.0),
            "task_load compute_energy_j",
        ),
        task_transmission_energy_j=_number(
            task_load.get("transmission_energy_j", 0.0),
            "task_load transmission_energy_j",
        ),
        deferred_tasks=_integer(counts.get("deferred", 0), "deferred tasks"),
    )


def snapshot_context_from_record(record: JsonObject) -> SnapshotContext:
    value = record.get("snapshot_context")
    if value is None:
        return SnapshotContext("ECI x-y projection; Sun direction was not recorded")
    context = _object(value, "snapshot_context")
    xy = context.get("sun_xy_unit")
    eci = context.get("sun_eci_unit")
    return SnapshotContext(
        projection_label=_string(context.get("projection_label"), "projection_label"),
        sun_xy_unit=None if xy is None else _vector(xy, "sun_xy_unit", 2),
        sun_eci_unit=None if eci is None else _vector(eci, "sun_eci_unit", 3),
    )


def task_record_from_lifecycle(lifecycle: TaskLifecycle) -> TaskRecord:
    generated = lifecycle.events[0]
    terminal = lifecycle.terminal_event or lifecycle.events[-1]
    workload = _object(generated.get("workload", {}), "task workload")
    location_value = generated.get("location")
    location = None if location_value is None else _object(location_value, "task location")
    source_sat = -1 if lifecycle.source_sat is None else lifecycle.source_sat
    target_sat = source_sat if lifecycle.target_sat is None else lifecycle.target_sat
    energy_value = terminal.get("energy_j")
    energy = {} if energy_value is None else _object(energy_value, "task energy")
    waiting_time_s = _number(terminal.get("waiting_time_s", 0.0), "waiting_time_s")
    compute_time_s = _number(terminal.get("compute_time_s", 0.0), "compute_time_s")
    transmission_time_s = _number(
        terminal.get("transmission_time_s", 0.0), "transmission_time_s"
    )
    total_time_s = _number(
        terminal.get("total_time_s", waiting_time_s), "total_time_s"
    )
    return TaskRecord(
        task_id=lifecycle.task_id,
        created_time_s=lifecycle.created_time_s,
        source_sat=source_sat,
        target_sat=target_sat,
        mode=lifecycle.mode or "unassigned",
        lat_deg=None if location is None else _number(location.get("lat_deg"), "task lat_deg"),
        lon_deg=None if location is None else _number(location.get("lon_deg"), "task lon_deg"),
        compute_cycles=_number(workload.get("compute_cycles", 0.0), "compute_cycles"),
        input_bits=_number(workload.get("input_bits", 0.0), "input_bits"),
        output_bits=_number(workload.get("output_bits", 0.0), "output_bits"),
        deadline_s=_number(generated.get("deadline_s", 0.0), "deadline_s"),
        waiting_time_s=waiting_time_s,
        compute_time_s=compute_time_s,
        transmission_time_s=transmission_time_s,
        total_time_s=total_time_s,
        energy_j=_number(energy.get("source", 0.0), "source energy"),
        completed=lifecycle.completed,
        failed_reason=_string(terminal.get("reason", ""), "failure reason"),
        source_energy_j=_number(energy.get("source", 0.0), "source energy"),
        target_energy_j=_number(energy.get("target", 0.0), "target energy"),
        total_energy_j=_number(energy.get("total", 0.0), "total energy"),
        status=lifecycle.status,
        score=_number(terminal.get("score", 0.0), "score"),
    )


def _record_time_s(record: TaskRecord) -> int:
    return int(record.created_time_s + record.waiting_time_s)


def _object(value: object, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _object_list(value: object, name: str) -> list[JsonObject]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _vector(value: object, name: str, length: int):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return tuple(_number(component, name) for component in value)


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
