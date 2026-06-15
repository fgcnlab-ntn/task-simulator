from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

from .models import DemandDistribution, DemandPoint, Task, TaskConfig
from .runtime import EnvironmentRuntime, SatelliteRuntime

T = TypeVar("T")


def demand_distribution(points: Iterable[DemandPoint]) -> DemandDistribution:
    loaded = tuple(points)
    cumulative: list[float] = []
    total = 0.0
    for point in loaded:
        if not math.isfinite(point.weight) or point.weight <= 0.0:
            raise ValueError("demand distribution weights must be finite and positive")
        total += point.weight
        cumulative.append(total)
    return DemandDistribution(loaded, tuple(cumulative), total)


def load_demand_points(path: Path | None) -> DemandDistribution:
    if path is None:
        return demand_distribution(())
    points: list[DemandPoint] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"lat", "lon", "weight"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("demand points CSV must contain lat, lon, weight columns")
        for row in reader:
            lat = float(row["lat"])
            lon = float(row["lon"])
            weight = float(row["weight"])
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"invalid demand latitude: {lat}")
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"invalid demand longitude: {lon}")
            if weight < 0.0:
                raise ValueError(f"invalid demand weight: {weight}")
            if weight > 0.0:
                points.append(DemandPoint(lat_deg=lat, lon_deg=lon, weight=weight))
    if not points:
        raise ValueError(f"no positive demand points in {path}")
    return demand_distribution(points)


def demand_points_provenance(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    distribution = load_demand_points(path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    conversion: dict[str, object] | None = None
    if metadata_path.exists():
        value = json.loads(metadata_path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"demand-point metadata must be an object: {metadata_path}")
        conversion = value
    return {
        "file": str(path),
        "points": len(distribution.points),
        "total_weight": distribution.total_weight,
        "metadata_file": str(metadata_path) if metadata_path.exists() else None,
        "conversion": conversion,
    }


def validate_distribution(name: str, choices: Sequence[object], weights: Sequence[float]) -> None:
    if not choices:
        raise ValueError(f"{name} choices must not be empty")
    if len(choices) != len(weights):
        raise ValueError(f"{name} choices and weights must have the same length")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"{name} weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError(f"{name} weights must contain a positive value")


def validate_task_config(task_config: TaskConfig) -> None:
    if task_config.interval_s <= 0:
        raise ValueError("task interval must be positive")
    if task_config.tasks_per_sat < 0:
        raise ValueError("tasks per satellite must be non-negative")
    if task_config.cpu_cycles <= 0:
        raise ValueError("task CPU cycles must be positive")
    if task_config.input_bits < 0 or task_config.output_bits < 0:
        raise ValueError("task input/output bits must be non-negative")
    if task_config.deadline_s <= 0:
        raise ValueError("task deadline must be positive")
    if task_config.cpu_rate_cycles_s <= 0:
        raise ValueError("CPU rate must be positive")
    if task_config.joule_per_cycle < 0:
        raise ValueError("joule per cycle must be non-negative")
    if not 0.0 <= task_config.min_elevation_deg <= 90.0:
        raise ValueError("minimum elevation must be within [0, 90]")
    if task_config.generation_mode not in {"satellite-deterministic", "demand-points"}:
        raise ValueError(f"unknown task generation mode: {task_config.generation_mode}")
    validate_distribution(
        "tasks_per_step", task_config.tasks_per_step_choices, task_config.tasks_per_step_weights
    )
    validate_distribution("cpu_cycles", task_config.cpu_cycles_choices, task_config.cpu_cycles_weights)
    validate_distribution("input_bits", task_config.input_bits_choices, task_config.input_bits_weights)
    validate_distribution("output_bits", task_config.output_bits_choices, task_config.output_bits_weights)
    if task_config.generation_mode == "demand-points" and not task_config.demand_distribution:
        raise ValueError("demand-points task generation requires a demand_points_file")


def weighted_choice(rng: random.Random, choices: Sequence[T], weights: Sequence[float]) -> T:
    return rng.choices(list(choices), weights=list(weights), k=1)[0]


def choose_demand_point(
    rng: random.Random,
    distribution: DemandDistribution,
) -> DemandPoint:
    return rng.choices(
        distribution.points,
        cum_weights=distribution.cumulative_weights,
        k=1,
    )[0]


@lru_cache(maxsize=1)
def skyfield_timescale():
    try:
        from skyfield.api import load
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for demand-point coordinates. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return load.timescale()


@lru_cache(maxsize=1)
def skyfield_visibility_modules():
    try:
        from skyfield.api import wgs84
        from skyfield.constants import AU_KM
        from skyfield.positionlib import Geocentric
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for demand-point visibility. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return wgs84, AU_KM, Geocentric


def ground_position_km(point: DemandPoint, time_utc: dt.datetime) -> tuple[float, float, float]:
    wgs84, _, _ = skyfield_visibility_modules()
    t = skyfield_timescale().from_datetime(time_utc)
    position = wgs84.latlon(point.lat_deg, point.lon_deg).at(t).position.km
    return tuple(float(component) for component in position)


def satellite_altitude_distance(
    sat: SatelliteRuntime,
    point: DemandPoint,
    time_utc: dt.datetime,
) -> tuple[float, float]:
    wgs84, au_km, geocentric = skyfield_visibility_modules()
    t = skyfield_timescale().from_datetime(time_utc)
    ground = wgs84.latlon(point.lat_deg, point.lon_deg).at(t)
    return satellite_altitude_distance_from_ground(sat, t, ground, au_km, geocentric)


def satellite_altitude_distance_from_ground(
    sat: SatelliteRuntime,
    t,
    ground,
    au_km: float,
    geocentric,
) -> tuple[float, float]:
    satellite = geocentric(
        [component / au_km for component in sat.pos_km],
        t=t,
    )
    altitude, _, distance = (satellite - ground).altaz()
    return float(altitude.degrees), float(distance.km)


def nearest_satellite_id(
    satellites: Iterable[SatelliteRuntime],
    point: DemandPoint,
    time_utc: dt.datetime,
    min_elevation_deg: float = 30.0,
) -> int | None:
    best_sat_id = -1
    best_distance_km = float("inf")
    wgs84, au_km, geocentric = skyfield_visibility_modules()
    t = skyfield_timescale().from_datetime(time_utc)
    ground = wgs84.latlon(point.lat_deg, point.lon_deg).at(t)
    for sat in satellites:
        altitude_deg, distance_km = satellite_altitude_distance_from_ground(
            sat, t, ground, au_km, geocentric
        )
        if altitude_deg >= min_elevation_deg and distance_km < best_distance_km:
            best_sat_id = sat.sat_id
            best_distance_km = distance_km
    return None if best_sat_id < 0 else best_sat_id


def generate_step_tasks(
    env: EnvironmentRuntime,
    task_config: TaskConfig,
) -> tuple[list[Task], list[Task]]:
    if not task_config.enabled:
        return [], []
    if task_config.generation_mode == "satellite-deterministic":
        if env.time_s <= 0 or env.time_s % task_config.interval_s != 0:
            return [], []
        return generate_satellite_deterministic_tasks(env, task_config), []
    if task_config.generation_mode == "demand-points":
        if env.time_s > 0 and env.time_s % task_config.interval_s == 0:
            env.pending_tasks.extend(generate_demand_point_tasks(env, task_config))
        return resolve_pending_tasks(env, task_config)
    raise ValueError(f"unknown task generation mode: {task_config.generation_mode}")


def generate_satellite_deterministic_tasks(env: EnvironmentRuntime, task_config: TaskConfig) -> list[Task]:
    tasks: list[Task] = []
    for sat in env.satellites:
        for _ in range(task_config.tasks_per_sat):
            task = Task(
                task_id=env.next_task_id,
                created_time_s=env.time_s,
                source_sat=sat.sat_id,
                cpu_cycles=task_config.cpu_cycles,
                input_bits=task_config.input_bits,
                output_bits=task_config.output_bits,
                deadline_s=task_config.deadline_s,
            )
            tasks.append(task)
            emit_generated_task(env, task)
            env.next_task_id += 1
    return tasks


def generate_demand_point_tasks(env: EnvironmentRuntime, task_config: TaskConfig) -> list[Task]:
    if env.time_utc is None:
        raise ValueError("demand-point task generation requires an absolute UTC simulation time")
    task_count = weighted_choice(
        env.rng, task_config.tasks_per_step_choices, task_config.tasks_per_step_weights
    )
    tasks: list[Task] = []
    for _ in range(task_count):
        point = choose_demand_point(env.rng, task_config.demand_distribution)
        task = Task(
            task_id=env.next_task_id,
            created_time_s=env.time_s,
            source_sat=None,
            cpu_cycles=weighted_choice(
                env.rng, task_config.cpu_cycles_choices, task_config.cpu_cycles_weights
            ),
            input_bits=weighted_choice(
                env.rng, task_config.input_bits_choices, task_config.input_bits_weights
            ),
            output_bits=weighted_choice(
                env.rng, task_config.output_bits_choices, task_config.output_bits_weights
            ),
            deadline_s=task_config.deadline_s,
            lat_deg=point.lat_deg,
            lon_deg=point.lon_deg,
        )
        tasks.append(task)
        emit_generated_task(env, task)
        env.next_task_id += 1
    return tasks


def emit_generated_task(env: EnvironmentRuntime, task: Task) -> None:
    env.emit_task_event(
        "task_generated",
        task.task_id,
        source_sat=task.source_sat,
        location=(
            None
            if task.lat_deg is None
            else {"lat_deg": task.lat_deg, "lon_deg": task.lon_deg}
        ),
        workload={
            "cpu_cycles": task.cpu_cycles,
            "input_bits": task.input_bits,
            "output_bits": task.output_bits,
        },
        deadline_s=task.deadline_s,
    )


def resolve_pending_tasks(
    env: EnvironmentRuntime,
    task_config: TaskConfig,
) -> tuple[list[Task], list[Task]]:
    if env.time_utc is None:
        raise ValueError("demand-point task generation requires an absolute UTC simulation time")
    ready: list[Task] = []
    expired: list[Task] = []
    still_pending: list[Task] = []
    for task in env.pending_tasks:
        if task.lat_deg is None or task.lon_deg is None:
            raise ValueError("pending demand-point task requires latitude and longitude")
        point = DemandPoint(lat_deg=task.lat_deg, lon_deg=task.lon_deg, weight=1.0)
        source_sat = nearest_satellite_id(
            env.satellites,
            point,
            env.time_utc,
            task_config.min_elevation_deg,
        )
        if source_sat is not None:
            env.emit_task_event(
                "task_coverage_acquired",
                task.task_id,
                source_sat=source_sat,
                waiting_time_s=env.time_s - task.created_time_s,
            )
            ready.append(
                Task(
                    task_id=task.task_id,
                    created_time_s=task.created_time_s,
                    source_sat=source_sat,
                    cpu_cycles=task.cpu_cycles,
                    input_bits=task.input_bits,
                    output_bits=task.output_bits,
                    deadline_s=task.deadline_s,
                    lat_deg=task.lat_deg,
                    lon_deg=task.lon_deg,
                )
            )
        elif env.time_s - task.created_time_s >= task.deadline_s:
            expired.append(task)
        else:
            env.emit_task_event(
                "task_waiting_for_coverage",
                task.task_id,
                waiting_time_s=env.time_s - task.created_time_s,
            )
            still_pending.append(task)
    env.pending_tasks = still_pending
    return ready, expired
