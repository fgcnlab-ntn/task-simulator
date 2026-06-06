from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

from .constants import EARTH_RADIUS_KM
from .models import DemandPoint, Task, TaskConfig
from .runtime import EnvironmentRuntime, SatelliteRuntime

T = TypeVar("T")


def load_demand_points(path: Path | None) -> tuple[DemandPoint, ...]:
    if path is None:
        return ()
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
    return tuple(points)


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
    if task_config.generation_mode not in {"satellite-deterministic", "demand-points"}:
        raise ValueError(f"unknown task generation mode: {task_config.generation_mode}")
    validate_distribution(
        "tasks_per_step", task_config.tasks_per_step_choices, task_config.tasks_per_step_weights
    )
    validate_distribution("cpu_cycles", task_config.cpu_cycles_choices, task_config.cpu_cycles_weights)
    validate_distribution("input_bits", task_config.input_bits_choices, task_config.input_bits_weights)
    validate_distribution("output_bits", task_config.output_bits_choices, task_config.output_bits_weights)
    if task_config.generation_mode == "demand-points" and not task_config.demand_points:
        raise ValueError("demand-points task generation requires a demand_points_file")


def weighted_choice(rng: random.Random, choices: Sequence[T], weights: Sequence[float]) -> T:
    return rng.choices(list(choices), weights=list(weights), k=1)[0]

# Apply library, using ECI
def ground_xyz(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    cos_lat = math.cos(lat)
    return (
        EARTH_RADIUS_KM * cos_lat * math.cos(lon),
        EARTH_RADIUS_KM * cos_lat * math.sin(lon),
        EARTH_RADIUS_KM * math.sin(lat),
    )


def nearest_satellite_id(satellites: Iterable[SatelliteRuntime], point: DemandPoint) -> int:
    gx, gy, gz = ground_xyz(point.lat_deg, point.lon_deg)
    best_sat_id = -1
    best_dist2 = float("inf")
    for sat in satellites:
        dx = sat.pos_km[0] - gx
        dy = sat.pos_km[1] - gy
        dz = sat.pos_km[2] - gz
        dist2 = dx * dx + dy * dy + dz * dz
        if dist2 < best_dist2:
            best_sat_id = sat.sat_id
            best_dist2 = dist2
    if best_sat_id < 0:
        raise ValueError("cannot assign demand task without satellites")
    return best_sat_id


def generate_step_tasks(env: EnvironmentRuntime, task_config: TaskConfig) -> list[Task]:
    if not task_config.enabled:
        return []
    if env.time_s <= 0 or env.time_s % task_config.interval_s != 0:
        return []
    if task_config.generation_mode == "satellite-deterministic":
        return generate_satellite_deterministic_tasks(env, task_config)
    if task_config.generation_mode == "demand-points":
        return generate_demand_point_tasks(env, task_config)
    raise ValueError(f"unknown task generation mode: {task_config.generation_mode}")


def generate_satellite_deterministic_tasks(env: EnvironmentRuntime, task_config: TaskConfig) -> list[Task]:
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


def generate_demand_point_tasks(env: EnvironmentRuntime, task_config: TaskConfig) -> list[Task]:
    task_count = weighted_choice(
        env.rng, task_config.tasks_per_step_choices, task_config.tasks_per_step_weights
    )
    demand_weights = tuple(point.weight for point in task_config.demand_points)
    tasks: list[Task] = []
    for _ in range(task_count):
        point = weighted_choice(env.rng, task_config.demand_points, demand_weights)
        tasks.append(
            Task(
                task_id=env.next_task_id,
                created_time_s=env.time_s,
                source_sat=nearest_satellite_id(env.satellites, point),
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
        )
        env.next_task_id += 1
    return tasks
