from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    capacity_j: float
    initial_j: float
    min_safe_j: float
    harvest_w: float
    idle_w: float


@dataclass(frozen=True)
class DemandPoint:
    lat_deg: float
    lon_deg: float
    weight: float


@dataclass(frozen=True)
class DemandDistribution:
    points: tuple[DemandPoint, ...]
    cumulative_weights: tuple[float, ...]
    total_weight: float

    def __bool__(self) -> bool:
        return bool(self.points)


@dataclass(frozen=True)
class TaskConfig:
    enabled: bool
    interval_s: int
    generation_mode: str
    random_seed: int | None
    tasks_per_sat: int
    tasks_per_step_choices: tuple[int, ...]
    tasks_per_step_weights: tuple[float, ...]
    cpu_cycles: float
    cpu_cycles_choices: tuple[float, ...]
    cpu_cycles_weights: tuple[float, ...]
    input_bits: float
    input_bits_choices: tuple[float, ...]
    input_bits_weights: tuple[float, ...]
    output_bits: float
    output_bits_choices: tuple[float, ...]
    output_bits_weights: tuple[float, ...]
    deadline_s: float
    cpu_rate_cycles_s: float
    joule_per_cycle: float
    demand_distribution: DemandDistribution
    min_elevation_deg: float


@dataclass(frozen=True)
class ISLConfig:
    isl_forward_rate_bps: float
    isl_return_rate_bps: float
    isl_tx_energy_per_bit_j: float
    isl_rx_energy_per_bit_j: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    max_tasks_per_sat_per_slot: int = 4
    defer_penalty: float = 3.0
    fail_penalty: float = 1000.0
    time_weight: float = 1.0
    energy_weight: float = 2.0
    battery_weight: float = 5.0
    load_weight: float = 0.1
    eclipse_local_penalty: float = 2.0
    low_battery_threshold_pct: float = 35.0


@dataclass(frozen=True)
class Task:
    task_id: int
    created_time_s: int
    source_sat: int | None
    cpu_cycles: float
    input_bits: float
    output_bits: float
    deadline_s: float
    lat_deg: float | None = None
    lon_deg: float | None = None


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    created_time_s: int
    source_sat: int
    target_sat: int
    mode: str
    lat_deg: float | None
    lon_deg: float | None
    cpu_cycles: float
    input_bits: float
    output_bits: float
    deadline_s: float
    waiting_time_s: float
    compute_time_s: float
    transmission_time_s: float
    total_time_s: float
    energy_j: float
    completed: bool
    failed_reason: str
    source_energy_j: float = 0.0
    target_energy_j: float = 0.0
    total_energy_j: float = 0.0
    status: str = "completed"
    remaining_deadline_s: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class Assignment:
    task_id: int
    source_sat: int
    target_sat: int
    mode: str
    score: float = 0.0
    failed_reason: str = ""


@dataclass(frozen=True)
class SatelliteView:
    sat_id: int
    x_km: float
    y_km: float
    z_km: float
    sunlit: bool
    battery_j: float = 0.0
    load: float = 0.0


@dataclass(frozen=True)
class SatelliteState:
    time_s: int
    sat_id: int
    name: str
    plane: int
    slot: int
    x_km: float
    y_km: float
    z_km: float
    vx_km_s: float
    vy_km_s: float
    vz_km_s: float
    lat_deg: float | None
    lon_deg: float | None
    elevation_km: float | None
    sunlit: bool
    battery_j: float
    battery_pct: float
    harvested_j: float
    consumed_j: float
    safe_battery: bool
    generated_tasks: int
    completed_tasks: int
    failed_tasks: int
    task_energy_j: float
    deferred_tasks: int = 0


@dataclass(frozen=True)
class SnapshotContext:
    projection_label: str
    sun_xy_unit: tuple[float, float] | None = None
    sun_eci_unit: tuple[float, float, float] | None = None
