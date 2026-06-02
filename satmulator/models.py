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
class TaskConfig:
    enabled: bool
    interval_s: int
    tasks_per_sat: int
    cpu_cycles: float
    input_bits: float
    output_bits: float
    deadline_s: float
    cpu_rate_cycles_s: float
    joule_per_cycle: float


@dataclass(frozen=True)
class ISLConfig:
    isl_forward_rate_bps: float
    isl_return_rate_bps: float
    isl_tx_energy_per_bit_j: float
    isl_rx_energy_per_bit_j: float


@dataclass(frozen=True)
class Task:
    task_id: int
    created_time_s: int
    source_sat: int
    cpu_cycles: float
    input_bits: float
    output_bits: float
    deadline_s: float


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    created_time_s: int
    source_sat: int
    target_sat: int
    mode: str
    cpu_cycles: float
    input_bits: float
    output_bits: float
    deadline_s: float
    compute_time_s: float
    transmission_time_s: float
    total_time_s: float
    energy_j: float
    completed: bool
    failed_reason: str
    source_energy_j: float = 0.0
    target_energy_j: float = 0.0
    total_energy_j: float = 0.0


@dataclass(frozen=True)
class Assignment:
    task_id: int
    source_sat: int
    target_sat: int
    mode: str


@dataclass(frozen=True)
class SatelliteView:
    sat_id: int
    x_km: float
    y_km: float
    z_km: float
    sunlit: bool


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


@dataclass(frozen=True)
class SnapshotContext:
    projection_label: str
    sun_xy_unit: tuple[float, float] | None = None
