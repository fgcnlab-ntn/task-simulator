from __future__ import annotations

from .models import BatteryConfig


def validate_battery_config(battery: BatteryConfig) -> None:
    if battery.capacity_j <= 0:
        raise ValueError("battery capacity must be positive")
    if battery.harvest_w < 0 or battery.idle_w < 0:
        raise ValueError("battery power values must be non-negative")
    if not 0 <= battery.initial_j <= battery.capacity_j:
        raise ValueError("initial battery must be within [0, capacity]")
    if not 0 <= battery.min_safe_j <= battery.capacity_j:
        raise ValueError("minimum safe battery must be within [0, capacity]")


def apply_battery_step(
    *,
    battery_now: float,
    sunlit: bool,
    step_s: int,
    battery: BatteryConfig,
    task_energy_j: float,
    update: bool,
) -> tuple[float, float, float]:
    consumed_j = battery.idle_w * step_s
    harvested_j = battery.harvest_w * step_s if sunlit else 0.0
    if not update:
        return battery_now, 0.0, 0.0
    battery_now = max(
        0.0,
        projected_battery_after_step(
            battery_now=battery_now,
            sunlit=sunlit,
            step_s=step_s,
            battery=battery,
            task_energy_j=task_energy_j,
            update=update,
        ),
    )
    return battery_now, harvested_j, consumed_j


def projected_battery_after_step(
    *,
    battery_now: float,
    sunlit: bool,
    step_s: int,
    battery: BatteryConfig,
    task_energy_j: float,
    update: bool,
) -> float:
    if not update:
        return battery_now
    consumed_j = battery.idle_w * step_s
    harvested_j = battery.harvest_w * step_s if sunlit else 0.0
    return min(battery.capacity_j, battery_now - consumed_j - task_energy_j + harvested_j)
