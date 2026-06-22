from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Callable

from .models import BatteryConfig, SatelliteState, SatelliteView, Task


Vector3 = tuple[float, float, float]
TaskEventSink = Callable[[dict[str, object]], None]


@dataclass
class SatelliteRuntime:
    """Mutable per-satellite simulation state.

    This is the owner of state that changes during the simulation.  Snapshot
    dataclasses are for output only; policies and energy accounting should not
    pass around parallel arrays for battery, load, and task ownership.
    """

    sat_id: int
    name: str
    plane: int
    slot: int
    battery_j: float
    load: float = 0.0
    running_tasks: list[int] = field(default_factory=list)
    pos_km: Vector3 = (0.0, 0.0, 0.0)
    vel_km_s: Vector3 = (0.0, 0.0, 0.0)
    lat_deg: float | None = None
    lon_deg: float | None = None
    elevation_km: float | None = None
    sunlit: bool = False

    def update_orbit(
        self,
        *,
        pos_km: Vector3,
        vel_km_s: Vector3,
        sunlit: bool,
        lat_deg: float | None = None,
        lon_deg: float | None = None,
        elevation_km: float | None = None,
    ) -> None:
        self.pos_km = pos_km
        self.vel_km_s = vel_km_s
        self.sunlit = sunlit
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.elevation_km = elevation_km

    def view(self) -> SatelliteView:
        return SatelliteView(
            sat_id=self.sat_id,
            x_km=self.pos_km[0],
            y_km=self.pos_km[1],
            z_km=self.pos_km[2],
            sunlit=self.sunlit,
            battery_j=self.battery_j,
            load=self.load,
            plane=self.plane,
            slot=self.slot,
        )

    def snapshot(
        self,
        *,
        time_s: int,
        harvested_j: float,
        consumed_j: float,
        battery: BatteryConfig,
        generated_tasks: int,
        completed_tasks: int,
        failed_tasks: int,
        deferred_tasks: int,
        task_energy_j: float,
    ) -> SatelliteState:
        return SatelliteState(
            time_s=time_s,
            sat_id=self.sat_id,
            name=self.name,
            plane=self.plane,
            slot=self.slot,
            x_km=self.pos_km[0],
            y_km=self.pos_km[1],
            z_km=self.pos_km[2],
            vx_km_s=self.vel_km_s[0],
            vy_km_s=self.vel_km_s[1],
            vz_km_s=self.vel_km_s[2],
            lat_deg=self.lat_deg,
            lon_deg=self.lon_deg,
            elevation_km=self.elevation_km,
            sunlit=self.sunlit,
            battery_j=self.battery_j,
            battery_pct=100.0 * self.battery_j / battery.capacity_j,
            harvested_j=harvested_j,
            consumed_j=consumed_j,
            safe_battery=self.battery_j >= battery.min_safe_j,
            generated_tasks=generated_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            deferred_tasks=deferred_tasks,
            task_energy_j=task_energy_j,
        )


@dataclass
class EnvironmentRuntime:
    """Mutable simulation state shared by orbit, scheduler, and accounting."""

    satellites: list[SatelliteRuntime]
    rng: random.Random = field(default_factory=random.Random)
    time_s: int = 0
    time_utc: dt.datetime | None = None
    next_task_id: int = 0
    pending_tasks: list[Task] = field(default_factory=list)
    completed_tasks: list[int] = field(default_factory=list)
    deferred_tasks: list[Task] = field(default_factory=list)
    failed_tasks: list[int] = field(default_factory=list)
    task_event_sink: TaskEventSink | None = None

    def views(self) -> list[SatelliteView]:
        return [sat.view() for sat in self.satellites]

    def emit_task_event(self, event_type: str, task_id: int, **details: object) -> None:
        if self.task_event_sink is None:
            return
        self.task_event_sink(
            {
                "type": event_type,
                "time_s": self.time_s,
                "task_id": task_id,
                **details,
            }
        )
