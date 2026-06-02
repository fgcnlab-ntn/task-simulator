from __future__ import annotations

from .models import Assignment, SatelliteView, Task


def distance_km(a: SatelliteView, b: SatelliteView) -> float:
    dx = a.x_km - b.x_km
    dy = a.y_km - b.y_km
    dz = a.z_km - b.z_km
    return (dx * dx + dy * dy + dz * dz) ** 0.5


class Scheduler:
    name = "base"

    def assign_task(self, *, task: Task, satellite_views: list[SatelliteView]) -> Assignment:
        raise NotImplementedError


class LocalOnlyScheduler(Scheduler):
    name = "local"

    def assign_task(self, *, task: Task, satellite_views: list[SatelliteView]) -> Assignment:
        return Assignment(
            task_id=task.task_id,
            source_sat=task.source_sat,
            target_sat=task.source_sat,
            mode=self.name,
        )


class NearestSunlitScheduler(Scheduler):
    name = "nearest-sunlit"

    def assign_task(self, *, task: Task, satellite_views: list[SatelliteView]) -> Assignment:
        by_id = {sat.sat_id: sat for sat in satellite_views}
        source = by_id[task.source_sat]
        sunlit_targets = [sat for sat in satellite_views if sat.sunlit]
        target = source
        mode = "local"
        if not source.sunlit and sunlit_targets:
            target = min(sunlit_targets, key=lambda sat: distance_km(source, sat))
            mode = "offload"
        return Assignment(
            task_id=task.task_id,
            source_sat=task.source_sat,
            target_sat=target.sat_id,
            mode=mode,
        )


def create_scheduler(name: str) -> Scheduler:
    if name == LocalOnlyScheduler.name:
        return LocalOnlyScheduler()
    if name == NearestSunlitScheduler.name:
        return NearestSunlitScheduler()
    raise ValueError(f"unknown scheduler: {name}")
