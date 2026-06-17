from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .constants import EARTH_RADIUS_KM
from .models import ISLConfig, Route, SatelliteView


@dataclass(frozen=True)
class ISLGraph:
    adjacency: dict[int, tuple[int, ...]]

    def neighbors(self, sat_id: int) -> tuple[int, ...]:
        return self.adjacency.get(sat_id, ())


def fully_connected_isl_graph(satellites: Iterable[SatelliteView]) -> ISLGraph:
    sat_ids = sorted(sat.sat_id for sat in satellites)
    return ISLGraph(
        {
            sat_id: tuple(other for other in sat_ids if other != sat_id)
            for sat_id in sat_ids
        }
    )


def distance_km(a: SatelliteView, b: SatelliteView) -> float:
    dx = a.x_km - b.x_km
    dy = a.y_km - b.y_km
    dz = a.z_km - b.z_km
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def has_line_of_sight(
    a: SatelliteView,
    b: SatelliteView,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> bool:
    """Return true when the segment between satellites clears Earth."""

    ax, ay, az = a.x_km, a.y_km, a.z_km
    bx, by, bz = b.x_km, b.y_km, b.z_km
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    length_sq = dx * dx + dy * dy + dz * dz
    if length_sq == 0.0:
        return True

    closest_t = -(ax * dx + ay * dy + az * dz) / length_sq
    closest_t = max(0.0, min(1.0, closest_t))
    closest_x = ax + closest_t * dx
    closest_y = ay + closest_t * dy
    closest_z = az + closest_t * dz
    closest_distance_sq = (
        closest_x * closest_x
        + closest_y * closest_y
        + closest_z * closest_z
    )
    return closest_distance_sq > earth_radius_km * earth_radius_km


def range_limited_isl_graph(
    satellites: Iterable[SatelliteView],
    max_range_km: float,
) -> ISLGraph:
    sat_list = sorted(satellites, key=lambda sat: sat.sat_id)
    adjacency = {sat.sat_id: [] for sat in sat_list}
    for index, first in enumerate(sat_list):
        for second in sat_list[index + 1 :]:
            if distance_km(first, second) > max_range_km:
                continue
            if not has_line_of_sight(first, second):
                continue
            adjacency[first.sat_id].append(second.sat_id)
            adjacency[second.sat_id].append(first.sat_id)
    return ISLGraph(
        {sat_id: tuple(neighbors) for sat_id, neighbors in adjacency.items()}
    )


def build_isl_graph(satellites: Iterable[SatelliteView], config: ISLConfig) -> ISLGraph:
    if config.topology == "fully-connected":
        return fully_connected_isl_graph(satellites)
    if config.topology == "range-limited":
        if config.max_range_km is None or config.max_range_km <= 0.0:
            raise ValueError("range-limited ISL topology requires positive max_range_km")
        return range_limited_isl_graph(satellites, config.max_range_km)
    raise ValueError(f"unknown ISL topology: {config.topology}")


def shortest_route(graph: ISLGraph, source_sat: int, target_sat: int) -> Route | None:
    if source_sat == target_sat:
        return Route((source_sat,))
    if source_sat not in graph.adjacency or target_sat not in graph.adjacency:
        return None

    parents: dict[int, int | None] = {source_sat: None}
    queue: deque[int] = deque([source_sat])

    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == target_sat:
                return build_route(parents, target_sat)
            queue.append(neighbor)
    return None


def build_route(parents: dict[int, int | None], target_sat: int) -> Route:
    nodes = [target_sat]
    current = target_sat
    while parents[current] is not None:
        current = parents[current]
        nodes.append(current)
    nodes.reverse()
    return Route(tuple(nodes))
