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


@dataclass(frozen=True)
class ConstellationLayout:
    topology: str
    candidate_graph: ISLGraph


def fully_connected_isl_graph(satellites: Iterable[SatelliteView]) -> ISLGraph:
    sat_ids = sorted(sat.sat_id for sat in satellites)
    if len(sat_ids) != len(set(sat_ids)):
        raise ValueError("duplicate satellite id")
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


def grid_constellation_layout(
    satellites: Iterable[SatelliteView],
    walker_phase: int = 0,
) -> ConstellationLayout:
    """Build the fixed four-neighbor topology for a Walker constellation."""

    sat_list = sorted(satellites, key=lambda sat: sat.sat_id)
    if len(sat_list) != len({sat.sat_id for sat in sat_list}):
        raise ValueError("duplicate satellite id")
    by_position: dict[tuple[int, int], SatelliteView] = {}
    for sat in sat_list:
        if sat.plane is None or sat.plane < 0 or sat.slot is None or sat.slot < 0:
            raise ValueError(
                "grid ISL topology requires non-negative plane and slot metadata"
            )
        position = (sat.plane, sat.slot)
        if position in by_position:
            raise ValueError(f"duplicate satellite grid position: {position}")
        by_position[position] = sat

    if not sat_list:
        return ConstellationLayout("grid", ISLGraph({}))

    plane_count = max(plane for plane, _ in by_position) + 1
    slots_per_plane = max(slot for _, slot in by_position) + 1
    expected_positions = {
        (plane, slot)
        for plane in range(plane_count)
        for slot in range(slots_per_plane)
    }
    if set(by_position) != expected_positions:
        raise ValueError("grid ISL topology requires a complete rectangular layout")

    adjacency: dict[int, set[int]] = {sat.sat_id: set() for sat in sat_list}

    def add_link(
        first_position: tuple[int, int],
        second_position: tuple[int, int],
    ) -> None:
        first = by_position[first_position].sat_id
        second = by_position[second_position].sat_id
        if first == second:
            return
        adjacency[first].add(second)
        adjacency[second].add(first)

    for plane in range(plane_count):
        for slot in range(slots_per_plane):
            add_link((plane, slot), (plane, (slot + 1) % slots_per_plane))

    for plane in range(plane_count - 1):
        for slot in range(slots_per_plane):
            add_link((plane, slot), (plane + 1, slot))

    if plane_count > 1:
        seam_offset = walker_phase % slots_per_plane
        for slot in range(slots_per_plane):
            add_link(
                (plane_count - 1, slot),
                (0, (slot + seam_offset) % slots_per_plane),
            )

    candidate_graph = ISLGraph(
        {sat_id: tuple(sorted(neighbors)) for sat_id, neighbors in adjacency.items()}
    )
    return ConstellationLayout("grid", candidate_graph)


def build_constellation_layout(
    satellites: Iterable[SatelliteView],
    config: ISLConfig,
    *,
    walker_phase: int = 0,
) -> ConstellationLayout:
    sat_list = tuple(satellites)
    if config.topology == "fully-connected":
        return ConstellationLayout(
            "fully-connected",
            fully_connected_isl_graph(sat_list),
        )
    if config.topology == "grid":
        return grid_constellation_layout(sat_list, walker_phase)
    raise ValueError(f"unknown ISL topology: {config.topology}")


def filter_link_availability(
    satellites: Iterable[SatelliteView],
    layout: ConstellationLayout,
    max_range_km: float,
) -> ISLGraph:
    if max_range_km <= 0.0:
        raise ValueError("grid ISL topology requires positive max_range_km")

    sat_list = tuple(satellites)
    by_id = {sat.sat_id: sat for sat in sat_list}
    if len(by_id) != len(sat_list):
        raise ValueError("duplicate satellite id")
    if set(by_id) != set(layout.candidate_graph.adjacency):
        raise ValueError("satellites do not match constellation layout")

    adjacency: dict[int, list[int]] = {sat_id: [] for sat_id in by_id}
    for first_id, neighbors in layout.candidate_graph.adjacency.items():
        first = by_id[first_id]
        for second_id in neighbors:
            if second_id <= first_id:
                continue
            second = by_id[second_id]
            if distance_km(first, second) > max_range_km:
                continue
            if not has_line_of_sight(first, second):
                continue
            adjacency[first_id].append(second_id)
            adjacency[second_id].append(first_id)

    return ISLGraph(
        {sat_id: tuple(sorted(neighbors)) for sat_id, neighbors in adjacency.items()}
    )


def grid_isl_graph(
    satellites: Iterable[SatelliteView],
    max_range_km: float,
    *,
    walker_phase: int = 0,
) -> ISLGraph:
    """Compatibility helper that builds and filters a grid in one call."""

    sat_list = tuple(satellites)
    layout = grid_constellation_layout(sat_list, walker_phase)
    return filter_link_availability(sat_list, layout, max_range_km)


def build_isl_graph(
    satellites: Iterable[SatelliteView],
    config: ISLConfig,
    *,
    layout: ConstellationLayout | None = None,
    walker_phase: int = 0,
) -> ISLGraph:
    sat_list = tuple(satellites)
    if layout is None:
        layout = build_constellation_layout(
            sat_list,
            config,
            walker_phase=walker_phase,
        )
    if layout.topology != config.topology:
        raise ValueError("constellation layout does not match ISL topology")
    if config.topology == "fully-connected":
        return layout.candidate_graph
    if config.topology == "grid":
        if config.max_range_km is None:
            raise ValueError("grid ISL topology requires positive max_range_km")
        return filter_link_availability(sat_list, layout, config.max_range_km)
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
