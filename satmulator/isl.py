from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .models import Route, SatelliteView


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
