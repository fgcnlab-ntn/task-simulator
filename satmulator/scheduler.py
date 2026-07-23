from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .battery import battery_is_safe, projected_battery_after_step
from .isl import ISLGraph, shortest_route
from .models import (
    Assignment,
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    Route,
    SatelliteView,
    SchedulerConfig,
    Task,
    TaskConfig,
)
from .route_cost import (
    RouteCost,
    RouteTiming,
    compute_cycles,
    estimate_route_cost,
    estimate_route_timing,
    task_compute_time_s,
    transmission_energy_j,
)


def distance_km(a: SatelliteView, b: SatelliteView) -> float:
    dx = a.x_km - b.x_km
    dy = a.y_km - b.y_km
    dz = a.z_km - b.z_km
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def route_or_raise(graph: ISLGraph, source_sat: int, target_sat: int) -> Route:
    route = shortest_route(graph, source_sat, target_sat)
    if route is None:
        raise ValueError(f"no ISL route from {source_sat} to {target_sat}")
    return route


@dataclass(frozen=True)
class PhoenixCandidateCache:
    sunlit_by_plane: dict[int, tuple[SatelliteView, ...]]
    sunlit_global: tuple[SatelliteView, ...]
    sunlit_counts_by_plane: dict[int, int]


@dataclass(frozen=True)
class GreedyEnergyCandidate:
    assignment: Assignment
    finish_time_s: float
    energy_j: float
    battery_cost_j: float


def route_parents_from_source(
    graph: ISLGraph, source_sat: int
) -> dict[int, int | None]:
    """Return the shortest-route parent tree rooted at one source."""
    if source_sat not in graph.adjacency:
        return {}

    parents: dict[int, int | None] = {source_sat: None}
    queue: deque[int] = deque([source_sat])

    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)

    return parents


def route_parents_avoiding_relays(
    graph: ISLGraph,
    source_sat: int,
    blocked_relays: set[int],
) -> dict[int, int | None]:
    """Build a shortest-path tree without traversing blocked relay nodes."""

    if source_sat not in graph.adjacency:
        return {}

    parents: dict[int, int | None] = {source_sat: None}
    queue: deque[int] = deque([source_sat])
    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in parents or neighbor in blocked_relays:
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    return parents


def route_from_parents(
    parents: dict[int, int | None],
    target_sat: int,
) -> Route | None:
    nodes = route_nodes_from_parents(parents, target_sat)
    if nodes is None:
        return None
    return Route(nodes)


def route_nodes_from_parents(
    parents: dict[int, int | None],
    target_sat: int,
) -> tuple[int, ...] | None:
    nodes = reversed_route_nodes_from_parents(parents, target_sat)
    if nodes is None:
        return None
    nodes.reverse()
    return tuple(nodes)


def reversed_route_nodes_from_parents(
    parents: dict[int, int | None],
    target_sat: int,
) -> list[int] | None:
    if target_sat not in parents:
        return None

    nodes = [target_sat]
    current = target_sat
    while parents[current] is not None:
        current = parents[current]
        nodes.append(current)
    return nodes


def reserved_energy_for_sat(reserved_energy, sat_id: int) -> float:
    if isinstance(reserved_energy, BatteryReservation):
        return reserved_energy.spent_transmission_j.get(sat_id, 0.0)
    if isinstance(reserved_energy, dict):
        return reserved_energy.get(sat_id, 0.0)
    if 0 <= sat_id < len(reserved_energy):
        return reserved_energy[sat_id]
    return 0.0


class BatteryReservation:
    """Batch-local battery headroom and transmission-energy ledger."""

    def __init__(
        self,
        *,
        remaining_j: dict[int, float],
        free_sunlit_compute_s: dict[int, float],
        compute_config: ComputeConfig,
    ) -> None:
        self.remaining_j = remaining_j
        self.free_sunlit_compute_s = free_sunlit_compute_s
        self.spent_transmission_j = {sat_id: 0.0 for sat_id in remaining_j}
        self.compute_config = compute_config

    @classmethod
    def build(
        cls,
        *,
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
    ) -> BatteryReservation:
        remaining_j = {}
        free_sunlit_compute_s = {}
        for sat in satellite_views:
            minimum_j = minimum_projected_battery_until_recharge(
                sat=sat,
                available_time_s=float(time_s) + sat.queue_backlog_s,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
                extra_energy_j=sat.pending_task_energy_j,
            )
            remaining_j[sat.sat_id] = max(0.0, minimum_j - battery.min_safe_j)
            next_eclipse = (
                sat.next_eclipse_time_s
                if sat.next_eclipse_time_s is not None
                else sat.illumination_horizon_time_s
            )
            free_sunlit_compute_s[sat.sat_id] = (
                max(
                    0.0,
                    next_eclipse
                    - float(time_s)
                    - sat.queue_backlog_s,
                )
                if (
                    sat.sunlit
                    and next_eclipse is not None
                    and next_eclipse > float(time_s)
                    and battery.harvest_w
                    >= battery.idle_w + compute_config.cpu_power_w
                )
                else 0.0
            )
        return cls(
            remaining_j=remaining_j,
            free_sunlit_compute_s=free_sunlit_compute_s,
            compute_config=compute_config,
        )

    def allows(self, *, route: Route, route_cost: RouteCost) -> bool:
        target_compute_j = (
            route_cost.compute_time_s * self.compute_config.cpu_power_w
        )
        charge_target_compute = (
            route_cost.compute_time_s
            > self.free_sunlit_compute_s.get(route.target_sat, 0.0) + 1.0e-9
        )
        for sat_id, energy_j in route_cost.energy_by_sat.items():
            if sat_id == route.target_sat:
                energy_j = max(0.0, energy_j - target_compute_j)
                if charge_target_compute:
                    energy_j += target_compute_j
            if energy_j > self.remaining_j.get(sat_id, 0.0) + 1.0e-9:
                return False
        return True

    def allows_compute(self, *, sat_id: int, compute_time_s: float) -> bool:
        free_compute_s = self.free_sunlit_compute_s.get(sat_id, 0.0)
        compute_j = (
            0.0
            if compute_time_s <= free_compute_s + 1.0e-9
            else compute_time_s * self.compute_config.cpu_power_w
        )
        return compute_j <= self.remaining_j.get(sat_id, 0.0) + 1.0e-9

    def reserve(self, *, route: Route, route_cost: RouteCost) -> None:
        transmission_by_sat = route_transmission_energy_by_sat(
            route=route,
            route_cost=route_cost,
            compute_config=self.compute_config,
        )
        for sat_id, energy_j in transmission_by_sat.items():
            self.spent_transmission_j[sat_id] = (
                self.spent_transmission_j.get(sat_id, 0.0) + energy_j
            )
        if (
            route_cost.compute_time_s
            > self.free_sunlit_compute_s.get(route.target_sat, 0.0) + 1.0e-9
        ):
            transmission_by_sat[route.target_sat] = (
                transmission_by_sat.get(route.target_sat, 0.0)
                + route_cost.compute_time_s * self.compute_config.cpu_power_w
            )
        for sat_id, energy_j in transmission_by_sat.items():
            self.remaining_j[sat_id] = self.remaining_j.get(sat_id, 0.0) - energy_j
        self.free_sunlit_compute_s[route.target_sat] = max(
            0.0,
            self.free_sunlit_compute_s.get(route.target_sat, 0.0)
            - route_cost.compute_time_s,
        )


def hard_limit_reserved_energy_by_sat(
    *,
    satellite_views: list[SatelliteView],
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
) -> BatteryReservation:
    """Build the batch-local energy ledger from the exact transition lookup."""

    return BatteryReservation.build(
        satellite_views=satellite_views,
        time_s=time_s,
        step_s=step_s,
        battery=battery,
        compute_config=compute_config,
    )


def eclipse_route_respects_hard_limit(
    *,
    route_cost,
    satellite_by_id: dict[int, SatelliteView],
    reserved_energy,
    battery: BatteryConfig,
) -> bool:
    for sat_id, energy_j in route_cost.energy_by_sat.items():
        sat = satellite_by_id[sat_id]
        if sat.sunlit:
            continue
        projected = sat.battery_j - reserved_energy_for_sat(reserved_energy, sat_id)
        projected -= energy_j
        if not battery_is_safe(projected, battery.min_safe_j):
            return False
    return True


def reserve_route_energy(
    *,
    route_cost,
    reserved_energy,
) -> None:
    for sat_id, energy_j in route_cost.energy_by_sat.items():
        if isinstance(reserved_energy, dict):
            reserved_energy[sat_id] = reserved_energy.get(sat_id, 0.0) + energy_j
        elif 0 <= sat_id < len(reserved_energy):
            reserved_energy[sat_id] += energy_j


def route_transmission_energy_by_sat(
    *,
    route: Route,
    route_cost: RouteCost,
    compute_config: ComputeConfig,
) -> dict[int, float]:
    """Return route energy excluding target CPU compute energy."""

    target_compute_j = route_cost.compute_time_s * compute_config.cpu_power_w
    transmission_by_sat = {}
    for sat_id, energy_j in route_cost.energy_by_sat.items():
        if sat_id == route.target_sat:
            energy_j = max(0.0, energy_j - target_compute_j)
        if energy_j > 0.0:
            transmission_by_sat[sat_id] = energy_j
    return transmission_by_sat


def reserve_route_transmission_energy(
    *,
    route: Route,
    route_cost: RouteCost,
    compute_config: ComputeConfig,
    reserved_energy,
) -> None:
    if isinstance(reserved_energy, BatteryReservation):
        reserved_energy.reserve(route=route, route_cost=route_cost)
        return
    for sat_id, energy_j in route_transmission_energy_by_sat(
        route=route,
        route_cost=route_cost,
        compute_config=compute_config,
    ).items():
        if isinstance(reserved_energy, dict):
            reserved_energy[sat_id] = reserved_energy.get(sat_id, 0.0) + energy_j
        elif 0 <= sat_id < len(reserved_energy):
            reserved_energy[sat_id] += energy_j


def _projection_horizon_s(
    *,
    sat: SatelliteView,
    time_s: int,
    step_s: int,
    workload_s: float,
) -> float:
    """Return the horizon through the next unavoidable discharge interval."""

    now = float(time_s)
    fallback = now + max(float(step_s), workload_s)
    horizon = sat.illumination_horizon_time_s
    if sat.sunlit:
        next_sunlit = sat.next_sunlit_time_s
        if next_sunlit is not None and next_sunlit > now:
            return next_sunlit
        return max(fallback, horizon) if horizon is not None else fallback

    next_sunlit = sat.next_sunlit_time_s
    if next_sunlit is not None and next_sunlit > now:
        return next_sunlit
    return max(fallback, horizon) if horizon is not None else fallback


def _project_interval(
    *,
    battery_j: float,
    minimum_j: float,
    duration_s: float,
    sunlit: bool,
    compute_s: float,
    battery: BatteryConfig,
    compute_power_w: float,
) -> tuple[float, float]:
    if duration_s <= 0.0:
        return battery_j, minimum_j

    harvest_w = battery.harvest_w if sunlit else 0.0
    compute_s = max(0.0, min(duration_s, compute_s))
    idle_s = duration_s - compute_s

    if compute_s > 0.0:
        net_w = harvest_w - battery.idle_w - compute_power_w
        battery_j += net_w * compute_s
        battery_j = min(battery.capacity_j, battery_j)
        minimum_j = min(minimum_j, battery_j)

    if idle_s > 0.0:
        net_w = harvest_w - battery.idle_w
        battery_j += net_w * idle_s
        battery_j = min(battery.capacity_j, battery_j)
        minimum_j = min(minimum_j, battery_j)

    return battery_j, minimum_j


def minimum_projected_battery_until_recharge(
    *,
    sat: SatelliteView,
    available_time_s: float,
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig | None = None,
    compute_power_w: float | None = None,
    extra_compute_time_s: float = 0.0,
    extra_energy_j: float = 0.0,
) -> float:
    """Project the minimum battery through the next eclipse/recharge window.

    ``available_time_s`` encodes compute already queued for the satellite.
    ``extra_compute_time_s`` is the candidate work appended to that queue.
    ``extra_energy_j`` is non-compute route energy already reserved or added by
    the candidate.  Runtime still updates the real battery step by step; this
    function only answers whether the scheduler may spend the margin.
    """

    if compute_power_w is None:
        if compute_config is None:
            raise ValueError("compute_config or compute_power_w is required")
        compute_power_w = compute_config.cpu_power_w

    now = float(time_s)
    queued_compute_s = max(0.0, available_time_s - now)
    workload_s = queued_compute_s + max(0.0, extra_compute_time_s)
    horizon_s = _projection_horizon_s(
        sat=sat,
        time_s=time_s,
        step_s=step_s,
        workload_s=workload_s,
    )

    battery_j = sat.battery_j - max(0.0, extra_energy_j)
    minimum_j = min(sat.battery_j, battery_j)
    remaining_compute_s = workload_s
    cursor_s = now

    def consume_until(end_s: float, sunlit: bool) -> None:
        nonlocal battery_j, minimum_j, remaining_compute_s, cursor_s
        end_s = min(end_s, horizon_s)
        if end_s <= cursor_s:
            return
        duration_s = end_s - cursor_s
        compute_s = min(remaining_compute_s, duration_s)
        battery_j, minimum_j = _project_interval(
            battery_j=battery_j,
            minimum_j=minimum_j,
            duration_s=duration_s,
            sunlit=sunlit,
            compute_s=compute_s,
            battery=battery,
            compute_power_w=compute_power_w,
        )
        remaining_compute_s -= compute_s
        cursor_s = end_s

    if sat.sunlit:
        next_eclipse = sat.next_eclipse_time_s
        if next_eclipse is not None and next_eclipse <= now:
            consume_until(horizon_s, False)
        elif next_eclipse is not None and next_eclipse < horizon_s:
            consume_until(next_eclipse, True)
            consume_until(horizon_s, False)
        else:
            consume_until(horizon_s, True)
    else:
        consume_until(horizon_s, False)

    return minimum_j


def route_respects_battery_projection(
    *,
    route: Route,
    route_cost: RouteCost,
    satellite_by_id: dict[int, SatelliteView],
    reserved_available_time: dict[int, float],
    reserved_energy,
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
) -> bool:
    if isinstance(reserved_energy, BatteryReservation):
        return reserved_energy.allows(route=route, route_cost=route_cost)

    transmission_by_sat = route_transmission_energy_by_sat(
        route=route,
        route_cost=route_cost,
        compute_config=compute_config,
    )
    touched_sat_ids = set(route_cost.energy_by_sat)
    touched_sat_ids.add(route.target_sat)
    for sat_id in touched_sat_ids:
        sat = satellite_by_id[sat_id]
        extra_compute_time_s = (
            route_cost.compute_time_s if sat_id == route.target_sat else 0.0
        )
        extra_energy_j = (
            reserved_energy_for_sat(reserved_energy, sat_id)
            + transmission_by_sat.get(sat_id, 0.0)
        )
        minimum_j = minimum_projected_battery_until_recharge(
            sat=sat,
            available_time_s=reserved_available_time.get(
                sat_id,
                float(time_s) + sat.queue_backlog_s,
            ),
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            extra_compute_time_s=extra_compute_time_s,
            extra_energy_j=extra_energy_j,
        )
        if not battery_is_safe(minimum_j, battery.min_safe_j):
            return False
    return True


def routes_from_source(graph: ISLGraph, source_sat: int) -> dict[int, Route]:
    """Return shortest routes from one source to every reachable satellite."""

    parents = route_parents_from_source(graph, source_sat)
    routes: dict[int, Route] = {}
    for target_sat in parents:
        route = route_from_parents(parents, target_sat)
        assert route is not None
        routes[target_sat] = route
    return routes


def routes_to_targets(
    graph: ISLGraph,
    source_sat: int,
    target_sats: set[int],
) -> dict[int, Route]:
    """Return shortest routes from one source to requested reachable targets."""

    if not target_sats or source_sat not in graph.adjacency:
        return {}

    remaining = set(target_sats)
    parents: dict[int, int | None] = {source_sat: None}
    queue: deque[int] = deque([source_sat])

    if source_sat in remaining:
        remaining.remove(source_sat)

    while queue and remaining:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor in remaining:
                remaining.remove(neighbor)
                if not remaining:
                    break
            queue.append(neighbor)

    found_targets = target_sats - remaining
    routes: dict[int, Route] = {}
    for target_sat in found_targets:
        nodes = [target_sat]
        current = target_sat
        while parents[current] is not None:
            current = parents[current]
            nodes.append(current)
        nodes.reverse()
        routes[target_sat] = Route(tuple(nodes))
    return routes


class Scheduler:
    name = "base"

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        raise NotImplementedError

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        return [
            self.assign_task(
                task=task,
                satellite_views=satellite_views,
                isl_graph=isl_graph,
            )
            for task in tasks
        ]


class LocalOnlyScheduler(Scheduler):
    name = "local"

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        if task.source_sat is None:
            raise ValueError("task.source_sat is required for local scheduling")
        if task.source_sat not in isl_graph.adjacency:
            raise ValueError(
                f"source satellite {task.source_sat} is not present in the ISL graph"
            )
        return Assignment(
            task_id=task.task_id,
            route=(task.source_sat,),
            mode=self.name,
        )

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        if battery is None or compute_config is None or isl_config is None:
            return [
                self.assign_task(
                    task=task,
                    satellite_views=satellite_views,
                    isl_graph=isl_graph,
                )
                for task in tasks
            ]

        by_id = {sat.sat_id: sat for sat in satellite_views}
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s
            for sat in satellite_views
        }
        assignments: list[Assignment] = []

        for task in tasks:
            assert task.source_sat is not None
            route = Route((task.source_sat,))
            route_cost = estimate_route_cost(
                task=task,
                route=route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if route_respects_battery_projection(
                route=route,
                route_cost=route_cost,
                satellite_by_id=by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                assignments.append(
                    Assignment(task_id=task.task_id, route=route, mode=self.name)
                )
                reserve_route_transmission_energy(
                    route=route,
                    route_cost=route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                reserved_available_time[route.target_sat] += route_cost.compute_time_s
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="battery_hard_constraint",
                    )
                )

        return assignments


class NearestSunlitScheduler(Scheduler):
    name = "nearest-sunlit"

    def _assignment_for_source(
        self,
        *,
        task_id: int,
        source: SatelliteView,
        sunlit_targets: tuple[SatelliteView, ...],
        isl_graph: ISLGraph,
    ) -> Assignment:
        mode = "local"
        route = route_or_raise(isl_graph, source.sat_id, source.sat_id)
        if not source.sunlit:
            routes_by_target = routes_from_source(isl_graph, source.sat_id)
            reachable_sunlit_targets = [
                sat for sat in sunlit_targets if sat.sat_id in routes_by_target
            ]
            if reachable_sunlit_targets:
                target = min(
                    reachable_sunlit_targets,
                    key=lambda sat: routes_by_target[sat.sat_id].hop_count,
                )
                route = routes_by_target[target.sat_id]
                mode = "offload"
        return Assignment(
            task_id=task_id,
            route=route,
            mode=mode,
        )

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        assert task.source_sat is not None
        satellite_by_id = {sat.sat_id: sat for sat in satellite_views}
        source = satellite_by_id[task.source_sat]
        return self._assignment_for_source(
            task_id=task.task_id,
            source=source,
            sunlit_targets=tuple(sat for sat in satellite_views if sat.sunlit),
            isl_graph=isl_graph,
        )

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        if battery is None or compute_config is None or isl_config is None:
            return [
                self.assign_task(
                    task=task,
                    satellite_views=satellite_views,
                    isl_graph=isl_graph,
                )
                for task in tasks
            ]

        satellite_by_id = {sat.sat_id: sat for sat in satellite_views}
        sunlit_targets = tuple(sat for sat in satellite_views if sat.sunlit)
        assignment_by_source: dict[int, Assignment] = {}
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s
            for sat in satellite_views
        }
        assignments: list[Assignment] = []

        for task in tasks:
            assert task.source_sat is not None
            template = assignment_by_source.get(task.source_sat)
            if template is None:
                template = self._assignment_for_source(
                    task_id=task.task_id,
                    source=satellite_by_id[task.source_sat],
                    sunlit_targets=sunlit_targets,
                    isl_graph=isl_graph,
                )
                assignment_by_source[task.source_sat] = template
            route_cost = estimate_route_cost(
                task=task,
                route=template.route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if route_respects_battery_projection(
                route=template.route,
                route_cost=route_cost,
                satellite_by_id=satellite_by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=template.route,
                        mode=template.mode,
                        score=template.score,
                        failed_reason=template.failed_reason,
                    )
                )
                reserve_route_transmission_energy(
                    route=template.route,
                    route_cost=route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                reserved_available_time[template.route.target_sat] += (
                    route_cost.compute_time_s
                )
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=template.route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="battery_hard_constraint",
                    )
                )

        return assignments


class GreedyEnergyScheduler(Scheduler):
    """Greedy baseline adapted from the LEO energy-allocation paper.

    Ground stations are intentionally not modeled.  The paper's DVFS compute
    model is represented by the simulator's existing compute-power model, and
    Friis link loss is represented by the existing tx-power-per-bit model.

    Feasibility is enforced first: deadline, source-local CPU quota, and the
    shadow battery guard are hard constraints.  Among feasible local and sunlit
    relay candidates, choose the lowest shadow-battery-impact option.  Energy
    spent on sunlit satellites is treated as cheaper than energy drained from
    eclipse satellites, matching the paper's battery-preservation intent.
    """

    name = "greedy-energy"
    max_remote_candidates_per_source = 64
    shadow_soft_guard_ratio = 0.65

    def _step_capacity_cycles(
        self,
        *,
        step_s: int,
        compute_config: ComputeConfig,
        scheduler_config: SchedulerConfig,
    ) -> float:
        return (
            step_s
            * compute_config.cpu_frequency_hz
            * scheduler_config.cpu_utilization_limit
        )

    def _local_quota_cycles(
        self,
        *,
        sat: SatelliteView,
        step_s: int,
        time_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        scheduler_config: SchedulerConfig,
    ) -> float:
        cpu_quota_cycles = self._step_capacity_cycles(
            step_s=step_s,
            compute_config=compute_config,
            scheduler_config=scheduler_config,
        )
        if sat.sunlit:
            return cpu_quota_cycles

        soft_guard_j = self.shadow_soft_guard_ratio * battery.capacity_j
        idle_energy_j = battery.idle_w * step_s if time_s > 0 else 0.0
        task_energy_budget_j = sat.battery_j - soft_guard_j - idle_energy_j
        if task_energy_budget_j <= 0.0:
            return 0.0

        battery_quota_cycles = (
            task_energy_budget_j
            / compute_config.cpu_power_w
            * compute_config.cpu_frequency_hz
        )
        return min(cpu_quota_cycles, battery_quota_cycles)

    def _local_quota_by_sat(
        self,
        *,
        satellite_views: list[SatelliteView],
        step_s: int,
        time_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        scheduler_config: SchedulerConfig,
    ) -> dict[int, float]:
        return {
            sat.sat_id: self._local_quota_cycles(
                sat=sat,
                step_s=step_s,
                time_s=time_s,
                battery=battery,
                compute_config=compute_config,
                scheduler_config=scheduler_config,
            )
            for sat in satellite_views
        }

    def _battery_cost_j(
        self,
        *,
        energy_by_sat: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
    ) -> float:
        return sum(
            energy_j
            for sat_id, energy_j in energy_by_sat.items()
            if not satellite_views_by_id[sat_id].sunlit
        )

    def _candidate_for_route(
        self,
        *,
        task: Task,
        route: Route,
        mode: str,
        time_s: int,
        reserved_available_time: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> GreedyEnergyCandidate | None:
        timing = estimate_route_timing(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        finish_time_s = (
            max(
                float(time_s) + timing.transmission_time_s,
                reserved_available_time[route.target_sat],
            )
            + timing.compute_time_s
        )
        deadline_time_s = task.created_time_s + task.deadline_s
        if finish_time_s > deadline_time_s:
            return None

        cost = estimate_route_cost(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        energy_j = cost.total_energy_j
        battery_cost_j = self._battery_cost_j(
            energy_by_sat=cost.energy_by_sat,
            satellite_views_by_id=satellite_views_by_id,
        )
        return GreedyEnergyCandidate(
            assignment=Assignment(
                task_id=task.task_id,
                route=route,
                mode=mode,
                score=battery_cost_j,
            ),
            finish_time_s=finish_time_s,
            energy_j=energy_j,
            battery_cost_j=battery_cost_j,
        )

    def _local_candidate(
        self,
        *,
        task: Task,
        source: SatelliteView,
        time_s: int,
        reserved_available_time: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> GreedyEnergyCandidate | None:
        return self._candidate_for_route(
            task=task,
            route=Route((source.sat_id,)),
            mode="local",
            time_s=time_s,
            reserved_available_time=reserved_available_time,
            satellite_views_by_id=satellite_views_by_id,
            compute_config=compute_config,
            isl_config=isl_config,
        )

    def _remote_compute_candidates(
        self,
        *,
        task: Task,
        remote_routes: tuple[Route, ...],
        time_s: int,
        reserved_available_time: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> list[GreedyEnergyCandidate]:
        candidates: list[GreedyEnergyCandidate] = []
        for route in remote_routes:
            candidate = self._candidate_for_route(
                task=task,
                route=route,
                mode="relay",
                time_s=time_s,
                reserved_available_time=reserved_available_time,
                satellite_views_by_id=satellite_views_by_id,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _nearest_sunlit_compute_routes(
        self,
        *,
        source: SatelliteView,
        satellite_views_by_id: dict[int, SatelliteView],
        isl_graph: ISLGraph,
    ) -> tuple[Route, ...]:
        """Return a small stable set of nearest sunlit compute routes.

        The simplified link-energy model charges per hop, not per distance.
        Scanning every sunlit satellite per task is therefore just wasted work:
        lower-hop routes dominate higher-hop routes on energy.  Keep a bounded
        set so queue/deadline tie-breaks still have alternatives without
        turning each scheduling slot into tasks x constellation_size work.
        """

        if source.sat_id not in isl_graph.adjacency:
            return ()

        parents: dict[int, int | None] = {source.sat_id: None}
        queue: deque[int] = deque([source.sat_id])
        routes: list[Route] = []

        while queue and len(routes) < self.max_remote_candidates_per_source:
            current = queue.popleft()
            for neighbor in isl_graph.neighbors(current):
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                sat = satellite_views_by_id.get(neighbor)
                if sat is not None and sat.sunlit:
                    route = route_from_parents(parents, neighbor)
                    assert route is not None
                    routes.append(route)
                    if len(routes) >= self.max_remote_candidates_per_source:
                        break
                queue.append(neighbor)

        return tuple(routes)

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        by_id = {sat.sat_id: sat for sat in satellite_views}
        local_quota_cycles = self._local_quota_by_sat(
            satellite_views=satellite_views,
            step_s=step_s,
            time_s=time_s,
            battery=battery,
            compute_config=compute_config,
            scheduler_config=scheduler_config,
        )
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        reserved_local_cycles = {
            sat.sat_id: sat.queue_backlog_s * compute_config.cpu_frequency_hz
            for sat in satellite_views
        }
        remote_routes_by_source: dict[int, tuple[Route, ...]] = {}
        assignments: list[Assignment] = []
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]
            candidates: list[GreedyEnergyCandidate] = []
            task_cycles = compute_cycles(task, compute_config)
            local_fits_quota = (
                reserved_local_cycles[source.sat_id] + task_cycles
                <= local_quota_cycles[source.sat_id]
            )

            if local_fits_quota:
                local = self._local_candidate(
                    task=task,
                    source=source,
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
                    satellite_views_by_id=by_id,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if local is not None:
                    candidates.append(local)

            remote_routes = remote_routes_by_source.get(source.sat_id)
            if remote_routes is None:
                remote_routes = self._nearest_sunlit_compute_routes(
                    source=source,
                    satellite_views_by_id=by_id,
                    isl_graph=isl_graph,
                )
                remote_routes_by_source[source.sat_id] = remote_routes
            candidates.extend(
                self._remote_compute_candidates(
                    task=task,
                    remote_routes=remote_routes,
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
                    satellite_views_by_id=by_id,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
            )

            if not candidates:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=float("inf"),
                    )
                )
                continue

            safe_candidates = []
            for candidate in candidates:
                candidate_cost = estimate_route_cost(
                    task=task,
                    route=candidate.assignment.route,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if route_respects_battery_projection(
                    route=candidate.assignment.route,
                    route_cost=candidate_cost,
                    satellite_by_id=by_id,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    time_s=time_s,
                    step_s=step_s,
                    battery=battery,
                    compute_config=compute_config,
                ):
                    safe_candidates.append((candidate, candidate_cost))

            if not safe_candidates:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=candidates[0].assignment.route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="battery_hard_constraint",
                    )
                )
                continue

            chosen, chosen_cost = min(
                safe_candidates,
                key=lambda candidate: (
                    candidate[0].battery_cost_j,
                    candidate[0].finish_time_s,
                    candidate[0].energy_j,
                    candidate[0].assignment.hop_count,
                    candidate[0].assignment.target_sat,
                ),
            )

            assignments.append(chosen.assignment)
            reserved_available_time[chosen.assignment.target_sat] = chosen.finish_time_s
            reserve_route_transmission_energy(
                route=chosen.assignment.route,
                route_cost=chosen_cost,
                compute_config=compute_config,
                reserved_energy=reserved_energy,
            )
            if chosen.assignment.mode == "local":
                reserved_local_cycles[source.sat_id] += task_cycles

        return assignments


class Method3Scheduler(Scheduler):
    name = "method3"

    def _local_cost(
        self,
        *,
        sat: SatelliteView,
        available_time_s: float,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_energy_j: float,
        battery: BatteryConfig,
        warning_ratio: float,
        sunlit_local_load_weight: float,
        sunlit_local_battery_weight: float,
        eclipse_local_battery_weight: float,
        eclipse_local_warning_penalty: float,
    ) -> tuple[float, float] | None:
        t_fin = max(float(time_s), available_time_s) + compute_time_s
        if t_fin > deadline_time:
            return None

        projected = projected_battery_after_step(
            battery_now=sat.battery_j,
            sunlit=sat.sunlit,
            step_s=step_s,
            battery=battery,
            task_energy_j=compute_energy_j,
            update=time_s > 0,
        )

        eps = 1e-6
        slack_term = 1.0 / max((deadline_time - t_fin) / step_s, eps)
        current_load = max(0.0, available_time_s - float(time_s))
        load_term = current_load / step_s

        margin_j = projected - battery.min_safe_j
        margin_ratio = max(margin_j / battery.capacity_j, eps)
        battery_term = 1.0 / margin_ratio

        warn_j = battery.min_safe_j + warning_ratio * battery.capacity_j

        if sat.sunlit:
            cost = (
                sunlit_local_load_weight * load_term
                + sunlit_local_battery_weight * battery_term
                + slack_term
            )
        else:
            warning_term = eclipse_local_warning_penalty if projected < warn_j else 0.0
            cost = (
                eclipse_local_battery_weight * battery_term + warning_term + slack_term
            )

        return cost, t_fin

    def _sunlit_cost(
        self,
        *,
        available_time_s: float,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        load_weight: float,
    ) -> tuple[float, float] | None:
        t_fin = max(float(time_s), available_time_s) + compute_time_s
        if t_fin > deadline_time:
            return None

        eps = 1e-6
        current_load = max(0.0, available_time_s - float(time_s))
        load_term = current_load / step_s
        slack_term = 1.0 / max((deadline_time - t_fin) / step_s, eps)

        cost = load_weight * load_term + slack_term
        return cost, t_fin

    def _defer_cost(
        self,
        *,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
    ) -> float:
        eps = 1e-6
        t_fin_defer = float(time_s) + step_s + compute_time_s
        if t_fin_defer > deadline_time:
            return float("inf")
        return 1.0 / max((deadline_time - t_fin_defer) / step_s, eps)

    def _peek_least_loaded_sunlit(
        self,
        *,
        sunlit_heap,
        reserved_available_time: dict[int, float],
        satellite_by_id: dict[int, SatelliteView],
        time_s: int,
        exclude_sat_id: int,
    ) -> tuple[int, float] | None:
        import heapq

        skipped = []

        while sunlit_heap:
            recorded_load, sat_id = heapq.heappop(sunlit_heap)
            sat = satellite_by_id[sat_id]

            if sat_id == exclude_sat_id:
                skipped.append((recorded_load, sat_id))
                continue

            current_load = max(0.0, reserved_available_time[sat_id] - float(time_s))

            # lazy heap update
            if abs(recorded_load - current_load) > 1e-9:
                heapq.heappush(sunlit_heap, (current_load, sat_id))
                continue

            for item in skipped:
                heapq.heappush(sunlit_heap, item)
            return sat_id, reserved_available_time[sat_id]

        for item in skipped:
            heapq.heappush(sunlit_heap, item)
        return None

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        import heapq

        by_id = {sat.sat_id: sat for sat in satellite_views}

        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        unique_sources = {
            task.source_sat for task in ordered_tasks if task.source_sat is not None
        }
        route_parents_by_source: dict[int, dict[int, int | None]] = {
            source_sat: route_parents_from_source(isl_graph, source_sat)
            for source_sat in unique_sources
        }

        warning_ratio = getattr(scheduler_config, "warning_ratio", 0.10)
        sunlit_local_load_weight = getattr(
            scheduler_config, "sunlit_local_load_weight", 1.0
        )
        sunlit_local_battery_weight = getattr(
            scheduler_config, "sunlit_local_battery_weight", 0.25
        )
        eclipse_local_battery_weight = getattr(
            scheduler_config, "eclipse_local_battery_weight", 3.0
        )
        eclipse_local_warning_penalty = getattr(
            scheduler_config, "eclipse_local_warning_penalty", 2.0
        )
        sunlit_offload_load_weight = getattr(
            scheduler_config, "sunlit_offload_load_weight", 1.0
        )

        # min-heap of current sunlit loads
        sunlit_heap = []
        for sat in satellite_views:
            if sat.sunlit:
                heapq.heappush(
                    sunlit_heap,
                    (
                        max(0.0, reserved_available_time[sat.sat_id] - float(time_s)),
                        sat.sat_id,
                    ),
                )

        assignments: list[Assignment] = []

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]
            deadline_time = task.created_time_s + task.deadline_s
            compute_time_s = task_compute_time_s(task, compute_config)
            compute_energy_j = compute_time_s * compute_config.cpu_power_w

            # Action 1: local
            local_result = self._local_cost(
                sat=source,
                available_time_s=reserved_available_time[source.sat_id],
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_energy_j=compute_energy_j,
                battery=battery,
                warning_ratio=warning_ratio,
                sunlit_local_load_weight=sunlit_local_load_weight,
                sunlit_local_battery_weight=sunlit_local_battery_weight,
                eclipse_local_battery_weight=eclipse_local_battery_weight,
                eclipse_local_warning_penalty=eclipse_local_warning_penalty,
            )
            local_cost = float("inf")
            local_finish = None
            if local_result is not None:
                local_route = Route((source.sat_id,))
                local_route_cost = estimate_route_cost(
                    task=task,
                    route=local_route,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if route_respects_battery_projection(
                    route=local_route,
                    route_cost=local_route_cost,
                    satellite_by_id=by_id,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    time_s=time_s,
                    step_s=step_s,
                    battery=battery,
                    compute_config=compute_config,
                ):
                    local_cost, local_finish = local_result

            # Action 2: least-loaded sunlit
            sun_cost = float("inf")
            sun_finish = None
            sun_sat_id = None
            sun_route = None
            sun_route_cost = None

            best_sunlit = self._peek_least_loaded_sunlit(
                sunlit_heap=sunlit_heap,
                reserved_available_time=reserved_available_time,
                satellite_by_id=by_id,
                time_s=time_s,
                exclude_sat_id=source.sat_id,
            )
            if best_sunlit is not None:
                candidate_sat_id, candidate_available_time = best_sunlit
                route_parents = route_parents_by_source[source.sat_id]
                reversed_route_nodes = reversed_route_nodes_from_parents(
                    route_parents,
                    candidate_sat_id,
                )
                if reversed_route_nodes is not None:
                    sun_result = self._sunlit_cost(
                        available_time_s=candidate_available_time,
                        time_s=time_s,
                        step_s=step_s,
                        deadline_time=deadline_time,
                        compute_time_s=compute_time_s,
                        load_weight=sunlit_offload_load_weight,
                    )
                    if sun_result is not None:
                        route = Route(tuple(reversed(reversed_route_nodes)))
                        route_cost = estimate_route_cost(
                            task=task,
                            route=route,
                            compute_config=compute_config,
                            isl_config=isl_config,
                        )
                        if route_respects_battery_projection(
                            route=route,
                            route_cost=route_cost,
                            satellite_by_id=by_id,
                            reserved_available_time=reserved_available_time,
                            reserved_energy=reserved_energy,
                            time_s=time_s,
                            step_s=step_s,
                            battery=battery,
                            compute_config=compute_config,
                        ):
                            sun_cost, sun_finish = sun_result
                            sun_sat_id = candidate_sat_id
                            sun_route = route
                            sun_route_cost = route_cost

            # Action 3: defer
            defer_cost = self._defer_cost(
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
            )

            action, best_cost = min(
                [
                    ("local", local_cost),
                    ("sunlit", sun_cost),
                    ("defer", defer_cost),
                ],
                key=lambda x: x[1],
            )

            if action == "local" and local_finish is not None:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="local",
                        score=local_cost,
                    )
                )
                reserved_available_time[source.sat_id] = local_finish
                reserve_route_transmission_energy(
                    route=Route((source.sat_id,)),
                    route_cost=local_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                if source.sunlit:
                    heapq.heappush(
                        sunlit_heap,
                        (
                            max(0.0, local_finish - float(time_s)),
                            source.sat_id,
                        ),
                    )

            elif (
                action == "sunlit"
                and sun_sat_id is not None
                and sun_finish is not None
                and sun_route is not None
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=sun_route,
                        mode="offload",
                        score=sun_cost,
                    )
                )
                reserved_available_time[sun_sat_id] = sun_finish
                assert sun_route_cost is not None
                reserve_route_transmission_energy(
                    route=sun_route,
                    route_cost=sun_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                heapq.heappush(
                    sunlit_heap,
                    (
                        max(0.0, sun_finish - float(time_s)),
                        sun_sat_id,
                    ),
                )

            elif defer_cost < float("inf"):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=defer_cost,
                    )
                )

            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="fail",
                        score=float("inf"),
                        failed_reason="no_feasible_action",
                    )
                )

        return assignments


class Method3ModScheduler(Method3Scheduler):
    """Method 3 with lossless sunlit candidates and normalized action risks.

    Execution cost is the fraction of remaining deadline slots occupied after
    accepting the task.  Eclipse-local execution additionally considers the
    projected battery risk.  Deferral is expensive while usable sunlit
    capacity remains and reserves one task's compute time on the source for
    the rest of the current scheduling batch.
    """

    name = "method3_mod"

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _allow_unsafe_local_emergency(self) -> bool:
        """Whether deadline pressure may override eclipse battery safety."""
        return False

    def _execution_load_cost(
        self,
        *,
        available_time_s: float,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
    ) -> tuple[float, float] | None:
        """Return normalized projected slot occupancy and finish time."""
        import math

        now = float(time_s)
        finish_time = max(now, available_time_s) + compute_time_s
        if finish_time > deadline_time:
            return None

        remaining_time_s = deadline_time - now
        remaining_slots = max(1, math.floor(remaining_time_s / step_s))
        committed_workload_s = max(0.0, available_time_s - now)
        occupied_slots = math.ceil(
            (committed_workload_s + compute_time_s) / step_s
        )
        load_cost = self._clip_unit(occupied_slots / remaining_slots)
        return load_cost, finish_time

    def _normalized_local_cost(
        self,
        *,
        sat: SatelliteView,
        available_time_s: float,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_power_w: float,
        battery: BatteryConfig,
    ) -> tuple[float, float, bool] | None:
        execution = self._execution_load_cost(
            available_time_s=available_time_s,
            time_s=time_s,
            step_s=step_s,
            deadline_time=deadline_time,
            compute_time_s=compute_time_s,
        )
        if execution is None:
            return None

        load_cost, finish_time = execution
        if sat.sunlit:
            return load_cost, finish_time, True

        # Local eclipse work cannot be moved after it enters the execution
        # queue.  Project the battery through all work already committed to
        # this satellite, including reservations made earlier in this batch.
        committed_workload_s = max(0.0, available_time_s - float(time_s))
        projected_workload_s = committed_workload_s + compute_time_s
        projected_energy_j = (
            battery.idle_w + compute_power_w
        ) * projected_workload_s
        projected_battery_j = sat.battery_j - projected_energy_j
        projected_battery_safe = projected_battery_j >= battery.min_safe_j

        safe_span_j = battery.capacity_j - battery.min_safe_j
        if safe_span_j <= 0.0:
            battery_cost = (
                0.0 if projected_battery_j >= battery.capacity_j else 1.0
            )
        else:
            battery_cost = self._clip_unit(
                (battery.capacity_j - projected_battery_j) / safe_span_j
            )

        return (
            max(load_cost, battery_cost),
            finish_time,
            projected_battery_safe,
        )

    def _normalized_defer_cost(
        self,
        *,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        deferred_workload_s: float,
        has_feasible_sunlit_execution: bool,
        local_eclipse_cost: float | None,
    ) -> float:
        remaining_time_s = deadline_time - float(time_s)
        if remaining_time_s <= 0.0:
            return float("inf")

        finish_after_defer = (
            float(time_s) + step_s + deferred_workload_s + compute_time_s
        )
        if finish_after_defer > deadline_time:
            return float("inf")

        urgency_cost = self._clip_unit(step_s / remaining_time_s)
        deferred_workload_cost = self._clip_unit(
            (deferred_workload_s + compute_time_s) / step_s
        )
        if has_feasible_sunlit_execution:
            # Deferring would discard an immediately feasible, energy-safe
            # execution opportunity.  Execution wins the cost-1 tie.
            immediate_opportunity_cost = 1.0
        elif local_eclipse_cost is not None:
            # With no usable sunlit target, wait only when local eclipse risk
            # is worse than the complementary value of preserving it.
            immediate_opportunity_cost = 1.0 - self._clip_unit(
                local_eclipse_cost
            )
        else:
            immediate_opportunity_cost = 0.0

        return max(
            urgency_cost,
            deferred_workload_cost,
            immediate_opportunity_cost,
        )

    def _peek_least_loaded_sunlit_mod(
        self,
        *,
        sunlit_heap,
        reserved_available_time: dict[int, float],
        satellite_by_id: dict[int, SatelliteView],
        time_s: int,
        exclude_sat_id: int,
    ) -> tuple[int, float] | None:
        """Peek a valid candidate without consuming its heap entry."""
        import heapq

        excluded_entries = []
        candidate = None

        while sunlit_heap:
            recorded_load, sat_id = heapq.heappop(sunlit_heap)
            current_load = max(
                0.0,
                reserved_available_time[sat_id] - float(time_s),
            )

            # Updated reservations always push a new entry, so an old entry
            # can be discarded instead of being inserted again.
            if abs(recorded_load - current_load) > 1e-9:
                continue

            if sat_id == exclude_sat_id:
                excluded_entries.append((recorded_load, sat_id))
                continue

            candidate = (sat_id, reserved_available_time[sat_id])
            heapq.heappush(sunlit_heap, (recorded_load, sat_id))
            break

        for entry in excluded_entries:
            heapq.heappush(sunlit_heap, entry)

        return candidate

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        import heapq

        by_id = {sat.sat_id: sat for sat in satellite_views}

        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }
        reserved_transmission_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        unique_sources = {
            task.source_sat for task in ordered_tasks if task.source_sat is not None
        }
        route_parents_by_source: dict[int, dict[int, int | None]] = {
            source_sat: route_parents_from_source(isl_graph, source_sat)
            for source_sat in unique_sources
        }

        # Min-heap of current sunlit loads.  Entries are only discarded when
        # stale, so merely evaluating local execution or deferral cannot lose
        # a usable candidate.
        sunlit_heap = []
        for sat in satellite_views:
            if sat.sunlit:
                heapq.heappush(
                    sunlit_heap,
                    (
                        max(0.0, reserved_available_time[sat.sat_id] - float(time_s)),
                        sat.sat_id,
                    ),
                )

        assignments: list[Assignment] = []
        deferred_workload_s = {sat.sat_id: 0.0 for sat in satellite_views}

        for task in ordered_tasks:
            assert task.source_sat is not None

            source = by_id[task.source_sat]
            deadline_time = task.created_time_s + task.deadline_s
            compute_time_s = task_compute_time_s(task, compute_config)

            # Action 1: local
            local_result = self._normalized_local_cost(
                sat=source,
                available_time_s=reserved_available_time[source.sat_id],
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_power_w=compute_config.cpu_power_w,
                battery=battery,
            )

            local_cost = float("inf")
            local_finish = None
            local_battery_safe = source.sunlit
            local_route = Route((source.sat_id,))
            local_route_cost = None
            if local_result is not None:
                candidate_cost, candidate_finish, candidate_battery_safe = local_result
                local_route_cost = estimate_route_cost(
                    task=task,
                    route=local_route,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if candidate_battery_safe and route_respects_battery_projection(
                    route=local_route,
                    route_cost=local_route_cost,
                    satellite_by_id=by_id,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_transmission_energy,
                    time_s=time_s,
                    step_s=step_s,
                    battery=battery,
                    compute_config=compute_config,
                ):
                    local_cost = candidate_cost
                    local_finish = candidate_finish
                    local_battery_safe = True
                else:
                    local_battery_safe = False

            # Action 2: least-loaded sunlit
            sun_cost = float("inf")
            sun_finish = None
            sun_sat_id = None
            sun_route = None
            sun_route_cost = None

            best_sunlit = self._peek_least_loaded_sunlit_mod(
                sunlit_heap=sunlit_heap,
                reserved_available_time=reserved_available_time,
                satellite_by_id=by_id,
                time_s=time_s,
                exclude_sat_id=source.sat_id,
            )

            if best_sunlit is not None:
                candidate_sat_id, candidate_available_time = best_sunlit

                route_parents = route_parents_by_source[source.sat_id]
                reversed_route_nodes = reversed_route_nodes_from_parents(
                    route_parents,
                    candidate_sat_id,
                )

                if reversed_route_nodes is not None:
                    sun_result = self._normalized_local_cost(
                        sat=by_id[candidate_sat_id],
                        available_time_s=candidate_available_time,
                        time_s=time_s,
                        step_s=step_s,
                        deadline_time=deadline_time,
                        compute_time_s=compute_time_s,
                        compute_power_w=compute_config.cpu_power_w,
                        battery=battery,
                    )

                    if sun_result is not None:
                        (
                            candidate_cost,
                            candidate_finish,
                            candidate_battery_safe,
                        ) = sun_result
                        if candidate_battery_safe:
                            route = Route(tuple(reversed(reversed_route_nodes)))
                            route_cost = estimate_route_cost(
                                task=task,
                                route=route,
                                compute_config=compute_config,
                                isl_config=isl_config,
                            )
                            if route_respects_battery_projection(
                                route=route,
                                route_cost=route_cost,
                                satellite_by_id=by_id,
                                reserved_available_time=reserved_available_time,
                                reserved_energy=reserved_transmission_energy,
                                time_s=time_s,
                                step_s=step_s,
                                battery=battery,
                                compute_config=compute_config,
                            ):
                                sun_cost = candidate_cost
                                sun_finish = candidate_finish
                                sun_sat_id = candidate_sat_id
                                sun_route = route
                                sun_route_cost = route_cost

            # Action 3: defer.  Any immediately feasible sunlit execution has
            # full opportunity cost.  Only when no sunlit execution exists do
            # we compare waiting with the complementary eclipse-local risk.
            has_feasible_sunlit_execution = (
                source.sunlit
                and local_battery_safe
                and local_cost < float("inf")
            ) or sun_cost < float("inf")
            local_eclipse_cost = (
                local_cost
                if not source.sunlit and local_cost < float("inf")
                else None
            )
            defer_cost = self._normalized_defer_cost(
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                deferred_workload_s=deferred_workload_s[source.sat_id],
                has_feasible_sunlit_execution=has_feasible_sunlit_execution,
                local_eclipse_cost=local_eclipse_cost,
            )

            # Stable ordering supplies policy without a tunable bonus.  An
            # eclipse-local action whose committed queue would cross E_safe is
            # retained only as an emergency fallback after deferral.
            if source.sunlit and local_battery_safe:
                action_costs = [
                    ("local", local_cost),
                    ("sunlit", sun_cost),
                    ("defer", defer_cost),
                ]
            elif local_battery_safe:
                action_costs = [
                    ("sunlit", sun_cost),
                    ("local", local_cost),
                    ("defer", defer_cost),
                ]
            else:
                action_costs = [
                    ("sunlit", sun_cost),
                    ("defer", defer_cost),
                ]
                if self._allow_unsafe_local_emergency():
                    action_costs.append(("local", local_cost))
            action, _ = min(
                action_costs,
                key=lambda x: x[1],
            )

            if action == "local" and local_finish is not None:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=local_route,
                        mode="local",
                        score=local_cost,
                    )
                )

                reserved_available_time[source.sat_id] = local_finish
                assert local_route_cost is not None
                reserve_route_transmission_energy(
                    route=local_route,
                    route_cost=local_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_transmission_energy,
                )

                if source.sunlit:
                    heapq.heappush(
                        sunlit_heap,
                        (
                            max(0.0, local_finish - float(time_s)),
                            source.sat_id,
                        ),
                    )

            elif (
                action == "sunlit"
                and sun_sat_id is not None
                and sun_finish is not None
                and sun_route is not None
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=sun_route,
                        mode="offload",
                        score=sun_cost,
                    )
                )

                reserved_available_time[sun_sat_id] = sun_finish
                assert sun_route_cost is not None
                reserve_route_transmission_energy(
                    route=sun_route,
                    route_cost=sun_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_transmission_energy,
                )

                heapq.heappush(
                    sunlit_heap,
                    (
                        max(0.0, sun_finish - float(time_s)),
                        sun_sat_id,
                    ),
                )

            elif defer_cost < float("inf"):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=defer_cost,
                    )
                )
                deferred_workload_s[source.sat_id] += compute_time_s

            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="fail",
                        score=float("inf"),
                        failed_reason="no_feasible_action",
                    )
                )

        return assignments


class Method5Scheduler(Method3ModScheduler):
    """Method3-mod optimized for task completion under a hard DoD limit.

    Every immediately feasible, energy-safe execution is preferred over
    deferral.  Future illumination is used only for hard feasibility, not as
    a ranking signal, and no execution may project battery below the safe
    threshold before the next recharge opportunity.
    """

    name = "method5"

    def _allow_unsafe_local_emergency(self) -> bool:
        return False

    def _minimum_battery_until_next_sun(
        self,
        *,
        sat: SatelliteView,
        available_time_s: float,
        time_s: int,
        step_s: int,
        compute_time_s: float,
        compute_power_w: float,
        battery: BatteryConfig,
    ) -> float | None:
        """Project the minimum battery before the next recharge interval."""
        return minimum_projected_battery_until_recharge(
            sat=sat,
            available_time_s=available_time_s,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_power_w=compute_power_w,
            extra_compute_time_s=compute_time_s,
        )

    def _normalized_local_cost(
        self,
        *,
        sat: SatelliteView,
        available_time_s: float,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_power_w: float,
        battery: BatteryConfig,
    ) -> tuple[float, float, bool] | None:
        base_result = super()._normalized_local_cost(
            sat=sat,
            available_time_s=available_time_s,
            time_s=time_s,
            step_s=step_s,
            deadline_time=deadline_time,
            compute_time_s=compute_time_s,
            compute_power_w=compute_power_w,
            battery=battery,
        )
        if base_result is None:
            return None

        cost, finish_time, _ = base_result
        minimum_battery_j = self._minimum_battery_until_next_sun(
            sat=sat,
            available_time_s=available_time_s,
            time_s=time_s,
            step_s=step_s,
            compute_time_s=compute_time_s,
            compute_power_w=compute_power_w,
            battery=battery,
        )
        if minimum_battery_j is None:
            return cost, finish_time, False

        return (
            cost,
            finish_time,
            minimum_battery_j >= battery.min_safe_j,
        )

    def _normalized_defer_cost(
        self,
        *,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        deferred_workload_s: float,
        has_feasible_sunlit_execution: bool,
        local_eclipse_cost: float | None,
    ) -> float:
        cost = super()._normalized_defer_cost(
            time_s=time_s,
            step_s=step_s,
            deadline_time=deadline_time,
            compute_time_s=compute_time_s,
            deferred_workload_s=deferred_workload_s,
            has_feasible_sunlit_execution=has_feasible_sunlit_execution,
            local_eclipse_cost=local_eclipse_cost,
        )
        if cost == float("inf"):
            return cost

        # Under the new objective, waiting must never beat an execution that
        # is feasible now.  Execution costs are normalized to [0, 1], and the
        # stable action order resolves a cost-1 tie in favor of execution.
        if has_feasible_sunlit_execution or local_eclipse_cost is not None:
            return 1.0
        return cost

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        """Fill immediate safe capacity before allowing a one-slot wait.

        Sunlit and eclipse capacity are kept in separate lazy heaps.  Eclipse
        peers are a fallback pool: they are used only after local execution
        and sunlit offload cannot meet the deadline, and only while their
        queue remains safe until the next sunlit interval.
        """
        import heapq

        now = float(time_s)
        by_id = {sat.sat_id: sat for sat in satellite_views}
        reserved_available_time = {
            sat.sat_id: now + sat.queue_backlog_s for sat in satellite_views
        }
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )
        if not ordered_tasks:
            return []

        route_parents_by_source = {
            source_sat: route_parents_from_source(isl_graph, source_sat)
            for source_sat in {
                task.source_sat
                for task in ordered_tasks
                if task.source_sat is not None
            }
        }
        minimum_compute_time_s = min(
            task_compute_time_s(task, compute_config) for task in ordered_tasks
        )

        sunlit_heap: list[tuple[float, int]] = []
        eclipse_heap: list[tuple[float, int]] = []
        for sat in satellite_views:
            heap = sunlit_heap if sat.sunlit else eclipse_heap
            heapq.heappush(
                heap,
                (reserved_available_time[sat.sat_id], sat.sat_id),
            )
        reserved_transmission_energy_j = {
            sat.sat_id: 0.0 for sat in satellite_views
        }

        def execution_result(
            *,
            sat: SatelliteView,
            available_time_s: float,
            deadline_time: float,
            compute_time_s: float,
        ) -> tuple[float, float, bool] | None:
            if not sat.sunlit:
                result = self._normalized_local_cost(
                    sat=sat,
                    available_time_s=available_time_s,
                    time_s=time_s,
                    step_s=step_s,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                    compute_power_w=compute_config.cpu_power_w,
                    battery=battery,
                )
                if result is None:
                    return None
                cost, finish_time, battery_safe = result
                if battery_safe:
                    minimum_battery_j = self._minimum_battery_until_next_sun(
                        sat=sat,
                        available_time_s=available_time_s,
                        time_s=time_s,
                        step_s=step_s,
                        compute_time_s=compute_time_s,
                        compute_power_w=compute_config.cpu_power_w,
                        battery=battery,
                    )
                    battery_safe = (
                        minimum_battery_j is not None
                        and minimum_battery_j
                        - reserved_transmission_energy_j[sat.sat_id]
                        >= battery.min_safe_j
                    )
                return cost, finish_time, battery_safe

            result = self._execution_load_cost(
                available_time_s=available_time_s,
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
            )
            if result is None:
                return None
            cost, finish_time = result

            projected = self._normalized_local_cost(
                sat=sat,
                available_time_s=available_time_s,
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_power_w=compute_config.cpu_power_w,
                battery=battery,
            )
            return projected

        def route_transmission_energy(
            *, task: Task, route: Route, compute_time_s: float
        ) -> dict[int, float]:
            route_cost = estimate_route_cost(
                task=task,
                route=route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            transmission_by_sat = dict(route_cost.energy_by_sat)
            target_compute_energy_j = (
                compute_config.cpu_power_w * compute_time_s
            )
            target_energy_j = transmission_by_sat.get(route.target_sat, 0.0)
            target_transmission_j = max(
                0.0,
                target_energy_j - target_compute_energy_j,
            )
            if target_transmission_j > 0.0:
                transmission_by_sat[route.target_sat] = target_transmission_j
            else:
                transmission_by_sat.pop(route.target_sat, None)
            return transmission_by_sat

        def route_is_energy_safe(
            transmission_by_sat: dict[int, float],
        ) -> bool:
            for sat_id, additional_energy_j in transmission_by_sat.items():
                sat = by_id[sat_id]
                total_reserved_j = (
                    reserved_transmission_energy_j[sat_id]
                    + additional_energy_j
                )
                if sat.sunlit:
                    # Transmission is tiny and is paid while the route's task
                    # executes.  Reserving it against the current battery is
                    # conservative because this omits concurrent harvesting.
                    if sat.battery_j - total_reserved_j < battery.min_safe_j:
                        return False
                    continue

                minimum_battery_j = self._minimum_battery_until_next_sun(
                    sat=sat,
                    available_time_s=reserved_available_time[sat_id],
                    time_s=time_s,
                    step_s=step_s,
                    compute_time_s=0.0,
                    compute_power_w=compute_config.cpu_power_w,
                    battery=battery,
                )
                if (
                    minimum_battery_j is None
                    or minimum_battery_j - total_reserved_j
                    < battery.min_safe_j
                ):
                    return False
            return True

        def peer_candidate(
            *,
            heap: list[tuple[float, int]],
            task: Task,
            source: SatelliteView,
            deadline_time: float,
            compute_time_s: float,
        ) -> tuple[float, float, int, Route, dict[int, float]] | None:
            temporarily_skipped: list[tuple[float, int]] = []
            chosen = None

            while heap:
                recorded_available_time, candidate_sat_id = heapq.heappop(heap)
                current_available_time = reserved_available_time[candidate_sat_id]
                if abs(recorded_available_time - current_available_time) > 1e-9:
                    continue
                if candidate_sat_id == source.sat_id:
                    temporarily_skipped.append(
                        (recorded_available_time, candidate_sat_id)
                    )
                    continue

                reversed_route_nodes = reversed_route_nodes_from_parents(
                    route_parents_by_source[source.sat_id],
                    candidate_sat_id,
                )
                if reversed_route_nodes is None:
                    temporarily_skipped.append(
                        (recorded_available_time, candidate_sat_id)
                    )
                    continue

                candidate = by_id[candidate_sat_id]
                result = execution_result(
                    sat=candidate,
                    available_time_s=current_available_time,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                )
                if result is None:
                    # Later heap entries have no earlier CPU availability.
                    temporarily_skipped.append(
                        (recorded_available_time, candidate_sat_id)
                    )
                    break

                cost, finish_time, battery_safe = result
                if not battery_safe:
                    # With identical task sizes (the paper workload), this
                    # candidate cannot become safe again during this batch.
                    # Preserve it only when a smaller task later in the EDF
                    # sequence could still use its remaining energy margin.
                    if compute_time_s > minimum_compute_time_s:
                        temporarily_skipped.append(
                            (recorded_available_time, candidate_sat_id)
                        )
                    continue

                route = Route(tuple(reversed(reversed_route_nodes)))
                transmission_by_sat = route_transmission_energy(
                    task=task,
                    route=route,
                    compute_time_s=compute_time_s,
                )
                if not route_is_energy_safe(transmission_by_sat):
                    temporarily_skipped.append(
                        (recorded_available_time, candidate_sat_id)
                    )
                    continue
                chosen = (
                    cost,
                    finish_time,
                    candidate_sat_id,
                    route,
                    transmission_by_sat,
                )
                temporarily_skipped.append(
                    (recorded_available_time, candidate_sat_id)
                )
                break

            for entry in temporarily_skipped:
                heapq.heappush(heap, entry)
            return chosen

        assignments: list[Assignment] = []
        deferred_workload_s = {sat.sat_id: 0.0 for sat in satellite_views}

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]
            deadline_time = task.created_time_s + task.deadline_s
            compute_time_s = task_compute_time_s(task, compute_config)

            local = execution_result(
                sat=source,
                available_time_s=reserved_available_time[source.sat_id],
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
            )
            local_choice = None
            if local is not None and local[2]:
                local_choice = (
                    local[0],
                    local[1],
                    source.sat_id,
                    Route((source.sat_id,)),
                    {},
                )

            sunlit_choice = peer_candidate(
                heap=sunlit_heap,
                task=task,
                source=source,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
            )

            immediate_choices = []
            if source.sunlit:
                if local_choice is not None:
                    immediate_choices.append((0, local_choice))
                if sunlit_choice is not None:
                    immediate_choices.append((1, sunlit_choice))
            else:
                if sunlit_choice is not None:
                    immediate_choices.append((0, sunlit_choice))
                if local_choice is not None:
                    immediate_choices.append((1, local_choice))

            chosen = None
            if immediate_choices:
                _, chosen = min(
                    immediate_choices,
                    key=lambda item: (item[1][1], item[0]),
                )
            else:
                chosen = peer_candidate(
                    heap=eclipse_heap,
                    task=task,
                    source=source,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                )

            if chosen is not None:
                (
                    cost,
                    finish_time,
                    target_sat_id,
                    route,
                    transmission_by_sat,
                ) = chosen
                mode = "local" if target_sat_id == source.sat_id else "offload"
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route,
                        mode=mode,
                        score=cost,
                    )
                )
                reserved_available_time[target_sat_id] = finish_time
                for sat_id, energy_j in transmission_by_sat.items():
                    reserved_transmission_energy_j[sat_id] += energy_j
                target_heap = (
                    sunlit_heap if by_id[target_sat_id].sunlit else eclipse_heap
                )
                heapq.heappush(target_heap, (finish_time, target_sat_id))
                continue

            defer_cost = super()._normalized_defer_cost(
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                deferred_workload_s=deferred_workload_s[source.sat_id],
                has_feasible_sunlit_execution=False,
                local_eclipse_cost=None,
            )
            if defer_cost < float("inf"):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=defer_cost,
                    )
                )
                deferred_workload_s[source.sat_id] += compute_time_s
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="fail",
                        score=float("inf"),
                        failed_reason="no_safe_capacity_before_deadline",
                    )
                )

        return assignments


class Method6Scheduler(Method3ModScheduler):
    """Method3-mod with a conservative eclipse offload fallback.

    The primary policy remains the Method3-mod preference for local/sunlit
    execution.  Eclipse peers are admitted only when their projected queue
    stays above the battery threshold, and their cost is kept high while
    low-load sunlit capacity is still available.
    """

    name = "method6"

    def __init__(self) -> None:
        super().__init__()
        self._route_cache_key = None
        self._route_parents_cache: dict[int, dict[int, int | None]] = {}
        self._route_cost_cache: dict[tuple, RouteCost] = {}

    def _allow_unsafe_local_emergency(self) -> bool:
        return False

    def _restore_eclipse_candidate_after_route_rejection(self) -> bool:
        return False

    def _short_circuit_non_eclipse_actions(self) -> bool:
        return False

    def _short_circuit_local_action(
        self,
        *,
        source: SatelliteView,
        local_hard_safe: bool,
        local_finish: float | None,
        time_s: int,
        step_s: int,
    ) -> bool:
        """Whether a safe sunlit local action is cheap enough to stop search.

        This is an optimization, not policy.  It is only valid while local
        execution does not add cross-slot queue pressure.  Under high loading,
        blindly accepting every safe sunlit local task hides the load term from
        the original cost model and starves offload opportunities.
        """

        return (
            self._short_circuit_non_eclipse_actions()
            and source.sunlit
            and local_hard_safe
            and local_finish is not None
            and local_finish <= float(time_s) + step_s
        )

    def _source_can_offload(
        self,
        *,
        source: SatelliteView,
        task: Task,
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        isl_config: ISLConfig,
    ) -> bool:
        return True

    def _retry_route_after_energy_rejection(
        self,
        *,
        isl_graph: ISLGraph,
        source_sat: int,
        target_sat: int,
        task: Task,
        satellite_by_id: dict[int, SatelliteView],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        isl_config: ISLConfig,
    ) -> Route | None:
        return None

    def _blocked_route_relays(
        self,
        *,
        ordered_tasks: list[Task],
        satellite_by_id: dict[int, SatelliteView],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        isl_config: ISLConfig,
    ) -> set[int]:
        return set()

    def _peek_least_loaded_safe_eclipse_mod(
        self,
        *,
        eclipse_heap,
        compute_rejections: list[tuple[float, int]],
        route_cost_for,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        satellite_by_id: dict[int, SatelliteView],
        route_parents: dict[int, int | None],
        task: Task,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_power_w: float,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        exclude_sat_id: int,
        protected_cost: float | None,
    ) -> tuple[float, float, int, Route, RouteCost] | None:
        """Return the least-loaded reachable safe eclipse peer, if any."""
        import heapq

        skipped_entries = []
        chosen = None
        compute_energy_j = compute_time_s * compute_power_w
        while (
            compute_rejections
            and -compute_rejections[0][0] > compute_energy_j + 1.0e-9
        ):
            _neg_rejected_compute_j, sat_id = heapq.heappop(compute_rejections)
            heapq.heappush(
                eclipse_heap,
                (
                    max(
                        0.0,
                        reserved_available_time[sat_id] - float(time_s),
                    ),
                    sat_id,
                ),
            )
        eclipse_floor = (
            0.0
            if protected_cost is None
            else 1.0 - self._clip_unit(protected_cost)
        )

        while eclipse_heap:
            recorded_load, sat_id = heapq.heappop(eclipse_heap)
            current_load = max(
                0.0,
                reserved_available_time[sat_id] - float(time_s),
            )

            if abs(recorded_load - current_load) > 1e-9:
                heapq.heappush(eclipse_heap, (current_load, sat_id))
                continue

            if sat_id == exclude_sat_id:
                skipped_entries.append((recorded_load, sat_id))
                continue

            if (
                isinstance(reserved_energy, BatteryReservation)
                and not reserved_energy.allows_compute(
                    sat_id=sat_id,
                    compute_time_s=compute_time_s,
                )
            ):
                heapq.heappush(
                    compute_rejections,
                    (-compute_energy_j, sat_id),
                )
                continue

            result = self._normalized_local_cost(
                sat=satellite_by_id[sat_id],
                available_time_s=reserved_available_time[sat_id],
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_power_w=compute_power_w,
                battery=battery,
            )
            if result is None:
                skipped_entries.append((recorded_load, sat_id))
                break

            cost, finish_time, _battery_safe = result
            reversed_route_nodes = reversed_route_nodes_from_parents(
                route_parents,
                sat_id,
            )
            assert reversed_route_nodes is not None

            route = Route(tuple(reversed(reversed_route_nodes)))
            route_cost = route_cost_for(task, route)
            route_is_safe = route_respects_battery_projection(
                route=route,
                route_cost=route_cost,
                satellite_by_id=satellite_by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            )
            if not route_is_safe:
                retry_route = self._retry_route_after_energy_rejection(
                    isl_graph=isl_graph,
                    source_sat=exclude_sat_id,
                    target_sat=sat_id,
                    task=task,
                    satellite_by_id=satellite_by_id,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    isl_config=isl_config,
                )
                if retry_route is not None:
                    retry_cost = route_cost_for(task, retry_route)
                    if route_respects_battery_projection(
                        route=retry_route,
                        route_cost=retry_cost,
                        satellite_by_id=satellite_by_id,
                        reserved_available_time=reserved_available_time,
                        reserved_energy=reserved_energy,
                        time_s=time_s,
                        step_s=step_s,
                        battery=battery,
                        compute_config=compute_config,
                    ):
                        route = retry_route
                        route_cost = retry_cost
                        route_is_safe = True

            if not route_is_safe:
                if self._restore_eclipse_candidate_after_route_rejection():
                    skipped_entries.append((recorded_load, sat_id))
                continue

            chosen = (
                max(cost, eclipse_floor),
                finish_time,
                sat_id,
                route,
                route_cost,
            )
            skipped_entries.append((recorded_load, sat_id))
            break

        for entry in skipped_entries:
            heapq.heappush(eclipse_heap, entry)

        return chosen

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        import heapq

        by_id = {sat.sat_id: sat for sat in satellite_views}

        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s
            for sat in satellite_views
        }
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        def route_cost_for(task: Task, route: Route) -> RouteCost:
            key = (
                route.nodes,
                task.input_bits,
                task.output_bits,
                task.compute_time_s,
                compute_config.cycles_per_input_bit,
                compute_config.cpu_frequency_hz,
                compute_config.cpu_power_w,
                isl_config.rate_bps,
                isl_config.tx_power_w,
            )
            cost = self._route_cost_cache.get(key)
            if cost is None:
                cost = estimate_route_cost(
                    task=task,
                    route=route,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if len(self._route_cost_cache) >= 65536:
                    self._route_cost_cache.clear()
                self._route_cost_cache[key] = cost
            return cost

        blocked_route_relays = self._blocked_route_relays(
            ordered_tasks=ordered_tasks,
            satellite_by_id=by_id,
            reserved_energy=reserved_energy,
            battery=battery,
            isl_config=isl_config,
        )
        route_cache_key = (
            tuple(isl_graph.adjacency.items()),
            frozenset(blocked_route_relays),
        )
        if route_cache_key != self._route_cache_key:
            self._route_cache_key = route_cache_key
            self._route_parents_cache = {}
        route_parents_by_source = self._route_parents_cache

        def route_parents_for_source(source_sat: int) -> dict[int, int | None]:
            parents = route_parents_by_source.get(source_sat)
            if parents is None:
                parents = (
                    route_parents_avoiding_relays(
                        isl_graph,
                        source_sat,
                        blocked_route_relays - {source_sat},
                    )
                    if blocked_route_relays
                    else route_parents_from_source(isl_graph, source_sat)
                )
                route_parents_by_source[source_sat] = parents
            return parents

        sunlit_heap = []
        for sat in satellite_views:
            if sat.sunlit:
                heapq.heappush(
                    sunlit_heap,
                    (
                        max(
                            0.0,
                            reserved_available_time[sat.sat_id] - float(time_s),
                        ),
                        sat.sat_id,
                    ),
                )
        eclipse_heaps_by_source: dict[int, list[tuple[float, int]]] = {}
        eclipse_compute_rejections_by_source: dict[
            int,
            list[tuple[float, int]],
        ] = {}

        def eclipse_heap_for_source(
            source_sat: int,
            route_parents: dict[int, int | None],
        ) -> list[tuple[float, int]]:
            heap = eclipse_heaps_by_source.get(source_sat)
            if heap is None:
                heap = [
                    (
                        max(
                            0.0,
                            reserved_available_time[sat_id] - float(time_s),
                        ),
                        sat_id,
                    )
                    for sat_id in route_parents
                    if sat_id != source_sat and not by_id[sat_id].sunlit
                ]
                heapq.heapify(heap)
                eclipse_heaps_by_source[source_sat] = heap
            return heap

        assignments: list[Assignment] = []
        deferred_workload_s = {sat.sat_id: 0.0 for sat in satellite_views}

        for task in ordered_tasks:
            assert task.source_sat is not None

            source = by_id[task.source_sat]
            deadline_time = task.created_time_s + task.deadline_s
            compute_time_s = task_compute_time_s(task, compute_config)
            local_route = Route((source.sat_id,))

            local_result = self._normalized_local_cost(
                sat=source,
                available_time_s=reserved_available_time[source.sat_id],
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_power_w=compute_config.cpu_power_w,
                battery=battery,
            )

            local_cost = float("inf")
            local_finish = None
            local_hard_safe = False
            if local_result is not None:
                candidate_cost, candidate_finish, _battery_safe = local_result
                local_route_cost = RouteCost(
                    compute_time_s=compute_time_s,
                    transmission_time_s=0.0,
                    energy_by_sat={
                        source.sat_id: compute_time_s * compute_config.cpu_power_w
                    },
                )
                if route_respects_battery_projection(
                    route=local_route,
                    route_cost=local_route_cost,
                    satellite_by_id=by_id,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    time_s=time_s,
                    step_s=step_s,
                    battery=battery,
                    compute_config=compute_config,
                ):
                    local_cost = candidate_cost
                    local_finish = candidate_finish
                    local_hard_safe = True
            prefer_local = self._short_circuit_local_action(
                source=source,
                local_hard_safe=local_hard_safe,
                local_finish=local_finish,
                time_s=time_s,
                step_s=step_s,
            )

            sun_cost = float("inf")
            sun_finish = None
            sun_sat_id = None
            sun_route = None

            source_can_offload = self._source_can_offload(
                source=source,
                task=task,
                reserved_energy=reserved_energy,
                battery=battery,
                isl_config=isl_config,
            )

            best_sunlit = (
                self._peek_least_loaded_sunlit_mod(
                    sunlit_heap=sunlit_heap,
                    reserved_available_time=reserved_available_time,
                    satellite_by_id=by_id,
                    time_s=time_s,
                    exclude_sat_id=source.sat_id,
                )
                if source_can_offload and not prefer_local
                else None
            )

            if best_sunlit is not None:
                candidate_sat_id, candidate_available_time = best_sunlit
                reversed_route_nodes = reversed_route_nodes_from_parents(
                    route_parents_for_source(source.sat_id),
                    candidate_sat_id,
                )

                if reversed_route_nodes is not None:
                    sun_result = self._normalized_local_cost(
                        sat=by_id[candidate_sat_id],
                        available_time_s=candidate_available_time,
                        time_s=time_s,
                        step_s=step_s,
                        deadline_time=deadline_time,
                        compute_time_s=compute_time_s,
                        compute_power_w=compute_config.cpu_power_w,
                        battery=battery,
                    )

                    if sun_result is not None:
                        candidate_cost, candidate_finish, _battery_safe = sun_result
                        route = Route(tuple(reversed(reversed_route_nodes)))
                        route_cost = route_cost_for(task, route)
                        route_is_safe = route_respects_battery_projection(
                            route=route,
                            route_cost=route_cost,
                            satellite_by_id=by_id,
                            reserved_available_time=reserved_available_time,
                            reserved_energy=reserved_energy,
                            time_s=time_s,
                            step_s=step_s,
                            battery=battery,
                            compute_config=compute_config,
                        )
                        if not route_is_safe:
                            retry_route = self._retry_route_after_energy_rejection(
                                isl_graph=isl_graph,
                                source_sat=source.sat_id,
                                target_sat=candidate_sat_id,
                                task=task,
                                satellite_by_id=by_id,
                                reserved_energy=reserved_energy,
                                battery=battery,
                                isl_config=isl_config,
                            )
                            if retry_route is not None:
                                retry_cost = route_cost_for(task, retry_route)
                                if route_respects_battery_projection(
                                    route=retry_route,
                                    route_cost=retry_cost,
                                    satellite_by_id=by_id,
                                    reserved_available_time=reserved_available_time,
                                    reserved_energy=reserved_energy,
                                    time_s=time_s,
                                    step_s=step_s,
                                    battery=battery,
                                    compute_config=compute_config,
                                ):
                                    route = retry_route
                                    route_cost = retry_cost
                                    route_is_safe = True

                        if route_is_safe:
                            sun_cost = candidate_cost
                            sun_finish = candidate_finish
                            sun_sat_id = candidate_sat_id
                            sun_route = route
                            sun_route_cost = route_cost

            prefer_sunlit = (
                self._short_circuit_non_eclipse_actions()
                and sun_cost < float("inf")
            )

            protected_costs = []
            if source.sunlit and local_hard_safe and local_cost < float("inf"):
                protected_costs.append(local_cost)
            if sun_cost < float("inf"):
                protected_costs.append(sun_cost)
            protected_cost = min(protected_costs) if protected_costs else None

            eclipse_choice = (
                self._peek_least_loaded_safe_eclipse_mod(
                    eclipse_heap=eclipse_heap_for_source(
                        source.sat_id,
                        route_parents_for_source(source.sat_id),
                    ),
                    compute_rejections=eclipse_compute_rejections_by_source.setdefault(
                        source.sat_id,
                        [],
                    ),
                    route_cost_for=route_cost_for,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    satellite_by_id=by_id,
                    route_parents=route_parents_for_source(source.sat_id),
                    task=task,
                    time_s=time_s,
                    step_s=step_s,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                    compute_power_w=compute_config.cpu_power_w,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                    isl_graph=isl_graph,
                    exclude_sat_id=source.sat_id,
                    protected_cost=protected_cost,
                )
                if source_can_offload and not prefer_local and not prefer_sunlit
                else None
            )
            eclipse_cost = float("inf")
            eclipse_finish = None
            eclipse_sat_id = None
            eclipse_route = None
            eclipse_route_cost = None
            if eclipse_choice is not None:
                (
                    eclipse_cost,
                    eclipse_finish,
                    eclipse_sat_id,
                    eclipse_route,
                    eclipse_route_cost,
                ) = eclipse_choice

            has_immediate_execution = (
                (local_hard_safe and local_cost < float("inf"))
                or sun_cost < float("inf")
                or eclipse_cost < float("inf")
            )
            if has_immediate_execution:
                defer_cost = 1.0
            else:
                defer_cost = self._normalized_defer_cost(
                    time_s=time_s,
                    step_s=step_s,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                    deferred_workload_s=deferred_workload_s[source.sat_id],
                    has_feasible_sunlit_execution=False,
                    local_eclipse_cost=None,
                )

            if prefer_local:
                action = "local"
            elif prefer_sunlit:
                action = "sunlit"
            elif source.sunlit and local_hard_safe:
                action_costs = [
                    ("local", local_cost),
                    ("sunlit", sun_cost),
                    ("eclipse", eclipse_cost),
                    ("defer", defer_cost),
                ]
                action, _ = min(action_costs, key=lambda x: x[1])
            elif local_hard_safe:
                action_costs = [
                    ("sunlit", sun_cost),
                    ("local", local_cost),
                    ("eclipse", eclipse_cost),
                    ("defer", defer_cost),
                ]
                action, _ = min(action_costs, key=lambda x: x[1])
            else:
                action_costs = [
                    ("sunlit", sun_cost),
                    ("eclipse", eclipse_cost),
                    ("defer", defer_cost),
                ]
                action, _ = min(action_costs, key=lambda x: x[1])

            if action == "local" and local_finish is not None:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=local_route,
                        mode="local",
                        score=local_cost,
                    )
                )

                reserved_available_time[source.sat_id] = local_finish
                reserve_route_transmission_energy(
                    route=local_route,
                    route_cost=local_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )

                if source.sunlit:
                    heapq.heappush(
                        sunlit_heap,
                        (
                            max(0.0, local_finish - float(time_s)),
                            source.sat_id,
                        ),
                    )

            elif (
                action == "sunlit"
                and sun_sat_id is not None
                and sun_finish is not None
                and sun_route is not None
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=sun_route,
                        mode="offload",
                        score=sun_cost,
                    )
                )

                reserved_available_time[sun_sat_id] = sun_finish
                assert sun_route_cost is not None
                reserve_route_transmission_energy(
                    route=sun_route,
                    route_cost=sun_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                heapq.heappush(
                    sunlit_heap,
                    (
                        max(0.0, sun_finish - float(time_s)),
                        sun_sat_id,
                    ),
                )

            elif (
                action == "eclipse"
                and eclipse_sat_id is not None
                and eclipse_finish is not None
                and eclipse_route is not None
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=eclipse_route,
                        mode="offload",
                        score=eclipse_cost,
                    )
                )

                reserved_available_time[eclipse_sat_id] = eclipse_finish
                assert eclipse_route_cost is not None
                reserve_route_transmission_energy(
                    route=eclipse_route,
                    route_cost=eclipse_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )

            elif defer_cost < float("inf"):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=defer_cost,
                    )
                )
                deferred_workload_s[source.sat_id] += compute_time_s

            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="fail",
                        score=float("inf"),
                        failed_reason="no_safe_capacity_before_deadline",
                    )
                )

        return assignments


class Method7Scheduler(Method6Scheduler):
    """Method6 with stable eclipse heaps and battery-constrained routing."""

    name = "method7"

    def _restore_eclipse_candidate_after_route_rejection(self) -> bool:
        return True

    def _short_circuit_non_eclipse_actions(self) -> bool:
        return True

    def _blocked_route_relays(
        self,
        *,
        ordered_tasks: list[Task],
        satellite_by_id: dict[int, SatelliteView],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        isl_config: ISLConfig,
    ) -> set[int]:
        max_relay_energy_j = max(
            (
                transmission_energy_j(task.input_bits, isl_config)
                + transmission_energy_j(task.output_bits, isl_config)
                for task in ordered_tasks
            ),
            default=0.0,
        )
        return {
            sat_id
            for sat_id, sat in satellite_by_id.items()
            if not sat.sunlit
            and sat.battery_j
            - reserved_energy_for_sat(reserved_energy, sat_id)
            - max_relay_energy_j
            < battery.min_safe_j
        }


class Method8Scheduler(Method7Scheduler):
    """Method7 with one dynamic reroute and impossible-source pruning."""

    name = "method8"

    def _source_can_offload(
        self,
        *,
        source: SatelliteView,
        task: Task,
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        isl_config: ISLConfig,
    ) -> bool:
        if source.sunlit:
            return True
        source_tx_energy_j = transmission_energy_j(task.input_bits, isl_config)
        projected_battery_j = source.battery_j - reserved_energy_for_sat(
            reserved_energy,
            source.sat_id,
        )
        return projected_battery_j - source_tx_energy_j >= battery.min_safe_j

    def _retry_route_after_energy_rejection(
        self,
        *,
        isl_graph: ISLGraph,
        source_sat: int,
        target_sat: int,
        task: Task,
        satellite_by_id: dict[int, SatelliteView],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        isl_config: ISLConfig,
    ) -> Route | None:
        relay_energy_j = (
            transmission_energy_j(task.input_bits, isl_config)
            + transmission_energy_j(task.output_bits, isl_config)
        )
        blocked_relays = {
            sat_id
            for sat_id, sat in satellite_by_id.items()
            if not sat.sunlit
            and sat.battery_j
            - reserved_energy_for_sat(reserved_energy, sat_id)
            - relay_energy_j
            < battery.min_safe_j
        }
        parents = route_parents_avoiding_relays(
            isl_graph,
            source_sat,
            blocked_relays - {source_sat, target_sat},
        )
        return route_from_parents(parents, target_sat)


class _PhoenixSchedulerBase(Scheduler):
    """Shared PHOENIX helper logic without a public scheduler registration.

    The simulator has no ground-station model and no per-task scheduled start
    time, so Phoenix2 uses these bounded peer-selection helpers with its own
    batch-local scheduling state.
    """

    def __init__(self) -> None:
        self.plane_load_by_plane: dict[int, float] = {}
        # Backward-compatible alias for older tests/scripts.  The value is no
        # longer a task count; it is the compute/energy load assigned to a plane.
        self.task_count_by_plane = self.plane_load_by_plane

    def _candidate_finish_time(
        self,
        *,
        task: Task,
        route: Route,
        target: SatelliteView,
        time_s: int,
        reserved_available_time: dict[int, float],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[float, RouteTiming] | None:
        timing = estimate_route_timing(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        arrival_time = float(time_s) + timing.transmission_time_s
        start_time = max(arrival_time, reserved_available_time[target.sat_id])
        finish_time = start_time + timing.compute_time_s
        deadline_time = task.created_time_s + task.deadline_s
        if finish_time > deadline_time:
            return None
        return finish_time, timing

    @staticmethod
    def _plane_of(sat: SatelliteView) -> int | None:
        if sat.plane is None or sat.plane < 0:
            return None
        return sat.plane

    def _plane_sunlit_counts(
        self,
        satellite_views: list[SatelliteView],
    ) -> dict[int, int]:
        counts: dict[int, int] = {}
        for sat in satellite_views:
            plane = self._plane_of(sat)
            if plane is None:
                continue
            counts.setdefault(plane, 0)
            if sat.sunlit:
                counts[plane] += 1
        return counts

    def _candidate_cache(
        self,
        satellite_views: list[SatelliteView],
    ) -> PhoenixCandidateCache:
        sunlit_by_plane: dict[int, list[SatelliteView]] = {}
        sunlit_global: list[SatelliteView] = []
        sunlit_counts_by_plane: dict[int, int] = {}

        for sat in satellite_views:
            plane = self._plane_of(sat)
            if plane is not None:
                sunlit_counts_by_plane.setdefault(plane, 0)
            if not sat.sunlit:
                continue

            sunlit_global.append(sat)
            if plane is not None:
                sunlit_counts_by_plane[plane] += 1
                sunlit_by_plane.setdefault(plane, []).append(sat)

        def by_battery(candidates: list[SatelliteView]) -> tuple[SatelliteView, ...]:
            return tuple(
                sorted(
                    candidates,
                    key=lambda sat: (-sat.battery_j, sat.sat_id),
                )
            )

        return PhoenixCandidateCache(
            sunlit_by_plane={
                plane: by_battery(candidates)
                for plane, candidates in sunlit_by_plane.items()
            },
            sunlit_global=by_battery(sunlit_global),
            sunlit_counts_by_plane=sunlit_counts_by_plane,
        )

    def _target_plane(
        self,
        candidate_cache: PhoenixCandidateCache,
    ) -> int | None:
        sunlit_counts = candidate_cache.sunlit_counts_by_plane
        planes_with_sunlight = [
            plane for plane, sunlit_count in sunlit_counts.items() if sunlit_count > 0
        ]
        if not planes_with_sunlight:
            return None
        return min(
            planes_with_sunlight,
            key=lambda plane: (
                self.plane_load_by_plane.get(plane, 0.0) / max(1, sunlit_counts[plane]),
                self.plane_load_by_plane.get(plane, 0.0),
                plane,
            ),
        )

    def _candidate_energy_score(
        self,
        *,
        task: Task,
        route: Route,
        target: SatelliteView,
        finish_time: float,
        time_s: int,
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> float:
        """Estimate PHOENIX-style residual energy for a peer candidate.

        The paper uses future solar energy, current battery, and queued work.
        The simulator does not keep a full future sunlight matrix here, so use
        the cheap state we already have: current sunlight over this task window,
        current battery, already-reserved work, and this route's target energy.
        """

        route_cost = estimate_route_cost(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        window_s = max(0.0, finish_time - float(time_s))
        harvest_j = battery.harvest_w * window_s if target.sunlit else 0.0
        return min(
            battery.capacity_j,
            target.battery_j
            + harvest_j
            - reserved_energy_for_sat(reserved_energy, target.sat_id)
            - route_cost.energy_for(target.sat_id),
        )

    def _best_peer_in_planes(
        self,
        *,
        task: Task,
        source: SatelliteView,
        candidates: tuple[SatelliteView, ...],
        routes_by_target: dict[int, Route],
        satellite_views_by_id: dict[int, SatelliteView],
        time_s: int,
        step_s: int,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[Assignment, float] | None:
        best_assignment = None
        best_finish = None
        best_key = (float("inf"), float("inf"), float("inf"), float("inf"))

        for target in candidates:
            if target.sat_id == source.sat_id:
                continue

            route = routes_by_target.get(target.sat_id)
            if route is None:
                continue

            feasible = self._candidate_finish_time(
                task=task,
                route=route,
                target=target,
                time_s=time_s,
                reserved_available_time=reserved_available_time,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if feasible is None:
                continue
            finish_time, _timing = feasible
            route_cost = estimate_route_cost(
                task=task,
                route=route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if not route_respects_battery_projection(
                route=route,
                route_cost=route_cost,
                satellite_by_id=satellite_views_by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                continue
            energy_score = self._candidate_energy_score(
                task=task,
                route=route,
                target=target,
                finish_time=finish_time,
                time_s=time_s,
                reserved_energy=reserved_energy,
                battery=battery,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            key = (-energy_score, finish_time, route.hop_count, target.sat_id)

            if key < best_key:
                best_assignment = Assignment(
                    task_id=task.task_id,
                    route=route,
                    mode="offload",
                    score=energy_score,
                )
                best_finish = finish_time
                best_key = key

        if best_assignment is None:
            return None
        assert best_finish is not None
        return best_assignment, best_finish

    def _choose_peer(
        self,
        *,
        task: Task,
        source: SatelliteView,
        isl_graph: ISLGraph,
        routes_by_source: dict[int, dict[int, Route]],
        searched_targets_by_source: dict[int, set[int]],
        candidate_cache: PhoenixCandidateCache,
        satellite_views_by_id: dict[int, SatelliteView],
        time_s: int,
        step_s: int,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[Assignment, float] | None:
        target_plane = self._target_plane(candidate_cache)
        candidates = (
            candidate_cache.sunlit_global
            if target_plane is None
            else candidate_cache.sunlit_by_plane.get(target_plane, ())
        )
        if not candidates:
            return None

        routes_by_target = routes_by_source.setdefault(source.sat_id, {})
        searched_targets = searched_targets_by_source.setdefault(source.sat_id, set())
        candidate_target_ids = {target.sat_id for target in candidates}
        missing_target_ids = candidate_target_ids - searched_targets
        if missing_target_ids:
            routes_by_target.update(
                routes_to_targets(isl_graph, source.sat_id, missing_target_ids)
            )
            searched_targets.update(missing_target_ids)

        return self._best_peer_in_planes(
            task=task,
            source=source,
            candidates=candidates,
            routes_by_target=routes_by_target,
            satellite_views_by_id=satellite_views_by_id,
            time_s=time_s,
            step_s=step_s,
            reserved_available_time=reserved_available_time,
            reserved_energy=reserved_energy,
            battery=battery,
            compute_config=compute_config,
            isl_config=isl_config,
        )

    def _choose_local(
        self,
        *,
        task: Task,
        source: SatelliteView,
        isl_graph: ISLGraph,
        satellite_views_by_id: dict[int, SatelliteView],
        time_s: int,
        step_s: int,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        mode: str,
    ) -> tuple[Assignment, float] | None:
        route = route_or_raise(isl_graph, source.sat_id, source.sat_id)
        feasible = self._candidate_finish_time(
            task=task,
            route=route,
            target=source,
            time_s=time_s,
            reserved_available_time=reserved_available_time,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        if feasible is None:
            return None
        finish_time, _timing = feasible
        route_cost = estimate_route_cost(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        if not route_respects_battery_projection(
            route=route,
            route_cost=route_cost,
            satellite_by_id=satellite_views_by_id,
            reserved_available_time=reserved_available_time,
            reserved_energy=reserved_energy,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        ):
            return None
        return (
            Assignment(
                task_id=task.task_id,
                route=route,
                mode=mode,
                score=finish_time,
            ),
            finish_time,
        )

    def _remember_assignment_load(
        self,
        assignment: Assignment,
        by_id: dict[int, SatelliteView],
        load_j: float,
    ) -> None:
        target = by_id[assignment.target_sat]
        plane = self._plane_of(target)
        if plane is not None:
            self.plane_load_by_plane[plane] = (
                self.plane_load_by_plane.get(plane, 0.0) + load_j
            )


class Phoenix2Scheduler(_PhoenixSchedulerBase):
    """PHOENIX variant with bounded scheduling state.

    This keeps the current PHOENIX energy-aware peer scoring, but restores the
    f2fd36e state handling:

    * orbit-plane load is bounded to one assign_tasks() batch;
    * deferred local work reserves future source capacity in that batch.
    """

    name = "phoenix2"

    def _defer_time_if_deadline_safe_with_reservation(
        self,
        *,
        task: Task,
        source: SatelliteView,
        time_s: int,
        step_s: int,
        deferred_available_time: dict[int, float],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[float, float] | None:
        local_route = Route((source.sat_id,))
        timing = estimate_route_timing(
            task=task,
            route=local_route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        defer_until = source.next_sunlit_time_s
        if defer_until is None or defer_until <= float(time_s):
            defer_until = float(time_s + step_s)

        start_after_wait = max(
            defer_until,
            deferred_available_time[source.sat_id],
        )
        finish_after_wait = start_after_wait + timing.compute_time_s
        if finish_after_wait <= task.created_time_s + task.deadline_s:
            return defer_until, finish_after_wait
        return None

    def assign_tasks(
        self,
        *,
        tasks,
        satellite_views,
        time_s,
        step_s,
        battery,
        compute_config,
        task_config,
        isl_config,
        isl_graph,
        scheduler_config,
    ):
        by_id = {sat.sat_id: sat for sat in satellite_views}
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }
        deferred_available_time = dict(reserved_available_time)
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )
        candidate_cache = self._candidate_cache(satellite_views)

        # PHOENIX's orbit-level load is a scheduling-horizon signal, not a
        # lifetime counter.  Use one assign_tasks() batch as the horizon and
        # expose the last batch for diagnostics.
        self.plane_load_by_plane = {}
        self.task_count_by_plane = self.plane_load_by_plane

        assignments = []
        routes_by_source: dict[int, dict[int, Route]] = {}
        searched_targets_by_source: dict[int, set[int]] = {}

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]

            chosen = None
            deferred = None

            if source.sunlit:
                chosen = self._choose_local(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    satellite_views_by_id=by_id,
                    time_s=time_s,
                    step_s=step_s,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                    mode="local",
                )
            else:
                deferred = self._defer_time_if_deadline_safe_with_reservation(
                    task=task,
                    source=source,
                    time_s=time_s,
                    step_s=step_s,
                    deferred_available_time=deferred_available_time,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )

            if not source.sunlit and deferred is not None:
                defer_until, deferred_finish_time = deferred
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="defer",
                        score=defer_until,
                    )
                )
                deferred_available_time[source.sat_id] = deferred_finish_time
                continue

            if chosen is None:
                chosen = self._choose_peer(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    routes_by_source=routes_by_source,
                    searched_targets_by_source=searched_targets_by_source,
                    candidate_cache=candidate_cache,
                    satellite_views_by_id=by_id,
                    time_s=time_s,
                    step_s=step_s,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )

            if chosen is None and not source.sunlit:
                chosen = self._choose_local(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    satellite_views_by_id=by_id,
                    time_s=time_s,
                    step_s=step_s,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                    mode="local",
                )

            if chosen is None:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="fail",
                        score=float("inf"),
                        failed_reason="no_feasible_candidate",
                    )
                )
                continue

            assignment, finish_time = chosen
            cost = estimate_route_cost(
                task=task,
                route=assignment.route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if not route_respects_battery_projection(
                route=assignment.route,
                route_cost=cost,
                satellite_by_id=by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=assignment.route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="battery_hard_constraint",
                    )
                )
                continue

            assignments.append(assignment)
            reserved_available_time[assignment.target_sat] = finish_time
            reserve_route_transmission_energy(
                route=assignment.route,
                route_cost=cost,
                compute_config=compute_config,
                reserved_energy=reserved_energy,
            )
            self._remember_assignment_load(
                assignment,
                by_id,
                load_j=cost.energy_for(assignment.target_sat),
            )

        return assignments


def create_scheduler(name: str) -> Scheduler:
    if name == LocalOnlyScheduler.name:
        return LocalOnlyScheduler()
    if name == NearestSunlitScheduler.name:
        return NearestSunlitScheduler()
    if name == GreedyEnergyScheduler.name:
        return GreedyEnergyScheduler()
    if name == Method3Scheduler.name:
        return Method3Scheduler()
    if name == Method3ModScheduler.name:
        return Method3ModScheduler()
    if name == Method5Scheduler.name:
        return Method5Scheduler()
    if name == Method6Scheduler.name:
        return Method6Scheduler()
    if name == Method7Scheduler.name:
        return Method7Scheduler()
    if name == Method8Scheduler.name:
        return Method8Scheduler()
    if name in {Phoenix2Scheduler.name, "phoenix"}:
        return Phoenix2Scheduler()
    raise ValueError(f"unknown scheduler: {name}")
