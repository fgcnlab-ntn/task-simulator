from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .battery import projected_battery_after_step
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
    RouteTiming,
    compute_cycles,
    estimate_route_cost,
    estimate_route_timing,
    task_compute_time_s,
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
        satellite_by_id = {sat.sat_id: sat for sat in satellite_views}
        sunlit_targets = tuple(sat for sat in satellite_views if sat.sunlit)
        assignment_by_source: dict[int, Assignment] = {}
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
            assignments.append(
                Assignment(
                    task_id=task.task_id,
                    route=template.route,
                    mode=template.mode,
                    score=template.score,
                    failed_reason=template.failed_reason,
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

            chosen = min(
                candidates,
                key=lambda candidate: (
                    candidate.battery_cost_j,
                    candidate.finish_time_s,
                    candidate.energy_j,
                    candidate.assignment.hop_count,
                    candidate.assignment.target_sat,
                ),
            )
            assignments.append(chosen.assignment)
            reserved_available_time[chosen.assignment.target_sat] = chosen.finish_time_s
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
                local_cost, local_finish = local_result

            # Action 2: least-loaded sunlit
            sun_cost = float("inf")
            sun_finish = None
            sun_sat_id = None
            sun_route = None

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
                        sun_cost, sun_finish = sun_result
                        sun_sat_id = candidate_sat_id
                        sun_route = Route(tuple(reversed(reversed_route_nodes)))

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
    name = "method3_mod"

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

        # Min-heap of current sunlit loads.
        # Unlike the original method3 behavior, this method restores the
        # peeked candidate if it is not actually selected for sunlit execution.
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
                local_cost, local_finish = local_result

            # Action 2: least-loaded sunlit
            sun_cost = float("inf")
            sun_finish = None
            sun_sat_id = None
            sun_route = None

            # This records the satellite popped by _peek_least_loaded_sunlit().
            # If the final selected action is not sunlit execution, we must
            # push it back to avoid shrinking the candidate heap.
            peeked_sun_sat_id = None

            best_sunlit = self._peek_least_loaded_sunlit(
                sunlit_heap=sunlit_heap,
                reserved_available_time=reserved_available_time,
                satellite_by_id=by_id,
                time_s=time_s,
                exclude_sat_id=source.sat_id,
            )

            if best_sunlit is not None:
                candidate_sat_id, candidate_available_time = best_sunlit
                peeked_sun_sat_id = candidate_sat_id

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
                        sun_cost, sun_finish = sun_result
                        sun_sat_id = candidate_sat_id
                        sun_route = Route(tuple(reversed(reversed_route_nodes)))

            # Action 3: defer
            defer_cost = self._defer_cost(
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
            )

            action, _ = min(
                [
                    ("local", local_cost),
                    ("sunlit", sun_cost),
                    ("defer", defer_cost),
                ],
                key=lambda x: x[1],
            )

            def restore_peeked_sunlit_candidate() -> None:
                if peeked_sun_sat_id is None:
                    return

                heapq.heappush(
                    sunlit_heap,
                    (
                        max(
                            0.0,
                            reserved_available_time[peeked_sun_sat_id] - float(time_s),
                        ),
                        peeked_sun_sat_id,
                    ),
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

                if source.sunlit:
                    heapq.heappush(
                        sunlit_heap,
                        (
                            max(0.0, local_finish - float(time_s)),
                            source.sat_id,
                        ),
                    )

                # The sunlit candidate was only peeked, not selected.
                restore_peeked_sunlit_candidate()

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

                # Here we do not restore the old peeked heap entry.
                # Instead, we push the updated load of the actually selected
                # sunlit satellite.
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

                # The sunlit candidate was only peeked, not selected.
                restore_peeked_sunlit_candidate()

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

                # The sunlit candidate was only peeked, not selected.
                restore_peeked_sunlit_candidate()

        return assignments


class Method5Scheduler(Method3Scheduler):
    """
    Method5: fairness-gated predictive sunlit dispatching.

    Main idea:
    1. Fold queue load into predicted post-completion energy.
    2. Normally compare local execution and one safest current sunlit target.
    3. Allow deferral only if the source satellite's defer count is not above
       the active-source average defer count.
    4. If deferral is blocked, search sunlit targets in safety order and stop
       at the first deadline-feasible safe target.

    This avoids a fixed top-K parameter and prevents deferral from piling up on
    a small number of source satellites.
    """

    name = "method5"

    def _sunlit_at_time(
        self,
        *,
        sat: SatelliteView,
        query_time_s: float,
        time_s: int,
    ) -> bool:
        now = float(time_s)

        if sat.sunlit:
            next_eclipse = getattr(sat, "next_eclipse_time_s", None)
            if next_eclipse is not None and next_eclipse > now:
                return query_time_s < float(next_eclipse)
            return True

        next_sunlit = getattr(sat, "next_sunlit_time_s", None)
        if next_sunlit is not None and next_sunlit > now:
            return query_time_s >= float(next_sunlit)

        return False

    def _predict_post_completion_energy(
        self,
        *,
        sat: SatelliteView,
        available_time_s: float,
        time_s: int,
        step_s: int,
        compute_time_s: float,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
    ) -> tuple[float, float]:
        now = float(time_s)
        start_time = max(now, available_time_s)
        finish_time = start_time + compute_time_s

        duration_s = max(finish_time - now, 0.0)
        if duration_s <= 0.0:
            return float(sat.battery_j), finish_time

        mid_time = now + 0.5 * duration_s

        sunlit = self._sunlit_at_time(
            sat=sat,
            query_time_s=mid_time,
            time_s=time_s,
        )

        # The existing reserved queue plus this candidate task are approximated
        # as active compute over the busy window. This folds queue load into
        # energy prediction without per-step simulation.
        task_energy_j = duration_s * compute_config.cpu_power_w

        energy_j = projected_battery_after_step(
            battery_now=sat.battery_j,
            sunlit=sunlit,
            step_s=duration_s,
            battery=battery,
            task_energy_j=task_energy_j,
            update=time_s > 0,
        )

        return energy_j, finish_time

    def _safety_margin(
        self,
        *,
        energy_j: float,
        battery: BatteryConfig,
    ) -> float:
        denom = max(battery.capacity_j - battery.min_safe_j, 1e-6)
        return (energy_j - battery.min_safe_j) / denom

    def _baseline_sunlit_margin(
        self,
        *,
        sat: SatelliteView,
        reserved_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
    ) -> float:
        energy_j, _ = self._predict_post_completion_energy(
            sat=sat,
            available_time_s=reserved_available_time[sat.sat_id],
            time_s=time_s,
            step_s=step_s,
            compute_time_s=0.0,
            compute_config=compute_config,
            battery=battery,
        )
        return self._safety_margin(energy_j=energy_j, battery=battery)

    def _rebuild_sunlit_heap(
        self,
        *,
        sunlit_heap: list[tuple[float, float, int]],
        satellite_views: list[SatelliteView],
        reserved_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
    ) -> None:
        import heapq

        sunlit_heap.clear()

        for sat in satellite_views:
            if not sat.sunlit:
                continue

            margin = self._baseline_sunlit_margin(
                sat=sat,
                reserved_available_time=reserved_available_time,
                time_s=time_s,
                step_s=step_s,
                compute_config=compute_config,
                battery=battery,
            )

            heapq.heappush(
                sunlit_heap,
                (
                    -margin,
                    reserved_available_time[sat.sat_id],
                    sat.sat_id,
                ),
            )

    def _peek_safest_sunlit_from_heap(
        self,
        *,
        sunlit_heap: list[tuple[float, float, int]],
        by_id: dict[int, SatelliteView],
        reserved_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
        exclude_sat_id: int,
    ) -> int | None:
        import heapq

        skipped: list[tuple[float, float, int]] = []

        while sunlit_heap:
            neg_margin, recorded_available_time, sat_id = heapq.heappop(sunlit_heap)

            if sat_id == exclude_sat_id:
                skipped.append((neg_margin, recorded_available_time, sat_id))
                continue

            sat = by_id.get(sat_id)
            if sat is None or not sat.sunlit:
                continue

            current_available_time = reserved_available_time[sat_id]
            current_margin = self._baseline_sunlit_margin(
                sat=sat,
                reserved_available_time=reserved_available_time,
                time_s=time_s,
                step_s=step_s,
                compute_config=compute_config,
                battery=battery,
            )

            current_entry = (
                -current_margin,
                current_available_time,
                sat_id,
            )

            # Stale heap entry. Replace it with the current value.
            if (
                abs(recorded_available_time - current_available_time) > 1e-9
                or abs(neg_margin - current_entry[0]) > 1e-9
            ):
                heapq.heappush(sunlit_heap, current_entry)
                continue

            # True peek: restore the selected item before returning.
            heapq.heappush(sunlit_heap, current_entry)

            for item in skipped:
                heapq.heappush(sunlit_heap, item)

            return sat_id

        for item in skipped:
            heapq.heappush(sunlit_heap, item)

        return None

    def _make_local_candidate(
        self,
        *,
        source: SatelliteView,
        reserved_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
    ) -> dict | None:
        energy_j, finish_time = self._predict_post_completion_energy(
            sat=source,
            available_time_s=reserved_available_time[source.sat_id],
            time_s=time_s,
            step_s=step_s,
            compute_time_s=compute_time_s,
            compute_config=compute_config,
            battery=battery,
        )

        if finish_time > deadline_time:
            return None

        if energy_j < battery.min_safe_j:
            return None

        margin = self._safety_margin(
            energy_j=energy_j,
            battery=battery,
        )

        return {
            "mode": "local",
            "sat_id": source.sat_id,
            "route": Route((source.sat_id,)),
            "finish_time": finish_time,
            "energy_j": energy_j,
            "margin": margin,
            "score": -margin,
        }

    def _make_sunlit_candidate(
        self,
        *,
        sat: SatelliteView,
        source: SatelliteView,
        route_parents: dict[int, int | None],
        reserved_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
    ) -> dict | None:
        if not sat.sunlit:
            return None

        if sat.sat_id == source.sat_id:
            return None

        reversed_route_nodes = reversed_route_nodes_from_parents(
            route_parents,
            sat.sat_id,
        )
        if reversed_route_nodes is None:
            return None

        route = Route(tuple(reversed(reversed_route_nodes)))

        energy_j, finish_time = self._predict_post_completion_energy(
            sat=sat,
            available_time_s=reserved_available_time[sat.sat_id],
            time_s=time_s,
            step_s=step_s,
            compute_time_s=compute_time_s,
            compute_config=compute_config,
            battery=battery,
        )

        margin = self._safety_margin(
            energy_j=energy_j,
            battery=battery,
        )

        return {
            "mode": "offload",
            "sat_id": sat.sat_id,
            "route": route,
            "finish_time": finish_time,
            "energy_j": energy_j,
            "margin": margin,
            "deadline_feasible": finish_time <= deadline_time,
            "energy_safe": energy_j >= battery.min_safe_j,
            "score": -margin,
        }

    def _best_current_execution(
        self,
        *,
        local_candidate: dict | None,
        sunlit_candidate: dict | None,
    ) -> dict | None:
        feasible: list[dict] = []

        if local_candidate is not None:
            feasible.append(local_candidate)

        if (
            sunlit_candidate is not None
            and sunlit_candidate["deadline_feasible"]
            and sunlit_candidate["energy_safe"]
        ):
            feasible.append(sunlit_candidate)

        if not feasible:
            return None

        return max(
            feasible,
            key=lambda item: (
                item["margin"],
                -item["finish_time"],
            ),
        )

    def _defer_allowed(
        self,
        *,
        source_sat_id: int,
        defer_count_by_sat: dict[int, int],
        active_source_ids: set[int],
    ) -> bool:
        if not active_source_ids:
            return True

        total_defer = sum(
            defer_count_by_sat.get(sat_id, 0) for sat_id in active_source_ids
        )
        num_active_sources = len(active_source_ids)

        source_defer = defer_count_by_sat.get(source_sat_id, 0)

        # Equivalent to:
        # source_defer <= ceil(total_defer / num_active_sources)
        #
        # Written in integer form to avoid importing math.
        return source_defer * num_active_sources <= total_defer + num_active_sources - 1

    def _defer_deadline_feasible(
        self,
        *,
        source_sat_id: int,
        deferred_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
    ) -> tuple[bool, float]:
        start_time = max(
            float(time_s + step_s),
            deferred_available_time[source_sat_id],
        )
        finish_time = start_time + compute_time_s

        return finish_time <= deadline_time, finish_time

    def _fallback_sunlit_search(
        self,
        *,
        source: SatelliteView,
        by_id: dict[int, SatelliteView],
        sunlit_heap: list[tuple[float, float, int]],
        route_parents: dict[int, int | None],
        reserved_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
    ) -> dict | None:
        import heapq

        popped: list[tuple[float, float, int]] = []
        checked_sat_ids: set[int] = set()

        while sunlit_heap:
            neg_margin, recorded_available_time, sat_id = heapq.heappop(sunlit_heap)

            if sat_id in checked_sat_ids:
                continue

            if sat_id == source.sat_id:
                popped.append((neg_margin, recorded_available_time, sat_id))
                checked_sat_ids.add(sat_id)
                continue

            sat = by_id.get(sat_id)
            if sat is None or not sat.sunlit:
                checked_sat_ids.add(sat_id)
                continue

            current_available_time = reserved_available_time[sat_id]
            current_margin = self._baseline_sunlit_margin(
                sat=sat,
                reserved_available_time=reserved_available_time,
                time_s=time_s,
                step_s=step_s,
                compute_config=compute_config,
                battery=battery,
            )

            current_entry = (
                -current_margin,
                current_available_time,
                sat_id,
            )

            # Stale entry. Push the updated value and keep searching.
            if (
                abs(recorded_available_time - current_available_time) > 1e-9
                or abs(neg_margin - current_entry[0]) > 1e-9
            ):
                heapq.heappush(sunlit_heap, current_entry)
                continue

            checked_sat_ids.add(sat_id)

            candidate = self._make_sunlit_candidate(
                sat=sat,
                source=source,
                route_parents=route_parents,
                reserved_available_time=reserved_available_time,
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_config=compute_config,
                battery=battery,
            )

            if (
                candidate is not None
                and candidate["deadline_feasible"]
                and candidate["energy_safe"]
            ):
                # Restore previously popped candidates, but do not restore the
                # selected candidate. The caller will push its updated state
                # after assignment.
                for item in popped:
                    heapq.heappush(sunlit_heap, item)

                return candidate

            # Candidate was examined but not selected, so keep it available for
            # future tasks.
            popped.append(current_entry)

        for item in popped:
            heapq.heappush(sunlit_heap, item)

        return None

    def _push_updated_sunlit_state(
        self,
        *,
        sat_id: int,
        by_id: dict[int, SatelliteView],
        sunlit_heap: list[tuple[float, float, int]],
        reserved_available_time: dict[int, float],
        time_s: int,
        step_s: int,
        compute_config: ComputeConfig,
        battery: BatteryConfig,
    ) -> None:
        import heapq

        sat = by_id[sat_id]
        if not sat.sunlit:
            return

        margin = self._baseline_sunlit_margin(
            sat=sat,
            reserved_available_time=reserved_available_time,
            time_s=time_s,
            step_s=step_s,
            compute_config=compute_config,
            battery=battery,
        )

        heapq.heappush(
            sunlit_heap,
            (
                -margin,
                reserved_available_time[sat_id],
                sat_id,
            ),
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
        by_id = {sat.sat_id: sat for sat in satellite_views}

        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }

        deferred_available_time = dict(reserved_available_time)

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        unique_sources = {
            task.source_sat for task in ordered_tasks if task.source_sat is not None
        }

        active_source_ids: set[int] = set(unique_sources)

        route_parents_by_source: dict[int, dict[int, int | None]] = {
            source_sat: route_parents_from_source(isl_graph, source_sat)
            for source_sat in unique_sources
        }

        # Track defer count only over active source satellites. This avoids
        # diluting the average by satellites that have no tasks in this batch.
        defer_count_by_sat: dict[int, int] = {sat_id: 0 for sat_id in active_source_ids}

        sunlit_heap: list[tuple[float, float, int]] = []
        self._rebuild_sunlit_heap(
            sunlit_heap=sunlit_heap,
            satellite_views=satellite_views,
            reserved_available_time=reserved_available_time,
            time_s=time_s,
            step_s=step_s,
            compute_config=compute_config,
            battery=battery,
        )

        num_sunlit = sum(1 for sat in satellite_views if sat.sunlit)

        assignments: list[Assignment] = []

        for task in ordered_tasks:
            assert task.source_sat is not None

            source = by_id[task.source_sat]
            deadline_time = task.created_time_s + task.deadline_s
            compute_time_s = task_compute_time_s(task, compute_config)

            route_parents = route_parents_by_source[source.sat_id]

            # Candidate 1: local execution.
            local_candidate = self._make_local_candidate(
                source=source,
                reserved_available_time=reserved_available_time,
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_config=compute_config,
                battery=battery,
            )

            # Candidate 2: one safest currently sunlit target from the heap.
            sunlit_candidate = None
            safest_sunlit_id = self._peek_safest_sunlit_from_heap(
                sunlit_heap=sunlit_heap,
                by_id=by_id,
                reserved_available_time=reserved_available_time,
                time_s=time_s,
                step_s=step_s,
                compute_config=compute_config,
                battery=battery,
                exclude_sat_id=source.sat_id,
            )

            if safest_sunlit_id is not None:
                sunlit_candidate = self._make_sunlit_candidate(
                    sat=by_id[safest_sunlit_id],
                    source=source,
                    route_parents=route_parents,
                    reserved_available_time=reserved_available_time,
                    time_s=time_s,
                    step_s=step_s,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                    compute_config=compute_config,
                    battery=battery,
                )

            chosen = self._best_current_execution(
                local_candidate=local_candidate,
                sunlit_candidate=sunlit_candidate,
            )

            if chosen is not None:
                mode = "local" if chosen["mode"] == "local" else "offload"

                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=chosen["route"],
                        mode=mode,
                        score=chosen["score"],
                    )
                )

                reserved_available_time[chosen["sat_id"]] = chosen["finish_time"]

                self._push_updated_sunlit_state(
                    sat_id=chosen["sat_id"],
                    by_id=by_id,
                    sunlit_heap=sunlit_heap,
                    reserved_available_time=reserved_available_time,
                    time_s=time_s,
                    step_s=step_s,
                    compute_config=compute_config,
                    battery=battery,
                )

                if num_sunlit > 0 and len(sunlit_heap) > 3 * num_sunlit:
                    self._rebuild_sunlit_heap(
                        sunlit_heap=sunlit_heap,
                        satellite_views=satellite_views,
                        reserved_available_time=reserved_available_time,
                        time_s=time_s,
                        step_s=step_s,
                        compute_config=compute_config,
                        battery=battery,
                    )

                continue

            # If immediate execution is not feasible, check fair deferral.
            defer_ok = self._defer_allowed(
                source_sat_id=source.sat_id,
                defer_count_by_sat=defer_count_by_sat,
                active_source_ids=active_source_ids,
            )

            defer_deadline_ok, defer_finish = self._defer_deadline_feasible(
                source_sat_id=source.sat_id,
                deferred_available_time=deferred_available_time,
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
            )

            if defer_ok and defer_deadline_ok:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=defer_finish,
                    )
                )

                defer_count_by_sat[source.sat_id] = (
                    defer_count_by_sat.get(source.sat_id, 0) + 1
                )
                deferred_available_time[source.sat_id] = defer_finish
                continue

            # If deferral is blocked, search sunlit targets lazily in safety
            # order and stop at the first feasible safe target.
            fallback = self._fallback_sunlit_search(
                source=source,
                by_id=by_id,
                sunlit_heap=sunlit_heap,
                route_parents=route_parents,
                reserved_available_time=reserved_available_time,
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_config=compute_config,
                battery=battery,
            )

            if fallback is not None:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=fallback["route"],
                        mode="offload",
                        score=fallback["score"],
                    )
                )

                reserved_available_time[fallback["sat_id"]] = fallback["finish_time"]

                self._push_updated_sunlit_state(
                    sat_id=fallback["sat_id"],
                    by_id=by_id,
                    sunlit_heap=sunlit_heap,
                    reserved_available_time=reserved_available_time,
                    time_s=time_s,
                    step_s=step_s,
                    compute_config=compute_config,
                    battery=battery,
                )

                if num_sunlit > 0 and len(sunlit_heap) > 3 * num_sunlit:
                    self._rebuild_sunlit_heap(
                        sunlit_heap=sunlit_heap,
                        satellite_views=satellite_views,
                        reserved_available_time=reserved_available_time,
                        time_s=time_s,
                        step_s=step_s,
                        compute_config=compute_config,
                        battery=battery,
                    )

                continue

            assignments.append(
                Assignment(
                    task_id=task.task_id,
                    route=Route((source.sat_id,)),
                    mode="fail",
                    score=float("inf"),
                    failed_reason="no_safe_execution_or_fair_defer",
                )
            )

        return assignments


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
            - reserved_energy[target.sat_id]
            - route_cost.energy_for(target.sat_id),
        )

    def _best_peer_in_planes(
        self,
        *,
        task: Task,
        source: SatelliteView,
        candidates: tuple[SatelliteView, ...],
        routes_by_target: dict[int, Route],
        time_s: int,
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
        time_s: int,
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
            time_s=time_s,
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
        time_s: int,
        reserved_available_time: dict[int, float],
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
        finish_time, _cost = feasible
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
        reserved_energy = {
            sat.sat_id: sat.queue_backlog_s * compute_config.cpu_power_w
            for sat in satellite_views
        }
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
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
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
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )

            if chosen is None:
                chosen = self._choose_local(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
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
            assignments.append(assignment)
            reserved_available_time[assignment.target_sat] = finish_time
            cost = estimate_route_cost(
                task=task,
                route=assignment.route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if assignment.mode == "offload":
                for sat_id, energy_j in cost.energy_by_sat.items():
                    reserved_energy[sat_id] += energy_j
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
    if name in {Phoenix2Scheduler.name, "phoenix"}:
        return Phoenix2Scheduler()
    raise ValueError(f"unknown scheduler: {name}")
