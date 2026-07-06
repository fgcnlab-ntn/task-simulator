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
from .route_cost import RouteCost, estimate_route_cost


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


def routes_from_source(graph: ISLGraph, source_sat: int) -> dict[int, Route]:
    """Return shortest routes from one source to every reachable satellite."""

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

    routes: dict[int, Route] = {}
    for target_sat in parents:
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
        assert task.source_sat is not None
        return Assignment(
            task_id=task.task_id,
            route=route_or_raise(isl_graph, task.source_sat, task.source_sat),
            mode=self.name,
        )


class NearestSunlitScheduler(Scheduler):
    name = "nearest-sunlit"

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        assert task.source_sat is not None
        by_id = {sat.sat_id: sat for sat in satellite_views}
        source = by_id[task.source_sat]
        target = source
        mode = "local"
        route = route_or_raise(isl_graph, source.sat_id, source.sat_id)
        if not source.sunlit:
            reachable_sunlit_targets = [
                sat
                for sat in satellite_views
                if sat.sunlit
                and shortest_route(isl_graph, source.sat_id, sat.sat_id) is not None
            ]
            if reachable_sunlit_targets:
                target = min(
                    reachable_sunlit_targets,
                    key=lambda sat: distance_km(source, sat),
                )
                route = route_or_raise(isl_graph, source.sat_id, target.sat_id)
                mode = "offload"
        return Assignment(
            task_id=task.task_id,
            route=route,
            mode=mode,
        )


class Method1Scheduler(Scheduler):
    name = "method1"

    def _estimate_unsafe_and_margin_risk(
        self,
        *,
        cost,
        satellite_views,
        by_id,
        reserved_energy,
        battery,
        step_s,
        time_s,
    ) -> tuple[int, float]:
        unsafe_increase = 0
        margin_risk = 0.0
        eps_j = 1.0

        for sat_id in cost.energy_by_sat:
            sat = by_id[sat_id]
            if sat.sunlit:
                continue

            before_unsafe = sat.battery_j < battery.min_safe_j

            projected = projected_battery_after_step(
                battery_now=sat.battery_j,
                sunlit=sat.sunlit,
                step_s=step_s,
                battery=battery,
                task_energy_j=reserved_energy[sat_id] + cost.energy_for(sat_id),
                update=time_s > 0,
            )

            after_unsafe = projected < battery.min_safe_j
            unsafe_increase += int(after_unsafe) - int(before_unsafe)

            margin_j = max(projected - battery.min_safe_j, eps_j)
            margin_risk += 1.0 / margin_j

        return max(unsafe_increase, 0), margin_risk

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

        reserved_energy = {sat.sat_id: 0.0 for sat in satellite_views}
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        assignments: list[Assignment] = []

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]

            best_candidate = None
            best_key = (float("inf"), float("inf"), float("inf"), float("inf"))
            best_cost = None
            best_finish = None

            for target in satellite_views:
                route = shortest_route(isl_graph, source.sat_id, target.sat_id)
                if route is None:
                    continue

                cost = estimate_route_cost(
                    task=task,
                    route=route,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )

                arrival_time = float(time_s) + cost.transmission_time_s
                z_q = max(arrival_time, reserved_available_time[target.sat_id])
                t_fin = z_q + cost.compute_time_s
                deadline_time = task.created_time_s + task.deadline_s

                if t_fin > deadline_time:
                    continue

                U, R = self._estimate_unsafe_and_margin_risk(
                    cost=cost,
                    satellite_views=satellite_views,
                    by_id=by_id,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    step_s=step_s,
                    time_s=time_s,
                )

                key = (U, R, t_fin, route.hop_count)

                if key < best_key:
                    mode = "local" if target.sat_id == source.sat_id else "offload"
                    best_candidate = Assignment(
                        task_id=task.task_id,
                        route=route,
                        mode=mode,
                        score=float(U),
                    )
                    best_key = key
                    best_cost = cost
                    best_finish = t_fin

            if best_candidate is not None:
                assignments.append(best_candidate)

                assert best_cost is not None
                assert best_finish is not None

                reserved_available_time[best_candidate.route.target_sat] = best_finish

                for sat_id, energy_j in best_cost.energy_by_sat.items():
                    reserved_energy[sat_id] += energy_j
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="defer",
                        score=float("inf"),
                    )
                )

        return assignments


class PhoenixLiteScheduler(Scheduler):
    """PHOENIX-inspired scheduler without ground-station support.

    This is deliberately not the full PHOENIX paper algorithm.  The simulator
    has no ground-station model and no per-task scheduled start time, so the
    only honest approximation is:

    * prefer local execution when the source is sunlit;
    * defer eclipse-side local work by one simulator step when that can still
      meet the task deadline;
    * otherwise offload to a reachable sunlit peer, balancing by orbit plane
      and then choosing the highest-battery feasible satellite.
    """

    name = "phoenix"
    peer_candidate_limit = 4

    def __init__(self) -> None:
        self.task_count_by_plane: dict[int, int] = {}

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        assert task.source_sat is not None
        by_id = {sat.sat_id: sat for sat in satellite_views}
        source = by_id[task.source_sat]
        if source.sunlit:
            return Assignment(
                task_id=task.task_id,
                route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                mode="local",
            )
        return NearestSunlitScheduler().assign_task(
            task=task,
            satellite_views=satellite_views,
            isl_graph=isl_graph,
        )

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
    ) -> tuple[float, RouteCost] | None:
        cost = estimate_route_cost(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        arrival_time = float(time_s) + cost.transmission_time_s
        start_time = max(arrival_time, reserved_available_time[target.sat_id])
        finish_time = start_time + cost.compute_time_s
        deadline_time = task.created_time_s + task.deadline_s
        if finish_time > deadline_time:
            return None
        return finish_time, cost

    def _defer_is_deadline_safe(
        self,
        *,
        task: Task,
        source: SatelliteView,
        time_s: int,
        step_s: int,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> bool:
        local_route = Route((source.sat_id,))
        cost = estimate_route_cost(
            task=task,
            route=local_route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        finish_after_one_step_wait = (
            float(time_s) + step_s + source.queue_backlog_s + cost.compute_time_s
        )
        return finish_after_one_step_wait <= task.created_time_s + task.deadline_s

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
                )[: self.peer_candidate_limit]
            )

        return PhoenixCandidateCache(
            sunlit_by_plane={
                plane: by_battery(candidates)
                for plane, candidates in sunlit_by_plane.items()
            },
            sunlit_global=by_battery(sunlit_global),
            sunlit_counts_by_plane=sunlit_counts_by_plane,
        )

    def _preferred_planes(
        self,
        candidate_cache: PhoenixCandidateCache,
    ) -> list[int]:
        sunlit_counts = candidate_cache.sunlit_counts_by_plane
        return sorted(
            sunlit_counts,
            key=lambda plane: (
                self.task_count_by_plane.get(plane, 0) / max(1, sunlit_counts[plane]),
                self.task_count_by_plane.get(plane, 0),
                plane,
            ),
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
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[Assignment, float] | None:
        best_assignment = None
        best_finish = None
        best_key = None

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
            finish_time, _cost = feasible

            key = (
                -target.battery_j,
                finish_time,
                route.hop_count,
                target.sat_id,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_assignment = Assignment(
                    task_id=task.task_id,
                    route=route,
                    mode="offload",
                    score=finish_time,
                )
                best_finish = finish_time

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
        candidate_cache: PhoenixCandidateCache,
        time_s: int,
        reserved_available_time: dict[int, float],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[Assignment, float] | None:
        preferred_planes = self._preferred_planes(candidate_cache)
        routes_by_target = routes_by_source.get(source.sat_id)
        if routes_by_target is None:
            routes_by_target = routes_from_source(isl_graph, source.sat_id)
            routes_by_source[source.sat_id] = routes_by_target

        for plane in preferred_planes:
            best = self._best_peer_in_planes(
                task=task,
                source=source,
                candidates=candidate_cache.sunlit_by_plane.get(plane, ()),
                routes_by_target=routes_by_target,
                time_s=time_s,
                reserved_available_time=reserved_available_time,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if best is not None:
                return best

        return self._best_peer_in_planes(
            task=task,
            source=source,
            candidates=candidate_cache.sunlit_global,
            routes_by_target=routes_by_target,
            time_s=time_s,
            reserved_available_time=reserved_available_time,
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

    def _remember_assignment(
        self,
        assignment: Assignment,
        by_id: dict[int, SatelliteView],
    ) -> None:
        target = by_id[assignment.target_sat]
        plane = self._plane_of(target)
        if plane is not None:
            self.task_count_by_plane[plane] = self.task_count_by_plane.get(plane, 0) + 1

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
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )
        candidate_cache = self._candidate_cache(satellite_views)

        assignments = []
        routes_by_source: dict[int, dict[int, Route]] = {}

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]

            chosen = None

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
            elif self._defer_is_deadline_safe(
                task=task,
                source=source,
                time_s=time_s,
                step_s=step_s,
                compute_config=compute_config,
                isl_config=isl_config,
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="defer",
                        score=float(time_s + step_s),
                    )
                )
                continue

            if chosen is None:
                chosen = self._choose_peer(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    routes_by_source=routes_by_source,
                    candidate_cache=candidate_cache,
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
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
            self._remember_assignment(assignment, by_id)

        return assignments


def create_scheduler(name: str) -> Scheduler:
    if name == LocalOnlyScheduler.name:
        return LocalOnlyScheduler()
    if name == NearestSunlitScheduler.name:
        return NearestSunlitScheduler()
    if name == Method1Scheduler.name:
        return Method1Scheduler()
    if name == PhoenixLiteScheduler.name:
        return PhoenixLiteScheduler()
    raise ValueError(f"unknown scheduler: {name}")
