from __future__ import annotations

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
from .route_cost import estimate_route_cost


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

    def _estimate_unsafe_increase(
        self,
        *,
        route,
        cost,
        satellite_views,
        by_id,
        reserved_energy,
        battery,
        step_s,
        time_s,
    ) -> int:
        affected = set(cost.energy_by_sat.keys())
        increase = 0

        for sat_id in affected:
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
            increase += int(after_unsafe) - int(before_unsafe)

        return max(increase, 0)

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

        reserved_energy = {sat.sat_id: 0.0 for sat in satellite_views}
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        assignments = []

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]

            best_candidate = None
            best_key = (float("inf"), float("inf"), float("inf"))
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

                # deadline infeasible -> discard this candidate
                if t_fin > deadline_time:
                    continue

                U = self._estimate_unsafe_increase(
                    route=route,
                    cost=cost,
                    satellite_views=satellite_views,
                    by_id=by_id,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    step_s=step_s,
                    time_s=time_s,
                )

                key = (U, t_fin, route.hop_count)

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

                reserved_available_time[best_candidate.target_sat] = best_finish

                for sat_id, energy_j in best_cost.energy_by_sat.items():
                    reserved_energy[sat_id] += energy_j

            else:
                remaining_deadline = task.created_time_s + task.deadline_s - time_s
                if remaining_deadline > step_s:
                    assignments.append(
                        Assignment(
                            task_id=task.task_id,
                            route=route_or_raise(
                                isl_graph, source.sat_id, source.sat_id
                            ),
                            mode="defer",
                            score=float("inf"),
                        )
                    )
                else:
                    assignments.append(
                        Assignment(
                            task_id=task.task_id,
                            route=route_or_raise(
                                isl_graph, source.sat_id, source.sat_id
                            ),
                            mode="fail",
                            score=float("inf"),
                            failed_reason="no_feasible_candidate",
                        )
                    )

        return assignments


def create_scheduler(name: str) -> Scheduler:
    if name == LocalOnlyScheduler.name:
        return LocalOnlyScheduler()
    if name == NearestSunlitScheduler.name:
        return NearestSunlitScheduler()
    if name == Method1Scheduler.name:
        return Method1Scheduler()
    raise ValueError(f"unknown scheduler: {name}")
