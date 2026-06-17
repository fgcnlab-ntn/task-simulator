from __future__ import annotations

from .battery import projected_battery_after_step
from .isl import ISLGraph, shortest_route
from .models import (
    Assignment,
    BatteryConfig,
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


class SlackAwareScheduler(Scheduler):
    name = "slack-aware"

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        task_config: TaskConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        by_id = {sat.sat_id: sat for sat in satellite_views}
        reserved_energy = {sat.sat_id: 0.0 for sat in satellite_views}
        reserved_load_cycles = {sat.sat_id: 0.0 for sat in satellite_views}

        def cost_for(task: Task, route):
            return estimate_route_cost(
                task=task,
                route=route,
                task_config=task_config,
                isl_config=isl_config,
            )

        def best_possible_time(task: Task) -> float:
            assert task.source_sat is not None
            times = []
            for target in satellite_views:
                route = shortest_route(isl_graph, task.source_sat, target.sat_id)
                if route is not None:
                    times.append(cost_for(task, route).total_time_s)
            return min(times) if times else float("inf")

        def task_slack(task: Task) -> float:
            remaining = task.created_time_s + task.deadline_s - time_s
            return remaining - best_possible_time(task)

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (
                task_slack(task),
                -(time_s - task.created_time_s),
            ),
        )

        assignments: list[Assignment] = []

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]
            remaining_deadline = task.created_time_s + task.deadline_s - time_s

            best_assignment: Assignment | None = None
            best_score = float("inf")

            for target in satellite_views:
                mode = "local" if target.sat_id == source.sat_id else "offload"

                projected_load_cycles = (
                    reserved_load_cycles[target.sat_id] + task.cpu_cycles
                )
                if projected_load_cycles > scheduler_config.load_max_cycles_per_slot:
                    continue
                route = shortest_route(isl_graph, source.sat_id, target.sat_id)
                if route is None:
                    continue

                cost = cost_for(task, route)
                total_time = cost.total_time_s

                if total_time > remaining_deadline:
                    continue

                projected_battery_pct: list[float] = []
                violates_dod = False
                for sat_id in route.nodes:
                    sat = by_id[sat_id]
                    projected = projected_battery_after_step(
                        battery_now=sat.battery_j,
                        sunlit=sat.sunlit,
                        step_s=step_s,
                        battery=battery,
                        task_energy_j=(
                            reserved_energy[sat_id] + cost.energy_for(sat_id)
                        ),
                        update=time_s > 0,
                    )
                    if projected < battery.min_safe_j:
                        violates_dod = True
                        break
                    projected_battery_pct.append(100.0 * projected / battery.capacity_j)
                if violates_dod:
                    continue

                energy_score = cost.total_energy_j
                time_score = total_time
                load_score = (
                    projected_load_cycles / scheduler_config.load_max_cycles_per_slot
                )
                battery_risk = max(
                    0.0,
                    scheduler_config.low_battery_threshold_pct
                    - min(projected_battery_pct),
                )

                eclipse_penalty = 0.0
                if mode == "local" and not source.sunlit:
                    eclipse_penalty += scheduler_config.eclipse_local_penalty

                score = (
                    scheduler_config.time_weight * time_score
                    + scheduler_config.energy_weight * energy_score
                    + scheduler_config.load_weight * load_score
                    + scheduler_config.battery_weight * battery_risk
                    + eclipse_penalty
                )

                if score < best_score:
                    best_score = score
                    best_assignment = Assignment(
                        task_id=task.task_id,
                        route=route,
                        mode=mode,
                        score=score,
                    )

            can_defer = remaining_deadline > step_s
            defer_score = float("inf")
            if can_defer:
                defer_score = scheduler_config.defer_penalty + max(
                    0.0, step_s - task_slack(task)
                )

            fail_score = scheduler_config.fail_penalty

            if (
                best_assignment is not None
                and best_score <= defer_score
                and best_score <= fail_score
            ):
                assignments.append(best_assignment)
                reserved_load_cycles[best_assignment.target_sat] += task.cpu_cycles
                cost = estimate_route_cost(
                    task=task,
                    route=best_assignment.route,
                    task_config=task_config,
                    isl_config=isl_config,
                )
                for sat_id, energy_j in cost.energy_by_sat.items():
                    reserved_energy[sat_id] += energy_j

            elif can_defer and defer_score <= fail_score:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="defer",
                        score=defer_score,
                    )
                )
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="fail",
                        score=fail_score,
                        failed_reason="no_feasible_candidate",
                    )
                )

        return assignments


def create_scheduler(name: str) -> Scheduler:
    if name == LocalOnlyScheduler.name:
        return LocalOnlyScheduler()
    if name == NearestSunlitScheduler.name:
        return NearestSunlitScheduler()
    if name == SlackAwareScheduler.name:
        return SlackAwareScheduler()
    raise ValueError(f"unknown scheduler: {name}")
