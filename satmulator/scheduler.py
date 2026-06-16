from __future__ import annotations

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


def route_nodes(source_sat: int, target_sat: int) -> tuple[int, ...]:
    return (source_sat,) if source_sat == target_sat else (source_sat, target_sat)


class Scheduler:
    name = "base"

    def assign_task(
        self, *, task: Task, satellite_views: list[SatelliteView]
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
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        return [
            self.assign_task(task=task, satellite_views=satellite_views)
            for task in tasks
        ]


class LocalOnlyScheduler(Scheduler):
    name = "local"

    def assign_task(
        self, *, task: Task, satellite_views: list[SatelliteView]
    ) -> Assignment:
        return Assignment(
            task_id=task.task_id,
            route=route_nodes(task.source_sat, task.source_sat),
            mode=self.name,
        )


class NearestSunlitScheduler(Scheduler):
    name = "nearest-sunlit"

    def assign_task(
        self, *, task: Task, satellite_views: list[SatelliteView]
    ) -> Assignment:
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
            route=route_nodes(source.sat_id, target.sat_id),
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
        scheduler_config: SchedulerConfig,
    ) -> list[Assignment]:
        by_id = {sat.sat_id: sat for sat in satellite_views}
        reserved_energy = {sat.sat_id: 0.0 for sat in satellite_views}
        reserved_load = {sat.sat_id: 0 for sat in satellite_views}

        def cost_for(task: Task, source_sat: int, target_sat: int):
            return estimate_route_cost(
                task=task,
                route=Route(route_nodes(source_sat, target_sat)),
                task_config=task_config,
                isl_config=isl_config,
            )

        def best_possible_time(task: Task) -> float:
            assert task.source_sat is not None
            local_t = cost_for(task, task.source_sat, task.source_sat).total_time_s
            offload_targets = [
                sat for sat in satellite_views if sat.sat_id != task.source_sat
            ]
            if not offload_targets:
                return local_t
            offload_t = cost_for(
                task, task.source_sat, offload_targets[0].sat_id
            ).total_time_s
            return min(local_t, offload_t)

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

                if (
                    reserved_load[target.sat_id]
                    >= scheduler_config.max_tasks_per_sat_per_slot
                ):
                    continue

                if mode == "local":
                    cost = cost_for(task, source.sat_id, source.sat_id)
                    total_time = cost.total_time_s
                    source_energy = cost.energy_for(source.sat_id)
                    target_energy = 0.0
                else:
                    cost = cost_for(task, source.sat_id, target.sat_id)
                    total_time = cost.total_time_s
                    source_energy = cost.energy_for(source.sat_id)
                    target_energy = cost.energy_for(target.sat_id)

                if total_time > remaining_deadline:
                    continue

                source_after = (
                    source.battery_j - reserved_energy[source.sat_id] - source_energy
                )
                target_after = (
                    target.battery_j - reserved_energy[target.sat_id] - target_energy
                )

                if source_after < battery.min_safe_j:
                    continue
                if target_after < battery.min_safe_j:
                    continue

                energy_score = cost.total_energy_j
                time_score = total_time
                load_score = reserved_load[target.sat_id]
                source_battery_pct = 100.0 * source_after / battery.capacity_j
                target_battery_pct = 100.0 * target_after / battery.capacity_j
                battery_risk = max(
                    0.0,
                    scheduler_config.low_battery_threshold_pct
                    - min(source_battery_pct, target_battery_pct),
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
                        route=route_nodes(source.sat_id, target.sat_id),
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
                reserved_load[best_assignment.target_sat] += 1
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
                        route=route_nodes(source.sat_id, source.sat_id),
                        mode="defer",
                        score=defer_score,
                    )
                )
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_nodes(source.sat_id, source.sat_id),
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
