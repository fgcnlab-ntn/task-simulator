import unittest
from unittest.mock import patch

from satmulator.isl import ISLGraph
from satmulator.models import (
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    SatelliteView,
    SchedulerConfig,
    Task,
    TaskConfig,
)
import satmulator.scheduler as scheduler_module
from satmulator.scheduler import (
    Phoenix2Scheduler,
    PhoenixLiteScheduler,
    create_scheduler,
    routes_to_targets,
)
from satmulator.workload import demand_distribution


def view(
    sat_id: int,
    *,
    sunlit: bool,
    battery_j: float = 100.0,
    queue_backlog_s: float = 0.0,
    plane: int | None = 0,
    slot: int | None = 0,
    next_sunlit_time_s: float | None = None,
) -> SatelliteView:
    return SatelliteView(
        sat_id=sat_id,
        x_km=float(sat_id),
        y_km=0.0,
        z_km=0.0,
        sunlit=sunlit,
        battery_j=battery_j,
        queue_backlog_s=queue_backlog_s,
        plane=plane,
        slot=slot,
        next_sunlit_time_s=next_sunlit_time_s,
    )


def task(
    task_id: int = 1,
    *,
    source_sat: int = 0,
    deadline_s: float = 30.0,
    input_bits: float = 1.0,
    output_bits: float = 0.0,
) -> Task:
    return Task(
        task_id=task_id,
        created_time_s=0,
        source_sat=source_sat,
        input_bits=input_bits,
        output_bits=output_bits,
        deadline_s=deadline_s,
    )


def assign_one(
    scheduler: PhoenixLiteScheduler,
    *,
    task_: Task,
    views: list[SatelliteView],
    graph: ISLGraph,
    time_s: int = 0,
    step_s: int = 10,
):
    return scheduler.assign_tasks(
        tasks=[task_],
        satellite_views=views,
        time_s=time_s,
        step_s=step_s,
        battery=BatteryConfig(100.0, 100.0, 10.0, 0.0, 0.0),
        compute_config=ComputeConfig(
            cycles_per_input_bit=1.0,
            cpu_frequency_hz=1.0,
            cpu_power_w=1.0,
        ),
        task_config=TaskConfig(
            enabled=True,
            interval_s=10,
            generation_mode="satellite-deterministic",
            random_seed=1,
            tasks_per_sat=1,
            tasks_per_step_choices=(1,),
            tasks_per_step_weights=(1.0,),
            input_bits=1.0,
            input_bits_choices=(1.0,),
            input_bits_weights=(1.0,),
            output_bits=0.0,
            output_bits_choices=(0.0,),
            output_bits_weights=(1.0,),
            deadline_s=30.0,
            demand_distribution=demand_distribution(()),
            min_elevation_deg=30.0,
        ),
        isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
        isl_graph=graph,
        scheduler_config=SchedulerConfig(name="phoenix"),
    )[0]


class PhoenixLiteSchedulerTests(unittest.TestCase):
    def test_routes_to_targets_returns_only_requested_reachable_routes(self) -> None:
        routes = routes_to_targets(
            ISLGraph(
                {
                    0: (1,),
                    1: (0, 2),
                    2: (1, 3),
                    3: (2,),
                    9: (),
                }
            ),
            0,
            {2, 3, 9},
        )

        self.assertEqual(set(routes), {2, 3})
        self.assertEqual(routes[2].nodes, (0, 1, 2))
        self.assertEqual(routes[3].nodes, (0, 1, 2, 3))

    def test_create_scheduler_registers_phoenix(self) -> None:
        self.assertIsInstance(create_scheduler("phoenix"), PhoenixLiteScheduler)

    def test_create_scheduler_registers_phoenix2(self) -> None:
        self.assertIsInstance(create_scheduler("phoenix2"), Phoenix2Scheduler)

    def test_sunlit_source_uses_local_execution(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(),
            views=[
                view(0, sunlit=True, plane=0, slot=0),
                view(1, sunlit=True, battery_j=1000.0, plane=1, slot=0),
            ],
            graph=ISLGraph({0: (1,), 1: (0,)}),
        )

        self.assertEqual(assignment.mode, "local")
        self.assertEqual(assignment.route.nodes, (0,))

    def test_eclipse_source_defers_when_one_step_wait_is_safe(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=20.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(1, sunlit=True, battery_j=1000.0, plane=1, slot=0),
            ],
            graph=ISLGraph({0: (1,), 1: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "defer")
        self.assertEqual(assignment.route.nodes, (0,))

    def test_eclipse_source_defers_until_predicted_sunlight(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=50.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0, next_sunlit_time_s=30.0),
                view(1, sunlit=True, battery_j=1000.0, plane=1, slot=0),
            ],
            graph=ISLGraph({0: (1,), 1: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "defer")
        self.assertEqual(assignment.route.nodes, (0,))
        self.assertEqual(assignment.score, 30.0)

    def test_phoenix2_deferred_tasks_reserve_future_source_capacity(self) -> None:
        scheduler = Phoenix2Scheduler()

        assignments = scheduler.assign_tasks(
            tasks=[
                task(task_id=1, deadline_s=11.5),
                task(task_id=2, deadline_s=11.5),
            ],
            satellite_views=[
                view(0, sunlit=False, plane=0, slot=0, next_sunlit_time_s=10.0),
                view(1, sunlit=True, battery_j=1000.0, plane=1, slot=0),
            ],
            time_s=0,
            step_s=10,
            battery=BatteryConfig(100.0, 100.0, 10.0, 0.0, 0.0),
            compute_config=ComputeConfig(
                cycles_per_input_bit=1.0,
                cpu_frequency_hz=1.0,
                cpu_power_w=1.0,
            ),
            task_config=TaskConfig(
                enabled=True,
                interval_s=10,
                generation_mode="satellite-deterministic",
                random_seed=1,
                tasks_per_sat=1,
                tasks_per_step_choices=(1,),
                tasks_per_step_weights=(1.0,),
                input_bits=1.0,
                input_bits_choices=(1.0,),
                input_bits_weights=(1.0,),
                output_bits=0.0,
                output_bits_choices=(0.0,),
                output_bits_weights=(1.0,),
                deadline_s=30.0,
                demand_distribution=demand_distribution(()),
                min_elevation_deg=30.0,
            ),
            isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
            isl_graph=ISLGraph({0: (1,), 1: (0,)}),
            scheduler_config=SchedulerConfig(name="phoenix2"),
        )

        self.assertEqual([assignment.mode for assignment in assignments], ["defer", "offload"])
        self.assertEqual(assignments[0].score, 10.0)
        self.assertEqual(assignments[1].route.nodes, (0, 1))

    def test_eclipse_source_offloads_when_next_sunlight_misses_deadline(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=20.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0, next_sunlit_time_s=30.0),
                view(1, sunlit=True, battery_j=500.0, plane=1, slot=0),
            ],
            graph=ISLGraph({0: (1,), 1: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "offload")
        self.assertEqual(assignment.route.nodes, (0, 1))

    def test_eclipse_source_offloads_when_wait_would_miss_deadline(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=5.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(1, sunlit=True, battery_j=500.0, plane=1, slot=0),
            ],
            graph=ISLGraph({0: (1,), 1: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "offload")
        self.assertEqual(assignment.route.nodes, (0, 1))

    def test_peer_selection_records_energy_load_for_target_plane(self) -> None:
        scheduler = PhoenixLiteScheduler()
        assignment = assign_one(
            scheduler,
            task_=task(deadline_s=5.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(1, sunlit=True, battery_j=50.0, plane=1, slot=0),
                view(2, sunlit=True, battery_j=500.0, plane=1, slot=1),
            ],
            graph=ISLGraph({0: (1, 2), 1: (0,), 2: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.route.nodes, (0, 2))
        self.assertEqual(scheduler.plane_load_by_plane, {1: 1.0})

    def test_plane_load_uses_compute_energy_not_task_count(self) -> None:
        scheduler = PhoenixLiteScheduler()
        assignment = assign_one(
            scheduler,
            task_=task(deadline_s=5.0, input_bits=3.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(1, sunlit=True, battery_j=500.0, plane=1, slot=0),
            ],
            graph=ISLGraph({0: (1,), 1: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "offload")
        self.assertEqual(scheduler.plane_load_by_plane, {1: 3.0})
        self.assertIs(scheduler.task_count_by_plane, scheduler.plane_load_by_plane)

    def test_peer_selection_uses_energy_score_not_raw_battery(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=5.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(
                    1,
                    sunlit=True,
                    battery_j=100.0,
                    queue_backlog_s=80.0,
                    plane=1,
                    slot=0,
                ),
                view(
                    2,
                    sunlit=True,
                    battery_j=90.0,
                    queue_backlog_s=0.0,
                    plane=1,
                    slot=1,
                ),
            ],
            graph=ISLGraph({0: (1, 2), 1: (0,), 2: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "offload")
        self.assertEqual(assignment.route.nodes, (0, 2))

    def test_orbit_load_balances_planes_before_battery(self) -> None:
        scheduler = PhoenixLiteScheduler()
        scheduler.plane_load_by_plane[1] = 4.0

        assignment = assign_one(
            scheduler,
            task_=task(deadline_s=5.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(1, sunlit=True, battery_j=1000.0, plane=1, slot=0),
                view(2, sunlit=True, battery_j=100.0, plane=2, slot=0),
            ],
            graph=ISLGraph({0: (1, 2), 1: (0,), 2: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.route.nodes, (0, 2))

    def test_phoenix2_orbit_load_is_bounded_to_current_batch(self) -> None:
        scheduler = Phoenix2Scheduler()
        scheduler.plane_load_by_plane[1] = 4.0

        assignment = assign_one(
            scheduler,
            task_=task(deadline_s=5.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(1, sunlit=True, battery_j=1000.0, plane=1, slot=0),
                view(2, sunlit=True, battery_j=100.0, plane=2, slot=0),
            ],
            graph=ISLGraph({0: (1, 2), 1: (0,), 2: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.route.nodes, (0, 1))
        self.assertEqual(scheduler.plane_load_by_plane, {1: 1.0})

    def test_peer_selection_does_not_scan_unselected_planes(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=5.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(
                    1,
                    sunlit=True,
                    battery_j=1000.0,
                    queue_backlog_s=10.0,
                    plane=1,
                    slot=0,
                ),
                view(2, sunlit=True, battery_j=100.0, plane=2, slot=0),
            ],
            graph=ISLGraph({0: (1, 2), 1: (0,), 2: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "local")
        self.assertEqual(assignment.route.nodes, (0,))

    def test_no_plane_metadata_falls_back_to_global_sunlit_peer(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=5.0),
            views=[
                view(0, sunlit=False, plane=-1, slot=0),
                view(1, sunlit=True, battery_j=100.0, plane=-1, slot=1),
                view(2, sunlit=True, battery_j=900.0, plane=-1, slot=2),
            ],
            graph=ISLGraph({0: (1, 2), 1: (0,), 2: (0,)}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "offload")
        self.assertEqual(assignment.route.nodes, (0, 2))

    def test_no_feasible_candidate_fails(self) -> None:
        assignment = assign_one(
            PhoenixLiteScheduler(),
            task_=task(deadline_s=0.5, input_bits=1.0),
            views=[
                view(0, sunlit=False, plane=0, slot=0),
                view(1, sunlit=True, plane=1, slot=0),
            ],
            graph=ISLGraph({0: (), 1: ()}),
            step_s=10,
        )

        self.assertEqual(assignment.mode, "fail")
        self.assertEqual(assignment.failed_reason, "no_feasible_candidate")

    def test_reuses_route_map_for_tasks_from_same_source(self) -> None:
        scheduler = PhoenixLiteScheduler()
        views = [
            view(0, sunlit=False, plane=0, slot=0),
            view(1, sunlit=True, battery_j=100.0, plane=1, slot=0),
            view(2, sunlit=True, battery_j=200.0, plane=1, slot=1),
        ]
        graph = ISLGraph({0: (1, 2), 1: (0,), 2: (0,)})
        tasks = [
            task(task_id=1, deadline_s=5.0),
            task(task_id=2, deadline_s=5.0),
        ]

        with patch(
            "satmulator.scheduler.routes_to_targets",
            wraps=scheduler_module.routes_to_targets,
        ) as routes_to_targets_:
            assignments = scheduler.assign_tasks(
                tasks=tasks,
                satellite_views=views,
                time_s=0,
                step_s=10,
                battery=BatteryConfig(100.0, 100.0, 10.0, 0.0, 0.0),
                compute_config=ComputeConfig(
                    cycles_per_input_bit=1.0,
                    cpu_frequency_hz=1.0,
                    cpu_power_w=1.0,
                ),
                task_config=TaskConfig(
                    enabled=True,
                    interval_s=10,
                    generation_mode="satellite-deterministic",
                    random_seed=1,
                    tasks_per_sat=1,
                    tasks_per_step_choices=(1,),
                    tasks_per_step_weights=(1.0,),
                    input_bits=1.0,
                    input_bits_choices=(1.0,),
                    input_bits_weights=(1.0,),
                    output_bits=0.0,
                    output_bits_choices=(0.0,),
                    output_bits_weights=(1.0,),
                    deadline_s=30.0,
                    demand_distribution=demand_distribution(()),
                    min_elevation_deg=30.0,
                ),
                isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
                isl_graph=graph,
                scheduler_config=SchedulerConfig(name="phoenix"),
            )

        self.assertEqual(routes_to_targets_.call_count, 1)
        self.assertEqual([assignment.mode for assignment in assignments], ["offload", "offload"])

    def test_builds_sunlit_candidate_cache_once_per_step(self) -> None:
        scheduler = PhoenixLiteScheduler()
        views = [
            view(0, sunlit=False, plane=0, slot=0),
            view(1, sunlit=True, battery_j=100.0, plane=1, slot=0),
            view(2, sunlit=True, battery_j=200.0, plane=1, slot=1),
        ]
        graph = ISLGraph({0: (1, 2), 1: (0,), 2: (0,)})
        tasks = [
            task(task_id=1, deadline_s=5.0),
            task(task_id=2, deadline_s=5.0),
        ]

        with patch.object(
            scheduler,
            "_candidate_cache",
            wraps=scheduler._candidate_cache,
        ) as candidate_cache:
            scheduler.assign_tasks(
                tasks=tasks,
                satellite_views=views,
                time_s=0,
                step_s=10,
                battery=BatteryConfig(100.0, 100.0, 10.0, 0.0, 0.0),
                compute_config=ComputeConfig(
                    cycles_per_input_bit=1.0,
                    cpu_frequency_hz=1.0,
                    cpu_power_w=1.0,
                ),
                task_config=TaskConfig(
                    enabled=True,
                    interval_s=10,
                    generation_mode="satellite-deterministic",
                    random_seed=1,
                    tasks_per_sat=1,
                    tasks_per_step_choices=(1,),
                    tasks_per_step_weights=(1.0,),
                    input_bits=1.0,
                    input_bits_choices=(1.0,),
                    input_bits_weights=(1.0,),
                    output_bits=0.0,
                    output_bits_choices=(0.0,),
                    output_bits_weights=(1.0,),
                    deadline_s=30.0,
                    demand_distribution=demand_distribution(()),
                    min_elevation_deg=30.0,
                ),
                isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
                isl_graph=graph,
                scheduler_config=SchedulerConfig(name="phoenix"),
            )

        self.assertEqual(candidate_cache.call_count, 1)

    def test_candidate_cache_keeps_plane_satellites_sorted_by_battery(self) -> None:
        scheduler = PhoenixLiteScheduler()
        views = [
            view(sat_id, sunlit=True, battery_j=float(sat_id), plane=1, slot=sat_id)
            for sat_id in range(1, 7)
        ]
        views.append(view(10, sunlit=False, battery_j=1000.0, plane=1, slot=10))

        candidate_cache = scheduler._candidate_cache(views)

        self.assertEqual(
            [sat.sat_id for sat in candidate_cache.sunlit_by_plane[1]],
            [6, 5, 4, 3, 2, 1],
        )
        self.assertEqual(
            [sat.sat_id for sat in candidate_cache.sunlit_global],
            [6, 5, 4, 3, 2, 1],
        )


if __name__ == "__main__":
    unittest.main()
