import unittest

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
from satmulator.scheduler import PhoenixLiteScheduler, create_scheduler
from satmulator.workload import demand_distribution


def view(
    sat_id: int,
    *,
    sunlit: bool,
    battery_j: float = 100.0,
    queue_backlog_s: float = 0.0,
    plane: int | None = 0,
    slot: int | None = 0,
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
    def test_create_scheduler_registers_phoenix(self) -> None:
        self.assertIsInstance(create_scheduler("phoenix"), PhoenixLiteScheduler)

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

    def test_peer_selection_uses_highest_battery_within_preferred_plane(self) -> None:
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
        self.assertEqual(scheduler.task_count_by_plane, {1: 1})

    def test_orbit_task_counter_balances_planes_before_battery(self) -> None:
        scheduler = PhoenixLiteScheduler()
        scheduler.task_count_by_plane[1] = 4

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


if __name__ == "__main__":
    unittest.main()
