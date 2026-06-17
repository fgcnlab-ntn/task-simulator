import unittest

from satmulator.isl import fully_connected_isl_graph
from satmulator.models import (
    BatteryConfig,
    DemandDistribution,
    ISLConfig,
    SatelliteView,
    SchedulerConfig,
    Task,
    TaskConfig,
)
from satmulator.scheduler import SlackAwareScheduler


def task_config() -> TaskConfig:
    return TaskConfig(
        enabled=True,
        interval_s=30,
        generation_mode="satellite-deterministic",
        random_seed=1,
        tasks_per_sat=1,
        tasks_per_step_choices=(1,),
        tasks_per_step_weights=(1.0,),
        cpu_cycles=1.0,
        cpu_cycles_choices=(1.0,),
        cpu_cycles_weights=(1.0,),
        input_bits=0.0,
        input_bits_choices=(0.0,),
        input_bits_weights=(1.0,),
        output_bits=0.0,
        output_bits_choices=(0.0,),
        output_bits_weights=(1.0,),
        deadline_s=30.0,
        cpu_rate_cycles_s=1.0,
        joule_per_cycle=0.0,
        demand_distribution=DemandDistribution((), (), 0.0),
        min_elevation_deg=30.0,
    )


def scheduler_config(load_max_cycles_per_slot: float) -> SchedulerConfig:
    return SchedulerConfig(
        name="slack-aware",
        load_max_cycles_per_slot=load_max_cycles_per_slot,
        time_weight=0.0,
        energy_weight=0.0,
        battery_weight=0.0,
        load_weight=0.0,
        eclipse_local_penalty=100.0,
    )


class SchedulerLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.views = [
            SatelliteView(0, 0.0, 0.0, 0.0, False, battery_j=1000.0),
            SatelliteView(1, 1.0, 0.0, 0.0, True, battery_j=1000.0),
        ]
        self.graph = fully_connected_isl_graph(self.views)
        self.battery = BatteryConfig(1000.0, 1000.0, 0.0, 0.0, 0.0)
        self.isl = ISLConfig(1.0, 1.0, 0.0, 0.0)

    def task(self, task_id: int, cpu_cycles: float) -> Task:
        return Task(
            task_id=task_id,
            created_time_s=30,
            source_sat=0,
            cpu_cycles=cpu_cycles,
            input_bits=0.0,
            output_bits=0.0,
            deadline_s=30.0,
        )

    def test_load_limit_counts_cycles_not_tasks(self) -> None:
        assignments = SlackAwareScheduler().assign_tasks(
            tasks=[self.task(1, 4.0), self.task(2, 6.0)],
            satellite_views=self.views,
            time_s=30,
            step_s=30,
            battery=self.battery,
            task_config=task_config(),
            isl_config=self.isl,
            isl_graph=self.graph,
            scheduler_config=scheduler_config(10.0),
        )

        self.assertEqual([assignment.target_sat for assignment in assignments], [1, 1])

    def test_load_limit_rejects_candidate_when_cycles_exceed_limit(self) -> None:
        assignments = SlackAwareScheduler().assign_tasks(
            tasks=[self.task(1, 6.0), self.task(2, 6.0)],
            satellite_views=self.views,
            time_s=30,
            step_s=30,
            battery=self.battery,
            task_config=task_config(),
            isl_config=self.isl,
            isl_graph=self.graph,
            scheduler_config=scheduler_config(10.0),
        )

        self.assertEqual([assignment.target_sat for assignment in assignments], [1, 0])

    def test_task_larger_than_load_limit_fails(self) -> None:
        assignments = SlackAwareScheduler().assign_tasks(
            tasks=[self.task(1, 11.0)],
            satellite_views=self.views,
            time_s=30,
            step_s=30,
            battery=self.battery,
            task_config=task_config(),
            isl_config=self.isl,
            isl_graph=self.graph,
            scheduler_config=scheduler_config(10.0),
        )

        self.assertEqual(assignments[0].mode, "fail")
        self.assertEqual(assignments[0].failed_reason, "no_feasible_candidate")


if __name__ == "__main__":
    unittest.main()
