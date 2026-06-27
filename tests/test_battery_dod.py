import unittest

from satmulator.isl import fully_connected_isl_graph
from satmulator.models import (
    Assignment,
    BatteryConfig,
    DemandDistribution,
    ISLConfig,
    Route,
    SatelliteView,
    SchedulerConfig,
    Task,
    TaskConfig,
)
from satmulator.orbit import apply_step
from satmulator.runtime import EnvironmentRuntime, SatelliteRuntime
from satmulator.scheduler import SlackAwareScheduler


def task_config(*, joule_per_cycle: float = 1.0) -> TaskConfig:
    return TaskConfig(
        enabled=True,
        interval_s=30,
        generation_mode="satellite-deterministic",
        random_seed=1,
        tasks_per_sat=1,
        tasks_per_step_choices=(1,),
        tasks_per_step_weights=(1.0,),
        cpu_cycles=10.0,
        cpu_cycles_choices=(10.0,),
        cpu_cycles_weights=(1.0,),
        input_bits=0.0,
        input_bits_choices=(0.0,),
        input_bits_weights=(1.0,),
        output_bits=0.0,
        output_bits_choices=(0.0,),
        output_bits_weights=(1.0,),
        deadline_s=30.0,
        cpu_rate_cycles_s=1.0,
        joule_per_cycle=joule_per_cycle,
        demand_distribution=DemandDistribution((), (), 0.0),
        min_elevation_deg=30.0,
    )


class BatteryDoDTests(unittest.TestCase):
    def test_apply_step_rejects_task_that_would_cross_dod_limit(self) -> None:
        battery = BatteryConfig(
            capacity_j=100.0,
            initial_j=25.0,
            min_safe_j=20.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            cpu_cycles=10.0,
            input_bits=0.0,
            output_bits=0.0,
            deadline_s=30.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=25.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )

        states, records = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[
                Assignment(task_id=1, route=Route((0,)), mode="local")
            ],
        )

        self.assertEqual(records[0].status, "failed")
        self.assertEqual(records[0].failed_reason, "battery")
        self.assertEqual(states[0].battery_j, 25.0)
        self.assertTrue(states[0].safe_battery)

    def test_slack_aware_skips_candidate_that_would_cross_dod_limit(self) -> None:
        battery = BatteryConfig(
            capacity_j=100.0,
            initial_j=100.0,
            min_safe_j=20.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        views = [
            SatelliteView(
                sat_id=0,
                x_km=0.0,
                y_km=0.0,
                z_km=0.0,
                sunlit=False,
                battery_j=100.0,
            ),
            SatelliteView(
                sat_id=1,
                x_km=1.0,
                y_km=0.0,
                z_km=0.0,
                sunlit=True,
                battery_j=25.0,
            ),
        ]
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            cpu_cycles=10.0,
            input_bits=0.0,
            output_bits=0.0,
            deadline_s=30.0,
        )

        assignments = SlackAwareScheduler().assign_tasks(
            tasks=[task],
            satellite_views=views,
            time_s=30,
            step_s=30,
            battery=battery,
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            isl_graph=fully_connected_isl_graph(views),
            scheduler_config=SchedulerConfig(
                name="slack-aware",
                time_weight=0.0,
                energy_weight=0.0,
                battery_weight=0.0,
                load_weight=0.0,
                eclipse_local_penalty=100.0,
            ),
        )

        self.assertEqual(assignments[0].route.nodes, (0,))
        self.assertEqual(assignments[0].mode, "local")


if __name__ == "__main__":
    unittest.main()
