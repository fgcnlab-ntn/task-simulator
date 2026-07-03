import unittest

from satmulator.isl import fully_connected_isl_graph
from satmulator.models import (
    Assignment,
    BatteryConfig,
    ComputeConfig,
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


def task_config() -> TaskConfig:
    return TaskConfig(
        enabled=True,
        interval_s=30,
        generation_mode="satellite-deterministic",
        random_seed=1,
        tasks_per_sat=1,
        tasks_per_step_choices=(1,),
        tasks_per_step_weights=(1.0,),
        input_bits=0.0,
        input_bits_choices=(0.0,),
        input_bits_weights=(1.0,),
        output_bits=0.0,
        output_bits_choices=(0.0,),
        output_bits_weights=(1.0,),
        deadline_s=30.0,
        demand_distribution=DemandDistribution((), (), 0.0),
        min_elevation_deg=30.0,
)


def compute_config() -> ComputeConfig:
    return ComputeConfig(
        cycles_per_input_bit=1.0,
        cpu_frequency_hz=1.0,
        cpu_power_w=1.0,
    )


class BatteryDoDTests(unittest.TestCase):
    def test_apply_step_executes_task_that_crosses_dod_limit(self) -> None:
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
            input_bits=10.0,
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
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[
                Assignment(task_id=1, route=Route((0,)), mode="local")
            ],
        )

        self.assertEqual(records[0].status, "completed")
        self.assertEqual(records[0].failed_reason, "")
        self.assertEqual(states[0].battery_j, 15.0)
        self.assertFalse(states[0].safe_battery)

    def test_apply_step_keeps_oversized_task_running(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            input_bits=100.0,
            output_bits=0.0,
            deadline_s=1000.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )

        states, records = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[Assignment(task_id=1, route=Route((0,)), mode="local")],
        )

        self.assertEqual(records, [])
        self.assertEqual(len(env.running_tasks), 1)
        self.assertEqual(states[0].task_energy_j, 30.0)
        self.assertEqual(states[0].battery_j, 970.0)

    def test_apply_step_completes_running_task_later(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            input_bits=40.0,
            output_bits=0.0,
            deadline_s=120.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )

        apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[Assignment(task_id=1, route=Route((0,)), mode="local")],
        )
        env.time_s = 60
        states, records = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[],
            assignments=[],
        )

        self.assertEqual(len(env.running_tasks), 0)
        self.assertEqual(records[0].status, "completed")
        self.assertEqual(records[0].compute_time_s, 40.0)
        self.assertEqual(records[0].total_time_s, 40.0)
        self.assertEqual(states[0].task_energy_j, 10.0)
        self.assertEqual(states[0].battery_j, 960.0)

    def test_satellite_queue_runs_head_before_next_task(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        tasks = [
            Task(
                task_id=1,
                created_time_s=30,
                source_sat=0,
                input_bits=40.0,
                output_bits=0.0,
                deadline_s=1000.0,
            ),
            Task(
                task_id=2,
                created_time_s=30,
                source_sat=0,
                input_bits=10.0,
                output_bits=0.0,
                deadline_s=1000.0,
            ),
        ]
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )

        states, records = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=tasks,
            assignments=[
                Assignment(task_id=1, route=Route((0,)), mode="local"),
                Assignment(task_id=2, route=Route((0,)), mode="local"),
            ],
        )

        self.assertEqual(records, [])
        self.assertEqual([task.task.task_id for task in env.satellites[0].task_queue], [1, 2])
        self.assertEqual(states[0].task_energy_j, 30.0)

        env.time_s = 60
        states, records = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[],
            assignments=[],
        )

        self.assertEqual([record.task_id for record in records], [1, 2])
        self.assertEqual([record.status for record in records], ["completed", "completed"])
        self.assertEqual([record.total_time_s for record in records], [40.0, 50.0])
        self.assertEqual(len(env.running_tasks), 0)
        self.assertEqual(states[0].task_energy_j, 20.0)

    def test_apply_step_fails_running_task_that_finishes_after_deadline(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            input_bits=40.0,
            output_bits=0.0,
            deadline_s=35.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )

        apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[Assignment(task_id=1, route=Route((0,)), mode="local")],
        )
        env.time_s = 60
        states, records = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            task_config=task_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[],
            assignments=[],
        )

        self.assertEqual(len(env.running_tasks), 0)
        self.assertEqual(records[0].status, "failed")
        self.assertEqual(records[0].failed_reason, "deadline")
        self.assertEqual(records[0].compute_time_s, 40.0)
        self.assertEqual(states[0].failed_tasks, 1)

    def test_slack_aware_does_not_treat_dod_as_hard_limit(self) -> None:
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
            input_bits=10.0,
            output_bits=0.0,
            deadline_s=30.0,
        )

        assignments = SlackAwareScheduler().assign_tasks(
            tasks=[task],
            satellite_views=views,
            time_s=30,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
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

        self.assertEqual(assignments[0].route.nodes, (0, 1))
        self.assertEqual(assignments[0].mode, "offload")


if __name__ == "__main__":
    unittest.main()
