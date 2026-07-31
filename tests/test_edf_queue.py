import unittest

from satmulator.models import (
    Assignment,
    ComputeConfig,
    ISLConfig,
    QueuedTaskView,
    Route,
    SatelliteView,
    Task,
)
from satmulator.orbit import order_task_queue
from satmulator.runtime import RunningTask, SatelliteRuntime
from satmulator.scheduler import (
    Method7Scheduler,
    Phoenix2Scheduler,
    ProjectedQueueTask,
    enforce_edf_queue_feasibility,
    project_nonpreemptive_edf_queue,
)


def task(task_id: int, deadline_s: float) -> Task:
    return Task(
        task_id=task_id,
        created_time_s=0,
        source_sat=0,
        input_bits=0.0,
        output_bits=0.0,
        deadline_s=deadline_s,
        compute_time_s=5.0,
    )


def running_task(
    task_: Task,
    *,
    remaining_compute_time_s: float = 5.0,
    executed_compute_time_s: float = 0.0,
) -> RunningTask:
    return RunningTask(
        task=task_,
        route=Route((0,)),
        mode="local",
        total_compute_time_s=remaining_compute_time_s + executed_compute_time_s,
        remaining_compute_time_s=remaining_compute_time_s,
        executed_compute_time_s=executed_compute_time_s,
        transmission_time_s=0.0,
        transmission_energy_by_sat={},
        energy_by_sat={},
    )


class EDFQueueTests(unittest.TestCase):
    def test_method7_and_phoenix2_declare_edf_execution_queues(self) -> None:
        self.assertEqual(Method7Scheduler().queue_discipline, "edf")
        self.assertEqual(Phoenix2Scheduler().queue_discipline, "edf")

    def test_runtime_queue_keeps_active_task_and_sorts_waiting_tasks(self) -> None:
        active = running_task(
            task(1, 100.0),
            remaining_compute_time_s=4.0,
            executed_compute_time_s=1.0,
        )
        lax = running_task(task(2, 80.0))
        urgent = running_task(task(3, 20.0))
        sat = SatelliteRuntime(0, "sat_0", 0, 0, 100.0)
        sat.task_queue[:] = [active, lax, urgent]

        order_task_queue(sat, "edf")

        self.assertEqual(
            [running.task.task_id for running in sat.task_queue],
            [1, 3, 2],
        )

    def test_projection_checks_every_task_after_edf_insertion(self) -> None:
        projection = project_nonpreemptive_edf_queue(
            [
                ProjectedQueueTask(1, 30.0, 10.0, 0.0),
                ProjectedQueueTask(2, 15.0, 5.0, 0.0),
            ],
            time_s=0.0,
        )

        self.assertIsNotNone(projection)
        assert projection is not None
        self.assertEqual([entry.task_id for entry in projection.tasks], [2, 1])
        self.assertEqual(projection.finish_time_by_task, {2: 5.0, 1: 15.0})

        self.assertIsNone(
            project_nonpreemptive_edf_queue(
                [
                    ProjectedQueueTask(1, 12.0, 10.0, 0.0),
                    ProjectedQueueTask(2, 8.0, 5.0, 0.0),
                ],
                time_s=0.0,
            )
        )

    def test_assignment_is_rejected_if_it_breaks_existing_queue_deadline(self) -> None:
        new_task = task(2, 8.0)
        view = SatelliteView(
            sat_id=0,
            x_km=0.0,
            y_km=0.0,
            z_km=0.0,
            sunlit=True,
            queued_tasks=(
                QueuedTaskView(
                    task_id=1,
                    absolute_deadline_s=12.0,
                    remaining_compute_time_s=5.0,
                    started=False,
                    transmission_time_s=10.0,
                ),
            ),
        )

        checked = enforce_edf_queue_feasibility(
            tasks=[new_task],
            assignments=[
                Assignment(task_id=2, route=Route((0,)), mode="local")
            ],
            satellite_views=[view],
            time_s=0,
            compute_config=ComputeConfig(1.0, 1.0, 1.0),
            isl_config=ISLConfig(1.0e9, 0.0),
        )

        self.assertEqual(checked[0].mode, "fail")
        self.assertEqual(checked[0].failed_reason, "edf_queue_infeasible")


if __name__ == "__main__":
    unittest.main()
