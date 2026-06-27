import datetime as dt
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from satmulator.models import (
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    SatelliteState,
    SnapshotContext,
    SchedulerConfig,
    TaskConfig,
)
from satmulator.orbit import iter_circular_states
from satmulator.runlog import RunLog, iter_state_steps, iter_task_events, load_run
from satmulator.scheduler import LocalOnlyScheduler
from satmulator.workload import load_demand_points


def sample_state(time_s: int = 0) -> SatelliteState:
    return SatelliteState(
        time_s=time_s,
        sat_id=0,
        name="sat_0",
        plane=0,
        slot=0,
        x_km=1.0,
        y_km=2.0,
        z_km=3.0,
        vx_km_s=4.0,
        vy_km_s=5.0,
        vz_km_s=6.0,
        lat_deg=None,
        lon_deg=None,
        elevation_km=None,
        sunlit=True,
        battery_j=80.0,
        battery_pct=80.0,
        harvested_j=1.0,
        consumed_j=2.0,
        safe_battery=True,
        generated_tasks=1,
        completed_tasks=0,
        failed_tasks=0,
        task_energy_j=0.0,
    )


class RunLogTests(unittest.TestCase):
    def test_writes_self_contained_snapshot_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = dt.datetime(2026, 6, 14, tzinfo=dt.timezone.utc)
            log = RunLog(output, start, {})
            context = SnapshotContext(
                projection_label="ECI",
                sun_xy_unit=(0.6, 0.8),
                sun_eci_unit=(0.6, 0.8, 0.0),
            )

            log.write_step([sample_state()], context)
            log.complete([[sample_state()]])

            record = next(iter_state_steps(output))
            self.assertEqual(
                record["snapshot_context"],
                {
                    "projection_label": "ECI",
                    "sun_eci_unit": [0.6, 0.8, 0.0],
                    "sun_xy_unit": [0.6, 0.8],
                },
            )

    def test_writes_one_valid_state_object_per_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = dt.datetime(2026, 6, 14, tzinfo=dt.timezone.utc)
            log = RunLog(output, start, {"test": True})

            log.write_step([sample_state(0)])
            log.write_step([sample_state(30)])
            log.complete([[sample_state(0)], [sample_state(30)]])

            lines = (output / "states.jsonl").read_text().splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual([record["time_s"] for record in records], [0, 30])
            self.assertNotIn("battery_pct", records[0]["satellites"][0])
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "completed")

    def test_records_eclipse_energy_per_step_and_total_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = dt.datetime(2026, 6, 14, tzinfo=dt.timezone.utc)
            log = RunLog(output, start, {})
            first_step = [
                replace(
                    sample_state(0),
                    sunlit=False,
                    consumed_j=2.0,
                    task_energy_j=3.0,
                ),
                replace(
                    sample_state(0),
                    sat_id=1,
                    sunlit=True,
                    consumed_j=10.0,
                    task_energy_j=20.0,
                ),
            ]
            second_step = [
                replace(
                    sample_state(30),
                    sunlit=False,
                    consumed_j=5.0,
                    task_energy_j=7.0,
                )
            ]

            log.write_step(first_step)
            log.write_step(second_step)
            log.complete()

            records = list(iter_state_steps(output))
            self.assertEqual(
                records[0]["energy_summary"]["eclipse"],
                {"idle_j": 2.0, "task_j": 3.0, "total_j": 5.0},
            )
            self.assertEqual(
                records[1]["energy_summary"]["eclipse"],
                {"idle_j": 5.0, "task_j": 7.0, "total_j": 12.0},
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                summary["energy"]["eclipse"],
                {"idle_j": 7.0, "task_j": 10.0, "total_j": 17.0},
            )

    def test_task_summary_counts_pending_from_lifecycle_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = dt.datetime(2026, 6, 14, tzinfo=dt.timezone.utc)
            log = RunLog(output, start, {})

            log.write_task_event({"type": "task_generated", "time_s": 0, "task_id": 1})
            log.write_task_event({"type": "task_generated", "time_s": 0, "task_id": 2})
            log.write_task_event({"type": "task_completed", "time_s": 30, "task_id": 1})
            log.write_step([sample_state(30)])
            log.complete([[sample_state(30)]])

            summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(
            summary["tasks"],
            {
                "generated": 2,
                "completed": 1,
                "deferred": 0,
                "failed": 0,
                "pending": 1,
            },
        )

    def test_failed_run_keeps_parseable_jsonl_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = dt.datetime(2026, 6, 14, tzinfo=dt.timezone.utc)
            log = RunLog(output, start, {})
            log.write_step([sample_state()])
            log.fail(ValueError("broken"))

            manifest = json.loads((output / "run.json").read_text())
            json.loads((output / "states.jsonl").read_text().strip())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["type"], "ValueError")

    def test_simulator_emits_task_lifecycle_without_changing_step_results(self) -> None:
        events: list[dict[str, object]] = []
        battery = BatteryConfig(1000.0, 1000.0, 0.0, 0.0, 0.0)
        compute = ComputeConfig(1.0, 1.0, 0.0)
        task = TaskConfig(
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
            demand_distribution=load_demand_points(None),
            min_elevation_deg=30.0,
        )
        isl = ISLConfig(1.0, 0.0)

        steps = list(
            iter_circular_states(
                satellites=1,
                planes=1,
                altitude_km=550.0,
                inclination_deg=0.0,
                duration_s=30,
                step_s=30,
                battery=battery,
                compute_config=compute,
                task_config=task,
                isl_config=isl,
                scheduler=LocalOnlyScheduler(),
                scheduler_config=SchedulerConfig(name="local"),
                task_event_sink=events.append,
            )
        )

        self.assertEqual(len(steps), 2)
        self.assertEqual(
            [event["type"] for event in events],
            ["task_generated", "task_assigned", "task_completed"],
        )


class RunLogReaderTests(unittest.TestCase):
    def test_reads_manifest_state_steps_and_task_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = dt.datetime(2026, 6, 14, tzinfo=dt.timezone.utc)
            log = RunLog(output, start, {"test": True})
            log.write_task_event({"type": "task_generated", "time_s": 0, "task_id": 1})
            log.write_step([sample_state(0)])
            log.write_step([sample_state(30)])
            log.complete([[sample_state(0)], [sample_state(30)]])

            self.assertEqual(load_run(output)["status"], "completed")
            self.assertEqual(
                [record["time_s"] for record in iter_state_steps(output)],
                [0, 30],
            )
            self.assertEqual(
                [record["type"] for record in iter_task_events(output)],
                ["task_generated"],
            )

    def test_rejects_unsupported_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "run.json").write_text('{"schema_version": 999}\n')

            with self.assertRaisesRegex(ValueError, "unsupported schema_version 999"):
                load_run(output)

    def test_reports_invalid_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "states.jsonl").write_text(
                '{"schema_version":1,"time_s":0}\nnot-json\n'
            )

            with self.assertRaisesRegex(ValueError, r"states\.jsonl:2"):
                list(iter_state_steps(output))


if __name__ == "__main__":
    unittest.main()
