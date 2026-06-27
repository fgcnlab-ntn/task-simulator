import datetime as dt
import tempfile
import unittest
from pathlib import Path

from satmulator.models import SnapshotContext
from satmulator.plotting import render_run_plots
from satmulator.runlog import RunLog
from tests.test_runlog import sample_state


class PlottingTests(unittest.TestCase):
    def test_regenerates_all_plots_from_run_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = dt.datetime(2026, 6, 14, tzinfo=dt.timezone.utc)
            config = {
                "battery": {"capacity_j": 100.0, "min_safe_pct": 20.0},
            }
            context = SnapshotContext(
                projection_label="ECI",
                sun_xy_unit=(1.0, 0.0),
                sun_eci_unit=(1.0, 0.0, 0.0),
            )
            log = RunLog(output, start, config)
            log.write_task_event(
                {
                    "type": "task_generated",
                    "time_s": 0,
                    "task_id": 1,
                    "source_sat": 0,
                    "location": None,
                    "workload": {
                        "compute_cycles": 1.0,
                        "input_bits": 0.0,
                        "output_bits": 0.0,
                    },
                    "deadline_s": 30.0,
                }
            )
            log.write_task_event(
                {
                    "type": "task_completed",
                    "time_s": 0,
                    "task_id": 1,
                    "source_sat": 0,
                    "target_sat": 0,
                    "mode": "local",
                    "waiting_time_s": 0.0,
                    "compute_time_s": 1.0,
                    "transmission_time_s": 0.0,
                    "total_time_s": 1.0,
                    "energy_j": {"source": 1.0, "target": 0.0, "total": 1.0},
                }
            )
            log.write_step([sample_state(0)], context)
            log.write_step([sample_state(30)], context)
            log.complete()

            render_run_plots(output)

            expected = {
                "snapshot_start.svg",
                "snapshot_end.svg",
                "sunlight_summary.svg",
                "battery_summary.svg",
                "task_summary.svg",
                "sunlight_timeline.svg",
                "battery_timeline.svg",
                "task_mode_summary.svg",
                "offload_target_histogram.svg",
            }
            self.assertEqual({path.name for path in output.glob("*.svg")}, expected)
            original = {
                path.name: path.read_text()
                for path in output.glob("*.svg")
            }
            for path in output.glob("*.svg"):
                path.unlink()

            render_run_plots(output)
            self.assertEqual({path.name for path in output.glob("*.svg")}, expected)
            self.assertEqual(
                {path.name: path.read_text() for path in output.glob("*.svg")},
                original,
            )


if __name__ == "__main__":
    unittest.main()
