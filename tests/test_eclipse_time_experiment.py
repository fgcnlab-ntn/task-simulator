import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "eclipse_time_experiment.py"
SPEC = importlib.util.spec_from_file_location("eclipse_time_experiment", SCRIPT)
eclipse_time_experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eclipse_time_experiment)


class EclipseTimeExperimentTests(unittest.TestCase):
    def test_extracts_complete_eclipse_intervals_and_drops_censored_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = [
                # sat 0 starts eclipsed: this edge interval is left-censored and ignored.
                (0, {0: False, 1: True}),
                (30, {0: True, 1: False}),
                (60, {0: False, 1: False}),
                (90, {0: True, 1: True}),
                # sat 0 enters eclipse and never exits: right-censored and ignored.
                (120, {0: False, 1: True}),
            ]
            with (output / "states.jsonl").open("w") as stream:
                for time_s, states in records:
                    stream.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "time_s": time_s,
                                "satellites": [
                                    {"id": sat_id, "sunlit": sunlit}
                                    for sat_id, sunlit in states.items()
                                ],
                            }
                        )
                        + "\n"
                    )

            durations = eclipse_time_experiment.eclipse_durations_from_run(output)

        self.assertEqual(sorted(durations), [30.0, 60.0])

    def test_summary_reports_minutes(self) -> None:
        summary = eclipse_time_experiment.summarize_durations([1560.0, 1800.0, 2160.0])

        self.assertEqual(summary["intervals"], 3)
        self.assertEqual(summary["min_min"], 26.0)
        self.assertEqual(summary["mean_min"], 30.666666666666668)
        self.assertEqual(summary["max_min"], 36.0)


if __name__ == "__main__":
    unittest.main()
