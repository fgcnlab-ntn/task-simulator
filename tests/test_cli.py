import argparse
import tempfile
import unittest
from pathlib import Path

from satmulator.cli import DEFAULT_CONFIG, effective_run_config, load_json_config, run


def args_for(**overrides: object) -> argparse.Namespace:
    values = DEFAULT_CONFIG.copy()
    values.update(overrides)
    return argparse.Namespace(**values)


class EffectiveRunConfigTests(unittest.TestCase):
    def test_loads_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"time": {"duration_s": 60}}')

            self.assertEqual(load_json_config(path), {"duration_s": 60})

    def test_tle_orbit_config_omits_circular_only_fields(self) -> None:
        config = effective_run_config(
            args_for(
                orbit_model="tle",
                tle_file=Path("tle/stations.tle"),
            )
        )

        self.assertEqual(
            config["orbit"],
            {
                "orbit_model": "tle",
                "tle_file": "tle/stations.tle",
                "sun_position_file": "de440s.bsp",
            },
        )

    def test_circular_orbit_config_omits_tle_only_fields(self) -> None:
        config = effective_run_config(args_for(orbit_model="circular"))

        self.assertEqual(
            config["orbit"],
            {
                "orbit_model": "circular",
                "satellites": 66,
                "planes": 6,
                "altitude_km": 550.0,
                "inclination_deg": 53.0,
                "walker_phase": 1,
            },
        )

    def test_run_does_not_write_experiment_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(
                args_for(
                    satellites=1,
                    planes=1,
                    duration_s=0,
                    task_enable=False,
                    out=output,
                    tle_file=None,
                    task_demand_points_file=None,
                )
            )

            self.assertFalse(list(output.glob("*.csv")))
            self.assertTrue((output / "run.json").exists())
            self.assertFalse((output / "run_config.json").exists())
            self.assertTrue((output / "states.jsonl").exists())
            self.assertTrue((output / "tasks.jsonl").exists())
            self.assertTrue((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
