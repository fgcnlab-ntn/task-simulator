import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from satmulator.cli import (
    CONFIG_SECTIONS,
    DEFAULT_CONFIG,
    effective_run_config,
    load_json_config,
    load_standalone_json_config,
    run,
    validate_args,
)


def args_for(**overrides: object) -> argparse.Namespace:
    values = DEFAULT_CONFIG.copy()
    values.update(overrides)
    return argparse.Namespace(**values)


class EffectiveRunConfigTests(unittest.TestCase):
    def test_grid_is_the_default_isl_topology(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["isl_topology"], "grid")
        self.assertEqual(DEFAULT_CONFIG["isl_max_range_km"], 5000.0)

    def test_loads_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"time": {"duration_s": 60}}')

            self.assertEqual(load_json_config(path), {"duration_s": 60})


    def test_standalone_config_rejects_partial_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text('{"time": {"duration_s": 60}}')

            with self.assertRaisesRegex(ValueError, "standalone config"):
                load_standalone_json_config(path)

    def test_bundled_configs_are_complete_standalone_specs(self) -> None:
        required_sections = set(CONFIG_SECTIONS)
        for path in sorted(Path("configs").glob("*.json")):
            with self.subTest(config=str(path)):
                config = json.loads(path.read_text())
                self.assertEqual(set(config), required_sections)
                for section, mapping in CONFIG_SECTIONS.items():
                    self.assertEqual(set(config[section]), set(mapping))
                validate_args(args_for(**load_standalone_json_config(path)))

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
                "satellites": 1584,
                "planes": 72,
                "altitude_km": 550.0,
                "inclination_deg": 53.05,
                "walker_phase": 1,
            },
        )

    def test_run_config_records_demand_point_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"
            with path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["lat", "lon", "weight"])
                writer.writerow([25.0, 121.0, 10.0])
            metadata = {
                "source_url": "https://example.test/worldpop.tif",
                "aggregate_deg": 0.05,
                "output_population": 10.0,
            }
            path.with_suffix(".csv.metadata.json").write_text(json.dumps(metadata))

            config = effective_run_config(args_for(task_demand_points_file=path))
            provenance = config["task"]["demand_points_provenance"]

            self.assertEqual(provenance["points"], 1)
            self.assertEqual(provenance["total_weight"], 10.0)
            self.assertEqual(provenance["conversion"], metadata)

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
            state = json.loads((output / "states.jsonl").read_text())
            self.assertEqual(
                state["snapshot_context"]["sun_eci_unit"],
                [1.0, 0.0, 0.0],
            )

    def test_tle_requires_explicit_non_grid_topology(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable in TLE mode"):
            validate_args(args_for(orbit_model="tle"))

        validate_args(
            args_for(orbit_model="tle", isl_topology="fully-connected")
        )


if __name__ == "__main__":
    unittest.main()
