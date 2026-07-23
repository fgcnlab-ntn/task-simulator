import argparse
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from satmulator.cli import (
    CONFIG_SECTIONS,
    DEFAULT_CONFIG,
    OPTIONAL_CONFIG_KEYS,
    effective_run_config,
    load_json_config,
    load_standalone_json_config,
    parse_args,
    run,
    validate_args,
    walker_raan_spread_deg,
)


def args_for(**overrides: object) -> argparse.Namespace:
    values = DEFAULT_CONFIG.copy()
    values.update(overrides)
    return argparse.Namespace(**values)


class EffectiveRunConfigTests(unittest.TestCase):
    def test_grid_is_the_default_isl_topology(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["isl_topology"], "grid")
        self.assertEqual(DEFAULT_CONFIG["isl_max_range_km"], 5000.0)

    def test_objective_alpha_is_a_formal_config_section(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["objective_alpha"], 0.5)
        self.assertEqual(
            load_json_config(Path("configs/base/template.json"))["objective_alpha"],
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "objective.alpha"):
            validate_args(args_for(objective_alpha=1.1))

    def test_loads_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"time": {"duration_s": 60}}')

            self.assertEqual(load_json_config(path), {"duration_s": 60})


    def test_parse_args_keeps_only_run_control_overrides(self) -> None:
        with patch(
            "sys.argv",
            [
                "minimal_orbit.py",
                "--config",
                "configs/base/template.json",
                "--duration-s",
                "60",
                "--step-s",
                "10",
                "--no-task",
                "--out",
                "output/debug",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.duration_s, 60)
        self.assertEqual(args.step_s, 10)
        self.assertFalse(args.task_enable)
        self.assertEqual(args.out, Path("output/debug"))
        self.assertEqual(args.satellites, 1584)

    def test_parse_args_rejects_model_config_flags(self) -> None:
        with patch(
            "sys.argv",
            [
                "minimal_orbit.py",
                "--config",
                "configs/base/template.json",
                "--satellites",
                "1",
            ],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

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
                    expected_keys = set(mapping)
                    optional_keys = OPTIONAL_CONFIG_KEYS.get(section, set())
                    actual_keys = set(config[section])
                    self.assertTrue(expected_keys - optional_keys <= actual_keys)
                    self.assertTrue(actual_keys <= expected_keys)
                args = args_for(**load_standalone_json_config(path))
                validate_args(args)
                task_compute_time_s = (
                    args.task_compute_time_s
                    if args.task_compute_time_s is not None
                    else (
                        args.task_input_bits
                        * args.compute_cycles_per_input_bit
                        / args.satellite_cpu_frequency_hz
                    )
                )
                self.assertLessEqual(
                    task_compute_time_s,
                    args.step_s * args.scheduler_cpu_utilization_limit,
                )

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
        self.assertEqual(config["objective"], {"alpha": 0.5})
        self.assertEqual(config["logging"], {"task_events": "full"})

    def test_logging_task_events_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "logging.task_events"):
            validate_args(args_for(logging_task_events="chatty"))

    def test_oneweb_648_config_uses_walker_delta_layout(self) -> None:
        args = args_for(
            **load_standalone_json_config(Path("configs/base/oneweb_648.json"))
        )

        self.assertEqual(args.orbit_model, "circular")
        self.assertEqual(args.satellites, 648)
        self.assertEqual(args.planes, 18)
        self.assertEqual(args.satellites // args.planes, 36)
        self.assertEqual(args.walker_phase, 1)
        self.assertEqual(args.altitude_km, 1200.0)
        self.assertEqual(args.inclination_deg, 87.9)

    def test_kuiper_784_config_uses_walker_delta_layout(self) -> None:
        args = args_for(
            **load_standalone_json_config(Path("configs/base/kuiper_784.json"))
        )

        self.assertEqual(args.orbit_model, "circular")
        self.assertEqual(args.satellites, 784)
        self.assertEqual(args.planes, 28)
        self.assertEqual(args.satellites // args.planes, 28)
        self.assertEqual(args.walker_phase, 1)
        self.assertEqual(args.altitude_km, 590.0)
        self.assertEqual(args.inclination_deg, 33.0)

    def test_kuiper_1156_high_inclination_config_uses_walker_delta_layout(self) -> None:
        args = args_for(
            **load_standalone_json_config(
                Path("configs/base/kuiper_1156_630km_51p9deg.json")
            )
        )

        self.assertEqual(args.orbit_model, "circular")
        self.assertEqual(args.satellites, 1156)
        self.assertEqual(args.planes, 34)
        self.assertEqual(args.satellites // args.planes, 34)
        self.assertEqual(args.walker_phase, 1)
        self.assertEqual(args.altitude_km, 630.0)
        self.assertEqual(args.inclination_deg, 51.9)

    def test_iridium_66_config_uses_walker_star_layout(self) -> None:
        args = args_for(
            **load_standalone_json_config(Path("configs/base/iridium_66.json"))
        )

        self.assertEqual(args.orbit_model, "circular")
        self.assertEqual(args.satellites, 66)
        self.assertEqual(args.planes, 6)
        self.assertEqual(args.satellites // args.planes, 11)
        self.assertEqual(args.walker_phase, 2)
        self.assertEqual(args.altitude_km, 780.0)
        self.assertEqual(args.inclination_deg, 86.4)

    def test_iridium_uses_walker_star_raan_spread(self) -> None:
        self.assertEqual(walker_raan_spread_deg(args_for(run_name="iridium_66")), 180.0)
        self.assertEqual(walker_raan_spread_deg(args_for(run_name="template")), 360.0)

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

    def test_run_writes_logs_without_derived_artifacts(self) -> None:
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
            self.assertFalse(list(output.glob("*.svg")))
            self.assertTrue((output / "run.json").exists())
            self.assertFalse((output / "run_config.json").exists())
            self.assertTrue((output / "states.jsonl").exists())
            self.assertTrue((output / "tasks.jsonl").exists())
            self.assertTrue((output / "summary.json").exists())
            state = json.loads((output / "states.jsonl").read_text())
            sun_unit = state["snapshot_context"]["sun_eci_unit"]
            self.assertAlmostEqual(sum(component**2 for component in sun_unit), 1.0)
            self.assertIn("ephemeris Sun vector", state["snapshot_context"]["projection_label"])

    def test_tle_requires_explicit_non_grid_topology(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable in TLE mode"):
            validate_args(args_for(orbit_model="tle"))

        validate_args(
            args_for(orbit_model="tle", isl_topology="fully-connected")
        )


if __name__ == "__main__":
    unittest.main()
