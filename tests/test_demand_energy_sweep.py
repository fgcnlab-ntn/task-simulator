import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "demand_energy_sweep.py"
SPEC = importlib.util.spec_from_file_location("demand_energy_sweep", SCRIPT)
demand_energy_sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demand_energy_sweep)


class DemandEnergySweepTests(unittest.TestCase):
    def test_default_output_is_breach_ratio_experiment_dir(self) -> None:
        self.assertEqual(
            demand_energy_sweep.DEFAULT_OUTPUT,
            Path("experiments/breach_ratio/demand_energy_sweep"),
        )

    def test_one_circular_orbit_duration_rounds_to_step(self) -> None:
        values = {
            "orbit_model": "circular",
            "altitude_km": 550.0,
            "step_s": 30,
            "duration_s": 1800,
        }

        duration_s = demand_energy_sweep.one_circular_orbit_duration_s(values)

        self.assertEqual(duration_s % 30, 0)
        self.assertGreaterEqual(duration_s, 5738)
        self.assertLess(duration_s, 5800)

    def test_scenario_args_forces_local_fixed_all_workload(self) -> None:
        values = {
            "run_name": "base",
            "run_description": "",
            "orbit_model": "circular",
            "tle_file": None,
            "sun_position_file": "de440s.bsp",
            "satellites": 1,
            "planes": 1,
            "altitude_km": 550.0,
            "inclination_deg": 53.05,
            "duration_s": 30,
            "step_s": 30,
            "walker_phase": 1,
            "battery_capacity_j": 1000.0,
            "battery_initial_pct": 100.0,
            "battery_min_safe_pct": 70.0,
            "harvest_w": 0.0,
            "idle_w": 0.0,
            "task_enable": True,
            "scheduler": "nearest-sunlit",
            "task_interval_s": 300,
            "task_generation_mode": "demand-points",
            "task_random_seed": 42,
            "tasks_per_sat": 1,
            "tasks_per_step_choices": [1],
            "tasks_per_step_weights": [1.0],
            "task_input_bits": 1.0,
            "task_input_bits_choices": [1.0],
            "task_input_bits_weights": [1.0],
            "task_output_bits": 1.0,
            "task_output_bits_choices": [1.0],
            "task_output_bits_weights": [1.0],
            "task_demand_points_file": Path("data/demand/sample_population_points.csv"),
            "task_min_elevation_deg": 0.0,
            "task_deadline_s": 120.0,
            "compute_cycles_per_input_bit": 1.0,
            "satellite_cpu_frequency_hz": 1.0e9,
            "satellite_cpu_power_w": 1.0,
            "isl_rate_bps": 1.0e9,
            "isl_tx_power_w": 0.0,
            "isl_topology": "grid",
            "isl_max_range_km": 5000.0,
            "out": Path("output/base"),
            "scheduler_cpu_utilization_limit": 1.0,
            "scheduler_defer_penalty": 3.0,
            "scheduler_fail_penalty": 1000.0,
            "scheduler_time_weight": 1.0,
            "scheduler_energy_weight": 2.0,
            "scheduler_battery_weight": 5.0,
            "scheduler_load_weight": 0.1,
            "scheduler_eclipse_local_penalty": 2.0,
            "scheduler_low_battery_threshold_pct": 35.0,
        }

        args = demand_energy_sweep.scenario_args(
            values,
            out=Path("output/scenario"),
            data_size_bits=123.0,
            slot_interval_s=30,
            duration_s=60,
            deadline_s=None,
            cpu_power_w=25.0,
        )

        self.assertEqual(args.scheduler, "local")
        self.assertEqual(args.task_generation_mode, "demand-points-fixed-all")
        self.assertEqual(args.task_input_bits, 123.0)
        self.assertEqual(args.task_output_bits, 0.0)
        self.assertEqual(args.task_interval_s, 30)
        self.assertTrue(math.isclose(args.task_deadline_s, 1.0e12))
        self.assertEqual(args.satellite_cpu_power_w, 25.0)

    def test_scenario_args_can_split_global_total_demand(self) -> None:
        values = {
            "run_name": "base",
            "run_description": "",
            "orbit_model": "circular",
            "tle_file": None,
            "sun_position_file": "de440s.bsp",
            "satellites": 1,
            "planes": 1,
            "altitude_km": 550.0,
            "inclination_deg": 53.05,
            "duration_s": 30,
            "step_s": 30,
            "walker_phase": 1,
            "battery_capacity_j": 1000.0,
            "battery_initial_pct": 100.0,
            "battery_min_safe_pct": 70.0,
            "harvest_w": 0.0,
            "idle_w": 0.0,
            "task_enable": True,
            "scheduler": "nearest-sunlit",
            "task_interval_s": 300,
            "task_generation_mode": "demand-points",
            "task_random_seed": 42,
            "tasks_per_sat": 1,
            "tasks_per_step_choices": [1],
            "tasks_per_step_weights": [1.0],
            "task_input_bits": 1.0,
            "task_input_bits_choices": [1.0],
            "task_input_bits_weights": [1.0],
            "task_output_bits": 1.0,
            "task_output_bits_choices": [1.0],
            "task_output_bits_weights": [1.0],
            "task_demand_points_file": Path("data/demand/sample_population_points.csv"),
            "task_min_elevation_deg": 0.0,
            "task_deadline_s": 120.0,
            "compute_cycles_per_input_bit": 1.0,
            "satellite_cpu_frequency_hz": 1.0e9,
            "satellite_cpu_power_w": 1.0,
            "isl_rate_bps": 1.0e9,
            "isl_tx_power_w": 0.0,
            "isl_topology": "grid",
            "isl_max_range_km": 5000.0,
            "out": Path("output/base"),
            "scheduler_cpu_utilization_limit": 1.0,
            "scheduler_defer_penalty": 3.0,
            "scheduler_fail_penalty": 1000.0,
            "scheduler_time_weight": 1.0,
            "scheduler_energy_weight": 2.0,
            "scheduler_battery_weight": 5.0,
            "scheduler_load_weight": 0.1,
            "scheduler_eclipse_local_penalty": 2.0,
            "scheduler_low_battery_threshold_pct": 35.0,
        }

        args = demand_energy_sweep.scenario_args(
            values,
            out=Path("output/scenario"),
            data_size_bits=123.0,
            slot_interval_s=30,
            duration_s=60,
            deadline_s=None,
            global_total_demand=True,
        )

        self.assertEqual(
            args.task_generation_mode,
            "demand-points-fixed-weighted-all",
        )


    def test_config_float_values_accepts_scalar_or_list(self) -> None:
        self.assertEqual(
            demand_energy_sweep.config_float_values(30.0, "compute.cpu_power_w"),
            [30.0],
        )
        self.assertEqual(
            demand_energy_sweep.config_float_values([10, 20.5], "compute.cpu_power_w"),
            [10.0, 20.5],
        )

    def test_scenario_output_dir_separates_constellation_and_cpu_power(self) -> None:
        path = demand_energy_sweep.scenario_output_dir(
            Path("output/root"),
            1.0e8,
            30,
            cpu_power_w=20.0,
            constellation="Kuiper 784",
        )

        self.assertEqual(
            path,
            Path("output/root/runs/kuiper_784/cpu_20w/data_1e08_bits/slot_30s"),
        )

    def test_write_line_outputs_groups_by_constellation_slot_and_cpu(self) -> None:
        rows = [
            {
                "constellation": "Starlink 1584",
                "cpu_power_w": 10.0,
                "data_size_bits": 1.0e6,
                "slot_interval_s": 30,
                "unique_breached_ratio": 0.1,
                "unique_breached_satellites": 1,
                "unique_eclipse_breached_ratio": 0.1,
                "unique_eclipse_breached_satellites": 1,
                "tasks_generated": 10,
                "tasks_completed": 9,
                "tasks_failed": 1,
                "tasks_pending": 0,
                "run_dir": "run/a",
            },
            {
                "constellation": "Starlink 1584",
                "cpu_power_w": 20.0,
                "data_size_bits": 1.0e6,
                "slot_interval_s": 30,
                "unique_breached_ratio": 0.2,
                "unique_breached_satellites": 2,
                "unique_eclipse_breached_ratio": 0.2,
                "unique_eclipse_breached_satellites": 2,
                "tasks_generated": 10,
                "tasks_completed": 8,
                "tasks_failed": 2,
                "tasks_pending": 0,
                "run_dir": "run/b",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "battery_breach_ratio_line"
            demand_energy_sweep.write_line_outputs(prefix, rows)

            csv_text = prefix.with_suffix(".csv").read_text()
            svg_text = prefix.with_suffix(".svg").read_text()

        self.assertIn("cpu_power_w", csv_text)
        self.assertIn("10W", svg_text)
        self.assertIn("20W", svg_text)
        self.assertIn("polyline", svg_text)
        self.assertIn(">0.0</text>", svg_text)
        self.assertIn(">1.0</text>", svg_text)


if __name__ == "__main__":
    unittest.main()
