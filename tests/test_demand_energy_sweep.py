import importlib.util
import math
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "demand_energy_sweep.py"
SPEC = importlib.util.spec_from_file_location("demand_energy_sweep", SCRIPT)
demand_energy_sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demand_energy_sweep)


class DemandEnergySweepTests(unittest.TestCase):
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
        )

        self.assertEqual(args.scheduler, "local")
        self.assertEqual(args.task_generation_mode, "demand-points-fixed-all")
        self.assertEqual(args.task_input_bits, 123.0)
        self.assertEqual(args.task_output_bits, 0.0)
        self.assertEqual(args.task_interval_s, 30)
        self.assertTrue(math.isclose(args.task_deadline_s, 1.0e12))

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


if __name__ == "__main__":
    unittest.main()
