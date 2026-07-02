import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "tools" / "p_cut_experiment.py"
SPEC = importlib.util.spec_from_file_location("p_cut_experiment", SCRIPT)
p_cut_experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(p_cut_experiment)


class PCutExperimentTests(unittest.TestCase):
    def test_energy_sweep_is_linear_in_cpu_power(self) -> None:
        results = p_cut_experiment.energy_sweep(
            cpu_powers_w=[0.0, 10.0, 20.0],
            eclipse_duration_s=100.0,
            idle_w=4.0,
        )

        self.assertEqual(results[0]["cpu_energy_j"], 0.0)
        self.assertEqual(results[1]["cpu_energy_j"], 1000.0)
        self.assertEqual(results[2]["cpu_energy_j"], 2000.0)
        self.assertEqual(results[2]["total_eclipse_energy_j"], 2400.0)
        self.assertEqual(results[2]["scope"], "single_satellite")

    def test_safe_battery_energy_uses_initial_to_safe_capacity(self) -> None:
        args = SimpleNamespace(
            battery_capacity_j=1000.0,
            battery_initial_pct=80.0,
            battery_min_safe_pct=20.0,
            satellites=3,
        )

        self.assertEqual(p_cut_experiment.energy_to_safe_battery_j(args), 600.0)

    def test_uses_kj_for_single_satellite_scale(self) -> None:
        self.assertEqual(p_cut_experiment.energy_unit(172800.0), ("kJ", 1000.0))

    def test_tick_values_are_readable(self) -> None:
        ticks, axis_max = p_cut_experiment.nice_tick_values(
            172.8,
            target_count=5,
        )

        self.assertEqual(axis_max, 200)
        self.assertEqual(ticks, [0, 50, 100, 150, 200])

    def test_p_cut_power_subtracts_idle_energy(self) -> None:
        self.assertEqual(
            p_cut_experiment.p_cut_power_w(
                safe_energy_j=108000.0,
                idle_energy_j=7680.0,
                eclipse_seconds=1920.0,
            ),
            52.25,
        )


if __name__ == "__main__":
    unittest.main()
