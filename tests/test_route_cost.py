import unittest

from satmulator.models import ComputeConfig, DemandDistribution, ISLConfig, Route, Task, TaskConfig
from satmulator.route_cost import estimate_route_cost


def task_config() -> TaskConfig:
    return TaskConfig(
        enabled=True,
        interval_s=30,
        generation_mode="satellite-deterministic",
        random_seed=1,
        tasks_per_sat=1,
        tasks_per_step_choices=(1,),
        tasks_per_step_weights=(1.0,),
        input_bits=100.0,
        input_bits_choices=(100.0,),
        input_bits_weights=(1.0,),
        output_bits=10.0,
        output_bits_choices=(10.0,),
        output_bits_weights=(1.0,),
        deadline_s=30.0,
        demand_distribution=DemandDistribution((), (), 0.0),
        min_elevation_deg=30.0,
    )


def task() -> Task:
    return Task(
        task_id=1,
        created_time_s=0,
        source_sat=0,
        input_bits=100.0,
        output_bits=10.0,
        deadline_s=30.0,
    )


class RouteCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = task()
        self.task_config = task_config()
        self.compute = ComputeConfig(
            cycles_per_input_bit=10.0,
            cpu_frequency_hz=100.0,
            cpu_power_w=50.0,
        )
        self.isl = ISLConfig(
            rate_bps=10.0,
            tx_power_w=2.0,
        )

    def test_local_route_matches_local_execution(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0,)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(cost.compute_time_s, 10.0)
        self.assertEqual(cost.transmission_time_s, 0.0)
        self.assertEqual(cost.energy_by_sat, {0: 500.0})
        self.assertEqual(cost.total_energy_j, 500.0)

    def test_one_hop_route_charges_transmit_power(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0, 1)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(cost.compute_time_s, 10.0)
        self.assertEqual(cost.transmission_time_s, 11.0)
        self.assertEqual(cost.energy_by_sat, {0: 20.0, 1: 502.0})
        self.assertEqual(cost.total_energy_j, 522.0)

    def test_multi_hop_charges_relay_for_forward_and_return(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0, 2, 1)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(cost.transmission_time_s, 22.0)
        self.assertEqual(cost.energy_by_sat[0], 20.0)
        self.assertEqual(cost.energy_by_sat[2], 22.0)
        self.assertEqual(cost.energy_by_sat[1], 502.0)

    def test_explicit_compute_time_overrides_cycle_model(self) -> None:
        explicit = Task(
            task_id=2,
            created_time_s=0,
            source_sat=0,
            input_bits=100.0,
            output_bits=10.0,
            deadline_s=30.0,
            compute_time_s=5.0,
        )

        cost = estimate_route_cost(
            task=explicit,
            route=Route((0,)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(cost.compute_time_s, 5.0)
        self.assertEqual(cost.energy_by_sat, {0: 250.0})


if __name__ == "__main__":
    unittest.main()
