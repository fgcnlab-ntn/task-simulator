import unittest

from satmulator.models import DemandDistribution, ISLConfig, Route, Task, TaskConfig
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
        cpu_cycles=1000.0,
        cpu_cycles_choices=(1000.0,),
        cpu_cycles_weights=(1.0,),
        input_bits=100.0,
        input_bits_choices=(100.0,),
        input_bits_weights=(1.0,),
        output_bits=10.0,
        output_bits_choices=(10.0,),
        output_bits_weights=(1.0,),
        deadline_s=30.0,
        cpu_rate_cycles_s=100.0,
        joule_per_cycle=0.5,
        demand_distribution=DemandDistribution((), (), 0.0),
        min_elevation_deg=30.0,
    )


def task() -> Task:
    return Task(
        task_id=1,
        created_time_s=0,
        source_sat=0,
        cpu_cycles=1000.0,
        input_bits=100.0,
        output_bits=10.0,
        deadline_s=30.0,
    )


class RouteCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = task()
        self.task_config = task_config()
        self.isl = ISLConfig(
            isl_forward_rate_bps=10.0,
            isl_return_rate_bps=5.0,
            isl_tx_energy_per_bit_j=2.0,
            isl_rx_energy_per_bit_j=1.0,
        )

    def test_local_route_matches_local_execution(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0,)),
            task_config=self.task_config,
            isl_config=self.isl,
        )

        self.assertEqual(cost.compute_time_s, 10.0)
        self.assertEqual(cost.transmission_time_s, 0.0)
        self.assertEqual(cost.energy_by_sat, {0: 500.0})
        self.assertEqual(cost.total_energy_j, 500.0)

    def test_one_hop_route_matches_existing_endpoint_formula(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0, 1)),
            task_config=self.task_config,
            isl_config=self.isl,
        )

        old_source = 100.0 * 2.0 + 10.0 * 1.0
        old_target = 100.0 * 1.0 + 500.0 + 10.0 * 2.0
        self.assertEqual(cost.compute_time_s, 10.0)
        self.assertEqual(cost.transmission_time_s, 12.0)
        self.assertEqual(cost.energy_by_sat, {0: old_source, 1: old_target})
        self.assertEqual(cost.total_energy_j, old_source + old_target)

    def test_multi_hop_charges_relay_for_forward_and_return(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0, 2, 1)),
            task_config=self.task_config,
            isl_config=self.isl,
        )

        self.assertEqual(cost.transmission_time_s, 24.0)
        self.assertEqual(cost.energy_by_sat[0], 100.0 * 2.0 + 10.0 * 1.0)
        self.assertEqual(
            cost.energy_by_sat[2],
            100.0 * 1.0 + 100.0 * 2.0 + 10.0 * 1.0 + 10.0 * 2.0,
        )
        self.assertEqual(cost.energy_by_sat[1], 100.0 * 1.0 + 500.0 + 10.0 * 2.0)


if __name__ == "__main__":
    unittest.main()
