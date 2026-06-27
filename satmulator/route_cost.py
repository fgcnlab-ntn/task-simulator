from __future__ import annotations

from dataclasses import dataclass

from .models import ISLConfig, Route, Task, TaskConfig


@dataclass(frozen=True)
class RouteCost:
    compute_time_s: float
    transmission_time_s: float
    energy_by_sat: dict[int, float]

    @property
    def total_time_s(self) -> float:
        return self.compute_time_s + self.transmission_time_s

    @property
    def total_energy_j(self) -> float:
        return sum(self.energy_by_sat.values())

    def energy_for(self, sat_id: int) -> float:
        return self.energy_by_sat.get(sat_id, 0.0)


def add_energy(energy_by_sat: dict[int, float], sat_id: int, energy_j: float) -> None:
    if energy_j == 0.0:
        return
    energy_by_sat[sat_id] = energy_by_sat.get(sat_id, 0.0) + energy_j


def transfer_time_s(bits: float, isl_config: ISLConfig) -> float:
    return bits / isl_config.rate_bps


def transmission_energy_j(bits: float, isl_config: ISLConfig) -> float:
    return isl_config.tx_power_w * transfer_time_s(bits, isl_config)


def estimate_route_cost(
    *,
    task: Task,
    route: Route,
    task_config: TaskConfig,
    isl_config: ISLConfig,
) -> RouteCost:
    """Estimate execution cost for a task over a route.

    A single-node route means local execution.  A multi-node route sends input
    forward from source to target, computes at the target, then sends output
    back along the same path in reverse.  The two-node route is exactly the
    historical one-hop model.
    """

    compute_time_s = task.cpu_cycles / task_config.cpu_rate_cycles_s
    compute_energy_j = task.cpu_cycles * task_config.joule_per_cycle
    energy_by_sat: dict[int, float] = {}

    add_energy(energy_by_sat, route.target_sat, compute_energy_j)

    if route.hop_count == 0:
        return RouteCost(
            compute_time_s=compute_time_s,
            transmission_time_s=0.0,
            energy_by_sat=energy_by_sat,
        )

    forward_hops = tuple(zip(route.nodes, route.nodes[1:]))
    input_tx_energy_j = transmission_energy_j(task.input_bits, isl_config)
    output_tx_energy_j = transmission_energy_j(task.output_bits, isl_config)

    for sender, _receiver in forward_hops:
        add_energy(
            energy_by_sat,
            sender,
            input_tx_energy_j,
        )

    for _receiver, sender in reversed(forward_hops):
        add_energy(
            energy_by_sat,
            sender,
            output_tx_energy_j,
        )

    transmission_time_s = route.hop_count * (
        transfer_time_s(task.input_bits, isl_config)
        + transfer_time_s(task.output_bits, isl_config)
    )
    return RouteCost(
        compute_time_s=compute_time_s,
        transmission_time_s=transmission_time_s,
        energy_by_sat=energy_by_sat,
    )
