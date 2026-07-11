from __future__ import annotations

from satmulator.isl import ISLGraph
from satmulator.models import (
    BatteryConfig,
    ComputeConfig,
    DemandDistribution,
    ISLConfig,
    SatelliteView,
    SchedulerConfig,
    Task,
    TaskConfig,
)
from satmulator.scheduler import GreedyEnergyScheduler, create_scheduler


def _battery(
    *,
    min_safe_j: float = 0.0,
) -> BatteryConfig:
    return BatteryConfig(
        capacity_j=100.0,
        initial_j=100.0,
        min_safe_j=min_safe_j,
        harvest_w=0.0,
        idle_w=0.0,
    )


def _compute() -> ComputeConfig:
    return ComputeConfig(
        cycles_per_input_bit=1.0,
        cpu_frequency_hz=1.0,
        cpu_power_w=1.0,
    )


def _task_config() -> TaskConfig:
    return TaskConfig(
        enabled=True,
        interval_s=1,
        generation_mode="satellite-deterministic",
        random_seed=None,
        tasks_per_sat=1,
        tasks_per_step_choices=(1,),
        tasks_per_step_weights=(1.0,),
        input_bits=1.0,
        input_bits_choices=(1.0,),
        input_bits_weights=(1.0,),
        output_bits=0.0,
        output_bits_choices=(0.0,),
        output_bits_weights=(1.0,),
        deadline_s=1.5,
        demand_distribution=DemandDistribution((), (), 0.0),
        min_elevation_deg=0.0,
    )


def _isl(*, tx_power_w: float = 0.0) -> ISLConfig:
    return ISLConfig(
        rate_bps=1.0,
        tx_power_w=tx_power_w,
        topology="grid",
        max_range_km=5000.0,
    )


def _task(
    task_id: int,
    *,
    deadline_s: float = 1.5,
    input_bits: float = 0.0,
) -> Task:
    return Task(
        task_id=task_id,
        created_time_s=0,
        source_sat=0,
        input_bits=input_bits,
        output_bits=0.0,
        deadline_s=deadline_s,
        compute_time_s=1.0,
    )


def _assign(
    *,
    tasks: list[Task],
    satellites: list[SatelliteView],
    graph: ISLGraph,
    step_s: int = 1,
    battery: BatteryConfig | None = None,
    isl: ISLConfig | None = None,
) -> list:
    return GreedyEnergyScheduler().assign_tasks(
        tasks=tasks,
        satellite_views=satellites,
        time_s=0,
        step_s=step_s,
        battery=_battery() if battery is None else battery,
        compute_config=_compute(),
        task_config=_task_config(),
        isl_config=_isl() if isl is None else isl,
        isl_graph=graph,
        scheduler_config=SchedulerConfig(name="greedy-energy"),
    )


def test_factory_creates_greedy_energy_scheduler() -> None:
    assert isinstance(create_scheduler("greedy-energy"), GreedyEnergyScheduler)


def test_sunlit_source_uses_local_when_it_is_not_slower() -> None:
    satellites = [
        SatelliteView(sat_id=0, x_km=0, y_km=0, z_km=0, sunlit=True, battery_j=100),
        SatelliteView(sat_id=1, x_km=1, y_km=0, z_km=0, sunlit=True, battery_j=100),
    ]
    assignments = _assign(
        tasks=[_task(0)],
        satellites=satellites,
        graph=ISLGraph({0: (1,), 1: (0,)}),
        step_s=4,
    )

    assert len(assignments) == 1
    assert assignments[0].route.nodes == (0,)
    assert assignments[0].mode == "local"


def test_eclipse_source_prefers_sunlit_relay_to_preserve_battery() -> None:
    satellites = [
        SatelliteView(sat_id=0, x_km=0, y_km=0, z_km=0, sunlit=False, battery_j=66.5),
        SatelliteView(sat_id=1, x_km=1, y_km=0, z_km=0, sunlit=True, battery_j=100),
    ]
    assignments = _assign(
        tasks=[_task(0, deadline_s=4.0), _task(1, deadline_s=4.0)],
        satellites=satellites,
        graph=ISLGraph({0: (1,), 1: (0,)}),
        step_s=4,
        battery=_battery(),
    )

    assert [assignment.mode for assignment in assignments] == ["relay", "relay"]
    assert [assignment.route.nodes for assignment in assignments] == [(0, 1), (0, 1)]


def test_sunlit_source_overflow_uses_remote_sunlit_compute() -> None:
    satellites = [
        SatelliteView(sat_id=0, x_km=0, y_km=0, z_km=0, sunlit=True, battery_j=100),
        SatelliteView(sat_id=1, x_km=1, y_km=0, z_km=0, sunlit=True, battery_j=100),
    ]
    assignments = _assign(
        tasks=[_task(0), _task(1)],
        satellites=satellites,
        graph=ISLGraph({0: (1,), 1: (0,)}),
        step_s=1,
    )

    assert [assignment.mode for assignment in assignments] == ["local", "relay"]
    assert [assignment.route.nodes for assignment in assignments] == [(0,), (0, 1)]


def test_source_defers_when_no_sunlit_remote_compute_is_available() -> None:
    satellites = [
        SatelliteView(sat_id=0, x_km=0, y_km=0, z_km=0, sunlit=False, battery_j=66.5),
        SatelliteView(sat_id=1, x_km=1, y_km=0, z_km=0, sunlit=False, battery_j=100),
    ]
    assignments = _assign(
        tasks=[_task(0, deadline_s=4.0), _task(1, deadline_s=4.0)],
        satellites=satellites,
        graph=ISLGraph({0: (1,), 1: (0,)}),
        step_s=4,
        battery=_battery(),
    )

    assert [assignment.mode for assignment in assignments] == ["local", "defer"]
    assert assignments[1].route.nodes == (0,)


def test_low_battery_eclipse_source_relays_before_local_processing() -> None:
    satellites = [
        SatelliteView(sat_id=0, x_km=0, y_km=0, z_km=0, sunlit=False, battery_j=64),
        SatelliteView(sat_id=1, x_km=1, y_km=0, z_km=0, sunlit=True, battery_j=100),
    ]
    assignments = _assign(
        tasks=[_task(0, deadline_s=4.0)],
        satellites=satellites,
        graph=ISLGraph({0: (1,), 1: (0,)}),
        step_s=4,
        battery=_battery(),
    )

    assert assignments[0].mode == "relay"
    assert assignments[0].route.nodes == (0, 1)


def test_equal_battery_cost_uses_finish_time_before_total_energy() -> None:
    satellites = [
        SatelliteView(
            sat_id=0,
            x_km=0,
            y_km=0,
            z_km=0,
            sunlit=True,
            battery_j=100,
            queue_backlog_s=2.0,
        ),
        SatelliteView(sat_id=1, x_km=1, y_km=0, z_km=0, sunlit=True, battery_j=100),
    ]
    assignments = _assign(
        tasks=[_task(0, deadline_s=10.0, input_bits=1.0)],
        satellites=satellites,
        graph=ISLGraph({0: (1,), 1: (0,)}),
        step_s=4,
        isl=_isl(tx_power_w=1.0),
    )

    assert assignments[0].mode == "relay"
    assert assignments[0].route.nodes == (0, 1)
    assert assignments[0].score == 0.0
