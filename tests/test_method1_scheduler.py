from __future__ import annotations

import pytest

import satmulator.scheduler as scheduler_module
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
from satmulator.scheduler import Method1Scheduler


def test_method1_uses_one_source_route_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    satellites = [
        SatelliteView(sat_id=0, x_km=0, y_km=0, z_km=0, sunlit=True, battery_j=100),
        SatelliteView(sat_id=1, x_km=1, y_km=0, z_km=0, sunlit=True, battery_j=100),
        SatelliteView(sat_id=2, x_km=2, y_km=0, z_km=0, sunlit=True, battery_j=100),
    ]
    graph = ISLGraph({0: (1,), 1: (0, 2), 2: (1,)})
    tasks = [
        Task(
            task_id=0,
            created_time_s=0,
            source_sat=0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=100.0,
        )
    ]

    def fail_shortest_route(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("method1 should use routes_from_source, not per-target BFS")

    monkeypatch.setattr(scheduler_module, "shortest_route", fail_shortest_route)

    assignments = Method1Scheduler().assign_tasks(
        tasks=tasks,
        satellite_views=satellites,
        time_s=0,
        step_s=30,
        battery=BatteryConfig(
            capacity_j=100.0,
            initial_j=100.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        ),
        compute_config=ComputeConfig(
            cycles_per_input_bit=1.0,
            cpu_frequency_hz=1.0,
            cpu_power_w=1.0,
        ),
        task_config=TaskConfig(
            enabled=True,
            interval_s=30,
            generation_mode="satellite-deterministic",
            random_seed=None,
            tasks_per_sat=1,
            tasks_per_step_choices=(1,),
            tasks_per_step_weights=(1.0,),
            input_bits=1.0,
            input_bits_choices=(1.0,),
            input_bits_weights=(1.0,),
            output_bits=1.0,
            output_bits_choices=(1.0,),
            output_bits_weights=(1.0,),
            deadline_s=100.0,
            demand_distribution=DemandDistribution((), (), 0.0),
            min_elevation_deg=0.0,
        ),
        isl_config=ISLConfig(
            rate_bps=1.0,
            tx_power_w=0.0,
            topology="grid",
            max_range_km=5000.0,
        ),
        isl_graph=graph,
        scheduler_config=SchedulerConfig(name="method1"),
    )

    assert len(assignments) == 1
    assert assignments[0].route.nodes == (0,)
    assert assignments[0].mode == "local"
