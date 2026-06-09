# Task Configuration Report

## Core Judgment: Worth doing

The old task model is satellite-oriented: each satellite creates a fixed number
of identical tasks.  That is predictable, but it is not a realistic demand
model.  The new config keeps that old mode for compatibility and adds a
`demand-points` mode where tasks are sampled from weighted ground demand points.

## Task fields

- `enabled`: enable or disable task generation.
- `interval_s`: generate tasks every N simulated seconds.
- `generation_mode`: `satellite-deterministic` or `demand-points`.
- `random_seed`: seed for reproducible stochastic workloads.
- `tasks_per_sat`: legacy deterministic task count per satellite.
- `tasks_per_step_choices`, `tasks_per_step_weights`: discrete distribution for
  the number of tasks created at each generation time in `demand-points` mode.
- `demand_points_file`: CSV file with `lat,lon,weight` columns.  The weight can
  come from population, nighttime lights, or measured traffic demand.
- `min_elevation_deg`: minimum ground-to-satellite elevation angle used when
  selecting a serving satellite. Defaults to 30 degrees. Tasks wait while no
  satellite meets the threshold and fail with `no_coverage` when their deadline
  expires.
- `cpu_cycles_choices`, `cpu_cycles_weights`: discrete CPU demand distribution.
- `input_bits_choices`, `input_bits_weights`: discrete input data distribution.
- `output_bits_choices`, `output_bits_weights`: discrete output data distribution.
- `deadline_s`: task deadline.
- `cpu_rate_cycles_s`: compute rate used by the time model.
- `joule_per_cycle`: compute energy coefficient.

Task records include `waiting_time_s`. Waiting for coverage counts toward
`total_time_s` and the task deadline.

## Data source direction

The immediate implementation uses a small CSV interface so the simulator is not
blocked by GeoTIFF processing.  The same `lat,lon,weight` format can later be
produced from NASA SEDAC GPWv4 population grids, WorldPop, or NASA Black Marble
nighttime-light data.

## Compatibility

The default config still uses `satellite-deterministic`, so old runs keep the
same behavior.  The new task-oriented mode is enabled by `configs/demand_points.json`.
