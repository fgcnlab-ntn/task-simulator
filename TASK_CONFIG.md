# Task Configuration Report

## Core Judgment: Worth doing

The old task model is satellite-oriented: each satellite creates a fixed number
of identical tasks.  That is predictable, but it is not a realistic demand
model.  The new config keeps that old mode for compatibility and adds a
`demand-points` mode where tasks are sampled from weighted ground demand points.
Controlled battery-load sweeps use `demand-points-fixed-all`, which emits one
fixed-size task for every demand point at every generation slot.

## Task fields

- `enabled`: enable or disable task generation.
- `interval_s`: generate tasks every N simulated seconds.
- `generation_mode`: `satellite-deterministic`, `demand-points`, or
  `demand-points-fixed-all`.
- `random_seed`: seed for reproducible stochastic workloads.
- `tasks_per_sat`: legacy deterministic task count per satellite.
- `tasks_per_step_choices`, `tasks_per_step_weights`: discrete distribution for
  the number of tasks created at each generation time in `demand-points` mode.
- `demand_points_file`: CSV file with `lat,lon,weight` columns.  The weight can
  come from population, nighttime lights, or measured traffic demand. The
  default demand-point config uses the checked-in 5° global WorldPop aggregate
  at `data/demand/global_population_2025_5deg.csv`.
- `min_elevation_deg`: minimum ground-to-satellite elevation angle used when
  selecting a serving satellite. Defaults to 30 degrees. Tasks wait while no
  satellite meets the threshold and fail with `no_coverage` when their deadline
  expires.
- `input_bits_choices`, `input_bits_weights`: discrete input data distribution.
- `output_bits_choices`, `output_bits_weights`: discrete output data distribution.
- `deadline_s`: task deadline.

`demand-points-fixed-all` deliberately does not use the random choice fields.
At each generation time after the initial state, every configured demand point
creates exactly one task using `input_bits` and `output_bits`. The nearest
satellite satisfying `min_elevation_deg` is selected immediately. If no
satellite is visible, the task fails immediately with `no_coverage`; it is not
queued or deferred.

## Compute fields

- `cycles_per_input_bit`: conversion from input data size to compute cycles.
  The simulator uses `compute_cycles = input_bits * cycles_per_input_bit`.
- `cpu_frequency_hz`: satellite CPU frequency used by the compute time model.
- `cpu_power_w`: active satellite CPU power used by the compute energy model.

Compute time and energy are derived:

```text
compute_time_s = compute_cycles / cpu_frequency_hz
compute_energy_j = compute_time_s * cpu_power_w
```

## Scheduler compute capacity

- `cpu_utilization_limit`: fraction of one CPU slot that the scheduler may
  reserve for tasks. The slot capacity is derived from the compute model:

```text
max_cycles_per_slot = cpu_frequency_hz * step_s * cpu_utilization_limit
```

The default is `1.0`, meaning the modeled CPU may be fully reserved for compute
work during a step. Use a lower value such as `0.8` only when you want explicit
headroom for unmodeled platform work or thermal throttling.

Task records include `waiting_time_s`. Waiting for coverage counts toward
`total_time_s` and the task deadline.

## Objective fields

Standalone configs include an `objective` section:

```json
"objective": {
  "alpha": 0.5
}
```

`alpha` is the reporting weight for the simulator objective summary and must be
within `[0, 1]`. It does not alter task generation, assignment, or scheduler
behavior. `summary.json` reports:

```text
objective.value =
  alpha * avg_eclipse_unsafe_ratio
  + (1 - alpha) * task_failure_ratio
```

`avg_eclipse_unsafe_ratio` averages the per-step ratio of unsafe eclipse-side
satellites over all eclipse-side satellites. `task_failure_ratio` uses the
existing task lifecycle counters: `failed / generated`. Tasks still pending at
the end of the simulation are treated as non-failed in this objective summary;
the summary records this policy as `pending_policy: "count_as_success"`.

## Population data source

The population-weighted baseline uses WorldPop 2025 R2025A constrained
population-count products at 1 km resolution for both Taiwan and global
experiments. They are converted offline into the common `lat,lon,weight`
format. Exact sources are recorded in `data/worldpop/README.md`.
When a demand CSV has an adjacent `.metadata.json` file, its source URL,
aggregation resolution, retained population, conversion parameters, and input
information are copied into `run.json`. The demand point count and total weight
are always recorded.

## Compatibility

The default config still uses `satellite-deterministic`, so old runs keep the
same behavior.  The new task-oriented mode is enabled by `configs/demand_points.json`.
