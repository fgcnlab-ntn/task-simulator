# Task Configuration Report

## Core Judgment: Worth doing

Tasks are sampled from weighted ground demand points. This keeps workload
generation tied to geographic demand instead of assigning an artificial fixed
task count to every satellite.

## Task fields

- `interval_s`: generate tasks every N simulated seconds.
- `random_seed`: seed for reproducible stochastic workloads.
- `tasks_per_step`: number of tasks created at each generation time.
- `demand_points_file`: CSV file with `lat,lon,weight` columns.  The weight can
  come from population, nighttime lights, or measured traffic demand. The
  default demand-point config uses the checked-in 5° global WorldPop aggregate
  at `data/demand/global_population_2025_5deg.csv`.
- `min_elevation_deg`: minimum ground-to-satellite elevation angle used when
  selecting a serving satellite. Defaults to 30 degrees. Tasks wait while no
  satellite meets the threshold and fail with `no_coverage` when their deadline
  expires.
- `input_bits`, `output_bits`: fixed input and output sizes for every generated
  task.
Task deadlines always use a lower-truncated normal distribution.

- `deadline_s`: mean of the source normal deadline distribution.
- `deadline_min_s`: lower bound for deadlines. The source normal's
  standard deviation is derived as `(deadline_s - deadline_min_s) / 3`.
  Samples below this bound are discarded and redrawn.

For example, this uses a 180-second source mean, rejects deadlines below 30
seconds, and derives a 50-second source standard deviation:

```json
"deadline_s": 180.0,
"deadline_min_s": 30.0
```

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

Task records include `waiting_time_s`. Waiting for coverage counts toward
`total_time_s` and the task deadline.

`starlit` and `phoenix` use non-preemptive EDF execution queues. Before an
assignment is admitted, the simulator inserts it into a projection of the
target satellite's complete queue, keeps any partially executed task at the
front, sorts the remaining tasks by absolute deadline, and verifies every
projected completion time. After admission, the runtime queue is ordered by
the same rule before the time slot executes.

## Summary metrics

`summary.json` reports `battery_violations.avg_eclipse_unsafe_ratio`, the
average per-step ratio of unsafe eclipse-side satellites over all eclipse-side
satellites, and `tasks.failure_ratio`, calculated from the task lifecycle
counters as `failed / generated`. Tasks still pending at the end of the
simulation are treated as non-failed; `tasks.pending_policy` records this as
`"count_as_success"`.

## Population data source

The population-weighted baseline uses WorldPop 2025 R2025A constrained
population-count products at 1 km resolution for both Taiwan and global
experiments. They are converted offline into the common `lat,lon,weight`
format. Exact sources are recorded in `data/worldpop/README.md`.
When a demand CSV has an adjacent `.metadata.json` file, its source URL,
aggregation resolution, retained population, conversion parameters, and input
information are copied into `run.json`. The demand point count and total weight
are always recorded.
