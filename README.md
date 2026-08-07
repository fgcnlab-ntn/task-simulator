# Satmulator

Minimal NTN satellite-state simulator for energy-aware task execution and
offloading experiments.

The current model supports:

- circular Walker-style orbit model
- sunlight/eclipse classification
- per-satellite battery state
- population-weighted demand-point task generation
- local and nearest-sunlit schedulers
- per-satellite FIFO execution queues for assigned tasks
- four-neighbor grid ISL routing with per-hop accounting
- target load limits in CPU cycles per slot
- summary metrics for eclipse unsafe ratio and task failure ratio
- structured JSON/JSONL experiment logs

It does not yet model queueing, link contention, or thermal throttling dynamics.

## Install

Sun-position calculation and demand-point coordinate conversion require
Skyfield:

```bash
python3 -m pip install -r requirements.txt
```

`de440s.bsp` is not tracked by git. The simulator uses it for Sun-position
calculation; Skyfield can download it on first use:

```bash
python3 -c "from skyfield.api import load; load('de440s.bsp')"
```

### Ephemeris data source
Solar positions are computed using the JPL DE440s ephemeris
(`de440s.bsp`) loaded through Skyfield.

References:

- Ephemeris data (DE440s): https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440s.bsp
- DE440 technical paper: https://doi.org/10.3847/1538-3881/abd414
- Skyfield documentation: https://rhodesmill.org/skyfield/planets.html

## Run

Use JSON config files for reproducible runs:

```bash
python3 minimal_orbit.py --config configs/loading-ratio/r100/local-only.json
```

CPU-power sweep for one satellite keeping its CPU fully active through a
32-minute eclipse interval:

```bash
python3 tools/p_cut_experiment.py --out experiments/P_cut
```

Deterministic demand-load sweep for fixed data size and time-slot intervals:

```bash
python3 tools/demand_energy_sweep.py \
  --data-sizes-bits 1e6,1e7,1e8 \
  --slot-intervals-s 30,60,120,300
```

Population-weighted demand inputs use WorldPop 2025 R2025A constrained 1 km
population-count products for both Taiwan and global experiments. See
`data/worldpop/README.md` for the exact sources and checksums.
The default global demand config uses `data/demand/global_population_2025_5deg.csv`,
which aggregates the global raster into 5° latitude/longitude cells. This keeps
the controlled demand-energy sweep small enough for repeated runs while
preserving essentially all source population.

`--config` is required. Every simulation and output setting comes from the
complete standalone config; the CLI does not override individual fields.

The effective config is written to:

```text
<output>/run.json
```

## Model

The grid builds a fixed candidate layout once: two in-plane links and two
cross-plane links per satellite. The plane seam is shifted by the configured
Walker phase. At each simulation step, only range and Earth line-of-sight are
reevaluated to determine which candidate links are active. Diagonal satellites
therefore require at least two hops.

## Outputs

Each run writes:

- `run.json` — structured run status, effective config, and satellite catalog
- `states.jsonl` — one append-safe satellite-state record per simulation step
- `tasks.jsonl` — append-safe task lifecycle events
- `summary.json` — final structured result summary

JSON/JSONL files are the structured experiment log. Paper figures are generated
separately by the scripts under `tools/`.

`battery_violations.avg_eclipse_unsafe_ratio` is the mean over simulation
steps of `unsafe eclipse satellites / eclipse satellites`.
`tasks.failure_ratio` is calculated as `failed / generated` using the task
lifecycle counters in the same summary. Tasks that are still pending when the
simulation ends are counted as non-failed; `tasks.pending_policy` records this
as `"count_as_success"`.

The `P_cut` experiment writes outputs under `experiments/P_cut`, including
`p_cut_results.csv`, `p_cut_results.jsonl`, safe-battery energy plots,
combined energy plots, and constellation P_cut tables.

`states.jsonl` stores one JSON object per simulation step, including the ECI Sun
direction without requiring consumers to reopen the BSP ephemeris. `tasks.jsonl`
stores task lifecycle events such as generation,
coverage waiting, assignment, completion, and failure. Both files remain valid
and readable if a long run stops early.

See `TASK_CONFIG.md` for the task-oriented config fields.

## Code structure

- `minimal_orbit.py` — CLI wrapper
- `configs/` — complete standalone experiment configs
- `satmulator/cli.py` — config parsing and run orchestration
- `satmulator/runtime.py` — mutable satellite/environment state
- `satmulator/models.py` — configs, tasks, assignments, snapshots
- `satmulator/orbit.py` — circular orbit model and timestep flow
- `satmulator/scheduler.py` — task assignment schedulers
- `satmulator/battery.py` — battery update logic
- `satmulator/runlog.py` — streaming JSON/JSONL experiment logs
- `satmulator/geometry.py` — geometry helpers

## Next work

1. queueing and task finish time
2. thermal throttling dynamics
3. workload read/write for controlled experiments
