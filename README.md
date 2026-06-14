# Satmulator

Minimal NTN satellite-state simulator for energy-aware task execution and
offloading experiments.

The current model supports:

- circular Walker-style orbit model and TLE/SGP4 orbit model
- sunlight/eclipse classification
- per-satellite battery state
- deterministic and demand-point task generation
- local and nearest-sunlit schedulers
- one-hop ISL time/energy accounting for offloaded tasks
- structured JSON/JSONL logs and SVG outputs for quick inspection

It does not yet model routing, hop count, queueing, link contention, or target
compute capacity.

## Install

TLE runs and demand-point coordinate conversion require Skyfield. Circular
runs without demand-point workloads use only the Python standard library:

```bash
python3 -m pip install -r requirements.txt
```

`de440s.bsp` is not tracked by git. Circular runs do not need it. TLE runs use it
for Sun-position calculation; Skyfield can download it on first use:

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

For offline TLE runs, provide a local path:

```bash
python3 minimal_orbit.py \
  --config configs/default.json \
  --orbit-model tle \
  --tle-file tle/stations.tle \
  --sun-position-file /path/to/de440s.bsp
```

## Run

Use JSON config files for reproducible runs:

```bash
python3 minimal_orbit.py --config configs/default.json
```

Nearest-sunlit offloading:

```bash
python3 minimal_orbit.py --config configs/nearest_sunlit.json
```

Task-oriented demand-point workload:

```bash
python3 minimal_orbit.py --config configs/demand_points.json
```

Regenerate plots from an existing run without rerunning the simulation:

```bash
python3 minimal_orbit.py --plot-run output/minimal_orbit
```

CLI flags override config values:

```bash
python3 minimal_orbit.py \
  --config configs/default.json \
  --scheduler nearest-sunlit \
  --duration-s 600 \
  --out output/debug
```

TLE run:

```bash
python3 minimal_orbit.py \
  --config configs/default.json \
  --orbit-model tle \
  --tle-file tle/stations.tle \
  --duration-s 1800 \
  --step-s 60 \
  --out output/tle_stations
```

Config precedence:

```text
built-in defaults < JSON config < CLI overrides
```

The effective merged config is written to:

```text
<output>/run.json
```

## Default model

`configs/default.json` defines the default smoke scenario:

- 66 satellites, 6 planes, 550 km altitude, 53° inclination
- start time `2026-05-22T12:00:00Z`, duration 1800 s, step 30 s
- battery capacity 100000 J, initial 80%, safe minimum 20%
- default legacy mode: one task per satellite every 300 s
- demand-point mode: task locations and workload sizes sampled from configured distributions
- default legacy task size 1e9 CPU cycles, 1e7 input bits, 1e6 output bits
- one-hop ISL: 10 Mbps forward/return, 1e-7 J/bit TX, 5e-8 J/bit RX
- scheduler: `local`

## Outputs

Each run writes:

- `run.json` — structured run status, effective config, and satellite catalog
- `states.jsonl` — one append-safe satellite-state record per simulation step
- `tasks.jsonl` — append-safe task lifecycle events
- `summary.json` — final structured result summary
- `*.svg` — quick visual checks for orbit, battery, sunlight, and task results

JSON/JSONL files are the structured experiment log. SVG files are quick
inspection outputs.

`states.jsonl` stores one JSON object per simulation step, including the ECI Sun
direction needed to reproduce TLE snapshot plots without reopening the BSP
ephemeris. `tasks.jsonl` stores task lifecycle events such as generation,
coverage waiting, assignment, completion, and failure. Both files remain valid
and readable if a long run stops early.

See `TASK_CONFIG.md` for the task-oriented config fields.

## Code structure

- `minimal_orbit.py` — CLI wrapper
- `configs/` — JSON configs
- `satmulator/cli.py` — config parsing and run orchestration
- `satmulator/runtime.py` — mutable satellite/environment state
- `satmulator/models.py` — configs, tasks, assignments, snapshots
- `satmulator/orbit.py` — orbit models and timestep flow
- `satmulator/scheduler.py` — task assignment schedulers
- `satmulator/battery.py` — battery update logic
- `satmulator/runlog.py` — streaming JSON/JSONL experiment logs
- `satmulator/plotting.py` — rebuilds SVG plots from experiment logs
- `satmulator/output.py` — SVG writers
- `satmulator/geometry.py` — geometry helpers

## Next work

1. target-side compute capacity / load accounting
2. queueing and task finish time
3. hop-count and routing model
4. workload read/write for controlled experiments
5. JSONL logging for long runs
