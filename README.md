# Satmulator

Minimal NTN satellite-state simulator for energy-aware task execution and
offloading experiments.

The current model supports:

- circular Walker-style orbit model and TLE/SGP4 orbit model
- sunlight/eclipse classification
- per-satellite battery state
- deterministic and demand-point task generation
- local and nearest-sunlit schedulers
- four-neighbor grid or fully-connected ISL routing with per-hop accounting
- target load limits in CPU cycles per slot
- structured JSON/JSONL logs and SVG outputs for quick inspection

It does not yet model queueing, link contention, or thermal throttling dynamics.

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

For offline TLE runs, copy `configs/template.json`, set the `orbit` section to `orbit_model: "tle"`, provide `tle_file` and `sun_position_file`, and use `isl.topology: "fully-connected"`.

## Run

Use JSON config files for reproducible runs:

```bash
python3 minimal_orbit.py --config configs/template.json
```

Nearest-sunlit offloading:

```bash
python3 minimal_orbit.py --config configs/nearest_sunlit.json
```

Task-oriented demand-point workload:

```bash
python3 minimal_orbit.py --config configs/demand_points.json
```

Population-weighted demand inputs use WorldPop 2025 R2025A constrained 1 km
population-count products for both Taiwan and global experiments. See
`data/worldpop/README.md` for the exact sources and checksums.

Regenerate plots from an existing run without rerunning the simulation:

```bash
python3 minimal_orbit.py --plot-run output/minimal_orbit
```

CLI only keeps run-control and debug overrides:

```bash
python3 minimal_orbit.py \
  --config configs/template.json \
  --duration-s 600 \
  --no-task \
  --out output/debug
```

TLE runs should be described by a complete standalone TLE config file; the CLI no longer carries orbit or topology fields.

Config behavior:

```text
no --config: built-in defaults
--config: complete standalone JSON config < run-control/debug CLI overrides
```

The effective merged config is written to:

```text
<output>/run.json
```

## Template model

`configs/template.json` is a complete, standalone Starlink-like template scenario:

- 1584 satellites, 72 planes, 550 km altitude, 53.05° inclination
- start time `2026-05-22T12:00:00Z`, duration 1800 s, step 30 s
- battery capacity 100000 J, initial 80%, safe minimum 20%
- default legacy mode: one task per satellite every 300 s
- demand-point mode: task locations and workload sizes sampled from configured distributions
- default legacy task size 1e9 CPU cycles, 1e7 input bits, 1e6 output bits
- default four-neighbor grid ISL with a 5000 km link range and Earth-obstruction filtering
- default ISL cost: 10 Mbps forward/return, 1e-7 J/bit TX, 5e-8 J/bit RX
- default scheduler target load limit: 4e9 CPU cycles per slot
- scheduler: `local`

The grid builds a fixed candidate layout once: two in-plane links and two
cross-plane links per satellite. The plane seam is shifted by the configured
Walker phase. At each simulation step, only range and Earth line-of-sight are
reevaluated to determine which candidate links are active. Diagonal satellites
therefore require at least two hops. Fully-connected routing remains available by setting `isl.topology` to
`fully-connected` in a standalone config.

TLE input does not carry reliable plane/slot assignments, so TLE configs must use
`isl.topology: "fully-connected"` until explicit constellation layout metadata is
provided.

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
- `configs/` — complete standalone JSON configs; `template.json` is the copyable baseline
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

1. queueing and task finish time
2. thermal throttling dynamics
3. TLE constellation layout metadata
4. workload read/write for controlled experiments
