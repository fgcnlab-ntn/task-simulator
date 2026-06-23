from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from .models import SatelliteState, SnapshotContext


SCHEMA_VERSION = 1
JsonObject = dict[str, object]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_json_line(stream: TextIO, value: object) -> None:
    stream.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
    stream.flush()


def _validate_record(value: object, *, source: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    version = value.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{source} has unsupported schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return value


def load_run(output_dir: Path) -> JsonObject:
    path = output_dir / "run.json"
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    return _validate_record(value, source=str(path))


def _iter_jsonl(path: Path) -> Iterator[JsonObject]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            source = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"{source} must contain a JSON object")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source}: {exc.msg}") from exc
            yield _validate_record(value, source=source)


def iter_state_steps(output_dir: Path) -> Iterator[JsonObject]:
    return _iter_jsonl(output_dir / "states.jsonl")


def iter_task_events(output_dir: Path) -> Iterator[JsonObject]:
    return _iter_jsonl(output_dir / "tasks.jsonl")


def satellite_catalog(states: list[SatelliteState]) -> list[dict[str, object]]:
    return [
        {
            "id": state.sat_id,
            "name": state.name,
            "plane": state.plane,
            "slot": state.slot,
        }
        for state in states
    ]


def eclipse_energy_summary(states: list[SatelliteState]) -> dict[str, float]:
    idle_j = 0.0
    task_j = 0.0
    for state in states:
        if state.sunlit:
            continue
        idle_j += state.consumed_j
        task_j += state.task_energy_j
    return {
        "idle_j": idle_j,
        "task_j": task_j,
        "total_j": idle_j + task_j,
    }


def state_record(
    start: dt.datetime,
    states: list[SatelliteState],
    context: SnapshotContext | None = None,
) -> dict[str, object]:
    time_s = states[0].time_s
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "time_s": time_s,
        "time_iso": (start + dt.timedelta(seconds=time_s)).isoformat(),
        "energy_summary": {
            "eclipse": eclipse_energy_summary(states),
        },
        "satellites": [
            {
                "id": state.sat_id,
                "position_km": [state.x_km, state.y_km, state.z_km],
                "velocity_km_s": [state.vx_km_s, state.vy_km_s, state.vz_km_s],
                "geodetic": (
                    None
                    if state.lat_deg is None
                    else {
                        "lat_deg": state.lat_deg,
                        "lon_deg": state.lon_deg,
                        "elevation_km": state.elevation_km,
                    }
                ),
                "sunlit": state.sunlit,
                "battery_j": state.battery_j,
                "energy_delta_j": {
                    "harvested": state.harvested_j,
                    "consumed": state.consumed_j,
                    "tasks": state.task_energy_j,
                },
                "task_counts": {
                    "generated": state.generated_tasks,
                    "completed": state.completed_tasks,
                    "failed": state.failed_tasks,
                    "deferred": state.deferred_tasks,
                },
            }
            for state in states
        ],
    }
    if context is not None:
        record["snapshot_context"] = {
            "projection_label": context.projection_label,
            "sun_eci_unit": context.sun_eci_unit,
            "sun_xy_unit": context.sun_xy_unit,
        }
    return record


class RunLog:
    """Streaming structured log for one simulation run.

    JSONL streams are flushed after every record so a failed long run still
    leaves valid, parseable observations. SVG inspection outputs are handled
    elsewhere.
    """

    def __init__(self, output_dir: Path, start: dt.datetime, config: dict[str, object]):
        self.output_dir = output_dir
        self.start = start
        self.run_path = output_dir / "run.json"
        self.summary_path = output_dir / "summary.json"
        self._states = (output_dir / "states.jsonl").open("w")
        self._tasks = (output_dir / "tasks.jsonl").open("w")
        self._generated_ids: set[int] = set()
        self._terminal_ids: set[int] = set()
        self._completed = 0
        self._failed = 0
        self._deferred = 0
        self._steps = 0
        self._eclipse_idle_j = 0.0
        self._eclipse_task_j = 0.0
        self._final_states: list[SatelliteState] | None = None
        self._manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at": utc_now_iso(),
            "config": config,
        }
        write_json(self.run_path, self._manifest)

    def write_step(
        self,
        states: list[SatelliteState],
        context: SnapshotContext | None = None,
    ) -> None:
        if "satellites" not in self._manifest:
            self._manifest["satellites"] = satellite_catalog(states)
            write_json(self.run_path, self._manifest)
        append_json_line(self._states, state_record(self.start, states, context))
        eclipse_energy = eclipse_energy_summary(states)
        self._eclipse_idle_j += eclipse_energy["idle_j"]
        self._eclipse_task_j += eclipse_energy["task_j"]
        self._steps += 1
        self._final_states = states

    def write_task_event(self, event: dict[str, object]) -> None:
        record = {"schema_version": SCHEMA_VERSION, **event}
        time_s = event.get("time_s")
        if isinstance(time_s, int):
            record["time_iso"] = (self.start + dt.timedelta(seconds=time_s)).isoformat()
        task_id = event.get("task_id")
        event_type = event.get("type")
        if isinstance(task_id, int):
            if event_type == "task_generated":
                self._generated_ids.add(task_id)
            elif event_type == "task_completed":
                self._terminal_ids.add(task_id)
                self._completed += 1
            elif event_type == "task_failed":
                self._terminal_ids.add(task_id)
                self._failed += 1
            elif event_type == "task_deferred":
                self._deferred += 1
        append_json_line(self._tasks, record)

    def complete(self, all_steps: list[list[SatelliteState]] | None = None) -> None:
        if all_steps is not None:
            final_states = all_steps[-1]
            steps = len(all_steps)
            eclipse_idle_j = 0.0
            eclipse_task_j = 0.0
            for states in all_steps:
                eclipse_energy = eclipse_energy_summary(states)
                eclipse_idle_j += eclipse_energy["idle_j"]
                eclipse_task_j += eclipse_energy["task_j"]
        elif self._final_states is not None:
            final_states = self._final_states
            steps = self._steps
            eclipse_idle_j = self._eclipse_idle_j
            eclipse_task_j = self._eclipse_task_j
        else:
            raise ValueError("cannot complete a run with no state records")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "steps": steps,
            "final_time_s": final_states[0].time_s,
            "satellites": len(final_states),
            "tasks": {
                "generated": len(self._generated_ids),
                "completed": self._completed,
                "deferred": self._deferred,
                "failed": self._failed,
                "pending": len(self._generated_ids - self._terminal_ids),
            },
            "final_battery_j": {
                "minimum": min(state.battery_j for state in final_states),
                "average": sum(state.battery_j for state in final_states)
                / len(final_states),
            },
            "energy": {
                "eclipse": {
                    "idle_j": eclipse_idle_j,
                    "task_j": eclipse_task_j,
                    "total_j": eclipse_idle_j + eclipse_task_j,
                },
            },
        }
        write_json(self.summary_path, summary)
        self._manifest.update(
            {
                "status": "completed",
                "finished_at": utc_now_iso(),
                "summary_file": self.summary_path.name,
            }
        )
        write_json(self.run_path, self._manifest)
        self.close()

    def fail(self, exc: BaseException) -> None:
        self._manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now_iso(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(self.run_path, self._manifest)
        self.close()

    def close(self) -> None:
        if not self._states.closed:
            self._states.close()
        if not self._tasks.closed:
            self._tasks.close()
