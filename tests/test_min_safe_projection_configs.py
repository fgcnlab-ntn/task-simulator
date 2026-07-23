from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import pytest

from satmulator.cli import DEFAULT_CONFIG, load_standalone_json_config, run


CONFIG_DIR = Path("configs/regression/min-safe-projection")
BASE_CONFIG_DIR = Path("configs/final-loading-ratio/r80")
ALLOWED_REGRESSION_DIFFS = {
    ("run", "description"),
    ("time", "duration_s"),
    ("task", "tasks_per_step_choices"),
    ("task", "tasks_per_step_weights"),
    ("output", "path"),
}


def _args_for_config(config_path: Path, out: Path) -> argparse.Namespace:
    values = DEFAULT_CONFIG.copy()
    values.update(load_standalone_json_config(config_path))
    values["config"] = config_path
    values["plot_run"] = None
    values["out"] = out
    values["tle_file"] = None if values["tle_file"] is None else Path(values["tle_file"])
    values["task_demand_points_file"] = (
        None
        if values["task_demand_points_file"] is None
        else Path(values["task_demand_points_file"])
    )
    return argparse.Namespace(**values)


def _changed_paths(left: object, right: object, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            paths |= _changed_paths(left.get(key), right.get(key), (*prefix, str(key)))
        return paths
    if left == right:
        return set()
    return {prefix}


@pytest.mark.parametrize(
    "config_path",
    sorted(CONFIG_DIR.glob("*.json")),
    ids=lambda path: path.stem,
)
def test_min_safe_projection_configs_only_change_load_and_duration(config_path: Path) -> None:
    """Regression configs must stay comparable with the paper r80 configs."""

    base_path = BASE_CONFIG_DIR / config_path.name
    base = json.loads(base_path.read_text())
    config = json.loads(config_path.read_text())

    changed = _changed_paths(base, config)
    assert changed <= ALLOWED_REGRESSION_DIFFS
    assert config["task"]["generation_mode"] == base["task"]["generation_mode"] == "demand-points"
    assert config["time"]["duration_s"] < base["time"]["duration_s"]
    assert config["task"]["tasks_per_step_choices"][0] > base["task"]["tasks_per_step_choices"][0]


@pytest.mark.skipif(
    os.environ.get("SATMULATOR_RUN_SLOW_REGRESSION") != "1",
    reason="set SATMULATOR_RUN_SLOW_REGRESSION=1 to run full high-load config simulations",
)
@pytest.mark.parametrize(
    "config_path",
    sorted(CONFIG_DIR.glob("*.json")),
    ids=lambda path: path.stem,
)
def test_min_safe_projection_configs_drop_unsafe_work(config_path: Path) -> None:
    """Slow integration: high-load runs must preserve min_safe by dropping/deferring work."""

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / config_path.stem
        args = _args_for_config(config_path, out)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            assert run(args) == 0

        summary = json.loads((out / "summary.json").read_text())
        tasks = summary["tasks"]
        assert tasks["generated"] > 0
        assert tasks["completed"] < tasks["generated"]

        capacity_j = float(args.battery_capacity_j)
        min_safe_j = capacity_j * float(args.battery_min_safe_pct) / 100.0
        min_battery_j = capacity_j
        max_below_min_safe = 0

        with (out / "states.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                max_below_min_safe = max(
                    max_below_min_safe,
                    int(record["battery_violation_summary"]["below_min_safe"]),
                )
                for satellite in record["satellites"]:
                    min_battery_j = min(min_battery_j, float(satellite["battery_j"]))

        assert max_below_min_safe == 0
        assert min_battery_j >= min_safe_j
