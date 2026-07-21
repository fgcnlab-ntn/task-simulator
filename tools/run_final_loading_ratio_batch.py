#!/usr/bin/env python3
"""Run the final-loading-ratio configs with a bounded worker pool.

This launcher keeps the simulator itself unchanged and only parallelizes the
outer orchestration layer. Each config writes to its own output directory, so
results do not collide while runs are processed eight at a time.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import threading
import time
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG_DIR = Path("configs/final-loading-ratio")
DEFAULT_OUTPUT_DIR = Path("output/final-loading-ratio")
DEFAULT_WORKERS = 9
HEARTBEAT_INTERVAL_S = 30
REPO_ROOT = Path(__file__).resolve().parents[1]
PRINT_LOCK = threading.Lock()


class ProgressTracker:
    def __init__(self, total: int) -> None:
        self.total = total
        self.started = 0
        self.running = 0
        self.done = 0
        self.failed = 0
        self._lock = threading.Lock()

    def job_started(self) -> None:
        with self._lock:
            self.started += 1
            self.running += 1

    def job_finished(self, success: bool) -> None:
        with self._lock:
            self.running -= 1
            if success:
                self.done += 1
            else:
                self.failed += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "total": self.total,
                "started": self.started,
                "running": self.running,
                "queued": self.total - self.started,
                "done": self.done,
                "failed": self.failed,
            }


def discover_configs(config_dir: Path) -> list[Path]:
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory does not exist: {config_dir}")
    configs = sorted(config_dir.rglob("*.json"))
    if not configs:
        raise FileNotFoundError(f"no config files found under: {config_dir}")
    return configs


def output_path_for(config_dir: Path, output_dir: Path, config_path: Path) -> Path:
    relative = config_path.relative_to(config_dir)
    return output_dir / relative.with_suffix("")


def should_skip(output_path: Path) -> bool:
    return (output_path / "summary.json").exists()


def archive_existing_output(output_path: Path) -> Path | None:
    if not output_path.exists():
        return None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archive_path = output_path.with_name(f"{output_path.name}.archived-{timestamp}")
    suffix = 1
    while archive_path.exists():
        archive_path = output_path.with_name(
            f"{output_path.name}.archived-{timestamp}-{suffix}"
        )
        suffix += 1
    shutil.move(str(output_path), str(archive_path))
    return archive_path


def tail_lines(path: Path, limit: int = 20) -> list[str]:
    if not path.exists():
        return [f"missing log file: {path}"]
    lines = path.read_text().splitlines()
    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def run_one(config_path: Path, output_path: Path) -> tuple[Path, int]:
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "run.log"
    command = [
        sys.executable,
        str(REPO_ROOT / "minimal_orbit.py"),
        "--config",
        str(config_path),
        "--out",
        str(output_path),
    ]
    with log_path.open("w") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return config_path, process.returncode


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def is_tmux_mode() -> bool:
    return bool(os.environ.get("TMUX"))


def print_line(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def heartbeat_loop(
    tracker: ProgressTracker,
    started_at: float,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(HEARTBEAT_INTERVAL_S):
        snapshot = tracker.snapshot()
        elapsed = format_elapsed(time.monotonic() - started_at)
        print_line(
            "heartbeat: "
            f"running={snapshot['running']} queued={snapshot['queued']} "
            f"done={snapshot['done']} fail={snapshot['failed']} "
            f"total={snapshot['total']} elapsed={elapsed}"
        )


def run_one_with_progress(
    config_path: Path,
    output_path: Path,
    tracker: ProgressTracker,
) -> tuple[Path, int]:
    tracker.job_started()
    try:
        result = run_one(config_path, output_path)
    except Exception:
        tracker.job_finished(False)
        raise
    tracker.job_finished(result[1] == 0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run final-loading-ratio configs with a bounded worker pool."
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--ratios",
        nargs="+",
        help="only run these loading-ratio folders, e.g. r70 r80 r90",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="only run these config stems, e.g. phoenix2 method6",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun configs even when summary.json already exists",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="move an existing output directory aside before rerunning it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected configs without running them",
    )
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.archive_existing and not args.force:
        raise ValueError("--archive-existing requires --force")

    config_dir = (REPO_ROOT / args.config_dir).resolve()
    output_dir = (REPO_ROOT / args.out).resolve()
    configs = discover_configs(config_dir)
    if args.ratios:
        allowed_ratios = set(args.ratios)
        configs = [
            config_path
            for config_path in configs
            if config_path.parent.name in allowed_ratios
        ]
    if args.methods:
        allowed_methods = set(args.methods)
        configs = [
            config_path
            for config_path in configs
            if config_path.stem in allowed_methods
        ]
    if not configs:
        raise FileNotFoundError("no configs match the requested filters")

    jobs: list[tuple[Path, Path]] = []
    for config_path in configs:
        output_path = output_path_for(config_dir, output_dir, config_path)
        if not args.force and should_skip(output_path):
            print(f"skip {config_path} -> {output_path} (summary.json exists)")
            continue
        jobs.append((config_path, output_path))

    if not jobs:
        print_line("nothing to do")
        return 0

    if args.dry_run:
        print_line(f"dry-run: {len(jobs)} configs selected")
        for config_path, output_path in jobs:
            print_line(f"{config_path} -> {output_path}")
        return 0

    if args.archive_existing:
        for _config_path, output_path in jobs:
            archive_path = archive_existing_output(output_path)
            if archive_path is not None:
                print_line(f"archived {output_path} -> {archive_path}")

    total = len(jobs)
    started_at = time.monotonic()
    tracker = ProgressTracker(total)
    print_line(
        f"running {total} configs from {config_dir} "
        f"with {args.workers} workers -> {output_dir}"
    )

    failures: list[tuple[Path, int]] = []
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if is_tmux_mode():
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            args=(tracker, started_at, heartbeat_stop),
            daemon=True,
        )
        heartbeat_thread.start()
        snapshot = tracker.snapshot()
        print_line(
            "heartbeat: "
            f"running={snapshot['running']} queued={snapshot['queued']} "
            f"done={snapshot['done']} fail={snapshot['failed']} "
            f"total={snapshot['total']} elapsed=0s"
        )

    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                run_one_with_progress,
                config_path,
                output_path,
                tracker,
            ): (config_path, output_path)
            for config_path, output_path in jobs
        }
        for future in cf.as_completed(future_map):
            config_path, output_path = future_map[future]
            try:
                finished_config, returncode = future.result()
            except Exception as exc:  # pragma: no cover - surfaced to the console
                failures.append((config_path, 1))
                elapsed = format_elapsed(time.monotonic() - started_at)
                snapshot = tracker.snapshot()
                print_line(
                    f"[{snapshot['done'] + snapshot['failed']}/{total}] fail "
                    f"{config_path} -> {output_path} ({elapsed}): {exc}"
                )
                continue
            elapsed = format_elapsed(time.monotonic() - started_at)
            snapshot = tracker.snapshot()
            if returncode == 0:
                print_line(
                    f"[{snapshot['done'] + snapshot['failed']}/{total}] done "
                    f"{finished_config} -> {output_path} ({elapsed})"
                )
            else:
                failures.append((finished_config, returncode))
                print_line(
                    f"[{snapshot['done'] + snapshot['failed']}/{total}] fail "
                    f"{finished_config} -> {output_path} ({elapsed}): exit code {returncode}"
                )
                log_path = output_path / "run.log"
                log_tail = tail_lines(log_path)
                print_line(f"last {len(log_tail)} lines of {log_path}:")
                for line in log_tail:
                    print_line(f"  {line}")

    heartbeat_stop.set()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=1.0)

    elapsed = format_elapsed(time.monotonic() - started_at)
    snapshot = tracker.snapshot()
    print_line(
        f"summary: total={total}, done={snapshot['done']}, failed={snapshot['failed']}, elapsed={elapsed}"
    )

    if failures:
        return 1

    print_line("all configs completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
