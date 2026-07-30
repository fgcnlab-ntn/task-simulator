#!/usr/bin/env python3
"""Generate three-day configs measuring the middle day of every month."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "final-loading-ratio" / "r80"
DESTINATION = ROOT / "configs" / "all-months"
METHODS = (
    "greedy-energy",
    "local-only",
    "method7",
    "nearest-sunlit",
    "phoenix2",
)
MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def main() -> None:
    for month, month_name in MONTHS.items():
        start_date = f"2026-{month:02d}-14"
        group = f"m{month:02d}"

        for method in METHODS:
            with (SOURCE / f"{method}.json").open(encoding="utf-8") as source:
                config = json.load(source)

            description_prefix = config["run"]["description"].split(" (", 1)[0]
            config["run"]["description"] = (
                f"{description_prefix} ({month_name} 15 measurement, "
                f"14-16 warm-up/run window)"
            )
            config["time"]["start_utc"] = f"{start_date}T00:00:00Z"
            config["time"]["duration_s"] = 259200
            config["output"]["path"] = (
                f"output/all-months-3day/{group}/{method}"
            )
            config["logging"]["state_steps"] = "off"
            config["logging"]["summary_start_s"] = 86400
            config["logging"]["summary_duration_s"] = 86400

            destination = DESTINATION / group / f"{method}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as output:
                json.dump(config, output, indent=2)
                output.write("\n")


if __name__ == "__main__":
    main()
