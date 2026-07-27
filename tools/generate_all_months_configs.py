#!/usr/bin/env python3
"""Generate configs for the seven months missing from the seasonal runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "final-seasons" / "spring-equinox"
DESTINATION = ROOT / "configs" / "all-months"
METHODS = (
    "greedy-energy",
    "local-only",
    "method7",
    "nearest-sunlit",
    "phoenix2",
)
MISSING_MONTHS = {
    1: "January",
    2: "February",
    4: "April",
    7: "July",
    8: "August",
    10: "October",
    11: "November",
}


def main() -> None:
    for month, month_name in MISSING_MONTHS.items():
        date = f"2026-{month:02d}-20"
        group = f"m{month:02d}"

        for method in METHODS:
            with (SOURCE / f"{method}.json").open(encoding="utf-8") as source:
                config = json.load(source)

            description_prefix = config["run"]["description"].split(" (", 1)[0]
            config["run"]["description"] = (
                f"{description_prefix} ({month_name} sample, {date})"
            )
            config["time"]["start_utc"] = f"{date}T12:00:00Z"
            config["output"]["path"] = f"output/all-months/{group}/{method}"
            config["logging"]["state_steps"] = "off"

            destination = DESTINATION / group / f"{method}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as output:
                json.dump(config, output, indent=2)
                output.write("\n")


if __name__ == "__main__":
    main()
