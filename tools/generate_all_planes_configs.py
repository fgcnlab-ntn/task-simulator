#!/usr/bin/env python3
"""Generate the unrun P*S=1584 experiment configurations."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "final-planes" / "p48"
DESTINATION = ROOT / "configs" / "all-planes"
METHODS = (
    "greedy-energy",
    "local-only",
    "method7",
    "nearest-sunlit",
    "phoenix2",
)
ALL_PLANES = (9, 11, 12, 16, 18, 22, 24, 33, 36, 44, 48, 66, 72, 88, 99, 132, 144, 176)
ALREADY_RUN = (48, 66, 72, 88, 99)


def description_for(config: dict[str, object], planes: int) -> str:
    run = config["run"]
    assert isinstance(run, dict)
    description = run["description"]
    assert isinstance(description, str)
    return description.replace("(48 orbital planes)", f"({planes} orbital planes)")


def main() -> None:
    for planes in ALL_PLANES:
        if planes in ALREADY_RUN:
            continue
        for method in METHODS:
            with (SOURCE / f"{method}.json").open(encoding="utf-8") as source:
                config = json.load(source)

            config["orbit"]["planes"] = planes
            config["run"]["description"] = description_for(config, planes)
            config["output"]["path"] = f"output/all-planes/p{planes}/{method}"
            config["logging"]["state_steps"] = "off"

            destination = DESTINATION / f"p{planes}" / f"{method}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as output:
                json.dump(config, output, indent=2)
                output.write("\n")


if __name__ == "__main__":
    main()
