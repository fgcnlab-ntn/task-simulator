import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_grouped_task_outcomes import (
    collect_rows,
    collect_rows_from_dirs,
    format_group_labels,
    load_fail_rate,
    validate_groups,
    validate_methods,
)


class GroupedTaskOutcomesTests(unittest.TestCase):
    def write_summary(
        self,
        path: Path,
        *,
        generated: int = 10,
        completed: int = 7,
        pending: int = 1,
        failed: int = 2,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "tasks": {
                        "generated": generated,
                        "completed": completed,
                        "pending": pending,
                        "failed": failed,
                    }
                }
            )
        )

    def test_loads_failure_count_and_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            self.write_summary(summary_path)

            self.assertEqual(load_fail_rate(summary_path), (2, 20.0))

    def test_rejects_inconsistent_task_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            self.write_summary(summary_path, completed=8)

            with self.assertRaisesRegex(ValueError, "does not equal generated"):
                load_fail_rate(summary_path)

    def test_collects_groups_and_methods_in_requested_order(self) -> None:
        groups = ["g1", "g2", "g3", "g4"]
        methods = [
            "local-only",
            "nearest-sunlit",
            "greedy-energy",
            "phoenix2",
            "method7",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            for group in groups:
                for method in methods:
                    self.write_summary(base_dir / group / method / "summary.json")

            rows = collect_rows(
                base_dir,
                groups=groups,
                group_labels=["one", "two", "three", "four"],
                methods=methods,
            )

            self.assertEqual(
                [(row["group"], row["method"]) for row in rows],
                [
                    (group, method)
                    for group in groups
                    for method in methods
                ],
            )

    def test_requires_four_to_eighteen_unique_groups(self) -> None:
        for groups in (
            ["g1", "g2", "g3"],
            ["g1"] * 4,
            [f"g{i}" for i in range(19)],
        ):
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    validate_groups(groups)

        validate_groups([f"g{i}" for i in range(18)])

    def test_requires_every_summary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "missing summary file"):
                collect_rows(
                    Path(tmp),
                    groups=["g1", "g2", "g3", "g4"],
                    group_labels=["one", "two", "three", "four"],
                    methods=[
                        "local-only",
                        "nearest-sunlit",
                        "greedy-energy",
                        "phoenix2",
                        "method7",
                    ],
                )

    def test_collects_group_directories_with_different_parents(self) -> None:
        methods = [
            "local-only",
            "nearest-sunlit",
            "greedy-energy",
            "phoenix2",
            "method7",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_dirs = [
                root / "planes" / "p48",
                root / "planes" / "p66",
                root / "loading" / "r80",
                root / "planes" / "p88",
                root / "planes" / "p99",
            ]
            for group_dir in group_dirs:
                for method in methods:
                    self.write_summary(group_dir / method / "summary.json")

            rows = collect_rows_from_dirs(
                group_dirs,
                group_keys=["p48", "p66", "r80", "p88", "p99"],
                group_labels=["48", "66", "72", "88", "99"],
                methods=methods,
            )

            self.assertEqual(len(rows), 25)
            self.assertEqual(
                [row["group_label"] for row in rows[::5]],
                ["48", "66", "72", "88", "99"],
            )

    def test_requires_five_unique_known_methods(self) -> None:
        invalid_methods = (
            ["local-only"] * 5,
            ["local-only", "nearest-sunlit", "greedy-energy", "phoenix2"],
            [
                "local-only",
                "nearest-sunlit",
                "greedy-energy",
                "phoenix2",
                "unknown",
            ],
        )
        for methods in invalid_methods:
            with self.subTest(methods=methods):
                with self.assertRaises(ValueError):
                    validate_methods(methods)

    def test_formats_four_season_labels_on_two_lines(self) -> None:
        labels, is_four_season = format_group_labels(
            [
                "spring-equinox",
                "Summer solstice",
                "Autumn\nequinox",
                "winter-solstice",
            ]
        )

        self.assertTrue(is_four_season)
        self.assertEqual(
            labels,
            [
                "Spring\nequinox",
                "Summer\nsolstice",
                "Autumn\nequinox",
                "Winter\nsolstice",
            ],
        )

    def test_expands_short_season_labels(self) -> None:
        labels, is_four_season = format_group_labels(
            ["Spring", "Summer", "Autumn", "Winter"]
        )

        self.assertTrue(is_four_season)
        self.assertEqual(
            labels,
            [
                "Spring\nequinox",
                "Summer\nsolstice",
                "Autumn\nequinox",
                "Winter\nsolstice",
            ],
        )

    def test_reserves_a_second_line_for_five_group_labels(self) -> None:
        labels = ["48×33", "66×24", "72×22", "88×18", "99×16"]

        formatted, is_four_season = format_group_labels(labels)

        self.assertFalse(is_four_season)
        self.assertEqual(formatted, [f"{label}\n " for label in labels])


if __name__ == "__main__":
    unittest.main()
