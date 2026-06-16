import unittest

from satmulator.models import Assignment, Route


class AssignmentRouteTests(unittest.TestCase):
    def test_route_exposes_endpoint_compatibility(self) -> None:
        assignment = Assignment(task_id=7, route=(2, 5, 9), mode="offload")

        self.assertEqual(assignment.source_sat, 2)
        self.assertEqual(assignment.target_sat, 9)
        self.assertEqual(assignment.hop_count, 2)
        self.assertEqual(assignment.route, Route((2, 5, 9)))

    def test_legacy_endpoint_constructor_builds_route(self) -> None:
        assignment = Assignment(
            task_id=1,
            source_sat=3,
            target_sat=4,
            mode="offload",
        )

        self.assertEqual(assignment.route.nodes, (3, 4))

    def test_local_legacy_constructor_uses_single_node_route(self) -> None:
        assignment = Assignment(
            task_id=1,
            source_sat=3,
            target_sat=3,
            mode="local",
        )

        self.assertEqual(assignment.route.nodes, (3,))
        self.assertEqual(assignment.hop_count, 0)

    def test_rejects_empty_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "route must contain"):
            Assignment(task_id=1, route=(), mode="offload")

    def test_rejects_invalid_route_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "route nodes"):
            Assignment(task_id=1, route=(-1,), mode="offload")


if __name__ == "__main__":
    unittest.main()
