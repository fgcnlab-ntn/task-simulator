import unittest

from satmulator.isl import (
    ISLGraph,
    build_isl_graph,
    fully_connected_isl_graph,
    has_line_of_sight,
    range_limited_isl_graph,
    shortest_route,
)
from satmulator.models import ISLConfig, SatelliteView, Task
from satmulator.scheduler import NearestSunlitScheduler


def view(sat_id: int, *, sunlit: bool = False, x: float = 0.0) -> SatelliteView:
    return SatelliteView(
        sat_id=sat_id,
        x_km=x,
        y_km=0.0,
        z_km=0.0,
        sunlit=sunlit,
    )


def point_view(
    sat_id: int,
    *,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
) -> SatelliteView:
    return SatelliteView(
        sat_id=sat_id,
        x_km=x,
        y_km=y,
        z_km=z,
        sunlit=False,
    )


class ISLGraphTests(unittest.TestCase):
    def test_fully_connected_graph_has_every_other_satellite(self) -> None:
        graph = fully_connected_isl_graph([view(2), view(0), view(1)])

        self.assertEqual(graph.neighbors(0), (1, 2))
        self.assertEqual(graph.neighbors(1), (0, 2))
        self.assertEqual(graph.neighbors(2), (0, 1))

    def test_range_limited_graph_connects_only_nearby_satellites(self) -> None:
        graph = range_limited_isl_graph(
            [view(0, x=10000.0), view(1, x=10003.0), view(2, x=10010.0)],
            max_range_km=5.0,
        )

        self.assertEqual(graph.neighbors(0), (1,))
        self.assertEqual(graph.neighbors(1), (0,))
        self.assertEqual(graph.neighbors(2), ())

    def test_line_of_sight_blocks_links_through_earth(self) -> None:
        first = point_view(0, x=7000.0)
        second = point_view(1, x=-7000.0)

        self.assertFalse(has_line_of_sight(first, second))

    def test_line_of_sight_allows_links_above_earth_limb(self) -> None:
        first = point_view(0, x=10000.0, y=0.0)
        second = point_view(1, x=0.0, y=10000.0)

        self.assertTrue(has_line_of_sight(first, second))

    def test_range_limited_graph_rejects_earth_blocked_links(self) -> None:
        graph = range_limited_isl_graph(
            [point_view(0, x=7000.0), point_view(1, x=-7000.0)],
            max_range_km=20000.0,
        )

        self.assertEqual(graph.neighbors(0), ())
        self.assertEqual(graph.neighbors(1), ())

    def test_build_isl_graph_uses_configured_topology(self) -> None:
        graph = build_isl_graph(
            [view(0, x=10000.0), view(1, x=10003.0), view(2, x=10010.0)],
            ISLConfig(1.0, 1.0, 0.0, 0.0, topology="range-limited", max_range_km=5.0),
        )

        self.assertEqual(graph.neighbors(0), (1,))
        self.assertEqual(graph.neighbors(2), ())

    def test_range_limited_topology_requires_positive_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive max_range_km"):
            build_isl_graph(
                [view(0), view(1)],
                ISLConfig(1.0, 1.0, 0.0, 0.0, topology="range-limited"),
            )

    def test_shortest_route_returns_direct_route_when_available(self) -> None:
        graph = ISLGraph({0: (1, 2), 1: (0,), 2: (0,)})

        route = shortest_route(graph, 0, 2)

        self.assertIsNotNone(route)
        self.assertEqual(route.nodes, (0, 2))

    def test_shortest_route_returns_multi_hop_route(self) -> None:
        graph = ISLGraph({0: (1,), 1: (0, 2), 2: (1,)})

        route = shortest_route(graph, 0, 2)

        self.assertIsNotNone(route)
        self.assertEqual(route.nodes, (0, 1, 2))

    def test_shortest_route_returns_none_when_disconnected(self) -> None:
        graph = ISLGraph({0: (), 1: ()})

        self.assertIsNone(shortest_route(graph, 0, 1))

    def test_nearest_sunlit_scheduler_uses_route_finder(self) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=0,
            cpu_cycles=1.0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )
        views = [
            view(0, sunlit=False, x=0.0),
            view(1, sunlit=False, x=1.0),
            view(2, sunlit=True, x=2.0),
        ]
        graph = ISLGraph({0: (1,), 1: (0, 2), 2: (1,)})

        assignment = NearestSunlitScheduler().assign_task(
            task=task,
            satellite_views=views,
            isl_graph=graph,
        )

        self.assertEqual(assignment.route.nodes, (0, 1, 2))
        self.assertEqual(assignment.mode, "offload")

    def test_nearest_sunlit_scheduler_ignores_unreachable_sunlit_target(self) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=0,
            cpu_cycles=1.0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )
        views = [
            view(0, sunlit=False, x=0.0),
            view(1, sunlit=True, x=1.0),
        ]
        graph = ISLGraph({0: (), 1: ()})

        assignment = NearestSunlitScheduler().assign_task(
            task=task,
            satellite_views=views,
            isl_graph=graph,
        )

        self.assertEqual(assignment.route.nodes, (0,))
        self.assertEqual(assignment.mode, "local")


if __name__ == "__main__":
    unittest.main()
