import unittest

from satmulator.isl import ISLGraph, fully_connected_isl_graph, shortest_route
from satmulator.models import SatelliteView, Task
from satmulator.scheduler import NearestSunlitScheduler


def view(sat_id: int, *, sunlit: bool = False, x: float = 0.0) -> SatelliteView:
    return SatelliteView(
        sat_id=sat_id,
        x_km=x,
        y_km=0.0,
        z_km=0.0,
        sunlit=sunlit,
    )


class ISLGraphTests(unittest.TestCase):
    def test_fully_connected_graph_has_every_other_satellite(self) -> None:
        graph = fully_connected_isl_graph([view(2), view(0), view(1)])

        self.assertEqual(graph.neighbors(0), (1, 2))
        self.assertEqual(graph.neighbors(1), (0, 2))
        self.assertEqual(graph.neighbors(2), (0, 1))

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
