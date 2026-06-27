import datetime as dt
import math
import random
import unittest
from types import SimpleNamespace

from satmulator.models import DemandPoint, Task
from satmulator.runtime import EnvironmentRuntime, SatelliteRuntime
from satmulator.workload import (
    choose_demand_point,
    demand_distribution,
    ground_position_km,
    nearest_satellite_id,
    resolve_pending_tasks,
    satellite_altitude_distance,
    weighted_choice,
)


class DemandPointCoordinateTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import skyfield  # noqa: F401
        except ImportError:
            self.skipTest("Skyfield is not installed")

    def test_demand_distribution_precomputes_cumulative_weights(self) -> None:
        points = (
            DemandPoint(1.0, 2.0, 2.0),
            DemandPoint(3.0, 4.0, 3.0),
            DemandPoint(5.0, 6.0, 5.0),
        )
        distribution = demand_distribution(points)

        self.assertEqual(distribution.points, points)
        self.assertEqual(distribution.cumulative_weights, (2.0, 5.0, 10.0))
        self.assertEqual(distribution.total_weight, 10.0)

    def test_demand_distribution_rejects_invalid_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            demand_distribution((DemandPoint(1.0, 2.0, float("nan")),))

    def test_demand_sampling_reuses_precomputed_distribution(self) -> None:
        points = (
            DemandPoint(1.0, 2.0, 1.0),
            DemandPoint(3.0, 4.0, 9.0),
        )
        distribution = demand_distribution(points)
        rng = random.Random(42)

        sampled = [choose_demand_point(rng, distribution) for _ in range(20)]

        self.assertGreater(sampled.count(points[1]), sampled.count(points[0]))

    def test_demand_sampling_preserves_seeded_results(self) -> None:
        points = (
            DemandPoint(1.0, 2.0, 1.0),
            DemandPoint(3.0, 4.0, 3.0),
            DemandPoint(5.0, 6.0, 6.0),
        )
        distribution = demand_distribution(points)
        old_rng = random.Random(42)
        new_rng = random.Random(42)

        old_samples = [
            weighted_choice(old_rng, points, tuple(point.weight for point in points))
            for _ in range(20)
        ]
        new_samples = [choose_demand_point(new_rng, distribution) for _ in range(20)]

        self.assertEqual(new_samples, old_samples)

    def test_ground_position_rotates_with_utc_time(self) -> None:
        point = DemandPoint(lat_deg=0.0, lon_deg=0.0, weight=1.0)
        start = dt.datetime(2026, 6, 7, tzinfo=dt.timezone.utc)
        later = start + dt.timedelta(hours=6)

        start_position = ground_position_km(point, start)
        later_position = ground_position_km(point, later)

        displacement_km = sum(
            (later_component - start_component) ** 2
            for start_component, later_component in zip(start_position, later_position)
        ) ** 0.5
        self.assertGreater(displacement_km, 8000.0)

    def test_nearest_satellite_uses_same_frame_as_ground_position(self) -> None:
        point = DemandPoint(lat_deg=25.033, lon_deg=121.5654, weight=1.0)
        time_utc = dt.datetime(2026, 6, 7, 12, tzinfo=dt.timezone.utc)
        ground = ground_position_km(point, time_utc)
        near = SatelliteRuntime(
            0,
            "near",
            0,
            0,
            1.0,
            pos_km=tuple(component * 1.1 for component in ground),
        )
        far = SatelliteRuntime(1, "far", 0, 1, 1.0, pos_km=(0.0, 0.0, 0.0))

        self.assertEqual(nearest_satellite_id([far, near], point, time_utc), 0)

    def test_nearest_satellite_prefers_visible_satellite(self) -> None:
        point = DemandPoint(lat_deg=25.033, lon_deg=121.5654, weight=1.0)
        time_utc = dt.datetime(2026, 6, 7, 12, tzinfo=dt.timezone.utc)
        ground = ground_position_km(point, time_utc)
        visible_position = tuple(component * 3.0 for component in ground)
        hidden_position = (0.0, 0.0, 0.0)
        visible = SatelliteRuntime(0, "visible", 0, 0, 1.0, pos_km=visible_position)
        hidden = SatelliteRuntime(1, "hidden", 0, 1, 1.0, pos_km=hidden_position)

        visible_altitude, _ = satellite_altitude_distance(visible, point, time_utc)
        hidden_altitude, _ = satellite_altitude_distance(hidden, point, time_utc)

        self.assertGreater(visible_altitude, 0.0)
        self.assertLess(hidden_altitude, 0.0)
        self.assertEqual(nearest_satellite_id([hidden, visible], point, time_utc), 0)

    def test_nearest_satellite_applies_minimum_elevation(self) -> None:
        point = DemandPoint(lat_deg=0.0, lon_deg=0.0, weight=1.0)
        time_utc = dt.datetime(2026, 6, 7, 12, tzinfo=dt.timezone.utc)
        ground = ground_position_km(point, time_utc)
        earth_radius = math.sqrt(sum(component**2 for component in ground))
        radial = tuple(component / earth_radius for component in ground)
        tangent = (-radial[1], radial[0], 0.0)
        tangent_norm = math.sqrt(sum(component**2 for component in tangent))
        tangent = tuple(component / tangent_norm for component in tangent)
        angle = math.radians(8.0)
        low_position = tuple(
            (radial[index] * math.cos(angle) + tangent[index] * math.sin(angle))
            * (earth_radius + 550.0)
            for index in range(3)
        )
        high_position = tuple(component * (earth_radius + 35786.0) for component in radial)
        low = SatelliteRuntime(0, "low", 0, 0, 1.0, pos_km=low_position)
        high = SatelliteRuntime(1, "high", 0, 1, 1.0, pos_km=high_position)

        self.assertEqual(nearest_satellite_id([low, high], point, time_utc, 0.0), 0)
        self.assertEqual(nearest_satellite_id([low, high], point, time_utc, 30.0), 1)
        self.assertIsNone(nearest_satellite_id([low], point, time_utc, 30.0))

    def test_pending_task_expires_without_coverage(self) -> None:
        point = DemandPoint(lat_deg=0.0, lon_deg=0.0, weight=1.0)
        time_utc = dt.datetime(2026, 6, 7, 12, tzinfo=dt.timezone.utc)
        hidden = SatelliteRuntime(0, "hidden", 0, 0, 1.0, pos_km=(0.0, 0.0, 0.0))
        task = Task(0, 0, None, 1.0, 0.0, 120.0, point.lat_deg, point.lon_deg)
        env = EnvironmentRuntime(
            satellites=[hidden],
            time_s=60,
            time_utc=time_utc,
            pending_tasks=[task],
        )
        config = SimpleNamespace(min_elevation_deg=30.0)

        ready, expired = resolve_pending_tasks(env, config)
        self.assertEqual((ready, expired), ([], []))
        self.assertEqual(env.pending_tasks, [task])

        env.time_s = 120
        ready, expired = resolve_pending_tasks(env, config)
        self.assertEqual(ready, [])
        self.assertEqual(expired, [task])
        self.assertEqual(env.pending_tasks, [])


if __name__ == "__main__":
    unittest.main()
