import datetime as dt
import math
import unittest
from types import SimpleNamespace

from satmulator.models import DemandPoint, Task
from satmulator.runtime import EnvironmentRuntime, SatelliteRuntime
from satmulator.workload import (
    ground_position_km,
    nearest_satellite_id,
    resolve_pending_tasks,
    satellite_altitude_distance,
)


class DemandPointCoordinateTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import skyfield  # noqa: F401
        except ImportError:
            self.skipTest("Skyfield is not installed")

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
        task = Task(0, 0, None, 1.0, 0.0, 0.0, 120.0, point.lat_deg, point.lon_deg)
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
