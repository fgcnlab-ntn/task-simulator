import datetime as dt
import unittest

from satmulator.models import DemandPoint
from satmulator.runtime import SatelliteRuntime
from satmulator.workload import (
    ground_position_km,
    nearest_satellite_id,
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
        near = SatelliteRuntime(0, "near", 0, 0, 1.0, pos_km=ground)
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


if __name__ == "__main__":
    unittest.main()
