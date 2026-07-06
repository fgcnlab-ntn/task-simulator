import datetime as dt
import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from satmulator.models import DemandPoint, Task
from satmulator.runtime import EnvironmentRuntime, SatelliteRuntime
from satmulator.workload import (
    choose_demand_point,
    demand_distribution,
    elevation_candidate_mask,
    fixed_all_demand_point_input_bits,
    generate_step_tasks,
    ground_position_km,
    nearest_satellite_id,
    nearest_satellite_ids_vectorized,
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

    def test_vectorized_nearest_matches_scalar_for_clear_visibility(self) -> None:
        points = (
            DemandPoint(lat_deg=0.0, lon_deg=0.0, weight=1.0),
            DemandPoint(lat_deg=25.033, lon_deg=121.5654, weight=1.0),
        )
        time_utc = dt.datetime(2026, 6, 7, 12, tzinfo=dt.timezone.utc)
        satellites = []
        for sat_id, point in enumerate(points):
            ground = ground_position_km(point, time_utc)
            satellites.append(
                SatelliteRuntime(
                    sat_id,
                    f"near_{sat_id}",
                    0,
                    sat_id,
                    1.0,
                    pos_km=tuple(component * 1.2 for component in ground),
                )
            )
        satellites.append(
            SatelliteRuntime(99, "hidden", 0, 99, 1.0, pos_km=(0.0, 0.0, 0.0))
        )

        scalar = [
            nearest_satellite_id(satellites, point, time_utc, 30.0)
            for point in points
        ]
        vectorized = nearest_satellite_ids_vectorized(
            satellites,
            points,
            time_utc,
            30.0,
        )

        self.assertEqual(vectorized, scalar)

    def test_elevation_candidate_mask_rejects_far_side_satellite(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")

        ground_unit = np.array([[1.0, 0.0, 0.0]])
        satellite_unit = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        ground_radii = np.array([6378.0])
        satellite_radii = np.array([6928.0, 6928.0])

        candidates = elevation_candidate_mask(
            ground_unit,
            satellite_unit,
            ground_radii,
            satellite_radii,
            30.0,
        )

        self.assertEqual(candidates.tolist(), [[True, False]])

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

    def test_pending_coverage_vectorizes_unique_coordinates(self) -> None:
        first = Task(0, 0, None, 1.0, 0.0, 120.0, 1.0, 2.0)
        second = Task(1, 0, None, 1.0, 0.0, 120.0, 1.0, 2.0)
        third = Task(2, 0, None, 1.0, 0.0, 120.0, 3.0, 4.0)
        env = EnvironmentRuntime(
            satellites=[],
            time_s=30,
            time_utc=dt.datetime(2026, 6, 7, 12, tzinfo=dt.timezone.utc),
            pending_tasks=[first, second, third],
        )
        config = SimpleNamespace(min_elevation_deg=30.0)

        with patch(
            "satmulator.workload.nearest_satellite_ids_vectorized",
            return_value=[7, None],
        ) as nearest:
            ready, expired = resolve_pending_tasks(env, config)

        nearest.assert_called_once()
        _satellites, points, _time_utc, min_elevation = nearest.call_args.args
        self.assertEqual(
            [(point.lat_deg, point.lon_deg) for point in points],
            [(1.0, 2.0), (3.0, 4.0)],
        )
        self.assertEqual(min_elevation, 30.0)
        self.assertEqual([task.task_id for task in ready], [0, 1])
        self.assertEqual([task.source_sat for task in ready], [7, 7])
        self.assertEqual(expired, [])
        self.assertEqual(env.pending_tasks, [third])

    def test_fixed_all_demand_generates_one_task_per_point_without_pending(self) -> None:
        time_utc = dt.datetime(2026, 6, 7, 12, tzinfo=dt.timezone.utc)
        points = (
            DemandPoint(lat_deg=0.0, lon_deg=0.0, weight=1.0),
            DemandPoint(lat_deg=25.0, lon_deg=121.0, weight=2.0),
        )
        satellites = []
        for sat_id, point in enumerate(points):
            ground = ground_position_km(point, time_utc)
            satellites.append(
                SatelliteRuntime(
                    sat_id,
                    f"sat_{sat_id}",
                    0,
                    sat_id,
                    1.0,
                    pos_km=tuple(component * 1.1 for component in ground),
                )
            )
        env = EnvironmentRuntime(
            satellites=satellites,
            time_s=30,
            time_utc=time_utc,
        )
        task_config = SimpleNamespace(
            enabled=True,
            generation_mode="demand-points-fixed-all",
            interval_s=30,
            demand_distribution=demand_distribution(points),
            min_elevation_deg=0.0,
            input_bits=1234.0,
            output_bits=0.0,
            deadline_s=9999.0,
        )
        compute_config = SimpleNamespace(cycles_per_input_bit=10.0)

        ready, expired = generate_step_tasks(env, task_config, compute_config)

        self.assertEqual(expired, [])
        self.assertEqual(len(ready), 2)
        self.assertEqual([task.input_bits for task in ready], [1234.0, 1234.0])
        self.assertEqual([task.source_sat for task in ready], [0, 1])
        self.assertEqual(env.pending_tasks, [])

    def test_weighted_fixed_all_splits_global_input_by_demand_weight(self) -> None:
        points = (
            DemandPoint(lat_deg=0.0, lon_deg=0.0, weight=1.0),
            DemandPoint(lat_deg=1.0, lon_deg=1.0, weight=3.0),
        )
        distribution = demand_distribution(points)
        task_config = SimpleNamespace(
            generation_mode="demand-points-fixed-weighted-all",
            input_bits=400.0,
            demand_distribution=distribution,
        )

        input_bits = [
            fixed_all_demand_point_input_bits(task_config, point)
            for point in points
        ]

        self.assertEqual(input_bits, [100.0, 300.0])


if __name__ == "__main__":
    unittest.main()
