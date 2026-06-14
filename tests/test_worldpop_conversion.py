import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "worldpop_to_demand_points.py"
SPEC = importlib.util.spec_from_file_location("worldpop_to_demand_points", SCRIPT)
worldpop = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = worldpop
SPEC.loader.exec_module(worldpop)


class WorldPopConversionTests(unittest.TestCase):
    def test_aggregation_preserves_population_and_uses_populated_source_point(self) -> None:
        bins = worldpop.aggregate_points(
            [
                (25.00, 121.00, 10.0),
                (25.08, 121.08, 30.0),
                (26.00, 122.00, 5.0),
                (0.0, 0.0, 0.0),
            ],
            aggregate_deg=0.1,
        )

        self.assertEqual(len(bins), 2)
        first = bins[worldpop.bin_key(25.00, 121.00, 0.1)]
        self.assertEqual(first.population, 40.0)
        self.assertAlmostEqual(first.lat, 25.08)
        self.assertAlmostEqual(first.lon, 121.08)
        self.assertEqual(sum(cell.population for cell in bins.values()), 45.0)

    def test_write_csv_filters_small_bins_and_reports_preserved_population(self) -> None:
        bins = worldpop.aggregate_points(
            [(25.0, 121.0, 10.0), (26.0, 122.0, 0.5)],
            aggregate_deg=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"
            count, population = worldpop.write_csv(path, bins, min_population=1.0)
            with path.open() as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(count, 1)
        self.assertEqual(population, 10.0)
        self.assertEqual(rows[0]["weight"], "10.000000")

    def test_reads_geotiff_and_applies_bbox(self) -> None:
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            self.skipTest("rasterio is not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.tif"
            values = np.array([[10, 20], [30, -999]], dtype="float32")
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype=values.dtype,
                crs="EPSG:4326",
                transform=from_origin(120.0, 25.0, 0.1, 0.1),
                nodata=-999,
            ) as dataset:
                dataset.write(values, 1)

            points = list(
                worldpop.iter_population_points(
                    path,
                    bbox=(120.0, 24.9, 120.1, 25.0),
                )
            )

        self.assertEqual(len(points), 1)
        lat, lon, population = points[0]
        self.assertAlmostEqual(lat, 24.95)
        self.assertAlmostEqual(lon, 120.05)
        self.assertEqual(population, 10.0)


if __name__ == "__main__":
    unittest.main()
