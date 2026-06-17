# WorldPop source data

GeoTIFF files in this directory are local inputs and are ignored by git.
Regional and global experiments use WorldPop 2025 R2025A constrained
population-count products at the same 1 km resolution.

## Global 2025, 1 km

- Local file: `global_pop_2025_CN_1km_R2025A_v1.tif`
- Product: WorldPop R2025A v1 constrained population count
- Reference year: 2025
- Resolution: 1 km (`0.0083333333` degrees)
- Source:
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/0_Mosaicked/v1/1km/constrained/global_pop_2025_CN_1km_R2025A_v1.tif

Download:

```bash
curl -fL \
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/0_Mosaicked/v1/1km/constrained/global_pop_2025_CN_1km_R2025A_v1.tif \
  -o data/worldpop/global_pop_2025_CN_1km_R2025A_v1.tif
```

The simulator does not read these rasters directly. Convert them into the
common `lat,lon,weight` demand-point format with
`tools/worldpop_to_demand_points.py`. By default, conversion aggregates source
pixels into `0.1` degree latitude/longitude cells and uses each cell center as
the demand-point coordinate; `weight` is the summed population in that cell.
Pass the canonical source URL so the generated metadata and every experiment
log retain it:

```bash
python3 tools/worldpop_to_demand_points.py \
  data/worldpop/global_pop_2025_CN_1km_R2025A_v1.tif \
  data/demand/global_population_2025_1km.csv \
  --source-url https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/0_Mosaicked/v1/1km/constrained/global_pop_2025_CN_1km_R2025A_v1.tif
```

## Taiwan 2025, 1 km

- Local file: `twn_pop_2025_CN_1km_R2025A_UA_v1.tif`
- Product: WorldPop R2025A v1 constrained population count
- Reference year: 2025
- Resolution: 1 km (`0.0083333333` degrees)
- Source:
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/TWN/v1/1km_ua/constrained/twn_pop_2025_CN_1km_R2025A_UA_v1.tif

Use the official country raster instead of cropping the global raster with a
rectangle. A Taiwan bounding box also contains parts of coastal China and
inflates the resulting population.

Download:

```bash
curl -fL \
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/TWN/v1/1km_ua/constrained/twn_pop_2025_CN_1km_R2025A_UA_v1.tif \
  -o data/worldpop/twn_pop_2025_CN_1km_R2025A_UA_v1.tif
```
