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
- SHA-256:
  `478d81441d39b0548c6f1a1a1d713ac7a2bf32483671c4fe0864f2d8f16cc56c`

Download:

```bash
curl -fL \
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/0_Mosaicked/v1/1km/constrained/global_pop_2025_CN_1km_R2025A_v1.tif \
  -o data/worldpop/global_pop_2025_CN_1km_R2025A_v1.tif
```

The simulator does not read these rasters directly. Convert them into the
common `lat,lon,weight` demand-point format with
`tools/worldpop_to_demand_points.py`.

## Taiwan 2025, 1 km

- Local file: `twn_pop_2025_CN_1km_R2025A_UA_v1.tif`
- Product: WorldPop R2025A v1 constrained population count
- Reference year: 2025
- Resolution: 1 km (`0.0083333333` degrees)
- Source:
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/TWN/v1/1km_ua/constrained/twn_pop_2025_CN_1km_R2025A_UA_v1.tif
- SHA-256:
  `eae984f0d741db820081f82ea989a743fb2a75c53da30af85a8618aaac682f0e`

Use the official country raster instead of cropping the global raster with a
rectangle. A Taiwan bounding box also contains parts of coastal China and
inflates the resulting population.

Download:

```bash
curl -fL \
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/TWN/v1/1km_ua/constrained/twn_pop_2025_CN_1km_R2025A_UA_v1.tif \
  -o data/worldpop/twn_pop_2025_CN_1km_R2025A_UA_v1.tif
```
