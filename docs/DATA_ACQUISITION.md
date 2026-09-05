# Data Acquisition Strategy & Specifications

This document outlines the data sources, access protocols, field definitions, spatial context integration, and satellite imagery acquisition strategies for the **Industrial Thermal Anomaly Detection & Monitoring System**.

---

## 1. NASA FIRMS (Fire Information for Resource Management System)

### Data Product & Specifications
- **Primary Product**: VIIRS Active Fire / Thermal Anomalies (375m resolution, VNP14IMGTDL / VNP14IMGTDL_NRT) & MODIS Active Fire (1km resolution).
- **Access Method**:
  - Direct CSV/GeoJSON API via `https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/{SOURCE}/{COUNTRY}/{DAYS}`
  - Archival NRT CSV bulk downloads via NASA Earthdata portal.
- **Required Credentials**: Free MAP_KEY requested from [NASA FIRMS API Key Request](https://firms.modaps.eosdis.nasa.gov/api/map_key/).
- **Key Schema Fields**:
  - `latitude`: Latitude of center of 375m / 1km fire pixel (decimal degrees).
  - `longitude`: Longitude of center of fire pixel (decimal degrees).
  - `bright_ti4` / `brightness`: Channel I-4 brightness temperature (Kelvin).
  - `scan`, `track`: Pixel spatial dimensions along and across scan.
  - `acq_date`: Acquisition date (`YYYY-MM-DD`).
  - `acq_time`: Acquisition time in UTC (`HHMM`).
  - `satellite`: Observing satellite platform (`N` = Suomi NPP, `1` = NOAA-20, `A` = Aqua, `T` = Terra).
  - `instrument`: Sensing instrument (`VIIRS` or `MODIS`).
  - `confidence`: Detection confidence metric (`l` = low, `n` = nominal, `h` = high, or 0-100%).
  - `bright_ti5` / `bright_t31`: Channel I-5 / T31 brightness temperature (Kelvin).
  - `frp`: Fire Radiative Power in Megawatts (MW).
  - `daynight`: Acquisition flag (`D` = Day, `N` = Night).

### Coverage & Temporal Characteristics
- **Geographic Coverage**: Global ($90^\circ\text{N}$ to $90^\circ\text{S}$).
- **Temporal Coverage**: 2000–present (MODIS), 2012–present (VIIRS).
- **Limitations**:
  - Overpass frequency: 2–4 passes per day per location.
  - Thermal masking by thick clouds, intense smoke, or heavy optical clutter.
  - Resolution limitations: Cannot pinpoint exact sub-pixel source without higher-resolution satellite or geospatial context.

---

## 2. OpenStreetMap (OSM) Geospatial Context

### Strategy & Access Protocol
- **Access Method**: Overpass API (`https://overpass-api.de/api/interpreter`) or local planet OSM dumps via `osmnx` / `pyrosm`.
- **Target Industrial Categories**:
  - `landuse=industrial`
  - `industrial=refinery` | `oil_gas` | `chemical` | `smelter` | `steel_works` | `factory`
  - `man_made=flare` | `chimney` | `petroleum_well` | `works`
  - `power=plant` | `generator`
- **Derived Feature Metrics**:
  - `dist_to_industrial`: Distance (km) to nearest registered industrial entity.
  - `industrial_count_2km`: Number of industrial entities within a 2.0 km radius.
  - `industrial_count_5km`: Number of industrial entities within a 5.0 km radius.
  - `nearest_facility_type`: Categorical label of nearest facility (e.g., `refinery`, `power_plant`, `chemical`, `steel_works`, `none`).

---

## 3. Satellite Optical & Infrared Imagery

### Strategy & Access Protocol
- **Primary Source**: Sentinel-2 L2A Multi-Spectral Instrument (MSI) via AWS Open Data Cloud-Optimized GeoTIFFs (STAC API).
- **Channels Used**:
  - B12 (Short-Wave Infrared 2 - $2202\text{ nm}$)
  - B11 (Short-Wave Infrared 1 - $1610\text{ nm}$)
  - B8A (Narrow Near Infrared - $865\text{ nm}$)
  - RGB Composite (B04, B03, B02)
- **Patch Extraction Process**:
  1. Query STAC API for optical imagery within $\pm 3$ days of detection date.
  2. Crop a $64 \times 64$ or $128 \times 128$ pixel patch centered at detection (latitude, longitude).
  3. Normalize surface reflectance values to $[0, 1]$.

---

## 4. Demo Data Strategy (`DEMO_MODE=true`)

When external APIs (NASA FIRMS, Overpass, Sentinel STAC) are unavailable or during offline demonstrations:
1. The application automatically falls back to bundled demo dataset (`data/demo/thermal_detections_demo.json`).
2. The demo dataset contains realistic, validated geographic locations across major industrial hubs (e.g., Permian Basin, Gulf Coast Refineries, Ruhr Valley, Persian Gulf Flares, Siberian Gas Terminals) and non-industrial control regions (wildfires, agricultural burning).
3. All UI indicators explicitly highlight `DEMO DATA` status to maintain full scientific transparency.
