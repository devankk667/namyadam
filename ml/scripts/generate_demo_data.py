"""
Script to generate realistic, scientifically defensible sample datasets for ML pipeline development,
baseline training, and DEMO_MODE execution.
"""

import json
import os
import random
import numpy as np
import pandas as pd

def generate_dataset(num_samples=1200, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # Key real-world regions with representative industrial or natural thermal activity
    regions_info = [
        # Industrial Regions (High density of flares/refineries/power plants)
        {"region": "Houston Ship Channel, USA", "lat_center": 29.70, "lon_center": -95.10, "primary_type": "refinery"},
        {"region": "Permian Basin, Texas, USA", "lat_center": 31.85, "lon_center": -102.35, "primary_type": "flare_stack"},
        {"region": "Ruhr Valley Industrial Hub, Germany", "lat_center": 51.45, "lon_center": 7.01, "primary_type": "steel_works"},
        {"region": "Jurong Island Chemical Hub, Singapore", "lat_center": 1.26, "lon_center": 103.69, "primary_type": "chemical"},
        {"region": "Ahvaz Oil & Gas Field, Iran", "lat_center": 31.32, "lon_center": 48.67, "primary_type": "gas_terminal"},
        {"region": "Jubail Industrial City, Saudi Arabia", "lat_center": 27.01, "lon_center": 49.61, "primary_type": "refinery"},

        # Natural / Agricultural Regions
        {"region": "Amazon Forest Basin, Brazil", "lat_center": -6.00, "lon_center": -62.00, "primary_type": "none"},
        {"region": "Northern Territory Savanna, Australia", "lat_center": -14.50, "lon_center": 132.50, "primary_type": "none"},
        {"region": "Punjab Agricultural Belt, India", "lat_center": 31.14, "lon_center": 75.34, "primary_type": "none"},
        {"region": "Boreal Forest, Boreal Siberia, Russia", "lat_center": 62.00, "lon_center": 129.00, "primary_type": "none"},
        {"region": "California Sierra Nevada, USA", "lat_center": 37.50, "lon_center": -119.50, "primary_type": "none"},
    ]

    classes = [
        "industrial_thermal_source",
        "industrial_fire",
        "wildfire",
        "agricultural_burning",
        "other_thermal_anomaly"
    ]

    records = []

    # We allocate samples across regions
    samples_per_region = num_samples // len(regions_info)

    id_counter = 1001

    for region_id, rinfo in enumerate(regions_info):
        is_industrial_hub = rinfo["primary_type"] != "none"

        for _ in range(samples_per_region):
            # Spatial jitter around regional cluster center
            lat = round(rinfo["lat_center"] + np.random.normal(0, 0.15), 5)
            lon = round(rinfo["lon_center"] + np.random.normal(0, 0.15), 5)

            if is_industrial_hub:
                # Distribution in industrial zones: mostly persistent sources, some industrial fires
                p_class = [0.65, 0.20, 0.05, 0.05, 0.05]
            else:
                # Distribution in natural zones: wildfires, ag burning, other
                p_class = [0.02, 0.03, 0.55, 0.30, 0.10]

            chosen_class = np.random.choice(classes, p=p_class)
            is_industrial = 1 if chosen_class in ["industrial_fire", "industrial_thermal_source"] else 0

            # Feature generation conditional on class
            if chosen_class == "industrial_thermal_source":
                # Continuous, high brightness, moderate FRP, highly persistent over time, close to industrial site
                dist_to_ind = round(float(np.random.exponential(scale=0.25) + 0.02), 3) # 20m to 1km
                ind_count_2k = int(np.random.poisson(lam=5) + 1)
                ind_count_5k = ind_count_2k + int(np.random.poisson(lam=8))
                facility_type = rinfo["primary_type"] if rinfo["primary_type"] != "none" else np.random.choice(["refinery", "flare_stack", "gas_terminal", "chemical"])

                brightness = round(float(np.random.normal(355, 18)), 2)
                bright_t31 = round(float(np.random.normal(295, 8)), 2)
                frp = round(float(np.random.gamma(shape=2.5, scale=12.0)), 2)
                confidence = round(float(np.random.uniform(85, 100)), 1)
                persistence_score = round(float(np.random.beta(a=8, b=2)), 3) # high persistence 0.7-1.0
                detection_count_30d = int(np.random.poisson(lam=22) + 5)

            elif chosen_class == "industrial_fire":
                # High FRP, high brightness, close to industrial infrastructure, sudden outbreak (moderate 30d persistence prior)
                dist_to_ind = round(float(np.random.exponential(scale=0.4) + 0.05), 3)
                ind_count_2k = int(np.random.poisson(lam=3) + 1)
                ind_count_5k = ind_count_2k + int(np.random.poisson(lam=6))
                facility_type = rinfo["primary_type"] if rinfo["primary_type"] != "none" else np.random.choice(["refinery", "chemical", "power_plant", "steel_works"])

                brightness = round(float(np.random.normal(385, 25)), 2)
                bright_t31 = round(float(np.random.normal(310, 12)), 2)
                frp = round(float(np.random.gamma(shape=4.0, scale=35.0) + 20), 2)
                confidence = round(float(np.random.uniform(80, 100)), 1)
                persistence_score = round(float(np.random.beta(a=3, b=5)), 3) # moderate/low historical persistence
                detection_count_30d = int(np.random.poisson(lam=6) + 1)

            elif chosen_class == "wildfire":
                # High FRP, large thermal spread, far from industrial sites, low persistence over long horizon
                dist_to_ind = round(float(np.random.exponential(scale=15.0) + 3.0), 3) # 3km to 50km+
                ind_count_2k = 0
                ind_count_5k = int(np.random.poisson(lam=0.3))
                facility_type = "none"

                brightness = round(float(np.random.normal(365, 20)), 2)
                bright_t31 = round(float(np.random.normal(300, 10)), 2)
                frp = round(float(np.random.gamma(shape=3.5, scale=40.0) + 15), 2)
                confidence = round(float(np.random.uniform(70, 100)), 1)
                persistence_score = round(float(np.random.beta(a=1.5, b=8)), 3) # low persistence
                detection_count_30d = int(np.random.poisson(lam=3) + 1)

            elif chosen_class == "agricultural_burning":
                # Moderate FRP, low/moderate brightness, far from heavy industrial, seasonal, transient
                dist_to_ind = round(float(np.random.exponential(scale=12.0) + 2.0), 3)
                ind_count_2k = 0
                ind_count_5k = int(np.random.poisson(lam=0.5))
                facility_type = "none"

                brightness = round(float(np.random.normal(330, 12)), 2)
                bright_t31 = round(float(np.random.normal(290, 6)), 2)
                frp = round(float(np.random.gamma(shape=2.0, scale=8.0) + 3), 2)
                confidence = round(float(np.random.uniform(60, 95)), 1)
                persistence_score = round(float(np.random.beta(a=1, b=10)), 3) # very low persistence
                detection_count_30d = int(np.random.poisson(lam=2) + 1)

            else: # other_thermal_anomaly
                dist_to_ind = round(float(np.random.exponential(scale=8.0) + 1.0), 3)
                ind_count_2k = int(np.random.poisson(lam=0.5))
                ind_count_5k = int(np.random.poisson(lam=1.5))
                facility_type = np.random.choice(["none", "factory", "power_plant"])

                brightness = round(float(np.random.normal(320, 10)), 2)
                bright_t31 = round(float(np.random.normal(288, 5)), 2)
                frp = round(float(np.random.gamma(shape=1.5, scale=5.0) + 1), 2)
                confidence = round(float(np.random.uniform(50, 85)), 1)
                persistence_score = round(float(np.random.beta(a=2, b=6)), 3)
                detection_count_30d = int(np.random.poisson(lam=3) + 1)

            # Metadata properties
            month = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            day = np.random.randint(1, 28)
            hour = np.random.randint(0, 24)
            minute = np.random.choice([0, 15, 30, 45])

            acq_date = f"2025-{month:02d}-{day:02d}"
            acq_time = f"{hour:02d}{minute:02d}"
            satellite = np.random.choice(["Suomi-NPP", "NOAA-20", "Aqua", "Terra"], p=[0.4, 0.4, 0.1, 0.1])
            instrument = "VIIRS" if satellite in ["Suomi-NPP", "NOAA-20"] else "MODIS"
            daynight = "D" if 6 <= hour <= 18 else "N"

            record = {
                "id": f"DET-2025-{id_counter:04d}",
                "latitude": lat,
                "longitude": lon,
                "acq_date": acq_date,
                "acq_time": acq_time,
                "brightness": brightness,
                "bright_t31": bright_t31,
                "frp": frp,
                "confidence": confidence,
                "satellite": satellite,
                "instrument": instrument,
                "daynight": daynight,
                "dist_to_industrial": dist_to_ind,
                "industrial_count_2km": ind_count_2k,
                "industrial_count_5km": ind_count_5k,
                "nearest_facility_type": facility_type,
                "persistence_score": persistence_score,
                "detection_count_30d": detection_count_30d,
                "region": rinfo["region"],
                "spatial_cluster_id": region_id,
                "true_class": chosen_class,
                "is_industrial": is_industrial
            }
            records.append(record)
            id_counter += 1

    df = pd.DataFrame(records)
    return df

def main():
    os.makedirs("data/demo", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = generate_dataset(num_samples=1200, seed=42)

    # Save full processed dataset CSV for training pipeline
    csv_path = "data/processed/thermal_detections_processed.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} dataset records saved to {csv_path}")

    # Save demo JSON format for DEMO_MODE frontend & backend
    demo_records = df.to_dict(orient="records")
    json_path = "data/demo/thermal_detections_demo.json"
    with open(json_path, "w") as f:
        json.dump(demo_records, f, indent=2)
    print(f"Saved {len(demo_records)} demo records to {json_path}")

if __name__ == "__main__":
    main()
