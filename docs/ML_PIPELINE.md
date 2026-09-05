# ML Pipeline & Spatial Leakage-Safe Strategy

## 1. Spatial Leakage Prevention Strategy
Standard random cross-validation or standard train/test splits lead to severe **data leakage** in geospatial machine learning tasks. Because satellite passes observe adjacent pixels or repeated occurrences over the same industrial facility or forest area, random row splitting causes samples from the exact same physical spatial location or facility cluster to appear in both train and test sets.

To ensure scientifically valid and generalizable evaluation, this project implements a **Facility-Aware Spatial Grouping Split Strategy**:

1. **Spatial Clustering Grouping (`spatial_cluster_id`)**: Detections are assigned to geographic clusters based on region and spatial proximity (e.g. facility or hub boundary).
2. **GroupKFold / GroupShuffleSplit**: Train/Test splits strictly group by `spatial_cluster_id`.
3. **Out-of-Region Validation**: The test set comprises entirely unseen geographical regions / industrial clusters. This measures the model's true capability to generalize to new, unseen global facilities or fire incidents.

---

## 2. Feature Engineering

### FIRMS Thermal Features
- `brightness`: Channel I-4 / T4 brightness temperature (Kelvin)
- `bright_t31`: Channel I-5 / T31 brightness temperature (Kelvin)
- `temp_difference`: Calculated as `brightness - bright_t31` (key physical indicator for intense high-temperature combustion)
- `frp`: Fire Radiative Power (MW)
- `confidence`: Satellite detection confidence percentage
- `is_nighttime`: Binary indicator derived from `daynight == 'N'`

### Geospatial Context Features (OSM Context)
- `dist_to_industrial`: Proximity (km) to nearest industrial entity
- `industrial_count_2km`: Infrastructure density within 2.0 km
- `industrial_count_5km`: Infrastructure density within 5.0 km
- `nearest_facility_type_encoded`: One-hot or label encoded facility type

### Persistence Metrics
- `persistence_score`: Normalized continuous score ($0.0$ to $1.0$) indicating historical recurring detections at location.
- `detection_count_30d`: Count of thermal anomalies recorded at location in trailing 30 days.

---

## 3. Multimodal & DL Architecture
For tabular data (FIRMS + OSM features), tree-based models (XGBoost, Random Forest) and Logistic Regression serve as robust baselines.
For deep learning, a PyTorch Multimodal Neural Network combines tabular embeddings with spatial/spectral embeddings to classify multi-class thermal sources.
