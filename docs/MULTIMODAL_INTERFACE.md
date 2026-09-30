# Multimodal ML Interface

## Tabular Input

Each thermal event must have:

- event_id
- brightness
- bright_t31
- temp_difference
- frp
- confidence
- is_nighttime
- dist_to_industrial
- industrial_count_2km
- industrial_count_5km
- persistence_score
- detection_count_30d
- nearest_facility_type

## Image Input

Person 1 provides one Sentinel-2 embedding per event.

Format:

event_id, embedding_0, embedding_1, ..., embedding_511

Embedding dimension:

512

## Join Key

The common key is:

event_id

## Final Fusion

Tabular features → tabular encoder
Image embedding → image encoder
Both representations → concatenation → classifier

## Target

5 classes:

1. industrial_thermal_source
2. industrial_fire
3. wildfire
4. agricultural_burning
5. other_thermal_anomaly