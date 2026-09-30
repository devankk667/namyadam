import pandas as pd

from app.repositories.detection_repository import ThermalDetectionRepository, _representative_sample


def test_representative_sample_is_reproducible_with_seed_and_preserves_class_mix():
    source = pd.DataFrame({
        "rf_predicted_class": (
            ["agricultural_burning"] * 8000
            + ["industrial_fire"] * 1500
            + ["mining_activity"] * 500
        ),
        "acq_date": [f"2024-06-{(index % 10) + 1:02d}" for index in range(10000)],
        "value": range(10000),
    })

    sample_a = _representative_sample(source, 500, seed=42)
    sample_b = _representative_sample(source, 500, seed=42)

    assert len(sample_a) == 500
    assert sample_a["value"].tolist() == sample_b["value"].tolist()
    ratios = sample_a["rf_predicted_class"].value_counts(normalize=True)
    assert abs(ratios["agricultural_burning"] - 0.8) < 0.03
    assert abs(ratios["industrial_fire"] - 0.15) < 0.03
    assert abs(ratios["mining_activity"] - 0.05) < 0.02


def test_representative_sample_handles_high_cardinality_strata():
    source = pd.DataFrame({
        "rf_predicted_class": ["class_a", "class_b"] * 600,
        "acq_date": [f"2024-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}" for index in range(1200)],
    })

    sample = _representative_sample(source, 100)

    assert len(sample) == 100
    assert set(sample["rf_predicted_class"]) == {"class_a", "class_b"}


def test_unseeded_representative_sample_varies_between_backend_loads():
    source = pd.DataFrame({
        "rf_predicted_class": ["class_a", "class_b"] * 600,
        "acq_date": [f"2024-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}" for index in range(1200)],
        "value": range(1200),
    })

    first_load = _representative_sample(source, 100)
    next_load = _representative_sample(source, 100)

    assert len(first_load) == len(next_load) == 100
    assert set(first_load["value"]) != set(next_load["value"])


def test_event_map_joins_sampled_rows_by_source_id(monkeypatch):
    repository = ThermalDetectionRepository.__new__(ThermalDetectionRepository)
    repository.total_source_records = 3
    source_map = pd.DataFrame({
        "obs_id": [0, 1, 2],
        "event_id": ["event-0", "event-1", "event-2"],
    })
    monkeypatch.setattr("app.repositories.detection_repository.os.path.isfile", lambda _path: True)
    monkeypatch.setattr(
        "app.repositories.detection_repository.pd.read_parquet",
        lambda *_args, **_kwargs: source_map.copy(),
    )

    sampled = pd.DataFrame({"value": [20, 0]}, index=[2, 0])

    assert repository._event_ids_for_sample(sampled) == ["event-2", "event-0"]


def test_event_map_rejects_non_exact_or_invalid_coverage(monkeypatch):
    repository = ThermalDetectionRepository.__new__(ThermalDetectionRepository)
    repository.total_source_records = 3
    monkeypatch.setattr("app.repositories.detection_repository.os.path.isfile", lambda _path: True)

    invalid_maps = [
        pd.DataFrame({"obs_id": [0, 1, 1], "event_id": ["e0", "e1", "e2"]}),
        pd.DataFrame({"obs_id": [0, 1.5, 2], "event_id": ["e0", "e1", "e2"]}),
        pd.DataFrame({"obs_id": [0, 1, 2], "event_id": ["e0", None, "e2"]}),
    ]
    for invalid_map in invalid_maps:
        monkeypatch.setattr(
            "app.repositories.detection_repository.pd.read_parquet",
            lambda *_args, mapping=invalid_map, **_kwargs: mapping.copy(),
        )
        assert repository._event_ids_for_sample(pd.DataFrame(index=[0, 2])) == [None, None]
