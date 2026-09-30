import pandas as pd

from app.repositories.detection_repository import _representative_sample


def test_representative_sample_is_stable_and_preserves_class_mix():
    source = pd.DataFrame({
        "rf_predicted_class": (
            ["agricultural_burning"] * 8000
            + ["industrial_fire"] * 1500
            + ["mining_activity"] * 500
        ),
        "acq_date": [f"2024-06-{(index % 10) + 1:02d}" for index in range(10000)],
        "value": range(10000),
    })

    sample_a = _representative_sample(source, 500)
    sample_b = _representative_sample(source, 500)

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
