import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_image_fusion.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("train_image_fusion", SCRIPT_PATH)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
fusion = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(fusion)

build_feature_sets = fusion.build_feature_sets
image_feature_groups = fusion.image_feature_groups
score_predictions = fusion.score_predictions
trusted_sample_weights = fusion.trusted_sample_weights
align_labels_to_spatial_folds = fusion.align_labels_to_spatial_folds


def test_image_feature_groups_exclude_full_embedding_and_metadata():
    columns = [
        "event_id", "label", "trusted", "fold", "p_agricultural_burning",
        "p_industrial_fire", "pca_00", "pca_31", "frozen_pca_00", "emb_000",
    ]

    groups = image_feature_groups(columns)

    assert groups == {
        "pca": ["pca_00", "pca_31"],
        "frozen_pca": ["frozen_pca_00"],
        "probabilities": ["p_agricultural_burning", "p_industrial_fire"],
    }


def test_fusion_feature_sets_follow_ablation_contract():
    groups = {
        "pca": ["pca_00"],
        "frozen_pca": ["frozen_pca_00"],
        "probabilities": ["p_industrial_fire"],
    }
    feature_sets = build_feature_sets(
        ["max_frp", "nearest_osm_distance_km"],
        ["ndvi_mean"],
        groups,
        "has_image",
    )

    assert feature_sets["A_tabular"] == ["max_frp", "nearest_osm_distance_km"]
    assert "ndvi_mean" in feature_sets["B_tabular_spectral"]
    assert "frozen_pca_00" in feature_sets["C_tabular_spectral_frozen"]
    assert "pca_00" in feature_sets["D_tabular_spectral_cnn"]
    assert "nearest_osm_distance_km" not in feature_sets["E_image_only"]
    assert "has_image" in feature_sets["E_image_only"]
    assert all(len(columns) == len(set(columns)) for columns in feature_sets.values())


def test_unassigned_local_labels_are_excluded_without_inventing_folds():
    labels = fusion.pd.DataFrame({
        "event_id": ["event-1", "event-2", "event-new"],
        "label": ["agricultural_burning", "industrial_fire", "mining_activity"],
        "label_confidence": ["low", "low", "low"],
    })
    folds = fusion.pd.DataFrame({
        "event_id": ["event-1", "event-2", "event-old"],
        "fold_label": ["agricultural_burning", "mining_activity", "industrial_fire"],
        "fold_label_confidence": ["low", "high", "low"],
    })

    aligned, report = align_labels_to_spatial_folds(labels, folds)

    assert aligned["event_id"].tolist() == ["event-1", "event-2", "event-old"]
    assert aligned.loc[aligned["event_id"] == "event-2", "label"].item() == "mining_activity"
    assert report["local_labels_without_fold"] == ["event-new"]
    assert report["fold_events_without_local_label"] == ["event-old"]
    assert report["local_label_mismatches"] == ["event-2"]
    assert report["local_confidence_mismatches"] == ["event-2"]


def test_trusted_labels_are_weighted_without_discarding_weak_labels():
    weights = trusted_sample_weights(np.array([True, False, True]), trusted_weight=5)

    assert weights.tolist() == [5.0, 1.0, 5.0]


def test_empty_fold_trusted_metrics_are_well_formed():
    scores = score_predictions(np.array([]), np.array([]))

    assert scores["n"] == 0
    assert scores["macro_f1"] == 0.0
    assert len(scores["confusion_matrix"]) == 4


def test_no_flare_metric_counts_flare_predictions_as_errors():
    scores = score_predictions(
        np.array(["agricultural_burning", "industrial_fire", "mining_activity"]),
        np.array(["gas_flare", "agricultural_burning", "mining_activity"]),
    )

    assert abs(scores["macro_f1_no_flare"] - (1 / 3)) < 1e-9


def test_scores_include_trusted_primary_metric_and_no_flare_variant():
    scores = score_predictions(
        np.array(["agricultural_burning", "industrial_fire", "mining_activity"]),
        np.array(["agricultural_burning", "agricultural_burning", "mining_activity"]),
    )

    assert scores["n"] == 3
    assert 0 <= scores["macro_f1"] <= 1
    assert 0 <= scores["macro_f1_no_flare"] <= 1
    assert len(scores["confusion_matrix"]) == 4
