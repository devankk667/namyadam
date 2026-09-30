import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

EVENT_FILE = "data/processed/event_training_table.parquet"

IMAGE_DIR = (
    r"C:\Users\Vedansh\Downloads\person2_fusion_image_features_v1"
    r"\person2_fusion_image_features_v1\data"
)

SPECTRAL_FILE = os.path.join(
    IMAGE_DIR,
    "image_spectral_features_v1.parquet"
)

EMBEDDING_FILE = os.path.join(
    IMAGE_DIR,
    "image_embeddings_v1.parquet"
)

FOLD_FILE = os.path.join(
    IMAGE_DIR,
    "spatial_folds_v1.csv"
)

OUTPUT_DIR = "ml/models/spatial_fusion"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

EXCLUDED_COLUMNS = [
    "event_id",
    "label",
    "label_confidence",
    "label_source"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

events = pd.read_parquet(EVENT_FILE)
spectral = pd.read_parquet(SPECTRAL_FILE)
embeddings = pd.read_parquet(EMBEDDING_FILE)
folds = pd.read_csv(FOLD_FILE)

print("Event table:", events.shape)
print("Spectral:", spectral.shape)
print("Embeddings:", embeddings.shape)
print("Folds:", folds.shape)


# ============================================================
# VALIDATION
# ============================================================

assert events["event_id"].is_unique
assert spectral["event_id"].is_unique
assert embeddings["event_id"].is_unique
assert folds["event_id"].is_unique

assert len(events) == len(spectral)
assert len(events) == len(embeddings)
assert len(events) == len(folds)


# ============================================================
# MERGE
# ============================================================

# We only take metadata from folds.
fold_metadata = folds[
    [
        "event_id",
        "label",
        "trusted",
        "fold",
        "block_id"
    ]
].copy()

merged = events.merge(
    spectral,
    on="event_id",
    how="left",
    suffixes=("", "_spectral")
)

merged = merged.merge(
    embeddings,
    on="event_id",
    how="left",
    suffixes=("", "_embedding")
)

merged = merged.merge(
    fold_metadata,
    on="event_id",
    how="left",
    suffixes=("", "_fold")
)

print("\nMerged dataset:", merged.shape)


# ============================================================
# TARGET
# ============================================================

# Use the event table label as the authoritative label.
labels = sorted(
    merged["label"].dropna().unique()
)

label_to_id = {
    label: i
    for i, label in enumerate(labels)
}

id_to_label = {
    i: label
    for label, i in label_to_id.items()
}

y = merged["label"].map(
    label_to_id
).values

n_classes = len(labels)

print("\nCLASS MAPPING")

for label, class_id in label_to_id.items():
    print(class_id, "->", label)

print("\nCLASS COUNTS")
print(
    merged["label"].value_counts()
)


# ============================================================
# FEATURE GROUPS
# ============================================================

# ------------------------------------------------------------
# A: Existing 74 tabular features
# ------------------------------------------------------------

A_features = [
    col
    for col in events.columns
    if col not in EXCLUDED_COLUMNS
]


# ------------------------------------------------------------
# B: A + 39 Sentinel spectral features
# ------------------------------------------------------------

spectral_features = [
    col
    for col in spectral.columns
    if col not in [
        "event_id",
        "has_image",
        "label"
    ]
]

B_features = (
    A_features
    + spectral_features
)


# ------------------------------------------------------------
# C: B + 32 frozen ImageNet PCA features
# ------------------------------------------------------------

frozen_pca_features = [
    f"frozen_pca_{i:02d}"
    for i in range(32)
]

C_features = (
    B_features
    + frozen_pca_features
)


# ------------------------------------------------------------
# D: B + 32 fine-tuned CNN PCA
#    + 4 CNN probabilities
# ------------------------------------------------------------

cnn_pca_features = [
    f"pca_{i:02d}"
    for i in range(32)
]

cnn_probability_features = [
    "p_agricultural_burning",
    "p_industrial_fire",
    "p_mining_activity",
    "p_gas_flare"
]

D_features = (
    B_features
    + cnn_pca_features
    + cnn_probability_features
)


experiments = {
    "A_tabular": A_features,
    "B_spectral": B_features,
    "C_spectral_frozen": C_features,
    "D_spectral_cnn": D_features
}


# ============================================================
# FEATURE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE COUNTS")
print("=" * 70)

for name, features in experiments.items():
    print(
        f"{name}: {len(features)} features"
    )


# ============================================================
# CHECK FEATURES EXIST
# ============================================================

for name, features in experiments.items():

    missing = [
        feature
        for feature in features
        if feature not in merged.columns
    ]

    if missing:

        print(
            f"\nMissing features in {name}:"
        )

        print(missing)

        raise ValueError(
            f"{name} has missing features."
        )


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(features):

    numeric_features = [
        col
        for col in features
        if pd.api.types.is_numeric_dtype(
            merged[col]
        )
    ]

    categorical_features = [
        col
        for col in features
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# MODEL
# ============================================================

def build_model():

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        num_class=n_classes
    )


# ============================================================
# OOF STORAGE
# ============================================================

oof_predictions = {}

oof_probabilities = {}

feature_importances = {}


# ============================================================
# TRAIN A / B / C / D
# ============================================================

for experiment_name, features in experiments.items():

    print("\n" + "=" * 70)
    print(experiment_name)
    print("=" * 70)

    X = merged[
        features
    ].copy()

    # --------------------------------------------------------
    # OOF predictions
    # --------------------------------------------------------

    oof_pred = np.zeros(
        len(merged),
        dtype=int
    )

    # --------------------------------------------------------
    # OOF probabilities
    # --------------------------------------------------------

    oof_prob = np.zeros(
        (len(merged), n_classes),
        dtype=float
    )

    # --------------------------------------------------------
    # Store feature importance from each fold
    # --------------------------------------------------------

    all_importances = []

    fold_ids = sorted(
        merged["fold"]
        .dropna()
        .unique()
    )

    # ========================================================
    # FIVE SPATIAL FOLDS
    # ========================================================

    for fold in fold_ids:

        print(
            f"\nFold {int(fold) + 1}/5"
        )

        train_idx = (
            merged["fold"].values
            != fold
        )

        test_idx = (
            merged["fold"].values
            == fold
        )

        X_train = X.loc[
            train_idx
        ]

        X_test = X.loc[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # ----------------------------------------------------
        # Build preprocessing
        # ----------------------------------------------------

        preprocessor = build_preprocessor(
            features
        )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = build_model()

        # ----------------------------------------------------
        # Fit preprocessing ONLY on training fold
        # ----------------------------------------------------

        X_train_processed = (
            preprocessor.fit_transform(
                X_train
            )
        )

        X_test_processed = (
            preprocessor.transform(
                X_test
            )
        )

        # ----------------------------------------------------
        # Train XGBoost
        # ----------------------------------------------------

        model.fit(
            X_train_processed,
            y_train
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test_processed
        )

        y_prob = model.predict_proba(
            X_test_processed
        )

        y_pred = np.asarray(
            y_pred
        ).reshape(-1)

        # ----------------------------------------------------
        # Store OOF predictions
        # ----------------------------------------------------

        oof_pred[
            test_idx
        ] = y_pred

        oof_prob[
            test_idx
        ] = y_prob

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        transformed_names = (
            preprocessor
            .get_feature_names_out()
        )

        fold_importance = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": (
                    model.feature_importances_
                )
            }
        )

        all_importances.append(
            fold_importance
        )

        # ----------------------------------------------------
        # Fold Macro F1
        # ----------------------------------------------------

        fold_macro = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        print(
            "All-event Macro F1:",
            round(
                fold_macro,
                4
            )
        )

    # ========================================================
    # SAVE OOF RESULTS
    # ========================================================

    oof_predictions[
        experiment_name
    ] = oof_pred

    oof_probabilities[
        experiment_name
    ] = oof_prob

    # ========================================================
    # AVERAGE FEATURE IMPORTANCE
    # ========================================================

    importance_all_folds = pd.concat(
        all_importances,
        ignore_index=True
    )

    mean_importance = (
        importance_all_folds
        .groupby(
            "feature"
        )["importance"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    feature_importances[
        experiment_name
    ] = mean_importance

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    print("\nOVERALL")

    # --------------------------------------------------------
    # All-event Macro F1
    # --------------------------------------------------------

    all_macro = f1_score(
        y,
        oof_pred,
        average="macro"
    )

    # --------------------------------------------------------
    # Weighted F1
    # --------------------------------------------------------

    weighted = f1_score(
        y,
        oof_pred,
        average="weighted"
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        oof_pred
    )

    # --------------------------------------------------------
    # Trusted mask
    # --------------------------------------------------------

    trusted_mask = (
        merged["trusted"].values
        == 1
    )

    # --------------------------------------------------------
    # Trusted Macro F1
    # --------------------------------------------------------

    trusted_macro = f1_score(
        y[trusted_mask],
        oof_pred[trusted_mask],
        average="macro"
    )

    # --------------------------------------------------------
    # Trusted without gas flare
    # --------------------------------------------------------

    trusted_no_flare_mask = (
        trusted_mask
        & (
            merged["label"].values
            != "gas_flare"
        )
    )

    trusted_no_flare_macro = f1_score(
        y[
            trusted_no_flare_mask
        ],
        oof_pred[
            trusted_no_flare_mask
        ],
        average="macro"
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        "All-event Macro F1:",
        round(
            all_macro,
            6
        )
    )

    print(
        "Trusted Macro F1:",
        round(
            trusted_macro,
            6
        )
    )

    print(
        "Trusted no-flare Macro F1:",
        round(
            trusted_no_flare_macro,
            6
        )
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            6
        )
    )

    print(
        "Weighted F1:",
        round(
            weighted,
            6
        )
    )

    # --------------------------------------------------------
    # Trusted classification report
    # --------------------------------------------------------

    print(
        "\nTrusted classification report:"
    )

    print(
        classification_report(
            y[trusted_mask],
            oof_pred[trusted_mask],
            labels=list(
                range(n_classes)
            ),
            target_names=[
                id_to_label[i]
                for i in range(n_classes)
            ],
            zero_division=0
        )
    )


# ============================================================
# A VS D PROBABILITY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("A VS D PROBABILITY COMPARISON")
print("=" * 70)

a_prob = oof_probabilities[
    "A_tabular"
]

d_prob = oof_probabilities[
    "D_spectral_cnn"
]

prob_diff = np.abs(
    a_prob - d_prob
)

print(
    "Mean absolute probability difference:",
    prob_diff.mean()
)

print(
    "Maximum probability difference:",
    prob_diff.max()
)

print(
    "Rows with probability change:",
    np.any(
        prob_diff > 1e-6,
        axis=1
    ).sum()
)

print(
    "Rows with probability change > 0.01:",
    np.any(
        prob_diff > 0.01,
        axis=1
    ).sum()
)

print(
    "Rows with probability change > 0.05:",
    np.any(
        prob_diff > 0.05,
        axis=1
    ).sum()
)


# ============================================================
# A VS B VS C VS D HARD PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("A VS B VS C VS D HARD PREDICTION COMPARISON")
print("=" * 70)

a_pred = oof_predictions[
    "A_tabular"
]

for name in [
    "B_spectral",
    "C_spectral_frozen",
    "D_spectral_cnn"
]:

    pred = oof_predictions[
        name
    ]

    different = (
        pred != a_pred
    ).sum()

    print(
        name,
        "different predictions:",
        different,
        "/",
        len(pred)
    )


# ============================================================
# IMAGE FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("IMAGE FEATURE IMPORTANCE")
print("=" * 70)


def is_image_feature(name):

    image_prefixes = (
        "num__ndvi_",
        "num__gndvi_",
        "num__mndwi_",
        "num__ndbi_",
        "num__nbr_",
        "num__nbr2_",
        "num__evi_",
        "num__frac_",
        "num__scl_",
        "num__B02_",
        "num__B03_",
        "num__B04_",
        "num__B08_",
        "num__B11_",
        "num__B12_",
        "num__b12_",
        "num__hot_px_",
        "num__frozen_pca_",
        "num__pca_",
        "num__p_"
    )

    return name.startswith(
        image_prefixes
    )


for experiment_name in [
    "B_spectral",
    "C_spectral_frozen",
    "D_spectral_cnn"
]:

    importance_series = (
        feature_importances[
            experiment_name
        ]
    )

    names = (
        importance_series
        .index
        .tolist()
    )

    importances = (
        importance_series
        .values
    )

    importance_df = pd.DataFrame(
        {
            "feature": names,
            "importance": importances
        }
    )

    image_importance = (
        importance_df[
            importance_df[
                "feature"
            ].apply(
                is_image_feature
            )
        ]
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(
        f"\n{experiment_name}"
    )

    if len(
        image_importance
    ) == 0:

        print(
            "No image features found."
        )

    else:

        print(
            image_importance
            .head(20)
            .to_string(
                index=False
            )
        )


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

prediction_output = merged[
    [
        "event_id",
        "label",
        "trusted",
        "fold"
    ]
].copy()


for experiment_name in experiments:

    prediction_output[
        f"{experiment_name}_pred"
    ] = oof_predictions[
        experiment_name
    ]

    for class_id, class_name in (
        id_to_label.items()
    ):

        prediction_output[
            f"{experiment_name}_p_{class_name}"
        ] = (
            oof_probabilities[
                experiment_name
            ][:, class_id]
        )


prediction_file = os.path.join(
    OUTPUT_DIR,
    "oof_predictions.csv"
)

prediction_output.to_csv(
    prediction_file,
    index=False
)

print(
    "\nSaved:",
    prediction_file
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {}


for experiment_name in experiments:

    pred = oof_predictions[
        experiment_name
    ]

    trusted_mask = (
        merged["trusted"].values
        == 1
    )

    trusted_no_flare_mask = (
        trusted_mask
        & (
            merged["label"].values
            != "gas_flare"
        )
    )

    metrics[
        experiment_name
    ] = {

        "all_macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro"
            )
        ),

        "trusted_macro_f1": float(
            f1_score(
                y[
                    trusted_mask
                ],
                pred[
                    trusted_mask
                ],
                average="macro"
            )
        ),

        "trusted_no_flare_macro_f1": float(
            f1_score(
                y[
                    trusted_no_flare_mask
                ],
                pred[
                    trusted_no_flare_mask
                ],
                average="macro"
            )
        ),

        "accuracy": float(
            accuracy_score(
                y,
                pred
            )
        ),

        "weighted_f1": float(
            f1_score(
                y,
                pred,
                average="weighted"
            )
        )
    }


metrics_file = os.path.join(
    OUTPUT_DIR,
    "metrics.json"
)

with open(
    metrics_file,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

print(
    "Saved:",
    metrics_file
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

for experiment_name in experiments:

    importance_df = (
        feature_importances[
            experiment_name
        ]
        .rename(
            "importance"
        )
        .reset_index()
        .rename(
            columns={
                "index": "feature"
            }
        )
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{experiment_name}_feature_importance.csv"
    )

    importance_df.to_csv(
        output_file,
        index=False
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

# ============================================================
# TOP A VS D PROBABILITY CHANGES
# ============================================================

print("\n" + "=" * 70)
print("EVENTS WITH LARGEST A VS D PROBABILITY CHANGES")
print("=" * 70)

a_prob = oof_probabilities["A_tabular"]
d_prob = oof_probabilities["D_spectral_cnn"]

# Maximum probability difference across the 4 classes
max_prob_diff = np.max(
    np.abs(a_prob - d_prob),
    axis=1
)

# Which class had the maximum difference
max_diff_class = np.argmax(
    np.abs(a_prob - d_prob),
    axis=1
)

diagnostic = pd.DataFrame({
    "event_id": merged["event_id"].values,
    "true_label": merged["label"].values,
    "trusted": merged["trusted"].values,
    "fold": merged["fold"].values,

    "A_prediction": [
        id_to_label[x]
        for x in a_pred
    ],

    "D_prediction": [
        id_to_label[x]
        for x in oof_predictions["D_spectral_cnn"]
    ],

    "max_probability_difference": max_prob_diff,

    "changed_class": [
        id_to_label[x]
        for x in max_diff_class
    ],

    "A_probability_changed_class": [
        a_prob[i, max_diff_class[i]]
        for i in range(len(a_prob))
    ],

    "D_probability_changed_class": [
        d_prob[i, max_diff_class[i]]
        for i in range(len(d_prob))
    ]
})

# Keep only > 0.01
changed_events = (
    diagnostic[
        diagnostic[
            "max_probability_difference"
        ] > 0.01
    ]
    .sort_values(
        "max_probability_difference",
        ascending=False
    )
)

print(
    "\nNumber of events with >1% probability change:",
    len(changed_events)
)

print(
    changed_events.to_string(
        index=False
    )
)

changed_file = os.path.join(
    OUTPUT_DIR,
    "a_vs_d_probability_changes.csv"
)

changed_events.to_csv(
    changed_file,
    index=False
)

print(
    "\nSaved:",
    changed_file
)