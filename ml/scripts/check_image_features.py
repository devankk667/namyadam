import pandas as pd


SPECTRAL_FILE = (
    r"C:\Users\Vedansh\Downloads\person2_fusion_image_features_v1"
    r"\person2_fusion_image_features_v1\data"
    r"\image_spectral_features_v1.parquet"
)

EMBEDDING_FILE = (
    r"C:\Users\Vedansh\Downloads\person2_fusion_image_features_v1"
    r"\person2_fusion_image_features_v1\data"
    r"\image_embeddings_v1.parquet"
)

FOLD_FILE = (
    r"C:\Users\Vedansh\Downloads\person2_fusion_image_features_v1"
    r"\person2_fusion_image_features_v1\data"
    r"\spatial_folds_v1.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

spectral = pd.read_parquet(SPECTRAL_FILE)
embeddings = pd.read_parquet(EMBEDDING_FILE)
folds = pd.read_csv(FOLD_FILE)


# ------------------------------------------------------------
# FEATURE GROUPS
# ------------------------------------------------------------

spectral_features = [
    c
    for c in spectral.columns
    if c not in ["event_id", "has_image", "label"]
]

frozen_features = [
    f"frozen_pca_{i:02d}"
    for i in range(32)
]

cnn_pca_features = [
    f"pca_{i:02d}"
    for i in range(32)
]

cnn_prob_features = [
    "p_agricultural_burning",
    "p_industrial_fire",
    "p_mining_activity",
    "p_gas_flare",
]


# ------------------------------------------------------------
# MERGE ONLY ON EVENT ID
# ------------------------------------------------------------

df = folds[
    [
        "event_id",
        "label",
        "fold",
        "trusted",
    ]
].merge(
    spectral[
        ["event_id"] + spectral_features
    ],
    on="event_id",
    how="left",
    validate="one_to_one",
).merge(
    embeddings[
        ["event_id"]
        + frozen_features
        + cnn_pca_features
        + cnn_prob_features
    ],
    on="event_id",
    how="left",
    validate="one_to_one",
)


# ------------------------------------------------------------
# BASIC CHECK
# ------------------------------------------------------------

print("\nDATASET")
print("Shape:", df.shape)
print("Unique events:", df.event_id.nunique())

print("\nCLASS COUNTS")
print(df.label.value_counts())


# ------------------------------------------------------------
# USABLE ROWS
# ------------------------------------------------------------

print("\nUSABLE ROWS")

print(
    "Spectral:",
    df[spectral_features]
    .notna()
    .all(axis=1)
    .sum()
)

print(
    "Frozen PCA:",
    df[frozen_features]
    .notna()
    .all(axis=1)
    .sum()
)

print(
    "CNN PCA:",
    df[cnn_pca_features]
    .notna()
    .all(axis=1)
    .sum()
)

print(
    "CNN probabilities:",
    df[cnn_prob_features]
    .notna()
    .all(axis=1)
    .sum()
)


# ------------------------------------------------------------
# VARIANCE
# ------------------------------------------------------------

print("\nFEATURE VARIANCE")

print(
    "Spectral mean variance:",
    df[spectral_features]
    .var()
    .mean()
)

print(
    "Frozen PCA mean variance:",
    df[frozen_features]
    .var()
    .mean()
)

print(
    "CNN PCA mean variance:",
    df[cnn_pca_features]
    .var()
    .mean()
)


# ------------------------------------------------------------
# CNN PROBABILITIES
# ------------------------------------------------------------

print("\nCNN PROBABILITY SUMMARY")

print(
    df[cnn_prob_features]
    .describe()
    .T[
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    ]
)


# ------------------------------------------------------------
# CNN PREDICTION DISTRIBUTION
# ------------------------------------------------------------

print("\nCNN PREDICTION DISTRIBUTION")

print(
    embeddings["cnn_pred"]
    .value_counts(dropna=False)
)


# ------------------------------------------------------------
# CORRELATION WITH LABEL
# ------------------------------------------------------------

print("\nMEAN CNN PROBABILITIES BY TRUE LABEL")

print(
    df.groupby("label")[
        cnn_prob_features
    ].mean()
)