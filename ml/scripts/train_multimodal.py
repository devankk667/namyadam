"""
Train the true multimodal thermal classification model.

Inputs:
- Tabular thermal/geospatial features
- 512-dimensional Sentinel-2 image embeddings
"""

import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from preprocessing import prepare_features, TARGET_CLASSES
from pytorch_model import MultimodalThermalNet


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/processed/thermal_detections_processed.csv"
IMAGE_FILE = "data/processed/sentinel_embeddings_demo.csv"
OUTPUT_DIR = "ml/models/multimodal_demo"

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading tabular data...")
df = pd.read_csv(DATA_FILE)

print("Loading image embeddings...")
image_df = pd.read_csv(IMAGE_FILE)

print(f"Tabular data shape: {df.shape}")
print(f"Image embedding shape: {image_df.shape}")


# ============================================================
# MERGE
# ============================================================

df = df.merge(
    image_df,
    on="id",
    how="inner",
    validate="one_to_one"
)

print(f"Merged data shape: {df.shape}")

assert len(df) == len(image_df)
assert df["id"].is_unique


# ============================================================
# IMAGE FEATURES
# ============================================================

IMAGE_FEATURES = [
    f"embedding_{i}"
    for i in range(512)
]


# ============================================================
# TARGET
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df["true_class"])

print("\nClasses:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i}: {cls}")


# ============================================================
# SPATIAL TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        df,
        y,
        groups=df["spatial_cluster_id"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print(
    f"\nSpatial Group Split complete: "
    f"Train samples={len(train_df)}, "
    f"Test samples={len(test_df)}"
)


# ============================================================
# TABULAR FEATURES
# ============================================================

X_tab_train, scaler, feature_names = prepare_features(
    train_df,
    fit=True
)

X_tab_test, _, _ = prepare_features(
    test_df,
    scaler=scaler,
    fit=False
)


# ============================================================
# IMAGE FEATURES
# ============================================================

X_img_train = train_df[IMAGE_FEATURES].values.astype(np.float32)
X_img_test = test_df[IMAGE_FEATURES].values.astype(np.float32)

y_train = y[train_idx]
y_test = y[test_idx]

print(f"\nTabular feature shape: {X_tab_train.shape}")
print(f"Image feature shape:   {X_img_train.shape}")


# ============================================================
# PYTORCH DATASET
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_tab_train, dtype=torch.float32),
    torch.tensor(X_img_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

model = MultimodalThermalNet(
    tabular_dim=X_tab_train.shape[1],
    image_dim=X_img_train.shape[1],
    num_classes=len(TARGET_CLASSES)
).to(device)


criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nTraining multimodal model...")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for bx_tab, bx_img, by in train_loader:

        bx_tab = bx_tab.to(device)
        bx_img = bx_img.to(device)
        by = by.to(device)

        optimizer.zero_grad()

        logits = model(bx_tab, bx_img)

        loss = criterion(logits, by)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {avg_loss:.4f}"
        )


# ============================================================
# EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    tab_inputs = torch.tensor(
        X_tab_test,
        dtype=torch.float32
    ).to(device)

    img_inputs = torch.tensor(
        X_img_test,
        dtype=torch.float32
    ).to(device)

    logits = model(
        tab_inputs,
        img_inputs
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    ).cpu().numpy()

    predictions = np.argmax(
        probabilities,
        axis=1
    )


# ============================================================
# METRICS
# ============================================================

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

macro_precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
    multi_class="ovr",
    average="macro"
)


print("\n--- MULTIMODAL MODEL RESULTS ---")

print(f"Macro F1:        {macro_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"ROC-AUC:         {roc_auc:.4f}")


# ============================================================
# SAVE ARTIFACTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    os.path.join(
        OUTPUT_DIR,
        "model.pt"
    )
)

import joblib

joblib.dump(
    scaler,
    os.path.join(
        OUTPUT_DIR,
        "scaler.joblib"
    )
)

metadata = {
    "model_type": "multimodal_thermal_net",
    "tabular_dim": X_tab_train.shape[1],
    "image_dim": X_img_train.shape[1],
    "num_classes": len(TARGET_CLASSES),
    "classes": list(label_encoder.classes_),
    "feature_names": feature_names,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "macro_f1": float(macro_f1),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "roc_auc": float(roc_auc),
    "image_embeddings": "DEMO_RANDOM_EMBEDDINGS"
}

with open(
    os.path.join(OUTPUT_DIR, "metadata.json"),
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )


print(
    f"\nArtifacts saved to: {OUTPUT_DIR}"
)

print(
    "\nNOTE: Image embeddings are random demo embeddings."
)
print(
    "These results validate the multimodal pipeline, "
    "not real satellite-image performance."
)