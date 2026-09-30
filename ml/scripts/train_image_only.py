"""
Train an image-only baseline using Sentinel-2 embeddings.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/processed/thermal_detections_processed.csv"
IMAGE_FILE = "data/processed/sentinel_embeddings_demo.csv"
OUTPUT_DIR = "ml/models/image_only_demo"

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
RANDOM_STATE = 42


# ============================================================
# MODEL
# ============================================================

class ImageOnlyNet(nn.Module):

    def __init__(self, image_dim=512, num_classes=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


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

assert df["id"].is_unique


# ============================================================
# IMAGE FEATURES
# ============================================================

IMAGE_FEATURES = [
    f"embedding_{i}"
    for i in range(512)
]

X_image = df[IMAGE_FEATURES].values.astype(np.float32)


# ============================================================
# TARGET
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["true_class"]
)


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

X_train = X_image[train_idx]
X_test = X_image[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

print(
    f"\nSpatial Group Split complete: "
    f"Train samples={len(train_idx)}, "
    f"Test samples={len(test_idx)}"
)

print(f"Image feature shape: {X_train.shape}")


# ============================================================
# PYTORCH DATASET
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
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

model = ImageOnlyNet(
    image_dim=512,
    num_classes=len(label_encoder.classes_)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nTraining image-only model...")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for bx, by in train_loader:

        bx = bx.to(device)
        by = by.to(device)

        optimizer.zero_grad()

        logits = model(bx)

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

    inputs = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    logits = model(inputs)

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


print("\n--- IMAGE-ONLY MODEL RESULTS ---")

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

metadata = {
    "model_type": "image_only",
    "image_dim": 512,
    "num_classes": len(label_encoder.classes_),
    "classes": list(label_encoder.classes_),
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
    "These results validate the image-only pipeline, "
    "not real satellite-image performance."
)