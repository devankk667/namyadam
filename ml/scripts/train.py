"""
Training script for evaluating Logistic Regression, Random Forest, XGBoost, and PyTorch DL models
using a spatial leakage-safe group split.
Saves model artifacts, scaler, and metadata to ml/models/best_model.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from preprocessing import prepare_features, TARGET_CLASSES
from pytorch_model import MultimodalThermalNet

def train_and_evaluate():
    data_path = "data/processed/thermal_detections_processed.csv"
    if not os.path.exists(data_path):
        from generate_demo_data import main as gen_main
        gen_main()

    df = pd.read_csv(data_path)

    # Target encoding
    class_to_idx = {cls_name: i for i, cls_name in enumerate(TARGET_CLASSES)}
    y = df["true_class"].map(class_to_idx).values
    groups = df["spatial_cluster_id"].values

    # Feature preparation & scaling
    X_scaled, scaler, feature_names = prepare_features(df, fit=True)

    # Spatial Leakage-Safe Split (GroupShuffleSplit on spatial_cluster_id)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X_scaled, y, groups=groups))

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Spatial Group Split complete: Train samples={len(train_idx)}, Test samples={len(test_idx)}")
    print(f"Unseen test spatial groups: {np.unique(groups[test_idx])}")

    models = {}
    eval_results = {}

    # 1. Logistic Regression Baseline
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)
    models["logistic_regression"] = log_reg
    eval_results["logistic_regression"] = {
        "macro_f1": float(f1_score(y_test, y_pred_lr, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred_lr, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred_lr, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba_lr, multi_class="ovr")),
    }

    # 2. Random Forest Baseline
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    models["random_forest"] = rf
    eval_results["random_forest"] = {
        "macro_f1": float(f1_score(y_test, y_pred_rf, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred_rf, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred_rf, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba_rf, multi_class="ovr")),
    }

    # 3. XGBoost Model
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)
    models["xgboost"] = xgb
    eval_results["xgboost"] = {
        "macro_f1": float(f1_score(y_test, y_pred_xgb, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred_xgb, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred_xgb, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba_xgb, multi_class="ovr")),
    }

    # 4. PyTorch Multimodal Net
    device = torch.device("cpu")
    pt_net = MultimodalThermalNet(input_dim=X_train.shape[1], num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pt_net.parameters(), lr=0.005)

    dataset_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

    pt_net.train()
    for epoch in range(25):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = pt_net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    pt_net.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        logits = pt_net(test_inputs)
        y_proba_pt = torch.softmax(logits, dim=1).numpy()
        y_pred_pt = np.argmax(y_proba_pt, axis=1)

    eval_results["pytorch_multimodal"] = {
        "macro_f1": float(f1_score(y_test, y_pred_pt, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred_pt, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred_pt, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba_pt, multi_class="ovr")),
    }

    print("\n--- MODEL COMPARISON BENCHMARK ---")
    for mname, metrics in eval_results.items():
        print(f"Model: {mname:<20} | Macro F1: {metrics['macro_f1']:.4f} | ROC-AUC: {metrics['roc_auc_ovr']:.4f}")

    # Select best model (XGBoost / Random Forest)
    best_model_name = max(eval_results, key=lambda k: eval_results[k]["macro_f1"])
    print(f"\nBest performing model: {best_model_name}")

    best_model = models[best_model_name]

    # Save artifacts to ml/models/best_model
    output_dir = "ml/models/best_model"
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    # Feature importance extraction (for Tree models)
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_names, best_model.feature_importances_.astype(float)))
    else:
        importances = {fn: 1.0 / len(feature_names) for fn in feature_names}

    cm = confusion_matrix(y_test, xgb.predict(X_test) if best_model_name == "xgboost" else y_pred_rf).tolist()

    metadata = {
        "model_name": best_model_name,
        "version": "1.0.0",
        "supported_classes": TARGET_CLASSES,
        "feature_names": feature_names,
        "metrics": eval_results[best_model_name],
        "all_model_benchmarks": eval_results,
        "feature_importances": importances,
        "confusion_matrix": cm,
        "split_method": "Facility-Aware Spatial Group Split",
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts successfully exported to {output_dir}/")

if __name__ == "__main__":
    train_and_evaluate()
