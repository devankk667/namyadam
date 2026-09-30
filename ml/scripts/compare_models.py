"""
Compare saved tabular, image-only, and multimodal model results.
"""

import json
import os
import pandas as pd


MODELS = {
    "Tabular-only": "ml/models/best_model/metadata.json",
    "Image-only": "ml/models/image_only_demo/metadata.json",
    "Multimodal": "ml/models/multimodal_demo/metadata.json"
}


results = []

for model_name, metadata_file in MODELS.items():

    if not os.path.exists(metadata_file):
        print(f"Missing: {metadata_file}")
        continue

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Existing tabular model stores metrics inside "metrics"
    if "metrics" in metadata:
        metrics = metadata["metrics"]

        macro_f1 = metrics["macro_f1"]
        precision = metrics["precision_macro"]
        recall = metrics["recall_macro"]
        roc_auc = metrics["roc_auc_ovr"]

    # New image-only / multimodal models store metrics directly
    else:
        macro_f1 = metadata["macro_f1"]
        precision = metadata["macro_precision"]
        recall = metadata["macro_recall"]
        roc_auc = metadata["roc_auc"]

    results.append({
        "Model": model_name,
        "Macro F1": macro_f1,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc
    })


results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("THERMOGUARD MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("=" * 70)