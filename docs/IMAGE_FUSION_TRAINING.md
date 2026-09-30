# Image-feature fusion training

The extracted handoff in `person2_fusion_image_features_v1/` contains event-level
Sentinel-2 spectral features and image embeddings. It is joined to
`data/processed/firms_thermal_events.parquet` and
`data/processed/firms_event_labels.parquet` using `event_id`.

## Train and compare fusion models

From the repository root, install only the training dependencies, then run:

```bash
python -m pip install -r ml/requirements-fusion.txt
python ml/scripts/train_image_fusion.py
```

The default run compares logistic regression and a 200-tree random forest across
five shared spatial folds for these feature ablations:

- `A_tabular`: event thermal and OSM features
- `B_tabular_spectral`: A plus Sentinel-2 spectral features
- `C_tabular_spectral_frozen`: B plus frozen ImageNet PCA features
- `D_tabular_spectral_cnn`: B plus out-of-fold CNN PCA and image probabilities
- `E_image_only`: image features without thermal/OSM inputs

The script performs feature-column selection before reading the embeddings, so
it does not load the full 512-dimensional `emb_*` vectors. It gives high/medium
(trusted) event labels five times the sample weight of low-confidence rule
labels by default. It reports out-of-fold macro-F1 on trusted events as the
primary metric and reports the score excluding gas flares separately. Flare
performance should be treated cautiously: the handoff reports only eight
trusted examples from two sites.

Artifacts are written under `ml/models/fusion_v1/`, with an independent model
and metadata for each feature set and estimator, plus
`fusion_metrics_v1.json`. This does **not** replace `self_trained_rf3` or change
the active backend model.

Useful smaller runs:

```bash
python ml/scripts/train_image_fusion.py --feature-sets A_tabular D_tabular_spectral_cnn
python ml/scripts/train_image_fusion.py --estimators logistic_regression --trusted-weight 5
```

The supplied fold CSV defines both the evaluation cohort and its versioned
labels/confidence tiers, which must stay aligned with the out-of-fold CNN image
features. Local labels outside that cohort are ignored; local label/confidence
disagreements and fold events missing from the local label table are warned about
and recorded in the metrics report. The run uses the handoff's labels for every
folded event and never guesses or generates folds. Image-bearing events must
still have embedding features.

## Evaluation and leakage

Use the five folds shipped with this package. The `pca_*` and `p_*` values are
out-of-fold CNN features and are only intended for evaluation with these same
spatial folds. Do not train/test-split the merged table randomly, and do not
include `emb_*` or label/trust/fold metadata as model inputs. Frozen PCA and
spectral features are the safer ablation if folds cannot be preserved.

Most labels are low-confidence rules based partly on OSM and FIRMS features.
The all-event metrics primarily measure agreement with those rules; the
trusted-event out-of-fold metrics are the more meaningful but still noisy
comparison. The script selects a best candidate for reporting by trusted
macro-F1 excluding gas flare; it does not claim that selection is independent
of the evaluation set.

## Show held-out fusion results in the app

The training run also writes `fusion_oof_predictions_v1.parquet`. The backend
selects the best tabular-plus-image candidate from the trusted, no-flare OOF
metric and joins those **held-out** predictions to dashboard detections through
`data/processed/firms_obs_event_map.parquet` (`obs_id` → `event_id`). That join
is accepted only when the map exactly covers the prediction parquet's source
rows. Restart the backend after training, then check `/api/fusion/status` and
open the Model Registry and detection detail pages.

The UI calls these CV holdout results. It does not display the final
all-labeled fitted artifact's in-sample predictions as validated predictions,
and it leaves unmatched detections on the existing Tier 2 output. The current
Tier 2 model remains active.

This is an honest historical evaluation overlay, not yet live image inference
for arbitrary new detections. Live use requires the upstream Sentinel-2 scene
retrieval/feature-extraction pipeline to produce the exact feature columns in
the selected artifact, plus evaluation on events independent of training.
Those inputs/model weights are not part of this feature package.
