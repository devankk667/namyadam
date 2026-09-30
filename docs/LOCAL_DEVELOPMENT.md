# Run the app locally

This guide covers a local, non-Docker setup for the FastAPI backend and React
frontend. Commands are provided for **Windows Command Prompt** and macOS/Linux
shells; the Windows commands do not require PowerShell.

## Prerequisites

- Git
- Python 3.10 or newer (3.10 is a known-supported version for this project)
- Node.js 18 or newer, including `npm`
- Enough disk space and memory for the Python dependencies; `backend/requirements.txt`
  includes PyTorch and geospatial libraries

Clone the repository and enter its root directory:

```text
git clone <repository-url>
cd nam_ya_dam
```

Replace `<repository-url>` with the GitHub clone URL. If the cloned folder has a
different name, use that name in subsequent `cd` commands.

## Install dependencies

Create a virtual environment from the repository root.

### Windows Command Prompt

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

The frontend dependencies are installed separately in the frontend section below.

## Data and model availability

The backend looks for data in `data/processed/` and falls back to
`data/demo/thermal_detections_demo.json` when the FIRMS parquet files are not
available. On startup, check `GET /api/health` for the selected `data_source`:

- `firms_predictions`: uses `data/processed/firms_1.1M_predictions.parquet`
- `processed_csv`: uses `data/processed/thermal_detections_processed.csv`
- `demo_json`: uses the bundled demo records
- `unavailable`: none of the configured sources could be loaded

Large FIRMS parquet files and trained model artifacts may not be included in a
GitHub clone. To use the real FIRMS dataset, place the prediction parquet at the
configured path. The default map/table sample size is 500 detections. The backend chooses a new
stratified random sample when it starts and keeps that sample consistent for the
lifetime of that backend process. Restart the backend to get another sample;
browser refreshes do not resample. When the prediction parquet is available,
analytics are aggregated over the full source and retained as compact summaries;
the map/table remain sampled. The dashboard labels these scopes separately. If
the data has 500 or fewer records, all records are used.

When trained Tier 2 artifacts are not present, the API can use its heuristic
fallback. The active mode and any fallback reason are returned by
`GET /api/health` and `GET /api/models`.

## Start the backend

Keep this terminal open while using the app. Activate the virtual environment in
each new terminal before running Python commands.

### Windows Command Prompt

From the repository root:

```bat
.venv\Scripts\activate.bat
cd backend
set "PYTHONPATH=%CD%;%CD%\..\ml\src"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### macOS / Linux

From the repository root:

```bash
source .venv/bin/activate
cd backend
PYTHONPATH=".:../ml/src" python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API is at `http://localhost:8000`; interactive API documentation is at
`http://localhost:8000/docs`.

### Optional: poll live FIRMS detections

Live polling is off by default. It requires a NASA FIRMS map key, an OSM cache
covering the selected area, outbound network access, and the Tier 2 model
artifacts. The key is a secret: set it in the process environment or a local
`.env` file that is not committed. Do not place it in source code.

For Windows Command Prompt, set values in the same terminal before starting the
backend:

```bat
set "LIVE_FIRMS_ENABLED=true"
set "FIRMS_MAP_KEY=your-private-map-key"
set "LIVE_FIRMS_SOURCE=VIIRS_NOAA20_SP"
set "LIVE_FIRMS_BBOX=68,8,97,37"
set "LIVE_FIRMS_POLL_INTERVAL_SECONDS=900"
set "LIVE_FIRMS_LOOKBACK_DAYS=2"
```

For macOS/Linux shell, use `export NAME=value` for those settings. The poller
runs in the FastAPI lifespan, requests a bounded date window, deduplicates
observations into `data/processed/live_firms.sqlite3`, and enriches against the
local `data/raw/osm/*.parquet` cache. It never calls Overpass per detection. A
record missing brightness, confidence, OSM context, or the trained Tier 2 model
is still retained, but its status remains unclassified instead of fabricating
feature values. Check `GET /api/ingestion/status` for poller, OSM, and model
status. Live API access cannot be validated without your key and network access.

## Start the frontend

Open a second terminal in the repository root.

### Windows Command Prompt

```bat
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

### macOS / Linux

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://localhost:5173`. Vite proxies `/api` requests to the backend on
port 8000 by default.

## Check the API and data loading

Use a third terminal, or open the URLs directly in a browser:

```text
http://localhost:8000/api/health
http://localhost:8000/api/ingestion/status
http://localhost:8000/api/detections?page=1&page_size=10
http://localhost:8000/api/analytics/summary
http://localhost:8000/api/analytics/temporal
http://localhost:8000/api/analytics/classification
http://localhost:8000/api/analytics/regions
http://localhost:8000/api/analytics/persistence
http://localhost:8000/api/alerts
http://localhost:8000/api/models
http://localhost:8000/api/models/available
http://localhost:8000/api/fusion/status
```

The detections endpoint should return an `items` array. The health endpoint
reports the loaded source, map sample, and analytics record counts. The Vite-
proxied health check is also available at `http://localhost:5173/api/health`.

## Run the backend and ML tests

From the repository root, with the virtual environment activated:

### Windows Command Prompt

```bat
set "PYTHONPATH=%CD%\backend;%CD%\ml\src"
python -m pytest backend\tests ml\tests
```

### macOS / Linux

```bash
PYTHONPATH="backend:ml/src" python -m pytest backend/tests ml/tests
```

## Optional: train the event-level image-fusion experiment

This requires the fusion feature package and event-level inputs. Expected inputs
include the event tables in `data/processed/` and the spectral features,
embeddings, and spatial folds under `person2_fusion_image_features_v1/data/`.

Install the fusion dependencies from the repository root:

### Windows Command Prompt

```bat
.venv\Scripts\activate.bat
python -m pip install -r ml\requirements-fusion.txt
python ml\scripts\train_image_fusion.py --feature-sets B_tabular_spectral C_tabular_spectral_frozen D_tabular_spectral_cnn --estimators logistic_regression random_forest --trees 100
```

### macOS / Linux

```bash
source .venv/bin/activate
python -m pip install -r ml/requirements-fusion.txt
python ml/scripts/train_image_fusion.py --feature-sets B_tabular_spectral C_tabular_spectral_frozen D_tabular_spectral_cnn --estimators logistic_regression random_forest --trees 100
```

Training outputs are written under `ml/models/fusion_v1/`. The handoff's
`spatial_folds_v1.csv` is authoritative for the evaluation cohort and its labels;
local label differences are reported in `fusion_metrics_v1.json`. The fusion
predictions displayed in the app are spatial out-of-fold validation results for
known image-covered events, not live satellite-image inference. Restart the
backend after training, then check `/api/fusion/status` and the Model Registry
page.

See [Image Fusion Training](IMAGE_FUSION_TRAINING.md) for the feature sets,
evaluation details, and leakage caveats. For an exploratory one-fold spatial
holdout diagnostic, run:

```bat
python ml\scripts\evaluate_spatial_holdout.py --holdout-fold 4
```

In a macOS/Linux shell, use `python ml/scripts/evaluate_spatial_holdout.py --holdout-fold 4`.

The report is written to `ml/models/fusion_v1/spatial_holdout_fold_4_exploratory.json`.
It is explicitly not an independent final-test estimate because the candidate
was selected using the same supplied folds. The event map is keyed by stable
FIRMS `obs_id` values and carries a metadata fingerprint. After downloading and
OSM-enriching source observations, run `ml/scripts/group_events.py` to build the
map before temporal features and predictions. Legacy maps without metadata are
deliberately not joined by row position; rebuild downstream artifacts after
changing from legacy row-number IDs.

## Troubleshooting

- **Backend reports `WinError 10013` or cannot bind the port:** use the documented
  `127.0.0.1` host, check whether another process is using port 8000, and stop
  only the duplicate backend process before retrying.
- **Frontend reports API errors:** check that the backend is running on port 8000
  and that `http://localhost:8000/api/health` responds before refreshing the UI.
- **No real detections appear:** check `data_source` and `data_error` in
  `/api/health`, and verify that the expected data file exists at its configured
  path. The API may be serving demo data instead.
- **Fusion status is disabled:** run the optional training step, restart the
  backend, and confirm that `/api/fusion/status` is enabled. Joining fusion
  predictions to FIRMS detections also requires
  `data/processed/firms_obs_event_map.parquet` to exactly map source observation
  rows to event IDs.
