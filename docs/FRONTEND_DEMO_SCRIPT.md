# Frontend demo script: 3–5 minutes

**Target duration:** about 4 minutes  
**Audience:** reviewers seeing the application for the first time  
**Flow:** Monitoring Console → Detection Detail → Analytics → Alerts → Inference Console → Model Registry

## Before the demo

- Start the backend and frontend, then open `http://localhost:5173`.
- Confirm the header shows the expected API and model status. If possible, check
  `/api/health` before presenting so you know whether this run uses FIRMS,
  processed, or demo data.
- If showing image fusion, make sure the Model Registry has fusion results and
  choose a detection whose detail page actually shows a fusion CV result.
- The FIRMS view is a stratified sample cached at backend startup. Restarting the
  backend can change which detections appear; refreshing the browser does not.
- Do not describe the dashboard as a live satellite feed. It displays the data
  currently loaded by the backend. Fusion results are spatial-CV validation
  outputs for known, image-covered events, not live image inference.

## Spoken script and actions

### 0:00–0:25 — Introduce the problem

**Show:** Monitoring Console and the API/model status in the top bar.

> “This is ThermalGuard, a decision-support dashboard for thermal anomalies.
> It brings FIRMS observations together with nearby OpenStreetMap facility
> context, model classifications, persistence indicators, and risk alerts. The
> status badges show whether the API and trained classifier are available.”

If the header shows demo data or heuristic mode, mention that accurately rather
than presenting the session as a real-data or trained-model deployment.

### 0:25–1:10 — Monitoring Console

**Show:** Summary cards, the map, a class or confidence filter, then select a
map marker or detection-table row.

> “The console gives a quick view of the currently loaded detections. I can filter
> by class, sensor confidence, or region, then inspect a specific point on the
> map. The summary cards and table provide context such as fire radiative power,
> nearest industrial facility, and persistence. Selecting a detection links the
> map and table to the same record.”

### 1:10–1:55 — Detection Detail

**Action:** Click **Inspect** or **Open full analysis** for a detection.

**Show:** The event summary, thermal characteristics, spatial context, model
explanation, and—if available—the fusion panel.

> “The detail view combines the original observation with its model result. Here
> we can inspect FRP, brightness temperatures, sensor confidence, and nearby OSM
> facilities. The current FIRMS/Tier 2 classifier provides a class, probability
> distribution, and explanation. If this event has an image-fusion result, this
> panel shows its held-out spatial-CV prediction and class probabilities. That
> fusion result is validation evidence, not a second live prediction.”

If the fusion panel is absent, say that the selected event has no image-CV
coverage; do not imply that the endpoint failed.

### 1:55–2:25 — Analytics & Persistence

**Action:** Select **Analytics & Persistence** in the sidebar.

**Show:** Temporal trend, classification chart, regional summary, and persistent
sources.

> “Analytics summarizes the loaded detection set: when observations occurred,
> how their classes are distributed, which regions have detections, and which
> sources show recurring activity. These charts describe the current backend
> dataset or sample, rather than a complete global FIRMS archive.”

### 2:25–2:50 — Early Warning Rules

**Action:** Select **Early Warning Rules**, choose a severity filter, and expand
one alert if available.

> “The alert page surfaces rule-engine results, with severity filters and an
> explanation for each alert. I can open the source detection from here. These
> are automated screening rules for review, not confirmed incident reports.”

If no alerts match, show the empty state and note that no loaded records currently
trigger that filter.

### 2:50–3:30 — Inference Console

**Action:** Select **Inference Console**, choose the **Refinery Flare Stack**
preset, and click **Execute inference**.

> “The inference console lets an analyst submit a feature vector without leaving
> the app. I’ll use a preset scenario, then review the predicted class,
> probabilities, confidence, and contributing features. This preset is an
> illustrative input—not a new satellite observation—and inference uses the
> currently active Tier 2 model or the indicated fallback.”

### 3:30–4:15 — Model Registry and close

**Action:** Select **Model Registry**. Show the active model and, if fusion
training artifacts are available, the A–D fusion ablation table.

> “The registry distinguishes the active FIRMS/Tier 2 deployment from the
> separate image-fusion experiment. The ablation table compares the tabular
> baseline with spectral and image-feature variants using the same estimator and
> spatial folds. In this run, frozen-image features improved the trusted
> no-flare score over the tabular baseline; the selected score is still a
> cross-validation result, not an independent final test. Overall, the app
> connects detection review, spatial context, analytics, rule alerts, and model
> evaluation in one frontend.”

If the comparison table is missing, fusion training artifacts have not been
loaded; finish with the active Tier 2 model and its registry metrics instead.

## Features to prioritize if time is short

For a three-minute version, prioritize these five moments:

1. Header health/model status and the monitoring map.
2. One detection’s FRP, OSM context, and Tier 2 explanation.
3. The analytics classification or persistence view.
4. A sample inference in the Inference Console.
5. The Model Registry’s active model and matched-estimator A–D comparison.

Alerts are a good optional stop if the audience is focused on operational
triage. Avoid spending time reading every table row or raw JSON panel.
