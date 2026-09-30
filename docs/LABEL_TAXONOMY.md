# ThermalGuard Label Taxonomy (Frozen v1.0)

Labels attach to **thermal events** (spatiotemporal groups), never to raw rows.
OSM proximity is evidence, never ground truth. `unknown` is a real state,
not a failure — most events SHOULD be unknown until evidence says otherwise.

## 7A. Classes

| Label | Meaning | Needs (besides OSM) |
|---|---|---|
| `industrial_fire` | Thermal event at/near an industrial facility | strong thermal signature + abnormal temporal pattern; capped at medium (no independent incident source yet) |
| `gas_flare` | Recurring flare-type source | flare facility ≤ 1 km + recurrence (≥5 distinct days, ≥30 d span) |
| `mining_activity` | Thermal activity at quarry/mine | quarry/mine context ≤ 2 km + repetition (≥3 distinct days) |
| `agricultural_burning` | Crop/residue burning | burning season (Oct–Nov, Mar–Apr) + rural (>5 km from industry) + low FRP + short duration |
| `wildfire` | Vegetation/forest fire | **PENDING — needs vegetation/forest mask; zero wildfire labels until then** |
| `unknown` | Insufficient evidence | default |

Target for first supervised model: ~20k trustworthy labeled observations
(high + medium confidence), rest stays UNLABELED — never 1M noisy pseudo-labels.

## 7B. Evidence rules (conjunctions, not single thresholds)

- NEVER `distance < 2 km → industrial_fire`. Every assigned label needs
  facility context AND thermal AND temporal evidence (see `label_events.py` config).
- `nearest industry = 80 km` while OSM tiles are missing means "no data",
  not "rural" — final labels only after the full OSM cache + re-enrichment.

## 7C. Label columns

Every labeled event carries: `label`, `label_confidence` (high/medium/low),
`label_source` (rule name / verified record / manual review), `label_reason`
(human-readable evidence summary). Confidence guide:

| Situation | Confidence |
|---|---|
| verified incident / known flare facility + recurrence | high |
| quarry + persistent hotspot; industrial + strong + recurrent | medium |
| industrial proximity only; seasonal + rural only | low |
| anything else | unknown (no confidence) |
