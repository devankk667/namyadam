import React, { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import GeospatialMap from '../components/GeospatialMap';
import {
  ArrowLeft,
  MapPin,
  Clock,
  Brain,
  ShieldAlert,
  Flame,
  Building2,
  GitCommitHorizontal,
} from 'lucide-react';
import {
  Panel,
  StatusBadge,
  Tabs,
  Disclosure,
  JsonViewer,
  ProbabilityBar,
  CopyButton,
  LoadingState,
  ErrorState,
  Metric,
  KeyValueRow,
} from '../components/ui';
import { classificationMeta, formatClassLabel } from '../lib/classification';

const FEATURE_GLOSSARY = {
  frp: 'Fire Radiative Power (MW) — energy release rate; higher values indicate more intense combustion.',
  dist_to_industrial: 'Straight-line distance to the nearest known industrial facility (OSM).',
  persistence_score: 'Rolling likelihood that this location is a recurring, not transient, thermal source.',
  brightness: 'Brightness temperature on the mid-infrared (I-4/T4) channel, in Kelvin.',
  bright_t31: 'Brightness temperature on the thermal-infrared (T31) channel, in Kelvin.',
  temp_difference: 'T4 − T31 — the spectral separation used to distinguish flaming combustion from background heat.',
  confidence: 'Sensor-reported detection confidence for the raw thermal pixel.',
  industrial_count_2km: 'Count of mapped industrial facilities within a 2km radius.',
  industrial_count_5km: 'Count of mapped industrial facilities within a 5km radius.',
  detection_count_30d: 'Number of thermal observations at this location in the trailing 30 days.',
};

function featureHelp(key) {
  return FEATURE_GLOSSARY[key] || 'Model-derived input feature contributing to the classification decision.';
}

function humanizeFeature(key) {
  return key.replace(/_/g, ' ');
}

const TABS = [
  { id: 'analysis', label: 'Analysis' },
  { id: 'technical', label: 'Technical Details' },
  { id: 'diagnostics', label: 'Diagnostics' },
  { id: 'history', label: 'History' },
];

export default function DetectionDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis');
  const [evaluatedAt] = useState(() => new Date());

  useEffect(() => {
    async function loadDetail() {
      setLoading(true);
      setError(null);
      try {
        const res = await api.getDetectionDetail(id);
        setData(res);
      } catch (err) {
        setError(err.message || 'Failed to load detection details.');
      } finally {
        setLoading(false);
      }
    }
    loadDetail();
  }, [id]);

  const diagnostics = useMemo(() => {
    if (!data) return null;
    const { record, ml_prediction } = data;
    const isHeuristic = ml_prediction?.model_version?.includes('fallback');
    const warnings = [];

    if (ml_prediction && ml_prediction.confidence < 0.7) {
      warnings.push({
        level: 'warning',
        text: `Model confidence (${(ml_prediction.confidence * 100).toFixed(1)}%) is below the 70% acceptable-certainty threshold.`,
      });
    }
    if (record.confidence < 70) {
      warnings.push({
        level: 'warning',
        text: `Sensor detection confidence (${record.confidence}%) is below the high-certainty threshold used for automated alerting.`,
      });
    }
    if (record.detection_count_30d < 3) {
      warnings.push({
        level: 'info',
        text: `Limited historical window: only ${record.detection_count_30d} observation(s) in the trailing 30 days — persistence score may be unstable.`,
      });
    }
    if (isHeuristic) {
      warnings.push({
        level: 'warning',
        text: 'Inference served by the rule-based heuristic fallback, not a trained model artifact.',
      });
    }
    if (warnings.length === 0) {
      warnings.push({ level: 'success', text: 'No data-quality or confidence warnings triggered for this record.' });
    }

    return { isHeuristic, warnings };
  }, [data]);

  if (loading) {
    return (
      <div className="space-y-4">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1.5 text-ink-muted hover:text-ink-primary text-[11px] font-mono">
          <ArrowLeft className="w-3.5 h-3.5" /> back
        </button>
        <Panel><LoadingState label={`fetching /api/detections/${id}…`} /></Panel>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="space-y-4 max-w-xl">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1.5 text-ink-muted hover:text-ink-primary text-[11px] font-mono">
          <ArrowLeft className="w-3.5 h-3.5" /> back
        </button>
        <ErrorState
          title="Detection record not found"
          message={error || `No record matching ID '${id}'.`}
          action={
            <button
              onClick={() => navigate('/')}
              className="mt-2 px-3 py-1.5 bg-panel2 hover:bg-line border border-line text-[11px] font-mono text-ink-secondary"
            >
              return to console
            </button>
          }
        />
      </div>
    );
  }

  const { record, ml_prediction } = data;
  const meta = classificationMeta(record.true_class);
  const tempDiff = (record.brightness - record.bright_t31).toFixed(1);

  return (
    <div className="space-y-5">
      {/* Nav + record id */}
      <div className="flex items-center justify-between">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1.5 text-ink-muted hover:text-ink-primary text-[11px] font-mono transition-colors">
          <ArrowLeft className="w-3.5 h-3.5" /> back to console
        </button>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-ink-muted">record id:</span>
          <span className="text-[11px] font-mono text-ink-primary font-semibold">{record.id}</span>
          <CopyButton value={record.id} />
        </div>
      </div>

      {/* Header */}
      <div className="border border-line bg-panel">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 px-4 py-3.5 border-b border-line">
          <div className="space-y-1.5 min-w-0">
            <div className="flex items-center gap-2.5 flex-wrap">
              <StatusBadge status={meta.status}>{meta.label}</StatusBadge>
              <span className="text-ink-muted text-[11px] flex items-center gap-1 font-mono">
                <MapPin className="w-3 h-3" />
                {record.region} ({record.latitude.toFixed(3)}, {record.longitude.toFixed(3)})
              </span>
            </div>
            <h1 className="text-[15px] font-semibold text-ink-primary tracking-tight capitalize">
              {formatClassLabel(record.true_class)} event
            </h1>
            <p className="text-[11px] text-ink-muted flex items-center gap-1.5 font-mono">
              <Clock className="w-3 h-3" />
              {record.acq_date} {record.acq_time} UTC · {record.satellite} / {record.instrument}
            </p>
          </div>

          {ml_prediction && (
            <div className="flex divide-x divide-line border border-line shrink-0">
              <Metric label="Model Confidence" value={`${(ml_prediction.confidence * 100).toFixed(1)}%`} tone="accent" caption={`v${ml_prediction.model_version}`} />
              <Metric label="Predicted Class" value={formatClassLabel(ml_prediction.predicted_class)} mono={false} caption="argmax(P)" />
            </div>
          )}
        </div>

        {/* Quick metric strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-y sm:divide-y-0 divide-line">
          <Metric label="FRP" value={`${record.frp} MW`} tone="accent" help={featureHelp('frp')} />
          <Metric label="Sensor Confidence" value={`${record.confidence}%`} help={featureHelp('confidence')} />
          <Metric label="Persistence" value={`${(record.persistence_score * 100).toFixed(0)}%`} tone="warning" help={featureHelp('persistence_score')} />
          <Metric label="Dist. to Industrial" value={`${record.dist_to_industrial} km`} help={featureHelp('dist_to_industrial')} />
        </div>
      </div>

      {/* Tabs */}
      <Tabs tabs={TABS} active={activeTab} onChange={setActiveTab} />

      {activeTab === 'analysis' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2 space-y-5">
            {ml_prediction && (
              <Panel title="Model Explanation" eyebrow="why this result" icon={Brain}>
                <p className="text-[12px] text-ink-secondary leading-relaxed border-l-2 border-accent/50 pl-3">
                  {ml_prediction.explanation.summary}
                </p>

                <div className="mt-4">
                  <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted mb-2">
                    Top Contributing Factors
                  </div>
                  <div className="space-y-2.5">
                    {Object.entries(ml_prediction.explanation.top_contributing_features).map(([fname, weight]) => (
                      <div key={fname} className="flex items-start gap-2">
                        <div className="flex-1">
                          <ProbabilityBar label={humanizeFeature(fname)} value={weight} highlighted />
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-[11px] text-ink-muted mt-2">
                    {Object.keys(ml_prediction.explanation.top_contributing_features)
                      .map((k) => `${humanizeFeature(k)} — ${featureHelp(k)}`)
                      .join('  ·  ')}
                  </p>
                </div>

                <div className="mt-4 pt-4 border-t border-line">
                  <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted mb-2">
                    Class Probability Distribution
                  </div>
                  <div className="space-y-2">
                    {Object.entries(ml_prediction.class_probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([cls, prob]) => (
                        <ProbabilityBar
                          key={cls}
                          label={formatClassLabel(cls)}
                          value={prob}
                          highlighted={cls === ml_prediction.predicted_class}
                        />
                      ))}
                  </div>
                </div>
              </Panel>
            )}

            <Panel title="Event Location" eyebrow="geospatial" icon={MapPin} noPadding>
              <GeospatialMap detections={[record]} selectedDetection={record} height="380px" />
            </Panel>
          </div>

          <div className="space-y-5">
            <Panel title="Thermal Characteristics" eyebrow="physical" icon={Flame}>
              <KeyValueRow label="Fire Radiative Power" value={`${record.frp} MW`} mono tone="accent" help={featureHelp('frp')} />
              <KeyValueRow label="Brightness (I-4/T4)" value={`${record.brightness} K`} mono />
              <KeyValueRow label="Brightness (T31)" value={`${record.bright_t31} K`} mono />
              <KeyValueRow label="Spectral difference (T4−T31)" value={`${tempDiff} K`} mono help={featureHelp('temp_difference')} />
              <KeyValueRow label="Day / Night" value={record.daynight === 'D' ? 'Day' : 'Night'} />
            </Panel>

            <Panel title="Spatial Context" eyebrow="OSM enrichment" icon={Building2}>
              <KeyValueRow label="Nearest facility" value={`${record.dist_to_industrial} km`} mono help={featureHelp('dist_to_industrial')} />
              <KeyValueRow label="Facility type" value={record.nearest_facility_type} mono tone="accent" />
              <KeyValueRow label="Facilities (2km / 5km)" value={`${record.industrial_count_2km} / ${record.industrial_count_5km}`} mono />
              <KeyValueRow label="Persistence score" value={`${(record.persistence_score * 100).toFixed(1)}%`} mono tone="warning" help={featureHelp('persistence_score')} />
              <KeyValueRow label="30-day observations" value={record.detection_count_30d} mono help={featureHelp('detection_count_30d')} />
            </Panel>
          </div>
        </div>
      )}

      {activeTab === 'technical' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="space-y-5">
            <Panel title="Input Parameters" eyebrow="model input vector">
              <KeyValueRow label="brightness" value={record.brightness} mono />
              <KeyValueRow label="bright_t31" value={record.bright_t31} mono />
              <KeyValueRow label="frp" value={record.frp} mono />
              <KeyValueRow label="confidence" value={record.confidence} mono />
              <KeyValueRow label="dist_to_industrial" value={record.dist_to_industrial} mono />
              <KeyValueRow label="industrial_count_2km" value={record.industrial_count_2km} mono />
              <KeyValueRow label="industrial_count_5km" value={record.industrial_count_5km} mono />
              <KeyValueRow label="nearest_facility_type" value={record.nearest_facility_type} mono />
              <KeyValueRow label="persistence_score" value={record.persistence_score} mono />
              <KeyValueRow label="detection_count_30d" value={record.detection_count_30d} mono />
              <KeyValueRow label="daynight" value={record.daynight} mono />
            </Panel>

            <Panel title="Model / Version Metadata" eyebrow="configuration">
              <KeyValueRow label="model_version" value={ml_prediction?.model_version || 'n/a'} mono />
              <KeyValueRow
                label="inference mode"
                value={ml_prediction?.model_version?.includes('fallback') ? 'heuristic fallback' : 'trained artifact'}
                mono
                tone={ml_prediction?.model_version?.includes('fallback') ? 'warning' : 'success'}
              />
              <KeyValueRow label="supported classes" value={Object.keys(ml_prediction?.class_probabilities || {}).length} mono />
              <KeyValueRow label="satellite / instrument" value={`${record.satellite} / ${record.instrument}`} mono />
            </Panel>
          </div>

          <div className="space-y-5">
            <Disclosure title="detection.json" eyebrow="raw record" mono defaultOpen>
              <JsonViewer data={record} title="record" />
            </Disclosure>
            <Disclosure title="prediction.json" eyebrow="raw model output" mono>
              <JsonViewer data={ml_prediction} title="ml_prediction" />
            </Disclosure>
          </div>
        </div>
      )}

      {activeTab === 'diagnostics' && diagnostics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <Panel title="Warnings & Data Quality" eyebrow="derived checks" icon={ShieldAlert}>
            <div className="space-y-2">
              {diagnostics.warnings.map((w, i) => (
                <div key={i} className="flex items-start gap-2 py-1.5 border-b border-line/70 last:border-b-0">
                  <StatusBadge status={w.level === 'success' ? 'success' : w.level === 'warning' ? 'warning' : 'info'} size="xs">
                    {w.level}
                  </StatusBadge>
                  <span className="text-[12px] text-ink-secondary">{w.text}</span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Inference Performance" eyebrow="runtime metadata">
            <KeyValueRow label="Model mode" value={diagnostics.isHeuristic ? 'HEURISTIC' : 'TRAINED ARTIFACT'} mono tone={diagnostics.isHeuristic ? 'warning' : 'success'} />
            <KeyValueRow label="Evaluated at (client)" value={evaluatedAt.toISOString()} mono />
            <KeyValueRow label="Endpoint" value={`GET /api/detections/${record.id}`} mono />
            <KeyValueRow label="Response fields" value={Object.keys(record).length + (ml_prediction ? Object.keys(ml_prediction).length : 0)} mono />
          </Panel>
        </div>
      )}

      {activeTab === 'history' && (
        <Panel title="Event Trace" eyebrow="observation timeline" icon={GitCommitHorizontal}>
          <div className="space-y-0">
            <div className="flex gap-3 pb-4 relative">
              <div className="flex flex-col items-center">
                <span className="w-2 h-2 rounded-full bg-status-info mt-1" />
                <span className="w-px flex-1 bg-line" />
              </div>
              <div className="pb-2">
                <div className="text-[11px] font-mono text-ink-muted">{record.acq_date} {record.acq_time} UTC</div>
                <div className="text-[12px] text-ink-primary font-medium">Satellite acquisition</div>
                <p className="text-[11px] text-ink-secondary mt-0.5">
                  Thermal pixel captured by {record.satellite} ({record.instrument}) and ingested as record {record.id}.
                </p>
              </div>
            </div>

            {record.persistence_score >= 0.5 && (
              <div className="flex gap-3 pb-4">
                <div className="flex flex-col items-center">
                  <span className="w-2 h-2 rounded-full bg-status-warning mt-1" />
                  <span className="w-px flex-1 bg-line" />
                </div>
                <div className="pb-2">
                  <div className="text-[11px] font-mono text-ink-muted">rolling 30-day window</div>
                  <div className="text-[12px] text-ink-primary font-medium">Recurring pattern detected</div>
                  <p className="text-[11px] text-ink-secondary mt-0.5">
                    {record.detection_count_30d} observation(s) at this location in the trailing 30 days, yielding a
                    persistence score of {(record.persistence_score * 100).toFixed(1)}%.
                  </p>
                </div>
              </div>
            )}

            <div className="flex gap-3">
              <div className="flex flex-col items-center">
                <span className="w-2 h-2 rounded-full bg-ink-muted mt-1" />
              </div>
              <div>
                <div className="text-[12px] text-ink-secondary">
                  Region-level persistence trends for <strong className="text-ink-primary">{record.region}</strong> are
                  available in{' '}
                  <button onClick={() => navigate('/analytics')} className="text-accent hover:underline">
                    Analytics & Persistence
                  </button>.
                </div>
              </div>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
