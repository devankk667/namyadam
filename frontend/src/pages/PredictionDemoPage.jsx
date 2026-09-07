import React, { useState } from 'react';
import { api } from '../services/api';
import { Play, Sliders, Brain } from 'lucide-react';
import { Disclosure, ErrorState, JsonViewer, Metric, PageHeader, Panel, ProbabilityBar, StatusBadge } from '../components/ui';
import { formatClassLabel } from '../lib/classification';

const SAMPLE_PRESETS = [
  {
    name: 'Refinery Flare Stack',
    data: {
      brightness: 368.5, bright_t31: 295.2, frp: 35.0, confidence: 96.0,
      dist_to_industrial: 0.12, industrial_count_2km: 6, industrial_count_5km: 14,
      nearest_facility_type: 'refinery', persistence_score: 0.92, detection_count_30d: 28, daynight: 'N'
    }
  },
  {
    name: 'Chemical Fire',
    data: {
      brightness: 395.0, bright_t31: 315.0, frp: 180.0, confidence: 99.0,
      dist_to_industrial: 0.25, industrial_count_2km: 4, industrial_count_5km: 10,
      nearest_facility_type: 'chemical', persistence_score: 0.35, detection_count_30d: 4, daynight: 'D'
    }
  },
  {
    name: 'Boreal Wildfire',
    data: {
      brightness: 375.0, bright_t31: 305.0, frp: 120.0, confidence: 92.0,
      dist_to_industrial: 18.5, industrial_count_2km: 0, industrial_count_5km: 0,
      nearest_facility_type: 'none', persistence_score: 0.10, detection_count_30d: 2, daynight: 'D'
    }
  },
  {
    name: 'Crop Stubble Burning',
    data: {
      brightness: 332.0, bright_t31: 292.0, frp: 14.5, confidence: 78.0,
      dist_to_industrial: 12.0, industrial_count_2km: 0, industrial_count_5km: 1,
      nearest_facility_type: 'none', persistence_score: 0.05, detection_count_30d: 1, daynight: 'D'
    }
  }
];

const FIELDS = [
  { name: 'brightness', label: 'Brightness Temp (K)', step: '0.1' },
  { name: 'bright_t31', label: 'Bright T31 Temp (K)', step: '0.1' },
  { name: 'frp', label: 'FRP (MW)', step: '0.1' },
  { name: 'confidence', label: 'Sensor Confidence (%)', step: '1' },
  { name: 'dist_to_industrial', label: 'Dist. to Industrial (km)', step: '0.01' },
  { name: 'persistence_score', label: 'Persistence Score (0-1)', step: '0.01', min: 0, max: 1 },
  { name: 'detection_count_30d', label: 'Trailing 30d Detections', step: '1' },
];

const FACILITY_TYPES = ['refinery', 'flare_stack', 'chemical', 'gas_terminal', 'power_plant', 'steel_works', 'factory', 'none'];

export default function PredictionDemoPage() {
  const [formData, setFormData] = useState(SAMPLE_PRESETS[0].data);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activePreset, setActivePreset] = useState(0);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: e.target.type === 'number' ? parseFloat(value) || 0 : value
    }));
    setActivePreset(null);
  };

  const handleApplyPreset = (presetData, idx) => {
    setFormData(presetData);
    setResult(null);
    setActivePreset(idx);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict(formData);
      setResult(res);
    } catch (err) {
      setError(err.message || 'Error running model inference');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Inference Console"
        description="Submit a feature vector to the trained classifier and inspect the full explanation, not just the label."
      />

      <Panel title="Preset Scenarios" eyebrow="quick load" noPadding bodyClassName="p-3">
        <div className="flex flex-wrap gap-1.5 p-3">
          {SAMPLE_PRESETS.map((p, idx) => (
            <button
              key={idx}
              onClick={() => handleApplyPreset(p.data, idx)}
              className={`px-2.5 py-1.5 border text-[11px] font-medium transition-colors ${
                activePreset === idx
                  ? 'bg-accent/10 border-accent/40 text-accent'
                  : 'bg-panel2 border-line text-ink-secondary hover:border-line2'
              }`}
            >
              {p.name}
            </button>
          ))}
        </div>
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
        <Panel title="Feature Input Vector" eyebrow="request payload" icon={Sliders}>
          <form onSubmit={handlePredict} className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {FIELDS.map((f) => (
                <div key={f.name}>
                  <label className="block text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">{f.label}</label>
                  <input
                    type="number"
                    step={f.step}
                    min={f.min}
                    max={f.max}
                    name={f.name}
                    value={formData[f.name]}
                    onChange={handleInputChange}
                    className="w-full bg-panel2 border border-line text-ink-primary font-mono text-[12px] px-2.5 py-1.5 focus:outline-none focus:border-accent"
                    required
                  />
                </div>
              ))}

              <div>
                <label className="block text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">Facility Type</label>
                <select
                  name="nearest_facility_type"
                  value={formData.nearest_facility_type}
                  onChange={handleInputChange}
                  className="w-full bg-panel2 border border-line text-ink-primary font-mono text-[12px] px-2.5 py-1.5 focus:outline-none focus:border-accent"
                >
                  {FACILITY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">Day / Night</label>
                <select
                  name="daynight"
                  value={formData.daynight}
                  onChange={handleInputChange}
                  className="w-full bg-panel2 border border-line text-ink-primary font-mono text-[12px] px-2.5 py-1.5 focus:outline-none focus:border-accent"
                >
                  <option value="D">Day</option>
                  <option value="N">Night</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-2.5 bg-accent hover:bg-accent-strong text-[#160d05] font-semibold text-[12px] uppercase tracking-wide transition-colors disabled:opacity-60"
            >
              {loading ? (
                <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-current" />
              )}
              Execute inference
            </button>

            <Disclosure title="request.json" eyebrow="preview" mono>
              <JsonViewer data={formData} title="POST /api/predictions" />
            </Disclosure>
          </form>
        </Panel>

        <Panel title="Inference Output" eyebrow="response" icon={Brain}>
          {error && <ErrorState title="Inference failed" message={error} />}

          {!result && !error && (
            <div className="flex flex-col items-center justify-center h-64 text-center text-ink-muted text-[12px] gap-2">
              <Brain className="w-8 h-8 stroke-[1.25]" />
              <p>Execute inference or select a preset scenario to evaluate.</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="flex divide-x divide-line border border-line">
                <Metric label="Predicted Class" value={formatClassLabel(result.predicted_class)} mono={false} tone="accent" />
                <Metric label="Confidence" value={`${(result.confidence * 100).toFixed(1)}%`} tone="success" />
              </div>

              <div>
                <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted mb-1.5">Explanation</div>
                <p className="text-[12px] text-ink-secondary leading-relaxed border-l-2 border-accent/50 pl-3">
                  {result.explanation.summary}
                </p>
              </div>

              <div>
                <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted mb-1.5">Top Contributing Features</div>
                <div className="space-y-2">
                  {Object.entries(result.explanation.top_contributing_features).map(([k, v]) => (
                    <ProbabilityBar key={k} label={k.replace(/_/g, ' ')} value={v} highlighted />
                  ))}
                </div>
              </div>

              <div>
                <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted mb-1.5">Class Probability Distribution</div>
                <div className="space-y-2">
                  {Object.entries(result.class_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, p]) => (
                      <ProbabilityBar key={cls} label={formatClassLabel(cls)} value={p} highlighted={cls === result.predicted_class} />
                    ))}
                </div>
              </div>

              <div className="flex items-center gap-2 text-[11px] font-mono text-ink-muted pt-1">
                <StatusBadge status="neutral" size="xs">model v{result.model_version}</StatusBadge>
              </div>

              <Disclosure title="response.json" eyebrow="raw output" mono>
                <JsonViewer data={result} title="200 OK" />
              </Disclosure>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
