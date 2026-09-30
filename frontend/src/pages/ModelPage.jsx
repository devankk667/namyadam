import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { BarChart2, Award, Cpu } from 'lucide-react';
import { EmptyState, ErrorState, LoadingState, Metric, PageHeader, Panel, ProbabilityBar, StatusBadge } from '../components/ui';

const CANDIDATE_LABELS = {
  random_forest: 'Random Forest Classifier',
  xgboost: 'XGBoost Gradient Boosted Trees',
  pytorch_multimodal: 'PyTorch Multimodal Neural Net',
  logistic_regression: 'Logistic Regression Baseline',
};

function candidateStatus(key, isDeployed) {
  if (isDeployed) return 'DEPLOYED';
  if (key === 'pytorch_multimodal') return 'EVALUATED (DL)';
  if (key === 'logistic_regression') return 'BASELINE';
  return 'EVALUATED';
}

const FUSION_ABLATIONS = [
  { key: 'A_tabular', label: 'Tabular baseline' },
  { key: 'B_tabular_spectral', label: 'Tabular + spectral' },
  { key: 'C_tabular_spectral_frozen', label: 'Tabular + spectral + frozen image' },
  { key: 'D_tabular_spectral_cnn', label: 'Tabular + spectral + CNN' },
];

function formatF1(value) {
  return value == null || !Number.isFinite(Number(value))
    ? '—'
    : `${(Number(value) * 100).toFixed(1)}%`;
}

export default function ModelPage() {
  const [modelStatus, setModelStatus] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [fusionStatus, setFusionStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const loadModelInfo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [status, available] = await Promise.all([api.getModels(), api.getAvailableModels()]);
      setModelStatus(status);
      setAvailableModels(Array.isArray(available.available_models) ? available.available_models : []);
      try {
        setFusionStatus(await api.getFusionStatus());
      } catch (fusionError) {
        setFusionStatus({ enabled: false, reason: fusionError.message || 'Fusion status unavailable.' });
      }
    } catch (err) {
      setError(err.message || 'Model registry could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadModelInfo();
  }, [loadModelInfo]);

  async function handleSwitchModel(event) {
    const modelName = event.target.value;
    if (!modelName || modelName === modelStatus?.active_model_key) return;
    setSwitching(true);
    setError(null);
    setNotice(null);
    try {
      const result = await api.switchModel(modelName);
      setNotice(`Active model: ${result.description}`);
      await loadModelInfo();
    } catch (err) {
      setError(err.message || 'Could not switch model.');
    } finally {
      setSwitching(false);
    }
  }

  const importances = modelStatus?.feature_importances || {};
  const sortedFeatures = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const fusionAblations = new Map(
    (fusionStatus?.ablation_comparison || [])
      .filter((candidate) => candidate.estimator === fusionStatus?.estimator)
      .map((candidate) => [candidate.feature_set, candidate]),
  );

  const benchmarks = Object.entries(modelStatus?.all_model_benchmarks || {})
    .map(([key, metrics]) => ({
      key,
      name: CANDIDATE_LABELS[key] || key,
      macroF1: Number(metrics?.macro_f1 || 0),
      rocAuc: Number(metrics?.roc_auc_ovr || 0),
      deployed: key === modelStatus?.active_model_key,
    }))
    .sort((a, b) => b.macroF1 - a.macroF1);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Model Registry"
        description="Evaluated on spatial group splits to prevent data leakage across unseen global industrial facilities."
        badges={
          <StatusBadge status={modelStatus?.model_loaded ? 'success' : 'warning'} dot size="xs">
            {modelStatus?.model_loaded ? 'artifact loaded' : 'heuristic fallback'}
          </StatusBadge>
        }
      />

      {notice && <div className="border border-status-success/30 bg-status-success/5 px-3 py-2 text-[11px] text-status-success">{notice}</div>}
      {error && <ErrorState title="Model registry error" message={error} action={<button onClick={loadModelInfo} className="mt-2 px-3 py-1.5 border border-line text-[11px]">Retry</button>} />}

      {fusionStatus && (
        <Panel title="OSM + FIRMS + Satellite Fusion" eyebrow="spatial holdout results" icon={Cpu}>
          {fusionStatus.enabled ? (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge status="info" size="xs">spatial CV holdout</StatusBadge>
                <span className="text-[11px] font-mono text-ink-muted">
                  {fusionStatus.feature_set} / {fusionStatus.estimator}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 divide-x divide-line border border-line">
                <Metric label="Trusted OOF Macro F1" value={`${(Number(fusionStatus.trusted_macro_f1 || 0) * 100).toFixed(1)}%`} tone="accent" />
                <Metric label="Trusted F1 · no flare" value={`${(Number(fusionStatus.trusted_macro_f1_no_flare || 0) * 100).toFixed(1)}%`} tone="info" />
                <Metric label="Image-covered events" value={fusionStatus.image_covered_events ?? 0} />
              </div>
              <p className="text-[11px] text-ink-muted">
                Dashboard fusion labels are out-of-fold validation predictions for known, imaged events—not live
                inference. Unmatched events continue to show the current FIRMS/Tier 2 result only.
              </p>
              <div className="border-t border-line pt-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted">
                    Feature ablation · {fusionStatus.estimator}
                  </div>
                  <span className="text-[10px] text-ink-muted">Same trusted events and spatial folds</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-[11px] whitespace-nowrap">
                    <thead className="border-b border-line bg-panel2 font-mono text-[9px] uppercase tracking-wide text-ink-muted">
                      <tr>
                        <th className="px-2 py-1.5 font-medium">Feature set</th>
                        <th className="px-2 py-1.5 font-medium">Trusted macro F1</th>
                        <th className="px-2 py-1.5 font-medium">No-flare macro F1</th>
                        <th className="px-2 py-1.5 font-medium">Trusted n</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {FUSION_ABLATIONS.map(({ key, label }) => {
                        const result = fusionAblations.get(key);
                        const isSelected = key === fusionStatus.feature_set;
                        return (
                          <tr key={key} className={isSelected ? 'bg-accent/[0.05]' : ''}>
                            <td className="px-2 py-1.5 text-ink-secondary">
                              {label}
                              {isSelected && <span className="ml-2 text-accent">selected</span>}
                            </td>
                            <td className="px-2 py-1.5 font-mono text-ink-primary">{formatF1(result?.trusted_macro_f1)}</td>
                            <td className="px-2 py-1.5 font-mono text-status-success">{formatF1(result?.trusted_macro_f1_no_flare)}</td>
                            <td className="px-2 py-1.5 font-mono text-ink-muted">{result?.trusted_events ?? '—'}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-[10px] text-ink-muted">
                  Missing values mean that feature set was not trained for this estimator. The selected result is chosen by
                  trusted macro F1 excluding gas flare; it is not an independent final test score.
                </p>
              </div>
            </div>
          ) : (
            <EmptyState message={fusionStatus.reason || 'Fusion validation predictions are not available yet.'} />
          )}
        </Panel>
      )}

      {loading ? (
        <Panel><LoadingState label="loading model registry…" /></Panel>
      ) : modelStatus && (
        <>
          <Panel title="Active Deployment" eyebrow="analytics/model" icon={Cpu} noPadding>
            {availableModels.length > 0 && <div className="flex items-center gap-2 border-b border-line px-3 py-2 text-[11px]">
              <label htmlFor="active-model" className="text-ink-muted">Active artifact</label>
              <select id="active-model" value={availableModels.find((model) => model.active)?.name || ''} onChange={handleSwitchModel} disabled={switching} className="bg-panel2 border border-line px-2 py-1 text-ink-primary disabled:opacity-60">
                {availableModels.map((model) => <option key={model.name} value={model.name}>{model.description} (F1 {Number(model.macro_f1 || 0).toFixed(3)}){model.compatible_with_prediction_form ? '' : ' — incomplete form features'}</option>)}
              </select>
              {switching && <span className="text-ink-muted">switching…</span>}
            </div>}
            <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-y sm:divide-y-0 divide-line">
              <Metric label="Active Algorithm" value={modelStatus?.model_name || 'Unavailable'} mono={false} caption={`v${modelStatus?.model_version || '—'}`} />
              <Metric label="Spatial Macro F1" value={modelStatus?.model_loaded ? `${(modelStatus.macro_f1 * 100).toFixed(2)}%` : '—'} tone="info" caption={modelStatus?.split_method || 'not available'} />
              <Metric label="Training observations" value={modelStatus?.train_samples ?? 0} caption={modelStatus?.split_method || 'not available'} />
              <Metric label="Supported Classes" value={modelStatus?.supported_classes?.length ?? 0} tone="accent" caption="multi-class" />
            </div>
          </Panel>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <Panel title="Top Predictive Feature Importances" eyebrow="explainability" icon={BarChart2}>
              {sortedFeatures.length > 0 ? (
                <div className="space-y-3">
                  {sortedFeatures.map(([fname, imp]) => (
                    <ProbabilityBar key={fname} label={fname} value={imp} tone="accent" />
                  ))}
                </div>
              ) : (
                <EmptyState message={modelStatus?.fallback_reason || 'No trained model artifact is loaded — feature importances are unavailable while the API is serving heuristic fallback predictions.'} />
              )}
            </Panel>

            <Panel title="Model Comparison Benchmarks" eyebrow="offline evaluation" icon={Award} noPadding subtitle={modelStatus?.split_method}>
              {benchmarks.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-[12px] whitespace-nowrap">
                    <thead className="bg-panel2 text-ink-muted font-mono text-[10px] uppercase tracking-wide border-b border-line">
                      <tr>
                        <th className="px-3 py-2 font-medium">Candidate</th>
                        <th className="px-3 py-2 font-medium">Macro F1</th>
                        <th className="px-3 py-2 font-medium">ROC-AUC</th>
                        <th className="px-3 py-2 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {benchmarks.map((b) => (
                        <tr key={b.key} className={b.deployed ? 'bg-accent/[0.05]' : ''}>
                          <td className={`px-3 py-2 ${b.deployed ? 'font-semibold text-ink-primary' : 'font-medium text-ink-secondary'}`}>{b.name}</td>
                          <td className="px-3 py-2 font-mono text-status-success">{b.macroF1.toFixed(4)}</td>
                          <td className="px-3 py-2 font-mono text-status-warning">{b.rocAuc.toFixed(4)}</td>
                          <td className="px-3 py-2">
                            <StatusBadge status={b.deployed ? 'accent' : 'neutral'} size="xs">{candidateStatus(b.key, b.deployed)}</StatusBadge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-4">
                  <EmptyState message="No offline evaluation benchmarks available — train candidate models via ml/scripts/train.py to populate this comparison." />
                </div>
              )}
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
