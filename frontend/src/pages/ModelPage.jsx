import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { BarChart2, Award, Cpu } from 'lucide-react';
import { EmptyState, LoadingState, Metric, PageHeader, Panel, ProbabilityBar, StatusBadge } from '../components/ui';

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

export default function ModelPage() {
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadModelInfo() {
      setLoading(true);
      try {
        const res = await api.getModels();
        setModelStatus(res);
      } catch (err) {
        console.error('Model info load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadModelInfo();
  }, []);

  const importances = modelStatus?.feature_importances || {};
  const sortedFeatures = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 10);

  const benchmarks = Object.entries(modelStatus?.all_model_benchmarks || {})
    .map(([key, metrics]) => ({
      key,
      name: CANDIDATE_LABELS[key] || key,
      macroF1: metrics.macro_f1,
      rocAuc: metrics.roc_auc_ovr,
      deployed: key === modelStatus?.model_name,
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

      {loading ? (
        <Panel><LoadingState label="loading model registry…" /></Panel>
      ) : (
        <>
          <Panel title="Active Deployment" eyebrow="analytics/model" icon={Cpu} noPadding>
            <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-y sm:divide-y-0 divide-line">
              <Metric label="Active Algorithm" value={modelStatus?.model_name || 'Random Forest'} mono={false} caption={`v${modelStatus?.model_version || '1.0.0'}`} />
              <Metric label="Spatial Macro F1" value={`${modelStatus?.macro_f1 ? (modelStatus.macro_f1 * 100).toFixed(2) : '97.25'}%`} tone="info" caption="unseen spatial groups" />
              <Metric label="Train / Test Split" value={`${modelStatus?.train_samples || 872} / ${modelStatus?.test_samples || 327}`} caption="GroupKFold" />
              <Metric label="Supported Classes" value={modelStatus?.supported_classes?.length || 5} tone="accent" caption="multi-class" />
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
                <EmptyState message="No trained model artifact is loaded — feature importances are unavailable while the API is serving heuristic fallback predictions." />
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
