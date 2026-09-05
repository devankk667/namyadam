import React, { useState } from 'react';
import { api } from '../services/api';
import { BrainCircuit, Play, CheckCircle2, Sliders, AlertCircle } from 'lucide-react';

const SAMPLE_PRESETS = [
  {
    name: 'Industrial Refinery Flare Stack',
    data: {
      brightness: 368.5,
      bright_t31: 295.2,
      frp: 35.0,
      confidence: 96.0,
      dist_to_industrial: 0.12,
      industrial_count_2km: 6,
      industrial_count_5km: 14,
      nearest_facility_type: 'refinery',
      persistence_score: 0.92,
      detection_count_30d: 28,
      daynight: 'N'
    }
  },
  {
    name: 'Industrial Explosive Chemical Fire',
    data: {
      brightness: 395.0,
      bright_t31: 315.0,
      frp: 180.0,
      confidence: 99.0,
      dist_to_industrial: 0.25,
      industrial_count_2km: 4,
      industrial_count_5km: 10,
      nearest_facility_type: 'chemical',
      persistence_score: 0.35,
      detection_count_30d: 4,
      daynight: 'D'
    }
  },
  {
    name: 'Boreal Forest Wildfire Outbreak',
    data: {
      brightness: 375.0,
      bright_t31: 305.0,
      frp: 120.0,
      confidence: 92.0,
      dist_to_industrial: 18.5,
      industrial_count_2km: 0,
      industrial_count_5km: 0,
      nearest_facility_type: 'none',
      persistence_score: 0.10,
      detection_count_30d: 2,
      daynight: 'D'
    }
  },
  {
    name: 'Seasonal Crop Stubble Agricultural Burning',
    data: {
      brightness: 332.0,
      bright_t31: 292.0,
      frp: 14.5,
      confidence: 78.0,
      dist_to_industrial: 12.0,
      industrial_count_2km: 0,
      industrial_count_5km: 1,
      nearest_facility_type: 'none',
      persistence_score: 0.05,
      detection_count_30d: 1,
      daynight: 'D'
    }
  }
];

export default function PredictionDemoPage() {
  const [formData, setFormData] = useState(SAMPLE_PRESETS[0].data);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: e.target.type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handleApplyPreset = (presetData) => {
    setFormData(presetData);
    setResult(null);
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
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
            <BrainCircuit className="w-6 h-6 text-orange-400" />
            <span>Interactive Machine Learning Prediction Studio</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Test custom or preset thermal features against trained models for live classification & explainability.
          </p>
        </div>
      </div>

      {/* Preset Selector Buttons */}
      <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl space-y-2">
        <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Quick Preset Scenarios</div>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_PRESETS.map((p, idx) => (
            <button
              key={idx}
              onClick={() => handleApplyPreset(p.data)}
              className="px-3 py-1.5 bg-slate-800 hover:bg-orange-600/20 hover:border-orange-500 border border-slate-700 text-slate-200 text-xs font-medium rounded-xl transition"
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Form + Results Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Parameters Form */}
        <form onSubmit={handlePredict} className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
          <h3 className="font-bold text-white text-base border-b border-slate-800 pb-2 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-orange-400" />
            <span>Feature Input Vector</span>
          </h3>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="block text-slate-400 mb-1">Brightness Temp (K)</label>
              <input
                type="number"
                step="0.1"
                name="brightness"
                value={formData.brightness}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Bright T31 Temp (K)</label>
              <input
                type="number"
                step="0.1"
                name="bright_t31"
                value={formData.bright_t31}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 mb-1">FRP (MW)</label>
              <input
                type="number"
                step="0.1"
                name="frp"
                value={formData.frp}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Confidence (%)</label>
              <input
                type="number"
                step="1"
                name="confidence"
                value={formData.confidence}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Dist to Industrial (km)</label>
              <input
                type="number"
                step="0.01"
                name="dist_to_industrial"
                value={formData.dist_to_industrial}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Facility Type</label>
              <select
                name="nearest_facility_type"
                value={formData.nearest_facility_type}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
              >
                <option value="refinery">Refinery</option>
                <option value="flare_stack">Flare Stack</option>
                <option value="chemical">Chemical</option>
                <option value="gas_terminal">Gas Terminal</option>
                <option value="power_plant">Power Plant</option>
                <option value="steel_works">Steel Works</option>
                <option value="factory">Factory</option>
                <option value="none">None</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Persistence Score (0.0-1.0)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                name="persistence_score"
                value={formData.persistence_score}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Trailing 30-Day Detections</label>
              <input
                type="number"
                step="1"
                name="detection_count_30d"
                value={formData.detection_count_30d}
                onChange={handleInputChange}
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 p-2 rounded-lg focus:outline-none focus:border-orange-500"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-gradient-to-r from-orange-600 to-amber-500 hover:from-orange-500 hover:to-amber-400 text-white font-bold rounded-xl shadow-lg transition flex items-center justify-center gap-2 text-sm"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Execute Model Classification Inference</span>
              </>
            )}
          </button>
        </form>

        {/* Prediction Output / Results */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl flex flex-col justify-between space-y-4">
          <div>
            <h3 className="font-bold text-white text-base border-b border-slate-800 pb-2 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span>Inference Classification Output</span>
            </h3>

            {error && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-xs flex items-center gap-2 mt-4">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {!result && !error && (
              <div className="flex flex-col items-center justify-center h-64 text-center text-slate-500 text-xs space-y-2">
                <BrainCircuit className="w-12 h-12 stroke-[1.5]" />
                <p>Click 'Execute Model Classification Inference' or select a preset scenario above to evaluate.</p>
              </div>
            )}

            {result && (
              <div className="space-y-4 mt-4">
                <div className="bg-slate-800/90 border border-slate-700 p-4 rounded-xl flex items-center justify-between">
                  <div>
                    <div className="text-xs text-slate-400">Predicted Classification</div>
                    <div className="text-xl font-black text-orange-400 capitalize mt-0.5">
                      {result.predicted_class.replace(/_/g, ' ')}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-slate-400">Confidence</div>
                    <div className="text-2xl font-black text-emerald-400">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="bg-slate-800/60 border border-slate-700/60 p-4 rounded-xl text-xs space-y-2">
                  <div className="font-semibold text-slate-300">Model Explanation</div>
                  <p className="text-slate-400 leading-relaxed">{result.explanation.summary}</p>
                </div>

                <div className="space-y-2 pt-2">
                  <div className="text-xs font-semibold text-slate-300">Multi-Class Probability Distribution</div>
                  {Object.entries(result.class_probabilities).map(([cls, p]) => (
                    <div key={cls} className="space-y-1">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span className="capitalize">{cls.replace(/_/g, ' ')}</span>
                        <span className="font-mono text-slate-200">{(p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-1.5 rounded-full ${cls === result.predicted_class ? 'bg-orange-500' : 'bg-slate-600'}`}
                          style={{ width: `${p * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
