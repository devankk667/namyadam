import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import GeospatialMap from '../components/GeospatialMap';
import {
  ArrowLeft,
  Flame,
  Building2,
  MapPin,
  Clock,
  Radio,
  Brain,
  ShieldAlert,
  BarChart2,
  Layers
} from 'lucide-react';

export default function DetectionDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadDetail() {
      setLoading(true);
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

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <div className="w-10 h-10 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 text-sm font-medium">Fetching satellite observation & ML model analysis...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl text-center max-w-lg mx-auto my-12 space-y-4">
        <ShieldAlert className="w-12 h-12 text-rose-500 mx-auto" />
        <h3 className="text-lg font-bold text-white">Detection Record Not Found</h3>
        <p className="text-sm text-slate-400">{error || `No record matching ID '${id}'`}</p>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white text-xs font-semibold rounded-lg transition"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  const { record, ml_prediction } = data;

  return (
    <div className="space-y-6">
      {/* Top Header & Navigation */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center space-x-2 text-slate-400 hover:text-white text-xs font-medium transition"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Monitoring Dashboard</span>
        </button>

        <span className="text-xs font-mono text-slate-500">
          EVENT RECORD ID: <strong className="text-orange-400">{record.id}</strong>
        </span>
      </div>

      {/* Main Banner Header */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-3">
            <span className={`px-3 py-1 rounded-md text-xs font-mono font-bold uppercase ${
              record.true_class === 'industrial_fire' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
              record.true_class === 'industrial_thermal_source' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/40' :
              'bg-amber-500/20 text-amber-400 border border-amber-500/40'
            }`}>
              {record.true_class ? record.true_class.replace(/_/g, ' ') : 'Thermal Anomaly'}
            </span>
            <span className="text-slate-400 text-xs flex items-center gap-1">
              <MapPin className="w-3.5 h-3.5 text-slate-500" />
              {record.region} ({record.latitude.toFixed(3)}, {record.longitude.toFixed(3)})
            </span>
          </div>
          <h2 className="text-xl font-black text-white tracking-wide pt-1">
            Observed {record.true_class ? record.true_class.replace(/_/g, ' ') : 'Thermal Anomaly'} Event
          </h2>
          <p className="text-xs text-slate-400 flex items-center gap-2">
            <Clock className="w-3.5 h-3.5" />
            Acquisition Timestamp: {record.acq_date} at {record.acq_time} UTC via {record.satellite} ({record.instrument})
          </p>
        </div>

        {ml_prediction && (
          <div className="bg-slate-800/90 border border-slate-700/80 p-3.5 rounded-xl text-center min-w-[200px]">
            <div className="text-[11px] text-slate-400 font-medium">Model Classification Confidence</div>
            <div className="text-2xl font-black text-orange-400 mt-0.5">
              {(ml_prediction.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              Version {ml_prediction.model_version}
            </div>
          </div>
        )}
      </div>

      {/* Main Grid: Metrics, Spatial Context & ML Explanation */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Physical & Spatial Attributes Column */}
        <div className="space-y-6">
          {/* Thermal Physical Features */}
          <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
            <h3 className="font-bold text-white text-sm border-b border-slate-800 pb-2 flex items-center gap-2">
              <Flame className="w-4 h-4 text-orange-400" />
              <span>Thermal Physical Characteristics</span>
            </h3>

            <div className="grid grid-cols-2 gap-4 text-xs">
              <div className="bg-slate-800/60 p-3 rounded-xl border border-slate-700/60">
                <div className="text-slate-400 font-medium">Fire Radiative Power</div>
                <div className="text-lg font-black text-amber-400 mt-1">{record.frp} MW</div>
              </div>

              <div className="bg-slate-800/60 p-3 rounded-xl border border-slate-700/60">
                <div className="text-slate-400 font-medium">Detection Confidence</div>
                <div className="text-lg font-black text-emerald-400 mt-1">{record.confidence}%</div>
              </div>

              <div className="bg-slate-800/60 p-3 rounded-xl border border-slate-700/60">
                <div className="text-slate-400 font-medium">Brightness Temp (I-4)</div>
                <div className="text-base font-bold text-slate-200 mt-1">{record.brightness} K</div>
              </div>

              <div className="bg-slate-800/60 p-3 rounded-xl border border-slate-700/60">
                <div className="text-slate-400 font-medium">Brightness Temp (T31)</div>
                <div className="text-base font-bold text-slate-200 mt-1">{record.bright_t31} K</div>
              </div>
            </div>
          </div>

          {/* Spatial Context (OSM Context) */}
          <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
            <h3 className="font-bold text-white text-sm border-b border-slate-800 pb-2 flex items-center gap-2">
              <Building2 className="w-4 h-4 text-cyan-400" />
              <span>Geospatial Context (OSM)</span>
            </h3>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
                <span className="text-slate-400">Nearest Industrial Facility:</span>
                <span className="font-bold text-slate-200">{record.dist_to_industrial} km</span>
              </div>

              <div className="flex justify-between items-center bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
                <span className="text-slate-400">Facility Type:</span>
                <span className="font-mono text-orange-400 uppercase font-semibold">{record.nearest_facility_type}</span>
              </div>

              <div className="flex justify-between items-center bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
                <span className="text-slate-400">Industrial Count (2 km / 5 km):</span>
                <span className="font-bold text-slate-200">{record.industrial_count_2km} / {record.industrial_count_5km}</span>
              </div>

              <div className="flex justify-between items-center bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
                <span className="text-slate-400">Historical Persistence Score:</span>
                <span className="font-bold text-amber-400">{(record.persistence_score * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* ML Model Explanation & Class Probabilities Column */}
        <div className="lg:col-span-2 space-y-6">
          {ml_prediction && (
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
              <h3 className="font-bold text-white text-sm border-b border-slate-800 pb-2 flex items-center gap-2">
                <Brain className="w-4 h-4 text-orange-400" />
                <span>Machine Learning Explanation & Class Probabilities</span>
              </h3>

              <div className="bg-slate-800/80 border border-slate-700/80 p-4 rounded-xl text-xs space-y-2">
                <p className="text-slate-200 leading-relaxed">{ml_prediction.explanation.summary}</p>
              </div>

              <div className="space-y-2 pt-2">
                <div className="text-xs text-slate-400 font-medium mb-1">Model Multi-Class Probability Distribution</div>
                {Object.entries(ml_prediction.class_probabilities).map(([clsName, prob]) => (
                  <div key={clsName} className="space-y-1">
                    <div className="flex justify-between text-xs text-slate-300">
                      <span className="capitalize font-mono">{clsName.replace(/_/g, ' ')}</span>
                      <span className="font-bold">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden border border-slate-700">
                      <div
                        className={`h-2 rounded-full ${
                          clsName === ml_prediction.predicted_class ? 'bg-orange-500' : 'bg-slate-600'
                        }`}
                        style={{ width: `${prob * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Event Map View */}
          <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
            <h3 className="font-bold text-white text-sm border-b border-slate-800 pb-2 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-orange-400" />
              <span>Event Geographic Location</span>
            </h3>

            <GeospatialMap
              detections={[record]}
              selectedDetection={record}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
