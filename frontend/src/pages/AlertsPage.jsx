import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { ShieldAlert, Bell, ExternalLink, Filter, AlertOctagon } from 'lucide-react';

export default function AlertsPage() {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [severityFilter, setSeverityFilter] = useState('ALL');

  useEffect(() => {
    async function loadAlerts() {
      setLoading(true);
      try {
        const res = await api.getAlerts();
        setAlerts(res || []);
      } catch (err) {
        console.error("Alerts load error:", err);
      } finally {
        setLoading(false);
      }
    }
    loadAlerts();
  }, []);

  const filteredAlerts = alerts.filter((a) => {
    if (severityFilter === 'ALL') return true;
    return a.severity === severityFilter;
  });

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <div className="w-10 h-10 border-4 border-red-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 text-sm font-medium">Evaluating automated thermal risk rules and early warnings...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
            <ShieldAlert className="w-6 h-6 text-red-500" />
            <span>Early Warning & Automated Risk Alerts</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Real-time rule engine evaluating high-FRP outbursts, industrial fires, and persistent flaring anomalies.
          </p>
        </div>

        {/* Severity Filter */}
        <div className="flex items-center space-x-2 bg-slate-800/80 p-1.5 rounded-xl border border-slate-700 text-xs">
          <Filter className="w-4 h-4 text-slate-400 ml-1" />
          {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'].map((sev) => (
            <button
              key={sev}
              onClick={() => setSeverityFilter(sev)}
              className={`px-3 py-1.5 rounded-lg font-mono font-bold transition ${
                severityFilter === sev
                  ? 'bg-orange-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              {sev}
            </button>
          ))}
        </div>
      </div>

      {/* Alerts Grid */}
      <div className="space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl text-center text-slate-400">
            No active alerts matching filter '{severityFilter}'
          </div>
        ) : (
          filteredAlerts.map((alt) => (
            <div
              key={alt.id}
              className={`bg-slate-900 border p-5 rounded-2xl shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4 transition hover:border-orange-500/50 ${
                alt.severity === 'CRITICAL' ? 'border-red-500/40 border-l-8 border-l-red-500' :
                alt.severity === 'HIGH' ? 'border-orange-500/40 border-l-8 border-l-orange-500' :
                'border-amber-500/40 border-l-8 border-l-amber-500'
              }`}
            >
              <div className="space-y-1.5 max-w-2xl">
                <div className="flex items-center space-x-3">
                  <span className={`px-2.5 py-0.5 rounded text-[11px] font-mono font-bold ${
                    alt.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                    alt.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/40' :
                    'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                  }`}>
                    {alt.severity}
                  </span>

                  <span className="text-xs font-mono text-slate-400">{alt.id}</span>
                  <span className="text-xs text-slate-500">&bull; {alt.timestamp}</span>
                </div>

                <h4 className="text-base font-bold text-white flex items-center gap-2">
                  <AlertOctagon className="w-4 h-4 text-orange-400" />
                  <span>{alt.type.replace(/_/g, ' ')}</span>
                </h4>

                <p className="text-xs text-slate-300 leading-relaxed">{alt.explanation}</p>

                <div className="text-[11px] text-slate-400 pt-1">
                  Region: <strong className="text-slate-200">{alt.region}</strong> ({alt.latitude.toFixed(3)}, {alt.longitude.toFixed(3)})
                </div>
              </div>

              <button
                onClick={() => navigate(`/detections/${alt.detection_id}`)}
                className="px-4 py-2 bg-slate-800 hover:bg-orange-600 text-slate-200 hover:text-white rounded-xl border border-slate-700 transition flex items-center gap-2 text-xs font-semibold whitespace-nowrap"
              >
                <span>Inspect Incident</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
