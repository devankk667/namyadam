import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { BarChart3, Activity, PieChart as PieIcon, Layers, Flame } from 'lucide-react';

const COLORS = ['#ef4444', '#f97316', '#eab308', '#10b981', '#6b7280'];

export default function AnalyticsPage() {
  const [temporalData, setTemporalData] = useState([]);
  const [classData, setClassData] = useState([]);
  const [regionData, setRegionData] = useState([]);
  const [persistenceData, setPersistenceData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadAnalytics() {
      setLoading(true);
      try {
        const [tempRes, classRes, regRes, persRes] = await Promise.all([
          api.getAnalyticsTemporal(),
          api.getAnalyticsClassification(),
          api.getAnalyticsRegions(),
          api.getAnalyticsPersistence()
        ]);
        setTemporalData(tempRes || []);
        setClassData(classRes || []);
        setRegionData(regRes || []);
        setPersistenceData(persRes || []);
      } catch (err) {
        console.error("Analytics load error:", err);
      } finally {
        setLoading(false);
      }
    }
    loadAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <div className="w-10 h-10 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 text-sm font-medium">Computing spatial-temporal metrics and regional persistence...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-orange-400" />
            <span>Geospatial Analytics & Persistence Studio</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Temporal activity trends, classification distributions, and recurring industrial thermal source detection.
          </p>
        </div>
      </div>

      {/* Grid: Temporal Trends & Classification Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Temporal Observation Trend */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
          <h3 className="font-bold text-white text-base border-b border-slate-800 pb-2 flex items-center gap-2">
            <Activity className="w-5 h-5 text-orange-400" />
            <span>Temporal Thermal Observation Trend</span>
          </h3>

          <div className="h-72 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={temporalData}>
                <defs>
                  <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="acq_date" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
                />
                <Area type="monotone" dataKey="total_events" stroke="#f97316" fillOpacity={1} fill="url(#colorEvents)" name="Total Events" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Classification Breakdown */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
          <h3 className="font-bold text-white text-base border-b border-slate-800 pb-2 flex items-center gap-2">
            <PieIcon className="w-5 h-5 text-amber-400" />
            <span>Thermal Classification Breakdown</span>
          </h3>

          <div className="h-72 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classData}
                  dataKey="count"
                  nameKey="classification"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ classification, percentage }) => `${classification.replace(/_/g, ' ')} (${percentage}%)`}
                  labelLine={false}
                >
                  {classData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Persistence Analysis & Regional Table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Regional Activity Breakdown */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
          <h3 className="font-bold text-white text-base border-b border-slate-800 pb-2 flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <span>Regional Industrial Thermal Density</span>
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-800 text-slate-400 uppercase font-mono border-b border-slate-700">
                <tr>
                  <th className="p-2.5">Region</th>
                  <th className="p-2.5">Detections</th>
                  <th className="p-2.5">Avg Persistence</th>
                  <th className="p-2.5">High Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {regionData.map((reg, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40">
                    <td className="p-2.5 font-semibold text-slate-200">{reg.region}</td>
                    <td className="p-2.5 text-slate-300">{reg.total_detections}</td>
                    <td className="p-2.5 text-amber-400 font-bold">{(reg.avg_persistence * 100).toFixed(1)}%</td>
                    <td className="p-2.5 text-red-400 font-bold">{reg.high_risk_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Top Persistent Thermal Clusters */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
          <h3 className="font-bold text-white text-base border-b border-slate-800 pb-2 flex items-center gap-2">
            <Flame className="w-5 h-5 text-orange-400" />
            <span>Top Persistent Operational Flares / Sources</span>
          </h3>

          <div className="space-y-2.5 max-h-[300px] overflow-y-auto pr-1">
            {persistenceData.slice(0, 8).map((pitem) => (
              <div key={pitem.detection_id} className="bg-slate-800/70 border border-slate-700/70 p-3 rounded-xl flex items-center justify-between text-xs">
                <div>
                  <div className="font-bold text-slate-100">{pitem.region} ({pitem.detection_id})</div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    Facility: <span className="text-orange-400 font-mono">{pitem.nearest_facility}</span> &bull; 30d Obs: {pitem.detection_count_30d}
                  </div>
                </div>
                <div className="text-right">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-orange-500/20 text-orange-400 border border-orange-500/30">
                    {pitem.status}
                  </span>
                  <div className="text-[10px] text-slate-400 mt-1 font-bold">
                    {(pitem.persistence_score * 100).toFixed(0)}% persistence
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
