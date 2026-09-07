import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { Activity, PieChart as PieIcon, Layers, Flame } from 'lucide-react';
import { Panel, PageHeader, StatusBadge, LoadingState, EmptyState } from '../components/ui';
import { classificationMeta, formatClassLabel } from '../lib/classification';

const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#0d1117',
  borderColor: '#2a3341',
  borderRadius: 2,
  color: '#e6e9ef',
  fontSize: 11,
  fontFamily: 'ui-monospace, monospace',
};

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
        console.error('Analytics load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadAnalytics();
  }, []);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Analytics & Persistence"
        description="Temporal activity trends, classification distributions, and recurring industrial thermal source detection."
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Panel title="Temporal Observation Trend" eyebrow="analytics/temporal" icon={Activity}>
          {loading ? (
            <LoadingState label="computing temporal trend…" />
          ) : temporalData.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={temporalData}>
                  <defs>
                    <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#e3934a" stopOpacity={0.5} />
                      <stop offset="95%" stopColor="#e3934a" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#1e2530" vertical={false} />
                  <XAxis dataKey="acq_date" stroke="#5f6b7a" fontSize={10} tickLine={false} axisLine={{ stroke: '#1e2530' }} />
                  <YAxis stroke="#5f6b7a" fontSize={10} tickLine={false} axisLine={{ stroke: '#1e2530' }} />
                  <RechartsTooltip contentStyle={CHART_TOOLTIP_STYLE} />
                  <Area type="monotone" dataKey="total_events" stroke="#e3934a" strokeWidth={1.5} fillOpacity={1} fill="url(#colorEvents)" name="Total Events" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>

        <Panel
          title="Classification Breakdown"
          eyebrow="analytics/classification"
          icon={PieIcon}
          className="h-full flex flex-col"
          bodyClassName="flex-1 flex items-center"
        >
          {loading ? (
            <LoadingState label="computing class distribution…" />
          ) : classData.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 items-center w-full">
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={classData} dataKey="count" nameKey="classification" cx="50%" cy="50%" outerRadius={80} innerRadius={44} strokeWidth={1} stroke="#090c10">
                      {classData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={classificationMeta(entry.classification).hex} />
                      ))}
                    </Pie>
                    <RechartsTooltip contentStyle={CHART_TOOLTIP_STYLE} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-1.5">
                {classData.map((entry, idx) => (
                  <div key={idx} className="flex items-center justify-between text-[11px]">
                    <span className="flex items-center gap-1.5 text-ink-secondary truncate">
                      <span className="w-2 h-2 shrink-0" style={{ backgroundColor: classificationMeta(entry.classification).hex }} />
                      {formatClassLabel(entry.classification)}
                    </span>
                    <span className="font-mono text-ink-primary">{entry.count} <span className="text-ink-muted">({entry.percentage}%)</span></span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Panel title="Regional Industrial Thermal Density" eyebrow="analytics/regions" icon={Layers} noPadding>
          {loading ? (
            <LoadingState label="aggregating regional metrics…" />
          ) : regionData.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[12px] whitespace-nowrap">
                <thead className="bg-panel2 text-ink-muted font-mono text-[10px] uppercase tracking-wide border-b border-line">
                  <tr>
                    <th className="px-3 py-2 font-medium">Region</th>
                    <th className="px-3 py-2 font-medium">Detections</th>
                    <th className="px-3 py-2 font-medium">Avg Persistence</th>
                    <th className="px-3 py-2 font-medium">High Risk</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {regionData.map((reg, idx) => (
                    <tr key={idx} className="hover:bg-panel2/50 transition-colors">
                      <td className="px-3 py-2 font-medium text-ink-primary">{reg.region}</td>
                      <td className="px-3 py-2 font-mono text-ink-secondary">{reg.total_detections}</td>
                      <td className="px-3 py-2 font-mono text-status-warning">{(reg.avg_persistence * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 font-mono text-status-critical">{reg.high_risk_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <Panel title="Top Persistent Operational Sources" eyebrow="analytics/persistence" icon={Flame} noPadding>
          {loading ? (
            <LoadingState label="ranking persistent clusters…" />
          ) : persistenceData.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="divide-y divide-line max-h-[300px] overflow-y-auto">
              {persistenceData.slice(0, 8).map((pitem) => (
                <div key={pitem.detection_id} className="flex items-center justify-between px-3 py-2.5 text-[12px]">
                  <div className="min-w-0">
                    <div className="font-medium text-ink-primary truncate">{pitem.region} <span className="text-ink-muted font-mono text-[11px]">({pitem.detection_id})</span></div>
                    <div className="text-[10px] font-mono text-ink-muted mt-0.5">
                      facility: <span className="text-accent">{pitem.nearest_facility}</span> · 30d obs: {pitem.detection_count_30d}
                    </div>
                  </div>
                  <div className="text-right shrink-0 ml-3">
                    <StatusBadge status="accent" size="xs">{pitem.status}</StatusBadge>
                    <div className="text-[10px] font-mono text-ink-muted mt-1">{(pitem.persistence_score * 100).toFixed(0)}%</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
