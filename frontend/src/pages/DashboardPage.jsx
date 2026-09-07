import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import GeospatialMap from '../components/GeospatialMap';
import {
  RefreshCw,
  ArrowUpRight,
  MapPin,
  Building2,
} from 'lucide-react';
import { EmptyState, KeyValueRow, LoadingState, Metric, PageHeader, Panel, StatusBadge } from '../components/ui';
import { classificationMeta, formatClassLabel, severityStatus } from '../lib/classification';

export default function DashboardPage() {
  const navigate = useNavigate();

  const [summary, setSummary] = useState(null);
  const [detections, setDetections] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastRefreshed, setLastRefreshed] = useState(null);

  const [minConf, setMinConf] = useState('');
  const [minFrp, setMinFrp] = useState('');
  const [selectedClass, setSelectedClass] = useState('');
  const [searchRegion, setSearchRegion] = useState('');

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [sumRes, detRes, altRes] = await Promise.all([
        api.getAnalyticsSummary(),
        api.getDetections({
          min_confidence: minConf || undefined,
          min_frp: minFrp || undefined,
          classification: selectedClass || undefined,
          region: searchRegion || undefined,
          page_size: 100
        }),
        api.getAlerts()
      ]);

      setSummary(sumRes);
      setDetections(detRes.items || []);
      setAlerts(altRes || []);
      setSelectedDetection((prev) => prev || (detRes.items && detRes.items[0]) || null);
      setLastRefreshed(new Date());
    } catch (err) {
      console.error('Dashboard error loading data:', err);
    } finally {
      setLoading(false);
    }
  }, [minConf, minFrp, selectedClass, searchRegion]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Monitoring Console"
        description="Real-time thermal anomaly identification, industrial facility spatial enrichment, and risk analysis."
        badges={<StatusBadge status="info" dot pulse size="xs">live feed</StatusBadge>}
        meta={[{ label: 'last sync', value: lastRefreshed ? lastRefreshed.toLocaleTimeString() : '—' }]}
        actions={
          <button
            onClick={loadData}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-panel2 hover:bg-line border border-line text-ink-secondary text-[11px] font-mono uppercase tracking-wide transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-accent' : ''}`} />
            Refresh
          </button>
        }
      />

      {/* Summary metrics */}
      <Panel title="Detection Summary" eyebrow="analytics/summary" noPadding>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 divide-x divide-y lg:divide-y-0 divide-line">
          <Metric label="Total Detections" value={summary?.total_detections ?? '—'} caption="VIIRS / MODIS" help="All thermal anomaly records currently loaded in the active dataset." />
          <Metric label="High Risk" value={summary?.high_risk_events ?? '—'} tone="critical" caption="FRP > 50MW or fire" help="Events flagged critical: fire radiative power above 50MW, or classified as an industrial fire." />
          <Metric label="Industrial Assoc." value={summary?.industrial_associated_detections ?? '—'} tone="accent" caption="near refinery/flare" help="Detections located within proximity thresholds of a known industrial facility (OSM)." />
          <Metric label="Persistent Sources" value={summary?.persistent_sources ?? '—'} tone="warning" caption="30-day recurring" help="Locations with repeated thermal activity over the trailing 30-day observation window." />
          <Metric label="Active Flare Stacks" value={summary?.active_thermal_sources ?? '—'} tone="info" caption="operational flares" help="Persistent thermal sources currently classified as operating flare stacks." />
          <Metric label="High Confidence" value={summary?.high_confidence_events ?? '—'} tone="success" caption="≥ 90% confidence" help="Detections where the sensor-reported confidence score meets or exceeds 90%." />
        </div>
      </Panel>

      {/* Map + inspector */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 items-start">
        <Panel
          title="Interactive Thermal Map"
          eyebrow="geospatial"
          className="xl:col-span-2"
          noPadding
          actions={
            <div className="flex flex-wrap items-center gap-1.5">
              <select
                value={selectedClass}
                onChange={(e) => setSelectedClass(e.target.value)}
                className="bg-panel2 border border-line text-[11px] font-mono text-ink-secondary px-2 py-1 focus:outline-none focus:border-accent"
              >
                <option value="">all classes</option>
                <option value="industrial_fire">industrial fire</option>
                <option value="industrial_thermal_source">persistent source</option>
                <option value="wildfire">wildfire</option>
                <option value="agricultural_burning">agricultural</option>
                <option value="other_thermal_anomaly">other anomaly</option>
              </select>
              <select
                value={minConf}
                onChange={(e) => setMinConf(e.target.value)}
                className="bg-panel2 border border-line text-[11px] font-mono text-ink-secondary px-2 py-1 focus:outline-none focus:border-accent"
              >
                <option value="">min confidence</option>
                <option value="70">≥ 70%</option>
                <option value="85">≥ 85%</option>
                <option value="95">≥ 95%</option>
              </select>
              <input
                type="text"
                placeholder="search region…"
                value={searchRegion}
                onChange={(e) => setSearchRegion(e.target.value)}
                className="bg-panel2 border border-line text-[11px] font-mono text-ink-secondary px-2 py-1 focus:outline-none focus:border-accent w-28"
              />
            </div>
          }
        >
          <GeospatialMap
            detections={detections}
            selectedDetection={selectedDetection}
            onSelectDetection={(det) => setSelectedDetection(det)}
          />
        </Panel>

        {/* Right column: inspector + warnings */}
        <div className="space-y-5">
          <Panel title="Selected Event" eyebrow="inspector" icon={MapPin}>
            {selectedDetection ? (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[12px] text-ink-primary font-semibold">{selectedDetection.id}</span>
                  <StatusBadge status={classificationMeta(selectedDetection.true_class).status} size="xs">
                    {formatClassLabel(selectedDetection.true_class)}
                  </StatusBadge>
                </div>
                <KeyValueRow label="Region" value={selectedDetection.region} />
                <KeyValueRow label="FRP" value={`${selectedDetection.frp} MW`} mono tone="accent" />
                <KeyValueRow label="Confidence" value={`${selectedDetection.confidence}%`} mono />
                <KeyValueRow
                  label="Nearest facility"
                  value={`${selectedDetection.dist_to_industrial} km · ${selectedDetection.nearest_facility_type}`}
                  mono
                />
                <button
                  onClick={() => navigate(`/detections/${selectedDetection.id}`)}
                  className="w-full mt-1 flex items-center justify-center gap-1.5 py-1.5 bg-panel2 hover:bg-line border border-line text-[11px] font-medium text-ink-secondary hover:text-ink-primary transition-colors"
                >
                  Open full analysis
                  <ArrowUpRight className="w-3 h-3" />
                </button>
              </div>
            ) : (
              <EmptyState message="Select a marker on the map or a row below to inspect." />
            )}
          </Panel>

          <Panel
            title="Active Early Warnings"
            eyebrow="rule engine"
            actions={<span className="text-[10px] font-mono text-ink-muted">{alerts.length} total</span>}
          >
            <div className="space-y-0 max-h-[320px] overflow-y-auto -mx-1 px-1">
              {alerts.slice(0, 6).map((alt) => (
                <button
                  key={alt.id}
                  onClick={() => navigate(`/detections/${alt.detection_id}`)}
                  className="w-full text-left border-b border-line/70 last:border-b-0 py-2 hover:bg-panel2/60 transition-colors"
                >
                  <div className="flex items-center justify-between gap-2">
                    <StatusBadge status={severityStatus(alt.severity)} size="xs">{alt.severity}</StatusBadge>
                    <span className="text-[10px] font-mono text-ink-muted truncate">{alt.timestamp}</span>
                  </div>
                  <p className="text-[11px] text-ink-secondary mt-1 line-clamp-2">{alt.explanation}</p>
                  <div className="text-[10px] font-mono text-ink-muted mt-0.5">{alt.region}</div>
                </button>
              ))}
              {alerts.length === 0 && !loading && <EmptyState message="No active warnings." />}
            </div>
            <button
              onClick={() => navigate('/alerts')}
              className="w-full mt-2 flex items-center justify-center gap-1.5 py-1.5 border border-line text-[11px] font-medium text-ink-secondary hover:text-ink-primary hover:bg-panel2 transition-colors"
            >
              View all {alerts.length} warnings
              <ArrowUpRight className="w-3 h-3" />
            </button>
          </Panel>
        </div>
      </div>

      {/* Detections table */}
      <Panel
        title="Observed Thermal Anomalies"
        eyebrow="detections"
        icon={Building2}
        actions={<span className="text-[11px] font-mono text-ink-muted">{detections.length} rows</span>}
        noPadding
      >
        {loading ? (
          <LoadingState label="fetching /api/detections…" />
        ) : detections.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[12px] whitespace-nowrap">
              <thead className="bg-panel2 text-ink-muted font-mono text-[10px] uppercase tracking-wide border-b border-line">
                <tr>
                  <th className="px-3 py-2 font-medium">ID / Acquired</th>
                  <th className="px-3 py-2 font-medium">Region</th>
                  <th className="px-3 py-2 font-medium">Classification</th>
                  <th className="px-3 py-2 font-medium">FRP</th>
                  <th className="px-3 py-2 font-medium">Confidence</th>
                  <th className="px-3 py-2 font-medium">Nearest Facility</th>
                  <th className="px-3 py-2 font-medium">Persistence</th>
                  <th className="px-3 py-2 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {detections.slice(0, 15).map((det) => {
                  const meta = classificationMeta(det.true_class);
                  return (
                    <tr
                      key={det.id}
                      onClick={() => setSelectedDetection(det)}
                      className={`cursor-pointer hover:bg-panel2/60 transition-colors ${selectedDetection?.id === det.id ? 'bg-accent/[0.05]' : ''}`}
                    >
                      <td className="px-3 py-2">
                        <div className="font-mono font-medium text-ink-primary">{det.id}</div>
                        <div className="text-[10px] font-mono text-ink-muted">{det.acq_date} {det.acq_time}</div>
                      </td>
                      <td className="px-3 py-2 text-ink-secondary">{det.region}</td>
                      <td className="px-3 py-2">
                        <StatusBadge status={meta.status} size="xs">{meta.label}</StatusBadge>
                      </td>
                      <td className="px-3 py-2 font-mono text-accent font-semibold">{det.frp} MW</td>
                      <td className="px-3 py-2 font-mono text-ink-secondary">{det.confidence}%</td>
                      <td className="px-3 py-2 font-mono text-ink-secondary">
                        {det.dist_to_industrial} km <span className="text-ink-muted">({det.nearest_facility_type})</span>
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div className="w-14 h-1.5 bg-panel2 border border-line overflow-hidden">
                            <div className="h-full bg-accent" style={{ width: `${det.persistence_score * 100}%` }} />
                          </div>
                          <span className="text-[10px] font-mono text-ink-muted">{(det.persistence_score * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <button
                          onClick={(e) => { e.stopPropagation(); navigate(`/detections/${det.id}`); }}
                          className="inline-flex items-center gap-1 text-[11px] font-medium text-ink-muted hover:text-accent transition-colors"
                        >
                          Inspect
                          <ArrowUpRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
