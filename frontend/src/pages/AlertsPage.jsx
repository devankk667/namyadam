import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { ArrowUpRight, ChevronRight, ChevronLeft } from 'lucide-react';
import { Panel, PageHeader, StatusBadge, LoadingState, EmptyState } from '../components/ui';
import { severityStatus } from '../lib/classification';

const SEVERITIES = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'];
const PAGE_SIZE = 25;

export default function AlertsPage() {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [expandedId, setExpandedId] = useState(null);
  const [page, setPage] = useState(1);

  useEffect(() => {
    async function loadAlerts() {
      setLoading(true);
      try {
        const res = await api.getAlerts();
        setAlerts(res || []);
      } catch (err) {
        console.error('Alerts load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadAlerts();
  }, []);

  useEffect(() => {
    setPage(1);
  }, [severityFilter]);

  const filteredAlerts = useMemo(
    () => alerts.filter((a) => severityFilter === 'ALL' || a.severity === severityFilter),
    [alerts, severityFilter]
  );

  const pageCount = Math.max(1, Math.ceil(filteredAlerts.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  const pagedAlerts = filteredAlerts.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Early Warning Rules"
        description="Automated rule engine evaluating high-FRP outbursts, industrial fires, and persistent flaring anomalies."
        meta={[{ label: 'active rules', value: 4 }, { label: 'firing', value: alerts.length }]}
      />

      <Panel
        title="Rule Evaluation Output"
        eyebrow="alerts"
        noPadding
        actions={
          <div className="flex items-center gap-1">
            {SEVERITIES.map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={`px-2.5 py-1 text-[10px] font-mono uppercase tracking-wide border transition-colors ${
                  severityFilter === sev
                    ? 'bg-accent/10 border-accent/40 text-accent'
                    : 'border-line text-ink-muted hover:text-ink-secondary hover:border-line2'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        }
      >
        {loading ? (
          <LoadingState label="evaluating risk rules…" />
        ) : filteredAlerts.length === 0 ? (
          <EmptyState message={`No active alerts matching filter '${severityFilter}'.`} />
        ) : (
          <div className="divide-y divide-line">
            {pagedAlerts.map((alt) => {
              const isOpen = expandedId === alt.id;
              return (
                <div key={alt.id}>
                  <button
                    onClick={() => setExpandedId(isOpen ? null : alt.id)}
                    className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-panel2/50 transition-colors"
                  >
                    <ChevronRight className={`w-3.5 h-3.5 text-ink-muted shrink-0 transition-transform ${isOpen ? 'rotate-90' : ''}`} />
                    <StatusBadge status={severityStatus(alt.severity)} size="xs">{alt.severity}</StatusBadge>
                    <span className="text-[11px] font-mono text-ink-muted shrink-0 hidden sm:inline">{alt.id}</span>
                    <span className="text-[12px] text-ink-primary font-medium truncate flex-1">
                      {alt.type.replace(/_/g, ' ')}
                    </span>
                    <span className="text-[11px] text-ink-muted shrink-0 hidden md:inline">{alt.region}</span>
                    <span className="text-[10px] font-mono text-ink-muted shrink-0">{alt.timestamp}</span>
                  </button>

                  {isOpen && (
                    <div className="px-4 pb-4 pl-10 space-y-3">
                      <p className="text-[12px] text-ink-secondary leading-relaxed border-l-2 border-accent/50 pl-3 max-w-2xl">
                        {alt.explanation}
                      </p>
                      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] font-mono text-ink-muted">
                        <span>detection_id: <span className="text-ink-secondary">{alt.detection_id}</span></span>
                        <span>lat/lon: <span className="text-ink-secondary">{alt.latitude.toFixed(3)}, {alt.longitude.toFixed(3)}</span></span>
                        <span>rule_type: <span className="text-ink-secondary">{alt.type}</span></span>
                      </div>
                      <button
                        onClick={() => navigate(`/detections/${alt.detection_id}`)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-panel2 hover:bg-line border border-line text-[11px] font-medium text-ink-secondary hover:text-ink-primary transition-colors"
                      >
                        Inspect source detection
                        <ArrowUpRight className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {!loading && filteredAlerts.length > 0 && (
          <div className="flex items-center justify-between px-4 py-2 border-t border-line text-[11px] font-mono text-ink-muted">
            <span>
              showing {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, filteredAlerts.length)} of {filteredAlerts.length}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={currentPage <= 1}
                className="flex items-center gap-1 px-2 py-1 border border-line hover:border-line2 disabled:opacity-30 disabled:hover:border-line transition-colors"
              >
                <ChevronLeft className="w-3 h-3" /> prev
              </button>
              <span className="text-ink-secondary">page {currentPage} / {pageCount}</span>
              <button
                onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
                disabled={currentPage >= pageCount}
                className="flex items-center gap-1 px-2 py-1 border border-line hover:border-line2 disabled:opacity-30 disabled:hover:border-line transition-colors"
              >
                next <ChevronRight className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}
