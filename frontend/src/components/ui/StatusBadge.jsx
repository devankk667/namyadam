import React from 'react';

// The only palette that is allowed to carry meaning in this UI.
// success = healthy/positive, warning = attention, critical = error/severe,
// info = active/informational, neutral = inactive/metadata.
const STYLES = {
  success: 'text-status-success border-status-success/35 bg-status-success/10',
  warning: 'text-status-warning border-status-warning/35 bg-status-warning/10',
  critical: 'text-status-critical border-status-critical/35 bg-status-critical/10',
  info: 'text-status-info border-status-info/35 bg-status-info/10',
  neutral: 'text-ink-muted border-line2 bg-panel2',
  accent: 'text-accent border-accent/35 bg-accent/10',
};

const DOT_STYLES = {
  success: 'bg-status-success',
  warning: 'bg-status-warning',
  critical: 'bg-status-critical',
  info: 'bg-status-info',
  neutral: 'bg-ink-muted',
  accent: 'bg-accent',
};

export default function StatusBadge({ status = 'neutral', children, dot = false, pulse = false, size = 'sm' }) {
  const sizeClass = size === 'xs' ? 'text-[10px] px-1.5 py-0.5' : 'text-[11px] px-2 py-0.5';
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-sm border font-mono font-semibold uppercase tracking-wide leading-none ${sizeClass} ${STYLES[status] || STYLES.neutral}`}
    >
      {dot && (
        <span className="relative flex w-1.5 h-1.5">
          {pulse && (
            <span className={`absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping ${DOT_STYLES[status]}`} />
          )}
          <span className={`relative inline-flex rounded-full w-1.5 h-1.5 ${DOT_STYLES[status] || DOT_STYLES.neutral}`} />
        </span>
      )}
      {children}
    </span>
  );
}
