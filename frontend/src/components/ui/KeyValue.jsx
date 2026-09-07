import React from 'react';
import Tooltip from './Tooltip';

export function KeyValueRow({ label, value, mono = false, tone = 'default', help, dense = false }) {
  const toneClass = {
    default: 'text-ink-primary',
    success: 'text-status-success',
    warning: 'text-status-warning',
    critical: 'text-status-critical',
    info: 'text-status-info',
    accent: 'text-accent',
    muted: 'text-ink-muted',
  }[tone] || 'text-ink-primary';

  return (
    <div className={`flex items-center justify-between gap-3 ${dense ? 'py-1' : 'py-1.5'} border-b border-line/70 last:border-b-0 text-[12px]`}>
      <span className="flex items-center gap-1 text-ink-muted shrink-0">
        {label}
        {help && <Tooltip content={help} />}
      </span>
      <span className={`text-right truncate font-medium ${mono ? 'font-mono' : ''} ${toneClass}`}>{value}</span>
    </div>
  );
}
