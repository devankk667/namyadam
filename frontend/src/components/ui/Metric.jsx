import React from 'react';
import Tooltip from './Tooltip';

const TONE_TEXT = {
  default: 'text-ink-primary',
  success: 'text-status-success',
  warning: 'text-status-warning',
  critical: 'text-status-critical',
  info: 'text-status-info',
  accent: 'text-accent',
};

/**
 * A single statistic cell. Intended to sit inside a divided grid
 * (`divide-x divide-y divide-line`) so values are separated by hairlines
 * instead of boxed in individual cards.
 */
export function Metric({ label, value, caption, tone = 'default', help, mono = true }) {
  return (
    <div className="px-4 py-3 min-w-0">
      <div className="flex items-center gap-1 text-[10px] font-mono uppercase tracking-wider text-ink-muted">
        <span className="truncate">{label}</span>
        {help && <Tooltip content={help} />}
      </div>
      <div className={`mt-1 text-xl font-semibold tabular-nums truncate ${mono ? 'font-mono' : ''} ${TONE_TEXT[tone] || TONE_TEXT.default}`}>
        {value}
      </div>
      {caption && <div className="text-[11px] text-ink-muted mt-0.5 truncate">{caption}</div>}
    </div>
  );
}
