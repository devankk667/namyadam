import React from 'react';

const TONE_BAR = {
  default: 'bg-ink-muted',
  accent: 'bg-accent',
  success: 'bg-status-success',
  warning: 'bg-status-warning',
  critical: 'bg-status-critical',
  info: 'bg-status-info',
};

/**
 * Thin labeled bar for probability / importance / score distributions.
 * Used for class probabilities, feature importances, and persistence scores.
 */
export default function ProbabilityBar({ label, value, highlighted = false, tone }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const resolvedTone = tone || (highlighted ? 'accent' : 'default');
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px]">
        <span className={`font-mono truncate ${highlighted ? 'text-ink-primary font-semibold' : 'text-ink-secondary'}`}>
          {label}
        </span>
        <span className={`font-mono tabular-nums ${highlighted ? 'text-ink-primary font-semibold' : 'text-ink-muted'}`}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="w-full h-1.5 bg-panel2 border border-line overflow-hidden">
        <div
          className={`h-full ${TONE_BAR[resolvedTone] || TONE_BAR.default}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
