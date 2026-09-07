import React from 'react';
import { Info } from 'lucide-react';

/**
 * Minimal hover affordance for explaining unfamiliar metrics/terms inline.
 * Pure CSS group-hover — no JS state, no portal.
 */
export default function Tooltip({ content, children }) {
  return (
    <span className="relative inline-flex group align-middle">
      {children || <Info className="w-3 h-3 text-ink-muted hover:text-ink-secondary cursor-help" strokeWidth={2} />}
      <span
        role="tooltip"
        className="pointer-events-none absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-1.5
        w-max max-w-[240px] rounded-sm border border-line2 bg-panel2 px-2.5 py-1.5
        text-[11px] font-sans font-normal normal-case leading-snug text-ink-secondary
        opacity-0 shadow-overlay transition-opacity duration-100 group-hover:opacity-100"
      >
        {content}
      </span>
    </span>
  );
}
