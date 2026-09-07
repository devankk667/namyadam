import React, { useState } from 'react';
import { ChevronRight } from 'lucide-react';

/**
 * Expandable technical section. Keeps the default view readable while
 * allowing drill-down into deeper diagnostic / raw detail.
 */
export default function Disclosure({ title, eyebrow, defaultOpen = false, children, mono = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-line bg-panel2/40">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-panel2 transition-colors"
        aria-expanded={open}
      >
        <ChevronRight
          className={`w-3.5 h-3.5 text-ink-muted shrink-0 transition-transform duration-150 ${open ? 'rotate-90' : ''}`}
          strokeWidth={2}
        />
        <span className="min-w-0">
          {eyebrow && <span className="text-[10px] font-mono uppercase tracking-wider text-ink-muted mr-2">{eyebrow}</span>}
          <span className={`text-[12px] font-medium text-ink-secondary ${mono ? 'font-mono' : ''}`}>{title}</span>
        </span>
      </button>
      {open && <div className="px-3 pb-3 pt-0.5 border-t border-line">{children}</div>}
    </div>
  );
}
