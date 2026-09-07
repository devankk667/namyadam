import React from 'react';

/**
 * Flat underline tabs for moving between levels of detail on a result page
 * (Analysis -> Technical Details -> Diagnostics -> History).
 */
export default function Tabs({ tabs, active, onChange }) {
  return (
    <div className="flex items-center gap-5 border-b border-line px-1 overflow-x-auto">
      {tabs.map((tab) => {
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`relative flex items-center gap-1.5 py-2.5 text-[12px] font-medium uppercase tracking-wide whitespace-nowrap transition-colors ${
              isActive ? 'text-ink-primary' : 'text-ink-muted hover:text-ink-secondary'
            }`}
          >
            {tab.label}
            {typeof tab.count === 'number' && (
              <span className={`text-[10px] font-mono px-1 rounded-sm ${isActive ? 'bg-accent/15 text-accent' : 'bg-panel2 text-ink-muted'}`}>
                {tab.count}
              </span>
            )}
            {isActive && <span className="absolute left-0 right-0 -bottom-px h-[2px] bg-accent" />}
          </button>
        );
      })}
    </div>
  );
}
