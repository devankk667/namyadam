import React from 'react';

/**
 * Consistent page header: title + status + contextual metadata + actions.
 * Replaces the large gradient "hero banner" pattern.
 */
export default function PageHeader({ title, description, meta = [], badges, actions }) {
  return (
    <div className="border-b border-line pb-4 flex flex-col lg:flex-row lg:items-end justify-between gap-4">
      <div className="min-w-0">
        <div className="flex items-center gap-2.5 flex-wrap">
          <h1 className="text-[15px] font-semibold text-ink-primary tracking-tight uppercase">{title}</h1>
          {badges}
        </div>
        {description && <p className="text-[12px] text-ink-muted mt-1 max-w-2xl">{description}</p>}
        {meta.length > 0 && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-[11px] font-mono text-ink-muted">
            {meta.map((m, i) => (
              <span key={i} className="flex items-center gap-1.5">
                <span className="text-ink-muted/70">{m.label}</span>
                <span className="text-ink-secondary">{m.value}</span>
              </span>
            ))}
          </div>
        )}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}
