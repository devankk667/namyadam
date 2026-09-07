import React from 'react';

/**
 * Structured panel — the base unit of the console UI.
 * Replaces ad-hoc "rounded-2xl shadow-xl card" blocks with a flat,
 * bordered surface with an optional labeled header + action slot.
 */
export default function Panel({
  title,
  eyebrow,
  icon: Icon,
  actions,
  subtitle,
  children,
  className = '',
  bodyClassName = '',
  noPadding = false,
  as: Tag = 'section',
}) {
  return (
    <Tag className={`bg-panel border border-line ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5 min-h-[42px]">
          <div className="flex items-center gap-2 min-w-0">
            {Icon && <Icon className="w-3.5 h-3.5 text-ink-muted shrink-0" strokeWidth={1.75} />}
            <div className="min-w-0">
              {eyebrow && (
                <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted leading-none mb-0.5">
                  {eyebrow}
                </div>
              )}
              <h3 className="text-[12px] font-semibold uppercase tracking-wide text-ink-secondary truncate leading-none">
                {title}
              </h3>
              {subtitle && (
                <p className="text-[11px] text-ink-muted truncate mt-1">{subtitle}</p>
              )}
            </div>
          </div>
          {actions && <div className="flex items-center gap-1.5 shrink-0">{actions}</div>}
        </header>
      )}
      <div className={noPadding ? '' : `p-4 ${bodyClassName}`}>{children}</div>
    </Tag>
  );
}
