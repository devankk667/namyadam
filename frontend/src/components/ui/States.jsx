import React from 'react';
import { Loader2, AlertTriangle, Inbox } from 'lucide-react';

/** Compact, log-style loading indicator — not a full-screen spinner-and-prose block. */
export function LoadingState({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-2.5 px-4 py-8 justify-center text-ink-muted">
      <Loader2 className="w-3.5 h-3.5 animate-spin" />
      <span className="text-[12px] font-mono">{label}</span>
    </div>
  );
}

export function ErrorState({ title = 'Request failed', message, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center border border-status-critical/30 bg-status-critical/5">
      <AlertTriangle className="w-5 h-5 text-status-critical" strokeWidth={1.75} />
      <div className="text-[13px] font-semibold text-ink-primary">{title}</div>
      {message && <div className="text-[11px] font-mono text-ink-muted max-w-md">{message}</div>}
      {action}
    </div>
  );
}

export function EmptyState({ message = 'No records match the current filters.' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center text-ink-muted">
      <Inbox className="w-5 h-5" strokeWidth={1.5} />
      <div className="text-[12px]">{message}</div>
    </div>
  );
}
