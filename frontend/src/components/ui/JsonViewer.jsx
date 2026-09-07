import React from 'react';
import CopyButton from './CopyButton';

/**
 * Raw structured-output viewer. Used to expose the exact payload behind a
 * summarized result so the underlying evidence is always inspectable.
 */
export default function JsonViewer({ data, title = 'Raw JSON', maxHeight = '320px' }) {
  const text = JSON.stringify(data, null, 2);
  return (
    <div className="border border-line bg-[#080a0e]">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-line bg-panel2">
        <span className="text-[10px] font-mono uppercase tracking-wider text-ink-muted">{title}</span>
        <CopyButton value={text} />
      </div>
      <pre
        className="p-3 text-[11px] font-mono text-ink-secondary overflow-auto leading-relaxed"
        style={{ maxHeight }}
      >
        {text}
      </pre>
    </div>
  );
}
