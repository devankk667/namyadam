import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

export default function CopyButton({ value, label }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(typeof value === 'string' ? value : JSON.stringify(value, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="inline-flex items-center gap-1 text-[10px] font-mono uppercase tracking-wide text-ink-muted hover:text-ink-secondary border border-line hover:border-line2 px-1.5 py-0.5 transition-colors"
      title="Copy to clipboard"
    >
      {copied ? <Check className="w-3 h-3 text-status-success" /> : <Copy className="w-3 h-3" />}
      {label || (copied ? 'Copied' : 'Copy')}
    </button>
  );
}
