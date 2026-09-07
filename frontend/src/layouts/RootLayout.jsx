import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutGrid,
  BarChart3,
  Bell,
  Cpu,
  BrainCircuit,
  Radio,
} from 'lucide-react';
import StatusBadge from '../components/ui/StatusBadge';

const NAV_ITEMS = [
  { to: '/', end: true, label: 'Monitoring Console', icon: LayoutGrid },
  { to: '/analytics', label: 'Analytics & Persistence', icon: BarChart3 },
  { to: '/alerts', label: 'Early Warning Rules', icon: Bell },
  { to: '/predict', label: 'Inference Console', icon: BrainCircuit },
  { to: '/model', label: 'Model Registry', icon: Cpu },
];

export default function RootLayout({ health }) {
  const isDemo = health?.data_mode === 'demo';
  const apiOk = health?.status === 'healthy';

  return (
    <div className="min-h-screen flex flex-col bg-base text-ink-primary font-sans text-sm">
      {/* Top bar */}
      <header className="h-12 shrink-0 bg-panel border-b border-line px-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-accent shrink-0" />
            <span className="font-mono font-semibold text-[13px] tracking-tight text-ink-primary">
              THERMALGUARD
            </span>
          </div>
          <span className="text-line2 select-none">/</span>
          <span className="text-[11px] text-ink-muted truncate hidden sm:inline">
            Industrial Thermal Anomaly & Fire Decision Platform
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {isDemo && <StatusBadge status="warning" size="xs">demo data</StatusBadge>}
          <StatusBadge status={apiOk ? 'success' : 'critical'} dot pulse={apiOk} size="xs">
            api: {health?.status || 'connecting'}
          </StatusBadge>
          <StatusBadge status={health?.model_loaded ? 'info' : 'warning'} size="xs">
            model: {health?.model_loaded ? 'active' : 'heuristic'}
          </StatusBadge>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <aside className="w-56 shrink-0 bg-panel border-r border-line flex flex-col justify-between">
          <nav className="py-3">
            <div className="px-3 pb-2 text-[10px] font-mono uppercase tracking-wider text-ink-muted">
              Navigation
            </div>
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 pl-3 pr-3 py-2 text-[12px] font-medium border-l-2 transition-colors ${
                    isActive
                      ? 'border-l-accent bg-accent/[0.06] text-ink-primary'
                      : 'border-l-transparent text-ink-muted hover:text-ink-secondary hover:bg-panel2'
                  }`
                }
              >
                <item.icon className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} />
                <span className="truncate">{item.label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="border-t border-line px-3 py-3 space-y-1.5">
            <div className="text-[10px] font-mono uppercase tracking-wider text-ink-muted mb-1.5 flex items-center gap-1.5">
              <Radio className="w-3 h-3" />
              System
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-ink-muted">version</span>
              <span className="text-ink-secondary">{health?.version || '1.0.0'}</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-ink-muted">source</span>
              <span className="text-ink-secondary">VIIRS/OSM</span>
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 min-w-0 overflow-y-auto bg-base">
          <div className="p-6 max-w-[1600px] mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
