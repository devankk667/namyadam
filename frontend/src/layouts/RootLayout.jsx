import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  Flame,
  LayoutDashboard,
  MapPin,
  BarChart3,
  Bell,
  Cpu,
  BrainCircuit,
  Database,
  Activity
} from 'lucide-react';

export default function RootLayout({ health }) {
  const isDemo = health?.data_mode === 'demo';

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Navigation Header */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-3 flex items-center justify-between sticky top-0 z-50 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-tr from-orange-600 to-amber-500 rounded-lg shadow-lg">
            <Flame className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wide text-white flex items-center gap-2">
              ThermalGuard <span className="text-xs px-2 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30">AI Pro</span>
            </h1>
            <p className="text-xs text-slate-400">Industrial Thermal Anomaly & Fire Decision Platform</p>
          </div>
        </div>

        {/* Global System Status & Indicators */}
        <div className="flex items-center space-x-4">
          {isDemo && (
            <div className="flex items-center space-x-1.5 px-3 py-1 bg-amber-500/10 border border-amber-500/30 rounded-full text-amber-400 text-xs font-medium">
              <Database className="w-3.5 h-3.5" />
              <span>DEMO DATA MODE</span>
            </div>
          )}

          <div className="flex items-center space-x-2 px-3 py-1 bg-slate-800 rounded-full border border-slate-700 text-xs">
            <Activity className={`w-3.5 h-3.5 ${health?.status === 'healthy' ? 'text-emerald-400 animate-pulse' : 'text-rose-400'}`} />
            <span className="text-slate-300">
              API: <span className="text-slate-100 font-semibold">{health?.status || 'connecting...'}</span>
            </span>
          </div>

          <div className="flex items-center space-x-2 px-3 py-1 bg-slate-800 rounded-full border border-slate-700 text-xs">
            <Cpu className={`w-3.5 h-3.5 ${health?.model_loaded ? 'text-cyan-400' : 'text-amber-400'}`} />
            <span className="text-slate-300">
              ML Model: <span className="text-slate-100 font-semibold">{health?.model_loaded ? 'ACTIVE' : 'HEURISTIC'}</span>
            </span>
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        {/* Sidebar Navigation */}
        <aside className="w-64 bg-slate-900/80 border-r border-slate-800 p-4 flex flex-col justify-between">
          <nav className="space-y-1.5">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-orange-600/10 text-orange-400 border border-orange-500/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Dashboard</span>
            </NavLink>

            <NavLink
              to="/analytics"
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-orange-600/10 text-orange-400 border border-orange-500/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <BarChart3 className="w-4 h-4" />
              <span>Analytics & Persistence</span>
            </NavLink>

            <NavLink
              to="/alerts"
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-orange-600/10 text-orange-400 border border-orange-500/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <Bell className="w-4 h-4" />
              <span>Early Warning Alerts</span>
            </NavLink>

            <NavLink
              to="/predict"
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-orange-600/10 text-orange-400 border border-orange-500/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <BrainCircuit className="w-4 h-4" />
              <span>Interactive ML Predictor</span>
            </NavLink>

            <NavLink
              to="/model"
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-orange-600/10 text-orange-400 border border-orange-500/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <Cpu className="w-4 h-4" />
              <span>ML Model Benchmarks</span>
            </NavLink>
          </nav>

          {/* Footer Metadata */}
          <div className="pt-4 border-t border-slate-800 text-xs text-slate-500 space-y-1">
            <p>System Ver: {health?.version || '1.0.0'}</p>
            <p>Data Source: NASA VIIRS & OSM</p>
          </div>
        </aside>

        {/* Main Content Viewport */}
        <main className="flex-1 overflow-y-auto p-6 bg-slate-950">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
