import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import GeospatialMap from '../components/GeospatialMap';
import {
  Flame,
  AlertTriangle,
  Building2,
  Activity,
  Filter,
  Eye,
  RefreshCw,
  Search,
  ShieldAlert,
  ChevronRight,
  SlidersHorizontal
} from 'lucide-react';

export default function DashboardPage() {
  const navigate = useNavigate();

  const [summary, setSummary] = useState(null);
  const [detections, setDetections] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [loading, setLoading] = useState(true);

  // Filter States
  const [minConf, setMinConf] = useState('');
  const [minFrp, setMinFrp] = useState('');
  const [selectedClass, setSelectedClass] = useState('');
  const [searchRegion, setSearchRegion] = useState('');

  const loadData = async () => {
    setLoading(true);
    try {
      const [sumRes, detRes, altRes] = await Promise.all([
        api.getAnalyticsSummary(),
        api.getDetections({
          min_confidence: minConf || undefined,
          min_frp: minFrp || undefined,
          classification: selectedClass || undefined,
          region: searchRegion || undefined,
          page_size: 100
        }),
        api.getAlerts()
      ]);

      setSummary(sumRes);
      setDetections(detRes.items || []);
      setAlerts(altRes || []);
      if (detRes.items && detRes.items.length > 0) {
        setSelectedDetection(detRes.items[0]);
      }
    } catch (err) {
      console.error("Dashboard error loading data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [minConf, minFrp, selectedClass, searchRegion]);

  return (
    <div className="space-y-6">
      {/* Top Banner / Hero Overview */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
        <div>
          <h2 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <span>Geospatial Thermal Monitoring Hub</span>
            <span className="text-xs px-2.5 py-1 bg-orange-500/20 text-orange-400 rounded-full border border-orange-500/30 font-semibold">
              LIVE SATELLITE FEED
            </span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Real-time thermal anomaly identification, industrial facility spatial enrichment, and risk analysis.
          </p>
        </div>

        <button
          onClick={loadData}
          className="flex items-center space-x-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-sm font-medium transition"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-orange-400' : ''}`} />
          <span>Refresh Data</span>
        </button>
      </div>

      {/* Analytics Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg">
          <div className="text-xs text-slate-400 font-medium">Total Detections</div>
          <div className="text-2xl font-black text-white mt-1">{summary?.total_detections ?? '-'}</div>
          <div className="text-[11px] text-slate-500 mt-1">NASA VIIRS / MODIS</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg border-l-4 border-l-red-500">
          <div className="text-xs text-slate-400 font-medium">High Risk Events</div>
          <div className="text-2xl font-black text-red-400 mt-1">{summary?.high_risk_events ?? '-'}</div>
          <div className="text-[11px] text-red-500/80 mt-1">FRP &gt; 50 MW or Fire</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg border-l-4 border-l-orange-500">
          <div className="text-xs text-slate-400 font-medium">Industrial Associated</div>
          <div className="text-2xl font-black text-orange-400 mt-1">{summary?.industrial_associated_detections ?? '-'}</div>
          <div className="text-[11px] text-orange-400/80 mt-1">Near Refinery/Flare</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg border-l-4 border-l-amber-500">
          <div className="text-xs text-slate-400 font-medium">Persistent Sources</div>
          <div className="text-2xl font-black text-amber-400 mt-1">{summary?.persistent_sources ?? '-'}</div>
          <div className="text-[11px] text-amber-400/80 mt-1">30-day recurring</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg border-l-4 border-l-cyan-500">
          <div className="text-xs text-slate-400 font-medium">Active Flare Stacks</div>
          <div className="text-2xl font-black text-cyan-400 mt-1">{summary?.active_thermal_sources ?? '-'}</div>
          <div className="text-[11px] text-cyan-400/80 mt-1">Operational flares</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg border-l-4 border-l-emerald-500">
          <div className="text-xs text-slate-400 font-medium">High Confidence</div>
          <div className="text-2xl font-black text-emerald-400 mt-1">{summary?.high_confidence_events ?? '-'}</div>
          <div className="text-[11px] text-emerald-400/80 mt-1">&ge; 90% confidence</div>
        </div>
      </div>

      {/* Main Map + Early Warning Sidebar Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Interactive Geospatial Map Component & Filters */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2">
              <SlidersHorizontal className="w-5 h-5 text-orange-400" />
              <h3 className="font-bold text-white text-base">Interactive Thermal Map</h3>
            </div>

            {/* Quick Map Filters */}
            <div className="flex flex-wrap items-center gap-2">
              <select
                value={selectedClass}
                onChange={(e) => setSelectedClass(e.target.value)}
                className="bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-orange-500"
              >
                <option value="">All Classifications</option>
                <option value="industrial_fire">Industrial Fire</option>
                <option value="industrial_thermal_source">Persistent Source</option>
                <option value="wildfire">Wildfire</option>
                <option value="agricultural_burning">Agricultural</option>
                <option value="other_thermal_anomaly">Other Anomaly</option>
              </select>

              <select
                value={minConf}
                onChange={(e) => setMinConf(e.target.value)}
                className="bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-orange-500"
              >
                <option value="">Min Confidence</option>
                <option value="70">70% +</option>
                <option value="85">85% +</option>
                <option value="95">95% +</option>
              </select>

              <input
                type="text"
                placeholder="Search Region..."
                value={searchRegion}
                onChange={(e) => setSearchRegion(e.target.value)}
                className="bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-orange-500 w-32"
              />
            </div>
          </div>

          <GeospatialMap
            detections={detections}
            selectedDetection={selectedDetection}
            onSelectDetection={(det) => setSelectedDetection(det)}
          />
        </div>

        {/* Early Warning Alerts & Quick Inspection Sidebar */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <ShieldAlert className="w-5 h-5 text-red-500 animate-pulse" />
                <h3 className="font-bold text-white text-base">Active Early Warnings</h3>
              </div>
              <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 font-semibold border border-red-500/30">
                {alerts.length} ALERTS
              </span>
            </div>

            <div className="space-y-3 mt-4 max-h-[460px] overflow-y-auto pr-1">
              {alerts.slice(0, 5).map((alt) => (
                <div
                  key={alt.id}
                  className="bg-slate-800/80 border border-slate-700/80 p-3.5 rounded-xl hover:border-orange-500/50 transition cursor-pointer space-y-1.5"
                  onClick={() => navigate(`/detections/${alt.detection_id}`)}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className={`px-2 py-0.5 rounded font-mono font-bold text-[10px] ${
                      alt.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                      alt.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/40' :
                      'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                    }`}>
                      {alt.severity}
                    </span>
                    <span className="text-slate-400 text-[11px]">{alt.timestamp}</span>
                  </div>

                  <p className="text-xs font-medium text-slate-200 line-clamp-2">{alt.explanation}</p>

                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-700/50">
                    <span>{alt.region}</span>
                    <span className="text-orange-400 hover:underline flex items-center gap-0.5">
                      View details <ChevronRight className="w-3 h-3" />
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={() => navigate('/alerts')}
            className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-semibold rounded-xl flex items-center justify-center gap-2 transition"
          >
            <span>View All {alerts.length} Early Warnings</span>
            <ChevronRight className="w-4 h-4 text-orange-400" />
          </button>
        </div>
      </div>

      {/* Detection Data Table */}
      <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <Flame className="w-5 h-5 text-orange-400" />
            <h3 className="font-bold text-white text-base">Observed Thermal Anomalies</h3>
          </div>
          <span className="text-xs text-slate-400">Showing {detections.length} detections</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-800/80 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-700">
              <tr>
                <th className="p-3">ID / Date</th>
                <th className="p-3">Region</th>
                <th className="p-3">Classification</th>
                <th className="p-3">FRP</th>
                <th className="p-3">Confidence</th>
                <th className="p-3">Nearest Facility</th>
                <th className="p-3">Persistence</th>
                <th className="p-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {detections.slice(0, 15).map((det) => (
                <tr
                  key={det.id}
                  className={`hover:bg-slate-800/50 transition ${selectedDetection?.id === det.id ? 'bg-slate-800/80 border-l-2 border-orange-500' : ''}`}
                >
                  <td className="p-3">
                    <div className="font-bold text-slate-100">{det.id}</div>
                    <div className="text-[10px] text-slate-400">{det.acq_date} {det.acq_time}</div>
                  </td>
                  <td className="p-3 font-medium text-slate-200">{det.region}</td>
                  <td className="p-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                      det.true_class === 'industrial_fire' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                      det.true_class === 'industrial_thermal_source' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                      det.true_class === 'wildfire' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                      'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    }`}>
                      {det.true_class ? det.true_class.replace(/_/g, ' ') : 'anomaly'}
                    </span>
                  </td>
                  <td className="p-3 font-bold text-amber-400">{det.frp} MW</td>
                  <td className="p-3">{det.confidence}%</td>
                  <td className="p-3 text-slate-300">
                    {det.dist_to_industrial} km ({det.nearest_facility_type})
                  </td>
                  <td className="p-3">
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-orange-500 h-1.5 rounded-full"
                          style={{ width: `${det.persistence_score * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-[10px] text-slate-400">{(det.persistence_score * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="p-3 text-right">
                    <button
                      onClick={() => navigate(`/detections/${det.id}`)}
                      className="px-2.5 py-1 bg-slate-800 hover:bg-orange-600 text-slate-200 hover:text-white rounded border border-slate-700 transition flex items-center gap-1 ml-auto text-xs"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      <span>Inspect</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
