import React, { useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { useNavigate } from 'react-router-dom';
import { ExternalLink, Flame } from 'lucide-react';

const CLASS_COLORS = {
  industrial_fire: '#ef4444',            // Red
  industrial_thermal_source: '#f97316',   // Orange
  wildfire: '#eab308',                    // Yellow
  agricultural_burning: '#10b981',        // Green
  other_thermal_anomaly: '#6b7280'        // Gray
};

function MapRecenter({ selectedDetection }) {
  const map = useMap();
  useEffect(() => {
    if (selectedDetection) {
      map.flyTo([selectedDetection.latitude, selectedDetection.longitude], 8, {
        duration: 1.5
      });
    }
  }, [selectedDetection, map]);
  return null;
}

export default function GeospatialMap({ detections = [], selectedDetection, onSelectDetection }) {
  const navigate = useNavigate();

  const getMarkerRadius = (frp, isSelected) => {
    const base = Math.min(Math.max(frp / 10, 6), 18);
    return isSelected ? base + 4 : base;
  };

  return (
    <div className="relative w-full h-[550px] rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
      <MapContainer
        center={[25.0, 10.0]}
        zoom={2}
        scrollWheelZoom={true}
        className="w-full h-full bg-slate-950"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapRecenter selectedDetection={selectedDetection} />

        {detections.map((det) => {
          const color = CLASS_COLORS[det.true_class] || CLASS_COLORS.other_thermal_anomaly;
          const isSelected = selectedDetection?.id === det.id;

          return (
            <CircleMarker
              key={det.id}
              center={[det.latitude, det.longitude]}
              radius={getMarkerRadius(det.frp, isSelected)}
              pathOptions={{
                color: isSelected ? '#ffffff' : color,
                fillColor: color,
                fillOpacity: isSelected ? 0.95 : 0.7,
                weight: isSelected ? 3 : 1.5
              }}
              eventHandlers={{
                click: () => onSelectDetection && onSelectDetection(det)
              }}
            >
              <Popup className="custom-leaflet-popup">
                <div className="p-1 bg-slate-900 text-slate-100 rounded text-xs space-y-2 min-w-[200px]">
                  <div className="flex items-center justify-between border-b border-slate-700 pb-1">
                    <span className="font-bold text-orange-400">{det.id}</span>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                      {det.true_class ? det.true_class.replace(/_/g, ' ') : 'Anomaly'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-300">
                    <div><span className="text-slate-500">Lat/Lon:</span> {det.latitude.toFixed(2)}, {det.longitude.toFixed(2)}</div>
                    <div><span className="text-slate-500">FRP:</span> <strong className="text-amber-400">{det.frp} MW</strong></div>
                    <div><span className="text-slate-500">Confidence:</span> {det.confidence}%</div>
                    <div><span className="text-slate-500">Near Facility:</span> {det.dist_to_industrial} km ({det.nearest_facility_type})</div>
                  </div>

                  <button
                    onClick={() => navigate(`/detections/${det.id}`)}
                    className="w-full mt-2 py-1 px-2 bg-orange-600 hover:bg-orange-500 text-white font-medium rounded flex items-center justify-center gap-1 transition text-xs"
                  >
                    <span>Inspect Event Details</span>
                    <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* Floating Interactive Map Legend */}
      <div className="absolute bottom-4 left-4 z-[1000] bg-slate-900/90 backdrop-blur-md p-3 rounded-lg border border-slate-800 shadow-xl text-xs space-y-1.5 min-w-[210px]">
        <div className="font-semibold text-slate-200 border-b border-slate-800 pb-1 flex items-center gap-1.5">
          <Flame className="w-3.5 h-3.5 text-orange-400" />
          <span>Thermal Classifications</span>
        </div>
        <div className="space-y-1 pt-0.5">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500 inline-block shadow-sm"></span>
            <span className="text-slate-300">Industrial Fire</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-orange-500 inline-block shadow-sm"></span>
            <span className="text-slate-300">Persistent Thermal Source</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-yellow-500 inline-block shadow-sm"></span>
            <span className="text-slate-300">Wildfire</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block shadow-sm"></span>
            <span className="text-slate-300">Agricultural Burning</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-gray-500 inline-block shadow-sm"></span>
            <span className="text-slate-300">Other Anomaly</span>
          </div>
        </div>
      </div>
    </div>
  );
}
