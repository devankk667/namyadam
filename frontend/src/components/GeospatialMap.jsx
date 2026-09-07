import React, { useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { useNavigate } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import { classificationMeta, formatClassLabel, CLASSIFICATIONS } from '../lib/classification';

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

export default function GeospatialMap({ detections = [], selectedDetection, onSelectDetection, height = '550px' }) {
  const navigate = useNavigate();

  const getMarkerRadius = (frp, isSelected) => {
    const base = Math.min(Math.max(frp / 10, 6), 18);
    return isSelected ? base + 4 : base;
  };

  return (
    <div className="relative w-full border border-line" style={{ height }}>
      <MapContainer
        center={[25.0, 10.0]}
        zoom={2}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapRecenter selectedDetection={selectedDetection} />

        {detections.map((det) => {
          const meta = classificationMeta(det.true_class);
          const isSelected = selectedDetection?.id === det.id;

          return (
            <CircleMarker
              key={det.id}
              center={[det.latitude, det.longitude]}
              radius={getMarkerRadius(det.frp, isSelected)}
              pathOptions={{
                color: isSelected ? '#e6e9ef' : meta.hex,
                fillColor: meta.hex,
                fillOpacity: isSelected ? 0.95 : 0.65,
                weight: isSelected ? 2.5 : 1.25
              }}
              eventHandlers={{
                click: () => onSelectDetection && onSelectDetection(det)
              }}
            >
              <Popup className="thermal-popup">
                <div className="text-[12px] font-sans">
                  <div className="flex items-center justify-between border-b border-line px-3 py-2">
                    <span className="font-mono font-semibold text-ink-primary text-[11px]">{det.id}</span>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 border border-line2 text-ink-secondary">
                      {formatClassLabel(det.true_class)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 px-3 py-2 text-[11px]">
                    <div className="text-ink-muted">lat/lon</div>
                    <div className="font-mono text-ink-secondary text-right">{det.latitude.toFixed(2)}, {det.longitude.toFixed(2)}</div>
                    <div className="text-ink-muted">FRP</div>
                    <div className="font-mono text-accent text-right font-semibold">{det.frp} MW</div>
                    <div className="text-ink-muted">confidence</div>
                    <div className="font-mono text-ink-secondary text-right">{det.confidence}%</div>
                    <div className="text-ink-muted">nearest facility</div>
                    <div className="font-mono text-ink-secondary text-right">{det.dist_to_industrial} km</div>
                  </div>

                  <button
                    onClick={() => navigate(`/detections/${det.id}`)}
                    className="w-full flex items-center justify-center gap-1.5 border-t border-line px-3 py-2 text-[11px] font-medium text-accent hover:bg-panel2 transition-colors"
                  >
                    Inspect event
                    <ArrowUpRight className="w-3 h-3" />
                  </button>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 z-[1000] bg-panel/95 border border-line px-3 py-2.5 text-[11px] space-y-1.5 min-w-[190px]">
        <div className="font-mono uppercase tracking-wider text-ink-muted text-[10px] border-b border-line pb-1.5 mb-1">
          Classification
        </div>
        {Object.entries(CLASSIFICATIONS).map(([key, meta]) => (
          <div key={key} className="flex items-center gap-2">
            <span className="w-2 h-2 shrink-0" style={{ backgroundColor: meta.hex }} />
            <span className="text-ink-secondary">{meta.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
