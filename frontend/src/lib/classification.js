// Single source of truth for how a classification maps to color/label/severity
// across the map, tables, badges and detail pages.

export const CLASSIFICATIONS = {
  industrial_fire: { label: 'Industrial Fire', hex: '#f85149', status: 'critical' },
  industrial_thermal_source: { label: 'Persistent Thermal Source', hex: '#e3934a', status: 'accent' },
  wildfire: { label: 'Wildfire', hex: '#d29922', status: 'warning' },
  agricultural_burning: { label: 'Agricultural Burning', hex: '#3fb950', status: 'success' },
  other_thermal_anomaly: { label: 'Other Anomaly', hex: '#6b7684', status: 'neutral' },
};

export function classificationMeta(key) {
  return CLASSIFICATIONS[key] || { label: key ? key.replace(/_/g, ' ') : 'Unclassified', hex: '#6b7684', status: 'neutral' };
}

export function severityStatus(severity) {
  switch (severity) {
    case 'CRITICAL':
      return 'critical';
    case 'HIGH':
      return 'accent';
    case 'MEDIUM':
      return 'warning';
    default:
      return 'neutral';
  }
}

export function formatClassLabel(key) {
  return key ? key.replace(/_/g, ' ') : 'unclassified';
}
