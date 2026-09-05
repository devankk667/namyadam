const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

async function fetchJson(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    });
    if (!res.ok) {
      throw new Error(`API Error ${res.status}: ${res.statusText}`);
    }
    return await res.json();
  } catch (error) {
    console.error(`Fetch error at ${endpoint}:`, error);
    throw error;
  }
}

export const api = {
  getHealth: () => fetchJson('/health'),
  getDetections: (params = {}) => {
    const query = new URLSearchParams();
    if (params.min_confidence) query.append('min_confidence', params.min_confidence);
    if (params.min_frp) query.append('min_frp', params.min_frp);
    if (params.classification) query.append('classification', params.classification);
    if (params.region) query.append('region', params.region);
    if (params.page) query.append('page', params.page);
    if (params.page_size) query.append('page_size', params.page_size);
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return fetchJson(`/detections${queryString}`);
  },
  getDetectionDetail: (id) => fetchJson(`/detections/${id}`),
  predict: (inputData) => fetchJson('/predictions', {
    method: 'POST',
    body: JSON.stringify(inputData),
  }),
  getAnalyticsSummary: () => fetchJson('/analytics/summary'),
  getAnalyticsTemporal: () => fetchJson('/analytics/temporal'),
  getAnalyticsClassification: () => fetchJson('/analytics/classification'),
  getAnalyticsRegions: () => fetchJson('/analytics/regions'),
  getAnalyticsPersistence: () => fetchJson('/analytics/persistence'),
  getAlerts: () => fetchJson('/alerts'),
  getModels: () => fetchJson('/models'),
};
