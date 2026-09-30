const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

async function fetchJson(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const { headers, ...requestOptions } = options;
    const res = await fetch(url, {
      ...requestOptions,
      headers: {
        Accept: 'application/json',
        ...(requestOptions.body ? { 'Content-Type': 'application/json' } : {}),
        ...headers,
      },
    });
    if (!res.ok) {
      const responseBody = await res.text();
      let detail = responseBody || res.statusText;
      try {
        const body = JSON.parse(responseBody);
        detail = body.detail || body.message || detail;
        if (typeof detail !== 'string') detail = JSON.stringify(detail);
      } catch {
        // Preserve non-JSON proxy or server error text.
      }
      throw new Error(`API Error ${res.status}: ${detail}`);
    }
    if (res.status === 204) return null;
    const responseBody = await res.text();
    return responseBody ? JSON.parse(responseBody) : null;
  } catch (error) {
    const requestError = error instanceof TypeError
      ? new Error(`Unable to reach API at ${url}. Check that the backend is running and the Vite proxy target is correct.`)
      : error;
    console.error(`Fetch error at ${endpoint}:`, requestError);
    throw requestError;
  }
}

export const api = {
  getHealth: () => fetchJson('/health'),
  getIngestionStatus: () => fetchJson('/ingestion/status'),

  getDetections: (params = {}) => {
    const query = new URLSearchParams();
    if (params.min_confidence !== undefined && params.min_confidence !== null && params.min_confidence !== '') query.append('min_confidence', params.min_confidence);
    if (params.min_frp !== undefined && params.min_frp !== null && params.min_frp !== '') query.append('min_frp', params.min_frp);
    if (params.classification) query.append('classification', params.classification);
    if (params.region) query.append('region', params.region);
    if (params.page !== undefined && params.page !== null && params.page !== '') query.append('page', params.page);
    if (params.page_size !== undefined && params.page_size !== null && params.page_size !== '') query.append('page_size', params.page_size);
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return fetchJson(`/detections${queryString}`);
  },

  getDetectionDetail: (id) => fetchJson(`/detections/${encodeURIComponent(id)}`),

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
  getFusionStatus: () => fetchJson('/fusion/status'),
  getAvailableModels: () => fetchJson('/models/available'),
  switchModel: (modelName) => fetchJson(`/models/switch/${encodeURIComponent(modelName)}`, {
    method: 'POST',
  }),
};
