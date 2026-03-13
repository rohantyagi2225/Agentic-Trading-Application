const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) throw new Error(`API ${res.status}`);
    return res.json();
  } catch (e) {
    console.warn(`API call failed for ${path}:`, e.message);
    return null;
  }
}

export const api = {
  getHealth: () => request('/health'),
  getSignals: (symbol) => request(`/signals/${symbol}`),
  getPortfolioMetrics: () => request('/portfolio/metrics'),
  getMarketPrice: (symbol) => request(`/market/price/${symbol}`),
  executeAgent: (payload) => request('/agents/execute', { method: 'POST', body: JSON.stringify(payload) }),
};

export const SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'];
