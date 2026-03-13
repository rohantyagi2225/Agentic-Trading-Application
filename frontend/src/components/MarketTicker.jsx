import { useState, useEffect, useCallback } from 'react';
import { api, SYMBOLS } from '../services/api';

// Mock price changes for demo when API returns null
function mockPrice(sym) {
  const bases = { AAPL: 189.5, MSFT: 378.2, GOOGL: 141.8, AMZN: 182.4, TSLA: 248.7, META: 503.1, NVDA: 875.4 };
  const base = bases[sym] || 100;
  const change = (Math.random() - 0.48) * base * 0.012;
  return { price: base + change, change, changePct: change / base };
}

export default function MarketTicker() {
  const [prices, setPrices] = useState({});

  const fetchPrices = useCallback(async () => {
    const results = await Promise.all(
      SYMBOLS.map(async (sym) => {
        const data = await api.getMarketPrice(sym);
        return [sym, data || mockPrice(sym)];
      })
    );
    setPrices(Object.fromEntries(results));
  }, []);

  useEffect(() => {
    fetchPrices();
    const t = setInterval(fetchPrices, 6000);
    return () => clearInterval(t);
  }, [fetchPrices]);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
      {SYMBOLS.map((sym) => {
        const p = prices[sym];
        const isPos = p?.change >= 0;
        return (
          <div
            key={sym}
            className="flex flex-col p-2.5 rounded border border-zinc-800 bg-zinc-900/60 hover:border-zinc-700 transition-colors"
          >
            <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">{sym}</span>
            <span className="text-sm font-light tabular-nums text-zinc-100 mt-0.5">
              {p ? `$${Number(p.price ?? p).toFixed(2)}` : '—'}
            </span>
            {p?.change != null && (
              <span className={`text-[10px] font-mono tabular-nums ${isPos ? 'text-emerald-400' : 'text-red-400'}`}>
                {isPos ? '+' : ''}{(p.change).toFixed(2)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
