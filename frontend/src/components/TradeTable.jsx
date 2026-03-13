import { useState } from 'react';

// Generates mock trade history for display (replace with API when endpoint available)
function generateMockTrades() {
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'];
  const sides = ['BUY', 'SELL'];
  const statuses = ['FILLED', 'FILLED', 'FILLED', 'PARTIAL', 'REJECTED'];
  return Array.from({ length: 20 }, (_, i) => {
    const sym = symbols[i % symbols.length];
    const side = sides[i % 2];
    const qty = Math.floor(Math.random() * 100) + 10;
    const price = (Math.random() * 800 + 100).toFixed(2);
    const ts = new Date(Date.now() - i * 1000 * 60 * (i + 1));
    return { id: `TRD-${1000 + i}`, symbol: sym, side, qty, price, status: statuses[i % statuses.length], ts };
  });
}

const MOCK_TRADES = generateMockTrades();

export default function TradeTable({ trades = MOCK_TRADES, maxRows = 15 }) {
  const [filter, setFilter] = useState('ALL');
  const FILTERS = ['ALL', 'BUY', 'SELL'];

  const filtered = trades
    .filter((t) => filter === 'ALL' || t.side === filter)
    .slice(0, maxRows);

  const STATUS_STYLE = {
    FILLED: 'text-emerald-400',
    PARTIAL: 'text-amber-400',
    REJECTED: 'text-red-400',
    PENDING: 'text-zinc-400',
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex gap-1 mb-3">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`text-[10px] font-mono uppercase tracking-widest px-2.5 py-1 rounded border transition-colors ${
              filter === f
                ? 'border-cyan-500 text-cyan-400 bg-cyan-400/5'
                : 'border-zinc-800 text-zinc-500 hover:border-zinc-600'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="overflow-auto flex-1">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-[10px] uppercase tracking-widest text-zinc-600 border-b border-zinc-800">
              <th className="text-left pb-2 pr-4">ID</th>
              <th className="text-left pb-2 pr-4">Symbol</th>
              <th className="text-left pb-2 pr-4">Side</th>
              <th className="text-right pb-2 pr-4">Qty</th>
              <th className="text-right pb-2 pr-4">Price</th>
              <th className="text-left pb-2 pr-4">Status</th>
              <th className="text-right pb-2">Time</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((t, i) => (
              <tr
                key={t.id}
                className="border-b border-zinc-900 hover:bg-zinc-800/30 transition-colors"
              >
                <td className="py-1.5 pr-4 text-zinc-600">{t.id}</td>
                <td className="py-1.5 pr-4 text-cyan-400">{t.symbol}</td>
                <td className={`py-1.5 pr-4 font-semibold ${t.side === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>
                  {t.side}
                </td>
                <td className="py-1.5 pr-4 text-right text-zinc-300 tabular-nums">{t.qty}</td>
                <td className="py-1.5 pr-4 text-right text-zinc-300 tabular-nums">${t.price}</td>
                <td className={`py-1.5 pr-4 ${STATUS_STYLE[t.status] || 'text-zinc-400'}`}>{t.status}</td>
                <td className="py-1.5 text-right text-zinc-600">
                  {new Date(t.ts).toLocaleTimeString('en-US', { hour12: false })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
