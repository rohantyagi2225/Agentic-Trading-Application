import { useCallback } from 'react';
import { api } from '../services/api';
import { usePolling } from '../hooks/usePolling';

function Metric({ label, value, sub, color = 'text-zinc-100' }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-widest text-zinc-500 font-mono">{label}</span>
      <span className={`text-xl font-light tabular-nums ${color}`}>{value}</span>
      {sub && <span className="text-xs text-zinc-600 font-mono">{sub}</span>}
    </div>
  );
}

function fmt(n, prefix = '$') {
  if (n == null) return '—';
  return prefix + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(n) {
  if (n == null) return '—';
  const val = (Number(n) * 100).toFixed(2);
  return (n >= 0 ? '+' : '') + val + '%';
}

export default function PortfolioCard() {
  const fetchMetrics = useCallback(() => api.getPortfolioMetrics(), []);
  const { data: metrics, loading } = usePolling(fetchMetrics, 8000);

  const pnl = metrics?.unrealized_pnl ?? metrics?.pnl ?? null;
  const pnlPct = metrics?.pnl_pct ?? null;
  const isPos = pnl >= 0;

  return (
    <div className="flex flex-col gap-4 h-full">
      {loading && !metrics ? (
        <div className="text-zinc-600 text-xs font-mono animate-pulse">Loading portfolio...</div>
      ) : metrics ? (
        <>
          <div className="grid grid-cols-2 gap-4">
            <Metric
              label="Portfolio Value"
              value={fmt(metrics.portfolio_value ?? metrics.total_value)}
            />
            <Metric
              label="Cash"
              value={fmt(metrics.cash)}
            />
            <Metric
              label="Unrealized P&L"
              value={fmt(pnl)}
              sub={pnlPct != null ? fmtPct(pnlPct) : undefined}
              color={pnl == null ? 'text-zinc-100' : isPos ? 'text-emerald-400' : 'text-red-400'}
            />
            <Metric
              label="Exposure"
              value={metrics.exposure != null ? fmtPct(metrics.exposure) : '—'}
            />
          </div>

          {metrics.positions && Object.keys(metrics.positions).length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-zinc-500 font-mono mb-2">Positions</div>
              <div className="space-y-1">
                {Object.entries(metrics.positions).map(([sym, pos]) => (
                  <div key={sym} className="flex justify-between items-center text-xs font-mono">
                    <span className="text-cyan-400">{sym}</span>
                    <span className="text-zinc-400">{pos.qty ?? pos.quantity ?? pos} shares</span>
                    <span className="text-zinc-300">{fmt(pos.value ?? pos.market_value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {metrics.sharpe_ratio != null && (
            <div className="grid grid-cols-3 gap-3 pt-2 border-t border-zinc-800">
              <Metric label="Sharpe" value={Number(metrics.sharpe_ratio).toFixed(2)} />
              <Metric label="Volatility" value={metrics.volatility != null ? fmtPct(metrics.volatility) : '—'} />
              <Metric label="Max DD" value={metrics.max_drawdown != null ? fmtPct(metrics.max_drawdown) : '—'} color="text-red-400" />
            </div>
          )}
        </>
      ) : (
        <div className="text-zinc-600 text-xs font-mono">No portfolio data available</div>
      )}
    </div>
  );
}
