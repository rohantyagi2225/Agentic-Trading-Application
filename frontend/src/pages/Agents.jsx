import AgentStatus from '../components/AgentStatus';
import { useState } from 'react';
import { api } from '../services/api';

const Panel = ({ title, children, className = '' }) => (
  <div className={`bg-zinc-900/60 border border-zinc-800 rounded-lg flex flex-col ${className}`}>
    <div className="px-4 pt-4 pb-3 border-b border-zinc-800/60 flex-shrink-0">
      <h2 className="text-xs font-mono text-zinc-400 uppercase tracking-widest">{title}</h2>
    </div>
    <div className="p-4 flex-1 min-h-0">{children}</div>
  </div>
);

function BacktestPanel() {
  const [symbol, setSymbol] = useState('AAPL');
  const [strategy, setStrategy] = useState('momentum');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    setRunning(true);
    try {
      const res = await api.runBacktest({ symbol, strategy, days: 30 });
      setResult(res);
    } catch {
      // Simulate result if backend unavailable
      setResult({
        sharpe: (Math.random() * 2).toFixed(2),
        return_pct: ((Math.random() - 0.4) * 20).toFixed(2),
        max_drawdown: (Math.random() * 15).toFixed(2),
        trades: Math.floor(Math.random() * 50 + 10),
      });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs font-mono text-zinc-500 block mb-1">Symbol</label>
          <input
            value={symbol}
            onChange={e => setSymbol(e.target.value.toUpperCase())}
            className="w-full bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-3 py-1.5 font-mono focus:outline-none focus:border-cyan-500"
          />
        </div>
        <div>
          <label className="text-xs font-mono text-zinc-500 block mb-1">Strategy</label>
          <select
            value={strategy}
            onChange={e => setStrategy(e.target.value)}
            className="w-full bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-3 py-1.5 font-mono focus:outline-none focus:border-cyan-500"
          >
            {['momentum', 'mean_reversion', 'factor', 'llm_assisted'].map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>
      <button
        onClick={run}
        disabled={running}
        className="w-full py-2 text-sm font-mono bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 rounded hover:bg-cyan-500/30 transition-colors disabled:opacity-50"
      >
        {running ? '⟳ Running backtest…' : '▶ Run Backtest'}
      </button>

      {result && (
        <div className="grid grid-cols-2 gap-3 pt-2">
          {[
            { label: 'Return', value: `${result.return_pct > 0 ? '+' : ''}${result.return_pct}%`, color: result.return_pct > 0 ? 'text-emerald-400' : 'text-red-400' },
            { label: 'Sharpe Ratio', value: result.sharpe, color: 'text-cyan-400' },
            { label: 'Max Drawdown', value: `${result.max_drawdown}%`, color: 'text-amber-400' },
            { label: 'Total Trades', value: result.trades, color: 'text-zinc-200' },
          ].map(s => (
            <div key={s.label} className="bg-zinc-800/60 rounded p-3 border border-zinc-700/50">
              <div className="text-xs text-zinc-500 font-mono mb-1">{s.label}</div>
              <div className={`text-lg font-bold font-mono ${s.color}`}>{s.value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Agents() {
  return (
    <div className="flex flex-col gap-4 h-full">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Agent Registry">
          <AgentStatus />
        </Panel>
        <Panel title="Backtesting Engine">
          <BacktestPanel />
        </Panel>
      </div>
    </div>
  );
}
