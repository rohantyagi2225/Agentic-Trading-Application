import { useState, useCallback, useEffect } from 'react';
import { useSignalStream } from '../hooks/useSignalStream';
import { api, SYMBOLS } from '../services/api';

function SignalBadge({ signal }) {
  const colors = {
    BUY: 'bg-emerald-400/10 text-emerald-400 border-emerald-400/30',
    SELL: 'bg-red-400/10 text-red-400 border-red-400/30',
    HOLD: 'bg-amber-400/10 text-amber-400 border-amber-400/30',
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded border text-[10px] font-bold tracking-wider font-mono ${colors[signal] || colors.HOLD}`}>
      {signal}
    </span>
  );
}

export default function Signals() {
  const [symbol, setSymbol] = useState('AAPL');
  const [restSignals, setRestSignals] = useState(null);
  const { messages, status } = useSignalStream(symbol, 100);

  const fetchRESTSignals = useCallback(async () => {
    const data = await api.getSignals(symbol);
    setRestSignals(data);
  }, [symbol]);

  useEffect(() => { fetchRESTSignals(); }, [fetchRESTSignals]);

  const STATUS = {
    connected: { dot: 'bg-emerald-400 animate-pulse', text: 'text-emerald-400', label: 'Connected' },
    connecting: { dot: 'bg-amber-400 animate-pulse', text: 'text-amber-400', label: 'Connecting...' },
    disconnected: { dot: 'bg-red-400', text: 'text-red-400', label: 'Disconnected' },
    error: { dot: 'bg-red-500', text: 'text-red-500', label: 'Error' },
    idle: { dot: 'bg-zinc-600', text: 'text-zinc-500', label: 'Idle' },
  };
  const s = STATUS[status] || STATUS.idle;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-zinc-900/60 border border-zinc-800 rounded-lg p-4">
        <div>
          <h2 className="text-zinc-100 font-light text-xl tracking-tight">Signal Monitor</h2>
          <p className="text-zinc-500 text-xs font-mono mt-0.5">Real-time WebSocket stream + REST snapshot</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${s.dot}`} />
            <span className={`text-xs font-mono ${s.text}`}>{s.label}</span>
          </div>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm font-mono px-3 py-1.5 rounded focus:outline-none focus:border-cyan-500"
          >
            {SYMBOLS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </div>
      </div>

      {/* REST Signals */}
      {restSignals && (
        <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-4">
          <div className="text-[10px] uppercase tracking-widest text-zinc-500 font-mono mb-3">Latest Signal · REST</div>
          <div className="flex flex-wrap gap-4 font-mono text-sm">
            {restSignals.signal && <div className="flex items-center gap-2"><span className="text-zinc-500">Signal</span><SignalBadge signal={restSignals.signal} /></div>}
            {restSignals.price && <div className="flex items-center gap-2"><span className="text-zinc-500">Price</span><span className="text-zinc-200">${Number(restSignals.price).toFixed(2)}</span></div>}
            {restSignals.confidence && <div className="flex items-center gap-2"><span className="text-zinc-500">Confidence</span><span className="text-zinc-200">{(restSignals.confidence * 100).toFixed(1)}%</span></div>}
            {restSignals.agent && <div className="flex items-center gap-2"><span className="text-zinc-500">Agent</span><span className="text-zinc-200">{restSignals.agent}</span></div>}
          </div>
        </div>
      )}

      {/* WebSocket Stream */}
      <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-4" style={{ minHeight: 400 }}>
        <div className="text-[10px] uppercase tracking-widest text-zinc-500 font-mono mb-3 flex items-center justify-between">
          <span>Live Stream · WebSocket</span>
          <span className="text-zinc-600">{messages.length} messages</span>
        </div>
        <div className="overflow-y-auto" style={{ maxHeight: 480 }}>
          {messages.length === 0 ? (
            <div className="text-zinc-600 text-center py-16 font-mono text-sm">
              {status === 'connected' ? `Streaming ${symbol}... waiting for signals` : 'Establishing connection...'}
            </div>
          ) : (
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-[10px] uppercase tracking-widest text-zinc-600 border-b border-zinc-800">
                  <th className="text-left pb-2 pr-4">Time</th>
                  <th className="text-left pb-2 pr-4">Signal</th>
                  <th className="text-left pb-2 pr-4">Symbol</th>
                  <th className="text-right pb-2 pr-4">Price</th>
                  <th className="text-right pb-2">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {messages.map((msg, i) => (
                  <tr key={msg._ts + i} className="border-b border-zinc-900 hover:bg-zinc-800/20">
                    <td className="py-1.5 pr-4 text-zinc-600">
                      {new Date(msg._ts).toLocaleTimeString('en-US', { hour12: false })}
                    </td>
                    <td className="py-1.5 pr-4">
                      {msg.signal ? <SignalBadge signal={msg.signal} /> : <span className="text-zinc-600">—</span>}
                    </td>
                    <td className="py-1.5 pr-4 text-cyan-400">{msg.symbol || symbol}</td>
                    <td className="py-1.5 pr-4 text-right text-zinc-300 tabular-nums">
                      {msg.price ? `$${Number(msg.price).toFixed(2)}` : '—'}
                    </td>
                    <td className="py-1.5 text-right tabular-nums text-zinc-400">
                      {msg.confidence ? `${(msg.confidence * 100).toFixed(0)}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
