import { useState } from 'react';
import { useSignalStream } from '../hooks/useSignalStream';
import { SYMBOLS } from '../services/api';

const STATUS_COLORS = {
  connected: 'text-emerald-400',
  connecting: 'text-amber-400',
  disconnected: 'text-red-400',
  error: 'text-red-500',
  idle: 'text-zinc-500',
};

const SIGNAL_COLORS = {
  BUY: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  SELL: 'text-red-400 bg-red-400/10 border-red-400/30',
  HOLD: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
};

export default function SignalStream({ compact = false }) {
  const [symbol, setSymbol] = useState('AAPL');
  const { messages, status } = useSignalStream(symbol);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${status === 'connected' ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600'}`} />
          <span className={`text-xs font-mono uppercase tracking-widest ${STATUS_COLORS[status]}`}>{status}</span>
        </div>
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-mono px-2 py-1 rounded focus:outline-none focus:border-cyan-500"
        >
          {SYMBOLS.map((s) => <option key={s}>{s}</option>)}
        </select>
      </div>

      <div className="flex-1 overflow-y-auto space-y-1.5 font-mono text-xs">
        {messages.length === 0 ? (
          <div className="text-zinc-600 text-center py-8">
            {status === 'connected' ? 'Awaiting signals...' : 'Connecting to stream...'}
          </div>
        ) : (
          messages.map((msg, i) => (
            <div
              key={msg._ts + i}
              className={`flex items-center gap-2 px-2 py-1.5 rounded border ${i === 0 ? 'bg-zinc-800/80' : 'bg-transparent border-transparent'}`}
              style={{ borderColor: i === 0 ? 'rgb(63,63,70)' : 'transparent' }}
            >
              <span className="text-zinc-600 shrink-0">
                {new Date(msg._ts).toLocaleTimeString('en-US', { hour12: false })}
              </span>
              {msg.signal && (
                <span className={`px-1.5 py-0.5 rounded border text-[10px] font-bold tracking-wider ${SIGNAL_COLORS[msg.signal] || SIGNAL_COLORS.HOLD}`}>
                  {msg.signal}
                </span>
              )}
              {msg.symbol && <span className="text-cyan-400">{msg.symbol}</span>}
              {msg.price && <span className="text-zinc-300">${Number(msg.price).toFixed(2)}</span>}
              {msg.confidence && (
                <span className="text-zinc-500">{(msg.confidence * 100).toFixed(0)}%</span>
              )}
              {!msg.signal && !msg.symbol && (
                <span className="text-zinc-500 truncate">{JSON.stringify(msg).slice(0, 80)}</span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
