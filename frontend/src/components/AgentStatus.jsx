import { useState, useCallback } from 'react';
import { api } from '../services/api';

const AGENTS = [
  { id: 'momentum', label: 'Momentum Agent', strategy: 'momentum' },
  { id: 'mean_reversion', label: 'Mean Reversion', strategy: 'mean_reversion' },
  { id: 'risk_manager', label: 'Risk Manager', strategy: 'risk' },
  { id: 'executor', label: 'Execution Agent', strategy: 'execution' },
];

export default function AgentStatus() {
  const [agentStates, setAgentStates] = useState({});
  const [executing, setExecuting] = useState(null);

  const execute = useCallback(async (agent) => {
    setExecuting(agent.id);
    setAgentStates((s) => ({ ...s, [agent.id]: 'running' }));
    try {
      const res = await api.executeAgent({ agent_id: agent.id, strategy: agent.strategy });
      setAgentStates((s) => ({ ...s, [agent.id]: res ? 'success' : 'idle' }));
    } catch {
      setAgentStates((s) => ({ ...s, [agent.id]: 'error' }));
    } finally {
      setExecuting(null);
      setTimeout(() => setAgentStates((s) => ({ ...s, [agent.id]: 'idle' })), 3000);
    }
  }, []);

  const STATE_STYLES = {
    running: 'border-amber-400/40 bg-amber-400/5',
    success: 'border-emerald-400/40 bg-emerald-400/5',
    error: 'border-red-400/40 bg-red-400/5',
    idle: 'border-zinc-800 bg-zinc-900/40',
  };

  const DOT_STYLES = {
    running: 'bg-amber-400 animate-pulse',
    success: 'bg-emerald-400',
    error: 'bg-red-400',
    idle: 'bg-zinc-600',
  };

  return (
    <div className="space-y-2">
      {AGENTS.map((agent) => {
        const state = agentStates[agent.id] || 'idle';
        return (
          <div
            key={agent.id}
            className={`flex items-center justify-between p-3 rounded border transition-colors ${STATE_STYLES[state]}`}
          >
            <div className="flex items-center gap-2.5">
              <span className={`w-2 h-2 rounded-full shrink-0 ${DOT_STYLES[state]}`} />
              <div>
                <div className="text-xs text-zinc-200 font-medium">{agent.label}</div>
                <div className="text-[10px] text-zinc-600 font-mono uppercase tracking-wider">{state}</div>
              </div>
            </div>
            <button
              onClick={() => execute(agent)}
              disabled={executing === agent.id}
              className="text-[10px] font-mono uppercase tracking-wider px-2.5 py-1 rounded border border-zinc-700 text-zinc-400 hover:border-cyan-500 hover:text-cyan-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {executing === agent.id ? 'Running...' : 'Execute'}
            </button>
          </div>
        );
      })}
    </div>
  );
}
