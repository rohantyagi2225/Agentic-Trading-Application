import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Signals from './pages/Signals';
import Portfolio from './pages/Portfolio';
import Agents from './pages/Agents';

const PAGES = {
  dashboard: { component: Dashboard, title: 'Dashboard' },
  signals: { component: Signals, title: 'Signals' },
  portfolio: { component: Portfolio, title: 'Portfolio' },
  agents: { component: Agents, title: 'Agents' },
};

export default function App() {
  const [page, setPage] = useState('dashboard');
  const { component: Page, title } = PAGES[page] || PAGES.dashboard;

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden">
      <Sidebar page={page} setPage={setPage} />
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="flex-shrink-0 h-12 border-b border-zinc-800/80 flex items-center px-6 gap-4">
          <h1 className="text-sm font-mono font-semibold text-zinc-200">{title}</h1>
          <div className="ml-auto flex items-center gap-3 text-xs font-mono text-zinc-500">
            <span>{new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}</span>
            <span className="text-zinc-700">|</span>
            <span className="text-cyan-400/70">Multi-Agent Trading Platform</span>
          </div>
        </header>

        {/* Page content */}
        <div className="flex-1 overflow-y-auto p-5">
          <Page />
        </div>
      </main>
    </div>
  );
}
