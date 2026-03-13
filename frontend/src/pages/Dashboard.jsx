import MarketTicker from '../components/MarketTicker';
import SignalStream from '../components/SignalStream';
import PortfolioCard from '../components/PortfolioCard';
import AgentStatus from '../components/AgentStatus';
import AnalyticsChart from '../components/AnalyticsChart';
import TradeTable from '../components/TradeTable';

function Panel({ title, children, className = '' }) {
  return (
    <div className={`bg-zinc-900/60 border border-zinc-800 rounded-lg p-4 flex flex-col ${className}`}>
      {title && (
        <div className="text-[10px] uppercase tracking-widest text-zinc-500 font-mono mb-3 pb-3 border-b border-zinc-800">
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export default function Dashboard() {
  return (
    <div className="space-y-4">
      {/* Market Ticker Bar */}
      <Panel title="Market Prices · Live">
        <MarketTicker />
      </Panel>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-4">
        {/* Signal Stream */}
        <Panel title="Signal Stream · WebSocket" className="col-span-12 lg:col-span-4" style={{ minHeight: 320 }}>
          <div className="flex-1 overflow-hidden" style={{ height: 300 }}>
            <SignalStream />
          </div>
        </Panel>

        {/* Portfolio */}
        <Panel title="Portfolio · Overview" className="col-span-12 lg:col-span-4">
          <PortfolioCard />
        </Panel>

        {/* Agents */}
        <Panel title="Agent Status" className="col-span-12 lg:col-span-4">
          <AgentStatus />
        </Panel>
      </div>

      {/* Analytics */}
      <Panel title="Analytics · Equity Curve">
        <AnalyticsChart />
      </Panel>

      {/* Trade History */}
      <Panel title="Trade History · Recent" className="flex-col" style={{ minHeight: 280 }}>
        <div style={{ height: 260 }}>
          <TradeTable maxRows={8} />
        </div>
      </Panel>
    </div>
  );
}
