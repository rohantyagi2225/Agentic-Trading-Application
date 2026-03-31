/**
 * PRODUCT TRANSFORMATION COMPLETE - INTEGRATION GUIDE
 * 
 * This file shows how to integrate all new educational and demo features
 */

// ============================================================================
// 1. TRADE EXPLANATION COMPONENT USAGE
// ============================================================================

// In your Dashboard or Portfolio page, when displaying a trade:

import TradeExplanation from '../components/educational/TradeExplanation';
import AIDecisionExplorer from '../components/educational/AIDecisionExplorer';
import demoData from '../data/demoData';

function TradeDetailPanel({ trade }) {
  // Prepare signal data for explanation
  const signalData = {
    contributing_signals: [
      { source: 'technical', type: 'RSI', strength: 0.75 },
      { source: 'sentiment', type: 'News Sentiment', strength: 0.68 },
      { source: 'macro', type: 'Economic Trend', strength: 0.72 }
    ]
  };
  
  return (
    <div className="space-y-6">
      {/* Traditional trade details */}
      <div>
        <h3>Trade Details</h3>
        {/* ... existing trade info ... */}
      </div>
      
      {/* NEW: Educational explanation */}
      <TradeExplanation 
        trade={trade}
        signalData={signalData}
      />
      
      {/* NEW: Interactive AI decision explorer */}
      <AIDecisionExplorer
        decisionData={{
          market_data: { symbol: trade.symbol, price: trade.entry_price },
          regime: { primary_regime: 'bull_market', confidence: 0.82 },
          signals: signalData.contributing_signals,
          fusion: { overall_confidence: 0.78 },
          risk: { circuit_breaker_status: 'active' },
          decision: { action: 'BUY', confidence: 0.75 }
        }}
      />
    </div>
  );
}

// ============================================================================
// 2. DEMO MODE INTEGRATION
// ============================================================================

// In your main App or Dashboard component:

import { useState } from 'react';
import demoData from '../data/demoData';

function Dashboard() {
  const [isDemoMode, setIsDemoMode] = useState(true);
  
  // Use demo data when in demo mode
  const portfolio = isDemoMode ? demoData.portfolio : realPortfolioData;
  const signals = isDemoMode ? demoData.signals : realTimeSignals;
  
  return (
    <div>
      {/* Demo mode banner */}
      {isDemoMode && (
        <div className="bg-blue-900/30 border border-blue-800 p-4 mb-6">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎮</span>
            <div>
              <strong>Demo Mode Active</strong>
              <p className="text-sm text-zinc-400">
                You're viewing sample data. Connect your account to see live trades.
              </p>
            </div>
            <button onClick={() => setIsDemoMode(false)}>
              Switch to Live Data
            </button>
          </div>
        </div>
      )}
      
      {/* Your existing dashboard content with demo data */}
      <PortfolioCard portfolio={portfolio} />
      <SignalStream signals={signals} />
    </div>
  );
}

// ============================================================================
// 3. EDUCATIONAL REGIME DETECTION SECTION
// ============================================================================

// Add this to your Learning Lab page:

function RegimeDetectionExplainer() {
  const regimes = demoData.educationalContent.regimes;
  
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">🌍 Understanding Market Regimes</h2>
      
      <div className="grid md:grid-cols-2 gap-6">
        {Object.entries(regimes).map(([key, regime]) => (
          <div key={key} className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-3">
            <h3 className="text-xl font-semibold">{regime.title}</h3>
            <p className="text-zinc-400">{regime.description}</p>
            
            <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-lg">💡</span>
                <strong>Think of it like:</strong>
              </div>
              <p className="text-sm text-zinc-300 italic">{regime.analogy}</p>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">Strategy Tips:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-zinc-400">
                {regime.tips.map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// 4. INDICATOR EXPLANATIONS
// ============================================================================

function TechnicalIndicatorsGuide() {
  const indicators = demoData.educationalContent.indicators;
  
  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold">📊 Technical Indicators Explained</h2>
      
      {Object.entries(indicators).map(([key, indicator]) => (
        <div key={key} className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
          <h3 className="text-xl font-semibold mb-4">{indicator.name}</h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-1">What It Is:</h4>
              <p className="text-zinc-400">{indicator.whatItIs}</p>
            </div>
            
            <div>
              <h4 className="font-medium mb-1">How to Use It:</h4>
              <ul className="list-disc list-inside space-y-1 text-zinc-400">
                {indicator.howToUse.map((point, i) => (
                  <li key={i}>{point}</li>
                ))}
              </ul>
            </div>
            
            <div className="bg-red-900/20 border border-red-800/50 rounded-lg p-4">
              <h4 className="font-medium text-red-400 mb-1">⚠️ Common Mistakes:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-zinc-400">
                {indicator.commonMistakes.map((mistake, i) => (
                  <li key={i}>{mistake}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// 5. PORTFOLIO SIMULATION WITH VIRTUAL BALANCE
// ============================================================================

function PortfolioSimulation() {
  const demo = demoData.portfolio;
  
  return (
    <div className="space-y-6">
      {/* Virtual Balance Card */}
      <div className="bg-gradient-to-br from-blue-900/50 to-purple-900/50 border border-zinc-800 rounded-xl p-6">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-2xl">💰</span>
          <h3 className="text-lg font-semibold">Paper Trading Account</h3>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className="text-xs text-zinc-400 mb-1">Total Value</div>
            <div className="text-2xl font-mono">${demo.balance.totalValue.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-xs text-zinc-400 mb-1">Cash</div>
            <div className="text-xl font-mono">${demo.balance.cash.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-xs text-zinc-400 mb-1">Invested</div>
            <div className="text-xl font-mono">${demo.balance.invested.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-xs text-zinc-400 mb-1">Day Change</div>
            <div className={`text-xl font-mono ${demo.balance.dayChange >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {demo.balance.dayChange >= 0 ? '+' : ''}{demo.balance.dayChangePct.toFixed(2)}%
            </div>
          </div>
        </div>
      </div>
      
      {/* Positions */}
      <div>
        <h3 className="font-semibold mb-4">Current Positions</h3>
        <div className="space-y-3">
          {demo.positions.map((pos, idx) => (
            <div key={idx} className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex justify-between items-center">
              <div>
                <div className="font-semibold">{pos.symbol}</div>
                <div className="text-sm text-zinc-400">{pos.quantity} shares @ ${pos.avgCost}</div>
              </div>
              <div className="text-right">
                <div className="font-mono">${pos.marketValue.toLocaleString()}</div>
                <div className={`text-sm ${pos.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {pos.unrealizedPnl >= 0 ? '+' : ''}${pos.unrealizedPnl.toFixed(2)} ({pos.unrealizedPnlPct.toFixed(2)}%)
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Performance Stats */}
      <div className="grid md:grid-cols-4 gap-4">
        <StatCard label="Win Rate" value={`${(demo.performance.winRate * 100).toFixed(1)}%`} />
        <StatCard label="Sharpe Ratio" value={demo.performance.sharpeRatio.toFixed(2)} />
        <StatCard label="Profit Factor" value={demo.performance.profitFactor.toFixed(2)} />
        <StatCard label="Max Drawdown" value={`${(demo.performance.maxDrawdown * 100).toFixed(1)}%`} negative />
      </div>
    </div>
  );
}

function StatCard({ label, value, negative = false }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="text-xs text-zinc-400 mb-1">{label}</div>
      <div className={`text-xl font-mono ${negative ? 'text-red-400' : 'text-zinc-100'}`}>{value}</div>
    </div>
  );
}

// ============================================================================
// 6. RECENT TRADES WITH EDUCATIONAL CONTEXT
// ============================================================================

function RecentTradesWithEducation() {
  const trades = demoData.portfolio.recentTrades;
  
  return (
    <div className="space-y-4">
      <h3 className="font-semibold">Recent Trades</h3>
      
      {trades.map((trade, idx) => (
        <div key={idx} className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
          {/* Trade Header */}
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                trade.type === 'BUY' ? 'bg-emerald-900/30 text-emerald-400' : 'bg-red-900/30 text-red-400'
              }`}>
                {trade.type}
              </span>
              <span className="font-semibold">{trade.symbol}</span>
            </div>
            <div className="text-right">
              {trade.status === 'CLOSED' ? (
                <div className={`font-mono ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {trade.pnl >= 0 ? '+' : ''}${trade.pnl?.toFixed(2)} ({trade.pnlPct?.toFixed(2)}%)
                </div>
              ) : (
                <span className="text-xs text-zinc-500">OPEN</span>
              )}
            </div>
          </div>
          
          {/* Educational Content */}
          <div className="border-t border-zinc-800 pt-3 space-y-2">
            <div className="flex items-start gap-2">
              <span className="text-lg">💡</span>
              <div>
                <div className="text-xs font-medium text-zinc-400 mb-1">Why this trade?</div>
                <p className="text-sm text-zinc-300">{trade.rationale}</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4 text-xs text-zinc-500">
              <div className="flex items-center gap-1">
                <span>🤖</span>
                <span>AI Confidence: {(trade.aiConfidence * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center gap-1">
                <span>🌍</span>
                <span>Regime: {trade.regime.replace('_', ' ')}</span>
              </div>
              <div className="flex items-center gap-1">
                <span>📚</span>
                <span>{trade.concepts.join(', ')}</span>
              </div>
            </div>
            
            {trade.lesson && (
              <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-3 mt-3">
                <div className="flex items-center gap-2">
                  <span className="text-lg">🎯</span>
                  <span className="text-xs font-medium text-blue-300">Key Lesson:</span>
                </div>
                <p className="text-xs text-blue-200 mt-1">{trade.lesson}</p>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default {
  TradeExplanation,
  AIDecisionExplorer,
  RegimeDetectionExplainer,
  TechnicalIndicatorsGuide,
  PortfolioSimulation,
  RecentTradesWithEducation,
  demoData
};
