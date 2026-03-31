import { useState } from 'react';

/**
 * TradingView Features Panel
 * Displays all available charting features and tools
 */
export default function ChartingFeaturesPanel() {
  const [isOpen, setIsOpen] = useState(false);

  const features = [
    {
      category: 'Chart Types',
      icon: '📊',
      items: [
        'Candlestick Charts',
        'Heikin Ashi Candles',
        'Hollow Candlesticks',
        'Bar Charts (OHLC)',
        'Line Charts',
        'Area Charts',
        'Baseline Chart',
        'Step Line',
        'Column Charts',
        'Pivot Points High/Low',
      ],
    },
    {
      category: 'Technical Indicators',
      icon: '📈',
      items: [
        'Moving Averages (SMA, EMA, WMA, VWMA)',
        'MACD (Moving Average Convergence Divergence)',
        'RSI (Relative Strength Index)',
        'Bollinger Bands',
        'Stochastic RSI',
        'ATR (Average True Range)',
        'ADX (Average Directional Index)',
        'Ichimoku Cloud',
        'Parabolic SAR',
        'Fibonacci Retracement',
        'Volume Profile',
        'On-Balance Volume (OBV)',
      ],
    },
    {
      category: 'Drawing Tools',
      icon: '✏️',
      items: [
        'Trend Lines',
        'Horizontal/Vertical Lines',
        'Regression Channel',
        'Fibonacci Extensions',
        'Gann Fan',
        'Pitchfork',
        'Schiff Pitchfork',
        'Rays',
        'Segments',
        'Curves',
        'Continuous Curves',
        'Elliott Waves',
        'Countertrend Lines',
        'Triangles',
        'Rectangles',
        'Arrows',
        'Text Labels',
      ],
    },
    {
      category: 'Timeframes',
      icon: '⏰',
      items: [
        '1 minute',
        '3 minutes',
        '5 minutes',
        '15 minutes',
        '30 minutes',
        '45 minutes',
        '1 hour',
        '2 hours',
        '3 hours',
        '4 hours',
        '1 day',
        '1 week',
        '1 month',
        '3 months',
        '6 months',
        '1 year',
        'All time',
      ],
    },
    {
      category: 'Analysis Tools',
      icon: '🔬',
      items: [
        'Compare Symbols',
        'Add Multiple Indicators',
        'Alert System',
        'Strategy Tester (Pine Script)',
        'Bar Replay Mode',
        'Measure Tool',
        'Price Scale Options',
        'Logarithmic Scale',
        'Percentage Scale',
        'Regular Scale',
      ],
    },
    {
      category: 'Smart Features',
      icon: '🧠',
      items: [
        'Real-time Data Streaming',
        'Auto Pattern Recognition',
        'Candlestick Patterns',
        'Classic Chart Patterns',
        'Supply & Demand Zones',
        'Support/Resistance Levels',
        'Pivot Points Classic',
        'Pivot Points Fibonacci',
        'Pivot Points Woodie',
        'Pivot Points Camarilla',
        'Pivot Points DeMark',
      ],
    },
  ];

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg shadow-blue-600/30 transition-all hover:scale-105"
      >
        <span className="text-xl">🎓</span>
        <span className="font-mono text-sm">{isOpen ? 'Hide' : 'Learn'} Features</span>
      </button>

      {/* Features Panel */}
      {isOpen && (
        <div className="absolute bottom-20 right-0 w-[800px] max-h-[600px] overflow-y-auto rounded-2xl border border-zinc-800 bg-zinc-950/95 backdrop-blur-xl shadow-2xl">
          {/* Header */}
          <div className="sticky top-0 flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-950/95 backdrop-blur-xl">
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">TradingView Pro Features</h2>
              <p className="text-xs text-zinc-500 mt-0.5">Professional-grade charting toolkit</p>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Content */}
          <div className="p-6 grid gap-6 md:grid-cols-2">
            {features.map((section, idx) => (
              <div
                key={idx}
                className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl">{section.icon}</span>
                  <h3 className="font-semibold text-zinc-200">{section.category}</h3>
                </div>
                <ul className="space-y-1.5">
                  {section.items.map((item, i) => (
                    <li
                      key={i}
                      className="text-sm text-zinc-400 flex items-start gap-2"
                    >
                      <span className="w-1 h-1 rounded-full bg-blue-500 mt-1.5 flex-shrink-0"></span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="sticky bottom-0 px-6 py-4 border-t border-zinc-800 bg-zinc-950/95 backdrop-blur-xl">
            <div className="flex items-center justify-between">
              <div className="text-xs text-zinc-500">
                📌 Tip: Right-click on chart to access more tools
              </div>
              <a
                href="https://www.tradingview.com/support/"
                target="_blank"
                rel="noreferrer"
                className="text-xs font-mono text-blue-400 hover:text-blue-300"
              >
                Full Documentation →
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
