/**
 * Demo Mode Data Provider
 * Provides realistic sample data for offline/demo functionality
 */

export const demoPortfolio = {
  user: {
    name: 'Demo User',
    accountType: 'Paper Trading',
    memberSince: '2024-01-01'
  },
  
  balance: {
    cash: 85420.50,
    invested: 14579.50,
    totalValue: 100000.00,
    dayChange: 1250.30,
    dayChangePct: 1.27,
    totalReturn: 0.00,
    totalReturnPct: 0.00
  },
  
  positions: [
    {
      symbol: 'AAPL',
      quantity: 50,
      avgCost: 172.45,
      currentPrice: 178.20,
      marketValue: 8910.00,
      unrealizedPnl: 287.50,
      unrealizedPnlPct: 3.33,
      allocation: 6.12
    },
    {
      symbol: 'MSFT',
      quantity: 15,
      avgCost: 378.90,
      currentPrice: 385.50,
      marketValue: 5782.50,
      unrealizedPnl: 99.00,
      unrealizedPnlPct: 1.74,
      allocation: 3.96
    }
  ],
  
  recentTrades: [
    {
      id: 'DEMO_001',
      symbol: 'AAPL',
      type: 'BUY',
      quantity: 50,
      price: 172.45,
      date: '2024-03-15T10:30:00Z',
      pnl: null,
      status: 'OPEN',
      rationale: 'Strong technical setup with RSI showing oversold conditions and positive momentum divergence',
      aiConfidence: 0.78,
      regime: 'bull_market',
      concepts: ['RSI Divergence', 'Support Bounce', 'Momentum Shift']
    },
    {
      id: 'DEMO_002',
      symbol: 'MSFT',
      type: 'BUY',
      quantity: 15,
      price: 378.90,
      date: '2024-03-18T14:15:00Z',
      pnl: null,
      status: 'OPEN',
      rationale: 'Breakout above resistance with high volume confirmation and positive sentiment',
      aiConfidence: 0.82,
      regime: 'low_volatility',
      concepts: ['Breakout', 'Volume Confirmation', 'Sentiment Analysis']
    },
    {
      id: 'DEMO_003',
      symbol: 'GOOGL',
      type: 'BUY',
      quantity: 25,
      price: 145.20,
      date: '2024-03-10T11:00:00Z',
      exitPrice: 152.80,
      exitDate: '2024-03-20T15:45:00Z',
      pnl: 190.00,
      pnlPct: 5.23,
      status: 'CLOSED',
      rationale: 'Undervalued relative to peers with improving technical setup',
      aiConfidence: 0.75,
      regime: 'bull_market',
      concepts: ['Relative Value', 'Technical Improvement'],
      outcome: 'WIN'
    },
    {
      id: 'DEMO_004',
      symbol: 'TSLA',
      type: 'BUY',
      quantity: 30,
      price: 178.50,
      date: '2024-03-05T09:45:00Z',
      exitPrice: 172.30,
      exitDate: '2024-03-12T10:30:00Z',
      pnl: -186.00,
      pnlPct: -3.47,
      status: 'CLOSED',
      rationale: 'Momentum reversal signal with deteriorating market structure',
      aiConfidence: 0.68,
      regime: 'high_volatility',
      concepts: ['Momentum Reversal', 'Risk Management'],
      outcome: 'LOSS',
      lesson: 'Stop-loss executed as planned, limiting loss to predefined risk level'
    }
  ],
  
  performance: {
    totalReturn: 0.00,
    totalReturnPct: 0.00,
    realizedPnl: 4520.00,
    unrealizedPnl: 386.50,
    winRate: 0.64,
    avgWin: 425.00,
    avgLoss: -185.00,
    profitFactor: 2.3,
    sharpeRatio: 1.45,
    maxDrawdown: -0.08,
    winningTrades: 16,
    losingTrades: 9,
    totalTrades: 25
  },
  
  monthlyReturns: [
    { month: 'Jan 2024', return: 2.4 },
    { month: 'Feb 2024', return: 1.8 },
    { month: 'Mar 2024', return: 3.2 },
    { month: 'Apr 2024', return: -0.5 },
    { month: 'May 2024', return: 1.9 },
    { month: 'Jun 2024', return: 2.1 }
  ]
};

export const demoSignals = [
  {
    id: 'SIG_DEMO_001',
    timestamp: new Date().toISOString(),
    symbol: 'NVDA',
    source: 'technical',
    type: 'bullish',
    strength: 0.82,
    confidence: 0.88,
    regime: 'bull_market',
    finalDecision: 'BUY',
    decisionConfidence: 0.85,
    rationale: 'Strong uptrend with RSI showing continued momentum without overbought conditions',
    concepts: ['Trend Following', 'RSI Analysis', 'Volume Profile'],
    targetPrice: 985.00,
    stopLoss: 890.00,
    expectedReturn: 0.08
  },
  {
    id: 'SIG_DEMO_002',
    timestamp: new Date().toISOString(),
    symbol: 'AMD',
    source: 'sentiment',
    type: 'bullish',
    strength: 0.75,
    confidence: 0.72,
    regime: 'bull_market',
    finalDecision: 'HOLD',
    decisionConfidence: 0.65,
    rationale: 'Positive sentiment but already fully valued, waiting for better entry',
    concepts: ['Sentiment Analysis', 'Valuation Check'],
    targetPrice: null,
    stopLoss: null,
    expectedReturn: 0.00
  },
  {
    id: 'SIG_DEMO_003',
    timestamp: new Date().toISOString(),
    symbol: 'SPY',
    source: 'macro',
    type: 'bearish',
    strength: 0.68,
    confidence: 0.70,
    regime: 'high_volatility',
    finalDecision: 'SELL',
    decisionConfidence: 0.72,
    rationale: 'Increasing volatility and deteriorating breadth suggest defensive positioning',
    concepts: ['Volatility Analysis', 'Market Breadth', 'Risk Management'],
    targetPrice: 485.00,
    stopLoss: 515.00,
    expectedReturn: -0.05
  }
];

export const demoRegimeHistory = [
  {
    date: '2024-03-20',
    primaryRegime: 'bull_market',
    confidence: 0.82,
    volatilityRegime: 'normal',
    trendStrength: 0.75,
    description: 'Steady uptrend with normal volatility - ideal for trend following'
  },
  {
    date: '2024-03-15',
    primaryRegime: 'low_volatility',
    confidence: 0.78,
    volatilityRegime: 'low',
    trendStrength: 0.65,
    description: 'Low volatility environment favors selling premium strategies'
  },
  {
    date: '2024-03-10',
    primaryRegime: 'high_volatility',
    confidence: 0.85,
    volatilityRegime: 'high',
    trendStrength: 0.45,
    description: 'High volatility requires smaller position sizes and wider stops'
  }
];

export const demoEducationalContent = {
  regimes: {
    bull_market: {
      title: 'Bull Market',
      description: 'Prices are rising, optimism is high, and the economy is strong.',
      strategy: 'Focus on buying stocks on dips. Trend-following strategies work well.',
      analogy: 'Like surfing - you want to ride the wave in the direction it\'s moving.',
      tips: [
        'Buy pullbacks to support levels',
        'Let winners run',
        'Use moving averages as dynamic support',
        "Don't fight the trend"
      ]
    },
    bear_market: {
      title: 'Bear Market',
      description: 'Prices are falling, pessimism dominates, economic outlook weakens.',
      strategy: 'Defensive positioning, focus on quality stocks, consider short-selling.',
      analogy: 'Like winter - time to hibernate and preserve capital until spring.',
      tips: [
        'Reduce position sizes',
        'Focus on defensive sectors',
        'Keep dry powder for opportunities',
        'Consider inverse ETFs or short positions'
      ]
    },
    high_volatility: {
      title: 'High Volatility',
      description: 'Large price swings, uncertainty, fear or excitement.',
      strategy: 'Smaller positions, wider stops, avoid overtrading.',
      analogy: 'Like driving in a storm - slow down and increase following distance.',
      tips: [
        'Cut position size in half',
        'Widen stop-losses to avoid noise',
        'Avoid leverage',
        'Wait for clarity'
      ]
    },
    low_volatility: {
      title: 'Low Volatility',
      description: 'Small price movements, complacency, stability.',
      strategy: 'Sell options premium, range-bound strategies.',
      analogy: 'Like calm seas - good for steady sailing but watch for storms.',
      tips: [
        'Sell options to collect premium',
        'Play range boundaries',
        'Watch for volatility expansion',
        'Use leverage cautiously'
      ]
    }
  },
  
  indicators: {
    rsi: {
      name: 'RSI (Relative Strength Index)',
      whatItIs: 'Measures the speed and magnitude of recent price changes to determine if a stock is overbought or oversold.',
      howToUse: [
        'Above 70 = Overbought (potential sell signal)',
        'Below 30 = Oversold (potential buy signal)',
        'Look for divergences with price for stronger signals'
      ],
      commonMistakes: [
        'Using RSI alone without other confirmation',
        'Ignoring the overall trend',
        'Acting on every overbought/oversold reading'
      ]
    },
    macd: {
      name: 'MACD (Moving Average Convergence Divergence)',
      whatItIs: 'Shows the relationship between two moving averages and indicates momentum and trend direction.',
      howToUse: [
        'Signal line crossovers indicate potential entries',
        'Histogram shows momentum strength',
        'Divergences with price can signal reversals'
      ],
      commonMistakes: [
        'Trading every crossover without trend context',
        'Ignoring the histogram',
        'Using in choppy, non-trending markets'
      ]
    },
    volume: {
      name: 'Volume Analysis',
      whatItIs: 'Shows how many shares/contracts are traded in a given period.',
      howToUse: [
        'High volume confirms price moves',
        'Low volume suggests weak conviction',
        'Volume spikes often mark turning points'
      ],
      commonMistakes: [
        'Ignoring volume when analyzing price moves',
        'Not comparing to average volume',
        'Assuming all volume is equal (institutional vs retail)'
      ]
    }
  }
};

export default {
  portfolio: demoPortfolio,
  signals: demoSignals,
  regimeHistory: demoRegimeHistory,
  educationalContent: demoEducationalContent
};
