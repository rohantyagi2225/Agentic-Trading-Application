import React, { useState } from 'react';

/**
 * Educational Trade Explanation Component
 * Shows WHY a trade was made with beginner-friendly explanations
 */

export default function TradeExplanation({ trade, signalData }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  if (!trade) return null;
  
  // Beginner-friendly explanations
  const getConceptExplanation = (concept) => {
    const explanations = {
      rsi: {
        name: 'RSI (Relative Strength Index)',
        analogy: 'Think of RSI like a thermometer for stock momentum. Just as a fever shows your body is too hot, high RSI (>70) shows a stock might be "overheated" and due for a cooldown.',
        beginner: 'Measures whether a stock has been going up or down too fast',
        range: '0-100 (Above 70 = Overbought, Below 30 = Oversold)'
      },
      trend: {
        name: 'Trend Analysis',
        analogy: 'Like surfing - you want to ride the wave in the direction it\'s already going. Trend analysis tells us which way the market wave is moving.',
        beginner: 'Identifies the general direction the price is moving',
        types: 'Uptrend (higher highs), Downtrend (lower lows), Sideways (range-bound)'
      },
      support_resistance: {
        name: 'Support & Resistance',
        analogy: 'Imagine a ball bouncing in a room. The floor is support (price bounces up), the ceiling is resistance (price gets pushed down).',
        beginner: 'Price levels where the stock has trouble moving past',
        levels: 'Support = Floor, Resistance = Ceiling'
      },
      volume: {
        name: 'Volume Analysis',
        analogy: 'Volume is like crowd enthusiasm at a concert. High volume means strong conviction in the price move, low volume means weak interest.',
        beginner: 'Shows how many shares are being traded',
        insight: 'High volume = Strong conviction, Low volume = Weak conviction'
      },
      volatility: {
        name: 'Volatility',
        analogy: 'Volatility is like a rollercoaster vs a train ride. High volatility = wild price swings (thrilling but risky). Low volatility = smooth, predictable movement.',
        beginner: 'Measures how much the price jumps around',
        measure: 'Standard deviation of returns'
      },
      momentum: {
        name: 'Momentum',
        analogy: 'Like a speeding car - momentum shows how fast price is changing and whether it\'s likely to keep going.',
        beginner: 'Speed of price movement',
        signal: 'Positive = Accelerating, Negative = Decelerating'
      },
      sentiment: {
        name: 'Market Sentiment',
        analogy: 'Market sentiment is like the mood at a party. When everyone\'s optimistic (bullish), prices tend to rise. When pessimistic (bearish), prices fall.',
        beginner: 'Overall feeling or attitude of investors',
        sources: 'News, social media, analyst ratings'
      },
      regime_detection: {
        name: 'Regime Detection',
        analogy: 'Like checking the weather before dressing. Different market regimes (bull, bear, sideways) require different strategies, just like you dress differently for sun vs rain.',
        beginner: 'AI identifies what type of market we\'re in',
        regimes: 'Bull Market, Bear Market, High Volatility, Low Volatility'
      }
    };
    
    return explanations[concept.toLowerCase()] || { name: concept, beginner: 'Technical indicator', analogy: '' };
  };
  
  // Risk explanation
  const getRiskExplanation = () => {
    const riskLevel = trade.risk_level || 'moderate';
    
    const risks = {
      low: {
        label: 'Low Risk',
        color: 'text-emerald-400',
        bg: 'bg-emerald-900/20',
        explanation: 'This trade has conservative position sizing and clear exit points. Lower potential reward, but also lower chance of significant loss.',
        analogy: 'Like driving in a school zone - slow and steady, prioritizing safety over speed.'
      },
      moderate: {
        label: 'Moderate Risk',
        color: 'text-amber-400',
        bg: 'bg-amber-900/20',
        explanation: 'Balanced risk-reward ratio. Standard position sizing with defined stop-loss. Suitable for most market conditions.',
        analogy: 'Like highway driving - reasonable speed with normal safety precautions.'
      },
      high: {
        label: 'High Risk',
        color: 'text-red-400',
        bg: 'bg-red-900/20',
        explanation: 'Aggressive position with higher volatility. Potential for significant gains OR losses. Only recommended for experienced traders.',
        analogy: 'Like racing on a track - high speed, high skill required, higher danger.'
      }
    };
    
    return risks[riskLevel] || risks.moderate;
  };
  
  // What-if scenario
  const getWhatIfScenario = () => {
    const scenarios = {
      correct: {
        title: 'If This Trade Goes Well...',
        outcome: `Your ${trade.direction === 'BUY' ? 'long' : 'short'} position could gain ${(trade.expected_return * 100).toFixed(1)}% if the price moves as expected.`,
        lesson: 'This would validate our analysis that the stock was undervalued and ready to move higher.',
        takeaway: 'Key learning: Patience and proper risk management pay off.'
      },
      incorrect: {
        title: 'If This Trade Goes Against You...',
        outcome: `Your maximum loss is limited to ${(trade.stop_loss_pct * 100).toFixed(1)}% thanks to the stop-loss at $${trade.stop_loss?.toFixed(2) || 'N/A'}.`,
        lesson: 'Even wrong trades can be profitable learning experiences when properly managed.',
        takeaway: 'Key learning: Losses are part of trading - controlling them is what matters.'
      }
    };
    
    return scenarios;
  };
  
  const riskInfo = getRiskExplanation();
  const whatIfScenarios = getWhatIfScenario();
  
  return (
    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-zinc-100">
          🎓 Why This Trade?
        </h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
        >
          {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
        </button>
      </div>
      
      {/* Main Decision Reason */}
      <div className="space-y-3">
        <div className="flex items-start gap-3">
          <span className="text-2xl">💡</span>
          <div>
            <h4 className="font-medium text-zinc-100 mb-1">Main Reason</h4>
            <p className="text-zinc-400 leading-relaxed">
              {trade.decision_rationale || 'Multiple technical indicators align to suggest this opportunity.'}
            </p>
          </div>
        </div>
      </div>
      
      {/* Concepts Used */}
      {signalData?.contributing_signals && signalData.contributing_signals.length > 0 && (
        <div className="space-y-3">
          <h4 className="font-medium text-zinc-100">📚 Concepts Used</h4>
          <div className="grid md:grid-cols-2 gap-3">
            {signalData.contributing_signals.map((signal, idx) => {
              const concept = getConceptExplanation(signal.source || signal.type);
              return (
                <div key={idx} className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{signal.source === 'technical' ? '📊' : signal.source === 'sentiment' ? '😊' : '🌍'}</span>
                    <span className="font-medium text-zinc-100 text-sm">{concept.name}</span>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {concept.beginner}
                  </p>
                  {showAdvanced && (
                    <>
                      <p className="text-xs text-zinc-500 italic mt-2">
                        💭 {concept.analogy}
                      </p>
                      <div className="text-xs text-zinc-500 mt-2 pt-2 border-t border-zinc-700">
                        <strong>Range:</strong> {concept.range || concept.types || concept.insight || 'N/A'}
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
      
      {/* Risk Explanation */}
      <div className={`${riskInfo.bg} rounded-lg p-4 space-y-3 border border-zinc-700`}>
        <div className="flex items-center gap-2">
          <span className="text-lg">⚠️</span>
          <span className={`font-medium ${riskInfo.color}`}>{riskInfo.label}</span>
        </div>
        <p className="text-sm text-zinc-300 leading-relaxed">
          {riskInfo.explanation}
        </p>
        {showAdvanced && (
          <p className="text-xs text-zinc-500 italic">
            💭 {riskInfo.analogy}
          </p>
        )}
      </div>
      
      {/* What-If Scenarios */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-emerald-900/20 border border-emerald-800/50 rounded-lg p-4 space-y-2">
          <h4 className="font-medium text-emerald-400 flex items-center gap-2">
            ✅ {whatIfScenarios.correct.title}
          </h4>
          <p className="text-sm text-zinc-300">{whatIfScenarios.correct.outcome}</p>
          <p className="text-xs text-zinc-500 mt-2">{whatIfScenarios.correct.lesson}</p>
        </div>
        
        <div className="bg-red-900/20 border border-red-800/50 rounded-lg p-4 space-y-2">
          <h4 className="font-medium text-red-400 flex items-center gap-2">
            ❌ {whatIfScenarios.incorrect.title}
          </h4>
          <p className="text-sm text-zinc-300">{whatIfScenarios.incorrect.outcome}</p>
          <p className="text-xs text-zinc-500 mt-2">{whatIfScenarios.incorrect.lesson}</p>
        </div>
      </div>
      
      {/* Key Takeaway */}
      <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
        <p className="text-sm text-blue-300 font-medium">
          🎯 {whatIfScenarios.correct.takeaway}
        </p>
      </div>
    </div>
  );
}
