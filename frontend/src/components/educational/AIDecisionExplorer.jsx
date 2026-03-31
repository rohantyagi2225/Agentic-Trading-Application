import React, { useState } from 'react';

/**
 * Interactive AI Decision Explorer
 * Visualizes how the AI makes trading decisions step-by-step
 */

export default function AIDecisionExplorer({ decisionData }) {
  const [activeStep, setActiveStep] = useState(0);
  
  if (!decisionData) return null;
  
  const steps = [
    {
      id: 'data',
      icon: '📊',
      title: 'Step 1: Collect Market Data',
      description: 'AI gathers real-time price data, volume, and market information',
      details: decisionData.market_data || {},
      analogy: 'Like a detective gathering clues before solving a case'
    },
    {
      id: 'regime',
      icon: '🌍',
      title: 'Step 2: Detect Market Regime',
      description: 'AI identifies what type of market we\'re in (bull, bear, volatile)',
      details: decisionData.regime || {},
      analogy: 'Like checking the weather - you dress differently for sun vs rain'
    },
    {
      id: 'signals',
      icon: '📡',
      title: 'Step 3: Generate Signals',
      description: 'Multiple AI agents analyze different aspects (technical, sentiment, macro)',
      details: decisionData.signals || [],
      analogy: 'Like asking multiple experts for their opinions'
    },
    {
      id: 'fusion',
      icon: '🔀',
      title: 'Step 4: Fuse Signals',
      description: 'AI combines all signals with confidence weighting',
      details: decisionData.fusion || {},
      analogy: 'Like averaging predictions from a panel of experts, giving more weight to confident ones'
    },
    {
      id: 'risk',
      icon: '⚠️',
      title: 'Step 5: Risk Check',
      description: 'Circuit breakers ensure this trade doesn\'t exceed risk limits',
      details: decisionData.risk || {},
      analogy: 'Like a safety net that prevents dangerous trades'
    },
    {
      id: 'decision',
      icon: '🎯',
      title: 'Step 6: Final Decision',
      description: 'AI makes the final BUY/SELL/HOLD decision with confidence score',
      details: decisionData.decision || {},
      analogy: 'The moment of truth - all analysis leads to action'
    }
  ];
  
  return (
    <div className="bg-zinc-900 rounded-xl border border-zinc-800 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 px-6 py-4 border-b border-zinc-800">
        <h3 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
          🧠 How AI Makes This Decision
        </h3>
        <p className="text-xs text-zinc-400 mt-1">
          Click through each step to see the decision-making process
        </p>
      </div>
      
      {/* Step Navigation */}
      <div className="flex overflow-x-auto border-b border-zinc-800 scrollbar-thin">
        {steps.map((step, idx) => (
          <button
            key={step.id}
            onClick={() => setActiveStep(idx)}
            className={`flex items-center gap-2 px-4 py-3 text-sm whitespace-nowrap transition-colors border-b-2 ${
              activeStep === idx
                ? 'border-blue-500 text-blue-400 bg-blue-900/20'
                : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
            }`}
          >
            <span className="text-lg">{step.icon}</span>
            <span className="hidden md:inline">{step.title}</span>
          </button>
        ))}
      </div>
      
      {/* Active Step Content */}
      <div className="p-6 space-y-4">
        <div className="flex items-start gap-4">
          <div className="text-4xl">{steps[activeStep].icon}</div>
          <div className="flex-1 space-y-3">
            <div>
              <h4 className="text-xl font-semibold text-zinc-100 mb-1">
                {steps[activeStep].title}
              </h4>
              <p className="text-zinc-400 leading-relaxed">
                {steps[activeStep].description}
              </p>
            </div>
            
            {/* Analogy */}
            <div className="bg-zinc-800/50 rounded-lg p-3 border-l-4 border-blue-500">
              <p className="text-sm text-zinc-300 italic">
                💭 <strong>Think of it like:</strong> {steps[activeStep].analogy}
              </p>
            </div>
            
            {/* Technical Details */}
            {Object.keys(steps[activeStep].details).length > 0 && (
              <div className="space-y-2">
                <h5 className="text-sm font-medium text-zinc-300">Technical Details:</h5>
                <div className="bg-zinc-950 rounded-lg p-4 font-mono text-xs overflow-x-auto">
                  <pre className="text-zinc-400">
                    {JSON.stringify(steps[activeStep].details, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Navigation Buttons */}
        <div className="flex justify-between pt-4 border-t border-zinc-800">
          <button
            onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
            disabled={activeStep === 0}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            ← Previous
          </button>
          <button
            onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))}
            disabled={activeStep === steps.length - 1}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
