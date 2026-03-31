import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { PREMIUM_COLORS, COMPONENTS, TYPOGRAPHY, ANIMATIONS } from '../styles/premiumDesignSystem';

const FEATURES = [
  {
    icon: '🤖',
    title: 'Multi-Agent Trading',
    desc: 'Watch momentum, mean reversion, and factor agents collaborate in real time to generate high-quality signals.',
    gradient: 'from-cyan-500/20 via-blue-500/20 to-purple-500/20',
  },
  {
    icon: '📡',
    title: 'Real-Time Signals',
    desc: 'Live WebSocket streams deliver signals with confidence scores and agent explanations as they happen.',
    gradient: 'from-emerald-500/20 via-teal-500/20 to-cyan-500/20',
  },
  {
    icon: '📊',
    title: 'Portfolio Analytics',
    desc: 'Sharpe ratio, Sortino, max drawdown, alpha/beta — institutional-grade metrics on your demo portfolio.',
    gradient: 'from-amber-500/20 via-orange-500/20 to-yellow-500/20',
  },
  {
    icon: '🎓',
    title: 'Learn by Doing',
    desc: 'Every agent explains its logic. Understand WHY trades are made, not just what happens.',
    gradient: 'from-violet-500/20 via-purple-500/20 to-fuchsia-500/20',
  },
];

const STEPS = [
  { n: '01', title: 'Create a free account', desc: 'Get a $100,000 paper trading account instantly — no credit card required.' },
  { n: '02', title: 'Watch agents trade', desc: 'Our AI agents analyze markets in real time and generate buy/sell signals with explanations.' },
  { n: '03', title: 'Practice & learn', desc: 'Execute simulated trades, track your P&L, and understand every decision.' },
  { n: '04', title: 'Go live when ready', desc: 'Connect to Alpaca, Interactive Brokers, or Binance when you\'re confident.' },
];

const TICKERS = [
  { symbol: 'AAPL', change: '+1.24%' },
  { symbol: 'NVDA', change: '+3.17%' },
  { symbol: 'TSLA', change: '-0.89%' },
  { symbol: 'META', change: '+2.05%' },
  { symbol: 'GOOGL', change: '+0.61%' },
  { symbol: 'MSFT', change: '+1.42%' },
  { symbol: 'AMZN', change: '-0.33%' },
  { symbol: 'BTC', change: '+4.21%' },
];

function TickerBanner() {
  return (
    <div className="relative bg-black/40 backdrop-blur-xl border-b border-zinc-800/50 overflow-hidden py-3">
      <div className="flex animate-ticker whitespace-nowrap">
        {[...TICKERS, ...TICKERS].map((t, i) => (
          <span 
            key={i} 
            className={`inline-flex items-center gap-2 mx-6 text-sm font-mono font-semibold ${
              t.change.includes('+') ? 'text-emerald-400' : 'text-red-400'
            }`}
          >
            <span className="text-zinc-300">{t.symbol}</span>
            <span className={t.change.includes('+') ? 'text-emerald-400' : 'text-red-400'}>{t.change}</span>
          </span>
        ))}
      </div>
      {/* Gradient overlays */}
      <div className="absolute inset-y-0 left-0 w-32 bg-gradient-to-r from-zinc-950 to-transparent"></div>
      <div className="absolute inset-y-0 right-0 w-32 bg-gradient-to-l from-zinc-950 to-transparent"></div>
    </div>
  );
}

export default function Landing() {
  const { isAuthenticated } = useAuth();
  const [email, setEmail] = useState('');
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Parallax effect for hero glow
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 20,
        y: (e.clientY / window.innerHeight - 0.5) * 20,
      });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 overflow-x-hidden selection:bg-cyan-500/30 selection:text-white">
      {/* Ticker */}
      <TickerBanner />

      {/* Hero Section - Premium Design */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Animated background grid */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#18181b_1px,transparent_1px),linear-gradient(to_bottom,#18181b_1px,transparent_1px)] bg-[size:3rem_3rem] opacity-30 animate-pulse"></div>
        
        {/* Dynamic glow effects */}
        <div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 rounded-full blur-[120px] pointer-events-none transition-transform duration-1000"
          style={{ 
            transform: `translate(calc(-50% + ${mousePosition.x}px), calc(-50% + ${mousePosition.y}px))` 
          }}
        />
        
        {/* Secondary glow */}
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-gradient-to-l from-emerald-500/10 to-transparent rounded-full blur-[100px] pointer-events-none animate-pulse" />

        <div className="relative max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-32 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 backdrop-blur-xl border border-white/10 rounded-full px-6 py-3 mb-10 shadow-2xl shadow-cyan-500/20 hover:shadow-cyan-500/30 transition-all duration-500 hover:scale-105">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-sm font-medium text-zinc-300">AI-Powered Trading Platform</span>
          </div>

          {/* Main heading with gradient text */}
          <h1 className={`${TYPOGRAPHY.headings.hero} mb-6 leading-tight`}>
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-100">
              Trade Smarter with
            </span>
            <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 drop-shadow-[0_0_30px_rgba(34,211,238,0.3)]">
              AI-Powered Intelligence
            </span>
          </h1>

          {/* Subheading */}
          <p className={`${TYPOGRAPHY.body.large} text-zinc-400 max-w-3xl mx-auto mb-12 leading-relaxed`}>
            Experience the future of trading with our multi-agent AI system. 
            Watch intelligent agents analyze markets, generate signals, and explain their decisions in real-time.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Link 
              to={isAuthenticated ? '/dashboard' : '/signup'} 
              className={`group relative px-8 py-4 rounded-2xl font-semibold text-white overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-[0_0_40px_-10px_rgba(34,211,238,0.5)] ${PREMIUM_COLORS.gradients.primary}`}
            >
              <span className="relative z-10 flex items-center gap-2">
                {isAuthenticated ? 'Open Dashboard →' : 'Start Trading Free'}
              </span>
              <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            </Link>
            
            <Link 
              to="/markets" 
              className={`px-8 py-4 rounded-2xl font-semibold transition-all duration-300 hover:scale-105 ${PREMIUM_COLORS.glass.light} text-white hover:bg-white/20`}
            >
              Explore Markets
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
            {[
              { value: '$100K', label: 'Paper Trading', sublabel: 'Virtual Balance' },
              { value: '24/7', label: 'Market Monitoring', sublabel: 'AI Analysis' },
              { value: '10+', label: 'AI Agents', sublabel: 'Working Together' },
              { value: '0%', label: 'Commission', sublabel: 'Risk-Free Practice' },
            ].map((stat, idx) => (
              <div 
                key={idx}
                className={`group p-6 rounded-3xl ${PREMIUM_COLORS.glass.dark} border border-zinc-800/50 hover:border-cyan-500/30 transition-all duration-500 hover:scale-105 hover:shadow-[0_0_30px_-10px_rgba(34,211,238,0.3)]`}
              >
                <div className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-400 mb-2">
                  {stat.value}
                </div>
                <div className="text-sm font-semibold text-zinc-300 mb-1">{stat.label}</div>
                <div className="text-xs text-zinc-600">{stat.sublabel}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 rounded-full border-2 border-zinc-700 flex items-start justify-center p-2">
            <div className="w-1 h-2 bg-zinc-500 rounded-full animate-scroll"></div>
          </div>
        </div>
      </section>

      {/* Features Section - Glassmorphism Cards */}
      <section id="features" className="py-32 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className={TYPOGRAPHY.headings.h2}>
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400">
                Powerful Features
              </span>
            </h2>
            <p className={`${TYPOGRAPHY.body.large} text-zinc-400 mt-4 max-w-2xl mx-auto`}>
              Everything you need to master algorithmic trading
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {FEATURES.map((feature, idx) => (
              <div
                key={idx}
                className={`group relative p-8 rounded-3xl ${PREMIUM_COLORS.backgrounds.card} ${PREMIUM_COLORS.glass.medium} border border-zinc-800/50 hover:border-white/20 transition-all duration-500 hover:-translate-y-2 overflow-hidden`}
              >
                {/* Gradient background on hover */}
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}></div>
                
                {/* Content */}
                <div className="relative z-10">
                  <div className="text-5xl mb-6 transform group-hover:scale-110 transition-transform duration-300">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold text-zinc-100 mb-3 group-hover:text-white">
                    {feature.title}
                  </h3>
                  <p className="text-zinc-400 leading-relaxed group-hover:text-zinc-300">
                    {feature.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works - Step Process */}
      <section id="how-it-works" className="py-32 relative bg-gradient-to-b from-zinc-950 via-zinc-900/50 to-zinc-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className={TYPOGRAPHY.headings.h2}>
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400">
                Get Started in Minutes
              </span>
            </h2>
            <p className={`${TYPOGRAPHY.body.large} text-zinc-400 mt-4 max-w-2xl mx-auto`}>
              Four simple steps to begin your algorithmic trading journey
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {STEPS.map((step, idx) => (
              <div key={idx} className="relative group">
                {/* Connector line */}
                {idx < STEPS.length - 1 && (
                  <div className="hidden lg:block absolute top-12 left-[60%] w-full h-0.5 bg-gradient-to-r from-cyan-500/50 to-transparent"></div>
                )}
                
                <div className={`${PREMIUM_COLORS.glass.dark} rounded-3xl p-8 border border-zinc-800/50 hover:border-cyan-500/30 transition-all duration-500 hover:-translate-y-2`}>
                  {/* Step number */}
                  <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 text-cyan-400 font-bold text-lg mb-6 group-hover:scale-110 transition-transform duration-300">
                    {step.n}
                  </div>
                  
                  <h3 className="text-xl font-bold text-zinc-100 mb-3">
                    {step.title}
                  </h3>
                  <p className="text-zinc-400 leading-relaxed">
                    {step.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 relative overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-purple-500/10 blur-3xl"></div>
        
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 relative">
          <div className={`${PREMIUM_COLORS.glass.strong} rounded-[3rem] p-12 md:p-16 text-center border border-zinc-800/50 shadow-2xl`}>
            <h2 className={`${TYPOGRAPHY.headings.h1} mb-6`}>
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400">
                Ready to Start Trading?
              </span>
            </h2>
            <p className={`${TYPOGRAPHY.body.large} text-zinc-400 mb-10 max-w-2xl mx-auto`}>
              Join thousands of traders using AI-powered tools to make smarter investment decisions.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link 
                to="/signup" 
                className={`w-full sm:w-auto px-8 py-4 rounded-2xl font-semibold text-white transition-all duration-300 hover:scale-105 hover:shadow-[0_0_40px_-10px_rgba(34,211,238,0.5)] ${PREMIUM_COLORS.gradients.primary}`}
              >
                Create Free Account
              </Link>
              <Link 
                to="/learn" 
                className={`w-full sm:w-auto px-8 py-4 rounded-2xl font-semibold transition-all duration-300 hover:scale-105 ${PREMIUM_COLORS.glass.light} text-white hover:bg-white/20`}
              >
                Learn More
              </Link>
            </div>
            
            <p className="text-sm text-zinc-500 mt-8">
              No credit card required • $100,000 paper trading included
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-800/50 py-12 bg-zinc-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 flex items-center justify-center">
                <span className="text-cyan-400 text-xs font-bold">AGT</span>
              </div>
              <span className="font-semibold text-zinc-300">AgenticTrading</span>
            </div>
            <div className="text-sm text-zinc-500">
              © 2024 AgenticTrading. Built for educational purposes.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
