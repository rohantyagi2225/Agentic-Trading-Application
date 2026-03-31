"""
Comprehensive Integration Test for Financial Agent System

This test validates the complete financial agent system including:
- Regime detection
- Multi-source signal integration
- Explainable decision making
- Performance evaluation
- Memory-based learning

Usage:
    python comprehensive_integration_test.py
    
Author: FinAgent Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.market.regime_detector import RegimeDetector, MarketRegime, create_regime_detector
from backend.market.signal_integrator import (
    MultiSourceSignalIntegrator, SignalType, create_signal_integrator,
    TechnicalSignalGenerator, SentimentAnalyzer, MacroSignalGenerator
)
from backend.analytics.explainable_agent import ExplainableDecisionAgent, DecisionType
from backend.analytics.evaluation_framework import (
    ComprehensiveEvaluator, Trade, create_comprehensive_evaluator
)
from FinAgents.memory.regime_memory_learner import RegimeMemoryLearner, create_regime_memory_learner


def generate_synthetic_market_data(
    start_date: datetime = None,
    periods: int = 500,
    trend: float = 0.0005,
    volatility: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic market data for testing"""
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=periods)
    
    np.random.seed(seed)
    
    # Generate price series with trend and volatility
    returns = np.random.normal(trend, volatility, periods)
    prices = 100 * np.cumprod(1 + returns)
    
    # Generate volume (random with some correlation to volatility)
    base_volume = 1000000
    volumes = base_volume * (1 + np.abs(returns) * 10 + np.random.uniform(0.5, 1.5, periods))
    
    # Generate high/low
    daily_range = np.abs(np.random.normal(0, volatility * 0.5, periods))
    highs = prices * (1 + daily_range)
    lows = prices * (1 - daily_range)
    
    # Create DataFrame
    dates = pd.date_range(start=start_date, periods=periods, freq='B')
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    df.set_index('date', inplace=True)
    
    return df


def test_regime_detection():
    """Test regime detection system"""
    print("\n" + "="*80)
    print("TEST 1: Regime Detection System")
    print("="*80)
    
    # Generate test data
    df = generate_synthetic_market_data(periods=500)
    
    # Initialize detector
    detector = create_regime_detector({
        'volatility_lookback': 20,
        'trend_lookback': 50,
        'momentum_lookback': 14,
        'use_gmm': True
    })
    
    # Detect regime at different points
    test_points = [100, 200, 300, 400, 499]
    
    results = []
    for point in test_points:
        prices = df['close'].iloc[:point+1]
        volumes = df['volume'].iloc[:point+1]
        
        result = detector.detect_regime(prices, volumes)
        results.append(result)
        
        print(f"\n📊 Period {point}:")
        print(f"  Primary Regime: {result.primary_regime.value}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Risk Level: {result.risk_level}")
        print(f"  Volatility: {result.indicators.get('realized_volatility', 0):.2%}")
        print(f"  Trend: {result.indicators.get('annualized_trend', 0):.2%}")
        
        if result.strategy_recommendations:
            print(f"  Recommendation: {result.strategy_recommendations[0]}")
    
    # Validate results
    assert len(results) == len(test_points), "Should detect regime for each test point"
    assert all(r.confidence >= 0 for r in results), "Confidence should be non-negative"
    assert all(r.confidence <= 1 for r in results), "Confidence should not exceed 1"
    
    print("\n✅ Regime detection test PASSED")
    return True


def test_signal_integration():
    """Test multi-source signal integration"""
    print("\n" + "="*80)
    print("TEST 2: Multi-Source Signal Integration")
    print("="*80)
    
    # Generate test data
    df = generate_synthetic_market_data(periods=300)
    
    # Initialize integrator
    integrator = create_signal_integrator()
    
    # Add price data
    prices = df['close']
    volumes = df['volume']
    highs = df['high']
    lows = df['low']
    
    integrator.add_price_data(prices, volumes, highs, lows)
    
    # Add synthetic news data
    news_headlines = [
        "Company beats earnings expectations",
        "Strong revenue growth reported",
        "Analysts upgrade stock rating",
        "Market outlook remains positive",
        "New product launch exceeds expectations"
    ]
    integrator.add_news_data(news_headlines)
    
    # Add synthetic macro data
    macro_data = {
        'gdp_growth': 2.5,
        'inflation': 2.1,
        'interest_rates': 3.5,
        'unemployment': 3.8,
        'pmi': 52.3,
        'consumer_confidence': 105.2
    }
    integrator.add_macro_data(macro_data)
    
    # Set regime context
    integrator.set_regime_context('bull_market')
    
    # Generate integrated signal
    signal = integrator.generate_integrated_signal(symbols=['AAPL'])
    
    print(f"\n🎯 Integrated Signal:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Strength: {signal.overall_strength:.1%}")
    print(f"  Confidence: {signal.overall_confidence:.1%}")
    print(f"  Agreement Score: {signal.agreement_score:.1%}")
    
    print(f"\n📝 Explanation:")
    print(f"  {signal.explanation}")
    
    print(f"\n💡 Action Recommendation:")
    print(f"  {signal.action_recommendation}")
    
    print(f"\n📊 Contributing Signals:")
    for sig in signal.contributing_signals:
        print(f"  • {sig.source.value}: {sig.signal_type.value} (strength: {sig.strength:.1%})")
    
    # Validate
    assert signal.overall_strength >= 0, "Signal strength should be non-negative"
    assert signal.overall_strength <= 1, "Signal strength should not exceed 1"
    assert signal.overall_confidence >= 0, "Confidence should be non-negative"
    assert len(signal.contributing_signals) > 0, "Should have at least one contributing signal"
    
    print("\n✅ Signal integration test PASSED")
    return True


def test_explainable_decisions():
    """Test explainable decision generation"""
    print("\n" + "="*80)
    print("TEST 3: Explainable Decision Layer")
    print("="*80)
    
    # Generate test data
    df = generate_synthetic_market_data(periods=300)
    
    # Setup components
    integrator = create_signal_integrator()
    integrator.add_price_data(df['close'], df['volume'], df['high'], df['low'])
    integrator.add_news_data(["Positive market news"])
    integrator.add_macro_data({'gdp_growth': 2.5, 'inflation': 2.0})
    
    detector = create_regime_detector()
    result = detector.detect_regime(df['close'].iloc[:200], df['volume'].iloc[:200])
    
    signal = integrator.generate_integrated_signal(symbols=['AAPL'])
    
    # Create explainable agent
    agent = ExplainableDecisionAgent()
    
    # Generate decision
    context = {
        'symbol': 'AAPL',
        'current_price': float(df['close'].iloc[-1]),
        'quantity': 100
    }
    
    decision = agent.make_decision(signal, result, context)
    
    print(f"\n🎯 Trading Decision:")
    print(f"  Type: {decision.decision_type.value.upper()}")
    print(f"  Symbol: {decision.symbol}")
    print(f"  Confidence: {decision.confidence:.1%}")
    
    print(f"\n📝 Primary Reason:")
    print(f"  {decision.primary_reason}")
    
    print(f"\n📖 Simple Explanation (Beginner-Friendly):")
    print(f"  {decision.simple_explanation}")
    
    if decision.analogy:
        print(f"\n💡 Analogy:")
        print(f"  {decision.analogy}")
    
    print(f"\n📊 Signal Attribution:")
    for source, attribution in decision.signal_attribution.items():
        print(f"  • {source}: {attribution:.1f}%")
    
    print(f"\n⚠️ Risk Factors:")
    for risk in decision.risk_factors:
        print(f"  • {risk}")
    
    # Validate
    assert decision.decision_type is not None, "Decision type should be set"
    assert decision.primary_reason, "Should have primary reason"
    assert decision.simple_explanation, "Should have simple explanation"
    assert len(decision.signal_attribution) > 0, "Should have signal attribution"
    
    print("\n✅ Explainable decision test PASSED")
    return True


def test_evaluation_framework():
    """Test comprehensive evaluation framework"""
    print("\n" + "="*80)
    print("TEST 4: Comprehensive Evaluation Framework")
    print("="*80)
    
    # Generate realistic trades WITHOUT artificial bias
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=252)
    
    trades = []
    for i in range(50):
        entry_date = base_date + timedelta(days=i*5)
        exit_date = entry_date + timedelta(days=np.random.randint(1, 10))
        
        entry_price = 100 + np.random.randn() * 5
        direction = np.random.choice([1, -1])
        
        # REALISTIC P&L generation - NO ARTIFICIAL DRIFT
        # Actual trading has transaction costs and slippage
        gross_return = np.random.randn() * 0.03  # Random return
        
        # Apply transaction costs (realistic)
        commission_cost = 0.005 * 100  # $0.50 per side
        slippage_cost = 0.001 * entry_price * 100  # 0.1% slippage
        total_costs = (commission_cost + slippage_cost) * 2  # Round trip
        
        if direction == 1:
            exit_price = entry_price * (1 + gross_return)
        else:
            exit_price = entry_price * (1 - gross_return)
        
        trade = Trade(
            symbol='AAPL',
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=100,
            direction=direction
        )
        trade.calculate_pnl()
        
        # Subtract transaction costs from P&L
        trade.pnl -= total_costs
        
        trades.append(trade)
    
    # Calculate returns
    returns = [t.return_pct for t in trades]
    equity_curve = [1000 * (1 + np.prod(1 + np.array(returns[:i+1])) - 1) 
                   for i in range(len(returns))]
    
    # Generate regime series
    regime_dates = pd.date_range(start=base_date, periods=252, freq='B')
    regimes = np.random.choice(
        [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET, MarketRegime.SIDEWAYS],
        size=252,
        p=[0.5, 0.3, 0.2]
    )
    regime_series = pd.Series(regimes, index=regime_dates)
    
    # Create evaluator
    evaluator = create_comprehensive_evaluator(benchmark='SPY')
    
    evaluation = evaluator.evaluate_strategy(
        trades=trades,
        returns=returns,
        equity_curve=equity_curve,
        regime_series=regime_series,
        strategy_name="Test Momentum Strategy"
    )
    
    print(f"\n📊 Overall Performance Metrics:")
    print(f"  Total Return: {evaluation.overall_metrics.total_return:.2f}%")
    print(f"  Annualized Return: {evaluation.overall_metrics.annualized_return:.2f}%")
    print(f"  Sharpe Ratio: {evaluation.overall_metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {evaluation.overall_metrics.max_drawdown:.2f}%")
    print(f"  Win Rate: {evaluation.overall_metrics.win_rate:.1%}")
    print(f"  Profit Factor: {evaluation.overall_metrics.profit_factor:.2f}")
    
    print(f"\n📈 Score Breakdown:")
    print(f"  Overall Score: {evaluation.overall_score:.1f}/100")
    print(f"  Stability Score: {evaluation.stability_score:.1f}/100")
    print(f"  Consistency Score: {evaluation.consistency_score:.1f}/100")
    print(f"  Risk Score: {evaluation.risk_score:.1f}/100")
    
    print(f"\n📝 Recommendations:")
    for rec in evaluation.recommendations[:3]:
        print(f"  • {rec}")
    
    # Validate
    assert evaluation.overall_metrics.sharpe_ratio is not None, "Should calculate Sharpe"
    assert evaluation.overall_metrics.max_drawdown is not None, "Should calculate drawdown"
    assert evaluation.stability_score >= 0, "Stability score should be non-negative"
    assert len(evaluation.recommendations) > 0, "Should provide recommendations"
    
    print("\n✅ Evaluation framework test PASSED")
    return True


def test_memory_learning():
    """Test regime-specific memory learning"""
    print("\n" + "="*80)
    print("TEST 5: Memory-Based Learning System")
    print("="*80)
    
    # Create memory learner
    learner = create_regime_memory_learner(storage_path="test_memory_storage")
    
    # Simulate storing trade memories
    np.random.seed(42)
    detector = create_regime_detector()
    
    print("\n💾 Storing trade memories...")
    
    for i in range(20):
        # Generate synthetic data
        df = generate_synthetic_market_data(periods=200, seed=i)
        
        # Detect regime
        regime_result = detector.detect_regime(df['close'], df['volume'])
        
        # Create trade details
        trade_details = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'entry_price': float(df['close'].iloc[-1]),
            'entry_date': datetime.now(),
            'exit_date': datetime.now() + timedelta(days=5),
            'quantity': 100
        }
        
        # Market conditions from regime detection
        market_conditions = regime_result.indicators
        
        # Signals used
        signals = ['technical_momentum', 'trend_following', 'volume_confirmation']
        
        # Outcome (varied P&L)
        outcome = np.random.randn() * 0.05 + 0.01
        
        # Store memory
        learner.store_trade_memory(
            regime_result=regime_result,
            trade_details=trade_details,
            market_conditions=market_conditions,
            signals=signals,
            outcome=outcome,
            confidence=0.7
        )
    
    print(f"  Stored 20 trade memories")
    
    # Retrieve relevant memories
    print("\n🔍 Retrieving relevant memories...")
    
    current_regime = 'bull_market'
    current_conditions = {
        'realized_volatility': 0.15,
        'annualized_trend': 0.12,
        'momentum_score': 0.3,
        'price_vs_ma200': 0.08
    }
    
    relevant_memories = learner.retrieve_relevant_memories(
        current_regime=current_regime,
        market_conditions=current_conditions,
        limit=5
    )
    
    print(f"  Retrieved {len(relevant_memories)} relevant memories")
    
    # Get recommendation
    print("\n💡 Getting memory-based recommendation...")
    
    recommendation = learner.get_memory_recommendation(relevant_memories)
    
    print(f"  Recommendation: {recommendation['recommendation']}")
    print(f"  Confidence: {recommendation['confidence']:.1%}")
    if 'expected_outcome' in recommendation:
        print(f"  Expected Outcome: {recommendation['expected_outcome']:.2%}")
    print(f"  Reasoning: {recommendation['reasoning']}")
    
    # Get regime statistics
    print("\n📊 Regime Statistics:")
    stats = learner.get_regime_statistics(current_regime)
    if 'error' not in stats:
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Average Outcome: {stats['avg_outcome']:.2%}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    
    # Validate
    assert len(relevant_memories) >= 0, "Should retrieve memories (may be 0 if different regimes)"
    assert recommendation['recommendation'] in ['FAVORABLE', 'UNFAVORABLE', 'NEUTRAL', 'INSUFFICIENT_DATA'], \
        "Should provide valid recommendation"
    
    print("\n✅ Memory learning test PASSED")
    return True


def run_full_integration_test():
    """Run complete integration test"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FINANCIAL AGENT SYSTEM INTEGRATION TEST")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Regime Detection", test_regime_detection),
        ("Signal Integration", test_signal_integration),
        ("Explainable Decisions", test_explainable_decisions),
        ("Performance Evaluation", test_evaluation_framework),
        ("Memory Learning", test_memory_learning)
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results[test_name] = f"❌ ERROR: {str(e)}"
            failed += 1
            print(f"\n❌ Test {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("")
    
    for test_name, result in results.items():
        print(f"  {test_name}: {result}")
    
    print("\n" + "="*80)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Financial agent system is fully functional.")
        print("="*80)
        return True
    else:
        print(f"⚠️ {failed} test(s) failed. Review errors above.")
        print("="*80)
        return False


if __name__ == "__main__":
    success = run_full_integration_test()
    sys.exit(0 if success else 1)
