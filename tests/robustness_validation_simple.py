"""
Simplified Robustness Validation - No emoji issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobustnessValidation")

print("="*80)
print("PHASE 3: ROBUSTNESS & QUANT VALIDATION (SIMPLIFIED)")
print("="*80)

# TEST 1: Real Market Data
print("\n[TEST 1] Real Market Data Integration...")
try:
    from backend.market.real_data_provider import create_market_data_fetcher
    
    fetcher = create_market_data_fetcher()
    spy = fetcher.fetch("^GSPC", period="2y")
    
    print(f"  Records: {len(spy)}")
    print(f"  Latest Close: ${spy['Close'].iloc[-1]:.2f}")
    assert len(spy) > 100
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# TEST 2: Walk-Forward Framework
print("\n[TEST 2] Walk-Forward Analysis Framework...")
try:
    from backend.analytics.walk_forward_analysis import create_walk_forward_analyzer
    
    analyzer = create_walk_forward_analyzer(
        train_window_days=126,
        test_window_days=63,
        step_days=63
    )
    
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    periods = analyzer.generate_periods(start_date, end_date)
    
    print(f"  Generated {len(periods)} periods")
    assert len(periods) >= 3
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# TEST 3: Monte Carlo Simulation
print("\n[TEST 3] Monte Carlo Simulation...")
try:
    from backend.analytics.walk_forward_analysis import create_monte_carlo_simulator
    from backend.analytics.evaluation_framework import Trade
    
    simulator = create_monte_carlo_simulator(n_simulations=500)
    
    trades = []
    for i in range(50):
        entry_date = datetime.now() - timedelta(days=i*5)
        trade = Trade(
            symbol='TEST',
            entry_date=entry_date,
            exit_date=entry_date + timedelta(days=5),
            entry_price=100,
            exit_price=100 * (1 + np.random.randn() * 0.03),
            quantity=100,
            direction=1
        )
        trade.calculate_pnl()
        trades.append(trade)
    
    results = simulator.bootstrap_analysis(trades)
    print(f"  Mean Return: {results['mean_return']:.2%}")
    print(f"  Prob Profit: {results['prob_profit']:.1%}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# TEST 4: Multi-Market Testing
print("\n[TEST 4] Multi-Market Testing...")
try:
    fetcher = create_market_data_fetcher()
    
    assets = ['AAPL', '^GSPC', 'BTC-USD']
    for symbol in assets:
        df = fetcher.fetch(symbol, period="1y")
        print(f"  {symbol}: {len(df)} records")
    
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# TEST 5: Realistic Metrics with Costs
print("\n[TEST 5] Realistic Performance Metrics (With Costs)...")
try:
    from backend.risk.transaction_cost_model import create_cost_model
    
    cost_model = create_cost_model()
    
    # Test cost calculation
    costs = cost_model.estimate_costs(price=100, quantity=100, direction=1)
    
    print(f"  Commission: ${costs.commission:.2f}")
    print(f"  Slippage: ${costs.slippage:.2f}")
    print(f"  Total Cost: ${costs.total_cost:.2f}")
    print(f"  Breakeven Move: {cost_model.get_breakeven_move(100, 100):.2f}%")
    
    # Verify costs are realistic (> 0)
    assert costs.total_cost > 0, "Costs should be positive"
    assert costs.total_cost < 10, "Costs should be reasonable"
    
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "="*80)
print("PHASE 3 VALIDATION COMPLETE")
print("="*80)
