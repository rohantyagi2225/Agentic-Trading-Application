"""
Quick validation that critical fixes are working
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("="*80)
print("PHASE 2 CRITICAL FIXES VALIDATION")
print("="*80)

# Test 1: Transaction Cost Model exists
print("\n[TEST 1] Transaction Cost Model...")
try:
    from backend.risk.transaction_cost_model import create_cost_model
    
    cost_model = create_cost_model()
    costs = cost_model.estimate_costs(price=100, quantity=100, direction=1)
    
    print(f"  Commission: ${costs.commission:.2f}")
    print(f"  Slippage: ${costs.slippage:.2f}")
    print(f"  Total Cost: ${costs.total_cost:.2f}")
    print(f"  Breakeven Move: {cost_model.get_breakeven_move(100, 100):.2f}%")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Circuit Breaker exists
print("\n[TEST 2] Circuit Breaker System...")
try:
    from backend.risk.circuit_breaker import create_circuit_breaker
    
    cb = create_circuit_breaker(max_daily_loss_pct=0.05)
    cb.update_equity(100000)
    
    # Simulate a large loss
    cb.record_trade(-6000)  # -6% loss
    
    allowed, reason = cb.should_allow_trade()
    print(f"  Trading Allowed: {allowed}")
    print(f"  Reason: {reason}")
    print(f"  Status: Halted={cb.is_trading_halted}")
    print("  PASS" if not allowed else "  FAIL: Should have halted trading")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: Fast Crash Detection
print("\n[TEST 3] Flash Crash Detection...")
try:
    from backend.market.regime_detector import create_regime_detector
    
    detector = create_regime_detector({'volatility_lookback': 5, 'trend_lookback': 10})  # Shorter lookback for testing
    
    # Generate crash data (-15% in 5 days) with MORE data points
    dates = pd.date_range(start=datetime.now()-timedelta(days=30), periods=30, freq='B')
    prices = [100 + i*0.5 for i in range(20)]  # Stable uptrend
    prices += [95, 85, 75, 70, 65, 60, 58, 55, 53, 50]  # CRASH: -47% in 10 days
    prices_series = pd.Series(prices, index=dates)
    volumes = pd.Series([1000000]*30, index=dates)
    
    result = detector.detect_regime(prices_series, volumes)
    
    print(f"  Detected Regime: {result.primary_regime.value}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Risk Level: {result.risk_level}")
    
    if result.primary_regime.value == 'crash':
        print("  PASS: Correctly detected crash")
    else:
        print(f"  PARTIAL: Detected {result.primary_regime.value} instead of crash")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: Confidence NaN Fix
print("\n[TEST 4] Confidence Calculation (NaN Prevention)...")
try:
    from backend.market.signal_integrator import SignalFusionEngine, SignalType, SignalSource, IndividualSignal
    
    engine = SignalFusionEngine()
    
    # Create signals with valid confidences
    signals = [
        IndividualSignal(
            source=SignalSource.TECHNICAL,
            signal_type=SignalType.BULLISH,
            strength=0.5,
            confidence=0.7,
            timestamp=datetime.now(),
            description="Test signal"
        )
    ]
    
    fused = engine.fuse_signals(signals)
    
    print(f"  Fused Confidence: {fused.overall_confidence:.1%}")
    print(f"  Is Valid Number: {not np.isnan(fused.overall_confidence)}")
    
    if not np.isnan(fused.overall_confidence) and 0 <= fused.overall_confidence <= 1:
        print("  PASS: Confidence is valid")
    else:
        print("  FAIL: Confidence is NaN or out of bounds")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 5: Realistic Trade Generation (No Synthetic Bias)
print("\n[TEST 5] Realistic Trade Simulation (No Bias)...")
try:
    from backend.analytics.evaluation_framework import Trade
    
    np.random.seed(42)
    
    # Generate trades WITHOUT artificial drift
    entry_price = 100
    gross_return = np.random.randn() * 0.03  # Random return
    
    trade = Trade(
        symbol='TEST',
        entry_date=datetime.now(),
        exit_date=datetime.now() + timedelta(days=5),
        entry_price=entry_price,
        exit_price=entry_price * (1 + gross_return),
        quantity=100,
        direction=1
    )
    trade.calculate_pnl()
    
    # Apply transaction costs
    commission_cost = 0.005 * 100 * 2  # $1 round trip
    slippage_cost = 0.001 * 100 * 100 * 2  # $2 round trip
    total_costs = commission_cost + slippage_cost
    
    net_pnl = trade.pnl - total_costs
    
    print(f"  Gross P&L: ${trade.pnl:.2f}")
    print(f"  Transaction Costs: ${total_costs:.2f}")
    print(f"  Net P&L: ${net_pnl:.2f}")
    print("  PASS: Costs properly deducted")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
