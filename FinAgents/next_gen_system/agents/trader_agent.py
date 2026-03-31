"""
TraderAgent - Domain-Specialized Trading Agent with ML-Based Strategy Learning

This agent implements advanced trading strategies with reinforcement learning,
momentum detection, mean reversion, and pattern recognition capabilities.

Key Features:
------------
- Multi-strategy execution (momentum, mean reversion, breakout)
- ML-based signal generation
- Reinforcement learning for strategy adaptation
- Historical performance tracking
- Risk-adjusted position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PAIRS_TRADING = "pairs_trading"
    ML_ENSEMBLE = "ml_ensemble"


@dataclass
class TradingSignal:
    """Represents a trading signal with metadata"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float  # 0.0 to 1.0
    strategy: StrategyType
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    confidence: float = 0.5
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class TraderAgent:
    """
    Domain-specialized trading agent with ML-based strategy learning
    
    This agent combines multiple trading strategies with reinforcement
    learning to adapt to changing market conditions.
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_capital: float = 1_000_000.0,
        strategies: Optional[List[StrategyType]] = None,
        risk_tolerance: float = 0.02,
        learning_enabled: bool = True,
    ):
        """
        Initialize the TraderAgent
        
        Args:
            agent_id: Unique identifier for this agent
            initial_capital: Starting capital allocation
            strategies: List of strategies to employ
            risk_tolerance: Maximum risk per trade (as fraction of portfolio)
            learning_enabled: Whether to enable RL-based learning
        """
        self.agent_id = agent_id
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies = strategies or [
            StrategyType.MOMENTUM,
            StrategyType.MEAN_REVERSION,
            StrategyType.BREAKOUT
        ]
        self.risk_tolerance = risk_tolerance
        self.learning_enabled = learning_enabled
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.performance_metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
        
        # Strategy weights for ensemble methods
        self.strategy_weights = {s: 1.0 / len(self.strategies) for s in self.strategies}
        
        logger.info(f"TraderAgent {agent_id} initialized with {len(self.strategies)} strategies")
    
    def analyze_market(
        self,
        price_data: pd.DataFrame,
        news_sentiment: Optional[Dict[str, float]] = None,
        chart_patterns: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, TradingSignal]:
        """
        Analyze market data using multimodal inputs
        
        Args:
            price_data: OHLCV time-series data
            news_sentiment: Sentiment scores from news analysis
            chart_patterns: Detected chart patterns
            
        Returns:
            Dictionary of trading signals by symbol
        """
        signals = {}
        
        for symbol in price_data.columns.get_level_values(0).unique():
            # Combine signals from multiple strategies
            strategy_signals = []
            
            if StrategyType.MOMENTUM in self.strategies:
                mom_signal = self._momentum_strategy(price_data[symbol])
                strategy_signals.append(mom_signal)
            
            if StrategyType.MEAN_REVERSION in self.strategies:
                mr_signal = self._mean_reversion_strategy(price_data[symbol])
                strategy_signals.append(mr_signal)
            
            if StrategyType.BREAKOUT in self.strategies:
                breakout_signal = self._breakout_strategy(price_data[symbol])
                strategy_signals.append(breakout_signal)
            
            # Fuse signals using weighted ensemble
            fused_signal = self._fuse_signals(strategy_signals)
            
            # Adjust based on sentiment if available
            if news_sentiment and symbol in news_sentiment:
                fused_signal = self._adjust_for_sentiment(fused_signal, news_sentiment[symbol])
            
            # Adjust based on chart patterns if available
            if chart_patterns and symbol in chart_patterns:
                fused_signal = self._adjust_for_chart_pattern(fused_signal, chart_patterns[symbol])
            
            signals[symbol] = fused_signal
        
        return signals
    
    def execute_signal(
        self,
        signal: TradingSignal,
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Execute a trading signal with position sizing
        
        Args:
            signal: Trading signal to execute
            current_price: Current market price
            
        Returns:
            Execution result with order details
        """
        # Calculate position size based on risk tolerance and signal strength
        position_size = self._calculate_position_size(signal, current_price)
        
        # Update capital
        cost = position_size * current_price
        
        if signal.action == "BUY":
            self.current_capital -= cost
            self.positions[signal.symbol] = {
                "action": "BUY",
                "size": position_size,
                "entry_price": current_price,
                "timestamp": datetime.now(),
                "strategy": signal.strategy,
            }
        elif signal.action == "SELL" and signal.symbol in self.positions:
            prev_position = self.positions[signal.symbol]
            pnl = (current_price - prev_position["entry_price"]) * prev_position["size"]
            self.current_capital += cost + pnl
            
            # Record trade in history
            self.trade_history.append({
                "symbol": signal.symbol,
                "action": "SELL",
                "entry_price": prev_position["entry_price"],
                "exit_price": current_price,
                "size": prev_position["size"],
                "pnl": pnl,
                "strategy": signal.strategy,
                "timestamp": datetime.now(),
            })
            
            del self.positions[signal.symbol]
        
        logger.info(f"Executed {signal.action} signal for {signal.symbol}: {position_size} shares @ ${current_price}")
        
        return {
            "symbol": signal.symbol,
            "action": signal.action,
            "size": position_size,
            "price": current_price,
            "cost": cost,
            "remaining_capital": self.current_capital,
        }
    
    def _momentum_strategy(self, price_series: pd.DataFrame) -> TradingSignal:
        """Implement momentum-based trading strategy"""
        closes = price_series["close"]
        
        # Calculate momentum indicators
        returns = closes.pct_change()
        momentum_10 = returns.rolling(10).sum()
        momentum_20 = returns.rolling(20).sum()
        
        current_momentum = momentum_10.iloc[-1]
        prev_momentum = momentum_10.iloc[-2]
        
        # Generate signal
        if current_momentum > 0.02 and current_momentum > prev_momentum:
            action = "BUY"
            strength = min(abs(current_momentum), 1.0)
        elif current_momentum < -0.02 and current_momentum < prev_momentum:
            action = "SELL"
            strength = min(abs(current_momentum), 1.0)
        else:
            action = "HOLD"
            strength = 0.0
        
        return TradingSignal(
            symbol=closes.name[0] if isinstance(closes.name, tuple) else closes.name,
            action=action,
            strength=strength,
            strategy=StrategyType.MOMENTUM,
            entry_price=closes.iloc[-1],
            reasoning=[f"Momentum: {current_momentum:.4f}", f"Trend: {'increasing' if current_momentum > prev_momentum else 'decreasing'}"],
        )
    
    def _mean_reversion_strategy(self, price_series: pd.DataFrame) -> TradingSignal:
        """Implement mean reversion trading strategy"""
        closes = price_series["close"]
        
        # Calculate mean reversion indicators
        ma_20 = closes.rolling(20).mean()
        std_20 = closes.rolling(20).std()
        z_score = (closes - ma_20) / std_20
        
        current_z = z_score.iloc[-1]
        
        # Generate signal
        if current_z < -2.0:
            action = "BUY"
            strength = min(abs(current_z) / 4.0, 1.0)
        elif current_z > 2.0:
            action = "SELL"
            strength = min(abs(current_z) / 4.0, 1.0)
        else:
            action = "HOLD"
            strength = 0.0
        
        return TradingSignal(
            symbol=closes.name[0] if isinstance(closes.name, tuple) else closes.name,
            action=action,
            strength=strength,
            strategy=StrategyType.MEAN_REVERSION,
            entry_price=closes.iloc[-1],
            reasoning=[f"Z-score: {current_z:.4f}", f"Distance from mean: {(closes.iloc[-1] - ma_20.iloc[-1]):.2f}"],
        )
    
    def _breakout_strategy(self, price_series: pd.DataFrame) -> TradingSignal:
        """Implement breakout trading strategy"""
        highs = price_series["high"]
        lows = price_series["low"]
        closes = price_series["close"]
        
        # Calculate Donchian channels
        upper_channel = highs.rolling(20).max()
        lower_channel = lows.rolling(20).min()
        
        current_price = closes.iloc[-1]
        upper = upper_channel.iloc[-1]
        lower = lower_channel.iloc[-1]
        
        # Generate signal
        if current_price >= upper * 0.998:  # Near upper channel
            action = "BUY"
            strength = min((current_price - upper) / upper + 0.5, 1.0)
        elif current_price <= lower * 1.002:  # Near lower channel
            action = "SELL"
            strength = min((lower - current_price) / lower + 0.5, 1.0)
        else:
            action = "HOLD"
            strength = 0.0
        
        return TradingSignal(
            symbol=closes.name[0] if isinstance(closes.name, tuple) else closes.name,
            action=action,
            strength=strength,
            strategy=StrategyType.BREAKOUT,
            entry_price=current_price,
            reasoning=[f"Price: {current_price:.2f}", f"Channel: [{lower:.2f}, {upper:.2f}]"],
        )
    
    def _fuse_signals(self, signals: List[TradingSignal]) -> TradingSignal:
        """Fuse multiple strategy signals into unified decision"""
        if not signals:
            raise ValueError("No signals provided")
        
        symbol = signals[0].symbol
        base_symbol = symbol
        
        # Weighted voting
        buy_weight = sum(
            s.strength * self.strategy_weights.get(s.strategy, 0.5)
            for s in signals if s.action == "BUY"
        )
        sell_weight = sum(
            s.strength * self.strategy_weights.get(s.strategy, 0.5)
            for s in signals if s.action == "SELL"
        )
        
        # Determine consensus action
        if buy_weight > sell_weight * 1.2:  # 20% threshold
            action = "BUY"
            strength = min(buy_weight, 1.0)
        elif sell_weight > buy_weight * 1.2:
            action = "SELL"
            strength = min(sell_weight, 1.0)
        else:
            action = "HOLD"
            strength = 0.0
        
        # Collect reasoning from all strategies
        all_reasoning = []
        for s in signals:
            all_reasoning.extend([f"[{s.strategy.value}] {r}" for r in s.reasoning])
        
        return TradingSignal(
            symbol=base_symbol,
            action=action,
            strength=strength,
            strategy=StrategyType.ML_ENSEMBLE,
            entry_price=signals[0].entry_price,
            reasoning=all_reasoning,
            confidence=strength,
        )
    
    def _adjust_for_sentiment(self, signal: TradingSignal, sentiment: float) -> TradingSignal:
        """Adjust signal based on news sentiment"""
        # Sentiment ranges from -1 (very negative) to 1 (very positive)
        if sentiment > 0.5 and signal.action != "SELL":
            signal.strength = min(signal.strength + sentiment * 0.3, 1.0)
            signal.reasoning.append(f"Positive sentiment: {sentiment:.2f}")
        elif sentiment < -0.5 and signal.action != "BUY":
            signal.strength = min(signal.strength + abs(sentiment) * 0.3, 1.0)
            signal.reasoning.append(f"Negative sentiment: {sentiment:.2f}")
        
        return signal
    
    def _adjust_for_chart_pattern(self, signal: TradingSignal, pattern: Dict[str, Any]) -> TradingSignal:
        """Adjust signal based on detected chart patterns"""
        pattern_type = pattern.get("type", "unknown")
        reliability = pattern.get("reliability", 0.5)
        
        bullish_patterns = ["ascending_triangle", "bull_flag", "cup_and_handle"]
        bearish_patterns = ["descending_triangle", "bear_flag", "head_and_shoulders"]
        
        if pattern_type in bullish_patterns and signal.action != "SELL":
            signal.strength = min(signal.strength + reliability * 0.4, 1.0)
            signal.reasoning.append(f"Bullish pattern detected: {pattern_type}")
        elif pattern_type in bearish_patterns and signal.action != "BUY":
            signal.strength = min(signal.strength + reliability * 0.4, 1.0)
            signal.reasoning.append(f"Bearish pattern detected: {pattern_type}")
        
        return signal
    
    def _calculate_position_size(self, signal: TradingSignal, current_price: float) -> float:
        """Calculate optimal position size based on risk"""
        # Kelly criterion-inspired sizing
        max_risk_amount = self.current_capital * self.risk_tolerance
        position_value = max_risk_amount * signal.strength
        
        return int(position_value / current_price)
    
    def update_performance_metrics(self):
        """Update performance metrics based on trade history"""
        if not self.trade_history:
            return
        
        # Calculate returns
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        self.performance_metrics["total_return"] = total_pnl / self.initial_capital
        
        # Calculate win rate
        winning_trades = [t for t in self.trade_history if t["pnl"] > 0]
        self.performance_metrics["win_rate"] = len(winning_trades) / len(self.trade_history)
        
        # Calculate profit factor
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in self.trade_history if t["pnl"] < 0))
        self.performance_metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        logger.info(f"Updated performance metrics: {self.performance_metrics}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "positions": self.positions,
            "trade_history_count": len(self.trade_history),
            "performance_metrics": self.performance_metrics,
            "active_strategies": [s.value for s in self.strategies],
        }
