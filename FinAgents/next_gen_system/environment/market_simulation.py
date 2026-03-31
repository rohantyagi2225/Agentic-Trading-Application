"""
Market Simulation Environment
=============================

Realistic market simulation where agents interact with:
- Dynamic price movements
- External shocks (news events)
- Market impact from agent actions
- Performance tracking

Features:
- Agent-based market dynamics
- Event-driven simulations
- Realistic frictions (slippage, commissions)
- Comprehensive performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current state of the simulated market"""
    timestamp: datetime
    prices: Dict[str, float]
    returns: Dict[str, float]
    volatility: Dict[str, float]
    volumes: Dict[str, float]
    market_regime: str
    active_events: List[Dict[str, Any]]


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    var_95: float
    cvar_95: float


class MarketSimulationEnvironment:
    """
    Realistic market simulation environment
    
    Simulates market dynamics with agent interactions,
    external events, and comprehensive performance tracking.
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize market simulation
        
        Args:
            initial_capital: Starting capital
            commission_rate: Trading commission rate
            slippage_rate: Expected slippage
            risk_free_rate: Risk-free rate for Sharpe
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        
        # Market state
        self.base_prices: Dict[str, float] = {}
        self.price_history: List[Dict[str, float]] = []
        self.return_history: List[Dict[str, float]] = []
        
        # Portfolio tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.portfolio_values: List[float] = [initial_capital]
        self.trade_log: List[Dict[str, Any]] = []
        
        # Event system
        self.event_schedule: List[Dict[str, Any]] = []
        self.active_events: List[Dict[str, Any]] = []
        
        # Time tracking
        self.current_time = datetime.now()
        self.trading_days = 0
        
        logger.info(f"Market simulation initialized with ${initial_capital:,.0f}")
    
    def initialize_market(
        self,
        symbols: List[str],
        initial_prices: Optional[Dict[str, float]] = None,
    ):
        """Initialize market with symbols"""
        if initial_prices:
            self.base_prices = initial_prices
        else:
            # Default prices
            self.base_prices = {s: 100.0 for s in symbols}
        
        logger.info(f"Initialized market with {len(symbols)} symbols")
    
    def step(
        self,
        agent_actions: Optional[Dict[str, Dict[str, Any]]] = None,
        external_events: Optional[List[Dict[str, Any]]] = None,
    ) -> MarketState:
        """
        Advance simulation by one time step
        
        Args:
            agent_actions: Actions taken by agents
            external_events: External market events
            
        Returns:
            New market state
        """
        self.current_time += timedelta(days=1)
        self.trading_days += 1
        
        # Process external events
        if external_events:
            self._process_events(external_events)
        
        # Calculate natural price movements
        new_prices = self._simulate_price_movement(agent_actions)
        
        # Apply event impacts
        for event in self.active_events:
            new_prices = self._apply_event_impact(new_prices, event)
        
        # Update history
        self.price_history.append(new_prices.copy())
        
        # Calculate returns
        if len(self.price_history) > 1:
            prev_prices = self.price_history[-2]
            returns = {
                s: (new_prices[s] - prev_prices[s]) / prev_prices[s]
                for s in symbols if s in new_prices and s in prev_prices
            }
            self.return_history.append(returns)
        
        # Update portfolio value
        self._update_portfolio_value(new_prices)
        
        # Clean up expired events
        self.active_events = [e for e in self.active_events if e.get("end_time", datetime.max) > self.current_time]
        
        # Determine market regime
        market_regime = self._determine_market_regime()
        
        return MarketState(
            timestamp=self.current_time,
            prices=new_prices,
            returns=returns if len(self.price_history) > 1 else {},
            volatility=self._calculate_volatility(),
            volumes=self._simulate_volumes(),
            market_regime=market_regime,
            active_events=self.active_events.copy(),
        )
    
    def _simulate_price_movement(
        self,
        agent_actions: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, float]:
        """Simulate natural price movement with agent impact"""
        new_prices = {}
        
        for symbol, base_price in self.base_prices.items():
            # Base volatility
            daily_vol = 0.02  # 2% daily volatility
            
            # Random shock
            shock = np.random.normal(0, daily_vol)
            
            # Agent impact
            impact = 0
            if agent_actions and symbol in agent_actions:
                action = agent_actions[symbol]
                if action.get("action") == "BUY":
                    impact = 0.001 * action.get("size", 1)  # Small upward pressure
                elif action.get("action") == "SELL":
                    impact = -0.001 * action.get("size", 1)
            
            # Mean reversion
            mean_reversion = 0.0001 * (base_price - self.base_prices[symbol])
            
            # Calculate new price
            new_price = base_price * (1 + shock + impact + mean_reversion)
            new_price = max(new_price, 1.0)  # Floor at $1
            
            new_prices[symbol] = new_price
        
        # Update base prices for next iteration
        self.base_prices = new_prices.copy()
        
        return new_prices
    
    def _process_events(self, events: List[Dict[str, Any]]):
        """Process external events"""
        for event in events:
            event["start_time"] = self.current_time
            event["end_time"] = self.current_time + timedelta(days=event.get("duration_days", 1))
            self.active_events.append(event)
        
        logger.info(f"Processed {len(events)} events")
    
    def _apply_event_impact(
        self,
        prices: Dict[str, float],
        event: Dict[str, Any],
    ) -> Dict[str, float]:
        """Apply event impact to prices"""
        impacted_symbols = event.get("symbols", [])
        impact_magnitude = event.get("impact", 0)
        
        for symbol in impacted_symbols:
            if symbol in prices:
                prices[symbol] *= (1 + impact_magnitude)
        
        return prices
    
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current prices"""
        position_value = 0
        
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                position_value += pos["size"] * current_prices[symbol]
        
        self.current_capital = position_value + getattr(self, "cash_position", 0)
        self.portfolio_values.append(self.current_capital)
    
    def _determine_market_regime(self) -> str:
        """Determine current market regime"""
        if not self.return_history:
            return "neutral"
        
        recent_returns = list(self.return_history[-20:]) if len(self.return_history) >= 20 else list(self.return_history)
        
        avg_return = np.mean([r for ret_dict in recent_returns for r in ret_dict.values()])
        avg_vol = np.std([r for ret_dict in recent_returns for r in ret_dict.values()])
        
        if avg_return > 0.001 and avg_vol < 0.02:
            return "bull_low_vol"
        elif avg_return > 0.001:
            return "bull_high_vol"
        elif avg_return < -0.001 and avg_vol > 0.03:
            return "crisis"
        elif avg_return < -0.001:
            return "bear"
        else:
            return "neutral"
    
    def _calculate_volatility(self) -> Dict[str, float]:
        """Calculate rolling volatility"""
        if len(self.return_history) < 20:
            return {s: 0.02 for s in self.base_prices.keys()}
        
        recent = self.return_history[-20:]
        
        volatilities = {}
        for symbol in self.base_prices.keys():
            symbol_returns = [r.get(symbol, 0) for r in recent]
            volatilities[symbol] = np.std(symbol_returns) * np.sqrt(252)
        
        return volatilities
    
    def _simulate_volumes(self) -> Dict[str, float]:
        """Simulate trading volumes"""
        volumes = {}
        for symbol in self.base_prices.keys():
            base_volume = 1_000_000
            random_factor = np.random.uniform(0.5, 2.0)
            volumes[symbol] = base_volume * random_factor
        return volumes
    
    def execute_trade(
        self,
        symbol: str,
        action: str,
        size: int,
        price: float,
    ) -> Dict[str, Any]:
        """Execute a trade in the simulation"""
        # Calculate costs
        gross_value = size * price
        commission = gross_value * self.commission_rate
        slippage = gross_value * self.slippage_rate
        total_cost = commission + slippage
        
        if action == "BUY":
            total_cost = gross_value + total_cost
            if symbol in self.positions:
                # Average up
                old_size = self.positions[symbol]["size"]
                old_price = self.positions[symbol]["entry_price"]
                new_size = old_size + size
                new_price = ((old_size * old_price) + (size * price)) / new_size
                
                self.positions[symbol] = {
                    "size": new_size,
                    "entry_price": new_price,
                    "last_action": action,
                }
            else:
                self.positions[symbol] = {
                    "size": size,
                    "entry_price": price,
                    "last_action": action,
                }
            
            self.cash_position = getattr(self, "cash_position", self.initial_capital) - total_cost
            
        elif action == "SELL":
            if symbol in self.positions:
                pos = self.positions[symbol]
                realized_pnl = (price - pos["entry_price"]) * pos["size"]
                
                self.positions[symbol]["size"] -= size
                if self.positions[symbol]["size"] <= 0:
                    del self.positions[symbol]
                
                self.cash_position = getattr(self, "cash_position", self.initial_capital) + (gross_value - total_cost)
                
                # Log trade
                self.trade_log.append({
                    "timestamp": self.current_time,
                    "symbol": symbol,
                    "action": action,
                    "size": size,
                    "price": price,
                    "pnl": realized_pnl,
                    "costs": total_cost,
                })
        
        return {
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "total_cost": total_cost,
            "timestamp": self.current_time,
        }
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if len(self.portfolio_values) < 2:
            return self._empty_metrics()
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        n_years = self.trading_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate / 252
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = np.mean(excess_returns) / downside_dev * np.sqrt(252) if downside_dev > 0 else 0
        
        # Drawdown
        rolling_max = np.maximum.accumulate(values)
        drawdowns = (values - rolling_max) / rolling_max
        max_drawdown = abs(np.min(drawdowns))
        current_drawdown = abs(drawdowns[-1])
        
        # Win rate
        winning_trades = sum(1 for t in self.trade_log if t.get("pnl", 0) > 0)
        total_trades = len(self.trade_log)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profits = sum(t["pnl"] for t in self.trade_log if t["pnl"] > 0)
        gross_losses = abs(sum(t["pnl"] for t in self.trade_log if t["pnl"] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            var_95=var_95,
            cvar_95=cvar_95,
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics"""
        return PerformanceMetrics(
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            current_drawdown=0,
            win_rate=0,
            profit_factor=0,
            calmar_ratio=0,
            var_95=0,
            cvar_95=0,
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get simulation state"""
        return {
            "current_capital": self.current_capital,
            "trading_days": self.trading_days,
            "num_positions": len(self.positions),
            "portfolio_values_count": len(self.portfolio_values),
            "active_events": len(self.active_events),
            "trade_log_count": len(self.trade_log),
        }
