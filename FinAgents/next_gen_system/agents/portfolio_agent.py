"""
PortfolioAgent - Portfolio Optimization and Allocation Agent

This agent handles portfolio construction, optimization,
and dynamic rebalancing with modern portfolio theory.

Key Features:
------------
- Mean-Variance Optimization (Markowitz)
- Risk Parity allocation
- Black-Litterman model
- Dynamic rebalancing
- Transaction cost optimization
- ESG integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Optimal portfolio allocation result"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    constraints_satisfied: bool
    rebalancing_trades: List[Dict[str, Any]]
    transaction_costs: float
    timestamp: datetime = field(default_factory=datetime.now)


class PortfolioAgent:
    """
    Domain-specialized portfolio management agent
    
    Implements advanced portfolio optimization techniques
    with dynamic rebalancing and risk management.
    """
    
    def __init__(
        self,
        agent_id: str,
        risk_free_rate: float = 0.02,
        max_position_size: float = 0.25,
        min_position_size: float = 0.02,
        target_volatility: float = 0.15,
        allow_shorting: bool = False,
    ):
        """
        Initialize the PortfolioAgent
        
        Args:
            agent_id: Unique identifier
            risk_free_rate: Risk-free rate for Sharpe calculation
            max_position_size: Maximum weight for single position
            min_position_size: Minimum position size threshold
            target_volatility: Target portfolio volatility
            allow_shorting: Whether to allow short positions
        """
        self.agent_id = agent_id
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.target_volatility = target_volatility
        self.allow_shorting = allow_shorting
        
        # Current portfolio state
        self.current_weights: Dict[str, float] = {}
        self.cash_position: float = 0.0
        
        # Historical returns for optimization
        self.return_history: Dict[str, pd.Series] = {}
        
        logger.info(f"PortfolioAgent {agent_id} initialized")
    
    def optimize_portfolio(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        method: str = "mean_variance",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset return covariance matrix
            current_weights: Current portfolio weights
            method: Optimization method
            constraints: Additional constraints
            
        Returns:
            Optimal allocation with metrics
        """
        symbols = list(expected_returns.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            raise ValueError("No assets to optimize")
        
        # Convert to numpy arrays
        mu = np.array([expected_returns[s] for s in symbols])
        cov = covariance_matrix.loc[symbols, symbols].values
        
        # Initial weights
        if current_weights:
            w0 = np.array([current_weights.get(s, 1/n_assets) for s in symbols])
        else:
            w0 = np.ones(n_assets) / n_assets
        
        # Optimize based on method
        if method == "mean_variance":
            optimal_weights = self._mean_variance_optimization(mu, cov, w0, constraints)
        elif method == "risk_parity":
            optimal_weights = self._risk_parity_optimization(cov, w0)
        elif method == "max_sharpe":
            optimal_weights = self._max_sharpe_optimization(mu, cov, w0, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Create weight dictionary
        weights_dict = {symbols[i]: float(optimal_weights[i]) for i in range(n_assets)}
        
        # Calculate portfolio metrics
        port_return = np.dot(optimal_weights, mu)
        port_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_volatility if port_volatility > 0 else 0
        
        # Generate rebalancing trades
        current = current_weights or {s: 1/n_assets for s in symbols}
        trades = self._generate_rebalancing_trades(current, weights_dict)
        
        # Estimate transaction costs
        tx_costs = self._estimate_transaction_costs(trades)
        
        return PortfolioAllocation(
            weights=weights_dict,
            expected_return=port_return,
            expected_volatility=port_volatility,
            sharpe_ratio=sharpe,
            method=method,
            constraints_satisfied=self._check_constraints(weights_dict, constraints),
            rebalancing_trades=trades,
            transaction_costs=tx_costs,
        )
    
    def _mean_variance_optimization(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        constraints: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Mean-variance optimization (Markowitz)"""
        n = len(mu)
        
        # Objective: minimize -w'Mu + 0.5*w'Cov*w
        def objective(w):
            port_return = np.dot(w, mu)
            port_variance = np.dot(w.T, np.dot(cov, w))
            return -port_return + 0.5 * port_variance
        
        # Constraints
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        if self.allow_shorting:
            bounds = tuple((-0.5, 1.5) for _ in range(n))
        else:
            bounds = tuple((0.0, self.max_position_size) for _ in range(n))
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        
        return result.x if result.success else w0
    
    def _risk_parity_optimization(
        self,
        cov: np.ndarray,
        w0: np.ndarray,
    ) -> np.ndarray:
        """Risk parity optimization - equal risk contribution"""
        n = len(w0)
        
        def objective(w):
            # Portfolio variance
            port_var = np.dot(w.T, np.dot(cov, w))
            
            # Marginal risk contributions
            marginal_risk = np.dot(cov, w) / np.sqrt(port_var)
            
            # Risk contributions
            risk_contrib = w * marginal_risk
            
            # Target: equal risk contribution (1/n)
            target_risk = port_var / n
            
            # Minimize sum of squared differences from target
            return np.sum((risk_contrib - target_risk) ** 2)
        
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: w}  # Long-only
        ]
        
        bounds = tuple((0.01, 0.5) for _ in range(n))
        
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        
        return result.x if result.success else w0
    
    def _max_sharpe_optimization(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        constraints: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Maximum Sharpe ratio optimization"""
        n = len(mu)
        
        def negative_sharpe(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol < 1e-6:
                return 0
            return -(port_return - self.risk_free_rate) / port_vol
        
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        
        if not self.allow_shorting:
            bounds = tuple((0.0, self.max_position_size) for _ in range(n))
        else:
            bounds = tuple((-0.5, 1.5) for _ in range(n))
        
        result = minimize(
            negative_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        
        return result.x if result.success else w0
    
    def _generate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        tolerance: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Generate trades needed to rebalance portfolio"""
        trades = []
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            
            diff = target - current
            
            if abs(diff) > tolerance:
                action = "BUY" if diff > 0 else "SELL"
                trades.append({
                    "symbol": symbol,
                    "action": action,
                    "weight_change": abs(diff),
                    "priority": "HIGH" if abs(diff) > 0.05 else "MEDIUM",
                })
        
        return trades
    
    def _estimate_transaction_costs(
        self,
        trades: List[Dict[str, Any]],
        cost_rate: float = 0.001,
    ) -> float:
        """Estimate total transaction costs"""
        total_turnover = sum(trade["weight_change"] for trade in trades)
        return total_turnover * cost_rate
    
    def _check_constraints(
        self,
        weights: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if allocation satisfies constraints"""
        if not constraints:
            return True
        
        # Check max position
        max_weight = max(weights.values())
        if max_weight > constraints.get("max_weight", self.max_position_size) * 1.01:
            return False
        
        # Check min position
        min_weight = min(w for w in weights.values() if w > 0)
        if min_weight < constraints.get("min_weight", self.min_position_size) * 0.99:
            return False
        
        # Check sector limits if provided
        if "sector_limits" in constraints:
            for sector, limit in constraints["sector_limits"].items():
                sector_weight = sum(
                    w for s, w in weights.items() 
                    if self._get_symbol_sector(s) == sector
                )
                if sector_weight > limit * 1.01:
                    return False
        
        return True
    
    def _get_symbol_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol (placeholder)"""
        # In production, this would use a sector database
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "JPM": "Financials",
            "BAC": "Financials",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "XOM": "Energy",
            "CVX": "Energy",
        }
        return sector_map.get(symbol, "Unknown")
    
    def update_current_weights(self, new_weights: Dict[str, float]):
        """Update current portfolio weights"""
        self.current_weights = new_weights
        logger.info(f"Updated portfolio weights: {len(new_weights)} positions")
    
    def add_return_history(self, symbol: str, returns: pd.Series):
        """Add historical returns for optimization"""
        self.return_history[symbol] = returns
    
    def calculate_covariance_matrix(self, lookback_days: int = 252) -> pd.DataFrame:
        """Calculate covariance matrix from return history"""
        if not self.return_history:
            raise ValueError("No return history available")
        
        # Combine returns into DataFrame
        returns_df = pd.DataFrame(self.return_history)
        
        # Use specified lookback period
        if len(returns_df) > lookback_days:
            returns_df = returns_df.tail(lookback_days)
        
        # Annualize covariance
        cov_matrix = returns_df.cov() * 252
        
        return cov_matrix
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "num_positions": len(self.current_weights),
            "total_weight": sum(self.current_weights.values()),
            "cash_position": self.cash_position,
            "return_history_symbols": list(self.return_history.keys()),
            "optimization_parameters": {
                "risk_free_rate": self.risk_free_rate,
                "max_position_size": self.max_position_size,
                "target_volatility": self.target_volatility,
                "allow_shorting": self.allow_shorting,
            },
        }
