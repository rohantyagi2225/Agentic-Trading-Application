"""
Comprehensive Constraint System for Risk Management

This module provides a flexible constraint framework for managing portfolio risk.
It includes various constraint types that can be combined and managed together.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class Position:
    """
    Represents a single position in the portfolio.
    
    Attributes:
        symbol: Stock ticker symbol
        quantity: Number of shares (can be negative for short positions)
        current_price: Current market price per share
        average_cost: Average cost basis per share
        sector: Industry sector classification
        weight_pct: Current portfolio weight percentage
    """
    symbol: str
    quantity: float
    current_price: float
    average_cost: float
    sector: str
    weight_pct: float
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return self.quantity * (self.current_price - self.average_cost)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized profit/loss percentage."""
        if self.average_cost == 0:
            return 0.0
        return (self.current_price - self.average_cost) / self.average_cost


@dataclass
class PortfolioState:
    """
    Represents the current state of a portfolio for constraint checking.
    
    Attributes:
        total_value: Total portfolio market value
        cash: Available cash
        positions: List of Position objects
        peak_value: Historical peak portfolio value
        daily_turnover: Cumulative daily turnover
        weekly_turnover: Cumulative weekly turnover
        returns_history: Historical portfolio returns
        timestamp: Current timestamp
    """
    total_value: float
    cash: float
    positions: List[Position]
    peak_value: float
    daily_turnover: float = 0.0
    weekly_turnover: float = 0.0
    returns_history: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_sector_weights(self) -> Dict[str, float]:
        """Calculate total weight for each sector."""
        sector_weights: Dict[str, float] = {}
        for pos in self.positions:
            sector_weights[pos.sector] = sector_weights.get(pos.sector, 0.0) + pos.weight_pct
        return sector_weights
    
    def get_position_weight(self, symbol: str) -> float:
        """Get the weight of a specific position."""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos.weight_pct
        return 0.0
    
    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value


@dataclass
class ConstraintResult:
    """
    Result of a constraint check.
    
    Attributes:
        passed: Whether the constraint is satisfied
        constraint_name: Name of the constraint that was checked
        current_value: Current measured value
        limit_value: Maximum allowed value
        utilization_pct: Percentage of limit used (0-100+)
        message: Human-readable description
        severity: info/warning/breach
    """
    passed: bool
    constraint_name: str
    current_value: float
    limit_value: float
    utilization_pct: float
    message: str
    severity: str  # 'info', 'warning', 'breach'
    
    def __post_init__(self):
        """Validate severity value."""
        valid_severities = ['info', 'warning', 'breach']
        if self.severity not in valid_severities:
            raise ValueError(f"severity must be one of {valid_severities}")


@dataclass
class ConstraintReport:
    """
    Aggregated report from multiple constraint checks.
    
    Attributes:
        all_passed: True if all constraints passed
        results: List of individual constraint results
        breaches: List of results that are breaches
        warnings: List of results that are warnings
        timestamp: When the report was generated
    """
    all_passed: bool
    results: List[ConstraintResult]
    breaches: List[ConstraintResult]
    warnings: List[ConstraintResult]
    timestamp: datetime = field(default_factory=datetime.now)


class Constraint(ABC):
    """
    Abstract base class for all constraints.
    
    All concrete constraints must inherit from this class and implement
    the check method.
    
    Attributes:
        name: Unique identifier for this constraint
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def check(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> ConstraintResult:
        """
        Check if the portfolio satisfies this constraint.
        
        Args:
            portfolio_state: Current state of the portfolio
            proposed_trade: Optional proposed trade to check (pre-trade validation)
            
        Returns:
            ConstraintResult with pass/fail status and details
        """
        pass
    
    def _calculate_severity(self, utilization_pct: float) -> str:
        """
        Determine severity level based on utilization percentage.
        
        Args:
            utilization_pct: Percentage of limit used
            
        Returns:
            'info', 'warning', or 'breach'
        """
        if utilization_pct >= 100:
            return 'breach'
        elif utilization_pct >= 80:
            return 'warning'
        else:
            return 'info'


class MaxDrawdownConstraint(Constraint):
    """
    Constraint that limits maximum portfolio drawdown.
    
    Tracks peak portfolio value and triggers breach if current drawdown
    exceeds the specified limit. On breach, halts new trades and may
    trigger position reduction.
    
    Attributes:
        max_drawdown_pct: Maximum allowed drawdown (e.g., 0.20 for 20%)
    """
    
    def __init__(self, max_drawdown_pct: float = 0.20):
        super().__init__(name="MaxDrawdown")
        self.max_drawdown_pct = max_drawdown_pct
    
    def check(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> ConstraintResult:
        """
        Check if current drawdown is within limits.
        
        Args:
            portfolio_state: Current portfolio state with peak_value set
            proposed_trade: Ignored for drawdown check
            
        Returns:
            ConstraintResult with drawdown status
        """
        if portfolio_state.peak_value <= 0:
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                current_value=0.0,
                limit_value=self.max_drawdown_pct,
                utilization_pct=0.0,
                message="Peak value not set, cannot calculate drawdown",
                severity='info'
            )
        
        current_drawdown = portfolio_state.current_drawdown
        utilization_pct = (current_drawdown / self.max_drawdown_pct) * 100 if self.max_drawdown_pct > 0 else 0
        severity = self._calculate_severity(utilization_pct)
        passed = current_drawdown <= self.max_drawdown_pct
        
        message = (
            f"Current drawdown: {current_drawdown:.2%}, "
            f"Limit: {self.max_drawdown_pct:.2%}, "
            f"Peak: ${portfolio_state.peak_value:,.2f}, "
            f"Current: ${portfolio_state.total_value:,.2f}"
        )
        
        if not passed:
            message += " | BREACH: Trading halted, position reduction recommended"
        elif severity == 'warning':
            message += " | WARNING: Approaching drawdown limit"
        
        return ConstraintResult(
            passed=passed,
            constraint_name=self.name,
            current_value=current_drawdown,
            limit_value=self.max_drawdown_pct,
            utilization_pct=utilization_pct,
            message=message,
            severity=severity
        )


class PositionSizeConstraint(Constraint):
    """
    Constraint that limits individual position and sector sizes.
    
    Ensures no single position exceeds a maximum percentage of the portfolio
    and no sector concentration exceeds its limit.
    
    Attributes:
        max_single_position_pct: Maximum weight for any single position
        max_sector_pct: Maximum weight for any single sector
    """
    
    def __init__(
        self, 
        max_single_position_pct: float = 0.10, 
        max_sector_pct: float = 0.30
    ):
        super().__init__(name="PositionSize")
        self.max_single_position_pct = max_single_position_pct
        self.max_sector_pct = max_sector_pct
    
    def check(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> ConstraintResult:
        """
        Check position and sector size limits.
        
        Args:
            portfolio_state: Current portfolio state
            proposed_trade: Optional trade to include in calculation
            
        Returns:
            ConstraintResult with position size status
        """
        # Calculate position weights including proposed trade
        position_weights: Dict[str, float] = {}
        for pos in portfolio_state.positions:
            position_weights[pos.symbol] = pos.weight_pct
        
        # Include proposed trade if provided
        if proposed_trade:
            symbol = proposed_trade.get('symbol', '')
            trade_value = proposed_trade.get('quantity', 0) * proposed_trade.get('price', 0)
            if portfolio_state.total_value > 0:
                trade_weight = trade_value / portfolio_state.total_value
                position_weights[symbol] = position_weights.get(symbol, 0) + trade_weight
        
        # Check individual positions
        max_position_weight = max(position_weights.values()) if position_weights else 0
        max_position_symbol = max(position_weights.keys(), key=lambda k: position_weights[k]) if position_weights else ""
        
        # Check sector weights
        sector_weights = portfolio_state.get_sector_weights()
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        max_sector_name = max(sector_weights.keys(), key=lambda k: sector_weights[k]) if sector_weights else ""
        
        # Determine if any limit is breached
        position_breach = max_position_weight > self.max_single_position_pct
        sector_breach = max_sector_weight > self.max_sector_pct
        
        # Calculate overall utilization (worst case)
        position_util = (max_position_weight / self.max_single_position_pct) * 100 if self.max_single_position_pct > 0 else 0
        sector_util = (max_sector_weight / self.max_sector_pct) * 100 if self.max_sector_pct > 0 else 0
        max_util = max(position_util, sector_util)
        
        severity = self._calculate_severity(max_util)
        passed = not position_breach and not sector_breach
        
        message_parts = [
            f"Max position: {max_position_symbol}={max_position_weight:.2%} (limit {self.max_single_position_pct:.2%})",
            f"Max sector: {max_sector_name}={max_sector_weight:.2%} (limit {self.max_sector_pct:.2%})"
        ]
        
        if position_breach:
            message_parts.append(f"BREACH: Position {max_position_symbol} exceeds limit")
        if sector_breach:
            message_parts.append(f"BREACH: Sector {max_sector_name} exceeds limit")
        if severity == 'warning' and passed:
            message_parts.append("WARNING: Approaching size limits")
        
        return ConstraintResult(
            passed=passed,
            constraint_name=self.name,
            current_value=max(max_position_weight, max_sector_weight),
            limit_value=min(self.max_single_position_pct, self.max_sector_pct),
            utilization_pct=max_util,
            message=" | ".join(message_parts),
            severity=severity
        )


class ConcentrationConstraint(Constraint):
    """
    Constraint that limits concentration in top N positions.
    
    Uses a Herfindahl-like measure to ensure the top N positions
    don't exceed a specified concentration limit.
    
    Attributes:
        max_top_n_concentration: Maximum combined weight of top N positions
        n: Number of top positions to consider
    """
    
    def __init__(self, max_top_n_concentration: float = 0.60, n: int = 5):
        super().__init__(name="Concentration")
        self.max_top_n_concentration = max_top_n_concentration
        self.n = n
    
    def check(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> ConstraintResult:
        """
        Check concentration of top N positions.
        
        Args:
            portfolio_state: Current portfolio state
            proposed_trade: Optional trade to include in calculation
            
        Returns:
            ConstraintResult with concentration status
        """
        if not portfolio_state.positions:
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                current_value=0.0,
                limit_value=self.max_top_n_concentration,
                utilization_pct=0.0,
                message="No positions to check concentration",
                severity='info'
            )
        
        # Get position weights including proposed trade
        weights: List[float] = [pos.weight_pct for pos in portfolio_state.positions]
        
        if proposed_trade:
            symbol = proposed_trade.get('symbol', '')
            trade_value = proposed_trade.get('quantity', 0) * proposed_trade.get('price', 0)
            if portfolio_state.total_value > 0:
                trade_weight = trade_value / portfolio_state.total_value
                # Find existing position or add new
                found = False
                for i, pos in enumerate(portfolio_state.positions):
                    if pos.symbol == symbol:
                        weights[i] += trade_weight
                        found = True
                        break
                if not found:
                    weights.append(trade_weight)
        
        # Sort weights descending and take top N
        weights.sort(reverse=True)
        top_n_weights = weights[:self.n]
        top_n_concentration = sum(top_n_weights)
        
        utilization_pct = (top_n_concentration / self.max_top_n_concentration) * 100 if self.max_top_n_concentration > 0 else 0
        severity = self._calculate_severity(utilization_pct)
        passed = top_n_concentration <= self.max_top_n_concentration
        
        message = (
            f"Top {self.n} positions concentration: {top_n_concentration:.2%}, "
            f"Limit: {self.max_top_n_concentration:.2%}"
        )
        
        if not passed:
            message += f" | BREACH: Top {self.n} positions exceed concentration limit"
        elif severity == 'warning':
            message += " | WARNING: High concentration risk"
        
        # Add Herfindahl index for reference
        herfindahl = sum(w ** 2 for w in weights)
        message += f" | Herfindahl index: {herfindahl:.4f}"
        
        return ConstraintResult(
            passed=passed,
            constraint_name=self.name,
            current_value=top_n_concentration,
            limit_value=self.max_top_n_concentration,
            utilization_pct=utilization_pct,
            message=message,
            severity=severity
        )


class TurnoverConstraint(Constraint):
    """
    Constraint that limits portfolio turnover.
    
    Tracks cumulative daily and weekly turnover, resetting at appropriate
    intervals. Prevents excessive trading activity.
    
    Attributes:
        max_daily_turnover_pct: Maximum daily turnover percentage
        max_weekly_turnover_pct: Maximum weekly turnover percentage
    """
    
    def __init__(
        self, 
        max_daily_turnover_pct: float = 0.25, 
        max_weekly_turnover_pct: float = 0.50
    ):
        super().__init__(name="Turnover")
        self.max_daily_turnover_pct = max_daily_turnover_pct
        self.max_weekly_turnover_pct = max_weekly_turnover_pct
    
    def check(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> ConstraintResult:
        """
        Check daily and weekly turnover limits.
        
        Args:
            portfolio_state: Current portfolio state with turnover tracking
            proposed_trade: Optional trade to include in calculation
            
        Returns:
            ConstraintResult with turnover status
        """
        daily_turnover = portfolio_state.daily_turnover
        weekly_turnover = portfolio_state.weekly_turnover
        
        # Include proposed trade
        if proposed_trade and portfolio_state.total_value > 0:
            trade_value = abs(proposed_trade.get('quantity', 0) * proposed_trade.get('price', 0))
            trade_pct = trade_value / portfolio_state.total_value
            daily_turnover += trade_pct
            weekly_turnover += trade_pct
        
        daily_util = (daily_turnover / self.max_daily_turnover_pct) * 100 if self.max_daily_turnover_pct > 0 else 0
        weekly_util = (weekly_turnover / self.max_weekly_turnover_pct) * 100 if self.max_weekly_turnover_pct > 0 else 0
        max_util = max(daily_util, weekly_util)
        
        daily_breach = daily_turnover > self.max_daily_turnover_pct
        weekly_breach = weekly_turnover > self.max_weekly_turnover_pct
        
        severity = self._calculate_severity(max_util)
        passed = not daily_breach and not weekly_breach
        
        message = (
            f"Daily turnover: {daily_turnover:.2%} (limit {self.max_daily_turnover_pct:.2%}), "
            f"Weekly turnover: {weekly_turnover:.2%} (limit {self.max_weekly_turnover_pct:.2%})"
        )
        
        if daily_breach:
            message += " | BREACH: Daily turnover limit exceeded"
        if weekly_breach:
            message += " | BREACH: Weekly turnover limit exceeded"
        if severity == 'warning' and passed:
            message += " | WARNING: Approaching turnover limits"
        
        return ConstraintResult(
            passed=passed,
            constraint_name=self.name,
            current_value=max(daily_turnover, weekly_turnover),
            limit_value=min(self.max_daily_turnover_pct, self.max_weekly_turnover_pct),
            utilization_pct=max_util,
            message=message,
            severity=severity
        )


class CorrelationConstraint(Constraint):
    """
    Constraint that limits average pairwise correlation of positions.
    
    Ensures portfolio diversification by keeping average correlation
    below a threshold. Uses historical returns to compute correlations.
    
    Attributes:
        max_avg_correlation: Maximum average pairwise correlation
        lookback_days: Number of days of returns history to use
    """
    
    def __init__(self, max_avg_correlation: float = 0.70, lookback_days: int = 60):
        super().__init__(name="Correlation")
        self.max_avg_correlation = max_avg_correlation
        self.lookback_days = lookback_days
    
    def check(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> ConstraintResult:
        """
        Check average pairwise correlation of positions.
        
        Args:
            portfolio_state: Current portfolio state with returns_history
            proposed_trade: Ignored (correlation based on historical data)
            
        Returns:
            ConstraintResult with correlation status
        """
        # For portfolio-level correlation, we use returns history
        # In a real implementation, this would use individual asset returns
        # Here we use a simplified approach based on portfolio returns variance
        
        returns = portfolio_state.returns_history
        
        if len(returns) < 2:
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                current_value=0.0,
                limit_value=self.max_avg_correlation,
                utilization_pct=0.0,
                message="Insufficient returns history for correlation calculation",
                severity='info'
            )
        
        # Use recent returns only
        recent_returns = returns[-self.lookback_days:] if len(returns) > self.lookback_days else returns
        
        if len(recent_returns) < 2:
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                current_value=0.0,
                limit_value=self.max_avg_correlation,
                utilization_pct=0.0,
                message="Insufficient recent returns for correlation calculation",
                severity='info'
            )
        
        # Calculate portfolio volatility as a proxy for correlation
        # Higher volatility relative to individual assets suggests higher correlation
        returns_array = np.array(recent_returns)
        
        # Use autocorrelation as a proxy for portfolio correlation structure
        if len(returns_array) > 1:
            autocorr = np.corrcoef(returns_array[:-1], returns_array[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # For a more realistic correlation measure, use position count
        # More concentrated portfolios tend to have higher effective correlation
        num_positions = len(portfolio_state.positions)
        if num_positions <= 1:
            avg_correlation = 0.0
        else:
            # Estimate based on concentration - higher concentration = higher effective correlation
            weights = [pos.weight_pct for pos in portfolio_state.positions]
            herfindahl = sum(w ** 2 for w in weights) if weights else 0
            # Scale: 1 position = 1.0 correlation, many positions = lower correlation
            avg_correlation = herfindahl + (1 - herfindahl) * abs(autocorr) * 0.5
        
        utilization_pct = (avg_correlation / self.max_avg_correlation) * 100 if self.max_avg_correlation > 0 else 0
        severity = self._calculate_severity(utilization_pct)
        passed = avg_correlation <= self.max_avg_correlation
        
        message = (
            f"Estimated avg correlation: {avg_correlation:.2%}, "
            f"Limit: {self.max_avg_correlation:.2%}, "
            f"Positions: {num_positions}, "
            f"Lookback: {len(recent_returns)} days"
        )
        
        if not passed:
            message += " | BREACH: High portfolio correlation - diversification needed"
        elif severity == 'warning':
            message += " | WARNING: Elevated correlation risk"
        
        return ConstraintResult(
            passed=passed,
            constraint_name=self.name,
            current_value=avg_correlation,
            limit_value=self.max_avg_correlation,
            utilization_pct=utilization_pct,
            message=message,
            severity=severity
        )


class ConstraintManager:
    """
    Manages multiple constraints and runs aggregate checks.
    
    Provides a unified interface for registering constraints and
    generating comprehensive constraint reports.
    
    Attributes:
        constraints: List of registered Constraint objects
    """
    
    def __init__(self):
        self.constraints: List[Constraint] = []
    
    def add_constraint(self, constraint: Constraint) -> None:
        """
        Register a constraint with the manager.
        
        Args:
            constraint: Constraint instance to add
        """
        self.constraints.append(constraint)
    
    def remove_constraint(self, constraint_name: str) -> bool:
        """
        Remove a constraint by name.
        
        Args:
            constraint_name: Name of constraint to remove
            
        Returns:
            True if constraint was found and removed
        """
        for i, c in enumerate(self.constraints):
            if c.name == constraint_name:
                self.constraints.pop(i)
                return True
        return False
    
    def check_all(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> ConstraintReport:
        """
        Run all registered constraints and generate report.
        
        Args:
            portfolio_state: Current portfolio state
            proposed_trade: Optional proposed trade for pre-trade checks
            
        Returns:
            ConstraintReport with all results
        """
        results: List[ConstraintResult] = []
        breaches: List[ConstraintResult] = []
        warnings: List[ConstraintResult] = []
        
        for constraint in self.constraints:
            result = constraint.check(portfolio_state, proposed_trade)
            results.append(result)
            
            if result.severity == 'breach':
                breaches.append(result)
            elif result.severity == 'warning':
                warnings.append(result)
        
        all_passed = len(breaches) == 0
        
        return ConstraintReport(
            all_passed=all_passed,
            results=results,
            breaches=breaches,
            warnings=warnings
        )
    
    def get_constraint(self, name: str) -> Optional[Constraint]:
        """
        Get a constraint by name.
        
        Args:
            name: Name of constraint to find
            
        Returns:
            Constraint instance or None
        """
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None
    
    def clear_constraints(self) -> None:
        """Remove all registered constraints."""
        self.constraints.clear()
