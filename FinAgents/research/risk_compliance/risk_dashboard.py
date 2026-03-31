"""
Risk Dashboard for Real-Time Risk Monitoring

This module provides the RiskDashboard class for computing and displaying
real-time risk metrics, limit utilization, breach history, and risk budgets.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import numpy as np

from .constraints import (
    ConstraintManager,
    PortfolioState,
    Position,
)
from .compliance_engine import ComplianceEngine
from .regulatory_checks import RegulatoryChecker


@dataclass
class RiskMetrics:
    """
    Core risk metrics for portfolio monitoring.
    
    Attributes:
        var_95: 95% Value at Risk
        cvar_95: 95% Conditional Value at Risk (Expected Shortfall)
        current_drawdown: Current drawdown from peak
        max_drawdown: Maximum historical drawdown
        volatility: Annualized portfolio volatility
        beta: Portfolio beta to market
        sharpe: Sharpe ratio
        concentration: Herfindahl concentration index
        avg_correlation: Average pairwise correlation
    """
    var_95: float
    cvar_95: float
    current_drawdown: float
    max_drawdown: float
    volatility: float
    beta: float
    sharpe: float
    concentration: float
    avg_correlation: float


@dataclass
class LimitUtilization:
    """
    Utilization of a specific risk limit.
    
    Attributes:
        constraint_name: Name of the constraint
        current_value: Current measured value
        limit_value: Maximum allowed value
        utilization_pct: Percentage of limit used
        status: Status indicator (green/yellow/red)
        headroom: Remaining capacity before limit
    """
    constraint_name: str
    current_value: float
    limit_value: float
    utilization_pct: float
    status: str  # 'green', 'yellow', 'red'
    headroom: float


@dataclass
class BreachRecord:
    """
    Record of a constraint breach.
    
    Attributes:
        timestamp: When the breach occurred
        constraint_name: Name of breached constraint
        severity: Severity level
        details: Detailed description
        resolution: How the breach was resolved
    """
    timestamp: datetime
    constraint_name: str
    severity: str
    details: str
    resolution: str


@dataclass
class RiskBudget:
    """
    Risk budget allocation and usage.
    
    Attributes:
        total_budget: Total risk budget available
        used: Amount of risk currently used
        remaining: Remaining risk capacity
        per_position_budget: Risk allocation per position
        allocation_suggestion: Suggested allocation strategy
    """
    total_budget: float
    used: float
    remaining: float
    per_position_budget: Dict[str, float]
    allocation_suggestion: str


class RiskDashboard:
    """
    Dashboard for real-time risk monitoring and reporting.
    
    Provides comprehensive risk metrics, limit utilization tracking,
    breach history, and risk budget management.
    
    Attributes:
        constraint_manager: Manager for constraint checking
        compliance_engine: Engine for compliance validation
        regulatory_checker: Checker for regulatory compliance
        breach_log: In-memory log of constraint breaches
    """
    
    def __init__(
        self,
        constraint_manager: ConstraintManager,
        compliance_engine: ComplianceEngine,
        regulatory_checker: RegulatoryChecker
    ):
        """
        Initialize the risk dashboard.
        
        Args:
            constraint_manager: ConstraintManager instance
            compliance_engine: ComplianceEngine instance
            regulatory_checker: RegulatoryChecker instance
        """
        self.constraint_manager = constraint_manager
        self.compliance_engine = compliance_engine
        self.regulatory_checker = regulatory_checker
        self.breach_log: List[BreachRecord] = []
    
    def get_real_time_risk_metrics(self, portfolio_state: PortfolioState) -> RiskMetrics:
        """
        Compute real-time risk metrics for the portfolio.
        
        Calculates VaR, CVaR, drawdown, volatility, beta, Sharpe ratio,
        concentration, and correlation metrics using numpy.
        
        Args:
            portfolio_state: Current portfolio state with returns history
            
        Returns:
            RiskMetrics with all computed values
        """
        returns = np.array(portfolio_state.returns_history) if portfolio_state.returns_history else np.array([])
        
        # Handle edge cases
        if len(returns) < 2:
            return RiskMetrics(
                var_95=0.0,
                cvar_95=0.0,
                current_drawdown=portfolio_state.current_drawdown,
                max_drawdown=portfolio_state.current_drawdown,
                volatility=0.0,
                beta=1.0,
                sharpe=0.0,
                concentration=self._calculate_concentration(portfolio_state),
                avg_correlation=0.0
            )
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
        
        # Conditional VaR / Expected Shortfall (95%)
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        # Current drawdown
        current_drawdown = portfolio_state.current_drawdown
        
        # Max drawdown calculation from returns
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Volatility (annualized, assuming daily returns)
        volatility = np.std(returns, ddof=1) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Beta (simplified - assumes market returns are embedded in portfolio returns)
        beta = self._calculate_beta(returns)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        mean_return = np.mean(returns)
        sharpe = (mean_return * 252) / volatility if volatility > 0 else 0.0
        
        # Concentration (Herfindahl index)
        concentration = self._calculate_concentration(portfolio_state)
        
        # Average correlation (proxy)
        avg_correlation = self._calculate_avg_correlation(portfolio_state, returns)
        
        return RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            sharpe=sharpe,
            concentration=concentration,
            avg_correlation=avg_correlation
        )
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from returns series.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown as a positive percentage
        """
        if len(returns) < 2:
            return 0.0
        
        # Convert returns to cumulative values
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Return maximum drawdown (as positive value)
        max_dd = np.min(drawdown)
        return abs(max_dd) if max_dd < 0 else 0.0
    
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """
        Calculate portfolio beta (simplified).
        
        In practice, this would use actual market returns.
        Here we use a proxy based on return characteristics.
        
        Args:
            returns: Array of portfolio returns
            
        Returns:
            Estimated beta
        """
        if len(returns) < 2:
            return 1.0
        
        # Simplified beta estimate based on volatility
        # Higher volatility generally indicates higher beta
        volatility = np.std(returns, ddof=1)
        
        # Assume market volatility of ~16% annualized (daily ~1%)
        market_vol_daily = 0.01
        
        # Rough beta estimate
        beta_estimate = volatility / market_vol_daily if market_vol_daily > 0 else 1.0
        
        # Clamp to reasonable range
        return max(0.5, min(2.0, beta_estimate))
    
    def _calculate_concentration(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate Herfindahl concentration index.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Herfindahl index (0 = perfectly diversified, 1 = single asset)
        """
        weights = [pos.weight_pct for pos in portfolio_state.positions]
        
        if not weights:
            return 0.0
        
        return sum(w ** 2 for w in weights)
    
    def _calculate_avg_correlation(
        self, 
        portfolio_state: PortfolioState, 
        returns: np.ndarray
    ) -> float:
        """
        Calculate average correlation proxy.
        
        Args:
            portfolio_state: Current portfolio state
            returns: Portfolio returns history
            
        Returns:
            Estimated average correlation
        """
        num_positions = len(portfolio_state.positions)
        
        if num_positions <= 1:
            return 0.0
        
        # Use concentration as proxy for correlation
        # More concentrated portfolios tend to have higher effective correlation
        concentration = self._calculate_concentration(portfolio_state)
        
        # Estimate correlation based on concentration and return autocorrelation
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # Blend concentration and autocorrelation
        estimated_corr = concentration * 0.7 + abs(autocorr) * 0.3
        
        return min(1.0, max(0.0, estimated_corr))
    
    def get_limit_utilization(self, portfolio_state: PortfolioState) -> List[LimitUtilization]:
        """
        Get utilization status for all risk limits.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            List of LimitUtilization for each constraint
        """
        report = self.constraint_manager.check_all(portfolio_state)
        utilizations: List[LimitUtilization] = []
        
        for result in report.results:
            # Determine status
            if result.utilization_pct >= 100:
                status = 'red'
            elif result.utilization_pct >= 80:
                status = 'yellow'
            else:
                status = 'green'
            
            # Calculate headroom
            headroom = result.limit_value - result.current_value
            if result.limit_value > 0:
                headroom_pct = (1 - result.utilization_pct / 100) * result.limit_value
            else:
                headroom_pct = 0.0
            
            utilizations.append(LimitUtilization(
                constraint_name=result.constraint_name,
                current_value=result.current_value,
                limit_value=result.limit_value,
                utilization_pct=result.utilization_pct,
                status=status,
                headroom=headroom_pct
            ))
        
        return utilizations
    
    def get_breach_history(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> List[BreachRecord]:
        """
        Get breach history for a date range.
        
        Args:
            start_date: Start date filter (inclusive)
            end_date: End date filter (inclusive)
            
        Returns:
            List of BreachRecord within the date range
        """
        filtered = self.breach_log
        
        if start_date:
            filtered = [
                b for b in filtered 
                if b.timestamp.date() >= start_date
            ]
        
        if end_date:
            filtered = [
                b for b in filtered 
                if b.timestamp.date() <= end_date
            ]
        
        return filtered
    
    def log_breach(
        self, 
        constraint_name: str, 
        severity: str, 
        details: str,
        resolution: str = "Pending"
    ) -> None:
        """
        Log a new breach to the dashboard.
        
        Args:
            constraint_name: Name of the breached constraint
            severity: Severity level
            details: Detailed description
            resolution: Resolution status
        """
        record = BreachRecord(
            timestamp=datetime.now(),
            constraint_name=constraint_name,
            severity=severity,
            details=details,
            resolution=resolution
        )
        self.breach_log.append(record)
    
    def get_risk_budget(
        self, 
        portfolio_state: PortfolioState, 
        total_risk_budget: float = 0.20
    ) -> RiskBudget:
        """
        Calculate risk budget allocation and usage.
        
        Args:
            portfolio_state: Current portfolio state
            total_risk_budget: Total risk budget (e.g., 20% max drawdown)
            
        Returns:
            RiskBudget with allocation details
        """
        # Calculate current risk usage
        metrics = self.get_real_time_risk_metrics(portfolio_state)
        
        # Risk usage is primarily driven by drawdown and volatility
        drawdown_risk = metrics.current_drawdown / total_risk_budget if total_risk_budget > 0 else 0
        vol_risk = metrics.volatility * 0.5  # Scale volatility contribution
        
        used = min(1.0, drawdown_risk + vol_risk * 0.3)
        remaining = max(0.0, total_risk_budget - used * total_risk_budget)
        
        # Calculate per-position budget
        num_positions = len(portfolio_state.positions)
        if num_positions > 0:
            base_budget = total_risk_budget / num_positions
            per_position: Dict[str, float] = {}
            
            for pos in portfolio_state.positions:
                # Scale by position weight
                weight_factor = pos.weight_pct / (1.0 / num_positions) if num_positions > 0 else 1.0
                per_position[pos.symbol] = base_budget * weight_factor
        else:
            per_position = {}
            base_budget = total_risk_budget
        
        # Generate suggestion
        if used > 0.8:
            suggestion = "CRITICAL: Risk budget nearly exhausted. Reduce position sizes immediately."
        elif used > 0.6:
            suggestion = "WARNING: Risk budget significantly utilized. Consider reducing exposure."
        elif num_positions < 5:
            suggestion = "Consider adding more positions to diversify risk allocation."
        else:
            suggestion = "Risk budget healthy. Maintain current allocation strategy."
        
        return RiskBudget(
            total_budget=total_risk_budget,
            used=used * total_risk_budget,
            remaining=remaining,
            per_position_budget=per_position,
            allocation_suggestion=suggestion
        )
    
    def get_dashboard_summary(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """
        Get complete dashboard summary data.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with complete dashboard data
        """
        # Get all components
        metrics = self.get_real_time_risk_metrics(portfolio_state)
        utilizations = self.get_limit_utilization(portfolio_state)
        risk_budget = self.get_risk_budget(portfolio_state)
        
        # Get recent breaches (last 7 days)
        recent_breaches = self.get_breach_history(
            start_date=date.today() - timedelta(days=7)
        )
        
        # Calculate overall risk level
        overall_risk_level = self._calculate_overall_risk_level(
            metrics, utilizations, recent_breaches
        )
        
        # Format recent breaches
        breach_summary = [
            {
                'timestamp': b.timestamp.isoformat(),
                'constraint': b.constraint_name,
                'severity': b.severity,
                'details': b.details[:100] + '...' if len(b.details) > 100 else b.details
            }
            for b in recent_breaches[-5:]  # Last 5 breaches
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_risk_level': overall_risk_level,
            'metrics': {
                'var_95': round(metrics.var_95, 4),
                'cvar_95': round(metrics.cvar_95, 4),
                'current_drawdown': round(metrics.current_drawdown, 4),
                'max_drawdown': round(metrics.max_drawdown, 4),
                'volatility': round(metrics.volatility, 4),
                'beta': round(metrics.beta, 2),
                'sharpe': round(metrics.sharpe, 2),
                'concentration': round(metrics.concentration, 4),
                'avg_correlation': round(metrics.avg_correlation, 4),
            },
            'limit_utilizations': [
                {
                    'constraint_name': u.constraint_name,
                    'current_value': round(u.current_value, 4),
                    'limit_value': round(u.limit_value, 4),
                    'utilization_pct': round(u.utilization_pct, 1),
                    'status': u.status,
                    'headroom': round(u.headroom, 4)
                }
                for u in utilizations
            ],
            'recent_breaches': breach_summary,
            'breach_count_7d': len(recent_breaches),
            'risk_budget': {
                'total_budget': round(risk_budget.total_budget, 4),
                'used': round(risk_budget.used, 4),
                'remaining': round(risk_budget.remaining, 4),
                'utilization_pct': round(
                    (risk_budget.used / risk_budget.total_budget * 100) 
                    if risk_budget.total_budget > 0 else 0, 1
                ),
                'allocation_suggestion': risk_budget.allocation_suggestion
            },
            'portfolio_summary': {
                'total_value': round(portfolio_state.total_value, 2),
                'cash': round(portfolio_state.cash, 2),
                'cash_pct': round(
                    portfolio_state.cash / portfolio_state.total_value * 100 
                    if portfolio_state.total_value > 0 else 0, 2
                ),
                'num_positions': len(portfolio_state.positions),
                'daily_turnover': round(portfolio_state.daily_turnover, 4),
                'weekly_turnover': round(portfolio_state.weekly_turnover, 4),
            }
        }
    
    def _calculate_overall_risk_level(
        self,
        metrics: RiskMetrics,
        utilizations: List[LimitUtilization],
        recent_breaches: List[BreachRecord]
    ) -> str:
        """
        Calculate overall portfolio risk level.
        
        Args:
            metrics: Risk metrics
            utilizations: Limit utilizations
            recent_breaches: Recent breach records
            
        Returns:
            Risk level: LOW, MEDIUM, HIGH, or CRITICAL
        """
        score = 0
        
        # Check utilizations
        red_count = sum(1 for u in utilizations if u.status == 'red')
        yellow_count = sum(1 for u in utilizations if u.status == 'yellow')
        
        score += red_count * 3
        score += yellow_count * 1
        
        # Check recent breaches
        critical_breaches = sum(1 for b in recent_breaches if b.severity == 'CRITICAL')
        high_breaches = sum(1 for b in recent_breaches if b.severity == 'HIGH')
        
        score += critical_breaches * 3
        score += high_breaches * 2
        
        # Check metrics
        if metrics.current_drawdown > 0.15:
            score += 3
        elif metrics.current_drawdown > 0.10:
            score += 2
        elif metrics.current_drawdown > 0.05:
            score += 1
        
        if metrics.volatility > 0.30:
            score += 2
        elif metrics.volatility > 0.20:
            score += 1
        
        if metrics.concentration > 0.20:
            score += 1
        
        # Determine level
        if score >= 6:
            return "CRITICAL"
        elif score >= 4:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
