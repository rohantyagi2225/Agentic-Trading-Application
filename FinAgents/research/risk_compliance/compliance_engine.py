"""
Compliance Engine for Pre-trade and Post-trade Validation

This module provides the ComplianceEngine class that orchestrates constraint
checking for both pre-trade approval and post-trade monitoring, with support
for trade modification and breach handling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import copy

from .constraints import (
    ConstraintManager,
    ConstraintResult,
    PortfolioState,
    Position,
)


@dataclass
class ComplianceDecision:
    """
    Result of a pre-trade compliance check.
    
    Attributes:
        approved: Whether the trade is approved
        original_trade: The trade as originally proposed
        modified_trade: Modified version that passes constraints, or None
        constraint_results: List of all constraint check results
        rejection_reasons: Reasons for rejection if not approved
        warnings: List of warning messages
    """
    approved: bool
    original_trade: Dict[str, Any]
    modified_trade: Optional[Dict[str, Any]]
    constraint_results: List[ConstraintResult]
    rejection_reasons: List[str]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceReport:
    """
    Result of a post-trade compliance check.
    
    Attributes:
        compliant: Whether portfolio is in compliance
        breaches: List of constraint breaches
        required_actions: Recommended corrective actions
        risk_summary: Summary of current risk metrics
        timestamp: When the report was generated
    """
    compliant: bool
    breaches: List[ConstraintResult]
    required_actions: List[str]
    risk_summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BreachResponse:
    """
    Response to a constraint breach.
    
    Attributes:
        severity: Severity level of the breach
        actions_required: List of required actions
        auto_actions: Actions that can be automated
        notification_required: Whether notification is needed
        details: Detailed description of the breach and response
    """
    severity: str
    actions_required: List[str]
    auto_actions: List[str]
    notification_required: bool
    details: str


class ComplianceEngine:
    """
    Engine for pre-trade and post-trade compliance checking.
    
    The ComplianceEngine orchestrates constraint validation, trade modification
    attempts, and breach response generation. It works with a ConstraintManager
    to evaluate all registered constraints.
    
    Attributes:
        constraint_manager: Manager containing all constraints to check
        config: Configuration dictionary for engine behavior
        breach_history: Historical record of breaches
    """
    
    def __init__(self, constraint_manager: ConstraintManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the compliance engine.
        
        Args:
            constraint_manager: ConstraintManager with registered constraints
            config: Optional configuration dict with keys:
                - auto_modify: Whether to attempt trade modification (default True)
                - max_modification_attempts: Max attempts to modify trade (default 5)
                - modification_step: Percentage to reduce trade by each attempt (default 0.1)
                - auto_reduce_on_breach: Whether to auto-generate reduction trades (default False)
        """
        self.constraint_manager = constraint_manager
        self.config = config or {}
        self.breach_history: List[Dict[str, Any]] = []
        
        # Set defaults
        self.auto_modify = self.config.get('auto_modify', True)
        self.max_modification_attempts = self.config.get('max_modification_attempts', 5)
        self.modification_step = self.config.get('modification_step', 0.1)
        self.auto_reduce_on_breach = self.config.get('auto_reduce_on_breach', False)
    
    def pre_trade_check(
        self, 
        portfolio_state: PortfolioState, 
        proposed_trade: Dict[str, Any]
    ) -> ComplianceDecision:
        """
        Check if a proposed trade passes all constraints.
        
        If the trade violates constraints, attempts to modify the trade
        (e.g., reduce size) to fit within limits. If modification fails,
        rejects the trade with clear reasons.
        
        Args:
            portfolio_state: Current portfolio state
            proposed_trade: Trade to validate with keys:
                - symbol: Stock symbol
                - action: 'BUY' or 'SELL'
                - quantity: Number of shares
                - price: Expected execution price
                
        Returns:
            ComplianceDecision with approval status and modifications
        """
        # First check with original trade
        report = self.constraint_manager.check_all(portfolio_state, proposed_trade)
        
        # If all passed, approve as-is
        if report.all_passed:
            return ComplianceDecision(
                approved=True,
                original_trade=proposed_trade,
                modified_trade=None,
                constraint_results=report.results,
                rejection_reasons=[],
                warnings=[r.message for r in report.warnings]
            )
        
        # Try to modify trade if enabled
        if self.auto_modify:
            modified_trade, modification_results = self._attempt_trade_modification(
                portfolio_state, proposed_trade, report.breaches
            )
            
            if modified_trade is not None:
                # Modified trade passes
                return ComplianceDecision(
                    approved=True,
                    original_trade=proposed_trade,
                    modified_trade=modified_trade,
                    constraint_results=modification_results,
                    rejection_reasons=[],
                    warnings=[f"Trade modified from {proposed_trade.get('quantity')} to {modified_trade.get('quantity')} shares"]
                )
        
        # Cannot approve - collect rejection reasons
        rejection_reasons = []
        for breach in report.breaches:
            rejection_reasons.append(f"{breach.constraint_name}: {breach.message}")
        
        return ComplianceDecision(
            approved=False,
            original_trade=proposed_trade,
            modified_trade=None,
            constraint_results=report.results,
            rejection_reasons=rejection_reasons,
            warnings=[r.message for r in report.warnings]
        )
    
    def _attempt_trade_modification(
        self,
        portfolio_state: PortfolioState,
        original_trade: Dict[str, Any],
        breaches: List[ConstraintResult]
    ) -> tuple[Optional[Dict[str, Any]], List[ConstraintResult]]:
        """
        Attempt to modify a trade to pass constraints.
        
        Reduces trade size iteratively until constraints pass or
        max attempts reached.
        
        Args:
            portfolio_state: Current portfolio state
            original_trade: Original proposed trade
            breaches: List of constraint breaches
            
        Returns:
            Tuple of (modified_trade or None, constraint_results)
        """
        modified_trade = copy.deepcopy(original_trade)
        original_quantity = abs(modified_trade.get('quantity', 0))
        
        if original_quantity == 0:
            return None, []
        
        for attempt in range(self.max_modification_attempts):
            # Reduce quantity by modification step
            reduction_factor = 1 - (self.modification_step * (attempt + 1))
            new_quantity = original_quantity * reduction_factor
            
            # Update trade quantity (preserve sign)
            if modified_trade.get('quantity', 0) < 0:
                modified_trade['quantity'] = -new_quantity
            else:
                modified_trade['quantity'] = new_quantity
            
            # Check if modified trade passes
            report = self.constraint_manager.check_all(portfolio_state, modified_trade)
            
            if report.all_passed:
                return modified_trade, report.results
            
            # If drawdown breach, size reduction won't help
            drawdown_breach = any(b.constraint_name == "MaxDrawdown" for b in report.breaches)
            if drawdown_breach:
                break
        
        return None, []
    
    def post_trade_check(self, portfolio_state: PortfolioState) -> ComplianceReport:
        """
        Check portfolio compliance after trade execution.
        
        Evaluates all constraints against current portfolio state and
        generates corrective actions for any breaches.
        
        Args:
            portfolio_state: Current portfolio state after trade
            
        Returns:
            ComplianceReport with compliance status and actions
        """
        report = self.constraint_manager.check_all(portfolio_state)
        
        required_actions: List[str] = []
        
        # Generate corrective actions for each breach
        for breach in report.breaches:
            actions = self._generate_corrective_actions(breach, portfolio_state)
            required_actions.extend(actions)
            
            # Log breach to history
            self.breach_history.append({
                'timestamp': datetime.now(),
                'constraint_name': breach.constraint_name,
                'current_value': breach.current_value,
                'limit_value': breach.limit_value,
                'severity': breach.severity,
                'message': breach.message
            })
        
        # Generate risk summary
        risk_summary = self._generate_risk_summary(portfolio_state, report)
        
        return ComplianceReport(
            compliant=report.all_passed,
            breaches=report.breaches,
            required_actions=required_actions,
            risk_summary=risk_summary
        )
    
    def _generate_corrective_actions(
        self, 
        breach: ConstraintResult, 
        portfolio_state: PortfolioState
    ) -> List[str]:
        """
        Generate corrective actions for a constraint breach.
        
        Args:
            breach: The constraint breach
            portfolio_state: Current portfolio state
            
        Returns:
            List of recommended actions
        """
        actions: List[str] = []
        
        if breach.constraint_name == "MaxDrawdown":
            actions.append("HALT: All new trading suspended")
            actions.append("ACTION: Reduce position sizes by 20-50%")
            actions.append("REVIEW: Reassess risk parameters and market conditions")
            
        elif breach.constraint_name == "PositionSize":
            # Find largest position
            if portfolio_state.positions:
                largest = max(portfolio_state.positions, key=lambda p: p.weight_pct)
                excess = largest.weight_pct - 0.10  # Assuming 10% limit
                if excess > 0:
                    actions.append(f"REDUCE: {largest.symbol} position by {excess:.1%}")
            actions.append("REBALANCE: Redistribute to smaller positions")
            
        elif breach.constraint_name == "Concentration":
            actions.append("DIVERSIFY: Add positions in uncorrelated sectors")
            actions.append("REDUCE: Trim top 5 positions to reduce concentration")
            
        elif breach.constraint_name == "Turnover":
            actions.append("PAUSE: Suspend new trades until period resets")
            actions.append("REVIEW: Evaluate trading strategy frequency")
            
        elif breach.constraint_name == "Correlation":
            actions.append("DIVERSIFY: Add uncorrelated assets to portfolio")
            actions.append("REVIEW: Current positions show high correlation")
        
        return actions
    
    def _generate_risk_summary(
        self, 
        portfolio_state: PortfolioState, 
        report: Any
    ) -> Dict[str, Any]:
        """
        Generate a summary of current risk metrics.
        
        Args:
            portfolio_state: Current portfolio state
            report: Constraint report
            
        Returns:
            Dictionary with risk summary
        """
        num_positions = len(portfolio_state.positions)
        
        # Calculate concentration (Herfindahl)
        weights = [p.weight_pct for p in portfolio_state.positions]
        herfindahl = sum(w ** 2 for w in weights) if weights else 0
        
        # Sector distribution
        sector_weights = portfolio_state.get_sector_weights()
        
        return {
            'total_value': portfolio_state.total_value,
            'cash_percentage': portfolio_state.cash / portfolio_state.total_value if portfolio_state.total_value > 0 else 0,
            'num_positions': num_positions,
            'concentration_index': herfindahl,
            'current_drawdown': portfolio_state.current_drawdown,
            'sector_allocation': sector_weights,
            'constraint_breaches': len(report.breaches),
            'constraint_warnings': len(report.warnings),
        }
    
    def breach_handler(
        self, 
        breach: ConstraintResult, 
        portfolio_state: PortfolioState
    ) -> BreachResponse:
        """
        Handle a constraint breach and generate response.
        
        For critical breaches, generates automatic position reduction trades.
        
        Args:
            breach: The constraint breach to handle
            portfolio_state: Current portfolio state
            
        Returns:
            BreachResponse with actions and notifications
        """
        actions_required: List[str] = []
        auto_actions: List[str] = []
        
        # Determine severity and response
        if breach.constraint_name == "MaxDrawdown":
            severity = "CRITICAL"
            actions_required.append("Immediately halt all new trading")
            actions_required.append("Reduce position sizes by minimum 25%")
            actions_required.append("Notify portfolio manager and risk committee")
            
            if self.auto_reduce_on_breach:
                auto_actions.append("Auto-generate position reduction orders")
                
        elif breach.constraint_name == "PositionSize":
            severity = "HIGH"
            actions_required.append("Reduce oversized positions within 1 trading day")
            actions_required.append("Review position sizing methodology")
            
            if portfolio_state.positions:
                largest = max(portfolio_state.positions, key=lambda p: p.weight_pct)
                if largest.weight_pct > 0.15:
                    auto_actions.append(f"Auto-reduce {largest.symbol} position")
                    
        elif breach.constraint_name == "Concentration":
            severity = "MEDIUM"
            actions_required.append("Diversify portfolio within 3 trading days")
            actions_required.append("Review sector allocation strategy")
            
        elif breach.constraint_name == "Turnover":
            severity = "MEDIUM"
            actions_required.append("Pause trading until next period")
            actions_required.append("Review trading frequency limits")
            
        elif breach.constraint_name == "Correlation":
            severity = "MEDIUM"
            actions_required.append("Add uncorrelated positions")
            actions_required.append("Review portfolio construction methodology")
            
        else:
            severity = "LOW"
            actions_required.append("Review constraint parameters")
        
        notification_required = severity in ["CRITICAL", "HIGH"]
        
        details = (
            f"Breach Type: {breach.constraint_name}\n"
            f"Severity: {severity}\n"
            f"Current Value: {breach.current_value:.4f}\n"
            f"Limit Value: {breach.limit_value:.4f}\n"
            f"Utilization: {breach.utilization_pct:.1f}%\n"
            f"Message: {breach.message}\n"
            f"Required Actions: {len(actions_required)}\n"
            f"Auto Actions Available: {len(auto_actions)}"
        )
        
        return BreachResponse(
            severity=severity,
            actions_required=actions_required,
            auto_actions=auto_actions,
            notification_required=notification_required,
            details=details
        )
    
    def generate_compliance_report(
        self, 
        portfolio_state: PortfolioState, 
        period: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Generate a full compliance report for a given period.
        
        Args:
            portfolio_state: Current portfolio state
            period: Reporting period ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary with comprehensive compliance status
        """
        # Run all constraint checks
        constraint_report = self.constraint_manager.check_all(portfolio_state)
        
        # Calculate utilizations
        utilizations = []
        for result in constraint_report.results:
            utilizations.append({
                'constraint_name': result.constraint_name,
                'current_value': result.current_value,
                'limit_value': result.limit_value,
                'utilization_pct': result.utilization_pct,
                'status': result.severity
            })
        
        # Get recent breach history
        recent_breaches = [
            b for b in self.breach_history
            if (datetime.now() - b['timestamp']).days <= 7
        ]
        
        # Calculate trend (compare to previous if available)
        trend = "stable"
        if recent_breaches:
            recent_count = len([b for b in recent_breaches if (datetime.now() - b['timestamp']).days <= 1])
            if recent_count > 2:
                trend = "deteriorating"
            elif recent_count == 0:
                trend = "improving"
        
        return {
            'period': period,
            'timestamp': datetime.now().isoformat(),
            'compliant': constraint_report.all_passed,
            'overall_risk_level': self._calculate_risk_level(constraint_report),
            'constraint_utilizations': utilizations,
            'active_breaches': [
                {
                    'constraint': b.constraint_name,
                    'severity': b.severity,
                    'message': b.message
                }
                for b in constraint_report.breaches
            ],
            'recent_breach_count': len(recent_breaches),
            'breach_trend': trend,
            'portfolio_summary': {
                'total_value': portfolio_state.total_value,
                'num_positions': len(portfolio_state.positions),
                'current_drawdown': portfolio_state.current_drawdown,
                'daily_turnover': portfolio_state.daily_turnover,
                'weekly_turnover': portfolio_state.weekly_turnover,
            }
        }
    
    def _calculate_risk_level(self, report: Any) -> str:
        """
        Calculate overall risk level from constraint report.
        
        Args:
            report: Constraint report
            
        Returns:
            Risk level string: LOW, MEDIUM, HIGH, or CRITICAL
        """
        if report.breaches:
            critical_breaches = [b for b in report.breaches if b.constraint_name == "MaxDrawdown"]
            if critical_breaches:
                return "CRITICAL"
            return "HIGH"
        
        if report.warnings:
            high_warnings = [w for w in report.warnings if w.utilization_pct > 90]
            if high_warnings:
                return "HIGH"
            return "MEDIUM"
        
        return "LOW"
