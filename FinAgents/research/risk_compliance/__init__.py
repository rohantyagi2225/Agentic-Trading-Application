"""
Risk and Compliance Constraints Module

This module provides comprehensive constraint management and regulatory compliance
for the trading system. It includes:

- Constraint system with various constraint types (drawdown, position size, etc.)
- Pre-trade and post-trade compliance checking
- Regulatory validation (wash sale, PDT, short selling rules)
- Risk dashboard for real-time monitoring
"""

from .constraints import (
    Constraint,
    ConstraintResult,
    ConstraintManager,
    ConstraintReport,
    PortfolioState,
    Position,
    MaxDrawdownConstraint,
    PositionSizeConstraint,
    ConcentrationConstraint,
    TurnoverConstraint,
    CorrelationConstraint,
)

from .compliance_engine import (
    ComplianceEngine,
    ComplianceDecision,
    ComplianceReport,
    BreachResponse,
)

from .regulatory_checks import (
    RegulatoryChecker,
    WashSaleResult,
    PDTResult,
    ShortSellResult,
    ReportingResult,
    RegulatoryReport,
    Trade,
)

from .risk_dashboard import (
    RiskDashboard,
    RiskMetrics,
    LimitUtilization,
    BreachRecord,
    RiskBudget,
)

__all__ = [
    # Constraints
    "Constraint",
    "ConstraintResult",
    "ConstraintManager",
    "ConstraintReport",
    "PortfolioState",
    "Position",
    "MaxDrawdownConstraint",
    "PositionSizeConstraint",
    "ConcentrationConstraint",
    "TurnoverConstraint",
    "CorrelationConstraint",
    # Compliance Engine
    "ComplianceEngine",
    "ComplianceDecision",
    "ComplianceReport",
    "BreachResponse",
    # Regulatory Checks
    "RegulatoryChecker",
    "WashSaleResult",
    "PDTResult",
    "ShortSellResult",
    "ReportingResult",
    "RegulatoryReport",
    "Trade",
    # Risk Dashboard
    "RiskDashboard",
    "RiskMetrics",
    "LimitUtilization",
    "BreachRecord",
    "RiskBudget",
]
