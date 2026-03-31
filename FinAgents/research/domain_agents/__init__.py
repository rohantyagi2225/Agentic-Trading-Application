"""Unified domain agents package for research-grade financial workflows.

This package exposes a consistent interface for domain-specialized agents
that sit on top of the existing agent pools infrastructure.
"""

from .base_agent import (
    Action,
    ActionType,
    Explanation,
    LearningUpdate,
    MarketContext,
    MarketData,
    ReasoningResult,
    ResearchAgent,
)
from .trader_agent import TraderAgent
from .risk_manager_agent import RiskManagerAgent
from .analyst_agent import AnalystAgent
from .portfolio_manager_agent import PortfolioManagerAgent
from .domain_adaptation import DomainAdapter, RegimeProfile

__all__ = [
    "Action",
    "ActionType",
    "Explanation",
    "LearningUpdate",
    "MarketContext",
    "MarketData",
    "ReasoningResult",
    "ResearchAgent",
    "TraderAgent",
    "RiskManagerAgent",
    "AnalystAgent",
    "PortfolioManagerAgent",
    "DomainAdapter",
    "RegimeProfile",
]
