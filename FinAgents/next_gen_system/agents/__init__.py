"""
Domain-Specialized Financial Agents
===================================

This module implements the next-generation domain-specialized agents
with advanced financial reasoning capabilities.

Agent Types:
-----------
1. TraderAgent - Strategy execution with ML-based learning
2. RiskAgent - Comprehensive risk modeling (VaR, CVaR, stress testing)
3. AnalystAgent - Multimodal analysis (macro, sentiment, events)
4. PortfolioAgent - Optimization and allocation
"""

from .trader_agent import TraderAgent
from .risk_agent import RiskAgent
from .analyst_agent import AnalystAgent
from .portfolio_agent import PortfolioAgent

__all__ = [
    "TraderAgent",
    "RiskAgent",
    "AnalystAgent",
    "PortfolioAgent",
]
