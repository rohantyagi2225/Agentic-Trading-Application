"""
FinAgents Next-Generation Research System
==========================================

A research-grade multi-agent financial system with domain-specialized agents,
multimodal intelligence, advanced coordination, and explainable AI.

Architecture Overview:
---------------------

1. DOMAIN-SPECIALIZED AGENTS (FFM Extension)
   - TraderAgent: Strategy learning (momentum, mean reversion, ML-based)
   - RiskAgent: VaR, CVaR, stress testing, regime detection
   - AnalystAgent: Macro + sentiment + event impact modeling
   - PortfolioAgent: Optimization, allocation, rebalancing

2. MULTIMODAL INTELLIGENCE PIPELINE
   - Time-series processing (prices, volumes)
   - Text processing (news, sentiment, filings)
   - Image processing (chart patterns, visual analysis)
   - Cross-modal fusion for unified decisions

3. ADVANCED COORDINATION PROTOCOL
   - Shared memory / blackboard system
   - Negotiation and voting mechanisms
   - Consensus-based decision making
   - Conflict resolution strategies

4. MEMORY & LEARNING SYSTEM
   - Episodic memory (past trades, outcomes)
   - Semantic memory (market knowledge)
   - Procedural memory (strategies, heuristics)
   - Reinforcement learning with feedback loops

5. EXPLAINABILITY FRAMEWORK
   - Decision reasoning chains
   - Data source attribution
   - Risk justification
   - Counterfactual analysis

6. MARKET SIMULATION ENVIRONMENT
   - Agent-based market dynamics
   - External shock modeling
   - Performance tracking (PnL, Sharpe, drawdown)
   - Regulatory compliance checks

Author: FinAgents Research Team
Version: 2.0.0 (Research Edition)
"""

__version__ = "2.0.0-research"
__author__ = "FinAgents Research Team"

from .agents.trader_agent import TraderAgent
from .agents.risk_agent import RiskAgent
from .agents.analyst_agent import AnalystAgent
from .agents.portfolio_agent import PortfolioAgent
from .coordination.agent_coordinator import AgentCoordinator
from .memory.unified_memory import UnifiedMemorySystem
from .explainability.explainer import ExplainabilityEngine
from .environment.market_simulation import MarketSimulationEnvironment
from .evaluation.evaluation_framework import EvaluationFramework

__all__ = [
    "TraderAgent",
    "RiskAgent", 
    "AnalystAgent",
    "PortfolioAgent",
    "AgentCoordinator",
    "UnifiedMemorySystem",
    "ExplainabilityEngine",
    "MarketSimulationEnvironment",
    "EvaluationFramework",
]
