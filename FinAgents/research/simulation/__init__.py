"""Market Simulation Environment for Multi-Agent Trading.

This module provides a realistic market simulation where trading agents
interact in a dynamic environment with price evolution, order books,
external events, and comprehensive performance tracking.

Key Components
--------------
MarketEnvironment
    Dynamic market simulation with GBM price evolution, order books,
    and market regime transitions.

EventEngine
    External event simulation including earnings surprises, Fed decisions,
    flash crashes, and geopolitical events.

SimulationRunner
    End-to-end simulation orchestrator coordinating agents, markets,
    and events.

PerformanceTracker
    Real-time simulation metrics including Sharpe ratio, drawdowns,
    agent attribution, and benchmark comparison.
"""

from FinAgents.research.simulation.market_environment import (
    L2OrderBook,
    MarketEnvironment,
    MarketRegime,
    MarketState,
    Order,
    OrderSide,
    OrderType,
)
from FinAgents.research.simulation.event_engine import (
    EventEngine,
    EventType,
    MarketEvent,
)
from FinAgents.research.simulation.simulation_runner import (
    SimulationConfig,
    SimulationResult,
    SimulationRunner,
    SimulationSnapshot,
)
from FinAgents.research.simulation.performance_tracker import (
    AgentAttribution,
    BenchmarkComparison,
    PerformanceTracker,
)

__all__ = [
    # Market Environment
    "MarketEnvironment",
    "MarketState",
    "MarketRegime",
    "L2OrderBook",
    "Order",
    "OrderType",
    "OrderSide",
    # Event Engine
    "EventEngine",
    "EventType",
    "MarketEvent",
    # Simulation Runner
    "SimulationRunner",
    "SimulationConfig",
    "SimulationResult",
    "SimulationSnapshot",
    # Performance Tracker
    "PerformanceTracker",
    "AgentAttribution",
    "BenchmarkComparison",
]
