"""Memory and Learning System for financial agents.

This module provides a comprehensive memory learning system that includes:
- Structured trade memory storage and retrieval
- Experience replay with prioritized sampling
- Reward signal computation for reinforcement learning
- Feedback-based learning with confidence calibration

Main Components
---------------
TradeMemoryStore :
    In-memory store for trade records with indexing and analytics.

ExperienceReplayBuffer :
    Prioritized experience replay buffer for learning.

RewardEngine :
    Multi-component reward signal computation.

LearningLoop :
    Feedback-based learning system with calibration and decay detection.
"""

from FinAgents.research.memory_learning.trade_memory import (
    TradeRecord,
    TradeMemoryStore,
    StrategyPerformance,
    RegimePerformance,
)
from FinAgents.research.memory_learning.experience_replay import (
    Experience,
    ExperienceReplayBuffer,
)
from FinAgents.research.memory_learning.reward_engine import (
    RewardConfig,
    RewardEngine,
    RewardBreakdown,
    REGIME_STRATEGY_MAP,
)
from FinAgents.research.memory_learning.learning_loop import (
    LearningLoop,
    PostTradeReport,
    LearningStepResult,
    CalibrationResult,
    DecayReport,
    ReviewReport,
)

__all__ = [
    # Trade Memory
    "TradeRecord",
    "TradeMemoryStore",
    "StrategyPerformance",
    "RegimePerformance",
    # Experience Replay
    "Experience",
    "ExperienceReplayBuffer",
    # Reward Engine
    "RewardConfig",
    "RewardEngine",
    "RewardBreakdown",
    "REGIME_STRATEGY_MAP",
    # Learning Loop
    "LearningLoop",
    "PostTradeReport",
    "LearningStepResult",
    "CalibrationResult",
    "DecayReport",
    "ReviewReport",
]
