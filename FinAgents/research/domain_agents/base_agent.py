"""Base abstractions and core data structures for domain-specialized research agents.

This module defines the abstract ResearchAgent interface and shared dataclasses
used across all domain-specific agents (trader, risk manager, analyst,
portfolio manager, and domain adapter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class ActionType(str, Enum):
    """Enumeration of supported trading action types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketData:
    """Structured representation of market data for a single symbol.

    Attributes
    ----------
    symbol:
        Ticker symbol associated with the data.
    prices:
        Sequence of OHLCV-like dictionaries, e.g.::

            {"open": float, "high": float, "low": float,
             "close": float, "volume": float, "timestamp": str}

        The exact keys are flexible but should at minimum contain
        a ``close`` field for price-based computations.
    volume:
        Aggregate or latest volume for the period represented by ``prices``.
    timestamp:
        ISO 8601 timestamp associated with the most recent datapoint.
    regime:
        Optional regime label such as "bull", "bear", or
        "high_volatility".
    """

    symbol: str
    prices: List[Mapping[str, Any]]
    volume: float
    timestamp: str
    regime: Optional[str] = None


@dataclass
class MarketContext:
    """Contextual information surrounding current market conditions.

    Attributes
    ----------
    regime:
        High-level market regime label.
    volatility_level:
        Qualitative or numeric indicator of volatility (e.g., "low", "high",
        or a realized volatility estimate).
    sentiment:
        Aggregate sentiment score in the range ``[-1, 1]``.
    macro_indicators:
        Mapping from macroeconomic indicator names to values
        (e.g., GDP growth, inflation).
    events:
        List of notable events impacting the market (earnings, FOMC, etc.).
    """

    regime: Optional[str] = None
    volatility_level: Optional[float] = None
    sentiment: Optional[float] = None
    macro_indicators: Dict[str, float] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Intermediate reasoning output from an agent.

    This captures the agent's observations, inferences, and derived signals,
    along with an explicit reasoning chain that can be used for inspection
    and explainability.
    """

    observations: List[str] = field(default_factory=list)
    inferences: List[str] = field(default_factory=list)
    confidence: float = 0.0
    signals: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)


@dataclass
class Action:
    """Concrete action taken or proposed by an agent.

    For trading agents, this represents an order-like decision. For other
    agents (risk, portfolio), the semantics of ``action_type`` may be more
    abstract but still follow the same structure.
    """

    action_type: ActionType
    symbol: Optional[str] = None
    quantity: float = 0.0
    price: Optional[float] = None
    confidence: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning_summary: str = ""


@dataclass
class LearningUpdate:
    """Represents parameter updates learned from outcomes.

    Attributes
    ----------
    parameter_changes:
        Mapping of parameter names to their updated values or deltas.
    confidence_adjustment:
        Adjustment applied to the agent's internal calibration of
        confidence, typically in ``[-1, 1]``.
    lessons:
        Human-readable lessons derived from the learning process.
    """

    parameter_changes: Dict[str, Any] = field(default_factory=dict)
    confidence_adjustment: float = 0.0
    lessons: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    """Human-readable explanation of an agent's behavior.

    Attributes
    ----------
    summary:
        Short textual description of the decision or reasoning.
    reasoning_chain:
        Step-by-step narrative of how the conclusion was reached.
    data_sources:
        High-level description of the data and features considered.
    risk_justification:
        Explanation of how risk was evaluated and managed.
    confidence_breakdown:
        Mapping from factor names to confidence contributions.
    """

    summary: str
    reasoning_chain: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    risk_justification: str = ""
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)


class ResearchAgent(ABC):
    """Abstract base class for all domain-specialized research agents.

    Agents implementing this interface should provide a consistent workflow:

    1. :meth:`reason` over :class:`MarketData` and :class:`MarketContext`.
    2. :meth:`act` based on the resulting :class:`ReasoningResult`.
    3. :meth:`learn` from realized outcomes.
    4. :meth:`explain` decisions for downstream consumers.
    5. :meth:`get_state` to support serialization and persistence.
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent with an identifier and configuration.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent instance.
        config:
            Optional configuration dictionary. Concrete agents may interpret
            and store this in a structured form.
        """

        self.agent_id: str = agent_id
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def reason(self, market_data: MarketData, context: Optional[MarketContext] = None) -> ReasoningResult:
        """Analyze inputs and form structured reasoning output.

        Parameters
        ----------
        market_data:
            Primary market data for the agent's analysis.
        context:
            Optional broader market context.
        """

    @abstractmethod
    def act(self, reasoning_result: ReasoningResult) -> Action:
        """Decide and describe an action based on reasoning output."""

    @abstractmethod
    def learn(self, outcome: Mapping[str, Any]) -> LearningUpdate:
        """Update internal state based on realized outcomes.

        The ``outcome`` mapping can contain realized PnL, tracking error,
        or any other performance metrics relevant to the concrete agent.
        """

    @abstractmethod
    def explain(self, action: Action) -> Explanation:
        """Produce a human-readable explanation for the given action."""

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the agent's current state."""
