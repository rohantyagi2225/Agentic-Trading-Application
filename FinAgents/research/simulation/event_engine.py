"""External Event Simulation Engine.

This module simulates external market events such as earnings surprises,
Fed decisions, flash crashes, and geopolitical events.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from FinAgents.research.simulation.market_environment import MarketState


class EventType(str, Enum):
    """Enumeration of market event types."""

    EARNINGS_SURPRISE = "EARNINGS_SURPRISE"
    FED_DECISION = "FED_DECISION"
    FLASH_CRASH = "FLASH_CRASH"
    GEOPOLITICAL = "GEOPOLITICAL"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    NEWS_SENTIMENT_SHIFT = "NEWS_SENTIMENT_SHIFT"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"
    MARKET_RALLY = "MARKET_RALLY"


@dataclass
class MarketEvent:
    """Represents a market-affecting event.

    Attributes
    ----------
    event_id : str
        Unique identifier for the event.
    event_type : EventType
        Type of the event.
    timestamp : datetime
        When the event occurs.
    affected_symbols : list of str
        Symbols affected by the event.
    magnitude : float
        Event magnitude in range [-1, 1].
    price_impact_pct : float
        Immediate price impact as percentage.
    volatility_impact_pct : float
        Volatility impact as percentage.
    duration_steps : int
        Number of steps the impact lasts.
    description : str
        Human-readable description.
    metadata : dict
        Additional event metadata.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.NEWS_SENTIMENT_SHIFT
    timestamp: datetime = field(default_factory=datetime.utcnow)
    affected_symbols: List[str] = field(default_factory=list)
    magnitude: float = 0.0
    price_impact_pct: float = 0.0
    volatility_impact_pct: float = 0.0
    duration_steps: int = 1
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Internal tracking
    _steps_remaining: int = field(default=0, repr=False)
    _initial_price_impact: float = field(default=0.0, repr=False)
    _initial_vol_impact: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal tracking fields."""
        if isinstance(self.event_type, str):
            self.event_type = EventType(self.event_type)
        self._steps_remaining = self.duration_steps
        self._initial_price_impact = self.price_impact_pct
        self._initial_vol_impact = self.volatility_impact_pct

    def get_current_impact(self) -> tuple:
        """Get current impact with linear decay.

        Returns
        -------
        tuple
            (price_impact_pct, volatility_impact_pct) with decay applied.
        """
        if self.duration_steps <= 0:
            return 0.0, 0.0

        decay_factor = self._steps_remaining / self.duration_steps
        price_impact = self._initial_price_impact * decay_factor
        vol_impact = self._initial_vol_impact * decay_factor
        return price_impact, vol_impact

    def step(self) -> bool:
        """Advance the event by one step.

        Returns
        -------
        bool
            True if the event is still active, False if expired.
        """
        self._steps_remaining = max(0, self._steps_remaining - 1)
        return self._steps_remaining > 0


class EventEngine:
    """Engine for generating and applying market events.

    Simulates both scheduled events (earnings, Fed decisions) and
    random shocks (flash crashes, geopolitical events).

    Attributes
    ----------
    event_queue : list
        Queue of scheduled future events.
    event_history : list
        History of all events that have occurred.
    active_events : list
        Currently active events with remaining impact.
    """

    # Event templates with typical parameters
    EVENT_TEMPLATES = {
        EventType.EARNINGS_SURPRISE: {
            "price_impact_range": (-0.10, 0.10),
            "vol_impact_range": (0.2, 0.5),
            "duration_range": (1, 5),
            "description_templates": [
                "{symbol} reported earnings {surprise:.1%} vs expectations",
                "{symbol} earnings beat by {surprise:.1%}",
                "{symbol} misses earnings estimates by {surprise:.1%}",
            ],
        },
        EventType.FED_DECISION: {
            "price_impact_range": (-0.03, 0.03),
            "vol_impact_range": (0.3, 0.8),
            "duration_range": (5, 15),
            "description_templates": [
                "Fed {action} interest rates by {bps} basis points",
                "Federal Reserve announces {action} policy shift",
                "FOMC signals {sentiment} stance on monetary policy",
            ],
        },
        EventType.FLASH_CRASH: {
            "price_impact_range": (-0.15, -0.05),
            "vol_impact_range": (1.0, 3.0),
            "duration_range": (1, 3),
            "description_templates": [
                "Flash crash detected in {symbols}",
                "Liquidity vacuum triggers rapid price decline",
                "Algorithm-driven selloff in {symbols}",
            ],
        },
        EventType.GEOPOLITICAL: {
            "price_impact_range": (-0.08, 0.02),
            "vol_impact_range": (0.5, 1.5),
            "duration_range": (5, 20),
            "description_templates": [
                "Geopolitical tensions rise in {region}",
                "Trade dispute escalates between major economies",
                "Sanctions announced affecting {sector} sector",
            ],
        },
        EventType.SECTOR_ROTATION: {
            "price_impact_range": (-0.05, 0.05),
            "vol_impact_range": (0.1, 0.3),
            "duration_range": (3, 10),
            "description_templates": [
                "Sector rotation from {from_sector} to {to_sector}",
                "Institutional flows shift to {to_sector}",
                "Growth to value rotation accelerates",
            ],
        },
        EventType.NEWS_SENTIMENT_SHIFT: {
            "price_impact_range": (-0.04, 0.04),
            "vol_impact_range": (0.1, 0.4),
            "duration_range": (1, 5),
            "description_templates": [
                "Major news outlet reports {sentiment} outlook",
                "Analyst sentiment turns {sentiment} on {symbol}",
                "Social media sentiment shifts to {sentiment}",
            ],
        },
        EventType.LIQUIDITY_CRISIS: {
            "price_impact_range": (-0.12, -0.03),
            "vol_impact_range": (1.5, 4.0),
            "duration_range": (3, 10),
            "description_templates": [
                "Liquidity dries up across {symbols}",
                "Credit spreads widen sharply",
                "Market makers withdraw liquidity",
            ],
        },
        EventType.MARKET_RALLY: {
            "price_impact_range": (0.03, 0.10),
            "vol_impact_range": (-0.2, 0.2),
            "duration_range": (3, 10),
            "description_templates": [
                "Market rally driven by {driver}",
                "Risk-on sentiment fuels broad rally",
                "Short covering drives prices higher",
            ],
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the event engine.

        Parameters
        ----------
        config : dict, optional
            Configuration parameters:
            - scheduled_event_probability: float (default 0.05)
            - shock_probability: float (default 0.01)
            - max_events_per_step: int (default 2)
        """
        self.config = config or {}

        self.scheduled_event_probability = self.config.get(
            "scheduled_event_probability", 0.05
        )
        self.shock_probability = self.config.get("shock_probability", 0.01)
        self.max_events_per_step = self.config.get("max_events_per_step", 2)

        self._rng = np.random.default_rng()
        self.event_queue: List[MarketEvent] = []
        self.event_history: List[MarketEvent] = []
        self.active_events: List[MarketEvent] = []

    def schedule_event(self, event: MarketEvent) -> None:
        """Schedule an event for future execution.

        Parameters
        ----------
        event : MarketEvent
            Event to schedule.
        """
        self.event_queue.append(event)
        # Sort by timestamp
        self.event_queue.sort(key=lambda e: e.timestamp)

    def generate_random_events(
        self, current_step: int, symbols: List[str]
    ) -> List[MarketEvent]:
        """Generate random events based on configured probabilities.

        Parameters
        ----------
        current_step : int
            Current simulation step.
        symbols : list of str
            Available symbols.

        Returns
        -------
        list of MarketEvent
            Generated events.
        """
        events = []

        # Generate scheduled events (earnings, Fed decisions)
        if self._rng.random() < self.scheduled_event_probability:
            # Choose event type: earnings or Fed decision
            if self._rng.random() < 0.7:
                symbol = self._rng.choice(symbols)
                event = self._create_earnings_event(symbol)
            else:
                event = self._create_fed_event(symbols)
            events.append(event)

        # Generate random shocks (flash crash, geopolitical, etc.)
        if self._rng.random() < self.shock_probability:
            shock_type = self._rng.choice(
                [
                    EventType.FLASH_CRASH,
                    EventType.GEOPOLITICAL,
                    EventType.LIQUIDITY_CRISIS,
                    EventType.MARKET_RALLY,
                ]
            )
            affected = self._rng.choice(
                symbols, size=min(3, len(symbols)), replace=False
            ).tolist()

            if shock_type == EventType.FLASH_CRASH:
                event = self._create_flash_crash(affected)
            elif shock_type == EventType.GEOPOLITICAL:
                event = self._create_geopolitical_event(affected)
            elif shock_type == EventType.LIQUIDITY_CRISIS:
                event = self._create_liquidity_crisis(affected)
            else:
                event = self._create_market_rally(affected)
            events.append(event)

        # Generate sentiment shifts
        if self._rng.random() < self.scheduled_event_probability * 0.5:
            symbol = self._rng.choice(symbols)
            event = self._create_sentiment_shift(symbol)
            events.append(event)

        # Limit events per step
        if len(events) > self.max_events_per_step:
            events = list(self._rng.choice(events, size=self.max_events_per_step))

        return events

    def apply_events(
        self, events: List[MarketEvent], market_states: Dict[str, "MarketState"]
    ) -> Dict[str, "MarketState"]:
        """Apply event impacts to market states.

        Parameters
        ----------
        events : list of MarketEvent
            Events to apply.
        market_states : dict
            Current market states per symbol.

        Returns
        -------
        dict
            Modified market states.
        """
        for event in events:
            for symbol in event.affected_symbols:
                if symbol in market_states:
                    state = market_states[symbol]
                    price_impact, vol_impact = event.get_current_impact()

                    # Apply price impact
                    state.current_price *= 1 + price_impact
                    state.current_price = max(0.01, state.current_price)

                    # Apply volatility impact
                    state.volatility *= 1 + vol_impact
                    state.volatility = max(0.001, state.volatility)

                    # Update high/low
                    state.high = max(state.high, state.current_price)
                    state.low = min(state.low, state.current_price)

            # Add to active events and history
            if event not in self.active_events:
                self.active_events.append(event)
            if event not in self.event_history:
                self.event_history.append(event)

        return market_states

    def step_active_events(self) -> List[MarketEvent]:
        """Advance all active events and remove expired ones.

        Returns
        -------
        list of MarketEvent
            Events that are still active.
        """
        still_active = []
        for event in self.active_events:
            if event.step():
                still_active.append(event)
        self.active_events = still_active
        return still_active

    def get_active_events(self, current_step: int) -> List[MarketEvent]:
        """Get events whose effects haven't fully decayed.

        Parameters
        ----------
        current_step : int
            Current simulation step (unused, for interface compatibility).

        Returns
        -------
        list of MarketEvent
            Active events.
        """
        return self.active_events.copy()

    def get_event_history(self) -> List[MarketEvent]:
        """Get all events that have occurred.

        Returns
        -------
        list of MarketEvent
            Historical events.
        """
        return self.event_history.copy()

    def _create_earnings_event(self, symbol: str) -> MarketEvent:
        """Create an earnings surprise event.

        Parameters
        ----------
        symbol : str
            Symbol for earnings announcement.

        Returns
        -------
        MarketEvent
            The earnings event.
        """
        template = self.EVENT_TEMPLATES[EventType.EARNINGS_SURPRISE]
        surprise = self._rng.uniform(*template["price_impact_range"])
        vol_impact = self._rng.uniform(*template["vol_impact_range"])
        duration = self._rng.integers(*template["duration_range"])

        description = self._rng.choice(template["description_templates"])
        description = description.format(
            symbol=symbol, surprise=surprise, surprise_pct=surprise * 100
        )

        return MarketEvent(
            event_type=EventType.EARNINGS_SURPRISE,
            affected_symbols=[symbol],
            magnitude=abs(surprise),
            price_impact_pct=surprise,
            volatility_impact_pct=vol_impact,
            duration_steps=int(duration),
            description=description,
            metadata={"surprise_pct": surprise * 100},
        )

    def _create_fed_event(self, symbols: List[str]) -> MarketEvent:
        """Create a Fed decision event.

        Parameters
        ----------
        symbols : list of str
            Symbols to affect.

        Returns
        -------
        MarketEvent
            The Fed event.
        """
        template = self.EVENT_TEMPLATES[EventType.FED_DECISION]
        price_impact = self._rng.uniform(*template["price_impact_range"])
        vol_impact = self._rng.uniform(*template["vol_impact_range"])
        duration = self._rng.integers(*template["duration_range"])

        action = "raises" if price_impact > 0 else "cuts" if price_impact < 0 else "holds"
        bps = abs(int(price_impact * 1000))
        sentiment = "hawkish" if price_impact > 0 else "dovish" if price_impact < 0 else "neutral"

        description = self._rng.choice(template["description_templates"])
        description = description.format(
            action=action, bps=bps, sentiment=sentiment
        )

        return MarketEvent(
            event_type=EventType.FED_DECISION,
            affected_symbols=symbols,
            magnitude=abs(price_impact),
            price_impact_pct=price_impact,
            volatility_impact_pct=vol_impact,
            duration_steps=int(duration),
            description=description,
            metadata={"bps_change": bps, "stance": sentiment},
        )

    def _create_flash_crash(self, symbols: List[str]) -> MarketEvent:
        """Create a flash crash event.

        Parameters
        ----------
        symbols : list of str
            Symbols to affect.

        Returns
        -------
        MarketEvent
            The flash crash event.
        """
        template = self.EVENT_TEMPLATES[EventType.FLASH_CRASH]
        price_impact = self._rng.uniform(*template["price_impact_range"])
        vol_impact = self._rng.uniform(*template["vol_impact_range"])
        duration = self._rng.integers(*template["duration_range"])

        description = self._rng.choice(template["description_templates"])
        description = description.format(symbols=", ".join(symbols[:3]))

        return MarketEvent(
            event_type=EventType.FLASH_CRASH,
            affected_symbols=symbols,
            magnitude=abs(price_impact),
            price_impact_pct=price_impact,
            volatility_impact_pct=vol_impact,
            duration_steps=int(duration),
            description=description,
            metadata={"crash_type": "flash"},
        )

    def _create_geopolitical_event(self, symbols: List[str]) -> MarketEvent:
        """Create a geopolitical event.

        Parameters
        ----------
        symbols : list of str
            Symbols to affect.

        Returns
        -------
        MarketEvent
            The geopolitical event.
        """
        template = self.EVENT_TEMPLATES[EventType.GEOPOLITICAL]
        price_impact = self._rng.uniform(*template["price_impact_range"])
        vol_impact = self._rng.uniform(*template["vol_impact_range"])
        duration = self._rng.integers(*template["duration_range"])

        regions = ["Middle East", "Europe", "Asia-Pacific", "Americas"]
        sectors = ["energy", "technology", "financials", "healthcare"]

        description = self._rng.choice(template["description_templates"])
        description = description.format(
            region=self._rng.choice(regions),
            sector=self._rng.choice(sectors),
        )

        return MarketEvent(
            event_type=EventType.GEOPOLITICAL,
            affected_symbols=symbols,
            magnitude=abs(price_impact),
            price_impact_pct=price_impact,
            volatility_impact_pct=vol_impact,
            duration_steps=int(duration),
            description=description,
            metadata={"severity": "moderate"},
        )

    def _create_sentiment_shift(self, symbol: str) -> MarketEvent:
        """Create a sentiment shift event.

        Parameters
        ----------
        symbol : str
            Symbol to affect.

        Returns
        -------
        MarketEvent
            The sentiment shift event.
        """
        template = self.EVENT_TEMPLATES[EventType.NEWS_SENTIMENT_SHIFT]
        price_impact = self._rng.uniform(*template["price_impact_range"])
        vol_impact = self._rng.uniform(*template["vol_impact_range"])
        duration = self._rng.integers(*template["duration_range"])

        sentiment = "bullish" if price_impact > 0 else "bearish"

        description = self._rng.choice(template["description_templates"])
        description = description.format(
            sentiment=sentiment, symbol=symbol
        )

        return MarketEvent(
            event_type=EventType.NEWS_SENTIMENT_SHIFT,
            affected_symbols=[symbol],
            magnitude=abs(price_impact),
            price_impact_pct=price_impact,
            volatility_impact_pct=vol_impact,
            duration_steps=int(duration),
            description=description,
            metadata={"sentiment": sentiment},
        )

    def _create_liquidity_crisis(self, symbols: List[str]) -> MarketEvent:
        """Create a liquidity crisis event.

        Parameters
        ----------
        symbols : list of str
            Symbols to affect.

        Returns
        -------
        MarketEvent
            The liquidity crisis event.
        """
        template = self.EVENT_TEMPLATES[EventType.LIQUIDITY_CRISIS]
        price_impact = self._rng.uniform(*template["price_impact_range"])
        vol_impact = self._rng.uniform(*template["vol_impact_range"])
        duration = self._rng.integers(*template["duration_range"])

        description = self._rng.choice(template["description_templates"])
        description = description.format(symbols=", ".join(symbols[:3]))

        return MarketEvent(
            event_type=EventType.LIQUIDITY_CRISIS,
            affected_symbols=symbols,
            magnitude=abs(price_impact),
            price_impact_pct=price_impact,
            volatility_impact_pct=vol_impact,
            duration_steps=int(duration),
            description=description,
            metadata={"severity": "high"},
        )

    def _create_market_rally(self, symbols: List[str]) -> MarketEvent:
        """Create a market rally event.

        Parameters
        ----------
        symbols : list of str
            Symbols to affect.

        Returns
        -------
        MarketEvent
            The market rally event.
        """
        template = self.EVENT_TEMPLATES[EventType.MARKET_RALLY]
        price_impact = self._rng.uniform(*template["price_impact_range"])
        vol_impact = self._rng.uniform(*template["vol_impact_range"])
        duration = self._rng.integers(*template["duration_range"])

        drivers = [
            "strong earnings",
            "positive economic data",
            "central bank support",
            "trade deal progress",
        ]

        description = self._rng.choice(template["description_templates"])
        description = description.format(driver=self._rng.choice(drivers))

        return MarketEvent(
            event_type=EventType.MARKET_RALLY,
            affected_symbols=symbols,
            magnitude=price_impact,
            price_impact_pct=price_impact,
            volatility_impact_pct=vol_impact,
            duration_steps=int(duration),
            description=description,
            metadata={"rally_type": "broad"},
        )

    def reset(self) -> None:
        """Reset the event engine to initial state."""
        self.event_queue.clear()
        self.event_history.clear()
        self.active_events.clear()

    def set_random_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        self._rng = np.random.default_rng(seed)
