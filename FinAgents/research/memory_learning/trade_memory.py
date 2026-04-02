"""Structured Trade Memory Store for financial agents.

This module provides a comprehensive trade memory system that stores,
indexes, and retrieves historical trade records with performance analytics.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TradeRecord:
    """Structured representation of a completed or open trade.

    Attributes
    ----------
    trade_id :
        Unique identifier for the trade (UUID string).
    symbol :
        Ticker symbol traded.
    action :
        Trade direction - 'BUY' or 'SELL'.
    entry_price :
        Price at which the position was entered.
    exit_price :
        Price at which the position was exited. None if still open.
    quantity :
        Number of shares/units traded.
    entry_time :
        Timestamp when the position was opened.
    exit_time :
        Timestamp when the position was closed. None if still open.
    pnl :
        Profit/loss in absolute terms. None if still open.
    pnl_pct :
        Profit/loss as percentage of entry value. None if still open.
    strategy :
        Name of the strategy that generated the trade signal.
    agent_id :
        Identifier of the agent that executed the trade.
    market_regime :
        Market regime classification at time of entry (e.g., 'bull', 'bear', 'sideways').
    volatility_at_entry :
        Volatility measure (e.g., realized vol) at time of entry.
    reasoning_summary :
        Brief summary of the reasoning behind the trade decision.
    confidence_at_entry :
        Agent's confidence level at time of entry (0.0 to 1.0).
    signals_at_entry :
        Dictionary of signal values/features present at entry.
    outcome_tags :
        List of descriptive tags for the trade outcome (e.g., ['winner', 'momentum', 'high_vol']).
    """

    trade_id: str
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    strategy: str
    agent_id: str
    market_regime: str
    volatility_at_entry: float
    reasoning_summary: str
    confidence_at_entry: float
    signals_at_entry: Dict[str, Any]
    outcome_tags: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        symbol: str,
        action: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        agent_id: str,
        market_regime: str,
        volatility_at_entry: float,
        reasoning_summary: str,
        confidence_at_entry: float,
        signals_at_entry: Dict[str, Any],
        entry_time: Optional[datetime] = None,
        trade_id: Optional[str] = None,
    ) -> "TradeRecord":
        """Factory method to create a new open trade record.

        Parameters
        ----------
        symbol :
            Ticker symbol.
        action :
            'BUY' or 'SELL'.
        entry_price :
            Entry price.
        quantity :
            Position size.
        strategy :
            Strategy name.
        agent_id :
            Agent identifier.
        market_regime :
            Market regime at entry.
        volatility_at_entry :
            Volatility at entry.
        reasoning_summary :
            Trade reasoning.
        confidence_at_entry :
            Confidence level (0-1).
        signals_at_entry :
            Signal dictionary at entry.
        entry_time :
            Entry timestamp. Defaults to current time.
        trade_id :
            Optional trade ID. Defaults to generated UUID.

        Returns
        -------
        TradeRecord
            New trade record with exit fields set to None.
        """
        return cls(
            trade_id=trade_id or str(uuid.uuid4()),
            symbol=symbol.upper(),
            action=action.upper(),
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            entry_time=entry_time or datetime.now(),
            exit_time=None,
            pnl=None,
            pnl_pct=None,
            strategy=strategy,
            agent_id=agent_id,
            market_regime=market_regime,
            volatility_at_entry=volatility_at_entry,
            reasoning_summary=reasoning_summary,
            confidence_at_entry=confidence_at_entry,
            signals_at_entry=signals_at_entry,
            outcome_tags=[],
        )

    def close(
        self,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        outcome_tags: Optional[List[str]] = None,
    ) -> None:
        """Close an open trade and compute PnL.

        Parameters
        ----------
        exit_price :
            Exit price.
        exit_time :
            Exit timestamp. Defaults to current time.
        outcome_tags :
            Optional list of outcome tags.
        """
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()

        if self.action == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl = (self.entry_price - exit_price) * self.quantity

        entry_value = self.entry_price * self.quantity
        self.pnl_pct = (self.pnl / entry_value) * 100 if entry_value != 0 else 0.0

        if outcome_tags:
            self.outcome_tags = outcome_tags
        else:
            self.outcome_tags = self._generate_outcome_tags()

    def _generate_outcome_tags(self) -> List[str]:
        """Generate outcome tags based on trade characteristics."""
        tags = []

        if self.pnl is not None:
            if self.pnl > 0:
                tags.append("winner")
            else:
                tags.append("loser")

            if abs(self.pnl_pct or 0) > 5:
                tags.append("high_impact")

        if self.volatility_at_entry > 0.02:
            tags.append("high_vol")
        elif self.volatility_at_entry < 0.01:
            tags.append("low_vol")

        if self.strategy:
            tags.append(self.strategy.lower().replace(" ", "_"))

        if self.market_regime:
            tags.append(self.market_regime.lower())

        return tags

    @property
    def holding_period(self) -> Optional[timedelta]:
        """Return the holding period for closed trades."""
        if self.exit_time and self.entry_time:
            return self.exit_time - self.entry_time
        return None

    @property
    def is_open(self) -> bool:
        """Check if the trade is still open."""
        return self.exit_price is None

    @property
    def is_closed(self) -> bool:
        """Check if the trade is closed."""
        return self.exit_price is not None


@dataclass
class StrategyPerformance:
    """Performance metrics for a trading strategy.

    Attributes
    ----------
    strategy :
        Strategy name.
    total_trades :
        Total number of trades executed.
    win_rate :
        Percentage of profitable trades (0.0 to 1.0).
    avg_pnl :
        Average profit/loss per trade.
    total_pnl :
        Cumulative profit/loss.
    sharpe :
        Approximate Sharpe ratio (annualized if possible).
    max_drawdown :
        Maximum peak-to-trough decline.
    avg_holding_period :
        Average time position was held.
    """

    strategy: str
    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_period: timedelta = field(default_factory=lambda: timedelta(0))


@dataclass
class RegimePerformance:
    """Performance metrics by market regime.

    Attributes
    ----------
    regime :
        Market regime identifier.
    total_trades :
        Total trades in this regime.
    win_rate :
        Win rate in this regime.
    avg_pnl :
        Average PnL in this regime.
    best_strategy :
        Best performing strategy in this regime.
    worst_strategy :
        Worst performing strategy in this regime.
    """

    regime: str
    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    best_strategy: str = ""
    worst_strategy: str = ""


class TradeMemoryStore:
    """In-memory store for trade records with indexing and analytics.

    This class provides a structured memory system for storing trade
    records, querying by various attributes, and computing performance
    analytics.

    Parameters
    ----------
    max_capacity :
        Maximum number of trades to store. Oldest trades are removed
        when capacity is exceeded.
    """

    def __init__(self, max_capacity: int = 10000) -> None:
        """Initialize the trade memory store."""
        self.max_capacity = max_capacity
        self._trades: List[TradeRecord] = []
        
        # Indexes for faster querying
        self._by_symbol: Dict[str, List[int]] = {}
        self._by_strategy: Dict[str, List[int]] = {}
        self._by_regime: Dict[str, List[int]] = {}
        self._by_agent: Dict[str, List[int]] = {}

    def store(self, record: TradeRecord) -> str:
        """Store a trade record in memory.

        If the store is at capacity, the oldest trade is removed.

        Parameters
        ----------
        record :
            Trade record to store.

        Returns
        -------
        str
            The trade_id of the stored record.
        """
        # Remove oldest if at capacity
        if len(self._trades) >= self.max_capacity:
            self._remove_oldest()

        # Add to main storage
        index = len(self._trades)
        self._trades.append(record)

        # Update indexes
        self._add_to_index(self._by_symbol, record.symbol.upper(), index)
        self._add_to_index(self._by_strategy, record.strategy, index)
        self._add_to_index(self._by_regime, record.market_regime, index)
        self._add_to_index(self._by_agent, record.agent_id, index)

        return record.trade_id

    def _add_to_index(self, index_dict: Dict[str, List[int]], key: str, idx: int) -> None:
        """Add an index entry."""
        if key not in index_dict:
            index_dict[key] = []
        index_dict[key].append(idx)

    def _remove_from_index(self, index_dict: Dict[str, List[int]], key: str, idx: int) -> None:
        """Remove an index entry."""
        if key in index_dict:
            if idx in index_dict[key]:
                index_dict[key].remove(idx)
            if not index_dict[key]:
                del index_dict[key]

    def _remove_oldest(self) -> None:
        """Remove the oldest trade record and update indexes."""
        if not self._trades:
            return

        # Remove from indexes
        record = self._trades[0]
        self._remove_from_index(self._by_symbol, record.symbol.upper(), 0)
        self._remove_from_index(self._by_strategy, record.strategy, 0)
        self._remove_from_index(self._by_regime, record.market_regime, 0)
        self._remove_from_index(self._by_agent, record.agent_id, 0)

        # Remove from storage
        self._trades.pop(0)

        # Decrement all remaining index values
        for key in self._by_symbol:
            self._by_symbol[key] = [i - 1 for i in self._by_symbol[key]]
        for key in self._by_strategy:
            self._by_strategy[key] = [i - 1 for i in self._by_strategy[key]]
        for key in self._by_regime:
            self._by_regime[key] = [i - 1 for i in self._by_regime[key]]
        for key in self._by_agent:
            self._by_agent[key] = [i - 1 for i in self._by_agent[key]]

    def get_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        """Retrieve a trade record by ID.

        Parameters
        ----------
        trade_id :
            Trade identifier.

        Returns
        -------
        TradeRecord or None
            The trade record if found, else None.
        """
        for trade in self._trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def query(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        agent_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        limit: int = 100,
    ) -> List[TradeRecord]:
        """Query trade records with multiple filters (AND logic).

        Parameters
        ----------
        symbol :
            Filter by ticker symbol.
        strategy :
            Filter by strategy name.
        regime :
            Filter by market regime.
        agent_id :
            Filter by agent identifier.
        start_date :
            Filter by entry time >= start_date.
        end_date :
            Filter by entry time <= end_date.
        min_pnl :
            Filter by minimum PnL.
        max_pnl :
            Filter by maximum PnL.
        limit :
            Maximum number of records to return.

        Returns
        -------
        List[TradeRecord]
            Matching trade records.
        """
        results: List[TradeRecord] = []

        for trade in self._trades:
            # Apply filters
            if symbol and trade.symbol.upper() != symbol.upper():
                continue
            if strategy and trade.strategy != strategy:
                continue
            if regime and trade.market_regime != regime:
                continue
            if agent_id and trade.agent_id != agent_id:
                continue
            if start_date and trade.entry_time < start_date:
                continue
            if end_date and trade.entry_time > end_date:
                continue
            if min_pnl is not None and (trade.pnl is None or trade.pnl < min_pnl):
                continue
            if max_pnl is not None and (trade.pnl is None or trade.pnl > max_pnl):
                continue

            results.append(trade)

            if len(results) >= limit:
                break

        return results

    def get_all(self) -> List[TradeRecord]:
        """Return all stored trade records."""
        return list(self._trades)

    def get_closed_trades(self) -> List[TradeRecord]:
        """Return all closed trade records."""
        return [t for t in self._trades if t.is_closed]

    def get_open_trades(self) -> List[TradeRecord]:
        """Return all open trade records."""
        return [t for t in self._trades if t.is_open]

    def get_performance_by_strategy(self) -> Dict[str, StrategyPerformance]:
        """Compute performance metrics grouped by strategy.

        Returns
        -------
        Dict[str, StrategyPerformance]
            Performance metrics for each strategy.
        """
        strategy_trades: Dict[str, List[TradeRecord]] = {}

        for trade in self._trades:
            if trade.is_closed:
                if trade.strategy not in strategy_trades:
                    strategy_trades[trade.strategy] = []
                strategy_trades[trade.strategy].append(trade)

        performance = {}
        for strategy, trades in strategy_trades.items():
            performance[strategy] = self._compute_strategy_performance(
                strategy, trades
            )

        return performance

    def _compute_strategy_performance(
        self, strategy: str, trades: List[TradeRecord]
    ) -> StrategyPerformance:
        """Compute performance metrics for a list of trades."""
        if not trades:
            return StrategyPerformance(strategy=strategy)

        pnls = [t.pnl for t in trades if t.pnl is not None]
        if not pnls:
            return StrategyPerformance(strategy=strategy, total_trades=len(trades))

        pnls_array = np.array(pnls)
        wins = pnls_array[pnls_array > 0]
        total_pnl = float(np.sum(pnls_array))
        avg_pnl = float(np.mean(pnls_array))
        win_rate = len(wins) / len(pnls) if pnls else 0.0

        # Sharpe approximation (assuming daily returns)
        if len(pnls) > 1 and np.std(pnls_array) > 0:
            sharpe = float(np.mean(pnls_array) / np.std(pnls_array) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumsum(pnls_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Average holding period
        holding_periods = [
            t.holding_period for t in trades if t.holding_period is not None
        ]
        if holding_periods:
            avg_holding = sum(holding_periods, timedelta()) / len(holding_periods)
        else:
            avg_holding = timedelta(0)

        return StrategyPerformance(
            strategy=strategy,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            avg_holding_period=avg_holding,
        )

    def get_performance_by_regime(self) -> Dict[str, RegimePerformance]:
        """Compute performance metrics grouped by market regime.

        Returns
        -------
        Dict[str, RegimePerformance]
            Performance metrics for each regime.
        """
        regime_trades: Dict[str, List[TradeRecord]] = {}

        for trade in self._trades:
            if trade.is_closed:
                if trade.market_regime not in regime_trades:
                    regime_trades[trade.market_regime] = []
                regime_trades[trade.market_regime].append(trade)

        performance = {}
        for regime, trades in regime_trades.items():
            performance[regime] = self._compute_regime_performance(regime, trades)

        return performance

    def _compute_regime_performance(
        self, regime: str, trades: List[TradeRecord]
    ) -> RegimePerformance:
        """Compute performance metrics for a regime."""
        if not trades:
            return RegimePerformance(regime=regime)

        pnls = [t.pnl for t in trades if t.pnl is not None]
        if not pnls:
            return RegimePerformance(regime=regime, total_trades=len(trades))

        pnls_array = np.array(pnls)
        wins = pnls_array[pnls_array > 0]
        avg_pnl = float(np.mean(pnls_array))
        win_rate = len(wins) / len(pnls) if pnls else 0.0

        # Find best and worst strategies
        strategy_pnls: Dict[str, float] = {}
        for trade in trades:
            if trade.pnl is not None:
                if trade.strategy not in strategy_pnls:
                    strategy_pnls[trade.strategy] = 0.0
                strategy_pnls[trade.strategy] += trade.pnl

        if strategy_pnls:
            best_strategy = max(strategy_pnls.keys(), key=lambda s: strategy_pnls[s])
            worst_strategy = min(strategy_pnls.keys(), key=lambda s: strategy_pnls[s])
        else:
            best_strategy = ""
            worst_strategy = ""

        return RegimePerformance(
            regime=regime,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            best_strategy=best_strategy,
            worst_strategy=worst_strategy,
        )

    def get_similar_trades(
        self,
        current_conditions: Dict[str, Any],
        top_k: int = 5,
    ) -> List[TradeRecord]:
        """Find past trades in similar market conditions.

        Similarity is computed as a weighted combination of:
        - Regime match (exact match bonus)
        - Volatility proximity
        - Signal cosine similarity

        Parameters
        ----------
        current_conditions :
            Dictionary containing 'regime', 'volatility', and 'signals' keys.
        top_k :
            Number of similar trades to return.

        Returns
        -------
        List[TradeRecord]
            Most similar historical trades.
        """
        current_regime = current_conditions.get("regime", "")
        current_volatility = current_conditions.get("volatility", 0.0)
        current_signals = current_conditions.get("signals", {})

        similarities: List[Tuple[float, int]] = []

        for idx, trade in enumerate(self._trades):
            if not trade.is_closed:
                continue

            similarity = self._compute_similarity(
                trade,
                current_regime,
                current_volatility,
                current_signals,
            )
            similarities.append((similarity, idx))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        return [self._trades[idx] for _, idx in similarities[:top_k]]

    def _compute_similarity(
        self,
        trade: TradeRecord,
        current_regime: str,
        current_volatility: float,
        current_signals: Dict[str, Any],
    ) -> float:
        """Compute similarity score between a trade and current conditions."""
        score = 0.0

        # Regime match (0-1, exact match = 1)
        if trade.market_regime == current_regime:
            score += 0.4

        # Volatility proximity (0-0.3)
        if current_volatility > 0 and trade.volatility_at_entry > 0:
            vol_diff = abs(trade.volatility_at_entry - current_volatility)
            vol_similarity = max(0, 1 - vol_diff / max(current_volatility, 0.05))
            score += 0.3 * vol_similarity

        # Signal cosine similarity (0-0.3)
        if current_signals and trade.signals_at_entry:
            signal_sim = self._cosine_similarity(current_signals, trade.signals_at_entry)
            score += 0.3 * signal_sim

        return score

    def _cosine_similarity(
        self,
        vec1: Dict[str, Any],
        vec2: Dict[str, Any],
    ) -> float:
        """Compute cosine similarity between two signal dictionaries."""
        # Find common keys
        all_keys = set(vec1.keys()) | set(vec2.keys())
        if not all_keys:
            return 0.0

        # Extract numeric values
        v1 = []
        v2 = []
        for key in all_keys:
            val1 = vec1.get(key, 0)
            val2 = vec2.get(key, 0)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                v1.append(val1)
                v2.append(val2)

        if not v1 or not v2:
            return 0.0

        v1_array = np.array(v1)
        v2_array = np.array(v2)

        dot_product = np.dot(v1_array, v2_array)
        norm1 = np.linalg.norm(v1_array)
        norm2 = np.linalg.norm(v2_array)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_attribution(self, trade_id: str) -> Dict[str, Any]:
        """Compute performance attribution for a trade.

        Attribution analyzes how much of PnL was due to:
        - Signal quality vs timing
        - Market movement vs alpha generation

        Parameters
        ----------
        trade_id :
            Trade identifier.

        Returns
        -------
        Dict[str, Any]
            Attribution breakdown.
        """
        trade = self.get_by_id(trade_id)
        if not trade or not trade.is_closed:
            return {"error": "Trade not found or still open"}

        attribution = {
            "trade_id": trade_id,
            "symbol": trade.symbol,
            "action": trade.action,
            "total_pnl": trade.pnl,
            "total_pnl_pct": trade.pnl_pct,
            "signal_contribution": 0.0,
            "timing_contribution": 0.0,
            "market_contribution": 0.0,
            "alpha_contribution": 0.0,
            "details": {},
        }

        # Signal quality: based on confidence and actual outcome
        if trade.pnl is not None:
            # If profitable, signal contributed positively
            signal_quality = trade.confidence_at_entry if trade.pnl > 0 else -trade.confidence_at_entry
            attribution["signal_contribution"] = signal_quality * abs(trade.pnl)

        # Market movement: estimate what market did during holding
        # Using volatility as proxy for expected move
        if trade.volatility_at_entry > 0 and trade.holding_period:
            holding_days = trade.holding_period.total_seconds() / 86400
            expected_move = trade.volatility_at_entry * np.sqrt(holding_days) * trade.entry_price
            
            if trade.pnl is not None:
                # Market contribution: expected move vs actual
                attribution["market_contribution"] = expected_move
                
                # Alpha: actual - expected
                attribution["alpha_contribution"] = trade.pnl - expected_move

        # Timing: compare entry price to potential better timing
        # Using volatility as a proxy for intra-period price range
        if trade.volatility_at_entry > 0:
            # Potential improvement from better timing
            potential_improvement = trade.volatility_at_entry * trade.entry_price * 0.5
            attribution["timing_contribution"] = potential_improvement

        attribution["details"] = {
            "confidence_at_entry": trade.confidence_at_entry,
            "volatility_at_entry": trade.volatility_at_entry,
            "holding_period": str(trade.holding_period) if trade.holding_period else None,
            "strategy": trade.strategy,
            "regime": trade.market_regime,
        }

        return attribution

    def clear(self) -> None:
        """Clear all stored trades and indexes."""
        self._trades.clear()
        self._by_symbol.clear()
        self._by_strategy.clear()
        self._by_regime.clear()
        self._by_agent.clear()

    def __len__(self) -> int:
        """Return the number of stored trades."""
        return len(self._trades)

    def __repr__(self) -> str:
        """String representation of the store."""
        return (
            f"TradeMemoryStore(trades={len(self._trades)}, "
            f"capacity={self.max_capacity}, "
            f"strategies={len(self._by_strategy)}, "
            f"regimes={len(self._by_regime)})"
        )
