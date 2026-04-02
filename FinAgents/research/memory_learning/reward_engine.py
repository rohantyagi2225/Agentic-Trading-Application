"""Reward Signal Computation for financial agent learning.

This module provides a configurable reward engine that computes multi-component
reward signals for trade outcomes, supporting reinforcement learning approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from FinAgents.research.memory_learning.trade_memory import TradeRecord


# Regime-strategy appropriateness mapping
REGIME_STRATEGY_MAP: Dict[str, List[str]] = {
    "bull": ["momentum", "trend_following", "breakout"],
    "bear": ["defensive", "short", "hedged"],
    "sideways": ["mean_reversion", "range_trading", "pairs"],
    "high_vol": ["reduced_position", "volatility_arbitrage", "hedged"],
    "low_vol": ["momentum", "carry"],
    "trending": ["momentum", "trend_following"],
    "range_bound": ["mean_reversion", "range_trading"],
}


@dataclass
class RewardConfig:
    """Configuration for reward computation.

    Attributes
    ----------
    risk_adjusted_weight :
        Weight for the risk-adjusted return component.
    drawdown_penalty_weight :
        Weight for the drawdown penalty component.
    accuracy_bonus_weight :
        Weight for the prediction accuracy bonus component.
    regime_bonus_weight :
        Weight for the regime-appropriate behavior bonus.
    max_reward :
        Maximum absolute reward value (clipping).
    regime_strategy_map :
        Mapping of regimes to appropriate strategies.
    """

    risk_adjusted_weight: float = 0.4
    drawdown_penalty_weight: float = 0.2
    accuracy_bonus_weight: float = 0.2
    regime_bonus_weight: float = 0.2
    max_reward: float = 10.0
    regime_strategy_map: Dict[str, List[str]] = field(
        default_factory=lambda: REGIME_STRATEGY_MAP
    )


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components.

    Attributes
    ----------
    total_reward :
        Final computed reward (weighted sum, clipped).
    components :
        Dictionary of individual component values.
    risk_adjusted_component :
        Risk-adjusted return contribution.
    drawdown_component :
        Drawdown penalty contribution.
    accuracy_component :
        Prediction accuracy bonus contribution.
    regime_component :
        Regime-appropriate behavior contribution.
    details :
        Human-readable explanation of the reward.
    """

    total_reward: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    risk_adjusted_component: float = 0.0
    drawdown_component: float = 0.0
    accuracy_component: float = 0.0
    regime_component: float = 0.0
    details: str = ""


class RewardEngine:
    """Engine for computing multi-component reward signals.

    This class computes rewards for trade outcomes based on configurable
    components including risk-adjusted returns, drawdown penalties,
    prediction accuracy, and regime-appropriate behavior.

    Parameters
    ----------
    config :
        Reward configuration. Uses defaults if not provided.
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        """Initialize the reward engine with optional configuration."""
        self.config = config or RewardConfig()

    def compute_reward(
        self,
        trade_record: TradeRecord,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> RewardBreakdown:
        """Compute reward for a single trade.

        Parameters
        ----------
        trade_record :
            The trade record to evaluate.
        portfolio_context :
            Optional portfolio context containing:
            - 'drawdown_contribution': Trade's contribution to portfolio drawdown
            - 'max_allowed_drawdown': Maximum allowed portfolio drawdown
            - 'predicted_direction': Agent's predicted direction ('UP' or 'DOWN')

        Returns
        -------
        RewardBreakdown
            Detailed breakdown of the computed reward.
        """
        if trade_record.pnl is None:
            return RewardBreakdown(
                total_reward=0.0,
                details="Trade still open - no reward computed yet",
            )

        components: Dict[str, float] = {}
        portfolio_context = portfolio_context or {}

        # 1. Risk-adjusted return component
        risk_adjusted = self._compute_risk_adjusted_component(trade_record)
        components["risk_adjusted"] = risk_adjusted

        # 2. Drawdown penalty component
        drawdown = self._compute_drawdown_component(trade_record, portfolio_context)
        components["drawdown_penalty"] = drawdown

        # 3. Prediction accuracy bonus
        accuracy = self._compute_accuracy_component(trade_record, portfolio_context)
        components["accuracy_bonus"] = accuracy

        # 4. Regime-appropriate behavior
        regime = self._compute_regime_component(trade_record)
        components["regime_bonus"] = regime

        # Weighted sum
        total = (
            self.config.risk_adjusted_weight * risk_adjusted
            + self.config.drawdown_penalty_weight * drawdown
            + self.config.accuracy_bonus_weight * accuracy
            + self.config.regime_bonus_weight * regime
        )

        # Clip to bounds
        total = float(np.clip(total, -self.config.max_reward, self.config.max_reward))

        # Generate details string
        details = self._generate_details(
            total, risk_adjusted, drawdown, accuracy, regime, trade_record
        )

        return RewardBreakdown(
            total_reward=total,
            components=components,
            risk_adjusted_component=risk_adjusted,
            drawdown_component=drawdown,
            accuracy_component=accuracy,
            regime_component=regime,
            details=details,
        )

    def _compute_risk_adjusted_component(
        self, trade_record: TradeRecord
    ) -> float:
        """Compute risk-adjusted return component.

        Uses a Sharpe-like measure per trade: pnl_pct / volatility_at_entry.
        Profits in volatile markets are rewarded more since they're harder to achieve.
        """
        if trade_record.pnl_pct is None:
            return 0.0

        volatility = trade_record.volatility_at_entry
        if volatility <= 0:
            # Avoid division by zero, use raw pnl_pct
            return trade_record.pnl_pct / 10.0  # Scale to reasonable range

        # Risk-adjusted: reward profits in volatile conditions
        # Higher volatility = harder to profit = higher reward when successful
        risk_adjusted = trade_record.pnl_pct / (volatility * 100)

        return risk_adjusted

    def _compute_drawdown_component(
        self,
        trade_record: TradeRecord,
        portfolio_context: Dict[str, Any],
    ) -> float:
        """Compute drawdown penalty component.

        Penalizes trades that contributed to portfolio drawdown.
        """
        drawdown_contribution = portfolio_context.get("drawdown_contribution", 0.0)
        max_allowed_drawdown = portfolio_context.get("max_allowed_drawdown", 0.1)

        if drawdown_contribution <= 0:
            return 0.0  # No penalty if no drawdown contribution

        # Penalty proportional to drawdown contribution
        penalty = -1.0 * min(1.0, drawdown_contribution / max_allowed_drawdown)

        return penalty

    def _compute_accuracy_component(
        self,
        trade_record: TradeRecord,
        portfolio_context: Dict[str, Any],
    ) -> float:
        """Compute prediction accuracy bonus component.

        Rewards correct predictions, scaled by confidence.
        """
        predicted_direction = portfolio_context.get("predicted_direction")

        if not predicted_direction or trade_record.pnl is None:
            return 0.0

        # Determine actual direction
        actual_direction = "UP" if trade_record.pnl > 0 else "DOWN"
        predicted_direction = predicted_direction.upper()

        if predicted_direction == actual_direction:
            # Correct prediction: +1 scaled by confidence
            return trade_record.confidence_at_entry
        else:
            # Incorrect prediction: -0.5 scaled by confidence
            return -0.5 * trade_record.confidence_at_entry

    def _compute_regime_component(
        self, trade_record: TradeRecord
    ) -> float:
        """Compute regime-appropriate behavior bonus.

        Rewards trades where strategy matches regime expectations.
        """
        regime = trade_record.market_regime.lower()
        strategy = trade_record.strategy.lower().replace(" ", "_")

        appropriate_strategies = self.config.regime_strategy_map.get(regime, [])

        if not appropriate_strategies:
            # Unknown regime, no bonus/penalty
            return 0.0

        # Check if strategy matches any appropriate strategy
        is_appropriate = any(
            strat in strategy or strategy in strat
            for strat in appropriate_strategies
        )

        if is_appropriate:
            return 0.5
        else:
            return -0.5

    def _generate_details(
        self,
        total: float,
        risk_adjusted: float,
        drawdown: float,
        accuracy: float,
        regime: float,
        trade_record: TradeRecord,
    ) -> str:
        """Generate human-readable details string."""
        parts = [
            f"Total reward: {total:.3f}",
            f"(risk_adj: {risk_adjusted:.3f}, "
            f"drawdown: {drawdown:.3f}, "
            f"accuracy: {accuracy:.3f}, "
            f"regime: {regime:.3f})",
        ]

        if trade_record.pnl is not None:
            parts.append(f"PnL: {trade_record.pnl:.2f} ({trade_record.pnl_pct:.2f}%)")

        parts.append(f"Strategy '{trade_record.strategy}' in '{trade_record.market_regime}' regime")

        return " | ".join(parts)

    def compute_batch_rewards(
        self,
        records: List[TradeRecord],
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> List[RewardBreakdown]:
        """Compute rewards for a batch of trades.

        Parameters
        ----------
        records :
            List of trade records to evaluate.
        portfolio_context :
            Optional portfolio context shared across all trades.

        Returns
        -------
        List[RewardBreakdown]
            Reward breakdowns for each trade.
        """
        return [
            self.compute_reward(record, portfolio_context)
            for record in records
        ]

    def get_reward_statistics(
        self, rewards: List[RewardBreakdown]
    ) -> Dict[str, Any]:
        """Compute statistics over a list of rewards.

        Parameters
        ----------
        rewards :
            List of reward breakdowns.

        Returns
        -------
        Dict[str, Any]
            Statistics including mean, std, min, max, and component distributions.
        """
        if not rewards:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "components": {},
            }

        totals = np.array([r.total_reward for r in rewards])

        stats = {
            "count": len(rewards),
            "mean": float(np.mean(totals)),
            "std": float(np.std(totals)),
            "min": float(np.min(totals)),
            "max": float(np.max(totals)),
            "median": float(np.median(totals)),
        }

        # Component-wise statistics
        components: Dict[str, List[float]] = {
            "risk_adjusted": [],
            "drawdown_penalty": [],
            "accuracy_bonus": [],
            "regime_bonus": [],
        }

        for reward in rewards:
            for comp_name, comp_list in components.items():
                comp_list.append(reward.components.get(comp_name, 0.0))

        stats["components"] = {}
        for comp_name, values in components.items():
            values_array = np.array(values)
            stats["components"][comp_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
            }

        return stats

    def update_config(self, **kwargs: Any) -> None:
        """Update reward configuration parameters.

        Parameters
        ----------
        **kwargs :
            Configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as a dictionary."""
        return {
            "risk_adjusted_weight": self.config.risk_adjusted_weight,
            "drawdown_penalty_weight": self.config.drawdown_penalty_weight,
            "accuracy_bonus_weight": self.config.accuracy_bonus_weight,
            "regime_bonus_weight": self.config.regime_bonus_weight,
            "max_reward": self.config.max_reward,
        }
