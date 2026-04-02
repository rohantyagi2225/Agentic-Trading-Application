"""TraderAgent implementation with multiple strategy modules.

This module defines :class:`TraderAgent`, a concrete :class:`ResearchAgent`
that combines momentum, mean-reversion, and ML-inspired strategies to
produce trading signals and position sizing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .base_agent import (
    Action,
    ActionType,
    Explanation,
    LearningUpdate,
    MarketData,
    MarketContext,
    ReasoningResult,
    ResearchAgent,
)


def _extract_closes(prices: List[Mapping[str, Any]]) -> np.ndarray:
    """Extract closing prices from a list of OHLCV-like dictionaries."""

    if not prices:
        return np.array([], dtype=float)
    return np.array([float(bar.get("close", np.nan)) for bar in prices], dtype=float)


@dataclass
class MomentumStrategy:
    """Simple momentum strategy based on rate of change and moving averages."""

    lookback_period: int = 20
    fast_ma: int = 10
    slow_ma: int = 50

    def generate_signal(self, prices: np.ndarray) -> float:
        """Generate a momentum signal in the range ``[-1, 1]``.

        The signal combines normalized rate of change with a moving average
        crossover indicator.
        """

        if prices.size < max(self.lookback_period, self.slow_ma) + 1:
            return 0.0

        recent = prices[-self.lookback_period :]
        roc = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0.0

        fast = prices[-self.fast_ma :].mean()
        slow = prices[-self.slow_ma :].mean()
        ma_diff = fast - slow

        roc_norm = np.tanh(roc)
        ma_norm = np.tanh(ma_diff / slow) if slow != 0 else 0.0

        signal = 0.6 * roc_norm + 0.4 * ma_norm
        return float(np.clip(signal, -1.0, 1.0))


@dataclass
class MeanReversionStrategy:
    """Mean-reversion strategy using z-score of price vs. rolling mean."""

    window: int = 20
    entry_threshold: float = 1.0
    exit_threshold: float = 0.2

    def generate_signal(self, prices: np.ndarray) -> float:
        """Generate a mean-reversion signal in the range ``[-1, 1]``.

        Positive values indicate expectation of price reverting upward,
        negative values indicate expectation of reversion downward.
        """

        if prices.size < self.window:
            return 0.0

        window_prices = prices[-self.window :]
        mean = window_prices.mean()
        std = window_prices.std(ddof=1) if self.window > 1 else 0.0

        if std == 0:
            return 0.0

        z = (window_prices[-1] - mean) / std

        if abs(z) < self.exit_threshold:
            raw = 0.0
        elif z > 0:
            raw = -min((abs(z) - self.exit_threshold) / (self.entry_threshold + 1e-6), 1.0)
        else:
            raw = min((abs(z) - self.exit_threshold) / (self.entry_threshold + 1e-6), 1.0)

        return float(np.clip(raw, -1.0, 1.0))


@dataclass
class MLBasedStrategy:
    """Simulated ML-based strategy using weighted feature combination."""

    feature_weights: Dict[str, float]

    def generate_signal(
        self,
        momentum: float,
        volatility: float,
        volume_trend: float,
        mean_reversion_score: float,
    ) -> float:
        """Generate an ML-inspired signal from engineered features."""

        features = {
            "momentum": momentum,
            "volatility": volatility,
            "volume_trend": volume_trend,
            "mean_reversion_score": mean_reversion_score,
        }

        score = 0.0
        for name, value in features.items():
            weight = self.feature_weights.get(name, 0.0)
            score += weight * value

        return float(np.clip(np.tanh(score), -1.0, 1.0))


class TraderAgent(ResearchAgent):
    """Domain-specialized trading agent combining multiple strategies."""

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(agent_id, config)

        cfg = config or {}

        self.momentum_strategy = MomentumStrategy(
            lookback_period=cfg.get("momentum_lookback", 20),
            fast_ma=cfg.get("momentum_fast_ma", 10),
            slow_ma=cfg.get("momentum_slow_ma", 50),
        )

        self.mean_reversion_strategy = MeanReversionStrategy(
            window=cfg.get("mr_window", 20),
            entry_threshold=cfg.get("mr_entry_threshold", 1.0),
            exit_threshold=cfg.get("mr_exit_threshold", 0.2),
        )

        default_feature_weights = {
            "momentum": 0.5,
            "volatility": -0.2,
            "volume_trend": 0.3,
            "mean_reversion_score": 0.4,
        }
        self.ml_strategy = MLBasedStrategy(
            feature_weights=cfg.get("ml_feature_weights", default_feature_weights),
        )

        self.strategy_weights: Dict[str, float] = cfg.get(
            "strategy_weights",
            {"momentum": 0.4, "mean_reversion": 0.3, "ml": 0.3},
        )

        self.risk_aversion: float = cfg.get("risk_aversion", 0.5)
        self._performance_history: List[Dict[str, float]] = []

    def _compute_features(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute common features for the ML strategy."""

        if prices.size < 2:
            return {"momentum": 0.0, "volatility": 0.0, "volume_trend": 0.0, "mean_reversion_score": 0.0}

        momentum_raw = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
        momentum = float(np.tanh(momentum_raw))

        volatility = float(np.tanh(prices.std() / (prices.mean() + 1e-6)))

        if volumes is not None and volumes.size > 1:
            volume_trend_raw = (volumes[-1] - volumes[0]) / (volumes[0] + 1e-6)
            volume_trend = float(np.tanh(volume_trend_raw))
        else:
            volume_trend = 0.0

        if prices.size > 5:
            mr_window = min(20, prices.size)
            window_prices = prices[-mr_window :]
            mean = window_prices.mean()
            std = window_prices.std(ddof=1) if mr_window > 1 else 0.0
            if std > 0:
                z = (window_prices[-1] - mean) / std
                mean_reversion_score = float(np.tanh(-z))
            else:
                mean_reversion_score = 0.0
        else:
            mean_reversion_score = 0.0

        return {
            "momentum": momentum,
            "volatility": volatility,
            "volume_trend": volume_trend,
            "mean_reversion_score": mean_reversion_score,
        }

    def reason(self, market_data: MarketData, context: Optional[MarketContext] = None) -> ReasoningResult:
        prices_arr = _extract_closes(market_data.prices)

        volumes_arr: Optional[np.ndarray]
        if market_data.prices:
            volumes_arr = np.array(
                [float(bar.get("volume", market_data.volume)) for bar in market_data.prices],
                dtype=float,
            )
        else:
            volumes_arr = None

        momentum_signal = self.momentum_strategy.generate_signal(prices_arr)
        mr_signal = self.mean_reversion_strategy.generate_signal(prices_arr)
        features = self._compute_features(prices_arr, volumes_arr)
        ml_signal = self.ml_strategy.generate_signal(
            momentum=features["momentum"],
            volatility=features["volatility"],
            volume_trend=features["volume_trend"],
            mean_reversion_score=features["mean_reversion_score"],
        )

        w_mom = self.strategy_weights.get("momentum", 0.0)
        w_mr = self.strategy_weights.get("mean_reversion", 0.0)
        w_ml = self.strategy_weights.get("ml", 0.0)
        weight_sum = w_mom + w_mr + w_ml
        if weight_sum <= 0:
            weights = (1 / 3, 1 / 3, 1 / 3)
        else:
            weights = (w_mom / weight_sum, w_mr / weight_sum, w_ml / weight_sum)

        combined_signal = (
            weights[0] * momentum_signal
            + weights[1] * mr_signal
            + weights[2] * ml_signal
        )

        confidence = float(min(1.0, abs(combined_signal) * 1.2))

        reasoning_chain = [
            f"Momentum signal: {momentum_signal:.3f}",
            f"Mean-reversion signal: {mr_signal:.3f}",
            f"ML signal: {ml_signal:.3f}",
            f"Strategy weights: momentum={weights[0]:.2f}, mean_reversion={weights[1]:.2f}, ml={weights[2]:.2f}",
            f"Combined signal: {combined_signal:.3f}",
        ]

        if context is not None and context.volatility_level is not None:
            reasoning_chain.append(f"Context volatility level: {context.volatility_level}")

        observations = [
            f"Prices length={prices_arr.size}",
            f"Latest price={prices_arr[-1] if prices_arr.size else 'n/a'}",
        ]

        inferences = []
        if combined_signal > 0:
            inferences.append("Bullish bias from aggregated strategies.")
        elif combined_signal < 0:
            inferences.append("Bearish bias from aggregated strategies.")
        else:
            inferences.append("Neutral signal; no clear directional edge.")

        signals = {
            "momentum": momentum_signal,
            "mean_reversion": mr_signal,
            "ml": ml_signal,
            "combined": combined_signal,
            "features": features,
        }

        return ReasoningResult(
            observations=observations,
            inferences=inferences,
            confidence=confidence,
            signals=signals,
            reasoning_chain=reasoning_chain,
        )

    def _kelly_position_size(self, edge: float, odds: float = 1.0) -> float:
        """Approximate position size using a Kelly-criterion-like formula."""

        if odds <= 0:
            return 0.0
        kelly_fraction = edge / odds if edge > 0 else 0.0
        kelly_fraction = max(min(kelly_fraction, 1.0), 0.0)
        adjusted = kelly_fraction * (1.0 - self.risk_aversion)
        return float(adjusted)

    def act(self, reasoning_result: ReasoningResult) -> Action:
        combined_signal = float(reasoning_result.signals.get("combined", 0.0))
        confidence = reasoning_result.confidence

        if combined_signal > 0:
            action_type = ActionType.BUY
        elif combined_signal < 0:
            action_type = ActionType.SELL
        else:
            action_type = ActionType.HOLD

        edge = abs(combined_signal) * confidence
        position_fraction = self._kelly_position_size(edge)

        latest_price = None
        market_data_price = reasoning_result.signals.get("latest_price")
        if isinstance(market_data_price, (int, float)):
            latest_price = float(market_data_price)

        reasoning_summary = reasoning_result.inferences[0] if reasoning_result.inferences else "No strong inference."

        return Action(
            action_type=action_type,
            symbol=reasoning_result.signals.get("symbol"),
            quantity=position_fraction,
            price=latest_price,
            confidence=confidence,
            stop_loss=None,
            take_profit=None,
            reasoning_summary=reasoning_summary,
        )

    def learn(self, outcome: Mapping[str, Any]) -> LearningUpdate:
        """Update strategy weights based on recent performance.

        The ``outcome`` mapping may include per-strategy performance metrics,
        e.g.::

            {
                "momentum_pnl": float,
                "mean_reversion_pnl": float,
                "ml_pnl": float,
            }
        """

        momentum_pnl = float(outcome.get("momentum_pnl", 0.0))
        mr_pnl = float(outcome.get("mean_reversion_pnl", 0.0))
        ml_pnl = float(outcome.get("ml_pnl", 0.0))

        self._performance_history.append(
            {"momentum": momentum_pnl, "mean_reversion": mr_pnl, "ml": ml_pnl}
        )

        total_pnl = momentum_pnl + mr_pnl + ml_pnl
        parameter_changes: Dict[str, Any] = {}
        lessons: List[str] = []

        if total_pnl != 0:
            mom_adj = momentum_pnl / (abs(total_pnl) + 1e-6)
            mr_adj = mr_pnl / (abs(total_pnl) + 1e-6)
            ml_adj = ml_pnl / (abs(total_pnl) + 1e-6)

            self.strategy_weights["momentum"] = float(
                np.clip(self.strategy_weights.get("momentum", 0.3) + 0.1 * mom_adj, 0.0, 1.0)
            )
            self.strategy_weights["mean_reversion"] = float(
                np.clip(self.strategy_weights.get("mean_reversion", 0.3) + 0.1 * mr_adj, 0.0, 1.0)
            )
            self.strategy_weights["ml"] = float(
                np.clip(self.strategy_weights.get("ml", 0.4) + 0.1 * ml_adj, 0.0, 1.0)
            )

            parameter_changes["strategy_weights"] = dict(self.strategy_weights)
            lessons.append("Adjusted strategy weights based on relative PnL contribution.")

        confidence_adjustment = float(np.tanh(total_pnl))
        if total_pnl > 0:
            lessons.append("Overall profitable period; increasing confidence slightly.")
        elif total_pnl < 0:
            lessons.append("Overall loss; reducing risk appetite and confidence.")

        parameter_changes["risk_aversion"] = float(
            np.clip(self.risk_aversion + (-0.05 if total_pnl < 0 else 0.02), 0.0, 1.0)
        )
        self.risk_aversion = parameter_changes["risk_aversion"]

        return LearningUpdate(
            parameter_changes=parameter_changes,
            confidence_adjustment=confidence_adjustment,
            lessons=lessons,
        )

    def explain(self, action: Action) -> Explanation:
        summary = f"TraderAgent decided to {action.action_type} with size {action.quantity:.3f} at confidence {action.confidence:.2f}."

        reasoning_chain = [
            f"Risk aversion level: {self.risk_aversion:.2f}",
            f"Current strategy weights: {self.strategy_weights}",
            "Position size derived from Kelly-style fraction adjusted for risk aversion.",
        ]

        data_sources = [
            "Historical close prices",
            "Volume series",
            "Derived features: momentum, volatility, volume trend, mean-reversion score",
        ]

        confidence_breakdown = {
            "signal_strength": float(abs(action.quantity)),
            "risk_aversion": float(1.0 - self.risk_aversion),
        }

        risk_justification = (
            "Position sizing is capped via risk_aversion and implicit Kelly fraction "
            "to avoid over-leveraging even when signals are strong."
        )

        return Explanation(
            summary=summary,
            reasoning_chain=reasoning_chain,
            data_sources=data_sources,
            risk_justification=risk_justification,
            confidence_breakdown=confidence_breakdown,
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "config": self.config,
            "strategy_weights": dict(self.strategy_weights),
            "risk_aversion": self.risk_aversion,
            "performance_history": list(self._performance_history),
        }
