"""PortfolioManagerAgent implementation for portfolio optimization and control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

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


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio optimization."""

    max_weight: float = 0.2
    min_weight: float = 0.0
    target_volatility: Optional[float] = None


class PortfolioManagerAgent(ResearchAgent):
    """Portfolio manager agent handling optimization and factor exposure."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(agent_id, config)
        cfg = config or {}
        self.constraints = PortfolioConstraints(
            max_weight=cfg.get("max_weight", 0.2),
            min_weight=cfg.get("min_weight", 0.0),
            target_volatility=cfg.get("target_volatility"),
        )

    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None,
    ) -> np.ndarray:
        """Optimize portfolio weights given expected returns and covariance.

        Objective combines maximizing return and minimizing variance. A simple
        transaction cost proxy penalizes large weights.
        """

        n = expected_returns.shape[0]
        if constraints is None:
            constraints = self.constraints

        def objective(w: np.ndarray) -> float:
            ret = float(w @ expected_returns)
            var = float(w @ cov_matrix @ w)
            cost = float(np.abs(w).sum()) * 0.001
            return -(ret - 0.5 * var - cost)

        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

        x0 = np.full(n, 1.0 / n, dtype=float)
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)

        if not result.success:
            return x0

        return result.x

    @staticmethod
    def calculate_rebalance(
        current_weights: Sequence[float],
        target_weights: Sequence[float],
        threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Generate rebalance trades when weight drift exceeds a threshold."""

        current = np.asarray(current_weights, dtype=float)
        target = np.asarray(target_weights, dtype=float)

        trades: List[Dict[str, Any]] = []
        for i, (cw, tw) in enumerate(zip(current, target)):
            diff = tw - cw
            if abs(diff) > threshold:
                trades.append({"asset_index": i, "delta_weight": float(diff)})

        return trades

    @staticmethod
    def manage_factor_exposure(
        positions: Mapping[str, float],
        factor_loadings: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, Any]:
        """Track and summarize factor exposures for a portfolio."""

        exposures: Dict[str, float] = {}
        for symbol, weight in positions.items():
            loads = factor_loadings.get(symbol, {})
            for factor, loading in loads.items():
                exposures[factor] = exposures.get(factor, 0.0) + weight * loading

        return {"factor_exposures": exposures}

    def reason(self, market_data: MarketData, context: Optional[MarketContext] = None) -> ReasoningResult:
        prices = np.array([bar.get("close", np.nan) for bar in market_data.prices], dtype=float)
        if prices.size < 2:
            expected_returns = np.array([0.0])
            cov_matrix = np.array([[0.0]])
        else:
            returns = np.diff(np.log(prices))
            expected_returns = np.array([returns.mean()])
            cov_matrix = np.array([[returns.var(ddof=1)]])

        target_weights = self.optimize_portfolio(expected_returns, cov_matrix)
        current_weights = np.array([1.0])

        trades = self.calculate_rebalance(current_weights, target_weights)

        exposures = {"market": float(target_weights.sum())}

        observations = [
            f"Estimated expected return={expected_returns.mean():.4f}",
            f"Estimated variance={cov_matrix.mean():.4f}",
            f"Target weights={target_weights}",
            f"Planned trades={trades}",
        ]

        inferences: List[str] = []
        if trades:
            inferences.append("Rebalancing required to align with optimized target weights.")
        else:
            inferences.append("Portfolio close to optimal; minimal rebalancing required.")

        confidence = 0.8 if trades else 0.6

        signals: Dict[str, Any] = {
            "target_weights": target_weights,
            "current_weights": current_weights,
            "trades": trades,
            "factor_exposures": exposures,
        }

        reasoning_chain = observations + [
            f"Confidence in optimization output={confidence:.2f}",
        ]

        return ReasoningResult(
            observations=observations,
            inferences=inferences,
            confidence=confidence,
            signals=signals,
            reasoning_chain=reasoning_chain,
        )

    def act(self, reasoning_result: ReasoningResult) -> Action:
        trades = reasoning_result.signals.get("trades", [])

        if trades:
            action_type = ActionType.BUY
            summary = f"Execute {len(trades)} rebalance trades to move toward target weights."
        else:
            action_type = ActionType.HOLD
            summary = "No significant drift; hold current portfolio weights."

        quantity = float(min(len(trades) / 10.0, 1.0)) if trades else 0.0

        return Action(
            action_type=action_type,
            symbol=None,
            quantity=quantity,
            price=None,
            confidence=reasoning_result.confidence,
            stop_loss=None,
            take_profit=None,
            reasoning_summary=summary,
        )

    def learn(self, outcome: Mapping[str, Any]) -> LearningUpdate:
        tracking_error = float(outcome.get("tracking_error", 0.0))

        lessons: List[str] = []
        parameter_changes: Dict[str, Any] = {}

        if tracking_error > 0.05:
            old_max_weight = self.constraints.max_weight
            self.constraints.max_weight = float(max(self.constraints.max_weight * 0.9, 0.05))
            parameter_changes["max_weight"] = self.constraints.max_weight
            lessons.append(
                f"High tracking error {tracking_error:.3f} observed; tightened max_weight from {old_max_weight:.2f} to {self.constraints.max_weight:.2f}."
            )

        confidence_adjustment = float(-np.tanh(tracking_error * 10))

        return LearningUpdate(
            parameter_changes=parameter_changes,
            confidence_adjustment=confidence_adjustment,
            lessons=lessons,
        )

    def explain(self, action: Action) -> Explanation:
        summary = (
            "PortfolioManagerAgent used mean-variance optimization and drift thresholds "
            f"to recommend action={action.action_type} with intensity={action.quantity:.2f}."
        )

        reasoning_chain = [
            "Objective balances expected return, variance, and a simple cost proxy.",
            "Trades are triggered only when weight deviations exceed a configurable threshold.",
        ]

        data_sources = [
            "Historical asset returns",
            "Estimated covariance matrix",
        ]

        confidence_breakdown = {"num_trades": float(action.quantity)}

        risk_justification = (
            "Rebalancing keeps portfolio aligned with desired risk/return trade-off "
            "and prevents unintended concentration through drift."
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
            "constraints": {
                "max_weight": self.constraints.max_weight,
                "min_weight": self.constraints.min_weight,
                "target_volatility": self.constraints.target_volatility,
            },
        }
