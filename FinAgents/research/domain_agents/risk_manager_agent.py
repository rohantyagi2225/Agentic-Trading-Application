"""RiskManagerAgent implementation for advanced risk analytics.

This module defines :class:`RiskManagerAgent`, a concrete
:class:`ResearchAgent` focused on portfolio risk evaluation and
risk-aware decision support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import norm

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
class RiskLimits:
    """Configuration of risk limits for the risk manager."""

    max_var: float = 0.05
    max_position_pct: float = 0.1
    max_drawdown: float = 0.2
    max_correlation: float = 0.8


class RiskManagerAgent(ResearchAgent):
    """Risk manager agent performing VaR, CVaR, and stress testing."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(agent_id, config)
        cfg = config or {}
        self.limits = RiskLimits(
            max_var=cfg.get("max_var", 0.05),
            max_position_pct=cfg.get("max_position_pct", 0.1),
            max_drawdown=cfg.get("max_drawdown", 0.2),
            max_correlation=cfg.get("max_correlation", 0.8),
        )

    @staticmethod
    def calculate_var(
        returns: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """Calculate Value-at-Risk (VaR) for a series of returns.

        Parameters
        ----------
        returns:
            Array of portfolio returns.
        confidence:
            Confidence level, typically 0.95 or 0.99.
        method:
            Either ``"historical"`` or ``"parametric"``.
        """

        if returns.size == 0:
            return 0.0

        alpha = 1.0 - confidence

        if method == "historical":
            var = -np.percentile(returns, 100 * alpha)
        elif method == "parametric":
            mu = returns.mean()
            sigma = returns.std(ddof=1)
            z = norm.ppf(alpha)
            var = -(mu + z * sigma)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

        return float(max(var, 0.0))

    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value-at-Risk (CVaR / Expected Shortfall)."""

        if returns.size == 0:
            return 0.0

        alpha = 1.0 - confidence
        threshold = np.percentile(returns, 100 * alpha)
        tail_losses = returns[returns <= threshold]

        if tail_losses.size == 0:
            return 0.0

        cvar = -tail_losses.mean()
        return float(max(cvar, 0.0))

    @staticmethod
    def monte_carlo_stress_test(
        portfolio: Mapping[str, float],
        num_simulations: int = 10000,
        horizon_days: int = 10,
        mean_return: float = 0.0,
        vol: float = 0.2,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Perform Monte Carlo stress test using geometric Brownian motion.

        This simulation is intentionally simple and is meant as a research-grade
        scaffold rather than a production pricing engine.
        """

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        weights = np.array(list(portfolio.values()), dtype=float)
        if weights.size == 0:
            return {"expected_pnl": 0.0, "worst_pnl": 0.0, "var_95": 0.0}

        dt = 1.0 / 252.0
        drift = (mean_return - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt)

        pnl_scenarios = []
        for _ in range(num_simulations):
            shocks = rng.normal(loc=drift, scale=diffusion, size=horizon_days)
            cumulative_return = float(np.exp(shocks.sum()) - 1.0)
            pnl = float((weights.sum()) * cumulative_return)
            pnl_scenarios.append(pnl)

        pnl_arr = np.array(pnl_scenarios, dtype=float)
        expected_pnl = float(pnl_arr.mean())
        worst_pnl = float(pnl_arr.min())
        var_95 = float(-np.percentile(pnl_arr, 5))

        return {
            "expected_pnl": expected_pnl,
            "worst_pnl": worst_pnl,
            "var_95": max(var_95, 0.0),
        }

    @staticmethod
    def correlation_risk_decomposition(
        positions: Mapping[str, float],
        returns_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """Decompose portfolio risk by correlation structure.

        Parameters
        ----------
        positions:
            Mapping from symbol to weight.
        returns_matrix:
            2D array of shape (T, N) with asset returns.
        """

        symbols = list(positions.keys())
        weights = np.array(list(positions.values()), dtype=float)

        if returns_matrix.size == 0 or len(symbols) == 0:
            return {
                "symbols": symbols,
                "weights": weights.tolist(),
                "correlation_matrix": None,
                "concentration_flags": {},
            }

        corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
        concentration_flags: Dict[Tuple[str, str], bool] = {}

        n_assets = len(symbols)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                rho = corr_matrix[i, j]
                key = (symbols[i], symbols[j])
                concentration_flags[key] = bool(abs(rho) > 0.8)

        return {
            "symbols": symbols,
            "weights": weights.tolist(),
            "correlation_matrix": corr_matrix,
            "concentration_flags": concentration_flags,
        }

    def reason(self, market_data: MarketData, context: Optional[MarketContext] = None) -> ReasoningResult:
        """Evaluate current portfolio risk using multiple methods.

        The ``market_data`` object is used primarily as a carrier for recent
        returns and position information via its ``prices`` and ``regime``
        fields; concrete integrations can extend this pattern as needed.
        """

        prices = [bar.get("close") for bar in market_data.prices if "close" in bar]
        returns = np.diff(np.log(np.asarray(prices, dtype=float))) if len(prices) > 1 else np.array([], dtype=float)

        var_hist = self.calculate_var(returns, confidence=0.95, method="historical")
        var_param = self.calculate_var(returns, confidence=0.95, method="parametric")
        cvar = self.calculate_cvar(returns, confidence=0.95)

        mc_result = self.monte_carlo_stress_test({market_data.symbol: 1.0}, num_simulations=2000, horizon_days=10)

        observations = [
            f"Computed historical VaR (95%): {var_hist:.4f}",
            f"Computed parametric VaR (95%): {var_param:.4f}",
            f"Computed CVaR (95%): {cvar:.4f}",
            f"Monte Carlo expected PnL (10d): {mc_result['expected_pnl']:.4f}",
            f"Monte Carlo worst PnL (10d): {mc_result['worst_pnl']:.4f}",
        ]

        inferences: List[str] = []
        if var_hist > self.limits.max_var:
            inferences.append("Historical VaR exceeds configured limit.")
        if cvar > self.limits.max_var * 1.5:
            inferences.append("Tail risk (CVaR) is significantly elevated.")
        if not inferences:
            inferences.append("Risk metrics are within configured limits.")

        risk_score = min(1.0, (var_hist / (self.limits.max_var + 1e-6)))
        confidence = float(np.clip(1.0 - risk_score, 0.0, 1.0))

        reasoning_chain = observations + [
            f"Risk limits: max_var={self.limits.max_var}, max_drawdown={self.limits.max_drawdown}",
            f"Derived risk_score={risk_score:.3f}, confidence={confidence:.3f}",
        ]

        signals = {
            "var_hist": var_hist,
            "var_param": var_param,
            "cvar": cvar,
            "mc": mc_result,
            "risk_score": risk_score,
        }

        return ReasoningResult(
            observations=observations,
            inferences=inferences,
            confidence=confidence,
            signals=signals,
            reasoning_chain=reasoning_chain,
        )

    def act(self, reasoning_result: ReasoningResult) -> Action:
        risk_score = float(reasoning_result.signals.get("risk_score", 0.0))

        if risk_score > 1.2:
            decision = ActionType.SELL
            summary = "Risk excessive; recommend de-risking and reducing positions."
        elif risk_score > 1.0:
            decision = ActionType.HOLD
            summary = "Risk at or slightly above limits; hold and avoid new risk."
        else:
            decision = ActionType.BUY
            summary = "Risk within limits; approve incremental risk-taking within bounds."

        quantity = float(np.clip(1.0 - risk_score, 0.0, 1.0))

        return Action(
            action_type=decision,
            symbol=None,
            quantity=quantity,
            price=None,
            confidence=reasoning_result.confidence,
            stop_loss=None,
            take_profit=None,
            reasoning_summary=summary,
        )

    def learn(self, outcome: Mapping[str, Any]) -> LearningUpdate:
        """Update risk limits based on realized drawdowns and breaches."""

        drawdown = float(outcome.get("max_drawdown", 0.0))
        breaches = int(outcome.get("var_breaches", 0))

        parameter_changes: Dict[str, Any] = {}
        lessons: List[str] = []

        if breaches > 0:
            old_max_var = self.limits.max_var
            self.limits.max_var = float(max(self.limits.max_var * 0.9, 0.01))
            parameter_changes["max_var"] = self.limits.max_var
            lessons.append(
                f"Observed {breaches} VaR breaches; tightened max_var from {old_max_var:.3f} to {self.limits.max_var:.3f}."
            )

        if drawdown > self.limits.max_drawdown:
            old_dd = self.limits.max_drawdown
            self.limits.max_drawdown = float(max(self.limits.max_drawdown * 0.9, 0.05))
            parameter_changes["max_drawdown"] = self.limits.max_drawdown
            lessons.append(
                f"Observed drawdown {drawdown:.3f} exceeds limit; tightened max_drawdown from {old_dd:.3f} to {self.limits.max_drawdown:.3f}."
            )

        confidence_adjustment = float(-np.tanh(max(breaches, 0) + max(drawdown - self.limits.max_drawdown, 0)))

        return LearningUpdate(
            parameter_changes=parameter_changes,
            confidence_adjustment=confidence_adjustment,
            lessons=lessons,
        )

    def explain(self, action: Action) -> Explanation:
        summary = (
            "RiskManagerAgent "
            f"recommends action={action.action_type} with scale={action.quantity:.2f} "
            f"under confidence={action.confidence:.2f}."
        )

        reasoning_chain = [
            f"Current limits: max_var={self.limits.max_var:.3f}, max_drawdown={self.limits.max_drawdown:.3f}",
            "Action reflects trade-off between current risk metrics and configured limits.",
        ]

        data_sources = [
            "Historical portfolio returns",
            "Monte Carlo simulated scenarios",
        ]

        confidence_breakdown = {
            "limit_headroom": float(action.quantity),
        }

        risk_justification = (
            "Recommendations are conservative when risk metrics approach or exceed limits, "
            "and only allow incremental risk when sufficient headroom exists."
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
            "limits": {
                "max_var": self.limits.max_var,
                "max_position_pct": self.limits.max_position_pct,
                "max_drawdown": self.limits.max_drawdown,
                "max_correlation": self.limits.max_correlation,
            },
        }
