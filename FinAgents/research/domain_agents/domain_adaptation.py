"""Domain adaptation utilities for regime-aware research agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .base_agent import LearningUpdate, ResearchAgent


@dataclass
class RegimeProfile:
    """Profile capturing regime-specific optimal parameters and performance."""

    regime_type: str
    optimal_parameters: Dict[str, Any] = field(default_factory=dict)
    historical_performance: float = 0.0
    sample_count: int = 0


class DomainAdapter:
    """Simulated fine-tuning and regime adaptation for research agents."""

    def __init__(self) -> None:
        self.regime_registry: Dict[str, RegimeProfile] = {}

    def adapt_to_regime(
        self,
        agent: ResearchAgent,
        regime: str,
        performance_history: Mapping[str, float],
    ) -> LearningUpdate:
        """Adjust agent parameters based on regime-specific performance.

        The ``performance_history`` is a mapping from metric names to values,
        e.g., Sharpe ratios or average PnL for the given regime.
        """

        regime_profile = self.regime_registry.get(regime) or RegimeProfile(regime_type=regime)

        sharpe = float(performance_history.get("sharpe", 0.0))
        pnl = float(performance_history.get("pnl", 0.0))

        prev_perf = regime_profile.historical_performance
        prev_count = regime_profile.sample_count
        new_count = prev_count + 1
        new_perf = (prev_perf * prev_count + sharpe) / new_count if new_count > 0 else sharpe

        regime_profile.historical_performance = new_perf
        regime_profile.sample_count = new_count

        parameter_changes: Dict[str, Any] = {}
        lessons: List[str] = []

        if hasattr(agent, "risk_aversion"):
            current_ra = float(getattr(agent, "risk_aversion"))
            if "volatility" in performance_history and performance_history["volatility"] > 0.3:
                new_ra = float(np.clip(current_ra + 0.1, 0.0, 1.0))
                lessons.append(
                    f"High volatility regime detected; increasing risk_aversion from {current_ra:.2f} to {new_ra:.2f}."
                )
            else:
                new_ra = float(np.clip(current_ra - 0.05, 0.0, 1.0))
                lessons.append(
                    f"Benign volatility regime; modestly decreasing risk_aversion from {current_ra:.2f} to {new_ra:.2f}."
                )
            setattr(agent, "risk_aversion", new_ra)
            parameter_changes["risk_aversion"] = new_ra

        regime_profile.optimal_parameters.update(parameter_changes)
        self.regime_registry[regime] = regime_profile

        confidence_adjustment = float(np.tanh(sharpe))
        lessons.append(f"Updated regime profile for {regime} with Sharpe={sharpe:.3f} and PnL={pnl:.3f}.")

        return LearningUpdate(
            parameter_changes=parameter_changes,
            confidence_adjustment=confidence_adjustment,
            lessons=lessons,
        )

    def simulate_fine_tuning(
        self,
        agent: ResearchAgent,
        training_data: List[Mapping[str, Any]],
        epochs: int = 10,
    ) -> LearningUpdate:
        """Simulate fine-tuning of an agent on historical training data.

        This routine iterates over the training dataset and applies simple
        heuristics to adjust agent parameters, returning an aggregated
        :class:`LearningUpdate` that summarizes the overall adaptation.
        """

        cumulative_changes: Dict[str, Any] = {}
        lessons: List[str] = []
        confidence_adjustments: List[float] = []

        for epoch in range(epochs):
            epoch_pnl = 0.0
            for sample in training_data:
                pnl = float(sample.get("pnl", 0.0))
                epoch_pnl += pnl

            avg_pnl = epoch_pnl / max(len(training_data), 1)
            conf_adj = float(np.tanh(avg_pnl))
            confidence_adjustments.append(conf_adj)

            if hasattr(agent, "strategy_weights") and isinstance(getattr(agent, "strategy_weights"), dict):
                weights = getattr(agent, "strategy_weights")
                for name in list(weights.keys()):
                    weights[name] = float(np.clip(weights[name] + 0.01 * np.sign(avg_pnl), 0.0, 1.0))
                setattr(agent, "strategy_weights", weights)
                cumulative_changes["strategy_weights"] = dict(weights)
                lessons.append(
                    f"Epoch {epoch}: adjusted strategy_weights based on avg_pnl={avg_pnl:.4f}."
                )

        confidence_adjustment = float(np.mean(confidence_adjustments)) if confidence_adjustments else 0.0

        return LearningUpdate(
            parameter_changes=cumulative_changes,
            confidence_adjustment=confidence_adjustment,
            lessons=lessons,
        )
