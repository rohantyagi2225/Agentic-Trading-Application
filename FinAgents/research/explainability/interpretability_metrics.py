"""Quantitative interpretability metrics for explainable agents.

This module implements simple, model-agnostic metrics that assess the
completeness, consistency, and faithfulness of explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from FinAgents.research.domain_agents.base_agent import Action

from .reasoning_chain import ReasoningChain


@dataclass
class CompletenessScore:
    """Score describing how complete a reasoning chain is."""

    score: float
    missing_factors: List[str] = field(default_factory=list)
    covered_factors: List[str] = field(default_factory=list)
    details: str = ""


@dataclass
class ConsistencyScore:
    """Score describing consistency across similar decisions."""

    score: float
    inconsistent_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = field(default_factory=list)
    details: str = ""


@dataclass
class FaithfulnessScore:
    """Score describing alignment between explanations and actual factors."""

    score: float
    aligned_factors: List[str] = field(default_factory=list)
    misaligned_factors: List[str] = field(default_factory=list)
    details: str = ""


@dataclass
class InterpretabilityReport:
    """Aggregated interpretability metrics."""

    overall_score: float
    completeness: CompletenessScore
    consistency: Optional[ConsistencyScore]
    faithfulness: Optional[FaithfulnessScore]
    grade: str
    recommendations: List[str] = field(default_factory=list)


class InterpretabilityScorer:
    """Compute quantitative interpretability metrics for decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------
    def score_completeness(
        self, reasoning_chain: ReasoningChain, action: Action
    ) -> CompletenessScore:
        """Evaluate whether key reasoning factors are covered.

        Required factors
        ----------------
        - market_data_observation
        - signal_analysis
        - risk_assessment
        - position_sizing_rationale

        Optional factors
        ----------------
        - regime_context
        - sentiment_analysis
        - alternative_scenarios
        """

        steps = reasoning_chain.get_chain()
        text = " ".join(
            [
                getattr(s, "observation", "") + " " + getattr(s, "inference", "")
                for s in steps
            ]
        ).lower()

        required = [
            "market_data_observation",
            "signal_analysis",
            "risk_assessment",
            "position_sizing_rationale",
        ]
        optional = [
            "regime_context",
            "sentiment_analysis",
            "alternative_scenarios",
        ]

        covered_required: List[str] = []
        covered_optional: List[str] = []

        # Very simple keyword-based heuristic for now.
        for factor in required:
            token = factor.split("_")[0]
            if token in text:
                covered_required.append(factor)
        for factor in optional:
            token = factor.split("_")[0]
            if token in text:
                covered_optional.append(factor)

        missing_required = [f for f in required if f not in covered_required]

        required_score = (
            len(covered_required) / len(required) if required else 0.0
        )
        optional_score = (
            len(covered_optional) / len(optional) if optional else 0.0
        )
        score = required_score * 0.7 + optional_score * 0.3

        details_lines = [
            f"Required factors covered: {covered_required}",
            f"Optional factors covered: {covered_optional}",
            f"Missing required factors: {missing_required}",
        ]

        return CompletenessScore(
            score=score,
            missing_factors=missing_required,
            covered_factors=covered_required + covered_optional,
            details="\n".join(details_lines),
        )

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------
    def score_consistency(
        self, decisions: List[Tuple[Dict[str, Any], Action]]
    ) -> ConsistencyScore:
        """Assess whether similar inputs lead to similar actions.

        The input is a list of ``(market_conditions, action)`` tuples.
        This implementation groups decisions by coarse market condition
        bins and computes the variance of action quantities within each
        group. Lower variance implies higher consistency.
        """

        if not decisions:
            return ConsistencyScore(score=0.0, inconsistent_pairs=[], details="No decisions provided.")

        # Group by regime and rounded volatility if present.
        groups: Dict[str, List[Action]] = {}
        for conditions, action in decisions:
            regime = conditions.get("regime", "unknown")
            vol = conditions.get("volatility", 0.0)
            try:
                vol_bucket = round(float(vol), 1)
            except (TypeError, ValueError):
                vol_bucket = 0.0
            key = f"{regime}|{vol_bucket}"
            groups.setdefault(key, []).append(action)

        import statistics

        variances: List[float] = []
        inconsistent_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        for key, actions in groups.items():
            if len(actions) < 2:
                continue
            quantities = [a.quantity for a in actions]
            try:
                var = statistics.pvariance(quantities)
            except statistics.StatisticsError:
                var = 0.0
            variances.append(var)

            # Identify a simple inconsistent pair: max vs min quantity.
            max_a = max(actions, key=lambda a: a.quantity)
            min_a = min(actions, key=lambda a: a.quantity)
            if max_a.quantity != min_a.quantity:
                inconsistent_pairs.append((asdict_action(min_a), asdict_action(max_a)))

        if not variances:
            return ConsistencyScore(score=1.0, inconsistent_pairs=[], details="All groups have a single decision.")

        max_var = max(variances) or 1.0
        avg_var = sum(variances) / len(variances)
        normalized_variance = avg_var / max_var
        score = max(0.0, 1.0 - normalized_variance)

        details = (
            f"Average variance across groups: {avg_var:.4f}, "
            f"max variance: {max_var:.4f}, normalized variance: {normalized_variance:.4f}"
        )

        return ConsistencyScore(score=score, inconsistent_pairs=inconsistent_pairs, details=details)

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------
    def score_faithfulness(
        self,
        reasoning_chain: ReasoningChain,
        action: Action,
        actual_factors: Dict[str, Any],
    ) -> FaithfulnessScore:
        """Compare explanation content to actual decision factors.

        The current implementation uses simple overlap between keys in
        ``actual_factors`` and textual mentions within the reasoning
        chain. In a more advanced system, this would integrate with
        feature importance tooling from the underlying model.
        """

        steps = reasoning_chain.get_chain()
        text = " ".join(
            [
                getattr(s, "observation", "") + " " + getattr(s, "inference", "")
                for s in steps
            ]
        ).lower()

        aligned: List[str] = []
        misaligned: List[str] = []

        for factor in actual_factors.keys():
            token = str(factor).lower()
            if token in text:
                aligned.append(factor)
            else:
                misaligned.append(factor)

        total = len(aligned) + len(misaligned)
        score = (len(aligned) / total) if total else 0.0

        details = (
            f"Aligned factors: {aligned}; Misaligned factors: {misaligned}; "
            f"Total factors considered: {total}"
        )
        return FaithfulnessScore(score=score, aligned_factors=aligned, misaligned_factors=misaligned, details=details)

    # ------------------------------------------------------------------
    # Overall interpretability
    # ------------------------------------------------------------------
    def compute_overall_interpretability(
        self,
        reasoning_chain: ReasoningChain,
        action: Action,
        decisions_history: Optional[List[Tuple[Dict[str, Any], Action]]] = None,
        actual_factors: Optional[Dict[str, Any]] = None,
    ) -> InterpretabilityReport:
        """Compute an overall interpretability score and grade.

        Weighted average:
        - completeness: 0.4
        - consistency: 0.3
        - faithfulness: 0.3

        If consistency or faithfulness inputs are not provided, the
        weights are re-normalized.
        """

        completeness = self.score_completeness(reasoning_chain, action)
        consistency: Optional[ConsistencyScore] = None
        faithfulness: Optional[FaithfulnessScore] = None

        weights = {"completeness": 0.4, "consistency": 0.3, "faithfulness": 0.3}

        if decisions_history:
            consistency = self.score_consistency(decisions_history)
        else:
            weights["completeness"] += weights["consistency"]
            weights["consistency"] = 0.0

        if actual_factors is not None:
            faithfulness = self.score_faithfulness(reasoning_chain, action, actual_factors)
        else:
            weights["completeness"] += weights["faithfulness"]
            weights["faithfulness"] = 0.0

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values()) or 1.0
        for k in weights.keys():
            weights[k] /= total_weight

        overall_score = (
            completeness.score * weights["completeness"]
            + (consistency.score if consistency else 0.0) * weights["consistency"]
            + (faithfulness.score if faithfulness else 0.0) * weights["faithfulness"]
        )

        # Grade mapping
        if overall_score >= 0.9:
            grade = "A"
        elif overall_score >= 0.75:
            grade = "B"
        elif overall_score >= 0.6:
            grade = "C"
        elif overall_score >= 0.4:
            grade = "D"
        else:
            grade = "F"

        recommendations: List[str] = []
        if completeness.score < 0.8:
            recommendations.append("Improve coverage of required reasoning factors.")
        if consistency and consistency.score < 0.8:
            recommendations.append("Increase consistency of actions under similar conditions.")
        if faithfulness and faithfulness.score < 0.8:
            recommendations.append("Align explanations more closely with actual decision drivers.")

        return InterpretabilityReport(
            overall_score=overall_score,
            completeness=completeness,
            consistency=consistency,
            faithfulness=faithfulness,
            grade=grade,
            recommendations=recommendations,
        )


def asdict_action(action: Action) -> Dict[str, Any]:
    """Utility helper to convert an :class:`Action` to a dictionary.

    We avoid importing :func:`dataclasses.asdict` here to keep the
    dependency surface small, since :class:`Action` may not be a simple
    dataclass in all contexts.
    """

    return {
        "action_type": getattr(action.action_type, "value", str(action.action_type)),
        "symbol": action.symbol,
        "quantity": action.quantity,
        "price": action.price,
        "confidence": action.confidence,
        "stop_loss": action.stop_loss,
        "take_profit": action.take_profit,
        "reasoning_summary": action.reasoning_summary,
    }
