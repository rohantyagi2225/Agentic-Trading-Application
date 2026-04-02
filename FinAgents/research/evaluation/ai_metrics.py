"""AI/Agent-Specific Metrics for Research Evaluation.

This module provides metrics specific to evaluating AI trading agents,
including decision accuracy, confidence calibration, and explainability quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AIMetricsReport:
    """Comprehensive AI/agent metrics report.

    Attributes
    ----------
    decision_accuracy : float
        Overall accuracy of trading decisions.
    precision : float
        Precision of buy signals.
    recall : float
        Recall of buy signals.
    f1_score : float
        F1 score combining precision and recall.
    confidence_calibration_ece : float
        Expected Calibration Error for confidence scores.
    explainability_score : float
        Average quality score of explanations.
    learning_rate_metric : float
        Trend of performance improvement over time.
    agent_agreement_rate : float
        Rate at which multiple agents agree on decisions.
    direction_accuracy : float
        Accuracy of price direction predictions.
    magnitude_accuracy : float
        Accuracy of price magnitude predictions.
    details : dict
        Additional details and breakdowns.
    """

    decision_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confidence_calibration_ece: float = 0.0
    explainability_score: float = 0.0
    learning_rate_metric: float = 0.0
    agent_agreement_rate: float = 0.0
    direction_accuracy: float = 0.0
    magnitude_accuracy: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class AIMetricsCalculator:
    """Calculate AI/agent-specific metrics.

    Provides methods for evaluating agent decision quality, calibration,
    explainability, and multi-agent coordination.

    Parameters
    ----------
    config : dict, optional
        Configuration parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the AI metrics calculator.

        Parameters
        ----------
        config : dict, optional
            Configuration parameters.
        """
        self.config = config or {}

    def compute_all_metrics(
        self,
        predictions: List[Dict[str, Any]],
        actuals: List[Dict[str, Any]],
        confidence_scores: Optional[List[float]] = None,
        explanations: Optional[List[Any]] = None,
        multi_agent_decisions: Optional[List[List[str]]] = None,
        performance_over_time: Optional[List[float]] = None,
    ) -> AIMetricsReport:
        """Compute comprehensive AI metrics.

        Parameters
        ----------
        predictions : list of dict
            Agent predictions with 'action' and optionally 'direction', 'magnitude'.
        actuals : list of dict
            Actual outcomes with 'action' and optionally 'direction', 'magnitude'.
        confidence_scores : list of float, optional
            Confidence scores for each prediction.
        explanations : list, optional
            Explanation objects (ReasoningChain or similar).
        multi_agent_decisions : list of list of str, optional
            Decisions from multiple agents for each round.
        performance_over_time : list of float, optional
            Performance scores over time for learning rate calculation.

        Returns
        -------
        AIMetricsReport
            Comprehensive AI metrics report.
        """
        if not predictions or not actuals:
            return AIMetricsReport()

        # Decision accuracy
        pred_actions = [p.get("action", "HOLD") for p in predictions]
        actual_actions = [a.get("action", "HOLD") for a in actuals]
        decision_accuracy = self.decision_accuracy(pred_actions, actual_actions)

        # Signal quality (precision, recall, F1)
        quality = self.signal_quality(pred_actions, actual_actions)
        precision = quality["precision"]
        recall = quality["recall"]
        f1_score = quality["f1"]

        # Confidence calibration
        ece = 0.0
        if confidence_scores and len(confidence_scores) == len(predictions):
            correct = [
                pred_actions[i] == actual_actions[i]
                for i in range(min(len(pred_actions), len(actual_actions)))
            ]
            ece = self.confidence_calibration(confidence_scores, correct)

        # Explainability quality
        explainability_score = 0.0
        if explanations:
            explainability_score = self.explainability_quality(
                explanations, pred_actions
            )

        # Learning rate
        learning_rate_metric = 0.0
        if performance_over_time:
            learning_rate_metric = self.learning_rate(performance_over_time)

        # Agent agreement rate
        agent_agreement_rate = 0.0
        if multi_agent_decisions:
            agent_agreement_rate = self.agent_agreement_rate(multi_agent_decisions)

        # Direction accuracy
        direction_accuracy = 0.0
        pred_directions = [p.get("direction") for p in predictions if "direction" in p]
        actual_directions = [a.get("direction") for a in actuals if "direction" in a]
        if pred_directions and actual_directions:
            min_len = min(len(pred_directions), len(actual_directions))
            direction_accuracy = self.direction_accuracy(
                pred_directions[:min_len], actual_directions[:min_len]
            )

        # Magnitude accuracy
        magnitude_accuracy = 0.0
        pred_magnitudes = [p.get("magnitude") for p in predictions if "magnitude" in p]
        actual_magnitudes = [a.get("magnitude") for a in actuals if "magnitude" in a]
        if pred_magnitudes and actual_magnitudes:
            min_len = min(len(pred_magnitudes), len(actual_magnitudes))
            magnitude_accuracy = self.magnitude_accuracy(
                pred_magnitudes[:min_len], actual_magnitudes[:min_len]
            )

        return AIMetricsReport(
            decision_accuracy=decision_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confidence_calibration_ece=ece,
            explainability_score=explainability_score,
            learning_rate_metric=learning_rate_metric,
            agent_agreement_rate=agent_agreement_rate,
            direction_accuracy=direction_accuracy,
            magnitude_accuracy=magnitude_accuracy,
            details={
                "num_predictions": len(predictions),
                "signal_quality": quality,
            },
        )

    def decision_accuracy(
        self, predictions: List[str], actuals: List[str]
    ) -> float:
        """Calculate simple decision accuracy.

        Parameters
        ----------
        predictions : list of str
            Predicted actions ("BUY", "SELL", "HOLD").
        actuals : list of str
            Actual optimal actions.

        Returns
        -------
        float
            Accuracy (correct / total).
        """
        if not predictions or not actuals:
            return 0.0

        min_len = min(len(predictions), len(actuals))
        if min_len == 0:
            return 0.0

        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        correct = sum(
            1 for p, a in zip(predictions, actuals)
            if p.upper() == a.upper()
        )

        return float(correct / min_len)

    def signal_quality(
        self, predictions: List[str], actuals: List[str]
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for signals.

        Treats BUY as positive class, SELL as negative class,
        and HOLD as neutral (excluded from calculation).

        Parameters
        ----------
        predictions : list of str
            Predicted actions.
        actuals : list of str
            Actual actions.

        Returns
        -------
        dict
            {precision, recall, f1} treating BUY as positive.
        """
        if not predictions or not actuals:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        min_len = min(len(predictions), len(actuals))
        predictions = [p.upper() for p in predictions[:min_len]]
        actuals = [a.upper() for a in actuals[:min_len]]

        # Calculate confusion matrix for BUY signals
        tp = sum(1 for p, a in zip(predictions, actuals) if p == "BUY" and a == "BUY")
        fp = sum(1 for p, a in zip(predictions, actuals) if p == "BUY" and a != "BUY")
        fn = sum(1 for p, a in zip(predictions, actuals) if p != "BUY" and a == "BUY")

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 = 2 * P * R / (P + R)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def confidence_calibration(
        self,
        confidences: List[float],
        correct: List[bool],
        num_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error (ECE).

        Measures the difference between predicted confidence and actual accuracy.
        Lower ECE indicates better calibration.

        Parameters
        ----------
        confidences : list of float
            Confidence scores in [0, 1].
        correct : list of bool
            Whether each prediction was correct.
        num_bins : int
            Number of bins for calibration calculation.

        Returns
        -------
        float
            Expected Calibration Error.
        """
        if not confidences or not correct:
            return 0.0

        min_len = min(len(confidences), len(correct))
        confidences = confidences[:min_len]
        correct = correct[:min_len]

        if min_len == 0:
            return 0.0

        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        
        ece = 0.0
        total_samples = min_len

        for i in range(num_bins):
            # Find samples in this bin
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = [
                j for j, c in enumerate(confidences)
                if lower <= c < upper or (i == num_bins - 1 and c == upper)
            ]

            if not in_bin:
                continue

            bin_count = len(in_bin)
            avg_confidence = np.mean([confidences[j] for j in in_bin])
            accuracy = np.mean([1.0 if correct[j] else 0.0 for j in in_bin])

            # Add weighted error
            ece += (bin_count / total_samples) * abs(avg_confidence - accuracy)

        return float(ece)

    def explainability_quality(
        self, reasoning_chains: List[Any], actions: List[str]
    ) -> float:
        """Evaluate explanation quality using InterpretabilityScorer.

        Parameters
        ----------
        reasoning_chains : list
            Reasoning chain objects.
        actions : list of str
            Corresponding actions.

        Returns
        -------
        float
            Average overall explainability score.
        """
        if not reasoning_chains:
            return 0.0

        scores = []

        try:
            from FinAgents.research.explainability.interpretability_metrics import (
                InterpretabilityScorer,
            )
            from FinAgents.research.explainability.reasoning_chain import ReasoningChain
            from FinAgents.research.domain_agents.base_agent import Action, ActionType

            scorer = InterpretabilityScorer()

            for i, chain in enumerate(reasoning_chains):
                try:
                    # Create Action object for scoring
                    action_str = actions[i] if i < len(actions) else "HOLD"
                    action_type = ActionType.HOLD
                    if action_str.upper() == "BUY":
                        action_type = ActionType.BUY
                    elif action_str.upper() == "SELL":
                        action_type = ActionType.SELL

                    action = Action(action_type=action_type)

                    # Score the chain
                    if isinstance(chain, ReasoningChain):
                        report = scorer.compute_overall_interpretability(chain, action)
                        scores.append(report.overall_score)
                    elif hasattr(chain, "get_chain"):
                        # Duck typing for ReasoningChain-like objects
                        report = scorer.compute_overall_interpretability(chain, action)
                        scores.append(report.overall_score)

                except Exception:
                    # Skip chains that can't be scored
                    continue

        except ImportError:
            # InterpretabilityScorer not available
            return 0.0

        return float(np.mean(scores)) if scores else 0.0

    def learning_rate(
        self, performance_over_time: List[float], window: int = 50
    ) -> float:
        """Calculate learning rate as performance trend.

        Fits a linear regression over the rolling window to detect
        improvement trend.

        Parameters
        ----------
        performance_over_time : list of float
            Performance scores at each time step.
        window : int
            Rolling window for trend calculation.

        Returns
        -------
        float
            Normalized slope (positive = improving).
        """
        if len(performance_over_time) < 2:
            return 0.0

        performance = np.array(performance_over_time)
        n = len(performance)

        # Use available data if less than window
        if n < window:
            window = n

        # Use the last 'window' data points
        recent = performance[-window:]
        x = np.arange(len(recent))

        # Linear regression: y = mx + b
        # m = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
        n_points = len(recent)
        sum_x = np.sum(x)
        sum_y = np.sum(recent)
        sum_xy = np.sum(x * recent)
        sum_x2 = np.sum(x ** 2)

        denominator = n_points * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n_points * sum_xy - sum_x * sum_y) / denominator

        # Normalize by mean performance
        mean_perf = np.mean(recent) if np.mean(recent) != 0 else 1.0
        normalized_slope = slope / abs(mean_perf)

        return float(normalized_slope)

    def agent_agreement_rate(
        self, multi_agent_decisions: List[List[str]]
    ) -> float:
        """Calculate rate of agreement among multiple agents.

        For each decision round, calculates the fraction of agents
        that agreed with the majority decision.

        Parameters
        ----------
        multi_agent_decisions : list of list of str
            Decisions from each agent for each round.

        Returns
        -------
        float
            Average agreement rate across all rounds.
        """
        if not multi_agent_decisions:
            return 0.0

        agreement_rates = []

        for decisions in multi_agent_decisions:
            if not decisions or len(decisions) < 2:
                continue

            # Count occurrences of each decision
            decision_counts: Dict[str, int] = {}
            for d in decisions:
                d_upper = d.upper() if isinstance(d, str) else str(d)
                decision_counts[d_upper] = decision_counts.get(d_upper, 0) + 1

            if not decision_counts:
                continue

            # Find majority decision
            majority_count = max(decision_counts.values())
            total = len(decisions)

            # Agreement rate = fraction agreeing with majority
            agreement_rate = majority_count / total
            agreement_rates.append(agreement_rate)

        return float(np.mean(agreement_rates)) if agreement_rates else 0.0

    def direction_accuracy(
        self, predicted_directions: List[int], actual_directions: List[int]
    ) -> float:
        """Calculate accuracy of direction predictions.

        Directions should be -1 (down), 0 (neutral), or 1 (up).

        Parameters
        ----------
        predicted_directions : list of int
            Predicted directions (-1, 0, 1).
        actual_directions : list of int
            Actual directions (-1, 0, 1).

        Returns
        -------
        float
            Direction accuracy (matches / total).
        """
        if not predicted_directions or not actual_directions:
            return 0.0

        min_len = min(len(predicted_directions), len(actual_directions))
        predicted = predicted_directions[:min_len]
        actual = actual_directions[:min_len]

        if min_len == 0:
            return 0.0

        matches = sum(
            1 for p, a in zip(predicted, actual)
            if p == a
        )

        return float(matches / min_len)

    def magnitude_accuracy(
        self, predicted_magnitudes: List[float], actual_magnitudes: List[float]
    ) -> float:
        """Calculate accuracy of magnitude predictions.

        Returns 1 - normalized MAE (Mean Absolute Error).

        Parameters
        ----------
        predicted_magnitudes : list of float
            Predicted magnitudes.
        actual_magnitudes : list of float
            Actual magnitudes.

        Returns
        -------
        float
            Magnitude accuracy (1 - normalized_MAE).
        """
        if not predicted_magnitudes or not actual_magnitudes:
            return 0.0

        min_len = min(len(predicted_magnitudes), len(actual_magnitudes))
        predicted = np.array(predicted_magnitudes[:min_len])
        actual = np.array(actual_magnitudes[:min_len])

        if min_len == 0:
            return 0.0

        # Mean Absolute Error
        mae = float(np.mean(np.abs(predicted - actual)))

        # Normalize by mean of absolute actual values
        mean_actual = float(np.mean(np.abs(actual)))
        if mean_actual == 0:
            # If all actuals are zero, use MAE directly
            normalized_mae = mae
        else:
            normalized_mae = mae / mean_actual

        # Convert to accuracy: 1 - normalized_mae (clamped to [0, 1])
        accuracy = max(0.0, min(1.0, 1.0 - normalized_mae))

        return float(accuracy)
