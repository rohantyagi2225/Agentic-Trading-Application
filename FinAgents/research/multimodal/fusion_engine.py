"""Fusion Engine for Multimodal Decision Fusion.

This module provides attention-based fusion of signals from multiple modalities
(time series, text, chart) into unified trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from FinAgents.research.multimodal.time_series_encoder import TimeSeriesFeatures
from FinAgents.research.multimodal.text_encoder import TextFeatures
from FinAgents.research.multimodal.chart_encoder import ChartFeatures


@dataclass
class FusionResult:
    """Container for fusion engine output.

    Attributes
    ----------
    fused_signal : float
        Combined signal in range [-1, 1] where -1 is strongly bearish
        and 1 is strongly bullish.
    confidence : float
        Overall confidence in the fused signal, range [0, 1].
    modality_contributions : Dict[str, float]
        Contribution of each modality to the final signal.
    cross_modal_consistency : float
        Agreement score between modalities, range [0, 1].
    detailed_breakdown : Dict[str, Any]
        Detailed breakdown of signals from each modality.
    metadata : Dict[str, Any]
        Additional metadata about the fusion process.
    """

    fused_signal: float
    confidence: float
    modality_contributions: Dict[str, float]
    cross_modal_consistency: float
    detailed_breakdown: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FusionEngine:
    """Fuses signals from multiple modalities into unified trading signals.

    This engine extracts directional signals from each modality, computes
    cross-modal consistency, and applies attention-based weighting to produce
    a final fused signal.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with the following options:
        - modality_weights: Dict[str, float] - Initial weights for each modality
          (default: {time_series: 0.4, text: 0.3, chart: 0.3})
        - consistency_threshold: float - Threshold for high consistency (default 0.6)
        - attention_learning_rate: float - Learning rate for weight updates (default 0.01)

    Example
    -------
    >>> engine = FusionEngine()
    >>> result = engine.fuse(time_series_features, text_features, chart_features)
    >>> print(result.fused_signal)  # -1 to 1, bearish to bullish
    """

    DEFAULT_WEIGHTS = {
        "time_series": 0.4,
        "text": 0.3,
        "chart": 0.3,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the FusionEngine.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. See class docstring for options.
        """
        self.config = config or {}

        # Modality weights
        self.modality_weights = self.config.get(
            "modality_weights", self.DEFAULT_WEIGHTS.copy()
        )
        # Normalize weights
        self._normalize_weights()

        # Thresholds and learning rate
        self.consistency_threshold = self.config.get("consistency_threshold", 0.6)
        self.learning_rate = self.config.get("attention_learning_rate", 0.01)

        # Track historical performance per modality
        self.modality_accuracy = {
            "time_series": [],
            "text": [],
            "chart": [],
        }

    def fuse(
        self,
        time_series_features: TimeSeriesFeatures,
        text_features: TextFeatures,
        chart_features: ChartFeatures,
    ) -> FusionResult:
        """Fuse signals from all modalities.

        Parameters
        ----------
        time_series_features : TimeSeriesFeatures
            Features from time series encoder.
        text_features : TextFeatures
            Features from text encoder.
        chart_features : ChartFeatures
            Features from chart encoder.

        Returns
        -------
        FusionResult
            Container with fused signal, confidence, and breakdown.
        """
        # Step 1: Extract directional signals from each modality
        signals = {}
        signal_confidences = {}

        # Time series signal (momentum/trend based)
        signals["time_series"], signal_confidences["time_series"] = (
            self._extract_time_series_signal(time_series_features)
        )

        # Text signal (sentiment based)
        signals["text"], signal_confidences["text"] = self._extract_text_signal(
            text_features
        )

        # Chart signal (pattern based)
        signals["chart"], signal_confidences["chart"] = self._extract_chart_signal(
            chart_features
        )

        # Step 2: Compute cross-modal consistency
        consistency = self.check_consistency(signals)

        # Step 3: Compute attention weights based on consistency
        attention_weights = self._compute_attention_weights(
            signals, signal_confidences, consistency
        )

        # Step 4: Compute fused signal
        fused_signal = 0.0
        total_weight = 0.0

        for modality, signal in signals.items():
            weight = attention_weights[modality]
            fused_signal += signal * weight
            total_weight += weight

        if total_weight > 0:
            fused_signal /= total_weight

        # Step 5: Compute confidence
        base_confidence = np.mean(list(signal_confidences.values()))
        consistency_factor = 0.5 + 0.5 * consistency  # Scale from 0.5 to 1.0
        confidence = base_confidence * consistency_factor

        # Build modality contributions
        contributions = {}
        for modality, signal in signals.items():
            contributions[modality] = signal * attention_weights[modality]

        # Build detailed breakdown
        detailed_breakdown = {
            "signals": signals,
            "signal_confidences": signal_confidences,
            "attention_weights": attention_weights,
            "base_weights": self.modality_weights.copy(),
            "consistency_score": consistency,
            "consistency_level": self._get_consistency_level(consistency),
        }

        # Add pattern details if available
        if chart_features.detected_patterns:
            detailed_breakdown["detected_patterns"] = [
                {
                    "name": p.pattern_name,
                    "direction": p.direction,
                    "confidence": p.confidence,
                }
                for p in chart_features.detected_patterns[:3]  # Top 3 patterns
            ]

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "modality_weights": self.modality_weights.copy(),
        }

        return FusionResult(
            fused_signal=float(np.clip(fused_signal, -1, 1)),
            confidence=float(np.clip(confidence, 0, 1)),
            modality_contributions=contributions,
            cross_modal_consistency=consistency,
            detailed_breakdown=detailed_breakdown,
            metadata=metadata,
        )

    def check_consistency(self, signals: Dict[str, float]) -> float:
        """Check consistency between modality signals.

        Parameters
        ----------
        signals : Dict[str, float]
            Dictionary of signal values from each modality.

        Returns
        -------
        float
            Consistency score in range [0, 1]. 1.0 means all signals agree
            completely, 0.0 means complete disagreement.
        """
        if not signals:
            return 0.0

        signal_values = list(signals.values())
        n = len(signal_values)

        if n < 2:
            return 1.0  # Single modality is trivially consistent

        # Check direction agreement
        directions = [np.sign(s) for s in signal_values]
        direction_agreement = sum(1 for d in directions if d == directions[0]) / n

        # Check magnitude consistency (using coefficient of variation)
        signal_array = np.array(signal_values)
        mean_signal = np.mean(np.abs(signal_array))

        if mean_signal < 0.01:
            # All signals near zero - high consistency
            magnitude_consistency = 1.0
        else:
            signal_std = np.std(signal_array)
            cv = signal_std / mean_signal
            magnitude_consistency = max(0, 1 - cv)

        # Combined consistency score
        consistency = 0.6 * direction_agreement + 0.4 * magnitude_consistency

        return float(np.clip(consistency, 0, 1))

    def update_weights(
        self,
        outcome: float,
        modality_contributions: Dict[str, float],
    ) -> Dict[str, float]:
        """Update modality weights based on observed outcome.

        After observing the outcome, adjust modality_weights toward modalities
        that predicted correctly. Uses simple gradient descent.

        Parameters
        ----------
        outcome : float
            The actual outcome, normalized to [-1, 1] (negative = bearish,
            positive = bullish).
        modality_contributions : Dict[str, float]
            Contribution of each modality to the prediction.

        Returns
        -------
        Dict[str, float]
            Updated modality weights.
        """
        for modality, contribution in modality_contributions.items():
            if modality not in self.modality_weights:
                continue

            # Compute how well this modality predicted the outcome
            if contribution != 0 and outcome != 0:
                prediction_correctness = np.sign(contribution) == np.sign(outcome)

                if prediction_correctness:
                    # Increase weight
                    adjustment = self.learning_rate * abs(outcome)
                else:
                    # Decrease weight
                    adjustment = -self.learning_rate * abs(outcome)

                self.modality_weights[modality] += adjustment

                # Track accuracy
                self.modality_accuracy[modality].append(float(prediction_correctness))

        # Normalize weights
        self._normalize_weights()

        return self.modality_weights.copy()

    def _extract_time_series_signal(
        self, features: TimeSeriesFeatures
    ) -> tuple[float, float]:
        """Extract directional signal from time series features.

        Returns
        -------
        tuple[float, float]
            (signal, confidence) where signal is in [-1, 1].
        """
        if features.features.size == 0:
            return 0.0, 0.0

        feature_matrix = features.features
        feature_names = features.feature_names

        # Get last row (most recent features)
        last_features = feature_matrix[-1] if len(feature_matrix.shape) > 1 else feature_matrix

        # Extract key features for signal
        signal = 0.0
        weight_sum = 0.0

        # Look for momentum and trend features
        feature_dict = dict(zip(feature_names, last_features))

        # Momentum features
        for window in [5, 10, 20]:
            roc_key = f"roc_{window}"
            if roc_key in feature_dict:
                weight = 1.0 / window  # Shorter windows have higher weight
                signal += feature_dict[roc_key] * weight
                weight_sum += weight

        # Trend strength
        if "trend_strength" in feature_dict:
            signal += feature_dict["trend_strength"] * 2
            weight_sum += 2

        # RSI (contrarian at extremes, momentum in middle)
        if "rsi_14" in feature_dict:
            rsi = feature_dict["rsi_14"]
            if rsi > 0.7:  # Overbought - bearish signal
                signal -= (rsi - 0.7) * 5
                weight_sum += 1
            elif rsi < 0.3:  # Oversold - bullish signal
                signal += (0.3 - rsi) * 5
                weight_sum += 1

        # Bollinger Band position
        if "bb_position" in feature_dict:
            bb_pos = feature_dict["bb_position"]
            signal += bb_pos * 0.5
            weight_sum += 0.5

        # Normalize signal
        if weight_sum > 0:
            signal /= weight_sum

        # Compute confidence based on feature consistency
        confidence = min(0.9, 0.3 + weight_sum * 0.1)

        return float(np.clip(signal, -1, 1)), float(confidence)

    def _extract_text_signal(self, features: TextFeatures) -> tuple[float, float]:
        """Extract directional signal from text features.

        Returns
        -------
        tuple[float, float]
            (signal, confidence) where signal is in [-1, 1].
        """
        if features.features.size == 0:
            return 0.0, 0.0

        # Use sentiment score as primary signal
        sentiment = features.sentiment_score

        # Get additional features
        feature_vector = features.features
        # [avg_sentiment, sentiment_std, max_sentiment, min_sentiment,
        #  news_count, avg_magnitude, entity_density, temporal_impact_score]

        if len(feature_vector) >= 8:
            news_count = feature_vector[4]
            magnitude = feature_vector[5]
            temporal_impact = feature_vector[7]

            # Confidence based on news volume and magnitude
            confidence = min(0.9, 0.2 + magnitude * 0.5 + min(news_count / 10, 0.2))

            # Weight signal by temporal impact
            signal = sentiment * (0.5 + 0.5 * abs(temporal_impact))
        else:
            signal = sentiment
            confidence = 0.3

        return float(np.clip(signal, -1, 1)), float(confidence)

    def _extract_chart_signal(self, features: ChartFeatures) -> tuple[float, float]:
        """Extract directional signal from chart features.

        Returns
        -------
        tuple[float, float]
            (signal, confidence) where signal is in [-1, 1].
        """
        if features.features.size == 0:
            return 0.0, 0.0

        feature_vector = features.features
        # [num_bullish, num_bearish, strongest_pattern_confidence,
        #  distance_to_support_pct, distance_to_resistance_pct, volume_profile_skew,
        #  trend_angle, pattern_agreement_score, volume_ratio, volatility]

        if len(feature_vector) < 10:
            return 0.0, 0.3

        num_bullish = feature_vector[0]
        num_bearish = feature_vector[1]
        strongest_confidence = feature_vector[2]
        dist_to_support = feature_vector[3]
        dist_to_resistance = feature_vector[4]
        volume_skew = feature_vector[5]
        trend_angle = feature_vector[6]
        pattern_agreement = feature_vector[7]

        # Pattern-based signal
        pattern_signal = 0.0
        if num_bullish + num_bearish > 0:
            pattern_signal = (num_bullish - num_bearish) / (num_bullish + num_bearish)

        # Support/resistance signal
        sr_signal = 0.0
        if dist_to_support > 0 or dist_to_resistance > 0:
            # Closer to support = bullish, closer to resistance = bearish
            total_dist = dist_to_support + dist_to_resistance
            if total_dist > 0:
                sr_signal = (dist_to_resistance - dist_to_support) / total_dist

        # Trend signal
        trend_signal = np.clip(trend_angle, -1, 1)

        # Combine signals
        signal = (
            pattern_signal * 0.4
            + sr_signal * 0.2
            + trend_signal * 0.3
            + volume_skew * 0.1
        )

        # Weight by pattern confidence and agreement
        confidence = min(0.9, strongest_confidence * 0.5 + pattern_agreement * 0.3 + 0.2)

        return float(np.clip(signal, -1, 1)), float(confidence)

    def _compute_attention_weights(
        self,
        signals: Dict[str, float],
        confidences: Dict[str, float],
        consistency: float,
    ) -> Dict[str, float]:
        """Compute attention-adjusted modality weights.

        If consistency is high, use base weights equally.
        If consistency is low, upweight the most confident modality.
        """
        attention_weights = {}

        if consistency >= self.consistency_threshold:
            # High consistency - use base weights
            for modality in signals:
                attention_weights[modality] = self.modality_weights.get(modality, 0.33)
        else:
            # Low consistency - upweight confident modalities
            total_confidence = sum(confidences.values())

            for modality in signals:
                base_weight = self.modality_weights.get(modality, 0.33)
                conf_weight = confidences.get(modality, 0.5)

                if total_confidence > 0:
                    # Blend base weight with confidence-based weighting
                    confidence_weight = conf_weight / total_confidence
                    # Interpolate between base weight and confidence weight
                    blend = 1 - consistency  # Lower consistency = more confidence weighting
                    attention_weights[modality] = (
                        base_weight * (1 - blend) + confidence_weight * blend
                    )
                else:
                    attention_weights[modality] = base_weight

        return attention_weights

    def _normalize_weights(self) -> None:
        """Normalize modality weights to sum to 1."""
        total = sum(self.modality_weights.values())
        if total > 0:
            self.modality_weights = {
                k: v / total for k, v in self.modality_weights.items()
            }
        else:
            # Reset to defaults if all weights are zero
            self.modality_weights = self.DEFAULT_WEIGHTS.copy()

    def _get_consistency_level(self, consistency: float) -> str:
        """Get human-readable consistency level."""
        if consistency >= 0.8:
            return "high"
        elif consistency >= 0.6:
            return "moderate"
        elif consistency >= 0.4:
            return "low"
        else:
            return "very_low"

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state of the fusion engine."""
        return {
            "modality_weights": self.modality_weights.copy(),
            "modality_accuracy": {
                k: np.mean(v) if v else 0.5 for k, v in self.modality_accuracy.items()
            },
            "config": self.config,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state into the fusion engine."""
        if "modality_weights" in state:
            self.modality_weights = state["modality_weights"]
            self._normalize_weights()
