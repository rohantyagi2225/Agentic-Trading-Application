"""Multimodal Agent for Financial Analysis.

This module provides a comprehensive agent that integrates time series encoding,
text sentiment analysis, and chart pattern detection for unified trading decisions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from FinAgents.research.domain_agents.base_agent import (
    Action,
    ActionType,
    Explanation,
    LearningUpdate,
    MarketContext,
    MarketData,
    ReasoningResult,
    ResearchAgent,
)
from FinAgents.research.multimodal.time_series_encoder import (
    TimeSeriesEncoder,
    TimeSeriesFeatures,
)
from FinAgents.research.multimodal.text_encoder import (
    TextEncoder,
    TextFeatures,
)
from FinAgents.research.multimodal.chart_encoder import (
    ChartEncoder,
    ChartFeatures,
)
from FinAgents.research.multimodal.fusion_engine import (
    FusionEngine,
    FusionResult,
)


class MultimodalAgent(ResearchAgent):
    """Multimodal agent that integrates multiple data modalities for trading decisions.

    This agent combines time series technical analysis, text sentiment analysis,
    and chart pattern detection into unified trading signals using attention-based
    fusion.

    Parameters
    ----------
    agent_id : str
        Unique identifier for the agent instance.
    config : dict, optional
        Configuration dictionary with encoder and fusion settings:
        - time_series_config: dict - Config for TimeSeriesEncoder
        - text_config: dict - Config for TextEncoder
        - chart_config: dict - Config for ChartEncoder
        - fusion_config: dict - Config for FusionEngine
        - position_sizing: str - 'equal' or 'confidence_weighted' (default)
        - max_position_pct: float - Maximum position size as % (default 0.1)

    Example
    -------
    >>> agent = MultimodalAgent("multi_001")
    >>> reasoning = agent.reason(market_data, context)
    >>> action = agent.act(reasoning)
    >>> print(action.action_type)  # BUY, SELL, or HOLD
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the MultimodalAgent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.
        config : dict, optional
            Configuration dictionary. See class docstring for options.
        """
        super().__init__(agent_id, config)

        # Initialize encoders
        self.time_series_encoder = TimeSeriesEncoder(
            self.config.get("time_series_config")
        )
        self.text_encoder = TextEncoder(self.config.get("text_config"))
        self.chart_encoder = ChartEncoder(self.config.get("chart_config"))

        # Initialize fusion engine
        self.fusion_engine = FusionEngine(self.config.get("fusion_config"))

        # Position sizing config
        self.position_sizing = self.config.get("position_sizing", "confidence_weighted")
        self.max_position_pct = self.config.get("max_position_pct", 0.1)

        # Track last reasoning and action for explanation
        self._last_reasoning: Optional[ReasoningResult] = None
        self._last_action: Optional[Action] = None
        self._last_fusion: Optional[FusionResult] = None

        # Performance tracking
        self._trade_history: List[Dict[str, Any]] = []
        self._accuracy_history: List[float] = []

    def reason(
        self,
        market_data: MarketData,
        context: Optional[MarketContext] = None,
    ) -> ReasoningResult:
        """Analyze market data using multimodal fusion.

        Parameters
        ----------
        market_data : MarketData
            Primary market data containing prices and volume.
        context : MarketContext, optional
            Additional context including news items.

        Returns
        -------
        ReasoningResult
            Structured reasoning output with multimodal analysis.
        """
        observations = []
        inferences = []
        signals = {}
        reasoning_chain = []

        # Convert prices to DataFrame
        prices_df = self._convert_prices_to_df(market_data.prices)

        # Step 1: Encode time series features
        reasoning_chain.append("Step 1: Extracting time series features...")
        try:
            ts_features = self.time_series_encoder.encode(prices_df)
            observations.append(
                f"Time series: {len(ts_features.feature_names)} features extracted "
                f"from {ts_features.features.shape[0] if ts_features.features.size > 0 else 0} timesteps"
            )
            signals["time_series_features"] = ts_features
        except Exception as e:
            observations.append(f"Time series encoding failed: {str(e)}")
            ts_features = TimeSeriesFeatures(
                features=np.array([]),
                feature_names=[],
                timestamp=datetime.now(),
                metadata={"error": str(e)},
            )

        # Step 2: Encode text features (from context or empty)
        reasoning_chain.append("Step 2: Processing text/news data...")
        news_items = self._get_news_items(context)
        try:
            text_features = self.text_encoder.encode(news_items)
            if text_features.per_item_features:
                observations.append(
                    f"Text: {len(text_features.per_item_features)} items processed, "
                    f"sentiment={text_features.sentiment_score:.3f}"
                )
            else:
                observations.append("Text: No news items available")
            signals["text_features"] = text_features
        except Exception as e:
            observations.append(f"Text encoding failed: {str(e)}")
            text_features = TextFeatures(
                features=np.zeros(8),
                per_item_features=[],
                sentiment_score=0.0,
                metadata={"error": str(e)},
            )

        # Step 3: Encode chart patterns
        reasoning_chain.append("Step 3: Detecting chart patterns...")
        try:
            chart_features = self.chart_encoder.encode(prices_df)
            pattern_names = [p.pattern_name for p in chart_features.detected_patterns]
            observations.append(
                f"Chart: {len(chart_features.detected_patterns)} patterns detected, "
                f"{len(chart_features.support_levels)} support levels, "
                f"{len(chart_features.resistance_levels)} resistance levels"
            )
            if pattern_names:
                inferences.append(f"Detected patterns: {', '.join(pattern_names[:3])}")
            signals["chart_features"] = chart_features
        except Exception as e:
            observations.append(f"Chart encoding failed: {str(e)}")
            chart_features = ChartFeatures(
                features=np.zeros(10),
                detected_patterns=[],
                support_levels=[],
                resistance_levels=[],
                metadata={"error": str(e)},
            )

        # Step 4: Fuse all modalities
        reasoning_chain.append("Step 4: Fusing multimodal signals...")
        fusion_result = self.fusion_engine.fuse(ts_features, text_features, chart_features)

        # Store for later use
        self._last_fusion = fusion_result

        # Add fusion results to signals
        signals["fused_signal"] = fusion_result.fused_signal
        signals["fusion_confidence"] = fusion_result.confidence
        signals["cross_modal_consistency"] = fusion_result.cross_modal_consistency
        signals["modality_contributions"] = fusion_result.modality_contributions

        # Add inferences
        direction = "bullish" if fusion_result.fused_signal > 0.1 else \
                    "bearish" if fusion_result.fused_signal < -0.1 else "neutral"
        inferences.append(
            f"Fused signal: {fusion_result.fused_signal:.3f} ({direction}), "
            f"confidence: {fusion_result.confidence:.3f}"
        )
        inferences.append(
            f"Cross-modal consistency: {fusion_result.cross_modal_consistency:.3f}"
        )

        # Analyze modality contributions
        contributions = fusion_result.modality_contributions
        if contributions:
            dominant_modality = max(contributions, key=lambda k: abs(contributions[k]))
            inferences.append(
                f"Dominant modality: {dominant_modality} "
                f"(contribution: {contributions[dominant_modality]:.3f})"
            )

        # Build reasoning result
        result = ReasoningResult(
            observations=observations,
            inferences=inferences,
            confidence=fusion_result.confidence,
            signals=signals,
            reasoning_chain=reasoning_chain,
        )

        self._last_reasoning = result
        return result

    def act(self, reasoning_result: ReasoningResult) -> Action:
        """Convert fused signal to trading action.

        Parameters
        ----------
        reasoning_result : ReasoningResult
            Output from the reasoning step.

        Returns
        -------
        Action
            Trading action with position sizing.
        """
        fused_signal = reasoning_result.signals.get("fused_signal", 0.0)
        confidence = reasoning_result.confidence

        # Determine action type based on fused signal
        if fused_signal > 0.2:
            action_type = ActionType.BUY
            direction = 1
        elif fused_signal < -0.2:
            action_type = ActionType.SELL
            direction = -1
        else:
            action_type = ActionType.HOLD
            direction = 0

        # Compute position size
        if action_type == ActionType.HOLD:
            quantity = 0.0
        else:
            if self.position_sizing == "confidence_weighted":
                # Scale by signal strength and confidence
                position_size = abs(fused_signal) * confidence * self.max_position_pct
            else:  # equal
                position_size = self.max_position_pct

            quantity = position_size * direction

        # Get symbol from reasoning signals
        symbol = reasoning_result.signals.get("symbol", "UNKNOWN")

        # Build action
        action = Action(
            action_type=action_type,
            symbol=symbol,
            quantity=abs(quantity),
            confidence=confidence,
            reasoning_summary=self._build_action_summary(
                action_type, fused_signal, confidence
            ),
        )

        self._last_action = action
        return action

    def learn(self, outcome: Mapping[str, Any]) -> LearningUpdate:
        """Update fusion engine weights based on outcome.

        Parameters
        ----------
        outcome : Mapping[str, Any]
            Outcome dictionary with keys:
            - 'realized_return': float - Actual return realized
            - 'direction_correct': bool - Whether direction was correct
            - 'magnitude_error': float - Error in magnitude prediction

        Returns
        -------
        LearningUpdate
            Learning update with adjusted parameters.
        """
        realized_return = outcome.get("realized_return", 0.0)
        direction_correct = outcome.get("direction_correct", None)
        magnitude_error = outcome.get("magnitude_error", None)

        parameter_changes = {}
        lessons = []

        # Update fusion weights if we have modality contributions
        if self._last_fusion is not None:
            # Normalize outcome to [-1, 1]
            normalized_outcome = np.clip(realized_return * 10, -1, 1)

            old_weights = self.fusion_engine.modality_weights.copy()
            new_weights = self.fusion_engine.update_weights(
                normalized_outcome,
                self._last_fusion.modality_contributions,
            )

            for modality in new_weights:
                change = new_weights[modality] - old_weights.get(modality, 0)
                if abs(change) > 0.001:
                    parameter_changes[f"weight_{modality}"] = new_weights[modality]

            # Add lessons based on performance
            if direction_correct is not None:
                if direction_correct:
                    lessons.append("Direction prediction correct - reinforcing current weights")
                else:
                    lessons.append("Direction prediction incorrect - adjusting modality weights")

                    # Identify which modality was most wrong
                    if self._last_fusion:
                        contributions = self._last_fusion.modality_contributions
                        for modality, contrib in contributions.items():
                            if np.sign(contrib) != np.sign(normalized_outcome):
                                lessons.append(
                                    f"{modality} modality predicted wrong direction"
                                )

        # Track accuracy
        if direction_correct is not None:
            self._accuracy_history.append(float(direction_correct))

        # Compute confidence adjustment
        confidence_adjustment = 0.0
        if len(self._accuracy_history) >= 5:
            recent_accuracy = np.mean(self._accuracy_history[-5:])
            confidence_adjustment = (recent_accuracy - 0.5) * 0.1

        # Record trade in history
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "outcome": dict(outcome),
            "parameter_changes": parameter_changes.copy(),
        }
        self._trade_history.append(trade_record)

        return LearningUpdate(
            parameter_changes=parameter_changes,
            confidence_adjustment=confidence_adjustment,
            lessons=lessons,
        )

    def explain(self, action: Action) -> Explanation:
        """Generate human-readable explanation for the action.

        Parameters
        ----------
        action : Action
            The action to explain.

        Returns
        -------
        Explanation
            Detailed explanation of the decision.
        """
        reasoning_chain = []
        data_sources = []

        # Build reasoning chain
        if self._last_reasoning:
            reasoning_chain.extend(self._last_reasoning.observations)
            reasoning_chain.append("---")
            reasoning_chain.extend(self._last_reasoning.inferences)

        # Build data sources
        data_sources.append("OHLCV price data (time series)")
        if self._last_reasoning and self._last_reasoning.signals.get("text_features"):
            text_features = self._last_reasoning.signals["text_features"]
            if isinstance(text_features, TextFeatures) and text_features.per_item_features:
                sources = text_features.metadata.get("sources", [])
                data_sources.append(f"News data ({len(sources)} sources)")

        # Build confidence breakdown
        confidence_breakdown = {}

        if self._last_fusion:
            fusion = self._last_fusion

            # Modality contributions
            for modality, contrib in fusion.modality_contributions.items():
                confidence_breakdown[f"{modality}_contribution"] = contrib

            # Consistency factor
            confidence_breakdown["cross_modal_consistency"] = fusion.cross_modal_consistency

            # Detailed breakdown
            detailed = fusion.detailed_breakdown
            if "signals" in detailed:
                for modality, signal in detailed["signals"].items():
                    confidence_breakdown[f"{modality}_signal"] = signal
            if "signal_confidences" in detailed:
                for modality, conf in detailed["signal_confidences"].items():
                    confidence_breakdown[f"{modality}_confidence"] = conf

        # Pattern details
        pattern_details = ""
        if self._last_reasoning:
            chart_features = self._last_reasoning.signals.get("chart_features")
            if isinstance(chart_features, ChartFeatures):
                if chart_features.detected_patterns:
                    pattern_names = [
                        f"{p.pattern_name} ({p.direction}, {p.confidence:.2f})"
                        for p in chart_features.detected_patterns[:3]
                    ]
                    pattern_details = f" Key patterns: {', '.join(pattern_names)}."

        # Build summary
        direction = "buy" if action.action_type == ActionType.BUY else \
                    "sell" if action.action_type == ActionType.SELL else "hold"
        summary = (
            f"Multimodal analysis recommends {direction} action with "
            f"{action.confidence:.1%} confidence.{pattern_details}"
        )

        # Build risk justification
        risk_justification = self._build_risk_justification(action)

        return Explanation(
            summary=summary,
            reasoning_chain=reasoning_chain,
            data_sources=data_sources,
            risk_justification=risk_justification,
            confidence_breakdown=confidence_breakdown,
        )

    def get_state(self) -> Dict[str, Any]:
        """Return serializable state of the agent."""
        return {
            "agent_id": self.agent_id,
            "config": self.config,
            "fusion_engine": self.fusion_engine.get_state(),
            "trade_history": self._trade_history[-100:],  # Keep last 100 trades
            "accuracy_history": self._accuracy_history[-100:],
            "position_sizing": self.position_sizing,
            "max_position_pct": self.max_position_pct,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state into the agent."""
        if "fusion_engine" in state:
            self.fusion_engine.load_state(state["fusion_engine"])
        if "trade_history" in state:
            self._trade_history = state["trade_history"]
        if "accuracy_history" in state:
            self._accuracy_history = state["accuracy_history"]

    def _convert_prices_to_df(
        self, prices: List[Mapping[str, Any]]
    ) -> pd.DataFrame:
        """Convert price list to DataFrame."""
        if not prices:
            return pd.DataFrame()

        # Handle both dict and object-like items
        rows = []
        for p in prices:
            if isinstance(p, dict):
                rows.append(p)
            else:
                # Try to extract attributes
                try:
                    rows.append({
                        "open": getattr(p, "open", None),
                        "high": getattr(p, "high", None),
                        "low": getattr(p, "low", None),
                        "close": getattr(p, "close", None),
                        "volume": getattr(p, "volume", None),
                        "timestamp": getattr(p, "timestamp", None),
                    })
                except AttributeError:
                    continue

        df = pd.DataFrame(rows)
        return df

    def _get_news_items(
        self, context: Optional[MarketContext]
    ) -> List[Dict[str, Any]]:
        """Extract news items from context."""
        if context is None:
            return []

        # Check for news_items in context attributes or extra fields
        news_items = getattr(context, "news_items", None)
        if news_items:
            return news_items

        # Check events list for news-like items
        if context.events:
            return [
                {"text": event, "timestamp": datetime.now().isoformat(), "source": "events"}
                for event in context.events
            ]

        return []

    def _build_action_summary(
        self, action_type: ActionType, signal: float, confidence: float
    ) -> str:
        """Build a summary string for the action."""
        direction = action_type.value.lower()
        strength = "strong" if abs(signal) > 0.5 else "moderate" if abs(signal) > 0.3 else "weak"

        return f"{strength.title()} {direction} signal ({signal:.3f}) with {confidence:.1%} confidence"

    def _build_risk_justification(self, action: Action) -> str:
        """Build risk justification string."""
        if action.action_type == ActionType.HOLD:
            return "Holding position due to insufficient signal strength or low confidence."

        risk_factors = []

        if self._last_fusion:
            # Consistency risk
            consistency = self._last_fusion.cross_modal_consistency
            if consistency < 0.4:
                risk_factors.append(
                    f"Low cross-modal consistency ({consistency:.2f}) indicates uncertainty"
                )
            elif consistency < 0.6:
                risk_factors.append(
                    f"Moderate consistency ({consistency:.2f}) - some modalities disagree"
                )

            # Pattern risk
            if self._last_reasoning:
                chart_features = self._last_reasoning.signals.get("chart_features")
                if isinstance(chart_features, ChartFeatures):
                    conflicting = sum(
                        1 for p in chart_features.detected_patterns
                        if p.direction != ("bullish" if action.action_type == ActionType.BUY else "bearish")
                    )
                    if conflicting > 0:
                        risk_factors.append(
                            f"{conflicting} pattern(s) suggest opposite direction"
                        )

        if risk_factors:
            return "Risk factors: " + "; ".join(risk_factors)
        else:
            return "No significant risk factors identified from multimodal analysis."
