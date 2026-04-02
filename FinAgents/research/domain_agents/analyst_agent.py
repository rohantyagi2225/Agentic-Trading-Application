"""AnalystAgent implementation for multi-source market analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


POSITIVE_WORDS = {
    "beat",
    "outperform",
    "upgrade",
    "bullish",
    "strong",
    "record",
    "growth",
}

NEGATIVE_WORDS = {
    "miss",
    "downgrade",
    "bearish",
    "weak",
    "loss",
    "decline",
    "risk",
}


@dataclass
class MacroIndicators:
    """Container for macroeconomic indicators used by the analyst agent."""

    gdp_growth: float
    interest_rate: float
    inflation: float
    unemployment: float


class AnalystAgent(ResearchAgent):
    """Multi-source analyst combining macro, sentiment, and event analysis."""

    def analyze_macro(self, indicators: MacroIndicators) -> Tuple[float, str]:
        """Score macro environment and suggest a regime.

        Returns
        -------
        macro_score:
            Aggregate macro score in ``[-1, 1]``.
        regime_suggestion:
            Text label such as "expansion" or "contraction".
        """

        score = 0.0
        score += np.tanh(indicators.gdp_growth)
        score -= np.tanh(max(indicators.inflation - 0.02, 0.0))
        score -= np.tanh(max(indicators.interest_rate - 0.02, 0.0))
        score -= np.tanh(max(indicators.unemployment - 0.05, 0.0))
        macro_score = float(np.clip(score / 4.0, -1.0, 1.0))

        if macro_score > 0.3:
            regime = "expansion"
        elif macro_score < -0.3:
            regime = "contraction"
        else:
            regime = "transition"

        return macro_score, regime

    def analyze_sentiment(self, news_items: Sequence[str]) -> Tuple[float, List[float]]:
        """Score aggregate news sentiment using keyword matching."""

        if not news_items:
            return 0.0, []

        scores: List[float] = []
        for item in news_items:
            text = item.lower()
            pos_hits = sum(1 for w in POSITIVE_WORDS if w in text)
            neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text)
            raw = pos_hits - neg_hits
            scores.append(float(np.tanh(raw / 3.0)))

        aggregate = float(np.mean(scores)) if scores else 0.0
        return aggregate, scores

    def model_event_impact(self, event_type: str, magnitude: float) -> Dict[str, float]:
        """Model impact of various event types on price and volatility."""

        magnitude = float(np.clip(magnitude, 0.0, 1.0))

        if event_type == "earnings_surprise":
            expected_price_impact = 0.05 * magnitude
            volatility_impact = 0.1 * magnitude
            duration = 5.0 * magnitude
        elif event_type == "fed_decision":
            expected_price_impact = 0.02 * magnitude
            volatility_impact = 0.2 * magnitude
            duration = 10.0 * magnitude
        elif event_type == "geopolitical":
            expected_price_impact = -0.03 * magnitude
            volatility_impact = 0.25 * magnitude
            duration = 20.0 * magnitude
        elif event_type == "sector_rotation":
            expected_price_impact = 0.01 * magnitude
            volatility_impact = 0.05 * magnitude
            duration = 15.0 * magnitude
        else:
            expected_price_impact = 0.0
            volatility_impact = 0.0
            duration = 0.0

        return {
            "expected_price_impact": expected_price_impact,
            "volatility_impact": volatility_impact,
            "duration": duration,
        }

    def reason(self, market_data: MarketData, context: Optional[MarketContext] = None) -> ReasoningResult:
        macro_score = 0.0
        regime_suggestion = market_data.regime or (context.regime if context and context.regime else "neutral")

        if context is not None and context.macro_indicators:
            indicators = MacroIndicators(
                gdp_growth=context.macro_indicators.get("gdp_growth", 0.0),
                interest_rate=context.macro_indicators.get("interest_rate", 0.02),
                inflation=context.macro_indicators.get("inflation", 0.02),
                unemployment=context.macro_indicators.get("unemployment", 0.05),
            )
            macro_score, regime_suggestion = self.analyze_macro(indicators)

        news_items = context.events if context is not None else []
        sentiment_score, per_item_scores = self.analyze_sentiment(news_items)

        event_signal = 0.0
        if context is not None and context.events:
            impacts = []
            for ev in context.events:
                imp = self.model_event_impact(ev, magnitude=0.5)
                impacts.append(imp["expected_price_impact"])
            if impacts:
                event_signal = float(np.mean(impacts))

        aggregate_score = float(np.clip((macro_score + sentiment_score + event_signal) / 3.0, -1.0, 1.0))
        confidence = float(abs(aggregate_score))

        observations = [
            f"Macro score={macro_score:.3f} with regime={regime_suggestion}",
            f"Sentiment score={sentiment_score:.3f}",
            f"Event signal={event_signal:.3f}",
        ]

        inferences: List[str] = []
        if aggregate_score > 0.2:
            inferences.append("Overall positive environment; constructive risk stance justified.")
        elif aggregate_score < -0.2:
            inferences.append("Overall negative environment; defensive positioning warranted.")
        else:
            inferences.append("Mixed signals; maintain balanced, risk-aware stance.")

        reasoning_chain = observations + [
            f"Aggregate assessment score={aggregate_score:.3f}",
        ]

        signals: Dict[str, Any] = {
            "macro_score": macro_score,
            "regime_suggestion": regime_suggestion,
            "sentiment_score": sentiment_score,
            "sentiment_item_scores": per_item_scores,
            "event_signal": event_signal,
            "aggregate_score": aggregate_score,
        }

        return ReasoningResult(
            observations=observations,
            inferences=inferences,
            confidence=confidence,
            signals=signals,
            reasoning_chain=reasoning_chain,
        )

    def act(self, reasoning_result: ReasoningResult) -> Action:
        score = float(reasoning_result.signals.get("aggregate_score", 0.0))

        if score > 0.2:
            action_type = ActionType.BUY
            summary = "AnalystAgent indicates supportive backdrop; modestly increase risk exposure."
        elif score < -0.2:
            action_type = ActionType.SELL
            summary = "AnalystAgent indicates deteriorating backdrop; reduce risk exposure."
        else:
            action_type = ActionType.HOLD
            summary = "AnalystAgent sees mixed environment; maintain current exposure."

        quantity = float(abs(score))

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
        """Placeholder learning: track forecast accuracy for calibration."""

        forecast_return = float(outcome.get("forecast_return", 0.0))
        realized_return = float(outcome.get("realized_return", 0.0))
        error = realized_return - forecast_return

        confidence_adjustment = float(-np.tanh(abs(error)))
        lessons = [
            f"Forecast error of {error:.4f} observed; adjusting confidence calibration.",
        ]

        return LearningUpdate(
            parameter_changes={},
            confidence_adjustment=confidence_adjustment,
            lessons=lessons,
        )

    def explain(self, action: Action) -> Explanation:
        summary = (
            "AnalystAgent synthesized macro, sentiment, and event information to "
            f"recommend action={action.action_type} with intensity={action.quantity:.2f}."
        )

        reasoning_chain = [
            "Macro indicators drive medium-term regime assessment.",
            "News flow keywords provide a coarse sentiment proxy.",
            "Event modeling captures discrete catalysts and their expected impact.",
        ]

        data_sources = [
            "Macro indicator time series",
            "News headlines and articles",
            "Event calendars (earnings, central bank decisions, geopolitical events)",
        ]

        confidence_breakdown = {"aggregate_score": float(action.quantity)}

        risk_justification = (
            "Analyst-driven recommendations are directional signals; final risk "
            "sizing should be coordinated with dedicated risk and portfolio agents."
        )

        return Explanation(
            summary=summary,
            reasoning_chain=reasoning_chain,
            data_sources=data_sources,
            risk_justification=risk_justification,
            confidence_breakdown=confidence_breakdown,
        )

    def get_state(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "config": self.config}
