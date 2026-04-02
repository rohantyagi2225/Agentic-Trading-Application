"""
Explainable Decision Layer for Financial Trading Agents

This module provides transparent, human-readable explanations for every trading decision,
making AI-driven trading decisions interpretable and trustworthy.

Key Features:
- Trade rationale generation
- Signal attribution analysis
- Confidence breakdown
- Risk factor explanation
- Beginner-friendly translations
- Decision audit trail

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │          Explainable Decision Layer                     │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ Decision     │  │ Explanation  │  │ Attribution  │  │
    │  │ Logger       │  │ Generator    │  │ Analyzer     │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │         Natural Language Explanation Engine      │  │
    │  └──────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────┐
    │           Explanation Output                            │
    │  - WHY this trade was taken                             │
    │  - WHICH signals contributed                            │
    │  - HOW confident is the system                          │
    │  - WHAT are the key risks                               │
    │  - BEGINNER-FRIENDY translation                         │
    └─────────────────────────────────────────────────────────┘

Author: FinAgent Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from backend.market.signal_integrator import IntegratedSignal, SignalType, IndividualSignal
from backend.market.regime_detector import MarketRegime, RegimeResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("ExplainableAgent")


class DecisionType(Enum):
    """Type of trading decision"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"
    HEDGE = "hedge"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class TradeDecision:
    """Represents a complete trading decision with explanation"""
    decision_type: DecisionType
    symbol: str
    quantity: Optional[float]
    price: Optional[float]
    timestamp: datetime
    confidence: float
    
    # Explanation components
    primary_reason: str
    detailed_explanation: str
    signal_attribution: Dict[str, float]  # Which signals contributed
    regime_context: Optional[str]
    risk_factors: List[str]
    
    # Supporting evidence
    supporting_signals: List[Dict[str, Any]]
    contradictory_signals: List[Dict[str, Any]]
    
    # Confidence breakdown
    confidence_breakdown: Dict[str, float]
    
    # Beginner-friendly explanation
    simple_explanation: str
    analogy: Optional[str]
    
    # Audit trail
    decision_id: str = field(default_factory=lambda: f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000):04d}")
    model_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "primary_reason": self.primary_reason,
            "detailed_explanation": self.detailed_explanation,
            "signal_attribution": self.signal_attribution,
            "regime_context": self.regime_context,
            "risk_factors": self.risk_factors,
            "supporting_signals": self.supporting_signals,
            "contradictory_signals": self.contradictory_signals,
            "confidence_breakdown": self.confidence_breakdown,
            "simple_explanation": self.simple_explanation,
            "analogy": self.analogy,
            "model_version": self.model_version
        }


class ExplanationGenerator:
    """Generate human-readable explanations for trading decisions"""
    
    def __init__(self):
        self.explanation_templates = {
            DecisionType.BUY: [
                "Initiating LONG position based on {reason}",
                "BUY signal triggered by {reason}",
                "Opening position due to {reason}"
            ],
            DecisionType.SELL: [
                "Initiating SHORT position based on {reason}",
                "SELL signal triggered by {reason}",
                "Opening short position due to {reason}"
            ],
            DecisionType.HOLD: [
                "Maintaining current positions - {reason}",
                "NO ACTION - {reason}",
                "Standing pat because {reason}"
            ],
            DecisionType.STOP_LOSS: [
                "STOP LOSS triggered at {price} - {reason}",
                "Exiting position to limit losses - {reason}"
            ],
            DecisionType.TAKE_PROFIT: [
                "TAKING PROFITS at {price} - {reason}",
                "Closing position to realize gains - {reason}"
            ]
        }
        
        self.confidence_phrases = {
            (0.8, 1.0): "Very High Confidence",
            (0.6, 0.8): "High Confidence",
            (0.4, 0.6): "Moderate Confidence",
            (0.2, 0.4): "Low Confidence",
            (0.0, 0.2): "Very Low Confidence"
        }
    
    def generate_explanation(
        self,
        decision: DecisionType,
        integrated_signal: IntegratedSignal,
        regime_result: Optional[RegimeResult],
        context: Dict[str, Any]
    ) -> Tuple[str, str, str]:
        """
        Generate three levels of explanation:
        1. Primary reason (concise)
        2. Detailed explanation (comprehensive)
        3. Simple explanation (beginner-friendly)
        """
        
        # ===== PRIMARY REASON (Concise) =====
        primary_reason = self._generate_primary_reason(decision, integrated_signal)
        
        # ===== DETAILED EXPLANATION (Comprehensive) =====
        detailed_explanation = self._generate_detailed_explanation(
            decision, integrated_signal, regime_result, context
        )
        
        # ===== SIMPLE EXPLANATION (Beginner-Friendly) =====
        simple_explanation = self._generate_simple_explanation(
            decision, integrated_signal, context
        )
        
        return primary_reason, detailed_explanation, simple_explanation
    
    def _generate_primary_reason(
        self,
        decision: DecisionType,
        signal: IntegratedSignal
    ) -> str:
        """Generate concise primary reason"""
        
        # Identify strongest contributing signal
        if signal.contributing_signals:
            strongest_signal = max(
                signal.contributing_signals,
                key=lambda s: s.strength * s.confidence
            )
            
            source_name = strongest_signal.source.value.replace('_', ' ')
            signal_type = strongest_signal.signal_type.value
            
            reason_map = {
                DecisionType.BUY: f"strong {source_name} showing {signal_type} momentum",
                DecisionType.SELL: f"weak {source_name} indicating {signal_type} pressure",
                DecisionType.HOLD: f"mixed signals from {source_name} - no clear direction",
                DecisionType.STOP_LOSS: "predefined stop-loss level reached",
                DecisionType.TAKE_PROFIT: "profit target achieved"
            }
            
            return reason_map.get(decision, "market conditions warrant action")
        
        return "system-generated signal"
    
    def _generate_detailed_explanation(
        self,
        decision: DecisionType,
        signal: IntegratedSignal,
        regime: Optional[RegimeResult],
        context: Dict[str, Any]
    ) -> str:
        """Generate comprehensive explanation with all details"""
        
        parts = []
        
        # 1. Decision summary
        if decision == DecisionType.BUY:
            parts.append(f"DECISION: Initiate LONG position")
        elif decision == DecisionType.SELL:
            parts.append(f"DECISION: Initiate SHORT position")
        elif decision == DecisionType.HOLD:
            parts.append(f"DECISION: Maintain current positions (HOLD)")
        
        # 2. Signal strength and confidence
        parts.append(
            f"SIGNAL STRENGTH: {signal.overall_strength:.1%} | "
            f"CONFIDENCE: {signal.overall_confidence:.1%}"
        )
        
        # 3. Source attribution
        if signal.contributing_signals:
            parts.append("\nCONTRIBUTING SIGNALS:")
            for sig in signal.contributing_signals:
                source = sig.source.value.replace('_', ' ').title()
                parts.append(
                    f"  • {source}: {sig.signal_type.value.upper()} "
                    f"(strength: {sig.strength:.1%}, confidence: {sig.confidence:.1%})"
                )
                
                # Add brief description
                if sig.description:
                    parts.append(f"    → {sig.description}")
        
        # 4. Agreement/Disagreement
        if signal.agreement_score > 0.7:
            parts.append(f"\n✓ STRONG CONSENSUS: {signal.agreement_score:.1%} of sources agree")
        elif signal.agreement_score > 0.4:
            parts.append(f"◐ MODERATE AGREEMENT: {signal.agreement_score:.1%} of sources align")
        else:
            parts.append(f"✗ MIXED SIGNALS: Sources disagree (agreement: {signal.agreement_score:.1%})")
        
        # 5. Regime context
        if regime:
            parts.append(f"\nMARKET REGIME: {regime.primary_regime.value.replace('_', ' ').title()}")
            parts.append(f"  • Regime Confidence: {regime.confidence:.1%}")
            parts.append(f"  • Risk Level: {regime.risk_level.upper()}")
            
            if regime.strategy_recommendations:
                parts.append(f"  • Recommended Strategy: {regime.strategy_recommendations[0]}")
        
        # 6. Key drivers
        bullish_factors = [
            sig.source.value.replace('_', ' ')
            for sig in signal.contributing_signals
            if sig.signal_type == SignalType.BULLISH
        ]
        bearish_factors = [
            sig.source.value.replace('_', ' ')
            for sig in signal.contributing_signals
            if sig.signal_type == SignalType.BEARISH
        ]
        
        if bullish_factors:
            parts.append(f"\nBULLISH DRIVERS: {', '.join(bullish_factors)}")
        if bearish_factors:
            parts.append(f"BEARISH DRIVERS: {', '.join(bearish_factors)}")
        
        # 7. Risks and caveats
        if signal.regime_context == 'high_volatility':
            parts.append("\n⚠️ RISK WARNING: High volatility environment - expect larger price swings")
        elif signal.regime_context == 'crash':
            parts.append("\n⚠️ EXTREME RISK: Market crash detected - defensive positioning recommended")
        
        return "\n".join(parts)
    
    def _generate_simple_explanation(
        self,
        decision: DecisionType,
        signal: IntegratedSignal,
        context: Dict[str, Any]
    ) -> str:
        """Generate beginner-friendly explanation"""
        
        # Use simple language and analogies
        if decision == DecisionType.BUY:
            base_explanation = "The system wants to BUY because multiple indicators suggest the price will go up."
        elif decision == DecisionType.SELL:
            base_explanation = "The system wants to SELL because signals indicate the price will likely decline."
        elif decision == DecisionType.HOLD:
            base_explanation = "The system recommends WAITING because market signals are unclear or conflicting."
        else:
            base_explanation = "The system is taking action based on current market conditions."
        
        # Add confidence context
        confidence_text = ""
        if signal.overall_confidence > 0.7:
            confidence_text = " The system is quite confident about this decision."
        elif signal.overall_confidence > 0.4:
            confidence_text = " The system has moderate confidence."
        else:
            confidence_text = " The system is uncertain, so proceed with caution."
        
        # Add agreement context
        if signal.agreement_score > 0.7:
            agreement_text = " Most data sources are telling the same story, which strengthens the signal."
        elif signal.agreement_score < 0.4:
            agreement_text = " Different data sources are giving mixed messages, making this a riskier call."
        else:
            agreement_text = ""
        
        # Combine
        simple_explanation = base_explanation + confidence_text + agreement_text
        
        return simple_explanation
    
    def generate_analogy(
        self,
        decision: DecisionType,
        signal: IntegratedSignal
    ) -> Optional[str]:
        """Generate an analogy to help understand the decision"""
        
        if signal.agreement_score > 0.8 and signal.overall_confidence > 0.7:
            return (
                "Think of this like a weather forecast: when multiple meteorologists, "
                "radar systems, and satellite data all predict rain, you bring an umbrella. "
                "Similarly, when technical, sentiment, and macro signals all align, we take a position."
            )
        elif signal.agreement_score < 0.4:
            return (
                "Imagine asking five friends for restaurant recommendations and getting five different answers. "
                "That's what conflicting signals look like - it doesn't mean none are right, but it's harder to decide."
            )
        elif decision == DecisionType.STOP_LOSS:
            return (
                "A stop-loss is like a fire alarm - it doesn't mean there's definitely a fire, "
                "but it's a predefined safety mechanism that tells you when to exit to protect yourself."
            )
        elif decision == DecisionType.TAKE_PROFIT:
            return (
                "Taking profits is like harvesting crops at the right time - "
                "you could wait for more growth, but locking in gains ensures you have something to show for your work."
            )
        
        return None


class SignalAttributionAnalyzer:
    """Analyze which signals contributed most to a decision"""
    
    def analyze_attribution(
        self,
        integrated_signal: IntegratedSignal
    ) -> Dict[str, float]:
        """Calculate attribution percentages for each signal source"""
        
        if not integrated_signal.contributing_signals:
            return {}
        
        # Calculate weighted contribution for each signal
        contributions = {}
        total_contribution = 0
        
        for sig in integrated_signal.contributing_signals:
            # Contribution = strength * confidence * weight
            source_weight = integrated_signal.source_weights.get(sig.source.value, 0.5)
            contribution = sig.strength * sig.confidence * source_weight
            
            contributions[sig.source.value] = abs(contribution)
            total_contribution += abs(contribution)
        
        # Normalize to percentages
        if total_contribution > 0:
            normalized_contributions = {
                source: (contribution / total_contribution) * 100
                for source, contribution in contributions.items()
            }
        else:
            normalized_contributions = {
                source: 100.0 / len(contributions)
                for source in contributions
            }
        
        return dict(sorted(normalized_contributions.items(), key=lambda x: x[1], reverse=True))
    
    def identify_contradictory_signals(
        self,
        integrated_signal: IntegratedSignal
    ) -> List[IndividualSignal]:
        """Identify signals that contradict the overall decision"""
        
        if not integrated_signal.contributing_signals:
            return []
        
        # Determine dominant direction
        dominant_type = integrated_signal.signal_type
        
        # Find contradictory signals
        contradictory = [
            sig for sig in integrated_signal.contributing_signals
            if sig.signal_type != dominant_type and sig.signal_type != SignalType.NEUTRAL
        ]
        
        return contradictory


class ConfidenceBreakdownAnalyzer:
    """Break down confidence into component factors"""
    
    def analyze_confidence(
        self,
        integrated_signal: IntegratedSignal,
        regime_result: Optional[RegimeResult]
    ) -> Dict[str, float]:
        """Decompose confidence into contributing factors"""
        
        breakdown = {}
        
        # 1. Signal consistency confidence
        breakdown['signal_consistency'] = integrated_signal.agreement_score
        
        # 2. Data quality confidence
        num_sources = len(integrated_signal.contributing_signals)
        breakdown['data_availability'] = min(num_sources / 4.0, 1.0)  # Ideal: 4+ sources
        
        # 3. Individual signal confidence (weighted average)
        if integrated_signal.contributing_signals:
            avg_signal_confidence = np.mean([
                sig.confidence for sig in integrated_signal.contributing_signals
            ])
            breakdown['signal_quality'] = avg_signal_confidence
        else:
            breakdown['signal_quality'] = 0.0
        
        # 4. Regime stability confidence
        if regime_result:
            breakdown['regime_clarity'] = regime_result.confidence
        else:
            breakdown['regime_clarity'] = 0.5  # Neutral
        
        # 5. Strength confidence
        breakdown['signal_strength'] = integrated_signal.overall_strength
        
        return breakdown


class ExplainableDecisionAgent:
    """
    Main class for generating explainable trading decisions
    
    Usage:
        agent = ExplainableDecisionAgent()
        
        decision = agent.make_decision(
            integrated_signal=signal,
            regime_result=regime,
            context={'symbol': 'AAPL', 'current_price': 150.0}
        )
        
        print(decision.detailed_explanation)
        print(decision.simple_explanation)
    """
    
    def __init__(self):
        self.explanation_generator = ExplanationGenerator()
        self.attribution_analyzer = SignalAttributionAnalyzer()
        self.confidence_analyzer = ConfidenceBreakdownAnalyzer()
        
        logger.info("✅ Explainable decision agent initialized")
    
    def make_decision(
        self,
        integrated_signal: IntegratedSignal,
        regime_result: Optional[RegimeResult],
        context: Dict[str, Any]
    ) -> TradeDecision:
        """
        Generate a complete explainable trading decision
        
        Args:
            integrated_signal: Fused signal from multiple sources
            regime_result: Current market regime detection result
            context: Additional context (symbol, price, current positions, etc.)
            
        Returns:
            TradeDecision: Complete decision with full explanation
        """
        
        # ===== 1. DETERMINE DECISION TYPE =====
        decision_type = self._map_signal_to_decision(integrated_signal, context)
        
        # ===== 2. GENERATE EXPLANATIONS =====
        primary_reason, detailed_explanation, simple_explanation = \
            self.explanation_generator.generate_explanation(
                decision_type, integrated_signal, regime_result, context
            )
        
        # ===== 3. ANALYZE SIGNAL ATTRIBUTION =====
        signal_attribution = self.attribution_analyzer.analyze_attribution(integrated_signal)
        
        # ===== 4. IDENTIFY CONTRADICTORY SIGNALS =====
        contradictory_signals = self.attribution_analyzer.identify_contradictory_signals(integrated_signal)
        
        # ===== 5. BREAK DOWN CONFIDENCE =====
        confidence_breakdown = self.confidence_analyzer.analyze_confidence(
            integrated_signal, regime_result
        )
        
        # ===== 6. IDENTIFY RISK FACTORS =====
        risk_factors = self._identify_risk_factors(integrated_signal, regime_result, context)
        
        # ===== 7. GENERATE ANALOGY =====
        analogy = self.explanation_generator.generate_analogy(decision_type, integrated_signal)
        
        # ===== 8. PREPARE SUPPORTING EVIDENCE =====
        supporting_signals = [
            sig.to_dict() for sig in integrated_signal.contributing_signals
            if sig.signal_type == integrated_signal.signal_type
        ]
        
        contradictory_evidence = [
            sig.to_dict() for sig in contradictory_signals
        ]
        
        # ===== 9. CREATE TRADE DECISION =====
        decision = TradeDecision(
            decision_type=decision_type,
            symbol=context.get('symbol', 'UNKNOWN'),
            quantity=context.get('quantity'),
            price=context.get('current_price'),
            timestamp=datetime.now(),
            confidence=integrated_signal.overall_confidence,
            primary_reason=primary_reason,
            detailed_explanation=detailed_explanation,
            signal_attribution=signal_attribution,
            regime_context=regime_result.primary_regime.value if regime_result else None,
            risk_factors=risk_factors,
            supporting_signals=supporting_signals,
            contradictory_signals=contradictory_evidence,
            confidence_breakdown=confidence_breakdown,
            simple_explanation=simple_explanation,
            analogy=analogy
        )
        
        logger.info(f"🎯 Generated explainable decision: {decision_type.value} for {decision.symbol}")
        
        return decision
    
    def _map_signal_to_decision(
        self,
        signal: IntegratedSignal,
        context: Dict[str, Any]
    ) -> DecisionType:
        """Map integrated signal to specific decision type"""
        
        # Check for stop-loss trigger
        if context.get('current_price') and context.get('stop_loss_price'):
            if context['current_price'] <= context['stop_loss_price']:
                return DecisionType.STOP_LOSS
        
        # Check for take-profit trigger
        if context.get('current_price') and context.get('take_profit_price'):
            if context['current_price'] >= context['take_profit_price']:
                return DecisionType.TAKE_PROFIT
        
        # Map signal type to decision
        if signal.signal_type == SignalType.BULLISH:
            if signal.overall_strength > 0.5 and signal.overall_confidence > 0.6:
                return DecisionType.BUY
            elif signal.overall_strength > 0.3:
                return DecisionType.INCREASE_POSITION
            else:
                return DecisionType.HOLD
        
        elif signal.signal_type == SignalType.BEARISH:
            if signal.overall_strength > 0.5 and signal.overall_confidence > 0.6:
                return DecisionType.SELL
            elif signal.overall_strength > 0.3:
                return DecisionType.REDUCE_POSITION
            else:
                return DecisionType.HOLD
        
        else:  # NEUTRAL
            return DecisionType.HOLD
    
    def _identify_risk_factors(
        self,
        signal: IntegratedSignal,
        regime: Optional[RegimeResult],
        context: Dict[str, Any]
    ) -> List[str]:
        """Identify and list risk factors"""
        
        risks = []
        
        # 1. Low confidence risk
        if signal.overall_confidence < 0.4:
            risks.append("Low signal confidence increases uncertainty")
        
        # 2. Conflicting signals risk
        if signal.agreement_score < 0.4:
            risks.append("Conflicting signals from different sources")
        
        # 3. Regime-specific risks
        if regime:
            if regime.primary_regime == MarketRegime.HIGH_VOLATILITY:
                risks.append("High volatility may cause larger-than-expected price swings")
            elif regime.primary_regime == MarketRegime.CRASH:
                risks.append("Market crash conditions - extreme downside risk")
            elif regime.primary_regime == MarketRegime.SIDEWAYS:
                risks.append("Range-bound market may result in whipsaw trades")
        
        # 4. Position-specific risks
        if context.get('current_position_size', 0) > context.get('normal_position_size', 1):
            risks.append("Position size larger than normal - increased exposure")
        
        # 5. Concentration risk
        if context.get('portfolio_concentration', 0) > 0.3:
            risks.append("High concentration in single asset increases portfolio risk")
        
        return risks
    
    def export_decision_log(self, decision: TradeDecision, format: str = 'json') -> str:
        """Export decision log in various formats"""
        
        if format == 'json':
            return json.dumps(decision.to_dict(), indent=2)
        elif format == 'text':
            return self._format_decision_as_text(decision)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_decision_as_text(self, decision: TradeDecision) -> str:
        """Format decision as readable text report"""
        
        lines = [
            "=" * 80,
            f"TRADING DECISION REPORT - {decision.decision_id}",
            "=" * 80,
            "",
            f"Symbol: {decision.symbol}",
            f"Decision: {decision.decision_type.value.upper()}",
            f"Timestamp: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Confidence: {decision.confidence:.1%}",
            "",
            "-" * 80,
            "PRIMARY REASON",
            "-" * 80,
            decision.primary_reason,
            "",
            "-" * 80,
            "DETAILED EXPLANATION",
            "-" * 80,
            decision.detailed_explanation,
            "",
            "-" * 80,
            "SIMPLE EXPLANATION (For Beginners)",
            "-" * 80,
            decision.simple_explanation,
            "",
            "-" * 80,
            "ANALOGY",
            "-" * 80,
            decision.analogy if decision.analogy else "N/A",
            "",
            "-" * 80,
            "RISK FACTORS",
            "-" * 80
        ]
        
        for i, risk in enumerate(decision.risk_factors, 1):
            lines.append(f"{i}. {risk}")
        
        lines.extend([
            "",
            "-" * 80,
            "SIGNAL ATTRIBUTION",
            "-" * 80
        ])
        
        for source, attribution in decision.signal_attribution.items():
            lines.append(f"{source}: {attribution:.1f}%")
        
        lines.extend([
            "",
            "-" * 80,
            "CONFIDENCE BREAKDOWN",
            "-" * 80
        ])
        
        for factor, value in decision.confidence_breakdown.items():
            lines.append(f"{factor}: {value:.1%}")
        
        lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(lines)


def create_explainable_agent() -> ExplainableDecisionAgent:
    """Factory function to create explainable decision agent"""
    return ExplainableDecisionAgent()
