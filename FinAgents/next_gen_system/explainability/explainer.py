"""
Explainability Engine for Multi-Agent Financial Decisions
==========================================================

Provides human-readable explanations for:
- Trading decisions
- Risk assessments
- Portfolio allocations
- Agent coordination outcomes

Features:
- Decision reasoning chains
- Data source attribution
- Counterfactual analysis
- Confidence calibration
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Structured explanation of a decision"""
    decision_id: str
    decision_type: str
    decision_summary: str
    primary_reasoning: List[str]
    supporting_evidence: List[Dict[str, Any]]
    data_sources: List[str]
    agent_contributions: Dict[str, str]
    confidence_score: float
    alternative_scenarios: List[Dict[str, Any]]
    risk_factors: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ExplainabilityEngine:
    """
    Comprehensive explainability system for AI-driven financial decisions
    
    Generates human-interpretable explanations for complex multi-agent
    decision-making processes.
    """
    
    def __init__(self, engine_id: str = "main"):
        """
        Initialize explainability engine
        
        Args:
            engine_id: Unique identifier
        """
        self.engine_id = engine_id
        
        # Explanation templates
        self.templates = {
            "trade_decision": self._trade_explanation_template,
            "risk_assessment": self._risk_explanation_template,
            "portfolio_allocation": self._portfolio_explanation_template,
            "coordination_outcome": self._coordination_explanation_template,
        }
        
        # Explanation history
        self.explanation_history: List[Explanation] = []
        
        logger.info(f"ExplainabilityEngine {engine_id} initialized")
    
    def explain_trade_decision(
        self,
        symbol: str,
        action: str,
        strength: float,
        trader_reasoning: List[str],
        risk_assessment: Dict[str, Any],
        analyst_view: Dict[str, Any],
        final_confidence: float,
    ) -> Explanation:
        """Generate explanation for a trade decision"""
        
        # Build primary reasoning chain
        primary_reasoning = [
            f"Trader agent identified {action} opportunity with {strength*100:.1f}% signal strength",
        ]
        
        # Add risk perspective
        if risk_assessment.get("risk_level") == "LOW":
            primary_reasoning.append("Risk assessment favorable - within tolerance limits")
        elif risk_assessment.get("risk_level") == "MEDIUM":
            primary_reasoning.append("Risk assessment moderate - acceptable risk-reward")
        else:
            primary_reasoning.append(f"Risk concern: {risk_assessment.get('risk_level', 'unknown')} level noted")
        
        # Add analyst perspective
        if analyst_view.get("overall_sentiment", 0.5) > 0.6:
            primary_reasoning.append("Analyst sentiment positive - supportive market conditions")
        elif analyst_view.get("overall_sentiment", 0.5) < 0.4:
            primary_reasoning.append("Analyst sentiment cautious - headwinds present")
        
        # Collect supporting evidence
        supporting_evidence = []
        
        if trader_reasoning:
            supporting_evidence.append({
                "source": "trader_analysis",
                "evidence": trader_reasoning,
                "weight": 0.4,
            })
        
        if "technical_score" in analyst_view:
            supporting_evidence.append({
                "source": "technical_analysis",
                "evidence": [f"Technical score: {analyst_view['technical_score']}/10"],
                "weight": 0.3,
            })
        
        if "sentiment" in analyst_view:
            supporting_evidence.append({
                "source": "sentiment_analysis",
                "evidence": [f"News sentiment: {analyst_view['sentiment']:.2f}"],
                "weight": 0.2,
            })
        
        # Identify data sources
        data_sources = [
            "price_data",
            "volume_data",
            "technical_indicators",
        ]
        
        if analyst_view.get("news_analyzed"):
            data_sources.append("news_sentiment")
        
        if risk_assessment.get("var_calculated"):
            data_sources.append("risk_models")
        
        # Agent contributions
        agent_contributions = {
            "trader_agent": f"Proposed {action} with {strength*100:.1f}% confidence",
            "risk_agent": f"Assessed risk as {risk_assessment.get('risk_level', 'unknown')}",
            "analyst_agent": f"Market view: {analyst_view.get('overall_sentiment', 'neutral')}",
        }
        
        # Generate alternative scenarios
        alternative_scenarios = [
            {
                "scenario": "Wait for better entry",
                "probability": 0.3,
                "outcome": "Potentially better price but risk missing move",
            },
            {
                "scenario": "Reduce position size",
                "probability": 0.2,
                "outcome": "Lower risk exposure but reduced profit potential",
            },
        ]
        
        # Identify risk factors
        risk_factors = []
        
        if risk_assessment.get("volatility", 0) > 0.3:
            risk_factors.append("High market volatility")
        
        if analyst_view.get("sentiment_dispersion", 0) > 0.5:
            risk_factors.append("Mixed market signals")
        
        if strength < 0.6:
            risk_factors.append("Moderate signal conviction")
        
        explanation = Explanation(
            decision_id=f"trade_{symbol}_{datetime.now().strftime('%H%M%S')}",
            decision_type="TRADE",
            decision_summary=f"{action} {symbol} with {strength*100:.1f}% conviction",
            primary_reasoning=primary_reasoning,
            supporting_evidence=supporting_evidence,
            data_sources=data_sources,
            agent_contributions=agent_contributions,
            confidence_score=final_confidence,
            alternative_scenarios=alternative_scenarios,
            risk_factors=risk_factors,
        )
        
        self.explanation_history.append(explanation)
        
        return explanation
    
    def _trade_explanation_template(self, decision_data: Dict[str, Any]) -> str:
        """Template for trade decision explanation"""
        template = f"""
TRADE DECISION EXPLANATION
==========================

Decision: {decision_data.get('decision_summary')}
Confidence: {decision_data.get('confidence_score')*100:.1f}%

PRIMARY REASONING:
{chr(10).join('- ' + r for r in decision_data.get('primary_reasoning', []))}

SUPPORTING EVIDENCE:
{chr(10).join('- [' + e['source'] + '] ' + ', '.join(e['evidence']) for e in decision_data.get('supporting_evidence', []))}

DATA SOURCES USED:
{chr(10).join('- ' + s for s in decision_data.get('data_sources', []))}

AGENT CONTRIBUTIONS:
{chr(10).join('- ' + agent + ': ' + contrib for agent, contrib in decision_data.get('agent_contributions', {}).items())}

RISK FACTORS:
{chr(10).join('- ' + r for r in decision_data.get('risk_factors', []))}

ALTERNATIVE SCENARIOS CONSIDERED:
{chr(10).join('- ' + s['scenario'] + f" (prob: {s['probability']*100:.0f}%): " + s['outcome'] for s in decision_data.get('alternative_scenarios', []))}
"""
        return template.strip()
    
    def _risk_explanation_template(self, decision_data: Dict[str, Any]) -> str:
        """Template for risk assessment explanation"""
        return f"Risk Assessment: {decision_data.get('risk_level')}\n" + \
               f"VaR: {decision_data.get('var', 0)*100:.2f}%\n" + \
               f"Key Concerns: {', '.join(decision_data.get('warnings', []))}"
    
    def _portfolio_explanation_template(self, decision_data: Dict[str, Any]) -> str:
        """Template for portfolio allocation explanation"""
        return f"Portfolio Optimization Result\n" + \
               f"Method: {decision_data.get('method')}\n" + \
               f"Expected Return: {decision_data.get('expected_return')*100:.2f}%\n" + \
               f"Expected Volatility: {decision_data.get('expected_volatility')*100:.2f}%\n" + \
               f"Sharpe Ratio: {decision_data.get('sharpe_ratio', 0):.2f}"
    
    def _coordination_explanation_template(self, decision_data: Dict[str, Any]) -> str:
        """Template for coordination outcome explanation"""
        return f"Coordinated Decision: {decision_data.get('decision')}\n" + \
               f"Support: {len(decision_data.get('supporting_agents', []))} agents\n" + \
               f"Opposition: {len(decision_data.get('opposing_agents', []))} agents\n" + \
               f"Consensus Confidence: {decision_data.get('confidence')*100:.1f}%"
    
    def generate_natural_language_explanation(self, explanation: Explanation) -> str:
        """Convert structured explanation to natural language"""
        template_func = self.templates.get(explanation.decision_type.lower(), 
                                           lambda x: "Explanation not available")
        
        decision_data = {
            "decision_summary": explanation.decision_summary,
            "confidence_score": explanation.confidence_score,
            "primary_reasoning": explanation.primary_reasoning,
            "supporting_evidence": explanation.supporting_evidence,
            "data_sources": explanation.data_sources,
            "agent_contributions": explanation.agent_contributions,
            "risk_factors": explanation.risk_factors,
            "alternative_scenarios": explanation.alternative_scenarios,
            "decision": explanation.decision_type,
            "supporting_agents": explanation.agent_contributions.keys(),
            "opposing_agents": [],
        }
        
        return template_func(decision_data)
    
    def perform_counterfactual_analysis(
        self,
        explanation: Explanation,
        what_if_scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze how decision would change under different conditions"""
        original_decision = explanation.decision_type
        
        # Modify based on scenario
        modified_confidence = explanation.confidence_score
        
        if "lower_sentiment" in what_if_scenario:
            modified_confidence -= 0.15
        
        if "higher_volatility" in what_if_scenario:
            modified_confidence -= 0.20
        
        if "stronger_signal" in what_if_scenario:
            modified_confidence += 0.25
        
        # Determine if decision would change
        decision_change = False
        new_decision = original_decision
        
        if modified_confidence < 0.3 and original_decision == "BUY":
            decision_change = True
            new_decision = "HOLD"
        elif modified_confidence > 0.7 and original_decision == "HOLD":
            decision_change = True
            new_decision = "BUY"
        
        return {
            "original_decision": original_decision,
            "new_decision": new_decision,
            "decision_changed": decision_change,
            "original_confidence": explanation.confidence_score,
            "modified_confidence": modified_confidence,
            "scenario_applied": what_if_scenario,
        }
    
    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated explanations"""
        if not self.explanation_history:
            return {"total_explanations": 0}
        
        by_type = {}
        avg_confidence = np.mean([e.confidence_score for e in self.explanation_history])
        
        for exp in self.explanation_history:
            if exp.decision_type not in by_type:
                by_type[exp.decision_type] = 0
            by_type[exp.decision_type] += 1
        
        return {
            "total_explanations": len(self.explanation_history),
            "by_type": by_type,
            "average_confidence": avg_confidence,
        }
    
    def export_explanation(self, explanation: Explanation, format: str = "json") -> str:
        """Export explanation in various formats"""
        if format == "json":
            return json.dumps({
                "decision_id": explanation.decision_id,
                "decision_type": explanation.decision_type,
                "decision_summary": explanation.decision_summary,
                "primary_reasoning": explanation.primary_reasoning,
                "supporting_evidence": explanation.supporting_evidence,
                "confidence_score": explanation.confidence_score,
                "timestamp": explanation.timestamp.isoformat(),
            }, indent=2)
        elif format == "text":
            return self.generate_natural_language_explanation(explanation)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get engine state"""
        return {
            "engine_id": self.engine_id,
            "explanations_generated": len(self.explanation_history),
            "statistics": self.get_explanation_statistics(),
        }


# Import numpy at module level
import numpy as np
