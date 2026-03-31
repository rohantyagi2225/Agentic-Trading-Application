"""
Multi-Agent Coordination Protocol
=================================

Advanced coordination mechanisms including:
- Shared blackboard system
- Negotiation protocols
- Consensus-based decision making
- Conflict resolution
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AgentProposal:
    """Agent proposal for coordination"""
    agent_id: str
    action_type: str  # TRADE, REBALANCE, HOLD, etc.
    symbol: str
    proposal_data: Dict[str, Any]
    confidence: float
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CoordinationDecision:
    """Final coordinated decision"""
    decision: str
    confidence: float
    supporting_agents: List[str]
    opposing_agents: List[str]
    reasoning: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BlackboardSystem:
    """Shared memory blackboard for agent communication"""
    
    def __init__(self):
        self.posts: List[Dict[str, Any]] = []
        self.global_state: Dict[str, Any] = {}
        self.agent_contributions: Dict[str, List[Dict[str, Any]]] = {}
    
    def post(self, agent_id: str, content: Dict[str, Any], priority: int = 0):
        """Post information to blackboard"""
        post = {
            "agent_id": agent_id,
            "content": content,
            "priority": priority,
            "timestamp": datetime.now(),
        }
        self.posts.append(post)
        
        if agent_id not in self.agent_contributions:
            self.agent_contributions[agent_id] = []
        self.agent_contributions[agent_id].append(post)
        
        logger.info(f"Blackboard post from {agent_id}: {content.get('type', 'unknown')}")
    
    def get_recent_posts(
        self,
        limit: int = 10,
        agent_id: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent posts from blackboard"""
        filtered = self.posts
        
        if agent_id:
            filtered = [p for p in filtered if p["agent_id"] == agent_id]
        if content_type:
            filtered = [p for p in filtered if p["content"].get("type") == content_type]
        
        return sorted(filtered, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def update_global_state(self, key: str, value: Any):
        """Update global state"""
        self.global_state[key] = value
        logger.debug(f"Global state updated: {key}")
    
    def get_global_state(self, key: Optional[str] = None) -> Any:
        """Get global state or specific key"""
        if key:
            return self.global_state.get(key)
        return self.global_state


class AgentCoordinator:
    """
    Advanced multi-agent coordination with negotiation and voting
    
    Coordinates decisions between Trader, Risk, Analyst, and Portfolio agents
    using consensus mechanisms.
    """
    
    def __init__(self, coordinator_id: str = "main"):
        """
        Initialize the coordinator
        
        Args:
            coordinator_id: Unique identifier
        """
        self.coordinator_id = coordinator_id
        
        # Registered agents
        self.agents: Dict[str, Any] = {}
        
        # Blackboard system
        self.blackboard = BlackboardSystem()
        
        # Voting weights
        self.voting_weights = {
            "trader": 0.3,
            "risk": 0.4,
            "analyst": 0.2,
            "portfolio": 0.1,
        }
        
        # Decision history
        self.decision_history: List[CoordinationDecision] = []
        
        logger.info(f"AgentCoordinator {coordinator_id} initialized")
    
    def register_agent(self, agent_id: str, agent_type: str, agent_instance: Any):
        """Register an agent with the coordinator"""
        self.agents[agent_id] = {
            "type": agent_type,
            "instance": agent_instance,
            "status": "active",
        }
        logger.info(f"Registered agent {agent_id} as {agent_type}")
    
    def coordinate_decision(
        self,
        proposals: List[AgentProposal],
        context: Dict[str, Any],
    ) -> CoordinationDecision:
        """
        Coordinate decision among multiple agents
        
        Args:
            proposals: List of agent proposals
            context: Decision context
            
        Returns:
            Coordinated decision
        """
        if not proposals:
            return CoordinationDecision(
                decision="HOLD",
                confidence=1.0,
                supporting_agents=[],
                opposing_agents=[],
                reasoning=["No proposals received"],
            )
        
        # Group by action type
        action_groups = {}
        for prop in proposals:
            if prop.action_type not in action_groups:
                action_groups[prop.action_type] = []
            action_groups[prop.action_type].append(prop)
        
        # Score each action
        action_scores = {}
        for action, props in action_groups.items():
            score = self._score_action(props, context)
            action_scores[action] = score
        
        # Select best action
        best_action = max(action_scores.items(), key=lambda x: x[1]["score"])
        
        # Determine support/opposition
        supporting = [p.agent_id for p in action_groups[best_action[0]] if p.confidence > 0.5]
        opposing = [p.agent_id for p in proposals if p.action_type != best_action[0]]
        
        # Collect reasoning
        all_reasoning = []
        for prop in action_groups[best_action[0]]:
            all_reasoning.extend([f"[{prop.agent_id}] {r}" for r in prop.reasoning])
        
        decision = CoordinationDecision(
            decision=best_action[0],
            confidence=best_action[1]["score"],
            supporting_agents=supporting,
            opposing_agents=opposing,
            reasoning=all_reasoning,
            metadata={
                "all_scores": action_scores,
                "total_proposals": len(proposals),
            },
        )
        
        self.decision_history.append(decision)
        
        # Post to blackboard
        self.blackboard.post(
            agent_id=self.coordinator_id,
            content={
                "type": "coordination_decision",
                "decision": decision.decision,
                "confidence": decision.confidence,
            },
            priority=1,
        )
        
        return decision
    
    def _score_action(
        self,
        proposals: List[AgentProposal],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score an action based on proposals and agent weights"""
        total_score = 0
        total_confidence = 0
        
        for prop in proposals:
            # Get agent type
            agent_info = self.agents.get(prop.agent_id, {})
            agent_type = agent_info.get("type", "unknown")
            
            # Get voting weight
            weight = self.voting_weights.get(agent_type, 0.25)
            
            # Calculate weighted score
            score = prop.confidence * weight
            total_score += score
            total_confidence += prop.confidence
        
        avg_confidence = total_confidence / len(proposals) if proposals else 0
        
        return {
            "score": total_score,
            "num_supporters": len(proposals),
            "avg_confidence": avg_confidence,
        }
    
    def negotiate_trade(
        self,
        trader_proposal: AgentProposal,
        risk_assessment: Dict[str, Any],
        analyst_view: Dict[str, Any],
    ) -> CoordinationDecision:
        """
        Specialized negotiation for trade approval
        
        Args:
            trader_proposal: Trade proposal from trader agent
            risk_assessment: Risk assessment from risk agent
            analyst_view: Market view from analyst agent
            
        Returns:
            Approved/rejected decision
        """
        proposals = [trader_proposal]
        
        # Risk agent can veto high-risk trades
        if risk_assessment.get("risk_level") in ["HIGH", "CRITICAL"]:
            proposals.append(AgentProposal(
                agent_id="risk_agent",
                action_type="REJECT",
                symbol=trader_proposal.symbol,
                proposal_data={"reason": "High risk level"},
                confidence=0.9,
                reasoning=["Risk level exceeds tolerance"],
            ))
        
        # Analyst view influences decision
        if analyst_view.get("overall_sentiment", 0.5) < 0.3:
            proposals.append(AgentProposal(
                agent_id="analyst_agent",
                action_type="REDUCE_SIZE",
                symbol=trader_proposal.symbol,
                proposal_data={"reduction_factor": 0.5},
                confidence=0.7,
                reasoning=["Negative market sentiment"],
            ))
        
        # Coordinate final decision
        context = {
            "risk_level": risk_assessment.get("risk_level"),
            "sentiment": analyst_view.get("overall_sentiment"),
            "market_regime": risk_assessment.get("market_regime"),
        }
        
        return self.coordinate_decision(proposals, context)
    
    def get_consensus_view(
        self,
        symbol: str,
        trader_signal: Optional[Dict[str, Any]] = None,
        risk_level: Optional[str] = None,
        analyst_rating: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get consensus view on a symbol"""
        views = []
        
        if trader_signal:
            views.append({
                "source": "trader",
                "signal": trader_signal.get("action"),
                "strength": trader_signal.get("strength", 0),
            })
        
        if risk_level:
            risk_score = {"LOW": 1.0, "MEDIUM": 0.5, "HIGH": 0.0}.get(risk_level, 0.5)
            views.append({
                "source": "risk",
                "signal": "POSITIVE" if risk_score > 0.5 else "NEGATIVE",
                "strength": risk_score,
            })
        
        if analyst_rating:
            views.append({
                "source": "analyst",
                "signal": "BULLISH" if analyst_rating > 0.6 else "BEARISH" if analyst_rating < 0.4 else "NEUTRAL",
                "strength": analyst_rating,
            })
        
        # Aggregate
        if not views:
            return {"consensus": "NEUTRAL", "confidence": 0.0, "views": views}
        
        avg_strength = np.mean([v["strength"] for v in views])
        
        if avg_strength > 0.6:
            consensus = "BULLISH"
        elif avg_strength < 0.4:
            consensus = "BEARISH"
        else:
            consensus = "NEUTRAL"
        
        return {
            "consensus": consensus,
            "confidence": avg_strength,
            "views": views,
            "num_sources": len(views),
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get coordinator state"""
        return {
            "coordinator_id": self.coordinator_id,
            "registered_agents": list(self.agents.keys()),
            "blackboard_posts": len(self.blackboard.posts),
            "decisions_made": len(self.decision_history),
            "voting_weights": self.voting_weights,
        }
