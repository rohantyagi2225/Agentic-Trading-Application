"""Coordination Orchestrator for Multi-Agent Trading.

This module provides the main AgentCoordinator class that orchestrates
trading cycles across multiple specialized agents using shared memory,
voting, and structured workflows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from FinAgents.research.coordination.blackboard import Blackboard, BlackboardEntry, Section
from FinAgents.research.coordination.protocols import (
    Message,
    MessageBus,
    MessageType,
    PortfolioReallocation,
    RiskAssessment,
    TradeProposal,
    VoteRequest,
    VoteResponse,
)
from FinAgents.research.coordination.voting import (
    NegotiationResult,
    VoteResult,
    VoteSession,
    VotingMechanism,
)
from FinAgents.research.domain_agents.base_agent import (
    Action,
    ActionType,
    MarketContext,
    MarketData,
    ReasoningResult,
    ResearchAgent,
)
from FinAgents.research.domain_agents.analyst_agent import AnalystAgent
from FinAgents.research.domain_agents.portfolio_manager_agent import PortfolioManagerAgent
from FinAgents.research.domain_agents.risk_manager_agent import RiskManagerAgent
from FinAgents.research.domain_agents.trader_agent import TraderAgent


@dataclass
class TradingCycleResult:
    """Result of a complete trading coordination cycle.

    Attributes
    ----------
    final_decision : str
        The final decision (execute, reject, modify, etc.).
    executed_trades : list of dict
        Trades that were executed.
    agent_interactions : list of dict
        Full log of all agent interactions.
    reasoning_chains : dict
        Per-agent reasoning chains.
    vote_results : list of VoteResult
            Results from any voting that occurred.
    performance_snapshot : dict
            Performance metrics at cycle completion.
    cycle_id : str
            Unique identifier for this cycle.
    timestamp : datetime
            When the cycle completed.
    """

    final_decision: str = ""
    executed_trades: List[Dict[str, Any]] = field(default_factory=list)
    agent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chains: Dict[str, List[str]] = field(default_factory=dict)
    vote_results: List[VoteResult] = field(default_factory=list)
    performance_snapshot: Dict[str, Any] = field(default_factory=dict)
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cycle_id": self.cycle_id,
            "final_decision": self.final_decision,
            "executed_trades": self.executed_trades,
            "agent_interactions": self.agent_interactions,
            "reasoning_chains": self.reasoning_chains,
            "vote_results": [vr.to_dict() for vr in self.vote_results],
            "performance_snapshot": self.performance_snapshot,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class AgentCoordinator:
    """Main orchestrator for coordinating multiple trading agents.

    The coordinator manages a blackboard for shared memory, a message bus
    for agent communication, and a voting mechanism for collective decisions.
    It implements a structured workflow for trading cycles.

    Attributes
    ----------
    agents : dict
        Maps role names to agent instances.
    blackboard : Blackboard
        Shared memory system.
    message_bus : MessageBus
        Agent communication system.
    voting : VotingMechanism
        Voting and negotiation system.
    config : dict
        Configuration parameters.
    _cycle_history : list
        History of completed trading cycles.
    """

    # Default access control per role
    DEFAULT_ACCESS = {
        "analyst": {Section.MARKET_STATE},
        "trader": {Section.PROPOSALS},
        "risk_manager": {Section.RISK_ASSESSMENTS},
        "portfolio_manager": {Section.PORTFOLIO_STATE, Section.DECISIONS},
    }

    def __init__(
        self,
        agents: Dict[str, ResearchAgent],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the coordinator with agents and configuration.

        Parameters
        ----------
        agents : dict
            Maps role names to agent instances. Expected keys:
            "analyst", "trader", "risk_manager", "portfolio_manager".
        config : dict, optional
            Configuration overrides.
        """
        self.agents = agents
        self.config = config or {}

        # Initialize coordination systems
        self.blackboard = Blackboard()
        self.message_bus = MessageBus()
        self.voting = VotingMechanism(self.config.get("voting_config"))

        # Set up access control
        self._setup_access_control()

        # Register agents with message bus
        for agent_id in agents:
            self.message_bus.register_agent(agent_id)

        # Cycle history
        self._cycle_history: List[TradingCycleResult] = []

    def _setup_access_control(self) -> None:
        """Configure blackboard write access for each agent role."""
        access_config = self.config.get("access_control", self.DEFAULT_ACCESS)

        for role, agent in self.agents.items():
            writable_sections = access_config.get(role, set())
            self.blackboard.set_access(agent.agent_id, writable_sections)

    def run_trading_cycle(
        self,
        market_data: MarketData,
        context: Optional[MarketContext] = None,
    ) -> TradingCycleResult:
        """Execute a complete trading coordination cycle.

        The workflow:
        1. Analyst analyzes market -> publishes MarketAssessment
        2. Trader reads assessment + own analysis -> proposes TradeProposal
        3. Risk Manager reads proposal -> publishes RiskAssessment
        4. If approved, Portfolio Manager checks allocation fit -> publishes PortfolioReallocation
        5. Voting: all agents vote on final execution
        6. If consensus reached, record execution. If not, negotiate up to 3 rounds
        7. Log full cycle to blackboard EXECUTION_LOG section

        Parameters
        ----------
        market_data : MarketData
            Current market data for analysis.
        context : MarketContext, optional
            Additional market context.

        Returns
        -------
        TradingCycleResult
            Complete result of the trading cycle.
        """
        cycle_result = TradingCycleResult()
        interactions: List[Dict[str, Any]] = []

        # Get agent references
        analyst = self.agents.get("analyst")
        trader = self.agents.get("trader")
        risk_manager = self.agents.get("risk_manager")
        portfolio_manager = self.agents.get("portfolio_manager")

        # Step 1: Analyst analyzes market
        if analyst:
            analyst_reasoning = analyst.reason(market_data, context)
            self._record_reasoning("analyst", analyst_reasoning, cycle_result)

            # Publish market assessment to blackboard
            assessment_entry = self.blackboard.write(
                section=Section.MARKET_STATE,
                key=f"assessment_{cycle_result.cycle_id}",
                value={
                    "aggregate_score": analyst_reasoning.signals.get("aggregate_score", 0),
                    "regime_suggestion": analyst_reasoning.signals.get("regime_suggestion", "neutral"),
                    "confidence": analyst_reasoning.confidence,
                    "inferences": analyst_reasoning.inferences,
                },
                author=analyst.agent_id,
                metadata={"cycle_id": cycle_result.cycle_id},
            )

            interactions.append({
                "step": 1,
                "agent": "analyst",
                "action": "market_analysis",
                "blackboard_entry": assessment_entry.key,
                "confidence": analyst_reasoning.confidence,
            })

        # Step 2: Trader proposes trade
        trade_proposal: Optional[TradeProposal] = None
        if trader:
            trader_reasoning = trader.reason(market_data, context)
            self._record_reasoning("trader", trader_reasoning, cycle_result)

            trader_action = trader.act(trader_reasoning)

            # Only create proposal if action is not HOLD
            if trader_action.action_type != ActionType.HOLD:
                trade_proposal = TradeProposal(
                    proposer=trader.agent_id,
                    symbol=trader_action.symbol or market_data.symbol,
                    action=trader_action.action_type.value,
                    quantity=trader_action.quantity,
                    price=trader_action.price or (
                        market_data.prices[-1].get("close", 0) if market_data.prices else 0
                    ),
                    confidence=trader_action.confidence,
                    reasoning=trader_action.reasoning_summary,
                    stop_loss=trader_action.stop_loss,
                    take_profit=trader_action.take_profit,
                    urgency="medium",
                    expected_return=trader_reasoning.signals.get("combined", 0),
                    expected_risk=abs(trader_reasoning.signals.get("combined", 0)) * 0.5,
                )

                # Publish to blackboard
                proposal_entry = self.blackboard.write(
                    section=Section.PROPOSALS,
                    key=f"proposal_{cycle_result.cycle_id}",
                    value=trade_proposal.to_dict(),
                    author=trader.agent_id,
                    metadata={"cycle_id": cycle_result.cycle_id},
                )

                # Broadcast message
                self.message_bus.broadcast(trade_proposal.to_message())

                interactions.append({
                    "step": 2,
                    "agent": "trader",
                    "action": "trade_proposal",
                    "proposal_id": trade_proposal.proposal_id,
                    "blackboard_entry": proposal_entry.key,
                    "action_type": trade_proposal.action,
                    "symbol": trade_proposal.symbol,
                })
            else:
                interactions.append({
                    "step": 2,
                    "agent": "trader",
                    "action": "hold",
                    "reason": "No strong signal",
                })

        # Step 3: Risk Manager assesses proposal
        risk_assessment: Optional[RiskAssessment] = None
        if risk_manager and trade_proposal:
            risk_reasoning = risk_manager.reason(market_data, context)
            self._record_reasoning("risk_manager", risk_reasoning, cycle_result)

            risk_action = risk_manager.act(risk_reasoning)

            # Determine approval based on risk score
            risk_score = risk_reasoning.signals.get("risk_score", 0.5)
            approved = risk_score < 1.0 and risk_action.action_type != ActionType.SELL

            risk_assessment = RiskAssessment(
                assessor=risk_manager.agent_id,
                proposal_id=trade_proposal.proposal_id,
                approved=approved,
                risk_score=risk_score,
                var_impact=risk_reasoning.signals.get("var_hist", 0),
                max_loss=risk_reasoning.signals.get("cvar", 0),
                concerns=risk_reasoning.inferences if not approved else [],
                conditions=["Monitor position size"] if approved else [],
            )

            # Publish to blackboard
            assessment_entry = self.blackboard.write(
                section=Section.RISK_ASSESSMENTS,
                key=f"risk_{cycle_result.cycle_id}",
                value=risk_assessment.to_dict(),
                author=risk_manager.agent_id,
                metadata={"cycle_id": cycle_result.cycle_id},
            )

            interactions.append({
                "step": 3,
                "agent": "risk_manager",
                "action": "risk_assessment",
                "assessment_id": risk_assessment.assessment_id,
                "approved": approved,
                "risk_score": risk_score,
                "blackboard_entry": assessment_entry.key,
            })

        # Step 4: Portfolio Manager checks allocation
        reallocation: Optional[PortfolioReallocation] = None
        if portfolio_manager and trade_proposal and (risk_assessment and risk_assessment.approved):
            pm_reasoning = portfolio_manager.reason(market_data, context)
            self._record_reasoning("portfolio_manager", pm_reasoning, cycle_result)

            pm_action = portfolio_manager.act(pm_reasoning)

            # Create reallocation if trades are needed
            target_weights = pm_reasoning.signals.get("target_weights", [])
            current_weights = pm_reasoning.signals.get("current_weights", [])
            trades = pm_reasoning.signals.get("trades", [])

            reallocation = PortfolioReallocation(
                manager=portfolio_manager.agent_id,
                current_allocation={"cash": 1.0 - sum(current_weights)},
                proposed_allocation={trade_proposal.symbol: trade_proposal.quantity},
                trades_required=trades,
                expected_improvement={
                    "sharpe_delta": 0.1,
                    "risk_delta": -risk_score * 0.1,
                    "cost": len(trades) * 0.001,
                },
                reasoning=pm_action.reasoning_summary,
            )

            # Publish to blackboard
            reallocation_entry = self.blackboard.write(
                section=Section.PORTFOLIO_STATE,
                key=f"portfolio_{cycle_result.cycle_id}",
                value=reallocation.to_dict(),
                author=portfolio_manager.agent_id,
                metadata={"cycle_id": cycle_result.cycle_id},
            )

            interactions.append({
                "step": 4,
                "agent": "portfolio_manager",
                "action": "portfolio_reallocation",
                "reallocation_id": reallocation.reallocation_id,
                "trades_count": len(trades),
                "blackboard_entry": reallocation_entry.key,
            })

        # Step 5 & 6: Voting and negotiation
        if trade_proposal:
            vote_session = self.voting.create_vote(
                topic=f"Execute trade: {trade_proposal.action} {trade_proposal.symbol}",
                proposal=trade_proposal.to_dict(),
                options=["execute", "reject", "modify"],
                initiator="coordinator",
                deadline_seconds=self.config.get("vote_timeout_seconds", 30),
            )

            # Collect votes from all agents
            for role, agent in self.agents.items():
                # Determine vote based on agent type and previous analysis
                choice, confidence, reasoning = self._determine_agent_vote(
                    role, agent, trade_proposal, risk_assessment
                )

                self.voting.cast_vote(
                    session_id=vote_session.session_id,
                    voter=agent.agent_id,
                    choice=choice,
                    confidence=confidence,
                    reasoning=reasoning,
                )

                interactions.append({
                    "step": 5,
                    "agent": role,
                    "action": "vote",
                    "choice": choice,
                    "confidence": confidence,
                })

            # Tally votes
            vote_result = self.voting.tally_votes(vote_session.session_id)
            if vote_result:
                cycle_result.vote_results.append(vote_result)

                # If no consensus, negotiate
                if not vote_result.consensus_reached and not vote_result.vetoed:
                    negotiation_result = self.voting.negotiate(vote_session, rounds=3)
                    cycle_result.final_decision = negotiation_result.final_decision

                    interactions.append({
                        "step": 6,
                        "action": "negotiation",
                        "rounds_used": negotiation_result.rounds_used,
                        "final_decision": negotiation_result.final_decision,
                    })
                else:
                    cycle_result.final_decision = (
                        "rejected" if vote_result.vetoed else vote_result.decision
                    )
            else:
                cycle_result.final_decision = "error"
        else:
            cycle_result.final_decision = "no_proposal"

        # Record executed trades if approved
        if cycle_result.final_decision == "execute" and trade_proposal:
            cycle_result.executed_trades.append({
                "symbol": trade_proposal.symbol,
                "action": trade_proposal.action,
                "quantity": trade_proposal.quantity,
                "price": trade_proposal.price,
                "timestamp": datetime.utcnow().isoformat(),
            })

        # Step 7: Log to execution log
        self.blackboard.write(
            section=Section.EXECUTION_LOG,
            key=f"cycle_{cycle_result.cycle_id}",
            value={
                "cycle_id": cycle_result.cycle_id,
                "final_decision": cycle_result.final_decision,
                "executed_trades": cycle_result.executed_trades,
                "timestamp": datetime.utcnow().isoformat(),
            },
            author="coordinator",
            metadata={"cycle_id": cycle_result.cycle_id},
        )

        # Set interactions and save result
        cycle_result.agent_interactions = interactions
        self._cycle_history.append(cycle_result)

        return cycle_result

    def _record_reasoning(
        self,
        agent_role: str,
        reasoning: ReasoningResult,
        cycle_result: TradingCycleResult,
    ) -> None:
        """Record an agent's reasoning chain to the cycle result."""
        cycle_result.reasoning_chains[agent_role] = reasoning.reasoning_chain

    def _determine_agent_vote(
        self,
        role: str,
        agent: ResearchAgent,
        proposal: TradeProposal,
        risk_assessment: Optional[RiskAssessment],
    ) -> tuple:
        """Determine how an agent should vote.

        Parameters
        ----------
        role : str
            The agent's role.
        agent : ResearchAgent
            The agent instance.
        proposal : TradeProposal
            The trade being voted on.
        risk_assessment : RiskAssessment or None
            The risk assessment if available.

        Returns
        -------
        tuple of (choice, confidence, reasoning)
        """
        if role == "risk_manager":
            # Risk manager votes based on risk assessment
            if risk_assessment:
                if risk_assessment.approved and risk_assessment.risk_score < 0.5:
                    return "execute", 0.8, "Risk within acceptable limits"
                elif risk_assessment.approved:
                    return "modify", 0.6, "Risk acceptable but consider reducing position"
                else:
                    return "reject", 0.9, f"Risk concerns: {risk_assessment.concerns}"
            return "modify", 0.5, "Insufficient risk data"

        elif role == "portfolio_manager":
            # Portfolio manager considers allocation fit
            return "execute", 0.7, "Trade aligns with portfolio objectives"

        elif role == "analyst":
            # Analyst votes based on market assessment
            return "execute", proposal.confidence, "Market conditions support trade"

        elif role == "trader":
            # Trader votes for their own proposal
            return "execute", proposal.confidence, "Original proposer"

        return "modify", 0.5, "Neutral stance"

    def get_cycle_explanation(self, cycle_result: TradingCycleResult) -> str:
        """Generate a human-readable narrative of a trading cycle.

        Parameters
        ----------
        cycle_result : TradingCycleResult
            The cycle to explain.

        Returns
        -------
        str
            Human-readable explanation.
        """
        lines = [
            f"Trading Cycle: {cycle_result.cycle_id}",
            f"Timestamp: {cycle_result.timestamp}",
            "",
            "=== Agent Reasoning ===",
        ]

        for agent, reasoning in cycle_result.reasoning_chains.items():
            lines.append(f"\n{agent.upper()}:")
            for step in reasoning:
                lines.append(f"  - {step}")

        lines.extend([
            "",
            "=== Interactions ===",
        ])

        for interaction in cycle_result.agent_interactions:
            step = interaction.get("step", "?")
            agent = interaction.get("agent", "unknown")
            action = interaction.get("action", "unknown")
            lines.append(f"Step {step}: {agent} - {action}")

        lines.extend([
            "",
            "=== Voting Results ===",
        ])

        for vote_result in cycle_result.vote_results:
            lines.append(f"Decision: {vote_result.decision}")
            lines.append(f"Consensus: {'Yes' if vote_result.consensus_reached else 'No'}")
            if vote_result.vetoed:
                lines.append(f"Vetoed by: {vote_result.veto_by}")
            lines.append(f"Vote counts: {vote_result.vote_counts}")

        lines.extend([
            "",
            f"=== Final Decision: {cycle_result.final_decision} ===",
        ])

        if cycle_result.executed_trades:
            lines.append("\nExecuted Trades:")
            for trade in cycle_result.executed_trades:
                lines.append(
                    f"  {trade['action']} {trade['quantity']} {trade['symbol']} @ {trade['price']}"
                )

        return "\n".join(lines)

    def get_agent_interactions_log(self) -> List[Dict[str, Any]]:
        """Get full history of all cycles' agent interactions.

        Returns
        -------
        list of dict
            All interactions from all cycles.
        """
        all_interactions = []
        for cycle in self._cycle_history:
            for interaction in cycle.agent_interactions:
                interaction_with_cycle = dict(interaction)
                interaction_with_cycle["cycle_id"] = cycle.cycle_id
                all_interactions.append(interaction_with_cycle)
        return all_interactions

    def get_cycle_history(self) -> List[TradingCycleResult]:
        """Get all completed trading cycles.

        Returns
        -------
        list of TradingCycleResult
            All cycle results.
        """
        return list(self._cycle_history)

    def get_blackboard(self) -> Blackboard:
        """Get the shared blackboard instance.

        Returns
        -------
        Blackboard
            The coordinator's blackboard.
        """
        return self.blackboard

    def get_message_bus(self) -> MessageBus:
        """Get the message bus instance.

        Returns
        -------
        MessageBus
            The coordinator's message bus.
        """
        return self.message_bus

    def get_voting_mechanism(self) -> VotingMechanism:
        """Get the voting mechanism instance.

        Returns
        -------
        VotingMechanism
            The coordinator's voting mechanism.
        """
        return self.voting
