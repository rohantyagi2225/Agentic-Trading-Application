"""Multi-Agent Voting and Negotiation System.

This module implements voting mechanisms for agent coordination,
including weighted voting, veto capabilities, and iterative negotiation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class VoteSession:
    """A voting session for agent coordination.

    Attributes
    ----------
    session_id : str
        Unique identifier for this voting session.
    topic : str
        Description of what is being voted on.
    proposal : dict
        The proposal being voted on.
    options : list of str
        Available voting options.
    votes : dict
        Maps voter IDs to their vote details.
    status : str
        Current status: open, closed, or vetoed.
    result : str or None
        Final result if status is closed.
    created_at : datetime
        When the session was created.
    deadline : datetime
        Voting deadline.
    initiator : str
        Agent that initiated the vote.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    proposal: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=lambda: ["approve", "reject", "modify"])
    votes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: str = "open"  # open, closed, vetoed
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    initiator: str = ""

    def is_expired(self) -> bool:
        """Check if the voting deadline has passed."""
        if self.deadline is None:
            return False
        return datetime.utcnow() > self.deadline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "proposal": self.proposal,
            "options": self.options,
            "votes": self.votes,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "initiator": self.initiator,
        }


@dataclass
class VoteResult:
    """Result of a vote tally.

    Attributes
    ----------
    decision : str
        The winning option or "no_consensus".
    vote_counts : dict
        Raw vote counts per option.
    weighted_scores : dict
        Weighted scores per option.
    consensus_reached : bool
        Whether consensus threshold was met.
    vetoed : bool
        Whether the vote was vetoed.
    veto_by : str or None
            Agent that vetoed if vetoed is True.
    details : list of dict
            Detailed vote information per voter.
    session_id : str
            ID of the voting session.
    timestamp : datetime
            When the result was computed.
    """

    decision: str = ""
    vote_counts: Dict[str, int] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    consensus_reached: bool = False
    vetoed: bool = False
    veto_by: Optional[str] = None
    details: List[Dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision": self.decision,
            "vote_counts": self.vote_counts,
            "weighted_scores": self.weighted_scores,
            "consensus_reached": self.consensus_reached,
            "vetoed": self.vetoed,
            "veto_by": self.veto_by,
            "details": self.details,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class NegotiationResult:
    """Result of a negotiation process.

    Attributes
    ----------
    final_decision : str
        Final decision after negotiation.
    rounds_used : int
        Number of negotiation rounds performed.
    modifications : list of dict
        List of modifications made during negotiation.
    consensus_history : list of float
        Consensus scores from each round.
    final_proposal : dict
        The final modified proposal.
    session_id : str
        ID of the original voting session.
    timestamp : datetime
        When the negotiation completed.
    """

    final_decision: str = ""
    rounds_used: int = 0
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    consensus_history: List[float] = field(default_factory=list)
    final_proposal: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "final_decision": self.final_decision,
            "rounds_used": self.rounds_used,
            "modifications": self.modifications,
            "consensus_history": self.consensus_history,
            "final_proposal": self.final_proposal,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class VotingMechanism:
    """Multi-agent voting mechanism with weighted voting and negotiation.

    Supports configurable consensus thresholds, veto powers for specific
    agents, and iterative negotiation when initial votes don't reach
    consensus.

    Attributes
    ----------
    config : dict
        Configuration parameters.
    sessions : dict
        Active and completed voting sessions.
    """

    DEFAULT_CONFIG = {
        "consensus_threshold": 0.6,
        "allow_veto": True,
        "veto_agents": ["risk_manager"],
        "default_weights": {
            "analyst": 1.0,
            "trader": 1.0,
            "risk_manager": 1.5,
            "portfolio_manager": 1.2,
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the voting mechanism.

        Parameters
        ----------
        config : dict, optional
            Configuration overrides for default settings.
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.sessions: Dict[str, VoteSession] = {}

    def create_vote(
        self,
        topic: str,
        proposal: Dict[str, Any],
        options: Optional[List[str]] = None,
        initiator: str = "",
        deadline_seconds: int = 30,
    ) -> VoteSession:
        """Create a new voting session.

        Parameters
        ----------
        topic : str
            Description of what is being voted on.
        proposal : dict
            The proposal details.
        options : list of str, optional
            Voting options (default: ["approve", "reject", "modify"]).
        initiator : str
            Agent initiating the vote.
        deadline_seconds : int
            Seconds until voting deadline.

        Returns
        -------
        VoteSession
            The created voting session.
        """
        session = VoteSession(
            topic=topic,
            proposal=proposal,
            options=options or ["approve", "reject", "modify"],
            initiator=initiator,
            deadline=datetime.utcnow() + timedelta(seconds=deadline_seconds),
        )
        self.sessions[session.session_id] = session
        return session

    def cast_vote(
        self,
        session_id: str,
        voter: str,
        choice: str,
        confidence: float,
        reasoning: str = "",
        weight: float = 1.0,
    ) -> bool:
        """Cast a vote in a session.

        Parameters
        ----------
        session_id : str
            ID of the voting session.
        voter : str
            ID of the voting agent.
        choice : str
            Selected option.
        confidence : float
            Confidence level 0-1.
        reasoning : str
            Reasoning for the vote.
        weight : float
            Vote weight (overrides default if provided).

        Returns
        -------
        bool
            True if vote was accepted, False otherwise.
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if session.status != "open":
            return False

        if session.is_expired():
            session.status = "closed"
            return False

        if choice not in session.options:
            return False

        # Use default weight if not specified
        if weight == 1.0 and voter in self.config["default_weights"]:
            weight = self.config["default_weights"][voter]

        session.votes[voter] = {
            "choice": choice,
            "confidence": confidence,
            "reasoning": reasoning,
            "weight": weight,
            "timestamp": datetime.utcnow(),
        }

        return True

    def tally_votes(self, session_id: str) -> Optional[VoteResult]:
        """Tally votes and determine the result.

        Parameters
        ----------
        session_id : str
            ID of the voting session.

        Returns
        -------
        VoteResult or None
            The vote result, or None if session not found.
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check for expiration
        if session.is_expired() and session.status == "open":
            session.status = "closed"

        # Initialize result
        result = VoteResult(
            session_id=session_id,
            vote_counts={option: 0 for option in session.options},
            weighted_scores={option: 0.0 for option in session.options},
        )

        # Process votes
        veto_agents = set(self.config.get("veto_agents", []))
        allow_veto = self.config.get("allow_veto", True)

        for voter, vote in session.votes.items():
            choice = vote["choice"]
            confidence = vote["confidence"]
            weight = vote["weight"]

            # Count raw votes
            result.vote_counts[choice] = result.vote_counts.get(choice, 0) + 1

            # Calculate weighted score
            weighted_vote = confidence * weight
            result.weighted_scores[choice] = (
                result.weighted_scores.get(choice, 0.0) + weighted_vote
            )

            # Record details
            result.details.append(
                {
                    "voter": voter,
                    "choice": choice,
                    "confidence": confidence,
                    "weight": weight,
                    "weighted_vote": weighted_vote,
                    "reasoning": vote.get("reasoning", ""),
                }
            )

            # Check for veto
            if allow_veto and voter in veto_agents and choice == "reject":
                result.vetoed = True
                result.veto_by = voter

        # Determine winner
        if result.vetoed:
            result.decision = "rejected"
            result.consensus_reached = False
        elif result.weighted_scores:
            # Find option with highest weighted score
            max_score = max(result.weighted_scores.values())
            total_score = sum(result.weighted_scores.values())

            if total_score > 0:
                consensus_ratio = max_score / total_score
                result.consensus_reached = (
                    consensus_ratio >= self.config["consensus_threshold"]
                )

                # Get winning option
                for option, score in result.weighted_scores.items():
                    if score == max_score:
                        result.decision = option
                        break
            else:
                result.decision = "no_consensus"
                result.consensus_reached = False
        else:
            result.decision = "no_votes"
            result.consensus_reached = False

        # Update session
        session.status = "closed"
        session.result = result.decision

        return result

    def get_session(self, session_id: str) -> Optional[VoteSession]:
        """Get a voting session by ID.

        Parameters
        ----------
        session_id : str
            The session ID.

        Returns
        -------
        VoteSession or None
            The session if found.
        """
        return self.sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        """Manually close a voting session.

        Parameters
        ----------
        session_id : str
            The session ID.

        Returns
        -------
        bool
            True if session was closed, False if not found.
        """
        if session_id in self.sessions:
            self.sessions[session_id].status = "closed"
            return True
        return False

    def negotiate(
        self, session: VoteSession, rounds: int = 3
    ) -> NegotiationResult:
        """Perform iterative negotiation to reach consensus.

        If initial voting doesn't reach consensus, this method
        collects modification suggestions from agents and creates
        revised proposals for re-voting.

        Parameters
        ----------
        session : VoteSession
            The voting session to negotiate.
        rounds : int
            Maximum number of negotiation rounds.

        Returns
        -------
        NegotiationResult
            The final negotiation result.
        """
        result = NegotiationResult(session_id=session.session_id)
        result.final_proposal = dict(session.proposal)

        current_proposal = dict(session.proposal)

        for round_num in range(rounds):
            # Tally current votes
            vote_result = self.tally_votes(session.session_id)

            if vote_result is None:
                break

            # Record consensus level
            total_weighted = sum(vote_result.weighted_scores.values())
            max_weighted = max(vote_result.weighted_scores.values()) if vote_result.weighted_scores else 0
            consensus_level = max_weighted / total_weighted if total_weighted > 0 else 0
            result.consensus_history.append(consensus_level)

            # Check if consensus reached
            if vote_result.consensus_reached and not vote_result.vetoed:
                result.final_decision = vote_result.decision
                result.rounds_used = round_num + 1
                return result

            # Check if vetoed (can't negotiate past veto)
            if vote_result.vetoed:
                result.final_decision = "rejected"
                result.rounds_used = round_num + 1
                return result

            # Collect modifications from "modify" voters
            modifications = []
            for voter, vote in session.votes.items():
                if vote["choice"] == "modify":
                    mod = {
                        "voter": voter,
                        "reasoning": vote.get("reasoning", ""),
                        "suggested_changes": self._extract_suggestions(
                            vote.get("reasoning", "")
                        ),
                    }
                    modifications.append(mod)

            result.modifications.extend(modifications)

            # Create new proposal incorporating modifications
            if modifications:
                current_proposal = self._merge_modifications(
                    current_proposal, modifications
                )

                # Create new voting session for revised proposal
                new_session = self.create_vote(
                    topic=f"{session.topic} (Revision {round_num + 1})",
                    proposal=current_proposal,
                    options=session.options,
                    initiator=session.initiator,
                    deadline_seconds=30,
                )
                session = new_session
                result.final_proposal = dict(current_proposal)
            else:
                # No modifications suggested, can't improve
                break

        # Final tally
        final_vote = self.tally_votes(session.session_id)
        if final_vote:
            result.final_decision = final_vote.decision
        else:
            result.final_decision = "no_consensus"

        result.rounds_used = len(result.consensus_history)
        return result

    def _extract_suggestions(self, reasoning: str) -> Dict[str, Any]:
        """Extract modification suggestions from reasoning text.

        This is a simple implementation that looks for key phrases.
        In production, this could use NLP for better extraction.

        Parameters
        ----------
        reasoning : str
            The reasoning text.

        Returns
        -------
        dict
            Extracted suggestions.
        """
        suggestions = {}
        reasoning_lower = reasoning.lower()

        # Look for quantity adjustments
        if "reduce" in reasoning_lower or "decrease" in reasoning_lower:
            suggestions["adjustment_direction"] = "reduce"
        elif "increase" in reasoning_lower:
            suggestions["adjustment_direction"] = "increase"

        # Look for risk-related suggestions
        if "risk" in reasoning_lower:
            suggestions["concern"] = "risk"
        if "stop loss" in reasoning_lower or "stoploss" in reasoning_lower:
            suggestions["add_stop_loss"] = True

        return suggestions

    def _merge_modifications(
        self, proposal: Dict[str, Any], modifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge modification suggestions into a proposal.

        Parameters
        ----------
        proposal : dict
            The original proposal.
        modifications : list of dict
            List of modifications.

        Returns
        -------
        dict
            The modified proposal.
        """
        new_proposal = dict(proposal)

        # Count suggestion types
        reduce_count = sum(
            1
            for m in modifications
            if m.get("suggested_changes", {}).get("adjustment_direction") == "reduce"
        )
        increase_count = sum(
            1
            for m in modifications
            if m.get("suggested_changes", {}).get("adjustment_direction") == "increase"
        )
        risk_concerns = sum(
            1 for m in modifications if m.get("suggested_changes", {}).get("concern") == "risk"
        )
        stop_loss_suggestions = sum(
            1 for m in modifications if m.get("suggested_changes", {}).get("add_stop_loss")
        )

        # Apply modifications based on majority
        if "quantity" in new_proposal:
            if reduce_count > increase_count:
                new_proposal["quantity"] = new_proposal["quantity"] * 0.7
            elif increase_count > reduce_count:
                new_proposal["quantity"] = new_proposal["quantity"] * 1.3

        if risk_concerns > 0 and "risk_adjustment" not in new_proposal:
            new_proposal["risk_adjustment"] = "reduced"
            if "expected_risk" in new_proposal:
                new_proposal["expected_risk"] = new_proposal["expected_risk"] * 0.8

        if stop_loss_suggestions > 0 and "stop_loss" in new_proposal:
            if new_proposal["stop_loss"] is None:
                # Add a default stop loss
                if "price" in new_proposal and "action" in new_proposal:
                    price = new_proposal["price"]
                    action = new_proposal["action"]
                    if action == "BUY":
                        new_proposal["stop_loss"] = price * 0.95
                    else:
                        new_proposal["stop_loss"] = price * 1.05

        new_proposal["negotiation_rounds"] = new_proposal.get("negotiation_rounds", 0) + 1

        return new_proposal

    def get_all_sessions(self) -> List[VoteSession]:
        """Get all voting sessions.

        Returns
        -------
        list of VoteSession
            All sessions, both active and closed.
        """
        return list(self.sessions.values())

    def get_active_sessions(self) -> List[VoteSession]:
        """Get active (open) voting sessions.

        Returns
        -------
        list of VoteSession
            Sessions with status "open".
        """
        return [s for s in self.sessions.values() if s.status == "open"]

    def clear_sessions(self) -> None:
        """Clear all voting sessions."""
        self.sessions.clear()
