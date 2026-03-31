"""Communication Protocol Definitions for Multi-Agent Coordination.

This module defines the message types, structures, and message bus
for agent-to-agent communication in the trading system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class MessageType(Enum):
    """Enumeration of message types for agent communication."""

    TRADE_PROPOSAL = auto()
    """A trade proposal from a trading agent."""

    RISK_ASSESSMENT = auto()
    """Risk evaluation from a risk manager agent."""

    PORTFOLIO_REALLOCATION = auto()
    """Portfolio reallocation recommendation."""

    MARKET_UPDATE = auto()
    """Market data or state update."""

    VOTE_REQUEST = auto()
    """Request for agents to vote on a decision."""

    VOTE_RESPONSE = auto()
    """Response to a vote request."""

    EXECUTION_ORDER = auto()
    """Order to execute a trade."""

    STATUS_UPDATE = auto()
    """General status update from an agent."""


@dataclass
class Message:
    """A message for agent-to-agent communication.

    Attributes
    ----------
    id : str
        Unique identifier for this message.
    type : MessageType
        The type of message.
    sender : str
        Agent ID of the sender.
    recipient : str
        Agent ID of the recipient, or "ALL" for broadcast.
    payload : dict
        Message-specific data.
    timestamp : datetime
        When the message was sent.
    priority : int
        Priority level 0-9 (0 is highest).
    requires_response : bool
        Whether the recipient should respond.
    correlation_id : str
        Links related messages (request/response pairs).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.STATUS_UPDATE
    sender: str = ""
    recipient: str = "ALL"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 5
    requires_response: bool = False
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a serializable dictionary."""
        return {
            "id": self.id,
            "type": self.type.name,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "priority": self.priority,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType[data.get("type", "STATUS_UPDATE")],
            sender=data.get("sender", ""),
            recipient=data.get("recipient", "ALL"),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
            priority=data.get("priority", 5),
            requires_response=data.get("requires_response", False),
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class TradeProposal:
    """A structured trade proposal from a trading agent.

    Attributes
    ----------
    proposal_id : str
        Unique identifier for this proposal.
    proposer : str
        Agent ID that created the proposal.
    symbol : str
        Ticker symbol to trade.
    action : str
        BUY or SELL.
    quantity : float
        Number of shares/contracts.
    price : float
        Target price or limit price.
    confidence : float
        Confidence level 0-1.
    reasoning : str
        Human-readable reasoning for the trade.
    stop_loss : float
        Stop loss price level.
    take_profit : float
        Take profit price level.
    urgency : str
        low, medium, or high urgency.
    expected_return : float
        Expected return estimate.
    expected_risk : float
        Expected risk estimate.
    timestamp : datetime
        When the proposal was created.
    """

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposer: str = ""
    symbol: str = ""
    action: str = "BUY"  # BUY or SELL
    quantity: float = 0.0
    price: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    urgency: str = "medium"  # low, medium, high
    expected_return: float = 0.0
    expected_risk: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "proposal_id": self.proposal_id,
            "proposer": self.proposer,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "urgency": self.urgency,
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def to_message(self, recipient: str = "ALL") -> Message:
        """Convert this proposal to a Message."""
        return Message(
            type=MessageType.TRADE_PROPOSAL,
            sender=self.proposer,
            recipient=recipient,
            payload=self.to_dict(),
            priority=1 if self.urgency == "high" else (3 if self.urgency == "medium" else 5),
        )


@dataclass
class RiskAssessment:
    """Risk evaluation from a risk manager agent.

    Attributes
    ----------
    assessment_id : str
        Unique identifier for this assessment.
    assessor : str
        Agent ID that performed the assessment.
    proposal_id : str
        ID of the trade proposal being assessed.
    approved : bool
        Whether the proposal is approved.
    risk_score : float
        Overall risk score 0-1.
    var_impact : float
        Value at Risk impact estimate.
    max_loss : float
        Maximum potential loss estimate.
    concerns : list of str
        List of risk concerns identified.
    modifications : dict or None
        Suggested modifications if not fully approved.
    conditions : list of str
        Conditions that must be met for approval.
    timestamp : datetime
        When the assessment was created.
    """

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessor: str = ""
    proposal_id: str = ""
    approved: bool = False
    risk_score: float = 0.0
    var_impact: float = 0.0
    max_loss: float = 0.0
    concerns: List[str] = field(default_factory=list)
    modifications: Optional[Dict[str, Any]] = None
    conditions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "assessment_id": self.assessment_id,
            "assessor": self.assessor,
            "proposal_id": self.proposal_id,
            "approved": self.approved,
            "risk_score": self.risk_score,
            "var_impact": self.var_impact,
            "max_loss": self.max_loss,
            "concerns": self.concerns,
            "modifications": self.modifications,
            "conditions": self.conditions,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def to_message(self, recipient: str = "ALL") -> Message:
        """Convert this assessment to a Message."""
        return Message(
            type=MessageType.RISK_ASSESSMENT,
            sender=self.assessor,
            recipient=recipient,
            payload=self.to_dict(),
            priority=2 if not self.approved else 4,
        )


@dataclass
class PortfolioReallocation:
    """Portfolio reallocation recommendation.

    Attributes
    ----------
    reallocation_id : str
        Unique identifier for this reallocation.
    manager : str
        Agent ID that created the recommendation.
    current_allocation : dict
        Current portfolio allocation (symbol -> weight).
    proposed_allocation : dict
        Proposed new allocation (symbol -> weight).
    trades_required : list of dict
        List of trades needed to achieve the new allocation.
    expected_improvement : dict
        Expected improvements (sharpe_delta, risk_delta, cost).
    reasoning : str
        Human-readable reasoning for the reallocation.
    timestamp : datetime
        When the reallocation was created.
    """

    reallocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    manager: str = ""
    current_allocation: Dict[str, float] = field(default_factory=dict)
    proposed_allocation: Dict[str, float] = field(default_factory=dict)
    trades_required: List[Dict[str, Any]] = field(default_factory=list)
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "reallocation_id": self.reallocation_id,
            "manager": self.manager,
            "current_allocation": self.current_allocation,
            "proposed_allocation": self.proposed_allocation,
            "trades_required": self.trades_required,
            "expected_improvement": self.expected_improvement,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def to_message(self, recipient: str = "ALL") -> Message:
        """Convert this reallocation to a Message."""
        return Message(
            type=MessageType.PORTFOLIO_REALLOCATION,
            sender=self.manager,
            recipient=recipient,
            payload=self.to_dict(),
            priority=3,
        )


@dataclass
class VoteRequest:
    """Request for agents to vote on a decision.

    Attributes
    ----------
    topic : str
        Description of what is being voted on.
    options : list of str
        Available voting options.
    deadline : datetime
        Voting deadline.
    quorum : int
        Minimum number of votes required.
    request_id : str
        Unique identifier for this vote request.
    requester : str
        Agent ID requesting the vote.
    context : dict
        Additional context for the vote.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester: str = ""
    topic: str = ""
    options: List[str] = field(default_factory=lambda: ["approve", "reject", "modify"])
    deadline: Optional[datetime] = None
    quorum: int = 2
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "requester": self.requester,
            "topic": self.topic,
            "options": self.options,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "quorum": self.quorum,
            "context": self.context,
        }

    def to_message(self, recipient: str = "ALL") -> Message:
        """Convert this vote request to a Message."""
        return Message(
            type=MessageType.VOTE_REQUEST,
            sender=self.requester,
            recipient=recipient,
            payload=self.to_dict(),
            requires_response=True,
            priority=2,
            correlation_id=self.request_id,
        )


@dataclass
class VoteResponse:
    """Response to a vote request.

    Attributes
    ----------
    response_id : str
        Unique identifier for this response.
    vote_request_id : str
        ID of the vote request being responded to.
    voter : str
        Agent ID casting the vote.
    choice : str
        Selected option.
    confidence : float
        Confidence in the vote 0-1.
    reasoning : str
        Reasoning for the vote.
    timestamp : datetime
        When the vote was cast.
    """

    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vote_request_id: str = ""
    voter: str = ""
    choice: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "response_id": self.response_id,
            "vote_request_id": self.vote_request_id,
            "voter": self.voter,
            "choice": self.choice,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def to_message(self, recipient: str = "ALL", correlation_id: Optional[str] = None) -> Message:
        """Convert this vote response to a Message."""
        return Message(
            type=MessageType.VOTE_RESPONSE,
            sender=self.voter,
            recipient=recipient,
            payload=self.to_dict(),
            priority=3,
            correlation_id=correlation_id or self.vote_request_id,
        )


class MessageBus:
    """Message bus for agent-to-agent communication.

    Provides asynchronous messaging capabilities with support for
    direct messaging, broadcasting, and response correlation.

    Attributes
    ----------
    _inboxes : dict
        Maps agent IDs to their message queues.
    _correlation_index : dict
        Indexes messages by correlation ID for response tracking.
    """

    def __init__(self) -> None:
        """Initialize the message bus."""
        self._inboxes: Dict[str, List[Message]] = {}
        self._correlation_index: Dict[str, List[Message]] = {}

    def register_agent(self, agent_id: str) -> None:
        """Register a new agent to receive messages.

        Parameters
        ----------
        agent_id : str
            The agent to register.
        """
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = []

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent and clear its inbox.

        Parameters
        ----------
        agent_id : str
            The agent to unregister.

        Returns
        -------
        bool
            True if agent was registered, False otherwise.
        """
        if agent_id in self._inboxes:
            del self._inboxes[agent_id]
            return True
        return False

    def send(self, message: Message) -> None:
        """Send a message to a specific recipient.

        Parameters
        ----------
        message : Message
            The message to send.

        Raises
        ------
        ValueError
            If recipient is "ALL" (use broadcast instead).
        """
        if message.recipient == "ALL":
            raise ValueError("Use broadcast() for messages to ALL")

        # Auto-register if needed
        if message.recipient not in self._inboxes:
            self.register_agent(message.recipient)

        self._inboxes[message.recipient].append(message)

        # Index by correlation ID
        if message.correlation_id:
            if message.correlation_id not in self._correlation_index:
                self._correlation_index[message.correlation_id] = []
            self._correlation_index[message.correlation_id].append(message)

    def broadcast(self, message: Message) -> None:
        """Broadcast a message to all registered agents.

        Parameters
        ----------
        message : Message
            The message to broadcast.
        """
        message.recipient = "ALL"
        for agent_id in self._inboxes:
            # Create a copy for each recipient
            msg_copy = Message(
                id=message.id,
                type=message.type,
                sender=message.sender,
                recipient=agent_id,
                payload=dict(message.payload),
                timestamp=message.timestamp,
                priority=message.priority,
                requires_response=message.requires_response,
                correlation_id=message.correlation_id,
            )
            self._inboxes[agent_id].append(msg_copy)

        # Index by correlation ID
        if message.correlation_id:
            if message.correlation_id not in self._correlation_index:
                self._correlation_index[message.correlation_id] = []
            self._correlation_index[message.correlation_id].append(message)

    def receive(self, agent_id: str, clear: bool = True) -> List[Message]:
        """Get pending messages for an agent.

        Parameters
        ----------
        agent_id : str
            The agent to get messages for.
        clear : bool, default True
            Whether to clear the inbox after reading.

        Returns
        -------
        list of Message
            Pending messages, sorted by priority (highest first).
        """
        if agent_id not in self._inboxes:
            return []

        messages = self._inboxes[agent_id]
        # Sort by priority (lower number = higher priority)
        messages.sort(key=lambda m: m.priority)

        if clear:
            self._inboxes[agent_id] = []

        return messages

    def peek(self, agent_id: str) -> List[Message]:
        """Peek at pending messages without clearing.

        Parameters
        ----------
        agent_id : str
            The agent to peek messages for.

        Returns
        -------
        list of Message
            Pending messages, sorted by priority.
        """
        return self.receive(agent_id, clear=False)

    def get_responses(self, correlation_id: str) -> List[Message]:
        """Get all responses to a request.

        Parameters
        ----------
        correlation_id : str
            The correlation ID from the original request.

        Returns
        -------
        list of Message
            All messages with the given correlation ID.
        """
        return list(self._correlation_index.get(correlation_id, []))

    def has_messages(self, agent_id: str) -> bool:
        """Check if an agent has pending messages.

        Parameters
        ----------
        agent_id : str
            The agent to check.

        Returns
        -------
        bool
            True if there are pending messages.
        """
        return agent_id in self._inboxes and len(self._inboxes[agent_id]) > 0

    def count_messages(self, agent_id: str) -> int:
        """Count pending messages for an agent.

        Parameters
        ----------
        agent_id : str
            The agent to count messages for.

        Returns
        -------
        int
            Number of pending messages.
        """
        return len(self._inboxes.get(agent_id, []))

    def clear_inbox(self, agent_id: str) -> None:
        """Clear all messages from an agent's inbox.

        Parameters
        ----------
        agent_id : str
            The agent whose inbox to clear.
        """
        if agent_id in self._inboxes:
            self._inboxes[agent_id] = []

    def get_all_agents(self) -> List[str]:
        """Get list of all registered agents.

        Returns
        -------
        list of str
            All registered agent IDs.
        """
        return list(self._inboxes.keys())

    def clear_all(self) -> None:
        """Clear all inboxes and correlation index."""
        self._inboxes.clear()
        self._correlation_index.clear()
