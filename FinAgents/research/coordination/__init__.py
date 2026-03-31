"""Advanced Multi-Agent Coordination Module.

This module provides shared memory, voting, and structured agent workflows
for coordinating multiple research agents in the FinAgents framework.

Exports
-------
Blackboard : Shared memory system for agent communication
BlackboardEntry : Single entry in the blackboard
Section : Enum defining blackboard sections
MessageBus : Communication protocol for agent messaging
Message, MessageType : Message definitions
TradeProposal, RiskAssessment, PortfolioReallocation : Structured message payloads
VoteRequest, VoteResponse : Voting message types
VotingMechanism : Multi-agent voting and negotiation system
VoteSession, VoteResult, NegotiationResult : Voting data structures
AgentCoordinator : Main orchestrator for agent coordination
TradingCycleResult : Result of a trading coordination cycle
WorkflowEngine : Configurable workflow execution
WorkflowDefinition, WorkflowStep, WorkflowResult : Workflow data structures
"""

from __future__ import annotations

from FinAgents.research.coordination.blackboard import (
    Blackboard,
    BlackboardEntry,
    Section,
)
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
from FinAgents.research.coordination.coordinator import (
    AgentCoordinator,
    TradingCycleResult,
)
from FinAgents.research.coordination.workflow_engine import (
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowResult,
    WorkflowStep,
)

__all__ = [
    # Blackboard
    "Blackboard",
    "BlackboardEntry",
    "Section",
    # Protocols
    "Message",
    "MessageBus",
    "MessageType",
    "TradeProposal",
    "RiskAssessment",
    "PortfolioReallocation",
    "VoteRequest",
    "VoteResponse",
    # Voting
    "VotingMechanism",
    "VoteSession",
    "VoteResult",
    "NegotiationResult",
    # Coordinator
    "AgentCoordinator",
    "TradingCycleResult",
    # Workflow Engine
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowResult",
]
