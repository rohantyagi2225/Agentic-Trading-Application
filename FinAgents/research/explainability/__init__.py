"""Explainability and interpretability engine for research agents.

This package provides utilities for building structured reasoning chains,
rendering human-readable explanations, maintaining decision audit trails,
and computing quantitative interpretability metrics.
"""

from FinAgents.research.domain_agents.base_agent import (
    ResearchAgent,
    Action,
    ActionType,
    Explanation,
)

from .reasoning_chain import ReasoningStep, ReasoningBranch, ReasoningChain
from .explanation_renderer import (
    ExplanationFormat,
    DataAttribution,
    RenderedExplanation,
    ExplanationRenderer,
)
from .decision_audit import (
    AuditEntry,
    CounterfactualResult,
    AuditReport,
    DecisionAuditTrail,
)
from .interpretability_metrics import (
    CompletenessScore,
    ConsistencyScore,
    FaithfulnessScore,
    InterpretabilityReport,
    InterpretabilityScorer,
)

__all__ = [
    "ResearchAgent",
    "Action",
    "ActionType",
    "Explanation",
    # Reasoning chain
    "ReasoningStep",
    "ReasoningBranch",
    "ReasoningChain",
    # Explanation rendering
    "ExplanationFormat",
    "DataAttribution",
    "RenderedExplanation",
    "ExplanationRenderer",
    # Decision audit
    "AuditEntry",
    "CounterfactualResult",
    "AuditReport",
    "DecisionAuditTrail",
    # Interpretability metrics
    "CompletenessScore",
    "ConsistencyScore",
    "FaithfulnessScore",
    "InterpretabilityReport",
    "InterpretabilityScorer",
]
