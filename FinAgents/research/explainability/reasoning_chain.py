"""Structured reasoning chain representations for explainable agents.

This module defines data structures and utilities for building
step-by-step reasoning chains with optional branching to support
counterfactual analysis and rich explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


def _now_utc() -> datetime:
    """Return the current UTC time with timezone information."""

    return datetime.now(timezone.utc)


@dataclass
class ReasoningStep:
    """Single step in a reasoning chain.

    Attributes
    ----------
    step_id:
        Globally unique identifier for the step.
    observation:
        Description of what was observed at this step. For non-observation
        steps this may be an empty string.
    inference:
        Conclusion or intermediate inference drawn at this step.
    confidence:
        Confidence in the inference in the range ``[0, 1]``.
    data_source:
        Human-readable description of where the data originated
        (e.g., ``"price_history"``, ``"risk_model"``).
    evidence:
        Arbitrary mapping capturing supporting data for the step.
    step_type:
        Type of step: one of ``"observation"``, ``"analysis"``,
        ``"inference"``, or ``"decision"`.
    timestamp:
        Timestamp when the step was created, in UTC.
    parent_step_id:
        Optional identifier of the parent step to model branching
        relationships.
    supporting_step_ids:
        Optional list of step identifiers that support this step, used
        primarily for inference and decision types.
    """

    step_id: str
    observation: str
    inference: str
    confidence: float
    data_source: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    step_type: str = "analysis"
    timestamp: datetime = field(default_factory=_now_utc)
    parent_step_id: Optional[str] = None
    supporting_step_ids: List[str] = field(default_factory=list)


@dataclass
class ReasoningBranch:
    """Alternative reasoning branch representing a distinct hypothesis.

    Attributes
    ----------
    branch_id:
        Unique identifier for the branch.
    hypothesis:
        Short description of the hypothesis explored in this branch.
    steps:
        Ordered list of :class:`ReasoningStep` instances belonging to this
        branch.
    conclusion:
        Final conclusion for this branch.
    confidence:
        Confidence in the branch-level conclusion in ``[0, 1]``.
    was_selected:
        Whether this branch was ultimately selected as the primary
        reasoning path.
    rejection_reason:
        Optional explanation of why this branch was not selected.
    """

    branch_id: str
    hypothesis: str
    steps: List[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    was_selected: bool = False
    rejection_reason: Optional[str] = None


class ReasoningChain:
    """Container for structured reasoning steps and branches.

    The chain maintains a linear sequence of main-line steps as well as
    optional alternative branches. Branches can be used for
    counterfactual reasoning or to capture discarded hypotheses.
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        agent_id: str = "",
        decision_context: str = "",
    ) -> None:
        self.chain_id: str = chain_id or str(uuid.uuid4())
        self.agent_id: str = agent_id
        self.decision_context: str = decision_context
        self._steps: List[ReasoningStep] = []
        self._branches: Dict[str, ReasoningBranch] = {}
        self._selected_branch_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Step creation helpers
    # ------------------------------------------------------------------
    def add_observation(
        self,
        observation: str,
        data_source: str,
        evidence: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> ReasoningStep:
        """Add an observation step to the main chain.

        Parameters
        ----------
        observation:
            Factual observation recorded by the agent.
        data_source:
            Source of the observed data.
        evidence:
            Optional supporting evidence mapping.
        confidence:
            Confidence in the observation, defaulting to ``1.0``.
        """

        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            observation=observation,
            inference="",
            confidence=max(0.0, min(1.0, confidence)),
            data_source=data_source,
            evidence=evidence or {},
            step_type="observation",
            parent_step_id=self._steps[-1].step_id if self._steps else None,
        )
        self._steps.append(step)
        return step

    def add_analysis(
        self,
        analysis: str,
        data_source: str,
        evidence: Optional[Dict[str, Any]] = None,
        confidence: float = 0.8,
    ) -> ReasoningStep:
        """Add an analysis step that interprets prior observations.

        Parameters
        ----------
        analysis:
            Textual description of the analysis.
        data_source:
            Source of data informing the analysis.
        evidence:
            Optional supporting evidence mapping.
        confidence:
            Confidence in the analysis, defaulting to ``0.8``.
        """

        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            observation="",
            inference=analysis,
            confidence=max(0.0, min(1.0, confidence)),
            data_source=data_source,
            evidence=evidence or {},
            step_type="analysis",
            parent_step_id=self._steps[-1].step_id if self._steps else None,
        )
        self._steps.append(step)
        return step

    def add_inference(
        self,
        inference: str,
        confidence: float,
        supporting_steps: Optional[List[str]] = None,
    ) -> ReasoningStep:
        """Add an inference step to the chain.

        Parameters
        ----------
        inference:
            Conclusion drawn from previous observations and analyses.
        confidence:
            Confidence in the inference in ``[0, 1]``.
        supporting_steps:
            Optional list of step identifiers providing support for this
            inference.
        """

        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            observation="",
            inference=inference,
            confidence=max(0.0, min(1.0, confidence)),
            data_source="inference",
            evidence={},
            step_type="inference",
            parent_step_id=self._steps[-1].step_id if self._steps else None,
            supporting_step_ids=supporting_steps or [],
        )
        self._steps.append(step)
        return step

    def add_decision(
        self,
        decision: str,
        confidence: float,
        reasoning_summary: str,
    ) -> ReasoningStep:
        """Add a final decision step to the chain.

        Parameters
        ----------
        decision:
            Human-readable description of the decision.
        confidence:
            Confidence in the decision in ``[0, 1]``.
        reasoning_summary:
            Short textual summary of the reasoning leading to the decision.
        """

        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            observation=reasoning_summary,
            inference=decision,
            confidence=max(0.0, min(1.0, confidence)),
            data_source="decision",
            evidence={},
            step_type="decision",
            parent_step_id=self._steps[-1].step_id if self._steps else None,
        )
        self._steps.append(step)
        return step

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------
    def create_branch(self, hypothesis: str) -> ReasoningBranch:
        """Create a new reasoning branch for an alternative hypothesis.

        The branch initially has no steps; callers can directly mutate the
        returned branch's ``steps`` list or use the main chain API and
        later associate steps if needed.
        """

        branch_id = str(uuid.uuid4())
        branch = ReasoningBranch(branch_id=branch_id, hypothesis=hypothesis)
        self._branches[branch_id] = branch
        # If no branch selected yet, mark this as the default candidate.
        if self._selected_branch_id is None:
            self._selected_branch_id = branch_id
            branch.was_selected = True
        return branch

    def select_branch(self, branch_id: str, reason: str) -> None:
        """Mark one branch as selected and others as rejected.

        Parameters
        ----------
        branch_id:
            Identifier of the branch to select.
        reason:
            Explanation for why this branch was chosen. This is recorded as
            the rejection reason on all non-selected branches.
        """

        if branch_id not in self._branches:
            raise ValueError(f"Unknown branch_id: {branch_id}")

        self._selected_branch_id = branch_id
        for bid, branch in self._branches.items():
            if bid == branch_id:
                branch.was_selected = True
                branch.rejection_reason = None
            else:
                branch.was_selected = False
                if not branch.rejection_reason:
                    branch.rejection_reason = f"Rejected in favor of branch {branch_id}: {reason}"

    # ------------------------------------------------------------------
    # Accessors and serialization
    # ------------------------------------------------------------------
    def get_chain(self) -> List[ReasoningStep]:
        """Return the main reasoning chain.

        If branches are defined and one is selected, this returns that
        branch's steps. Otherwise it returns the linear main-line steps.
        """

        if self._branches and self._selected_branch_id:
            branch = self._branches.get(self._selected_branch_id)
            if branch is not None and branch.steps:
                return list(branch.steps)
        return list(self._steps)

    def get_all_branches(self) -> List[ReasoningBranch]:
        """Return all reasoning branches (may be empty)."""

        return list(self._branches.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire reasoning chain to a dictionary.

        The resulting structure is JSON-serializable when combined with
        suitable datetime handling (e.g., converting to ISO strings).
        """

        def _step_to_dict(step: ReasoningStep) -> Dict[str, Any]:
            data = asdict(step)
            # Convert datetime to ISO 8601 string for portability.
            ts = data.get("timestamp")
            if isinstance(ts, datetime):
                data["timestamp"] = ts.isoformat()
            return data

        return {
            "chain_id": self.chain_id,
            "agent_id": self.agent_id,
            "decision_context": self.decision_context,
            "steps": [_step_to_dict(s) for s in self._steps],
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "hypothesis": b.hypothesis,
                    "steps": [_step_to_dict(s) for s in b.steps],
                    "conclusion": b.conclusion,
                    "confidence": b.confidence,
                    "was_selected": b.was_selected,
                    "rejection_reason": b.rejection_reason,
                }
                for b in self._branches.values()
            ],
            "selected_branch_id": self._selected_branch_id,
        }

    def to_narrative(self) -> str:
        """Generate a human-readable narrative representation of the chain.

        The narrative enumerates steps in order with a concise label
        describing the type and contents of each step.
        """

        steps = self.get_chain()
        if not steps:
            return "No reasoning steps recorded."

        lines: List[str] = []
        for idx, step in enumerate(steps, start=1):
            if step.step_type == "observation":
                desc = step.observation
                label = "Observation"
            elif step.step_type == "analysis":
                desc = step.inference
                label = "Analysis"
            elif step.step_type == "inference":
                desc = step.inference
                label = "Inference"
            elif step.step_type == "decision":
                desc = step.inference
                label = "Decision"
            else:
                desc = step.inference or step.observation
                label = step.step_type.capitalize() or "Step"

            conf_pct = round(step.confidence * 100)
            lines.append(f"Step {idx}: [{label}] {desc} (confidence: {conf_pct}%)")

        return " -> ".join(lines)
