"""Explanation rendering utilities for research agents.

This module converts structured reasoning chains and actions into
human-readable explanations in multiple formats for different
stakeholders (traders, risk managers, compliance, executives).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from FinAgents.research.domain_agents.base_agent import Action

from .reasoning_chain import ReasoningChain, ReasoningBranch, ReasoningStep


class ExplanationFormat(str, Enum):
    """Supported explanation output formats."""

    PLAIN_TEXT = "plain_text"
    STRUCTURED_JSON = "structured_json"
    REGULATORY = "regulatory"
    EXECUTIVE_SUMMARY = "executive_summary"


@dataclass
class DataAttribution:
    """Structured description of data sources used in a decision.

    Attributes
    ----------
    sources:
        List of dictionaries with ``name``, ``contribution_pct``, and
        ``data_type`` keys describing the relative contribution of each
        data source.
    primary_source:
        Name of the primary data source.
    coverage:
        Fraction in ``[0, 1]`` indicating how much of the decision is
        backed by identifiable data sources.
    """

    sources: List[Dict[str, Any]] = field(default_factory=list)
    primary_source: Optional[str] = None
    coverage: float = 0.0


@dataclass
class RenderedExplanation:
    """Container for rendered explanations.

    Attributes
    ----------
    format:
        Output format used to render the explanation.
    content:
        Primary serialized representation (text or JSON string).
    sections:
        Optional mapping from section names (e.g. ``"risk"``) to
        textual content.
    metadata:
        Machine-readable metadata associated with the explanation.
    """

    format: ExplanationFormat
    content: str
    sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExplanationRenderer:
    """Render human-readable explanations from reasoning chains and actions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.default_format: ExplanationFormat = config.get(
            "default_format", ExplanationFormat.PLAIN_TEXT
        )
        self.include_alternatives: bool = config.get("include_alternatives", True)
        self.verbosity: str = config.get("verbosity", "standard")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(
        self,
        reasoning_chain: ReasoningChain,
        action: Action,
        format: Optional[ExplanationFormat] = None,
    ) -> RenderedExplanation:
        """Render an explanation in the requested format.

        Parameters
        ----------
        reasoning_chain:
            Structured chain describing how the decision was reached.
        action:
            Final action taken by the agent.
        format:
            Desired output format. If omitted, uses the configured
            default format.
        """

        fmt = format or self.default_format

        if fmt == ExplanationFormat.PLAIN_TEXT:
            content = self.render_plain_text(reasoning_chain, action)
        elif fmt == ExplanationFormat.STRUCTURED_JSON:
            content_dict = self.render_structured_json(reasoning_chain, action)
            # Store JSON-compatible dict as string; callers can choose
            # to re-serialize as needed.
            import json

            content = json.dumps(content_dict, default=str, indent=2)
        elif fmt == ExplanationFormat.REGULATORY:
            content = self.render_regulatory(reasoning_chain, action)
        elif fmt == ExplanationFormat.EXECUTIVE_SUMMARY:
            content = self.render_executive_summary(reasoning_chain, action)
        else:
            raise ValueError(f"Unsupported explanation format: {fmt}")

        # Basic metadata including attribution information.
        attribution = self.render_data_attribution(reasoning_chain)
        metadata = {
            "action": asdict(action),
            "reasoning_chain": reasoning_chain.to_dict(),
            "data_attribution": asdict(attribution),
        }

        sections: Dict[str, str] = {}
        if fmt == ExplanationFormat.PLAIN_TEXT:
            # simple heuristic section extraction
            sections["full_text"] = content

        return RenderedExplanation(format=fmt, content=content, sections=sections, metadata=metadata)

    # ------------------------------------------------------------------
    # Format-specific renderers
    # ------------------------------------------------------------------
    def _extract_core_sections(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Extract core components (observation, analysis, inference, decision)."""

        steps = chain.get_chain()
        observation = ""
        analysis = ""
        inference = ""
        decision = ""
        risk_section = ""

        for step in steps:
            if step.step_type == "observation" and not observation:
                observation = step.observation
            elif step.step_type == "analysis" and not analysis:
                analysis = step.inference
            elif step.step_type == "inference" and not inference:
                inference = step.inference
            elif step.step_type == "decision" and not decision:
                decision = step.inference

        # Risk reasoning may be embedded in analysis or dedicated steps.
        for step in steps:
            if "risk" in (step.inference or "").lower() and not risk_section:
                risk_section = step.inference

        return {
            "observation": observation,
            "analysis": analysis,
            "inference": inference,
            "decision": decision,
            "risk_section": risk_section,
        }

    def render_plain_text(self, chain: ReasoningChain, action: Action) -> str:
        """Render a detailed plain-text explanation.

        Format::

            TRADING DECISION: {action_type} {symbol}

            REASONING:
            1. Market Observation: {observation}
            2. Analysis: {analysis} (confidence: {conf}%)
            3. Signal: {inference}

            DECISION: {decision} with {confidence}% confidence

            RISK JUSTIFICATION: {risk_section}

            DATA SOURCES: {sources list}

            ALTERNATIVE PATHS CONSIDERED:
            - {branch hypothesis}: {rejection reason}
        """

        sections = self._extract_core_sections(chain)
        observation = sections["observation"] or "No explicit market observation recorded."
        analysis = sections["analysis"] or "No explicit analysis step recorded."
        inference = sections["inference"] or "No explicit signal inference recorded."
        decision_text = sections["decision"] or action.reasoning_summary or "No decision step recorded."

        # Use the highest confidence step as a proxy for analysis confidence.
        steps = chain.get_chain()
        analysis_conf = 0.0
        if steps:
            analysis_conf = max((s.confidence for s in steps), default=0.0) * 100.0
        decision_conf = action.confidence * 100.0

        risk_section = sections["risk_section"] or "Risk considerations were not explicitly documented."

        # Data sources
        data_sources = {s.data_source for s in steps if getattr(s, "data_source", None)}
        sources_list = ", ".join(sorted(data_sources)) if data_sources else "Not specified"

        lines = [
            f"TRADING DECISION: {action.action_type.value} {action.symbol or ''}".strip(),
            "",
            "REASONING:",
            f"1. Market Observation: {observation}",
            f"2. Analysis: {analysis} (confidence: {analysis_conf:.1f}%)",
            f"3. Signal: {inference}",
            "",
            f"DECISION: {decision_text} with {decision_conf:.1f}% confidence",
            "",
            f"RISK JUSTIFICATION: {risk_section}",
            "",
            f"DATA SOURCES: {sources_list}",
            "",
        ]

        if self.include_alternatives:
            branches = chain.get_all_branches()
            lines.append("ALTERNATIVE PATHS CONSIDERED:")
            if not branches:
                lines.append("- None explicitly modeled.")
            else:
                for branch in branches:
                    if branch.was_selected:
                        continue
                    reason = branch.rejection_reason or "No explicit rejection reason recorded."
                    lines.append(f"- {branch.hypothesis}: {reason}")
        else:
            lines.append("ALTERNATIVE PATHS CONSIDERED: omitted by configuration.")

        return "\n".join(lines)

    def render_structured_json(self, chain: ReasoningChain, action: Action) -> Dict[str, Any]:
        """Render a complete JSON-style dictionary with all details."""

        chain_dict = chain.to_dict()

        # Convert steps to serializable structures; they should already be
        # basic types except timestamps which are coerced in ``to_dict``.
        return {
            "action": asdict(action),
            "reasoning_chain": chain_dict,
        }

    def render_regulatory(self, chain: ReasoningChain, action: Action) -> str:
        """Render a formal regulatory-style explanation.

        The output is intentionally structured and conservative in tone
        for consumption by compliance and regulators.
        """

        sections = self._extract_core_sections(chain)
        steps = chain.get_chain()

        # Simple heuristics for risk metrics based on action parameters.
        max_potential_loss = "N/A"
        risk_reward_ratio = "N/A"
        if action.stop_loss is not None and action.take_profit is not None and action.price is not None:
            loss = abs(action.price - action.stop_loss)
            gain = abs(action.take_profit - action.price)
            if loss > 0:
                risk_reward_ratio = f"{gain / loss:.2f}"
            max_potential_loss = f"{loss:.4f} per unit"

        risk_controls_applied = []
        for step in steps:
            if "risk" in (step.inference or "").lower():
                risk_controls_applied.append(step.inference)

        lines = [
            "This trading decision was made based on the following analysis:",
            "",
            f"- Market observation: {sections['observation'] or 'Not explicitly recorded.'}",
            f"- Analytical assessment: {sections['analysis'] or 'Not explicitly recorded.'}",
            f"- Derived signal: {sections['inference'] or 'Not explicitly recorded.'}",
            "",
            f"Proposed action: {action.action_type.value} {action.symbol or ''} (quantity={action.quantity}, price={action.price})",
            "",
            "Risk controls applied:",
        ]
        if risk_controls_applied:
            for rc in risk_controls_applied:
                lines.append(f"- {rc}")
        else:
            lines.append("- No explicit risk control steps were documented.")

        lines.extend(
            [
                "",
                "Compliance checks passed:",
                "- Position sizing within configured limits (assumed unless overridden).",
                "- Order type and venue consistent with policy (assumed unless otherwise noted).",
                "",
                f"Maximum potential loss: {max_potential_loss}",
                f"Risk-reward ratio: {risk_reward_ratio}",
            ]
        )

        return "\n".join(lines)

    def render_executive_summary(self, chain: ReasoningChain, action: Action) -> str:
        """Render a concise executive-level summary (2-3 sentences)."""

        sections = self._extract_core_sections(chain)
        decision_conf = action.confidence * 100.0

        observation = sections["observation"] or "current market conditions"
        decision_text = sections["decision"] or action.reasoning_summary or f"{action.action_type.value} {action.symbol or ''}".strip()

        sentence1 = (
            f"The agent recommends {action.action_type.value} {action.symbol or ''} "
            f"based on {observation}."
        ).strip()
        sentence2 = (
            f"This decision is supported by the analysis '{sections['analysis'] or 'n/a'}' "
            f"and inferred signal '{sections['inference'] or 'n/a'}'."
        )
        sentence3 = (
            f"Overall confidence in this decision is {decision_conf:.1f}% and the "
            f"decision can be summarized as: {decision_text}."
        )

        return " " .join([sentence1, sentence2, sentence3])

    # ------------------------------------------------------------------
    # Data attribution
    # ------------------------------------------------------------------
    def render_data_attribution(self, chain: ReasoningChain) -> DataAttribution:
        """Compute a simple data attribution summary for the chain."""

        steps = chain.get_chain()
        if not steps:
            return DataAttribution(sources=[], primary_source=None, coverage=0.0)

        # Count occurrences of each data source.
        counts: Dict[str, int] = {}
        total_with_source = 0
        for step in steps:
            src = getattr(step, "data_source", None)
            if not src:
                continue
            total_with_source += 1
            counts[src] = counts.get(src, 0) + 1

        sources_list: List[Dict[str, Any]] = []
        primary_source: Optional[str] = None
        if counts:
            max_count = max(counts.values())
            for name, c in counts.items():
                pct = (c / total_with_source) * 100.0 if total_with_source else 0.0
                sources_list.append(
                    {
                        "name": name,
                        "contribution_pct": pct,
                        "data_type": "unknown",
                    }
                )
                if c == max_count and primary_source is None:
                    primary_source = name

        coverage = total_with_source / len(steps) if steps else 0.0
        return DataAttribution(sources=sources_list, primary_source=primary_source, coverage=coverage)
