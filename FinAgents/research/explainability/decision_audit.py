"""Decision audit trail for explainable trading agents.

This module provides an in-memory audit trail for recording trading
decisions along with their reasoning chains, market context, and
outcomes. It is designed for post-hoc analysis, compliance review, and
counterfactual reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import uuid

from FinAgents.research.domain_agents.base_agent import Action

from .reasoning_chain import ReasoningChain


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AuditEntry:
    """Single decision audit record.

    Attributes
    ----------
    audit_id:
        Globally unique identifier for this audit entry.
    decision_id:
        Identifier of the decision, typically aligned with an external
        trade identifier or order ID.
    agent_id:
        Identifier of the agent responsible for the decision.
    timestamp:
        Time at which the decision was logged.
    action:
        :class:`Action` object representing the decision.
    reasoning_chain:
        Serialized representation of the reasoning chain
        (:meth:`ReasoningChain.to_dict`).
    market_state:
        Snapshot of relevant market state at decision time.
    portfolio_state:
        Snapshot of relevant portfolio state at decision time.
    constraints_checked:
        List of constraints (e.g., risk limits, compliance rules)
        evaluated for the decision.
    compliance_status:
        Overall compliance outcome (e.g., ``"PASSED"``, ``"BLOCKED"``).
    outcome:
        Optional mapping with realized outcomes (PnL, slippage, etc.),
        filled once the trade has resolved.
    tags:
        Free-form tags for flexible filtering (e.g., "high_confidence").
    """

    audit_id: str
    decision_id: str
    agent_id: str
    timestamp: datetime
    action: Action
    reasoning_chain: Dict[str, Any]
    market_state: Dict[str, Any]
    portfolio_state: Dict[str, Any] = field(default_factory=dict)
    constraints_checked: List[Dict[str, Any]] = field(default_factory=list)
    compliance_status: str = "UNKNOWN"
    outcome: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CounterfactualResult:
    """Result of a simple counterfactual analysis on a decision."""

    original_decision: str
    counterfactual_decision: str
    parameter_changed: str
    original_value: Any
    new_value: Any
    would_have_been_better: Optional[bool]
    analysis: str


@dataclass
class AuditReport:
    """Aggregated statistics over a period of decisions."""

    period: str
    total_decisions: int
    by_agent: Dict[str, int]
    by_action_type: Dict[str, int]
    accuracy_rate: float
    avg_confidence: float
    constraint_breaches: int
    compliance_rate: float
    notable_decisions: List[Dict[str, Any]]


class DecisionAuditTrail:
    """In-memory audit trail for trading decisions.

    This implementation keeps a bounded list of :class:`AuditEntry`
    objects and provides basic querying and reporting utilities. It can
    be replaced or wrapped by a persistence layer in production.
    """

    def __init__(self, max_entries: int = 50000) -> None:
        self.max_entries = max_entries
        self._entries: List[AuditEntry] = []

    # ------------------------------------------------------------------
    # Core logging and update methods
    # ------------------------------------------------------------------
    def log_decision(
        self,
        agent_id: str,
        action: Action,
        reasoning_chain: ReasoningChain,
        market_state: Dict[str, Any],
        portfolio_state: Optional[Dict[str, Any]] = None,
        constraints_checked: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Record a new decision and return its audit identifier."""

        audit_id = str(uuid.uuid4())
        decision_id = audit_id  # can be overridden externally if needed
        entry = AuditEntry(
            audit_id=audit_id,
            decision_id=decision_id,
            agent_id=agent_id,
            timestamp=_now_utc(),
            action=action,
            reasoning_chain=reasoning_chain.to_dict(),
            market_state=market_state,
            portfolio_state=portfolio_state or {},
            constraints_checked=constraints_checked or [],
            compliance_status="PASSED",
            outcome=None,
            tags=[],
        )

        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            # Drop oldest entries when exceeding capacity.
            self._entries = self._entries[-self.max_entries :]

        return audit_id

    def update_outcome(self, audit_id: str, outcome: Dict[str, Any]) -> None:
        """Attach realized outcome information to an existing entry."""

        for entry in self._entries:
            if entry.audit_id == audit_id:
                entry.outcome = outcome
                return
        raise ValueError(f"Unknown audit_id: {audit_id}")

    # ------------------------------------------------------------------
    # Query and analysis
    # ------------------------------------------------------------------
    def query(
        self,
        agent_id: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action_type: Optional[str] = None,
        outcome_positive: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Query audit entries using multiple filters.

        Parameters
        ----------
        agent_id:
            Restrict to decisions taken by this agent.
        symbol:
            Restrict to decisions involving this symbol.
        start_time, end_time:
            Filter decisions whose timestamps fall within this window.
        action_type:
            Restrict to a specific action type (e.g., ``"BUY"``).
        outcome_positive:
            If ``True``, only include decisions with positive PnL (as
            inferred from the ``"pnl"`` field of the outcome mapping);
            if ``False``, only negative; if ``None``, ignore outcome
            sign.
        limit:
            Maximum number of entries to return.
        """

        results: List[AuditEntry] = []
        for entry in reversed(self._entries):  # newest first
            if agent_id is not None and entry.agent_id != agent_id:
                continue
            if symbol is not None and entry.action.symbol != symbol:
                continue
            if action_type is not None and entry.action.action_type.value != action_type:
                continue
            if start_time is not None and entry.timestamp < start_time:
                continue
            if end_time is not None and entry.timestamp > end_time:
                continue
            if outcome_positive is not None:
                pnl = None
                if entry.outcome is not None:
                    pnl = entry.outcome.get("pnl")
                if pnl is None:
                    continue
                if outcome_positive and pnl <= 0:
                    continue
                if not outcome_positive and pnl >= 0:
                    continue

            results.append(entry)
            if len(results) >= limit:
                break

        return list(reversed(results))  # return oldest-first order

    def counterfactual_analysis(
        self, audit_id: str, modified_params: Dict[str, Any]
    ) -> CounterfactualResult:
        """Perform a simple counterfactual analysis for the given decision.

        The current implementation focuses on position-size constraints
        and evaluates whether a stricter limit would have blocked the
        trade. This is intentionally conservative and does not attempt to
        re-simulate market dynamics.
        """

        entry = next((e for e in self._entries if e.audit_id == audit_id), None)
        if entry is None:
            raise ValueError(f"Unknown audit_id: {audit_id}")

        param_name = next(iter(modified_params.keys()), "unknown_param")
        new_value = modified_params[param_name]
        original_value = None

        # Example: {"max_position_size": 0.05}
        counterfactual_decision = "unchanged"
        would_have_been_better: Optional[bool] = None

        position_size = abs(entry.action.quantity)
        portfolio_equity = entry.portfolio_state.get("equity") or 0.0
        if portfolio_equity > 0:
            size_pct = position_size / portfolio_equity
        else:
            size_pct = 0.0

        if param_name == "max_position_size":
            original_value = entry.portfolio_state.get("max_position_size")
            if size_pct > new_value:
                counterfactual_decision = "BLOCKED_BY_CONSTRAINT"
            else:
                counterfactual_decision = "ALLOWED"

            # If we have realized PnL, we can assess whether the
            # counterfactual would have been better.
            if entry.outcome is not None and "pnl" in entry.outcome:
                pnl = entry.outcome["pnl"]
                if counterfactual_decision == "BLOCKED_BY_CONSTRAINT":
                    would_have_been_better = pnl < 0
                else:
                    # If trade was allowed and profitable, we assume the
                    # actual decision is at least as good.
                    would_have_been_better = pnl >= 0

        analysis_lines = [
            f"Counterfactual analysis for parameter '{param_name}':",
            f"- Original value: {original_value}",
            f"- New value: {new_value}",
            f"- Position size fraction: {size_pct:.4f}",
            f"- Counterfactual decision: {counterfactual_decision}",
        ]
        if would_have_been_better is not None:
            analysis_lines.append(
                f"- Would this have been better? {'yes' if would_have_been_better else 'no'}"
            )

        return CounterfactualResult(
            original_decision=entry.action.action_type.value,
            counterfactual_decision=counterfactual_decision,
            parameter_changed=param_name,
            original_value=original_value,
            new_value=new_value,
            would_have_been_better=would_have_been_better,
            analysis="\n".join(analysis_lines),
        )

    def generate_audit_report(
        self, start_time: datetime, end_time: datetime
    ) -> AuditReport:
        """Generate a summary report over the specified period."""

        entries = self.query(start_time=start_time, end_time=end_time, limit=self.max_entries)
        total = len(entries)
        by_agent: Dict[str, int] = {}
        by_action_type: Dict[str, int] = {}
        correct = 0
        total_with_outcome = 0
        sum_confidence = 0.0
        breaches = 0
        compliant = 0

        notable: List[Dict[str, Any]] = []

        for entry in entries:
            by_agent[entry.agent_id] = by_agent.get(entry.agent_id, 0) + 1
            atype = entry.action.action_type.value
            by_action_type[atype] = by_action_type.get(atype, 0) + 1

            sum_confidence += entry.action.confidence

            if entry.compliance_status != "PASSED":
                breaches += 1
            else:
                compliant += 1

            if entry.outcome is not None and "pnl" in entry.outcome:
                pnl = entry.outcome["pnl"]
                total_with_outcome += 1
                if pnl > 0:
                    correct += 1
                # Track notable decisions
                notable.append({"audit_id": entry.audit_id, "pnl": pnl, "confidence": entry.action.confidence})

        accuracy_rate = (correct / total_with_outcome) if total_with_outcome else 0.0
        avg_confidence = (sum_confidence / total) if total else 0.0
        compliance_rate = (compliant / total) if total else 0.0

        # Identify highest PnL, biggest loss, and most confident wrong.
        notable_sorted = sorted(notable, key=lambda x: x["pnl"])
        highest_pnl = max(notable, key=lambda x: x["pnl"], default=None)
        biggest_loss = notable_sorted[0] if notable_sorted else None
        most_confident_wrong = None
        wrong_trades = [n for n in notable if n["pnl"] < 0]
        if wrong_trades:
            most_confident_wrong = max(wrong_trades, key=lambda x: x["confidence"])

        notable_decisions: List[Dict[str, Any]] = []
        if highest_pnl is not None:
            notable_decisions.append({"type": "highest_pnl", **highest_pnl})
        if biggest_loss is not None:
            notable_decisions.append({"type": "biggest_loss", **biggest_loss})
        if most_confident_wrong is not None:
            notable_decisions.append({"type": "most_confident_wrong", **most_confident_wrong})

        period_str = f"{start_time.isoformat()} to {end_time.isoformat()}"

        return AuditReport(
            period=period_str,
            total_decisions=total,
            by_agent=by_agent,
            by_action_type=by_action_type,
            accuracy_rate=accuracy_rate,
            avg_confidence=avg_confidence,
            constraint_breaches=breaches,
            compliance_rate=compliance_rate,
            notable_decisions=notable_decisions,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, format: str = "json") -> str:
        """Serialize the full audit trail.

        Currently supports JSON only. Timestamps are converted to ISO 8601
        strings and actions are serialized via :func:`dataclasses.asdict`.
        """

        if format != "json":
            raise ValueError("Only JSON export is currently supported.")

        def _entry_to_dict(entry: AuditEntry) -> Dict[str, Any]:
            data = asdict(entry)
            # Convert timestamp to ISO 8601
            ts = data.get("timestamp")
            if isinstance(ts, datetime):
                data["timestamp"] = ts.isoformat()
            return data

        payload = [_entry_to_dict(e) for e in self._entries]
        return json.dumps(payload, indent=2, default=str)
