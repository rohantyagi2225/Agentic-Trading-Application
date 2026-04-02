from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from FinAgents.research.domain_agents.base_agent import Action, ActionType
from FinAgents.research.explainability.explanation_renderer import (
    ExplanationFormat,
)
from FinAgents.research.explainability.reasoning_chain import ReasoningChain, ReasoningStep
from FinAgents.research.integration.system_integrator import (
    ResearchSystem,
    ResearchSystemConfig,
    ResearchSystemIntegrator,
)
from FinAgents.research.evaluation.comparison_engine import ComparisonEngine


router = APIRouter(prefix="/api/research", tags=["Research"])

_lock = threading.Lock()
_system_cache: Dict[str, Any] = {
    "system": None,
    "last_result": None,
}


class SimulationRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOG"])
    num_steps: int = Field(50, ge=1, le=2000)
    initial_capital: float = Field(1_000_000.0, gt=0)
    use_coordinator: bool = True
    enable_events: bool = True
    random_seed: Optional[int] = 42


class SimulationResponse(BaseModel):
    total_return_pct: float
    final_portfolio_value: float
    performance_metrics: Dict[str, Any]
    decision_count: int
    sample_decision_id: Optional[str] = None


class CompareRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOG"])
    num_steps: int = Field(50, ge=1, le=2000)
    initial_capital: float = Field(1_000_000.0, gt=0)
    random_seed: Optional[int] = 42


def _build_system(payload: SimulationRequest) -> ResearchSystem:
    config = ResearchSystemConfig(
        symbols=payload.symbols,
        num_steps=payload.num_steps,
        initial_capital=payload.initial_capital,
        use_coordinator=payload.use_coordinator,
        enable_events=payload.enable_events,
        random_seed=payload.random_seed,
    )
    integrator = ResearchSystemIntegrator(config)
    return integrator.build_system()


def _log_decisions(system: ResearchSystem, result: Any) -> Optional[str]:
    sample_decision_id = None
    for decision in result.decision_log:
        action_str = decision.get("action")
        if action_str not in {"BUY", "SELL", "HOLD"}:
            continue

        action_type = ActionType(action_str)
        action = Action(
            action_type=action_type,
            symbol=decision.get("symbol"),
            quantity=float(decision.get("quantity", 0.0)),
            confidence=float(decision.get("confidence", 0.0)),
            reasoning_summary=str(decision.get("reasoning", "")),
        )

        chain = ReasoningChain(
            agent_id=str(decision.get("agent", "unknown")),
            decision_context=f"{action.action_type.value} {action.symbol or ''}".strip(),
        )
        if decision.get("reasoning"):
            chain.add_analysis(str(decision.get("reasoning")), data_source="agent_reasoning")
        chain.add_decision(
            decision=action.reasoning_summary or action.action_type.value,
            confidence=action.confidence,
            reasoning_summary=action.reasoning_summary or "",
        )

        audit_id = system.audit_trail.log_decision(
            agent_id=chain.agent_id,
            action=action,
            reasoning_chain=chain,
            market_state={"symbol": action.symbol, "step": decision.get("step")},
            portfolio_state={},
            constraints_checked=[],
        )
        if sample_decision_id is None:
            sample_decision_id = audit_id

    return sample_decision_id


def _find_audit_entry(system: ResearchSystem, decision_id: str) -> Optional[Any]:
    for entry in system.audit_trail._entries:
        if entry.audit_id == decision_id or entry.decision_id == decision_id:
            return entry
    return None


def _rebuild_chain(chain_dict: Dict[str, Any]) -> ReasoningChain:
    chain = ReasoningChain(
        chain_id=chain_dict.get("chain_id"),
        agent_id=chain_dict.get("agent_id", ""),
        decision_context=chain_dict.get("decision_context", ""),
    )
    steps = []
    for step in chain_dict.get("steps", []):
        steps.append(
            ReasoningStep(
                step_id=step.get("step_id", ""),
                observation=step.get("observation", ""),
                inference=step.get("inference", ""),
                confidence=float(step.get("confidence", 0.0)),
                data_source=step.get("data_source", ""),
                evidence=step.get("evidence", {}),
                step_type=step.get("step_type", "analysis"),
                parent_step_id=step.get("parent_step_id"),
                supporting_step_ids=step.get("supporting_step_ids", []),
            )
        )
    chain._steps = steps
    return chain


@router.post("/simulate", response_model=SimulationResponse)
def simulate(payload: SimulationRequest):
    system = _build_system(payload)
    result = system.simulation_runner.run(verbose=False)
    sample_decision_id = _log_decisions(system, result)

    with _lock:
        _system_cache["system"] = system
        _system_cache["last_result"] = result

    return SimulationResponse(
        total_return_pct=result.total_return_pct,
        final_portfolio_value=result.final_portfolio_value,
        performance_metrics=result.performance_metrics,
        decision_count=len(result.decision_log),
        sample_decision_id=sample_decision_id,
    )


@router.get("/metrics")
def get_metrics():
    with _lock:
        result = _system_cache.get("last_result")
    if result is None:
        raise HTTPException(status_code=404, detail="No simulation results available")

    return {
        "total_return_pct": result.total_return_pct,
        "final_portfolio_value": result.final_portfolio_value,
        "performance_metrics": result.performance_metrics,
        "agent_performance": result.agent_performance,
    }


@router.get("/explanation/{decision_id}")
def get_explanation(decision_id: str, format: str = "plain_text"):
    with _lock:
        system = _system_cache.get("system")

    if system is None:
        raise HTTPException(status_code=404, detail="No research system initialized")

    entry = _find_audit_entry(system, decision_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Decision not found")

    chain = _rebuild_chain(entry.reasoning_chain)
    fmt = ExplanationFormat(format) if format in ExplanationFormat._value2member_map_ else ExplanationFormat.PLAIN_TEXT
    rendered = system.explanation_renderer.render(chain, entry.action, format=fmt)

    return {
        "decision_id": entry.decision_id,
        "format": rendered.format.value,
        "content": rendered.content,
        "metadata": rendered.metadata,
    }


@router.post("/compare")
def compare(payload: CompareRequest):
    base_config = ResearchSystemConfig(
        symbols=payload.symbols,
        num_steps=payload.num_steps,
        initial_capital=payload.initial_capital,
        random_seed=payload.random_seed,
        use_coordinator=False,
        include_analyst_agent=False,
        include_risk_manager=False,
        include_portfolio_manager=False,
        include_multimodal_agent=False,
    )
    enhanced_config = ResearchSystemConfig(
        symbols=payload.symbols,
        num_steps=payload.num_steps,
        initial_capital=payload.initial_capital,
        random_seed=payload.random_seed,
        use_coordinator=True,
        include_analyst_agent=True,
        include_risk_manager=True,
        include_portfolio_manager=True,
        include_multimodal_agent=True,
    )

    base_system = ResearchSystemIntegrator(base_config).build_system()
    enhanced_system = ResearchSystemIntegrator(enhanced_config).build_system()

    base_result = base_system.simulation_runner.run(verbose=False)
    enhanced_result = enhanced_system.simulation_runner.run(verbose=False)

    comparison = ComparisonEngine().compare(base_result, enhanced_result)

    return {
        "base_metrics": comparison.base_metrics,
        "enhanced_metrics": comparison.enhanced_metrics,
        "improvements": comparison.improvements,
        "statistical_tests": comparison.statistical_tests,
        "summary": comparison.summary,
    }
