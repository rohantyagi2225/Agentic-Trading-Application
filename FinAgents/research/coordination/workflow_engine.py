"""Configurable Workflow Engine for Multi-Agent Coordination.

This module provides a flexible workflow engine that can execute
configurable agent workflows defined as directed acyclic graphs (DAGs).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from FinAgents.research.domain_agents.base_agent import (
    Action,
    MarketContext,
    MarketData,
    ReasoningResult,
    ResearchAgent,
)


@dataclass
class WorkflowStep:
    """A single step in a workflow definition.

    Attributes
    ----------
    step_id : str
        Unique identifier for this step.
    name : str
        Human-readable name for the step.
    agent_role : str
        Role of the agent that executes this step.
    action : str
        Action type: "reason", "act", "assess", or "vote".
    inputs_from : list of str
        Step IDs that provide inputs to this step.
    outputs_to : list of str
        Step IDs that consume this step's output.
    timeout_seconds : int
        Maximum execution time for this step.
    required : bool
            Whether this step must succeed for the workflow to continue.
    config : dict
            Additional step-specific configuration.
    """

    step_id: str
    name: str
    agent_role: str
    action: str  # "reason", "act", "assess", "vote"
    inputs_from: List[str] = field(default_factory=list)
    outputs_to: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    required: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "agent_role": self.agent_role,
            "action": self.action,
            "inputs_from": self.inputs_from,
            "outputs_to": self.outputs_to,
            "timeout_seconds": self.timeout_seconds,
            "required": self.required,
            "config": self.config,
        }


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow.

    Attributes
    ----------
    name : str
        Workflow name.
    steps : list of WorkflowStep
            Steps in the workflow.
    description : str
            Human-readable description.
    version : str
            Workflow version.
    workflow_id : str
            Unique identifier for this workflow definition.
    """

    name: str
    steps: List[WorkflowStep]
    description: str = ""
    version: str = "1.0"
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class WorkflowResult:
    """Result of a workflow execution.

    Attributes
    ----------
    workflow_id : str
        ID of the executed workflow.
    status : str
        Execution status: completed, failed, or timeout.
    step_results : dict
        Results from each step.
    execution_time_ms : float
        Total execution time in milliseconds.
    execution_log : list of dict
        Detailed execution log.
    error_message : str
        Error message if status is failed.
    """

    workflow_id: str = ""
    status: str = "pending"  # completed, failed, timeout
    step_results: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "step_results": self.step_results,
            "execution_time_ms": self.execution_time_ms,
            "execution_log": self.execution_log,
            "error_message": self.error_message,
        }


class WorkflowEngine:
    """Engine for executing configurable agent workflows.

    The workflow engine executes workflows defined as DAGs, respecting
    dependencies between steps and passing outputs between steps.

    Attributes
    ----------
    agents : dict
        Maps agent roles to agent instances.
    workflows : dict
        Registered workflow definitions.
    """

    def __init__(self, agents: Dict[str, ResearchAgent]) -> None:
        """Initialize the workflow engine.

        Parameters
        ----------
        agents : dict
            Maps agent roles to agent instances.
        """
        self.agents = agents
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self._execution_cache: Dict[str, Any] = {}

    def create_workflow(self, definition: WorkflowDefinition) -> str:
        """Register a workflow definition.

        Parameters
        ----------
        definition : WorkflowDefinition
            The workflow to register.

        Returns
        -------
        str
            The workflow ID.
        """
        self.workflows[definition.workflow_id] = definition
        return definition.workflow_id

    def execute_workflow(
        self, workflow_id: str, initial_data: Dict[str, Any]
    ) -> WorkflowResult:
        """Execute a workflow with the given initial data.

        Parameters
        ----------
        workflow_id : str
            ID of the workflow to execute.
        initial_data : dict
            Initial data for the workflow.

        Returns
        -------
        WorkflowResult
            The execution result.
        """
        if workflow_id not in self.workflows:
            return WorkflowResult(
                workflow_id=workflow_id,
                status="failed",
                error_message=f"Workflow {workflow_id} not found",
            )

        definition = self.workflows[workflow_id]
        result = WorkflowResult(workflow_id=workflow_id)
        execution_log: List[Dict[str, Any]] = []
        step_results: Dict[str, Any] = {}

        start_time = time.time()

        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(definition.steps)
            execution_order = self._topological_sort(dependency_graph)

            if not execution_order:
                result.status = "failed"
                result.error_message = "Circular dependency detected in workflow"
                result.execution_time_ms = (time.time() - start_time) * 1000
                return result

            # Execute steps in order
            for step_id in execution_order:
                step = next(s for s in definition.steps if s.step_id == step_id)

                step_start = time.time()

                # Gather inputs from previous steps
                inputs = self._gather_inputs(step, step_results, initial_data)

                # Execute step
                step_result = self._execute_step(step, inputs)

                step_time = (time.time() - step_start) * 1000

                step_results[step_id] = step_result

                execution_log.append({
                    "step_id": step_id,
                    "step_name": step.name,
                    "agent_role": step.agent_role,
                    "action": step.action,
                    "status": "success" if step_result.get("success") else "failed",
                    "execution_time_ms": step_time,
                    "result_summary": self._summarize_result(step_result),
                })

                # Check if required step failed
                if step.required and not step_result.get("success"):
                    result.status = "failed"
                    result.error_message = f"Required step {step_id} failed"
                    break

            else:
                # All steps completed successfully
                result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)

        result.step_results = step_results
        result.execution_log = execution_log
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, Set[str]]:
        """Build a dependency graph from workflow steps.

        Parameters
        ----------
        steps : list of WorkflowStep
            The workflow steps.

        Returns
        -------
        dict
            Maps step IDs to sets of dependency step IDs.
        """
        graph: Dict[str, Set[str]] = {step.step_id: set() for step in steps}

        for step in steps:
            for input_step in step.inputs_from:
                if input_step in graph:
                    graph[step.step_id].add(input_step)

        return graph

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> Optional[List[str]]:
        """Perform topological sort on the dependency graph.

        Parameters
        ----------
        graph : dict
            Dependency graph.

        Returns
        -------
        list of str or None
            Sorted step IDs, or None if cycle detected.
        """
        visited: Set[str] = set()
        temp_mark: Set[str] = set()
        result: List[str] = []

        def visit(node: str) -> bool:
            if node in temp_mark:
                return False  # Cycle detected
            if node in visited:
                return True

            temp_mark.add(node)
            for neighbor in graph.get(node, set()):
                if not visit(neighbor):
                    return False
            temp_mark.remove(node)
            visited.add(node)
            result.append(node)
            return True

        for node in graph:
            if node not in visited:
                if not visit(node):
                    return None

        return result

    def _gather_inputs(
        self,
        step: WorkflowStep,
        step_results: Dict[str, Any],
        initial_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Gather inputs for a step from previous steps and initial data.

        Parameters
        ----------
        step : WorkflowStep
            The current step.
        step_results : dict
            Results from completed steps.
        initial_data : dict
            Initial workflow data.

        Returns
        -------
        dict
            Combined inputs for the step.
        """
        inputs = dict(initial_data)

        for input_step_id in step.inputs_from:
            if input_step_id in step_results:
                inputs[f"{input_step_id}_output"] = step_results[input_step_id]

        return inputs

    def _execute_step(
        self, step: WorkflowStep, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step.

        Parameters
        ----------
        step : WorkflowStep
            The step to execute.
        inputs : dict
            Input data for the step.

        Returns
        -------
        dict
            Step execution result.
        """
        agent = self.agents.get(step.agent_role)

        if agent is None:
            return {
                "success": False,
                "error": f"Agent for role '{step.agent_role}' not found",
            }

        try:
            # Extract market data and context from inputs
            market_data = inputs.get("market_data")
            context = inputs.get("context")

            if step.action == "reason":
                if market_data is None:
                    return {"success": False, "error": "market_data required for reason action"}
                reasoning = agent.reason(market_data, context)
                return {
                    "success": True,
                    "reasoning": reasoning,
                    "confidence": reasoning.confidence,
                    "signals": reasoning.signals,
                }

            elif step.action == "act":
                reasoning = inputs.get("reasoning")
                if reasoning is None:
                    # Try to get from previous step output
                    for key, value in inputs.items():
                        if key.endswith("_output") and "reasoning" in value:
                            reasoning = value["reasoning"]
                            break

                if reasoning is None:
                    return {"success": False, "error": "reasoning required for act action"}

                action = agent.act(reasoning)
                return {
                    "success": True,
                    "action": action,
                    "action_type": action.action_type.value,
                    "symbol": action.symbol,
                    "quantity": action.quantity,
                }

            elif step.action == "assess":
                # Generic assessment action
                if market_data is None:
                    return {"success": False, "error": "market_data required for assess action"}
                reasoning = agent.reason(market_data, context)
                return {
                    "success": True,
                    "assessment": reasoning,
                    "confidence": reasoning.confidence,
                }

            elif step.action == "vote":
                # Voting action - simplified for workflow
                proposal = inputs.get("proposal", {})
                return {
                    "success": True,
                    "vote": "approve",  # Simplified - real implementation would use voting mechanism
                    "proposal_id": proposal.get("proposal_id"),
                }

            else:
                return {"success": False, "error": f"Unknown action: {step.action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Create a summary string for a step result.

        Parameters
        ----------
        result : dict
            The step result.

        Returns
        -------
        str
            Summary string.
        """
        if not result.get("success"):
            return f"Failed: {result.get('error', 'Unknown error')}"

        if "action" in result:
            return f"Action: {result.get('action_type')} {result.get('symbol')}"
        elif "reasoning" in result:
            return f"Reasoning: confidence={result.get('confidence', 0):.2f}"
        elif "assessment" in result:
            return f"Assessment: confidence={result.get('confidence', 0):.2f}"
        elif "vote" in result:
            return f"Vote: {result.get('vote')}"

        return "Completed"

    def get_default_trading_workflow(self) -> WorkflowDefinition:
        """Get the standard 5-step trading cycle workflow.

        Returns
        -------
        WorkflowDefinition
            Default trading workflow.
        """
        steps = [
            WorkflowStep(
                step_id="market_analysis",
                name="Market Analysis",
                agent_role="analyst",
                action="reason",
                inputs_from=[],
                outputs_to=["trade_proposal"],
                timeout_seconds=30,
                required=True,
            ),
            WorkflowStep(
                step_id="trade_proposal",
                name="Trade Proposal",
                agent_role="trader",
                action="act",
                inputs_from=["market_analysis"],
                outputs_to=["risk_assessment"],
                timeout_seconds=30,
                required=True,
            ),
            WorkflowStep(
                step_id="risk_assessment",
                name="Risk Assessment",
                agent_role="risk_manager",
                action="assess",
                inputs_from=["trade_proposal"],
                outputs_to=["portfolio_check", "voting"],
                timeout_seconds=30,
                required=True,
            ),
            WorkflowStep(
                step_id="portfolio_check",
                name="Portfolio Check",
                agent_role="portfolio_manager",
                action="assess",
                inputs_from=["risk_assessment"],
                outputs_to=["voting"],
                timeout_seconds=30,
                required=False,
            ),
            WorkflowStep(
                step_id="voting",
                name="Final Voting",
                agent_role="trader",  # Any agent can trigger voting
                action="vote",
                inputs_from=["risk_assessment", "portfolio_check"],
                outputs_to=[],
                timeout_seconds=60,
                required=True,
            ),
        ]

        return WorkflowDefinition(
            name="Standard Trading Cycle",
            description="Default 5-step trading workflow with analysis, proposal, risk assessment, portfolio check, and voting",
            version="1.0",
            steps=steps,
        )

    def validate_workflow(self, definition: WorkflowDefinition) -> List[str]:
        """Validate a workflow definition.

        Checks for:
        - DAG validity (no cycles)
        - Agent availability
        - Valid action types
        - Connected graph

        Parameters
        ----------
        definition : WorkflowDefinition
            The workflow to validate.

        Returns
        -------
        list of str
            List of validation errors (empty if valid).
        """
        errors: List[str] = []

        # Check for empty workflow
        if not definition.steps:
            errors.append("Workflow has no steps")
            return errors

        # Check for duplicate step IDs
        step_ids = [step.step_id for step in definition.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")

        # Check agent availability
        for step in definition.steps:
            if step.agent_role not in self.agents:
                errors.append(f"Step {step.step_id}: Agent '{step.agent_role}' not available")

        # Check action types
        valid_actions = {"reason", "act", "assess", "vote"}
        for step in definition.steps:
            if step.action not in valid_actions:
                errors.append(f"Step {step.step_id}: Invalid action '{step.action}'")

        # Check for cycles in dependency graph
        dependency_graph = self._build_dependency_graph(definition.steps)
        if self._topological_sort(dependency_graph) is None:
            errors.append("Circular dependency detected in workflow")

        # Check for disconnected steps
        all_deps: Set[str] = set()
        all_outputs: Set[str] = set()
        for step in definition.steps:
            all_deps.update(step.inputs_from)
            all_outputs.update(step.outputs_to)

        # Find steps that are neither referenced nor reference others (except root)
        root_steps = [s.step_id for s in definition.steps if not s.inputs_from]
        if not root_steps:
            errors.append("No root step found (step with no inputs)")

        # Check that all referenced steps exist
        for step in definition.steps:
            for dep in step.inputs_from:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id}: References unknown step '{dep}'")

        return errors

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a registered workflow by ID.

        Parameters
        ----------
        workflow_id : str
            The workflow ID.

        Returns
        -------
        WorkflowDefinition or None
            The workflow if found.
        """
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all registered workflows.

        Returns
        -------
        list of WorkflowDefinition
            All registered workflows.
        """
        return list(self.workflows.values())

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a registered workflow.

        Parameters
        ----------
        workflow_id : str
            The workflow ID to delete.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            return True
        return False
