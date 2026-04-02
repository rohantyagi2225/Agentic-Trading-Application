"""Research-grade extensions for the FinAgents platform.

This package hosts the next-generation research modules introduced in the
upgrade plan (domain agents, multimodal intelligence, coordination, learning,
explainability, simulation, data pipeline, risk/compliance, evaluation, and
integration tools).
"""

from FinAgents.research import (
    coordination,
    data_pipeline,
    domain_agents,
    evaluation,
    explainability,
    integration,
    memory_learning,
    multimodal,
    risk_compliance,
    simulation,
)

__all__ = [
    "coordination",
    "data_pipeline",
    "domain_agents",
    "evaluation",
    "explainability",
    "integration",
    "memory_learning",
    "multimodal",
    "risk_compliance",
    "simulation",
]
