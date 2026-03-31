"""Demo runner for the research-grade FinAgents system."""

from __future__ import annotations

from typing import Any, Dict, List

from FinAgents.research.domain_agents.base_agent import ActionType
from FinAgents.research.integration.system_integrator import (
    ResearchSystemConfig,
    ResearchSystemIntegrator,
)


def run_demo(
    num_steps: int = 50,
    symbols: List[str] | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a step-by-step simulation demo and return summary results."""
    symbols = symbols or ["AAPL", "MSFT", "GOOG"]
    config = ResearchSystemConfig(symbols=symbols, num_steps=num_steps)
    integrator = ResearchSystemIntegrator(config)
    system = integrator.build_system()

    if verbose:
        print("=" * 80)
        print("FINAGENTS RESEARCH-GRADE DEMO")
        print("=" * 80)
        print(f"Symbols: {', '.join(symbols)} | Steps: {num_steps}")
        print("Initializing agents, coordination, memory, explainability, and simulation...")

    # Run one coordinated trading cycle for an explanation sample
    if system.coordinator is not None:
        sample_symbol = symbols[0]
        market_data = system.simulation_runner.market.get_market_data(sample_symbol)
        context = system.simulation_runner._create_market_context(sample_symbol)
        cycle = system.coordinator.run_trading_cycle(market_data, context)
        if verbose:
            print("\nSample Trading Cycle:")
            print(system.coordinator.get_cycle_explanation(cycle))

    # Run full simulation
    result = system.simulation_runner.run(verbose=verbose)

    # Build sample explanation from trader agent
    trader = system.agents.get("trader")
    explanation_payload = {}
    if trader is not None:
        market_data = system.simulation_runner.market.get_market_data(symbols[0])
        context = system.simulation_runner._create_market_context(symbols[0])
        reasoning = trader.reason(market_data, context)
        action = trader.act(reasoning)
        chain = integrator.build_reasoning_chain(trader.agent_id, reasoning, action)
        explanation = system.explanation_renderer.render(chain, action)
        explanation_payload = {
            "action": action.action_type.value,
            "symbol": action.symbol,
            "confidence": action.confidence,
            "explanation": explanation.content,
        }

        if verbose:
            print("\nSample Explanation:")
            print(explanation.content)

            if action.action_type == ActionType.HOLD:
                print("Note: Sample explanation based on HOLD action.")

    summary = {
        "total_return_pct": result.total_return_pct,
        "final_portfolio_value": result.final_portfolio_value,
        "performance_metrics": result.performance_metrics,
        "agent_performance": result.agent_performance,
        "sample_explanation": explanation_payload,
    }

    if verbose:
        print("\nSimulation Summary:")
        print(f"Total Return: {result.total_return_pct * 100:.2f}%")
        print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
        print("Demo complete.")

    return summary


if __name__ == "__main__":
    run_demo()
