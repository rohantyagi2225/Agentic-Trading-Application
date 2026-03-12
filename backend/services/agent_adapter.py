from core.agents.base_agent import BaseAgent


class AgentAdapter(BaseAgent):
    """
    Wraps research agents to enforce production interface.
    """

    def __init__(self, research_agent):
        super().__init__()
        self.research_agent = research_agent
        self._last_signal = None

    def generate_signal(self, market_data):
        if not hasattr(self.research_agent, "generate_signal"):
            raise TypeError("Research agent must implement generate_signal")

        signal = self.research_agent.generate_signal(market_data)

        if not isinstance(signal, dict):
            raise ValueError("Research agent must return a dict signal")

        self._last_signal = signal
        return signal

    def confidence_score(self) -> float:
        return 0.75

    def explain_decision(self) -> str:
        if not self._last_signal:
            return "No signal generated yet."

        return f"Generated signal: {self._last_signal}"