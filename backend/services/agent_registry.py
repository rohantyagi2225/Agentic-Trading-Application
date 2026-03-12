from typing import Dict, Type, List
from core.agents.base_agent import BaseAgent


class AgentRegistry:
    """
    Central registry for all trading agents.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseAgent]] = {}

    def register(self, name: str, agent_cls: Type[BaseAgent]) -> None:
        key = name.lower()

        if key in self._registry:
            raise ValueError(f"Agent '{name}' already registered.")

        if not issubclass(agent_cls, BaseAgent):
            raise TypeError("agent_cls must inherit from BaseAgent")

        self._registry[key] = agent_cls

    def get_agent(self, name: str) -> BaseAgent:
        agent_cls = self._registry.get(name.lower())

        if not agent_cls:
            raise ValueError(f"Agent '{name}' is not registered.")

        return agent_cls()

    def list_agents(self) -> List[str]:
        return list(self._registry.keys())