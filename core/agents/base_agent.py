from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """
    Production-grade base interface for all trading agents.
    """

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Must return:
        {
            "symbol": str,
            "action": "BUY" | "SELL" | "HOLD",
            "quantity": float,
            "price": float
        }
        """
        pass

    @abstractmethod
    def confidence_score(self) -> float:
        """
        Return value between 0 and 1.
        """
        pass

    @abstractmethod
    def explain_decision(self) -> str:
        """
        Human-readable explanation.
        """
        pass