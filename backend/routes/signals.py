from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.dependencies import get_live_signal_service
from backend.services.live_signal_service import LiveSignalService
from core.agents.base_agent import BaseAgent


router = APIRouter(tags=["Signals"])


class SignalResponse(BaseModel):
    status: str
    data: Dict[str, Any]


class DummyAgent(BaseAgent):

    def __init__(self) -> None:
        self._last_explanation: str = "Initial dummy decision."
        self._last_confidence: float = 0.5

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:

        symbol = str(market_data.get("symbol") or "UNKNOWN")
        price = float(market_data.get("price") or 0.0)

        if price > 0:
            action = "BUY"
            self._last_confidence = 0.7
            self._last_explanation = "DummyAgent: positive price, BUY signal."
        else:
            action = "HOLD"
            self._last_confidence = 0.4
            self._last_explanation = "DummyAgent: non-positive price, HOLD."

        return {
            "symbol": symbol,
            "action": action,
            "quantity": 1.0,
            "price": price,
        }

    def confidence_score(self) -> float:
        return self._last_confidence

    def explain_decision(self) -> str:
        return self._last_explanation


_agent = DummyAgent()


@router.get("/{symbol}", response_model=SignalResponse)
def get_live_signal(
    symbol: str,
    service: LiveSignalService = Depends(get_live_signal_service),
):

    symbol = symbol.upper()

    try:
        signal = service.get_live_signal(symbol, _agent, agent_name="dummy")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "success",
        "data": signal
    }