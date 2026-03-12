import asyncio
from typing import Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config.settings import get_settings
from backend.cache.cache_service import CacheService
from backend.market.data_provider import MarketDataProvider
from backend.services.live_signal_service import LiveSignalService
from backend.risk.risk_engine import RiskEngine
from backend.routes.signals import DummyAgent


router = APIRouter(tags=["WebSocket Signals"])


def _get_cached_signal_service() -> LiveSignalService:
    settings = get_settings()
    cache = CacheService(redis_url=settings.REDIS_URL)
    provider = MarketDataProvider(cache=cache)
    return LiveSignalService(provider, risk_engine=RiskEngine(), cache=cache)


_service = _get_cached_signal_service()
_agent = DummyAgent()


@router.websocket("/signals/{symbol}")
async def websocket_signals(websocket: WebSocket, symbol: str) -> None:

    symbol = symbol.upper()

    await websocket.accept()

    try:

        while True:

            signal: Dict[str, Any] = _service.get_live_signal(
                symbol, _agent, agent_name="dummy"
            )

            await websocket.send_json(
                {
                    "status": "success",
                    "data": signal,
                }
            )

            await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        await websocket.close()