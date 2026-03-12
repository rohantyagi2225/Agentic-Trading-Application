from typing import Dict, Any, Optional

from backend.market.data_provider import MarketDataProvider
from backend.risk.risk_engine import RiskEngine
from backend.cache.cache_service import CacheService
from core.agents.base_agent import BaseAgent


class LiveSignalService:

    VALID_ACTIONS = {"BUY", "SELL", "HOLD"}

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        risk_engine: RiskEngine | None = None,
        cache: Optional[CacheService] = None,
    ) -> None:
        self._market_data_provider = market_data_provider
        self._risk_engine = risk_engine or RiskEngine()
        self._cache = cache

    def get_live_signal(
        self,
        symbol: str,
        agent: BaseAgent,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        if self._cache and self._cache.is_available and agent_name:
            cached = self._cache.get_signal(symbol, agent_name)
            if cached is not None:
                return cached

        market_data = self._market_data_provider.get_latest_price(symbol)

        if not market_data:
            raise ValueError(f"No market data available for symbol '{symbol}'")

        raw_signal = agent.generate_signal(market_data)

        if not isinstance(raw_signal, dict):
            raise ValueError("Agent generate_signal must return a dict")

        price = float(raw_signal.get("price", market_data.get("price")))
        action = str(raw_signal.get("action", "")).upper()
        quantity = float(raw_signal.get("quantity", 0.0))

        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid signal action '{action}'")

        confidence = float(agent.confidence_score())
        confidence = max(0.0, min(1.0, confidence))

        explanation = str(agent.explain_decision())

        # Risk validation
        position_size = price * quantity

        approved, reason = self._risk_engine.validate_trade(
            portfolio_value=100000,     # TODO replace with PortfolioService
            current_exposure=0,
            position_size=position_size,
        )

        if not approved:
            return {
                "symbol": symbol,
                "price": price,
                "signal": "REJECTED",
                "confidence": 0.0,
                "explanation": f"Risk rejected trade: {reason}",
            }

        out = {
            "symbol": symbol,
            "price": price,
            "signal": action,
            "confidence": confidence,
            "explanation": explanation,
        }
        if self._cache and self._cache.is_available and agent_name:
            self._cache.set_signal(symbol, agent_name, out)
        return out