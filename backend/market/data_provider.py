from typing import Dict, Any, Optional
import yfinance as yf

from backend.cache.cache_service import CacheService


class MarketDataProvider:
    """
    Production-ready market data provider.
    Fetches latest price from Yahoo Finance.
    Optional Redis cache to reduce external calls.
    """

    def __init__(self, cache: Optional[CacheService] = None) -> None:
        self._cache = cache

    def get_latest_price(self, symbol: str) -> Dict[str, Any]:

        if not symbol:
            raise ValueError("Symbol cannot be empty")

        symbol = symbol.upper()

        if self._cache and self._cache.is_available:
            cached = self._cache.get_market_data(symbol)
            if cached is not None:
                return cached

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")

            if data.empty:
                return {}

            latest = data.iloc[-1]

            result = {
                "symbol": symbol,
                "price": float(latest["Close"]),
            }
            if self._cache and self._cache.is_available:
                self._cache.set_market_data(symbol, result)
            return result

        except Exception:
            return {}