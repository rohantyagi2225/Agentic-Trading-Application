from fastapi import Depends
from sqlalchemy.orm import Session

from backend.config.settings import get_settings, Settings
from backend.db.session import SessionLocal
from backend.cache.cache_service import CacheService
from backend.market.data_provider import MarketDataProvider
from backend.risk.risk_engine import RiskEngine
from backend.services.live_signal_service import LiveSignalService


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_cache_service() -> CacheService:
    settings = get_settings()
    return CacheService(redis_url=settings.REDIS_URL)


def get_market_data_provider(
    cache: CacheService = Depends(get_cache_service),
) -> MarketDataProvider:
    return MarketDataProvider(cache=cache)


def get_risk_engine() -> RiskEngine:
    return RiskEngine()


def get_live_signal_service(
    provider: MarketDataProvider = Depends(get_market_data_provider),
    risk_engine: RiskEngine = Depends(get_risk_engine),
    cache: CacheService = Depends(get_cache_service),
) -> LiveSignalService:
    return LiveSignalService(provider, risk_engine, cache=cache)


def get_app_settings() -> Settings:
    return get_settings()