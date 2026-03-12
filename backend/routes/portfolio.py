from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db, get_cache_service
from backend.services.portfolio_service import PortfolioService
from backend.cache.cache_service import CacheService


router = APIRouter(tags=["Portfolio"])


@router.get("/metrics")
def get_portfolio_metrics(
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
):
    """
    Return high-level portfolio metrics computed from current positions.
    Includes analytics (sharpe, sortino, volatility, max_drawdown) when
    enough trade history is available.
    """
    try:
        service = PortfolioService(db, cache=cache)
        summary = service.get_portfolio_summary_with_analytics()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "data": summary
    }