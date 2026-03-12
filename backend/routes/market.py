from fastapi import APIRouter
from backend.market.data_provider import MarketDataProvider


router = APIRouter(tags=["Market"])

provider = MarketDataProvider()


@router.get("/price/{symbol}")
def get_price(symbol: str):

    data = provider.get_latest_price(symbol)

    if not data:
        return {
            "status": "error",
            "message": "No data available"
        }

    return {
        "status": "success",
        "data": data
    }