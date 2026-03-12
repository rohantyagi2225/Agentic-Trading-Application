from pydantic import BaseModel, Field, field_validator
from typing import Literal


class TradeRequest(BaseModel):

    symbol: str = Field(..., example="AAPL")

    action: Literal["BUY", "SELL"]

    quantity: float = Field(..., gt=0)

    price: float = Field(..., gt=0)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.upper()


class TradeResponse(BaseModel):

    status: Literal["executed", "rejected", "error"]

    trade_id: int | None = None

    reason: str | None = None