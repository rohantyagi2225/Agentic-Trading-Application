from sqlalchemy import Column, Integer, String, Float
from backend.db.base import Base


class PortfolioPosition(Base):

    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)

    symbol = Column(String, unique=True, index=True, nullable=False)

    quantity = Column(Float, nullable=False, default=0.0)