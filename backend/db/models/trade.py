from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from backend.db.base import Base


class Trade(Base):

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)

    symbol = Column(String, index=True, nullable=False)

    action = Column(String, nullable=False)

    quantity = Column(Float, nullable=False)

    price = Column(Float, nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)