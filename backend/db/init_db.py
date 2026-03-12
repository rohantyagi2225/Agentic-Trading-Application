from backend.db.base import Base
from backend.db.session import engine

from backend.db.models.trade import Trade
from backend.db.models.portfolio_position import PortfolioPosition


def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()