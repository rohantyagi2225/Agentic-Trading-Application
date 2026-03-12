class PositionSizer:
    """
    Determines position size based on portfolio value and risk limits.
    """

    def __init__(self, max_position_pct: float = 0.1):
        self.max_position_pct = max_position_pct

    def size_position(self, portfolio_value: float, price: float) -> float:

        if portfolio_value <= 0:
            raise ValueError("Invalid portfolio value")

        if price <= 0:
            raise ValueError("Invalid price")

        max_position_value = portfolio_value * self.max_position_pct

        quantity = max_position_value / price

        return round(quantity, 4)