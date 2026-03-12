class DrawdownController:
    """
    Controls trading based on portfolio drawdown.
    Stops trading if max drawdown limit is breached.
    """

    def __init__(self, max_drawdown_pct: float = 0.2):
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_value = None

    def update(self, portfolio_value: float):

        if portfolio_value <= 0:
            raise ValueError("Invalid portfolio value")

        if self.peak_value is None:
            self.peak_value = portfolio_value

        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

    def is_trading_allowed(self, portfolio_value: float) -> bool:

        if self.peak_value is None:
            self.peak_value = portfolio_value
            return True

        drawdown = (self.peak_value - portfolio_value) / self.peak_value

        return drawdown < self.max_drawdown_pct