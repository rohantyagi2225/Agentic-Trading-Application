class RiskEngine:

    def __init__(
        self,
        max_position_pct: float = 0.1,
        max_drawdown_pct: float = 0.2,
        max_portfolio_exposure_pct: float = 0.5,
        stop_loss_pct: float = 0.05
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.stop_loss_pct = stop_loss_pct

    def validate_trade(self, portfolio_value: float, current_exposure: float, position_size: float):

        if portfolio_value <= 0:
            return False, "Invalid portfolio value"

        position_size = abs(position_size)

        # Per-position allocation limit
        if position_size > portfolio_value * self.max_position_pct:
            return False, "Position size exceeds max allocation"

        # Portfolio exposure limit
        if current_exposure + position_size > portfolio_value * self.max_portfolio_exposure_pct:
            return False, "Portfolio exposure limit exceeded"

        return True, "Trade approved"

    def calculate_stop_loss(self, entry_price: float) -> float:

        if entry_price <= 0:
            raise ValueError("Invalid entry price")

        return entry_price * (1 - self.stop_loss_pct)