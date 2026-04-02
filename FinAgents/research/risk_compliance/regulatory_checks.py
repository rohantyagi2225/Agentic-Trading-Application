"""
Regulatory-Style Validation for Trading Compliance

This module provides regulatory checks for common trading rules including
wash sale detection, pattern day trading rules, short selling restrictions,
and position reporting requirements.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum


class TradeAction(Enum):
    """Enumeration of trade actions."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """
    Represents a historical trade for regulatory checking.
    
    Attributes:
        symbol: Stock ticker symbol
        action: BUY or SELL
        quantity: Number of shares traded
        price: Execution price per share
        timestamp: When the trade occurred
        trade_id: Unique identifier for the trade
    """
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    trade_id: str
    
    def __post_init__(self):
        """Validate action value."""
        if self.action not in ['BUY', 'SELL']:
            raise ValueError("action must be 'BUY' or 'SELL'")
    
    @property
    def value(self) -> float:
        """Calculate total trade value."""
        return self.quantity * self.price
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.action == 'BUY'
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.action == 'SELL'


@dataclass
class WashSaleResult:
    """
    Result of a wash sale check.
    
    Attributes:
        is_wash_sale: Whether a wash sale condition exists
        matching_trades: List of trades that trigger wash sale
        tax_implication: Description of tax implications
        details: Detailed explanation
    """
    is_wash_sale: bool
    matching_trades: List[Trade]
    tax_implication: str
    details: str


@dataclass
class PDTResult:
    """
    Result of a Pattern Day Trading check.
    
    Attributes:
        is_pdt: Whether account is flagged as pattern day trader
        day_trade_count: Number of day trades in window
        limit: Maximum allowed day trades
        details: Detailed explanation
    """
    is_pdt: bool
    day_trade_count: int
    limit: int
    details: str


@dataclass
class ShortSellResult:
    """
    Result of a short selling check.
    
    Attributes:
        is_short_sell: Whether the trade is a short sale
        allowed: Whether the short sale is permitted
        locate_required: Whether locate is required
        uptick_rule_applies: Whether uptick rule applies
        details: Detailed explanation
    """
    is_short_sell: bool
    allowed: bool
    locate_required: bool
    uptick_rule_applies: bool
    details: str


@dataclass
class ReportingResult:
    """
    Result of a position reporting check.
    
    Attributes:
        reporting_required: Whether reporting is required
        positions_above_threshold: List of positions exceeding threshold
        filing_type: Type of filing required
        details: Detailed explanation
    """
    reporting_required: bool
    positions_above_threshold: List[Dict[str, Any]]
    filing_type: str
    details: str


@dataclass
class RegulatoryReport:
    """
    Aggregated report of all regulatory checks.
    
    Attributes:
        timestamp: When the report was generated
        wash_sale_result: Wash sale check result
        pdt_result: Pattern day trading check result
        short_sell_result: Short sell check result
        reporting_result: Position reporting check result
        all_clear: True if no regulatory issues found
        warnings: List of warning messages
    """
    timestamp: datetime
    wash_sale_result: WashSaleResult
    pdt_result: PDTResult
    short_sell_result: ShortSellResult
    reporting_result: ReportingResult
    all_clear: bool
    warnings: List[str]


class RegulatoryChecker:
    """
    Checker for regulatory compliance rules.
    
    Provides methods to check various regulatory requirements including
    wash sale rules, pattern day trading restrictions, short selling
    rules, and position reporting thresholds.
    
    Attributes:
        config: Configuration dictionary for check parameters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the regulatory checker.
        
        Args:
            config: Optional configuration dict with keys:
                - wash_sale_window_days: Days for wash sale lookback (default 30)
                - pdt_window_days: Days for PDT lookback (default 5)
                - pdt_limit: Max day trades in window (default 3)
                - reporting_threshold_pct: Position reporting threshold (default 5.0)
        """
        self.config = config or {}
        self.wash_sale_window_days = self.config.get('wash_sale_window_days', 30)
        self.pdt_window_days = self.config.get('pdt_window_days', 5)
        self.pdt_limit = self.config.get('pdt_limit', 3)
        self.reporting_threshold_pct = self.config.get('reporting_threshold_pct', 5.0)
    
    def check_wash_sale(
        self, 
        trade_history: List[Trade], 
        proposed_trade: Trade,
        window_days: Optional[int] = None
    ) -> WashSaleResult:
        """
        Check for wash sale conditions.
        
        A wash sale occurs when a security is sold at a loss and the same
        or substantially identical security is repurchased within 30 days.
        
        Args:
            trade_history: Historical trades
            proposed_trade: Trade being considered
            window_days: Lookback window in days (default 30)
            
        Returns:
            WashSaleResult with wash sale status
        """
        window = window_days or self.wash_sale_window_days
        
        # Only check if proposed trade is a BUY
        if not proposed_trade.is_buy:
            return WashSaleResult(
                is_wash_sale=False,
                matching_trades=[],
                tax_implication="No wash sale - not a purchase",
                details="Wash sale rules only apply to repurchasing after a loss sale"
            )
        
        cutoff_date = proposed_trade.timestamp - timedelta(days=window)
        matching_trades: List[Trade] = []
        
        # Look for SELL trades of the same symbol at a loss within window
        for trade in trade_history:
            if (trade.symbol == proposed_trade.symbol and 
                trade.is_sell and 
                trade.timestamp >= cutoff_date and
                trade.timestamp < proposed_trade.timestamp):
                
                # For this check, we assume trades at a loss if we don't have cost basis
                # In practice, this would compare to the position's cost basis
                matching_trades.append(trade)
        
        is_wash_sale = len(matching_trades) > 0
        
        if is_wash_sale:
            total_loss_value = sum(t.value for t in matching_trades)
            tax_implication = (
                f"Wash sale detected. Loss of ${total_loss_value:,.2f} is disallowed "
                f"for tax purposes. The disallowed loss is added to the cost basis "
                f"of the new position."
            )
            details = (
                f"Found {len(matching_trades)} matching loss sale(s) within {window} days. "
                f"Symbol: {proposed_trade.symbol}. "
                f"Purchasing {proposed_trade.quantity} shares would trigger wash sale rules. "
                f"Consider waiting {(trade.timestamp + timedelta(days=window+1) - proposed_trade.timestamp).days} "
                f"more days to avoid wash sale."
            )
        else:
            tax_implication = "No wash sale condition detected"
            details = (
                f"No loss sales of {proposed_trade.symbol} found within {window} days. "
                f"Purchase is clear of wash sale rules."
            )
        
        return WashSaleResult(
            is_wash_sale=is_wash_sale,
            matching_trades=matching_trades,
            tax_implication=tax_implication,
            details=details
        )
    
    def check_pattern_day_trading(
        self, 
        trade_history: List[Trade], 
        window_days: Optional[int] = None
    ) -> PDTResult:
        """
        Check for Pattern Day Trading (PDT) status.
        
        A pattern day trader is defined as someone who executes 4 or more
        day trades within 5 business days.
        
        Args:
            trade_history: Historical trades
            window_days: Lookback window in days (default 5)
            
        Returns:
            PDTResult with PDT status and count
        """
        window = window_days or self.pdt_window_days
        
        if not trade_history:
            return PDTResult(
                is_pdt=False,
                day_trade_count=0,
                limit=self.pdt_limit,
                details="No trade history available"
            )
        
        # Get the most recent date in trade history
        latest_date = max(t.timestamp for t in trade_history)
        cutoff_date = latest_date - timedelta(days=window)
        
        # Group trades by day
        trades_by_day: Dict[datetime.date, List[Trade]] = {}
        for trade in trade_history:
            if trade.timestamp >= cutoff_date:
                trade_date = trade.timestamp.date()
                if trade_date not in trades_by_day:
                    trades_by_day[trade_date] = []
                trades_by_day[trade_date].append(trade)
        
        # Count day trades (round trip same day)
        day_trade_count = 0
        day_trade_details: List[str] = []
        
        for date, day_trades in trades_by_day.items():
            # Group by symbol
            symbol_trades: Dict[str, Dict[str, List[Trade]]] = {}
            for trade in day_trades:
                if trade.symbol not in symbol_trades:
                    symbol_trades[trade.symbol] = {'BUY': [], 'SELL': []}
                symbol_trades[trade.symbol][trade.action].append(trade)
            
            # Check for round trips
            for symbol, actions in symbol_trades.items():
                if actions['BUY'] and actions['SELL']:
                    day_trade_count += 1
                    day_trade_details.append(f"{symbol} on {date}")
        
        is_pdt = day_trade_count > self.pdt_limit
        
        if is_pdt:
            details = (
                f"PATTERN DAY TRADER STATUS: {day_trade_count} day trades in {window} days "
                f"(limit: {self.pdt_limit}). Day trades: {', '.join(day_trade_details)}. "
                f"Account flagged as Pattern Day Trader. Minimum $25,000 equity required to continue day trading."
            )
        elif day_trade_count >= self.pdt_limit - 1:
            details = (
                f"WARNING: {day_trade_count} day trades in {window} days. "
                f"One more day trade will flag account as Pattern Day Trader. "
                f"Recent day trades: {', '.join(day_trade_details[-3:])}"
            )
        else:
            details = (
                f"{day_trade_count} day trades in {window} days (limit: {self.pdt_limit}). "
                f"Account is clear of PDT restrictions."
            )
        
        return PDTResult(
            is_pdt=is_pdt,
            day_trade_count=day_trade_count,
            limit=self.pdt_limit,
            details=details
        )
    
    def check_short_selling(
        self, 
        proposed_trade: Trade, 
        portfolio_state: Any
    ) -> ShortSellResult:
        """
        Check short selling restrictions.
        
        Evaluates whether a proposed trade would create a short position
        and applies basic short selling rules including locate requirements
        and uptick rule considerations.
        
        Args:
            proposed_trade: Trade being considered
            portfolio_state: Current portfolio state with positions
            
        Returns:
            ShortSellResult with short sell status and restrictions
        """
        # Check if this is a sell that would exceed current position
        current_position = 0.0
        if hasattr(portfolio_state, 'positions'):
            for pos in portfolio_state.positions:
                if pos.symbol == proposed_trade.symbol:
                    current_position = pos.quantity
                    break
        
        # Calculate position after trade
        if proposed_trade.is_sell:
            position_after = current_position - proposed_trade.quantity
        else:
            position_after = current_position + proposed_trade.quantity
        
        is_short_sell = position_after < 0
        
        if not is_short_sell:
            return ShortSellResult(
                is_short_sell=False,
                allowed=True,
                locate_required=False,
                uptick_rule_applies=False,
                details=f"Trade does not create short position. Final position: {position_after} shares"
            )
        
        # Short selling rules
        short_quantity = abs(position_after)
        locate_required = True  # Always required for shorts
        
        # Uptick rule (short sale price test) - simplified
        # In practice, this compares to last sale price
        uptick_rule_applies = True
        
        allowed = True  # Assume allowed with proper locate
        
        details = (
            f"SHORT SELL DETECTED: Selling {proposed_trade.quantity} shares of "
            f"{proposed_trade.symbol} with current position of {current_position}. "
            f"Resulting short position: {short_quantity} shares. "
            f"REQUIREMENTS: (1) Locate/borrow required for {short_quantity} shares, "
            f"(2) Short sale price test (uptick rule) applies. "
            f"RESTRICTIONS: Short selling restricted during severe market declines."
        )
        
        return ShortSellResult(
            is_short_sell=True,
            allowed=allowed,
            locate_required=locate_required,
            uptick_rule_applies=uptick_rule_applies,
            details=details
        )
    
    def check_position_reporting(
        self, 
        portfolio_state: Any, 
        threshold_pct: Optional[float] = None
    ) -> ReportingResult:
        """
        Check if position reporting is required.
        
        Flags positions that exceed reporting thresholds (e.g., 5% of
        issuer's outstanding shares for 13D filing).
        
        Args:
            portfolio_state: Current portfolio state with positions
            threshold_pct: Reporting threshold percentage (default 5.0)
            
        Returns:
            ReportingResult with reporting requirements
        """
        threshold = threshold_pct or self.reporting_threshold_pct
        
        positions_above: List[Dict[str, Any]] = []
        
        if not hasattr(portfolio_state, 'positions'):
            return ReportingResult(
                reporting_required=False,
                positions_above_threshold=[],
                filing_type="None",
                details="No positions to check"
            )
        
        # Note: In practice, this would compare to issuer's outstanding shares
        # Here we use portfolio weight as a proxy for demonstration
        for pos in portfolio_state.positions:
            # Using portfolio weight as proxy - in reality would be % of issuer shares
            weight_pct = pos.weight_pct * 100  # Convert to percentage
            
            if weight_pct >= threshold:
                positions_above.append({
                    'symbol': pos.symbol,
                    'portfolio_weight_pct': weight_pct,
                    'shares': pos.quantity,
                    'market_value': pos.market_value if hasattr(pos, 'market_value') else pos.quantity * pos.current_price
                })
        
        reporting_required = len(positions_above) > 0
        
        if reporting_required:
            filing_type = "Schedule 13D (Beneficial Ownership Report)"
            details = (
                f"REPORTING REQUIRED: {len(positions_above)} position(s) exceed {threshold}% threshold. "
                f"Positions: {', '.join(p['symbol'] for p in positions_above)}. "
                f"FILING: {filing_type} must be filed within 10 days of crossing threshold. "
                f"ONGOING: Must amend filing for material changes. "
                f"DISCLOSURE: Position, purpose, and source of funds must be disclosed."
            )
        else:
            filing_type = "None"
            max_weight = max(
                (pos.weight_pct * 100 for pos in portfolio_state.positions),
                default=0
            )
            details = (
                f"No reporting required. Largest position: {max_weight:.2f}% "
                f"(threshold: {threshold}%). All positions below reporting threshold."
            )
        
        return ReportingResult(
            reporting_required=reporting_required,
            positions_above_threshold=positions_above,
            filing_type=filing_type,
            details=details
        )
    
    def run_all_regulatory_checks(
        self,
        trade_history: List[Trade],
        proposed_trade: Trade,
        portfolio_state: Any
    ) -> RegulatoryReport:
        """
        Run all regulatory checks and generate comprehensive report.
        
        Args:
            trade_history: Historical trades
            proposed_trade: Trade being considered
            portfolio_state: Current portfolio state
            
        Returns:
            RegulatoryReport with all check results
        """
        warnings: List[str] = []
        
        # Run all checks
        wash_sale_result = self.check_wash_sale(trade_history, proposed_trade)
        pdt_result = self.check_pattern_day_trading(trade_history)
        short_sell_result = self.check_short_selling(proposed_trade, portfolio_state)
        reporting_result = self.check_position_reporting(portfolio_state)
        
        # Collect warnings
        if wash_sale_result.is_wash_sale:
            warnings.append(f"Wash sale: {wash_sale_result.tax_implication}")
        
        if pdt_result.is_pdt:
            warnings.append(f"Pattern Day Trader: {pdt_result.day_trade_count} trades in {pdt_result.limit} day limit")
        elif pdt_result.day_trade_count >= self.pdt_limit - 1:
            warnings.append(f"Approaching PDT limit: {pdt_result.day_trade_count}/{self.pdt_limit} day trades")
        
        if short_sell_result.is_short_sell:
            warnings.append(f"Short sale: Locate required, uptick rule applies")
        
        if reporting_result.reporting_required:
            warnings.append(f"Reporting required: {len(reporting_result.positions_above_threshold)} positions above threshold")
        
        # Determine if all clear
        all_clear = (
            not wash_sale_result.is_wash_sale and
            not pdt_result.is_pdt and
            not short_sell_result.is_short_sell and
            not reporting_result.reporting_required and
            len(warnings) == 0
        )
        
        return RegulatoryReport(
            timestamp=datetime.now(),
            wash_sale_result=wash_sale_result,
            pdt_result=pdt_result,
            short_sell_result=short_sell_result,
            reporting_result=reporting_result,
            all_clear=all_clear,
            warnings=warnings
        )
