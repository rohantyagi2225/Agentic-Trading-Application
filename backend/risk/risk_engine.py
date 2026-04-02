"""
Risk Engine - Core Risk Management System

This module provides comprehensive risk controls for trading decisions:
- Position sizing limits
- Portfolio exposure monitoring
- Stop-loss calculations
- Drawdown protection
- Integration with circuit breakers

Author: FinAgent Team
Version: 2.0.0 (Enhanced with circuit breaker integration)
"""

import logging
from typing import Tuple, Dict, Any
from backend.risk.circuit_breaker import TradingCircuitBreaker

logger = logging.getLogger("RiskEngine")


class RiskEngine:
    """
    Enhanced risk engine with circuit breaker integration
    
    Provides:
    1. Pre-trade validation
    2. Position sizing
    3. Stop-loss calculation
    4. Circuit breaker monitoring
    5. Risk-adjusted position sizing
    """

    def __init__(
        self,
        max_position_pct: float = 0.1,
        max_drawdown_pct: float = 0.2,
        max_portfolio_exposure_pct: float = 0.5,
        stop_loss_pct: float = 0.05,
        enable_circuit_breakers: bool = True
    ):
        """
        Initialize risk engine
        
        Args:
            max_position_pct: Maximum % of capital per position
            max_drawdown_pct: Maximum allowed drawdown
            max_portfolio_exposure_pct: Maximum total exposure
            stop_loss_pct: Default stop-loss percentage
            enable_circuit_breakers: Enable automatic trading halts
        """
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Optional circuit breaker integration
        self.circuit_breaker = None
        if enable_circuit_breakers:
            self.circuit_breaker = TradingCircuitBreaker(
                max_daily_loss_pct=max_drawdown_pct * 0.5,  # Daily limit is half of total
                max_drawdown_pct=max_drawdown_pct,
                enable_auto_liquidation=False
            )
            logger.info("✅ Circuit breaker integration enabled")
        
        logger.info(f"✅ Risk engine initialized")
        logger.debug(f"   Max Position: {max_position_pct:.1%}")
        logger.debug(f"   Max Drawdown: {max_drawdown_pct:.1%}")
        logger.debug(f"   Stop Loss: {stop_loss_pct:.1%}")

    def validate_trade(
        self, 
        portfolio_value: float, 
        current_exposure: float, 
        position_size: float,
        symbol: str = "UNKNOWN"
    ) -> Tuple[bool, str]:
        """
        Validate a trade with comprehensive risk checks
        
        Args:
            portfolio_value: Total portfolio value
            current_exposure: Current total exposure
            position_size: Proposed position size
            symbol: Trading symbol (for logging)
            
        Returns:
            (approved, reason): Tuple of approval status and reason
        """
        try:
            # Basic validation
            if portfolio_value <= 0:
                logger.error(f"Invalid portfolio value: ${portfolio_value:.2f}")
                return False, "Invalid portfolio value (must be > 0)"

            position_size = abs(position_size)
            
            # Check circuit breaker first (if enabled)
            if self.circuit_breaker:
                cb_allowed, cb_reason = self.circuit_breaker.should_allow_trade()
                if not cb_allowed:
                    logger.warning(f"Trade blocked by circuit breaker: {cb_reason}")
                    return False, f"Circuit breaker active: {cb_reason}"

            # Per-position allocation limit
            max_position = portfolio_value * self.max_position_pct
            if position_size > max_position:
                reason = f"Position size ${position_size:.2f} exceeds max ${max_position:.2f} ({self.max_position_pct:.1%})"
                logger.warning(reason)
                return False, reason

            # Portfolio exposure limit
            max_exposure = portfolio_value * self.max_portfolio_exposure_pct
            if current_exposure + position_size > max_exposure:
                reason = f"Portfolio exposure ${current_exposure + position_size:.2f} exceeds max ${max_exposure:.2f}"
                logger.warning(reason)
                return False, reason
            
            # All checks passed
            logger.info(f"✓ Trade approved for {symbol}: ${position_size:.2f}")
            return True, "Trade approved"
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}", exc_info=True)
            return False, f"Risk validation error: {str(e)}"

    def calculate_stop_loss(self, entry_price: float, direction: int = 1) -> float:
        """
        Calculate stop-loss price
        
        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            
        Returns:
            stop_loss_price: Stop-loss price level
        """
        if entry_price <= 0:
            raise ValueError("Invalid entry price")
        
        if direction == 1:  # Long position
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        else:  # Short position
            stop_loss = entry_price * (1 + self.stop_loss_pct)
        
        logger.debug(f"Stop loss calculated: ${stop_loss:.2f} (direction={direction})")
        return stop_loss

    def calculate_position_size(
        self,
        portfolio_value: float,
        volatility: float = 0.20,
        risk_adjustment: float = 1.0
    ) -> float:
        """
        Calculate risk-adjusted position size
        
        Args:
            portfolio_value: Total portfolio value
            volatility: Asset volatility (higher = smaller position)
            risk_adjustment: Additional adjustment factor (0.0-1.0)
            
        Returns:
            position_size: Recommended position size in dollars
        """
        # Base position size
        base_size = portfolio_value * self.max_position_pct
        
        # Volatility adjustment (reduce size in high vol)
        vol_target = 0.15  # Target 15% annualized vol
        vol_adjustment = min(vol_target / max(volatility, 0.01), 2.0)
        
        # Apply adjustments
        adjusted_size = base_size * vol_adjustment * risk_adjustment
        
        # Apply bounds
        min_size = portfolio_value * 0.01  # Minimum 1%
        max_size = portfolio_value * self.max_position_pct
        
        final_size = max(min_size, min(adjusted_size, max_size))
        
        logger.debug(f"Position size calc: base=${base_size:.2f}, "
                    f"vol_adj={vol_adjustment:.2f}, final=${final_size:.2f}")
        
        return final_size

    def record_trade_pnl(self, pnl: float):
        """
        Record a completed trade P&L for circuit breaker monitoring
        
        Args:
            pnl: Profit/Loss from the trade
        """
        if self.circuit_breaker:
            self.circuit_breaker.record_trade(pnl)
            logger.info(f"Recorded trade P&L: ${pnl:.2f}")
            
            # Log circuit breaker status
            status = self.circuit_breaker.get_status()
            if status['trading_halted']:
                logger.warning(f"Trading halted after trade: {status['halt_reason']}")

    def update_equity(self, current_equity: float):
        """Update current equity for drawdown calculations"""
        if self.circuit_breaker:
            self.circuit_breaker.update_equity(current_equity)

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        status = {
            'max_position_pct': self.max_position_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'circuit_breaker_enabled': self.circuit_breaker is not None
        }
        
        if self.circuit_breaker:
            status['circuit_breaker_status'] = self.circuit_breaker.get_status()
        
        return status


def create_risk_engine(**kwargs) -> RiskEngine:
    """Factory function to create risk engine"""
    return RiskEngine(**kwargs)