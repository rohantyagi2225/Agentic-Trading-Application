"""
Trading Circuit Breakers - Emergency Risk Controls

This module implements mandatory risk controls that AUTOMATICALLY halt trading
when dangerous conditions are detected. Unlike passive risk checks, these
actively prevent further trading to protect capital.

Key Features:
- Daily loss limits
- Maximum drawdown protection
- Consecutive loss limits
- Volatility-based position reduction
- Emergency stop mechanism

Author: FinAgent Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from enum import Enum
import logging

logger = logging.getLogger("CircuitBreaker")


class CircuitBreakerLevel(Enum):
    """Severity levels for circuit breaker triggers"""
    WARNING = "warning"  # Alert sent, trading continues
    REDUCE = "reduce"    # Reduce position sizes by 50%
    HALT = "halt"        # Stop all new trading
    LIQUIDATE = "liquidate"  # Close all positions immediately


@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker trigger"""
    event_id: str
    trigger_type: str
    trigger_value: float
    threshold: float
    level: CircuitBreakerLevel
    timestamp: datetime
    message: str
    
    def to_dict(self):
        return {
            'event_id': self.event_id,
            'trigger_type': self.trigger_type,
            'trigger_value': self.trigger_value,
            'threshold': self.threshold,
            'level': self.level.value,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message
        }


class TradingCircuitBreaker:
    """
    Main circuit breaker system with multiple safety mechanisms
    
    Monitors in real-time and halts trading when limits breached
    """
    
    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,      # 5% daily loss limit
        max_drawdown_pct: float = 0.15,         # 15% max drawdown
        max_consecutive_losses: int = 5,        # Stop after 5 straight losses
        max_weekly_loss_pct: float = 0.10,      # 10% weekly loss limit
        volatility_halt_threshold: float = 0.50, # Halt if daily vol > 50%
        enable_auto_liquidation: bool = False   # Auto-close on extreme events
    ):
        """
        Initialize circuit breakers
        
        Args:
            max_daily_loss_pct: Maximum allowed daily loss (default 5%)
            max_drawdown_pct: Maximum peak-to-trough drawdown (default 15%)
            max_consecutive_losses: Max consecutive losing trades
            max_weekly_loss_pct: Maximum weekly loss
            volatility_halt_threshold: Volatility level triggering halt
            enable_auto_liquidation: Auto-liquidate on extreme breaches
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.volatility_halt_threshold = volatility_halt_threshold
        self.enable_auto_liquidation = enable_auto_liquidation
        
        # State tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_equity = 100000.0  # Starting equity
        self.current_equity = 100000.0
        self.consecutive_losses = 0
        self.is_trading_halted = False
        self.halt_reason: Optional[str] = None
        self.halt_until: Optional[datetime] = None
        
        # Event log
        self.events: List[CircuitBreakerEvent] = []
        
        logger.info(f"✅ Trading circuit breakers initialized")
        logger.info(f"   Daily Loss Limit: {max_daily_loss_pct:.1%}")
        logger.info(f"   Max Drawdown: {max_drawdown_pct:.1%}")
        logger.info(f"   Consecutive Loss Limit: {max_consecutive_losses}")
    
    def update_equity(self, current_equity: float):
        """Update current equity for drawdown calculations"""
        self.current_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)
    
    def record_trade(self, pnl: float):
        """Record a completed trade P&L"""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check all circuit breakers
        self._check_all_breakers()
    
    def _check_all_breakers(self):
        """Check all circuit breaker conditions"""
        
        # 1. Daily Loss Check
        daily_return = self.daily_pnl / self.peak_equity
        if daily_return < -self.max_daily_loss_pct:
            self._trigger_circuit_breaker(
                trigger_type='daily_loss',
                trigger_value=daily_return,
                threshold=-self.max_daily_loss_pct,
                level=CircuitBreakerLevel.HALT,
                message=f"Daily loss limit breached: {daily_return:.2%}"
            )
        
        # 2. Maximum Drawdown Check
        drawdown = (self.current_equity - self.peak_equity) / self.peak_equity
        if drawdown < -self.max_drawdown_pct:
            self._trigger_circuit_breaker(
                trigger_type='max_drawdown',
                trigger_value=drawdown,
                threshold=-self.max_drawdown_pct,
                level=CircuitBreakerLevel.LIQUIDATE if self.enable_auto_liquidation else CircuitBreakerLevel.HALT,
                message=f"Maximum drawdown breached: {drawdown:.2%}"
            )
        
        # 3. Consecutive Losses Check
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._trigger_circuit_breaker(
                trigger_type='consecutive_losses',
                trigger_value=self.consecutive_losses,
                threshold=self.max_consecutive_losses,
                level=CircuitBreakerLevel.HALT,
                message=f"Consecutive loss limit reached: {self.consecutive_losses}"
            )
        
        # 4. Weekly Loss Check
        weekly_return = self.weekly_pnl / self.peak_equity
        if weekly_return < -self.max_weekly_loss_pct:
            self._trigger_circuit_breaker(
                trigger_type='weekly_loss',
                trigger_value=weekly_return,
                threshold=-self.max_weekly_loss_pct,
                level=CircuitBreakerLevel.HALT,
                message=f"Weekly loss limit breached: {weekly_return:.2%}"
            )
    
    def check_volatility_halt(self, current_volatility: float):
        """Check if volatility warrants halting trading"""
        if current_volatility > self.volatility_halt_threshold:
            self._trigger_circuit_breaker(
                trigger_type='extreme_volatility',
                trigger_value=current_volatility,
                threshold=self.volatility_halt_threshold,
                level=CircuitBreakerLevel.HALT,
                message=f"Extreme volatility detected: {current_volatility:.1%}"
            )
    
    def _trigger_circuit_breaker(
        self,
        trigger_type: str,
        trigger_value: float,
        threshold: float,
        level: CircuitBreakerLevel,
        message: str
    ):
        """Trigger a circuit breaker"""
        
        event = CircuitBreakerEvent(
            event_id=f"cb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            threshold=threshold,
            level=level,
            timestamp=datetime.now(),
            message=message
        )
        
        self.events.append(event)
        
        logger.error(f"🚨 CIRCUIT BREAKER TRIGGERED: {message}")
        logger.error(f"   Level: {level.value}")
        
        # Take action based on level
        if level == CircuitBreakerLevel.WARNING:
            logger.warning("⚠️ Warning issued - trading continues with caution")
        
        elif level == CircuitBreakerLevel.REDUCE:
            logger.warning("⚠️ Position sizes reduced by 50%")
        
        elif level == CircuitBreakerLevel.HALT:
            self.is_trading_halted = True
            self.halt_reason = message
            
            # Halt until next trading day
            self.halt_until = datetime.now() + timedelta(days=1)
            
            logger.critical("🛑 TRADING HALTED - No new positions allowed")
        
        elif level == CircuitBreakerLevel.LIQUIDATE:
            self.is_trading_halted = True
            self.halt_reason = message
            
            logger.critical("🚨 EMERGENCY LIQUIDATION ORDERED")
            logger.critical("🚫 ALL POSITIONS MUST BE CLOSED IMMEDIATELY")
    
    def should_allow_trade(self) -> tuple[bool, Optional[str]]:
        """
        Check if trading is currently allowed
        
        Returns:
            (allowed, reason): Tuple of whether trade is allowed and why/why not
        """
        # Check if halted
        if self.is_trading_halted:
            # Check if halt period has expired
            if self.halt_until and datetime.now() > self.halt_until:
                logger.info("✅ Trading halt period expired - resuming trading")
                self.is_trading_halted = False
                self.halt_reason = None
                self.halt_until = None
                return True, None
            else:
                return False, f"Trading halted: {self.halt_reason}"
        
        # All clear
        return True, None
    
    def get_position_size_multiplier(self) -> float:
        """
        Get position size adjustment based on circuit breaker state
        
        Returns:
            multiplier: 1.0 for normal, 0.5 for reduce mode, 0.0 for halt
        """
        if self.is_trading_halted:
            return 0.0
        
        # Check recent events for REDUCE level triggers
        recent_events = [
            e for e in self.events
            if e.level == CircuitBreakerLevel.REDUCE and
            (datetime.now() - e.timestamp).total_seconds() < 86400  # Last 24h
        ]
        
        if recent_events:
            return 0.5
        
        return 1.0
    
    def reset_daily_metrics(self):
        """Reset daily P&L at start of each trading day"""
        logger.info("📊 Resetting daily metrics")
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        
        # Move daily P&L to weekly
        # (In production, would track week properly)
    
    def get_status(self) -> Dict:
        """Get current circuit breaker status"""
        return {
            'trading_halted': self.is_trading_halted,
            'halt_reason': self.halt_reason,
            'halt_until': self.halt_until.isoformat() if self.halt_until else None,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown': (self.current_equity - self.peak_equity) / self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'consecutive_losses': self.consecutive_losses,
            'position_size_multiplier': self.get_position_size_multiplier(),
            'recent_events': [e.to_dict() for e in self.events[-5:]]
        }


class VolatilityBasedPositionSizer:
    """
    Adjusts position sizes dynamically based on market volatility
    
    Reduces risk when volatility is high, increases when low
    """
    
    def __init__(
        self,
        base_position_size: float = 0.02,  # 2% of capital per trade
        vol_target: float = 0.15,          # Target 15% annualized vol
        max_position_size: float = 0.05,   # Max 5% per trade
        min_position_size: float = 0.005   # Min 0.5% per trade
    ):
        self.base_position_size = base_position_size
        self.vol_target = vol_target
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
    
    def calculate_position_size(
        self,
        current_volatility: float,
        circuit_breaker_multiplier: float = 1.0
    ) -> float:
        """
        Calculate risk-adjusted position size
        
        Args:
            current_volatility: Current market volatility
            circuit_breaker_multiplier: From circuit breaker (0.0-1.0)
            
        Returns:
            position_size: % of capital to risk
        """
        # Volatility scaling
        vol_adjustment = self.vol_target / max(current_volatility, 0.01)
        
        # Base size adjusted for volatility
        raw_size = self.base_position_size * vol_adjustment
        
        # Apply bounds
        bounded_size = max(self.min_position_size, min(raw_size, self.max_position_size))
        
        # Apply circuit breaker adjustment
        final_size = bounded_size * circuit_breaker_multiplier
        
        logger.debug(f"Position size calc: vol={current_volatility:.1%}, "
                    f"raw={raw_size:.2%}, final={final_size:.2%}")
        
        return final_size


def create_circuit_breaker(**kwargs) -> TradingCircuitBreaker:
    """Factory function to create circuit breaker"""
    return TradingCircuitBreaker(**kwargs)
