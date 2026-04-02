"""
Transaction Cost Model for Realistic Trading Simulation

This module provides comprehensive transaction cost modeling including:
- Commission fees
- Slippage estimation
- Market impact
- SEC/regulatory fees

Real-world trading costs can reduce returns by 20-50%.
This model ensures realistic performance expectations.

Author: FinAgent Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("TransactionCostModel")


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs"""
    commission: float
    slippage: float
    market_impact: float
    regulatory_fees: float
    total_cost: float
    
    def to_dict(self):
        return {
            'commission': self.commission,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'regulatory_fees': self.regulatory_fees,
            'total_cost': self.total_cost
        }


class TransactionCostModel:
    """
    Comprehensive transaction cost calculator
    
    Models all major cost components of trading:
    1. Commission: Broker fees per trade
    2. Slippage: Difference between expected and actual fill price
    3. Market Impact: Price movement caused by the trade itself
    4. Regulatory Fees: SEC, exchange, and other mandatory fees
    """
    
    def __init__(
        self,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_bps: float = 1.0,  # Basis points (0.01%)
        market_impact_bps: float = 0.5,
        sec_fee_rate: float = 0.0000229,  # Per dollar on sales
        exchange_fee_rate: float = 0.00003  # Per share on sales
    ):
        """
        Initialize cost model with realistic parameters
        
        Args:
            commission_per_share: Commission per share (e.g., $0.005)
            min_commission: Minimum commission per order ($1.00 typical)
            slippage_bps: Expected slippage in basis points (1 bp = 0.01%)
            market_impact_bps: Market impact in basis points
            sec_fee_rate: SEC fee rate (0.0000229 = $22.90 per $1M)
            exchange_fee_rate: Exchange fee rate
        """
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps / 10000  # Convert bps to decimal
        self.market_impact_bps = market_impact_bps / 10000
        self.sec_fee_rate = sec_fee_rate
        self.exchange_fee_rate = exchange_fee_rate
        
        logger.info(f"✅ Transaction cost model initialized")
        logger.debug(f"   Commission: ${commission_per_share:.4f}/share (min: ${min_commission:.2f})")
        logger.debug(f"   Slippage: {slippage_bps} bps")
        logger.debug(f"   Market Impact: {market_impact_bps} bps")
    
    def estimate_costs(
        self,
        price: float,
        quantity: int,
        direction: int = 1,
        spread: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> CostBreakdown:
        """
        Estimate total transaction costs for a trade
        
        Args:
            price: Trade execution price
            quantity: Number of shares (positive for long, negative for short)
            direction: 1 for buy, -1 for sell
            spread: Bid-ask spread (if known, otherwise estimated)
            volatility: Asset volatility (for dynamic slippage)
            
        Returns:
            CostBreakdown: Detailed cost breakdown
        """
        abs_quantity = abs(quantity)
        notional_value = price * abs_quantity
        
        # 1. Commission
        commission = max(
            self.commission_per_share * abs_quantity,
            self.min_commission
        )
        
        # 2. Slippage (dynamic based on volatility if provided)
        if volatility and spread is None:
            # Estimate spread from volatility (typical relationship)
            spread = volatility * 2
        
        if spread is None:
            # Default: use fixed slippage
            slippage = notional_value * self.slippage_bps
        else:
            # Slippage = half spread + market impact
            slippage = notional_value * (spread / 2 + self.market_impact_bps)
        
        # 3. Market Impact (larger trades have more impact)
        # Simple linear model: impact increases with trade size
        base_impact = notional_value * self.market_impact_bps
        size_adjustment = min(abs_quantity / 10000, 2.0)  # Cap at 2x for very large orders
        market_impact = base_impact * (1 + size_adjustment)
        
        # 4. Regulatory Fees (only on SELL orders typically)
        regulatory_fees = 0.0
        if direction == -1:  # Sell order
            sec_fee = notional_value * self.sec_fee_rate
            exchange_fee = abs_quantity * self.exchange_fee_rate
            regulatory_fees = sec_fee + exchange_fee
        
        # Total costs
        total_cost = commission + slippage + market_impact + regulatory_fees
        
        return CostBreakdown(
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            regulatory_fees=regulatory_fees,
            total_cost=total_cost
        )
    
    def calculate_cost_drag(
        self,
        gross_return: float,
        price: float,
        quantity: int,
        direction: int = 1,
        holding_period_days: int = 5
    ) -> float:
        """
        Calculate how much transaction costs reduce returns
        
        Args:
            gross_return: Gross P&L (before costs)
            price: Entry price
            quantity: Trade size
            direction: Trade direction
            holding_period_days: Holding period
            
        Returns:
            net_return: Return after transaction costs
        """
        costs = self.estimate_costs(price, quantity, direction)
        
        # Annualize cost impact for comparison
        annualization_factor = 252 / holding_period_days
        cost_drag = costs.total_cost * annualization_factor
        
        net_return = gross_return - cost_drag
        
        return net_return
    
    def get_breakeven_move(
        self,
        price: float,
        quantity: int,
        direction: int = 1
    ) -> float:
        """
        Calculate minimum price move needed to cover transaction costs
        
        Args:
            price: Entry price
            quantity: Trade size
            direction: Trade direction
            
        Returns:
            breakeven_percent: Minimum % move needed to profit
        """
        costs = self.estimate_costs(price, quantity, direction)
        
        # For long positions: need price to rise by (costs / notional)
        # For short positions: need price to fall by (costs / notional)
        notional = price * abs(quantity)
        breakeven_percent = (costs.total_cost / notional) * 100
        
        return breakeven_percent


class AdaptiveCostModel(TransactionCostModel):
    """
    Transaction cost model that adapts to market conditions
    
    Increases costs during:
    - High volatility
    - Low liquidity
    - Market stress
    """
    
    def __init__(self, base_config=None):
        super().__init__(**(base_config or {}))
        
        # Volatility multipliers
        self.vol_multipliers = {
            'low': 1.0,      # Normal vol
            'medium': 1.5,   # Elevated vol
            'high': 2.5,     # High vol
            'extreme': 5.0   # Crisis vol
        }
    
    def estimate_costs(
        self,
        price: float,
        quantity: int,
        direction: int = 1,
        spread: Optional[float] = None,
        volatility: Optional[float] = None,
        vol_regime: str = 'normal'
    ) -> CostBreakdown:
        """Estimate costs with volatility adjustment"""
        
        # Get base costs
        base_costs = super().estimate_costs(price, quantity, direction, spread, volatility)
        
        # Apply volatility multiplier
        vol_mult = self.vol_multipliers.get(vol_regime, 1.0)
        
        # Adjust slippage and market impact for volatility
        adjusted_slippage = base_costs.slippage * vol_mult
        adjusted_impact = base_costs.market_impact * vol_mult
        
        # Recalculate total
        adjusted_total = (
            base_costs.commission +
            adjusted_slippage +
            adjusted_impact +
            base_costs.regulatory_fees
        )
        
        return CostBreakdown(
            commission=base_costs.commission,
            slippage=adjusted_slippage,
            market_impact=adjusted_impact,
            regulatory_fees=base_costs.regulatory_fees,
            total_cost=adjusted_total
        )


def create_cost_model(adaptive: bool = False, **kwargs) -> TransactionCostModel:
    """Factory function to create cost model"""
    if adaptive:
        return AdaptiveCostModel(kwargs)
    else:
        return TransactionCostModel(**kwargs)
