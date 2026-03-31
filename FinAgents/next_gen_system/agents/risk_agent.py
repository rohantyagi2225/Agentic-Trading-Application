"""
RiskAgent - Comprehensive Risk Management Agent

This agent implements advanced risk modeling including VaR, CVaR,
stress testing, regime detection, and compliance monitoring.

Key Features:
------------
- Value at Risk (VaR) calculation
- Conditional VaR (CVaR/Expected Shortfall)
- Stress testing and scenario analysis
- Market regime detection
- Position limits and drawdown controls
- Regulatory compliance checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)


class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    BETA = "beta"


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_LOW_VOL = "bull_low_volatility"
    BULL_HIGH_VOL = "bull_high_volatility"
    BEAR_LOW_VOL = "bear_low_volatility"
    BEAR_HIGH_VOL = "bear_high_volatility"
    CRISIS = "crisis"


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    current_volatility: float
    market_regime: MarketRegime
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    position_limits: Dict[str, float]
    compliance_status: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class RiskAgent:
    """
    Domain-specialized risk management agent
    
    Implements comprehensive risk modeling and compliance monitoring
    for safe and regulated trading operations.
    """
    
    def __init__(
        self,
        agent_id: str,
        var_confidence: float = 0.95,
        max_portfolio_var: float = 0.05,
        max_position_size: float = 0.10,
        max_drawdown: float = 0.20,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize the RiskAgent
        
        Args:
            agent_id: Unique identifier
            var_confidence: Confidence level for VaR calculations
            max_portfolio_var: Maximum allowed portfolio VaR
            max_position_size: Maximum single position as % of portfolio
            max_drawdown: Maximum allowed drawdown before liquidation
            risk_free_rate: Risk-free rate for Sharpe calculations
        """
        self.agent_id = agent_id
        self.var_confidence = var_confidence
        self.max_portfolio_var = max_portfolio_var
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.risk_free_rate = risk_free_rate
        
        # Historical data for risk calculations
        self.return_history: List[float] = []
        self.price_history: Dict[str, List[float]] = {}
        
        # Compliance tracking
        self.compliance_violations: List[Dict[str, Any]] = []
        
        logger.info(f"RiskAgent {agent_id} initialized with VaR confidence {var_confidence*100}%")
    
    def assess_portfolio_risk(
        self,
        portfolio_values: pd.Series,
        positions: Dict[str, Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame],
    ) -> RiskAssessment:
        """
        Perform comprehensive portfolio risk assessment
        
        Args:
            portfolio_values: Time series of portfolio values
            positions: Current portfolio positions
            market_data: Market data for all holdings
            
        Returns:
            Complete risk assessment
        """
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        self.return_history.extend(returns.tolist())
        
        # Calculate risk metrics
        var_95 = self._calculate_var(returns, confidence=self.var_confidence)
        cvar_95 = self._calculate_cvar(returns, confidence=self.var_confidence)
        max_dd = self._calculate_max_drawdown(portfolio_values)
        volatility = self._calculate_volatility(returns)
        
        # Detect market regime
        regime = self._detect_market_regime(market_data)
        
        # Determine overall risk level
        risk_level = self._determine_risk_level(var_95, max_dd, volatility)
        
        # Calculate position limits
        position_limits = self._calculate_position_limits(positions, portfolio_values.iloc[-1])
        
        # Check compliance
        compliance_status, warnings = self._check_compliance(
            positions, portfolio_values.iloc[-1], var_95, max_dd
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level, regime, positions, warnings
        )
        
        return RiskAssessment(
            portfolio_var=var_95,
            portfolio_cvar=cvar_95,
            max_drawdown=max_dd,
            current_volatility=volatility,
            market_regime=regime,
            risk_level=risk_level,
            position_limits=position_limits,
            compliance_status=compliance_status,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def _calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Return series
            confidence: Confidence level
            method: Calculation method (historical, parametric, monte_carlo)
            
        Returns:
            VaR as positive number representing potential loss
        """
        if method == "historical":
            var = -np.percentile(returns, (1 - confidence) * 100)
        elif method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            var = -(mu + sigma * stats.norm.ppf(1 - confidence))
        else:
            # Default to historical
            var = -np.percentile(returns, (1 - confidence) * 100)
        
        return max(var, 0)
    
    def _calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        Args:
            returns: Return series
            confidence: Confidence level
            
        Returns:
            CVaR as expected loss beyond VaR threshold
        """
        var = self._calculate_var(returns, confidence)
        cvar = -returns[returns <= -var].mean()
        return max(cvar, 0) if not np.isnan(cvar) else var
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown from peak"""
        rolling_max = values.expanding(min_periods=1).max()
        drawdowns = (values - rolling_max) / rolling_max
        return abs(drawdowns.min())
    
    def _calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate annualized volatility"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Trading days
        return vol
    
    def _detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """Detect current market regime based on price action and volatility"""
        if not market_data:
            return MarketRegime.BULL_LOW_VOL
        
        # Aggregate market signals
        avg_return = 0
        avg_vol = 0
        count = 0
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                continue
            
            returns = data["close"].pct_change().dropna()
            avg_return += returns.mean()
            avg_vol += returns.std()
            count += 1
        
        if count == 0:
            return MarketRegime.BULL_LOW_VOL
        
        avg_return /= count
        avg_vol /= count
        
        # Classify regime
        if avg_return > 0.001 and avg_vol < 0.02:
            return MarketRegime.BULL_LOW_VOL
        elif avg_return > 0.001 and avg_vol >= 0.02:
            return MarketRegime.BULL_HIGH_VOL
        elif avg_return <= 0.001 and avg_vol < 0.02:
            return MarketRegime.BEAR_LOW_VOL
        elif avg_return <= 0.001 and avg_vol >= 0.02:
            return MarketRegime.BEAR_HIGH_VOL
        else:
            return MarketRegime.CRISIS
    
    def _determine_risk_level(
        self,
        var: float,
        max_dd: float,
        volatility: float,
    ) -> str:
        """Determine overall risk level"""
        risk_score = 0
        
        # VaR contribution
        if var > self.max_portfolio_var * 1.5:
            risk_score += 3
        elif var > self.max_portfolio_var:
            risk_score += 2
        elif var > self.max_portfolio_var * 0.7:
            risk_score += 1
        
        # Drawdown contribution
        if max_dd > self.max_drawdown * 0.8:
            risk_score += 3
        elif max_dd > self.max_drawdown * 0.5:
            risk_score += 2
        elif max_dd > self.max_drawdown * 0.3:
            risk_score += 1
        
        # Volatility contribution
        if volatility > 0.5:
            risk_score += 3
        elif volatility > 0.3:
            risk_score += 2
        elif volatility > 0.2:
            risk_score += 1
        
        # Map to risk level
        if risk_score >= 6:
            return "CRITICAL"
        elif risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_position_limits(
        self,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """Calculate maximum position sizes for each holding"""
        limits = {}
        
        for symbol, pos in positions.items():
            current_value = pos.get("size", 0) * pos.get("entry_price", 0)
            current_pct = current_value / portfolio_value if portfolio_value > 0 else 0
            
            # Dynamic limit based on volatility and correlation
            base_limit = self.max_position_size
            adjusted_limit = base_limit * (1 - current_pct)  # Reduce as position grows
            
            limits[symbol] = adjusted_limit
        
        return limits
    
    def _check_compliance(
        self,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        var: float,
        max_dd: float,
    ) -> Tuple[bool, List[str]]:
        """Check regulatory and internal compliance"""
        violations = []
        compliant = True
        
        # Check position concentration
        for symbol, pos in positions.items():
            current_value = pos.get("size", 0) * pos.get("entry_price", 0)
            position_pct = current_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > self.max_position_size * 1.2:  # 20% buffer
                violations.append(f"Position concentration: {symbol} at {position_pct*100:.1f}% (limit: {self.max_position_size*100}%)")
                compliant = False
        
        # Check VaR limit
        if var > self.max_portfolio_var * 1.1:
            violations.append(f"Portfolio VaR breach: {var*100:.2f}% (limit: {self.max_portfolio_var*100}%)")
            compliant = False
        
        # Check drawdown limit
        if max_dd > self.max_drawdown * 0.9:
            violations.append(f"Approaching max drawdown: {max_dd*100:.2f}% (limit: {self.max_drawdown*100}%)")
            compliant = False
        
        # Record violations
        if violations:
            self.compliance_violations.append({
                "timestamp": datetime.now(),
                "violations": violations,
                "severity": "WARNING" if compliant else "BREACH",
            })
        
        return compliant, violations
    
    def _generate_recommendations(
        self,
        risk_level: str,
        regime: MarketRegime,
        positions: Dict[str, Dict[str, Any]],
        warnings: List[str],
    ) -> List[str]:
        """Generate actionable risk recommendations"""
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.append("URGENT: Consider immediate portfolio de-risking")
            recommendations.append("Reduce position sizes by at least 50%")
        elif risk_level == "HIGH":
            recommendations.append("Consider reducing exposure to high-volatility positions")
            recommendations.append("Implement hedging strategies")
        
        if regime == MarketRegime.CRISIS:
            recommendations.append("Market in crisis mode - prioritize capital preservation")
            recommendations.append("Increase cash position and reduce leverage")
        elif regime == MarketRegime.BEAR_HIGH_VOL:
            recommendations.append("Bear market with high volatility - defensive positioning recommended")
        
        if any("concentration" in w.lower() for w in warnings):
            recommendations.append("Diversify portfolio to reduce concentration risk")
        
        if any("var" in w.lower() for w in warnings):
            recommendations.append("Reduce portfolio risk to bring VaR within limits")
        
        return recommendations
    
    def stress_test_portfolio(
        self,
        portfolio_values: pd.Series,
        scenarios: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio
        
        Args:
            portfolio_values: Portfolio value history
            scenarios: Custom stress scenarios
            
        Returns:
            Stress test results
        """
        default_scenarios = [
            {"name": "Market Crash (-20%)", "shock": -0.20},
            {"name": "Moderate Decline (-10%)", "shock": -0.10},
            {"name": "Volatility Spike (+50%)", "shock": 0.0, "vol_multiplier": 1.5},
            {"name": "Flash Crash (-5%)", "shock": -0.05},
        ]
        
        scenarios = scenarios or default_scenarios
        results = {}
        
        for scenario in scenarios:
            shock = scenario.get("shock", 0)
            vol_mult = scenario.get("vol_multiplier", 1.0)
            
            # Apply shock
            stressed_values = portfolio_values * (1 + shock)
            stressed_returns = stressed_values.pct_change().dropna()
            
            # Calculate impact
            impacted_var = self._calculate_var(stressed_returns) * vol_mult
            impacted_dd = self._calculate_max_drawdown(stressed_values)
            
            results[scenario["name"]] = {
                "portfolio_impact": shock * 100,
                "stressed_var": impacted_var,
                "stressed_drawdown": impacted_dd,
                "survival_probability": 1 - impacted_dd,
            }
        
        return {
            "scenarios_tested": len(scenarios),
            "results": results,
            "worst_case": max(results.items(), key=lambda x: x[1]["stressed_drawdown"]),
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "return_history_count": len(self.return_history),
            "compliance_violations_count": len(self.compliance_violations),
            "risk_parameters": {
                "var_confidence": self.var_confidence,
                "max_portfolio_var": self.max_portfolio_var,
                "max_position_size": self.max_position_size,
                "max_drawdown": self.max_drawdown,
            },
        }
