"""
Regime Detection System for Financial Trading Agents

This module provides comprehensive market regime detection capabilities to identify
different market states and adapt trading strategies accordingly.

Key Features:
- Multi-timeframe regime identification
- Statistical regime detection (volatility, trend, momentum)
- Market state classification (bull/bear/crash/expansion/stagnation)
- Regime confidence scoring
- Strategy adaptation recommendations

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │              Regime Detection System                    │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ Volatility   │  │ Trend        │  │ Momentum     │  │
    │  │ Detector     │  │ Detector     │  │ Detector     │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │           Regime Classifier & Scorer             │  │
    │  └──────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────┐
    │           Strategy Adaptation Layer                     │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ Bull Market  │  │ Bear Market  │  │ High Vol     │  │
    │  │ Strategies   │  │ Strategies   │  │ Strategies   │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    └─────────────────────────────────────────────────────────┘

Author: FinAgent Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("RegimeDetector")


class MarketRegime(Enum):
    """Enumeration of market regimes"""
    BULL_MARKET = "bull_market"  # Strong upward trend, low volatility
    BEAR_MARKET = "bear_market"  # Strong downward trend, high volatility
    HIGH_VOLATILITY = "high_volatility"  # Elevated uncertainty, large swings
    LOW_VOLATILITY = "low_volatility"  # Calm markets, small moves
    TRENDING_UP = "trending_up"  # Clear upward momentum
    TRENDING_DOWN = "trending_down"  # Clear downward momentum
    SIDEWAYS = "sideways"  # Range-bound, no clear direction
    CRASH = "crash"  # Extreme downward move with panic
    EXPANSION = "expansion"  # Economic growth phase
    CONTRACTION = "contraction"  # Economic decline phase
    RECOVERY = "recovery"  # Post-crisis recovery
    UNKNOWN = "unknown"  # Insufficient data or unclear regime


@dataclass
class RegimeResult:
    """Result of regime detection analysis"""
    primary_regime: MarketRegime
    secondary_regime: Optional[MarketRegime]
    confidence: float
    regime_probabilities: Dict[str, float]
    detected_at: datetime
    lookback_period: int
    indicators: Dict[str, Any]
    strategy_recommendations: List[str]
    risk_level: str  # low, medium, high, extreme
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "primary_regime": self.primary_regime.value,
            "secondary_regime": self.secondary_regime.value if self.secondary_regime else None,
            "confidence": self.confidence,
            "regime_probabilities": self.regime_probabilities,
            "detected_at": self.detected_at.isoformat(),
            "lookback_period": self.lookback_period,
            "indicators": self.indicators,
            "strategy_recommendations": self.strategy_recommendations,
            "risk_level": self.risk_level
        }


class RegimeDetector:
    """
    Advanced market regime detection system using multiple statistical methods.
    
    This detector combines:
    1. Volatility analysis (GARCH-like behavior, realized volatility)
    2. Trend detection (moving averages, ADX, linear regression)
    3. Momentum indicators (RSI, MACD, rate of change)
    4. Statistical methods (HMM, GMM clustering)
    5. Market microstructure (volume patterns, liquidity)
    
    The system outputs a probabilistic regime classification with confidence scores
    and strategy adaptation recommendations.
    """
    
    def __init__(
        self,
        volatility_lookback: int = 20,
        trend_lookback: int = 50,
        momentum_lookback: int = 14,
        long_term_lookback: int = 252,
        use_gmm: bool = True,
        gmm_components: int = 3
    ):
        """
        Initialize regime detector
        
        Args:
            volatility_lookback: Period for volatility calculations
            trend_lookback: Period for trend detection
            momentum_lookback: Period for momentum indicators
            long_term_lookback: Long-term analysis period (1 year default)
            use_gmm: Use Gaussian Mixture Model for clustering
            gmm_components: Number of GMM components
        """
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.momentum_lookback = momentum_lookback
        self.long_term_lookback = long_term_lookback
        self.use_gmm = use_gmm
        self.gmm_components = gmm_components
        
        # Predefined regime thresholds
        self.volatility_thresholds = {
            'low': 0.10,      # Annualized vol < 10%
            'medium': 0.20,   # 10-20%
            'high': 0.30,     # 20-30%
            'extreme': 0.50   # > 30%
        }
        
        self.trend_thresholds = {
            'strong_bull': 0.20,   # > 20% annualized
            'weak_bull': 0.05,     # 5-20%
            'weak_bear': -0.05,    # -5 to 5%
            'strong_bear': -0.20   # < -20%
        }
        
        logger.info("✅ Regime detector initialized")
    
    def detect_regime(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        returns: Optional[pd.Series] = None
    ) -> RegimeResult:
        """
        Detect current market regime from price/volume data
        
        Args:
            prices: Price series (close prices)
            volumes: Volume series (optional but recommended)
            returns: Pre-calculated returns (if None, calculated from prices)
            
        Returns:
            RegimeResult: Comprehensive regime detection result
        """
        # Validate input
        if len(prices) < self.volatility_lookback:
            raise ValueError(f"Insufficient data: need at least {self.volatility_lookback} periods")
        
        # Calculate returns if not provided
        if returns is None:
            returns = prices.pct_change().dropna()
        
        # ===== FAST CRASH DETECTION (Short lookback) =====
        crash_detected = self._detect_flash_crash(prices, returns)
        if crash_detected:
            # Immediate crash detection overrides other signals
            detected_at = datetime.now()
            return RegimeResult(
                primary_regime=MarketRegime.CRASH,
                secondary_regime=MarketRegime.HIGH_VOLATILITY,
                confidence=0.95,  # High confidence in crash detection
                regime_probabilities={'crash': 0.95, 'high_volatility': 0.05},
                detected_at=detected_at,
                lookback_period=5,  # Only 5 days needed
                indicators={'flash_crash': True, 'urgent': True},
                strategy_recommendations=[
                    "EXIT ALL RISK POSITIONS IMMEDIATELY",
                    "Move to cash and safe-haven assets",
                    "Do NOT attempt to catch falling knives"
                ],
                risk_level='extreme'
            )
        
        # Detect current time
        detected_at = datetime.now()
        
        # ===== STANDARD REGIME DETECTION (Full analysis) =====
        # 1. VOLATILITY ANALYSIS
        vol_indicators = self._analyze_volatility(prices, returns)
        
        # 2. TREND ANALYSIS
        trend_indicators = self._analyze_trend(prices)
        
        # 3. MOMENTUM ANALYSIS
        momentum_indicators = self._analyze_momentum(prices, returns)
        
        # 4. VOLUME ANALYSIS (if available)
        volume_indicators = {}
        if volumes is not None:
            volume_indicators = self._analyze_volume(prices, volumes)
        
        # 5. STATISTICAL REGIME DETECTION (GMM)
        gmm_result = {}
        if self.use_gmm and len(returns) >= self.long_term_lookback:
            gmm_result = self._detect_regime_gmm(returns)
        
        # 6. COMBINE ALL SIGNALS
        combined_indicators = {
            **vol_indicators,
            **trend_indicators,
            **momentum_indicators,
            **volume_indicators,
            **gmm_result
        }
        
        # 7. CLASSIFY REGIME
        regime_classification = self._classify_regime(combined_indicators)
        
        # 8. CALCULATE CONFIDENCE
        confidence = self._calculate_confidence(combined_indicators, regime_classification)
        
        # 9. GENERATE STRATEGY RECOMMENDATIONS
        recommendations = self._generate_recommendations(regime_classification, combined_indicators)
        
        # 10. DETERMINE RISK LEVEL
        risk_level = self._assess_risk_level(combined_indicators)
        
        # Create result
        result = RegimeResult(
            primary_regime=regime_classification,
            secondary_regime=self._get_secondary_regime(regime_classification, combined_indicators),
            confidence=confidence,
            regime_probabilities=self._calculate_regime_probabilities(combined_indicators),
            detected_at=detected_at,
            lookback_period=self.long_term_lookback,
            indicators=combined_indicators,
            strategy_recommendations=recommendations,
            risk_level=risk_level
        )
        
        logger.info(f"🎯 Detected regime: {regime_classification.value} (confidence: {confidence:.2%})")
        
        return result
    
    def _detect_flash_crash(self, prices: pd.Series, returns: pd.Series) -> bool:
        """
        Detect flash crash conditions using SHORT lookback (5 days)
        
        This is a fast emergency detection system that overrides normal regime detection.
        
        Returns:
            bool: True if flash crash detected
        """
        if len(returns) < 5:
            return False
        
        # Check last 5 days for extreme moves
        recent_returns = returns.iloc[-5:]
        
        # Criterion 1: >10% drop in 5 days
        five_day_return = (prices.iloc[-1] / prices.iloc[-5]) - 1
        if five_day_return < -0.10:
            logger.warning(f"⚠️ FLASH CRASH WARNING: {five_day_return:.1%} drop in 5 days")
            return True
        
        # Criterion 2: Multiple down days (>3 down days in last 5 with avg <-2%)
        down_days = sum(1 for r in recent_returns if r < -0.02)
        avg_return = recent_returns.mean()
        if down_days >= 3 and avg_return < -0.02:
            logger.warning(f"⚠️ FLASH CRASH WARNING: {down_days} down days, avg return {avg_return:.1%}")
            return True
        
        # Criterion 3: Single day extreme move (<-7% in one day)
        if any(r < -0.07 for r in recent_returns):
            worst_day = min(recent_returns)
            logger.warning(f"⚠️ FLASH CRASH WARNING: Single day drop of {worst_day:.1%}")
            return True
        
        return False
    
    def _analyze_volatility(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Analyze volatility characteristics"""
        result = {}
        
        # Realized volatility (rolling)
        realized_vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
        result['realized_volatility'] = realized_vol.iloc[-1]
        result['volatility_rank'] = (realized_vol.iloc[-1] - realized_vol.min()) / (realized_vol.max() - realized_vol.min() + 1e-10)
        
        # Volatility trend (is vol increasing or decreasing?)
        vol_ma_short = realized_vol.rolling(10).mean()
        vol_ma_long = realized_vol.rolling(30).mean()
        result['volatility_trend'] = 1 if vol_ma_short.iloc[-1] > vol_ma_long.iloc[-1] else -1
        
        # Historical volatility percentile
        hv_percentile = (realized_vol.iloc[-1] > realized_vol).mean()
        result['volatility_percentile'] = hv_percentile
        
        # Volatility regime (GARCH-like behavior)
        squared_returns = returns ** 2
        result['vol_clustering'] = squared_returns.rolling(self.volatility_lookback).mean().iloc[-1]
        
        # Classify volatility level
        if result['realized_volatility'] < self.volatility_thresholds['low']:
            result['vol_regime'] = 'low_volatility'
        elif result['realized_volatility'] < self.volatility_thresholds['medium']:
            result['vol_regime'] = 'normal_volatility'
        elif result['realized_volatility'] < self.volatility_thresholds['high']:
            result['vol_regime'] = 'high_volatility'
        else:
            result['vol_regime'] = 'extreme_volatility'
        
        return result
    
    def _analyze_trend(self, prices: pd.Series) -> Dict[str, Any]:
        """Analyze trend characteristics"""
        result = {}
        
        # Moving averages
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        ma_200 = prices.rolling(200).mean()
        
        current_price = prices.iloc[-1]
        
        # Price relative to MAs
        result['price_vs_ma20'] = (current_price - ma_20.iloc[-1]) / ma_20.iloc[-1]
        result['price_vs_ma50'] = (current_price - ma_50.iloc[-1]) / ma_50.iloc[-1]
        result['price_vs_ma200'] = (current_price - ma_200.iloc[-1]) / ma_200.iloc[-1]
        
        # MA alignment (golden cross vs death cross)
        ma_alignment = 0
        if ma_20.iloc[-1] > ma_50.iloc[-1] > ma_200.iloc[-1]:
            ma_alignment = 1  # Bullish alignment
        elif ma_20.iloc[-1] < ma_50.iloc[-1] < ma_200.iloc[-1]:
            ma_alignment = -1  # Bearish alignment
        result['ma_alignment'] = ma_alignment
        
        # Trend strength (ADX-like metric)
        tr1 = prices.diff().abs()
        tr2 = (prices - prices.shift(1)).abs()
        true_range = pd.concat([tr1, tr2], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        
        dm_plus = (prices - prices.shift(1)).clip(lower=0)
        dm_minus = (prices.shift(1) - prices).clip(lower=0)
        
        plus_di = 100 * (dm_plus.rolling(14).mean() / atr)
        minus_di = 100 * (dm_minus.rolling(14).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        result['adx'] = dx.rolling(14).mean().iloc[-1]
        result['trend_strength'] = result['adx'] / 100.0
        
        # Linear regression trend
        x = np.arange(len(prices.iloc[-self.trend_lookback:]))
        y = prices.iloc[-self.trend_lookback:].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        result['trend_slope'] = slope
        result['trend_r_squared'] = r_value ** 2
        result['trend_significance'] = 1 if p_value < 0.05 else 0
        
        # Annualized trend
        result['annualized_trend'] = slope * 252 / prices.iloc[-self.trend_lookback]
        
        # Classify trend
        if result['annualized_trend'] > self.trend_thresholds['strong_bull']:
            result['trend_regime'] = 'strong_uptrend'
        elif result['annualized_trend'] > self.trend_thresholds['weak_bull']:
            result['trend_regime'] = 'weak_uptrend'
        elif result['annualized_trend'] > self.trend_thresholds['weak_bear']:
            result['trend_regime'] = 'sideways'
        elif result['annualized_trend'] > self.trend_thresholds['strong_bear']:
            result['trend_regime'] = 'weak_downtrend'
        else:
            result['trend_regime'] = 'strong_downtrend'
        
        return result
    
    def _analyze_momentum(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Analyze momentum characteristics"""
        result = {}
        
        # RSI (Relative Strength Index)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.momentum_lookback).mean()
        loss = (-delta.clip(upper=0)).rolling(self.momentum_lookback).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        result['rsi'] = rsi.iloc[-1]
        
        # RSI regime
        if rsi.iloc[-1] > 70:
            result['rsi_regime'] = 'overbought'
        elif rsi.iloc[-1] < 30:
            result['rsi_regime'] = 'oversold'
        else:
            result['rsi_regime'] = 'neutral'
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - signal_line
        
        result['macd'] = macd_line.iloc[-1]
        result['macd_signal'] = signal_line.iloc[-1]
        result['macd_histogram'] = macd_histogram.iloc[-1]
        result['macd_crossover'] = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
        
        # Rate of Change (ROC)
        roc_1m = prices.pct_change(21).iloc[-1]  # 1 month
        roc_3m = prices.pct_change(63).iloc[-1]  # 3 months
        roc_6m = prices.pct_change(126).iloc[-1]  # 6 months
        roc_12m = prices.pct_change(252).iloc[-1]  # 1 year
        
        result['roc_1m'] = roc_1m
        result['roc_3m'] = roc_3m
        result['roc_6m'] = roc_6m
        result['roc_12m'] = roc_12m
        
        # Momentum score (composite)
        momentum_score = (
            0.3 * (rsi.iloc[-1] - 50) / 50 +
            0.3 * np.sign(macd_histogram.iloc[-1]) +
            0.2 * np.sign(roc_1m) +
            0.2 * np.sign(roc_3m)
        )
        result['momentum_score'] = momentum_score
        
        return result
    
    def _analyze_volume(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, Any]:
        """Analyze volume patterns"""
        result = {}
        
        # Volume trend
        vol_ma_20 = volumes.rolling(20).mean()
        result['volume_vs_avg'] = volumes.iloc[-1] / vol_ma_20.iloc[-1]
        
        # On-Balance Volume (OBV)
        obv = (np.sign(prices.diff()) * volumes).cumsum()
        result['obv_trend'] = obv.diff(20).iloc[-1]
        
        # Volume-price trend
        vpt = (prices.pct_change() * volumes).cumsum()
        result['vpt_trend'] = vpt.diff(20).iloc[-1]
        
        # Volume volatility
        result['volume_volatility'] = volumes.pct_change().rolling(20).std().iloc[-1]
        
        # Accumulation/Distribution
        clv = ((prices - prices.shift(1).min()) - (prices.shift(1).max() - prices)) / \
              ((prices.shift(1).max() - prices.shift(1).min()) + 1e-10)
        ad_line = (clv * volumes).cumsum()
        result['ad_trend'] = ad_line.diff(20).iloc[-1]
        
        return result
    
    def _detect_regime_gmm(self, returns: pd.Series) -> Dict[str, Any]:
        """Use Gaussian Mixture Model to detect volatility regimes"""
        result = {}
        
        try:
            # Prepare features
            features = pd.DataFrame({
                'returns': returns,
                'volatility': returns.rolling(self.volatility_lookback).std(),
                'momentum': returns.rolling(self.momentum_lookback).mean()
            }).dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=self.gmm_components,
                covariance_type='full',
                random_state=42
            )
            gmm.fit(scaled_features)
            
            # Predict current regime
            current_features = scaled_features[-1].reshape(1, -1)
            cluster = gmm.predict(current_features)[0]
            probabilities = gmm.predict_proba(current_features)[0]
            
            result['gmm_cluster'] = cluster
            result['gmm_probabilities'] = probabilities.tolist()
            result['gmm_confidence'] = max(probabilities)
            
            # Interpret cluster
            cluster_means = gmm.means_[cluster]
            if cluster_means[0] > 0.5:  # High positive returns
                result['gmm_interpretation'] = 'bull_regime'
            elif cluster_means[0] < -0.5:  # High negative returns
                result['gmm_interpretation'] = 'bear_regime'
            elif cluster_means[1] > 0.5:  # High volatility
                result['gmm_interpretation'] = 'high_vol_regime'
            else:
                result['gmm_interpretation'] = 'normal_regime'
                
        except Exception as e:
            logger.warning(f"GMM regime detection failed: {e}")
            result['gmm_cluster'] = -1
            result['gmm_interpretation'] = 'unknown'
        
        return result
    
    def _classify_regime(self, indicators: Dict[str, Any]) -> MarketRegime:
        """Classify the overall market regime based on all indicators"""
        
        # Extract key signals
        vol_regime = indicators.get('vol_regime', 'normal_volatility')
        trend_regime = indicators.get('trend_regime', 'sideways')
        rsi_regime = indicators.get('rsi_regime', 'neutral')
        momentum_score = indicators.get('momentum_score', 0)
        adx = indicators.get('adx', 0)
        
        # Check for crash conditions
        if (indicators.get('realized_volatility', 0) > self.volatility_thresholds['extreme'] and
            indicators.get('roc_1m', 0) < -0.15 and
            indicators.get('price_vs_ma200', 0) < -0.20):
            return MarketRegime.CRASH
        
        # Check for extreme volatility
        if vol_regime == 'extreme_volatility':
            return MarketRegime.HIGH_VOLATILITY
        
        # Check for strong trends
        if trend_regime == 'strong_uptrend' and adx > 25:
            return MarketRegime.TRENDING_UP
        elif trend_regime == 'strong_downtrend' and adx > 25:
            return MarketRegime.TRENDING_DOWN
        
        # Check for bull/bear markets
        if (indicators.get('price_vs_ma200', 0) > 0.20 and
            indicators.get('ma_alignment', 0) == 1 and
            momentum_score > 0.3):
            return MarketRegime.BULL_MARKET
        elif (indicators.get('price_vs_ma200', 0) < -0.20 and
              indicators.get('ma_alignment', 0) == -1 and
              momentum_score < -0.3):
            return MarketRegime.BEAR_MARKET
        
        # Check for high/low volatility
        if vol_regime == 'high_volatility':
            return MarketRegime.HIGH_VOLATILITY
        elif vol_regime == 'low_volatility':
            return MarketRegime.LOW_VOLATILITY
        
        # Check for sideways market
        if abs(momentum_score) < 0.2 and adx < 20:
            return MarketRegime.SIDEWAYS
        
        # Default to trending based on momentum
        if momentum_score > 0.2:
            return MarketRegime.TRENDING_UP
        elif momentum_score < -0.2:
            return MarketRegime.TRENDING_DOWN
        
        return MarketRegime.SIDEWAYS
    
    def _calculate_confidence(self, indicators: Dict[str, Any], regime: MarketRegime) -> float:
        """Calculate confidence score for the regime classification"""
        confidence_factors = []
        
        # Factor 1: Indicator agreement
        trend_signals = [
            indicators.get('ma_alignment', 0),
            np.sign(indicators.get('trend_slope', 0)),
            np.sign(indicators.get('momentum_score', 0))
        ]
        agreement = np.std(trend_signals)
        confidence_factors.append(1.0 - min(agreement, 1.0))
        
        # Factor 2: ADX (trend strength)
        adx = indicators.get('adx', 0)
        confidence_factors.append(min(adx / 50.0, 1.0))
        
        # Factor 3: RSI extremity (clearer signals at extremes)
        rsi = indicators.get('rsi', 50)
        rsi_extremity = abs(rsi - 50) / 50
        confidence_factors.append(rsi_extremity)
        
        # Factor 4: Volatility stability
        vol_percentile = indicators.get('volatility_percentile', 0.5)
        vol_stability = 1.0 - abs(vol_percentile - 0.5) * 2
        confidence_factors.append(vol_stability)
        
        # Factor 5: GMM confidence (if available)
        if 'gmm_confidence' in indicators:
            confidence_factors.append(indicators['gmm_confidence'])
        
        # Weighted average
        weights = [0.3, 0.25, 0.15, 0.15, 0.15]
        weighted_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return min(max(weighted_confidence, 0.0), 1.0)
    
    def _get_secondary_regime(self, primary: MarketRegime, indicators: Dict[str, Any]) -> Optional[MarketRegime]:
        """Determine secondary regime if applicable"""
        
        vol_regime = indicators.get('vol_regime', 'normal_volatility')
        
        # If primary is trend-based, secondary could be volatility-based
        if primary in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            if vol_regime == 'high_volatility':
                return MarketRegime.HIGH_VOLATILITY
            elif vol_regime == 'low_volatility':
                return MarketRegime.LOW_VOLATILITY
        
        # If primary is bull/bear, check for overbought/oversold conditions
        if primary == MarketRegime.BULL_MARKET:
            if indicators.get('rsi_regime') == 'overbought':
                return MarketRegime.HIGH_VOLATILITY
        elif primary == MarketRegime.BEAR_MARKET:
            if indicators.get('rsi_regime') == 'oversold':
                return MarketRegime.RECOVERY
        
        return None
    
    def _calculate_regime_probabilities(self, indicators: Dict[str, Any]) -> Dict[str, float]:
        """Calculate probabilities for each regime type"""
        probabilities = {}
        
        # Simplified probability estimation based on indicator values
        momentum_score = indicators.get('momentum_score', 0)
        vol_percentile = indicators.get('volatility_percentile', 0.5)
        adx = indicators.get('adx', 0)
        
        # Bull probability
        bull_prob = 0.5 + 0.3 * momentum_score + 0.2 * (1 - vol_percentile)
        probabilities['bull_market'] = min(max(bull_prob, 0.0), 1.0)
        
        # Bear probability
        bear_prob = 0.5 - 0.3 * momentum_score + 0.2 * vol_percentile
        probabilities['bear_market'] = min(max(bear_prob, 0.0), 1.0)
        
        # High vol probability
        high_vol_prob = vol_percentile * 0.7 + (1 - adx/100) * 0.3
        probabilities['high_volatility'] = min(max(high_vol_prob, 0.0), 1.0)
        
        # Sideways probability
        sideways_prob = (1 - abs(momentum_score)) * 0.6 + (1 - adx/50) * 0.4
        probabilities['sideways'] = min(max(sideways_prob, 0.0), 1.0)
        
        # Normalize
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        return probabilities
    
    def _generate_recommendations(self, regime: MarketRegime, indicators: Dict[str, Any]) -> List[str]:
        """Generate strategy recommendations based on detected regime"""
        recommendations = []
        
        if regime == MarketRegime.BULL_MARKET:
            recommendations.extend([
                "Consider momentum-based long strategies",
                "Use trend-following indicators for entry/exit",
                "Maintain higher position sizes with tight stop-losses",
                "Avoid counter-trend short positions"
            ])
        elif regime == MarketRegime.BEAR_MARKET:
            recommendations.extend([
                "Reduce overall portfolio exposure",
                "Consider defensive sectors and safe-haven assets",
                "Use inverse ETFs or put options for hedging",
                "Focus on capital preservation over growth"
            ])
        elif regime == MarketRegime.HIGH_VOLATILITY:
            recommendations.extend([
                "Reduce position sizes significantly",
                "Use wider stop-losses to avoid whipsaws",
                "Consider volatility-based products (VIX futures, options)",
                "Avoid highly leveraged positions"
            ])
        elif regime == MarketRegime.LOW_VOLATILITY:
            recommendations.extend([
                "Consider option selling strategies (premium collection)",
                "Use tighter stop-losses",
                "Look for breakout opportunities",
                "Increase position sizes moderately"
            ])
        elif regime == MarketRegime.TRENDING_UP:
            recommendations.extend([
                "Follow the trend with long positions",
                "Use moving average crossovers for timing",
                "Add to winners, cut losers quickly",
                "Avoid trying to pick tops"
            ])
        elif regime == MarketRegime.TRENDING_DOWN:
            recommendations.extend([
                "Consider short-selling or inverse products",
                "Use rallies to reduce long exposure",
                "Maintain defensive positioning",
                "Watch for oversold bounce opportunities"
            ])
        elif regime == MarketRegime.SIDEWAYS:
            recommendations.extend([
                "Use mean-reversion strategies",
                "Buy support, sell resistance",
                "Reduce position sizes",
                "Avoid trend-following strategies"
            ])
        elif regime == MarketRegime.CRASH:
            recommendations.extend([
                "EXIT ALL RISK POSITIONS IMMEDIATELY",
                "Move to cash and safe-haven assets",
                "Do NOT attempt to catch falling knives",
                "Wait for volatility to stabilize before re-entering"
            ])
        elif regime == MarketRegime.RECOVERY:
            recommendations.extend([
                "Gradually rebuild risk exposure",
                "Focus on high-quality assets",
                "Use dollar-cost averaging",
                "Monitor for secondary bottom formation"
            ])
        
        return recommendations
    
    def _assess_risk_level(self, indicators: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        risk_score = 0
        
        # Volatility risk
        vol_regime = indicators.get('vol_regime', 'normal_volatility')
        if vol_regime == 'extreme_volatility':
            risk_score += 4
        elif vol_regime == 'high_volatility':
            risk_score += 3
        elif vol_regime == 'normal_volatility':
            risk_score += 2
        else:
            risk_score += 1
        
        # Trend risk
        if indicators.get('price_vs_ma200', 0) < -0.20:
            risk_score += 3
        elif indicators.get('price_vs_ma200', 0) < -0.10:
            risk_score += 2
        elif indicators.get('price_vs_ma200', 0) > 0.20:
            risk_score += 1
        
        # Momentum risk
        momentum_score = indicators.get('momentum_score', 0)
        if momentum_score < -0.5:
            risk_score += 3
        elif momentum_score < -0.2:
            risk_score += 2
        elif momentum_score > 0.2:
            risk_score += 1
        
        # Map to risk level
        if risk_score >= 9:
            return 'extreme'
        elif risk_score >= 6:
            return 'high'
        elif risk_score >= 4:
            return 'medium'
        else:
            return 'low'


def create_regime_detector(config: Optional[Dict[str, Any]] = None) -> RegimeDetector:
    """Factory function to create regime detector with custom configuration"""
    if config is None:
        config = {}
    
    detector = RegimeDetector(
        volatility_lookback=config.get('volatility_lookback', 20),
        trend_lookback=config.get('trend_lookback', 50),
        momentum_lookback=config.get('momentum_lookback', 14),
        long_term_lookback=config.get('long_term_lookback', 252),
        use_gmm=config.get('use_gmm', True),
        gmm_components=config.get('gmm_components', 3)
    )
    
    logger.info(f"✅ Regime detector created with config: {config}")
    
    return detector
