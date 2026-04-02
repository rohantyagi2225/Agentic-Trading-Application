"""
Multi-Source Signal Integrator for Financial Trading Agents

This module integrates signals from multiple sources to generate comprehensive,
robust trading signals with higher predictive power than single-source approaches.

Key Features:
- Technical indicator signals (price-based)
- News sentiment analysis (NLP-based)
- External macro signals (economic data)
- Social media sentiment (alternative data)
- Signal fusion with adaptive weighting
- Confidence scoring and validation

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │          Multi-Source Signal Integrator                 │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ Technical    │  │ Sentiment    │  │ Macro        │  │
    │  │ Signals      │  │ Analysis     │  │ Indicators   │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │         Signal Fusion & Weighting Engine         │  │
    │  └──────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────┐
    │           Integrated Signal Output                      │
    │  - Direction (bullish/bearish/neutral)                 │
    │  - Strength (0-100%)                                   │
    │  - Confidence Score                                    │
    │  - Source Attribution                                  │
    └─────────────────────────────────────────────────────────┘

Author: FinAgent Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("SignalIntegrator")


class SignalSource(Enum):
    """Enumeration of signal sources"""
    TECHNICAL = "technical"
    SENTIMENT_NEWS = "sentiment_news"
    SENTIMENT_SOCIAL = "sentiment_social"
    MACRO_ECONOMIC = "macro_economic"
    FUNDAMENTAL = "fundamental"
    ALTERNATIVE_DATA = "alternative_data"
    CUSTOM = "custom"


class SignalType(Enum):
    """Type of trading signal"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class IndividualSignal:
    """Represents a single signal from one source"""
    source: SignalSource
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "description": self.description
        }


@dataclass
class IntegratedSignal:
    """Represents an integrated signal from multiple sources"""
    signal_type: SignalType
    overall_strength: float  # 0.0 to 1.0
    overall_confidence: float  # 0.0 to 1.0
    contributing_signals: List[IndividualSignal]
    source_weights: Dict[str, float]
    agreement_score: float  # How much sources agree (0-1)
    timestamp: datetime
    symbols: List[str]
    regime_context: Optional[str] = None
    explanation: str = ""
    action_recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "overall_strength": self.overall_strength,
            "overall_confidence": self.overall_confidence,
            "contributing_signals": [s.to_dict() for s in self.contributing_signals],
            "source_weights": self.source_weights,
            "agreement_score": self.agreement_score,
            "timestamp": self.timestamp.isoformat(),
            "symbols": self.symbols,
            "regime_context": self.regime_context,
            "explanation": self.explanation,
            "action_recommendation": self.action_recommendation
        }


class TechnicalSignalGenerator:
    """Generate signals from technical indicators"""
    
    def __init__(self):
        self.indicator_weights = {
            'trend': 0.35,      # Moving averages, ADX, etc.
            'momentum': 0.30,   # RSI, MACD, etc.
            'volatility': 0.20, # Bollinger Bands, ATR
            'volume': 0.15      # OBV, volume trends
        }
    
    def generate_signal(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None
    ) -> IndividualSignal:
        """Generate technical signal from price/volume data"""
        
        if high is None:
            high = prices
        if low is None:
            low = prices
        
        # Calculate component signals
        trend_signal = self._trend_signal(prices)
        momentum_signal = self._momentum_signal(prices)
        volatility_signal = self._volatility_signal(prices, high, low)
        volume_signal = self._volume_signal(prices, volumes) if volumes is not None else {'signal': 0, 'confidence': 0.5}
        
        # Combine signals with weights
        combined_signal = (
            trend_signal['signal'] * self.indicator_weights['trend'] +
            momentum_signal['signal'] * self.indicator_weights['momentum'] +
            volatility_signal['signal'] * self.indicator_weights['volatility'] +
            volume_signal['signal'] * self.indicator_weights['volume']
        )
        
        combined_confidence = (
            trend_signal['confidence'] * self.indicator_weights['trend'] +
            momentum_signal['confidence'] * self.indicator_weights['momentum'] +
            volatility_signal['confidence'] * self.indicator_weights['volatility'] +
            volume_signal['confidence'] * self.indicator_weights['volume']
        )
        
        # Normalize
        signal_strength = abs(combined_signal)
        signal_direction = 1 if combined_signal > 0 else (-1 if combined_signal < 0 else 0)
        
        signal_type = SignalType.NEUTRAL
        if signal_direction > 0.2:
            signal_type = SignalType.BULLISH
        elif signal_direction < -0.2:
            signal_type = SignalType.BEARISH
        
        # Generate description
        description = self._generate_technical_description(
            trend_signal, momentum_signal, volatility_signal, volume_signal
        )
        
        return IndividualSignal(
            source=SignalSource.TECHNICAL,
            signal_type=signal_type,
            strength=min(signal_strength, 1.0),
            confidence=min(combined_confidence, 1.0),
            timestamp=datetime.now(),
            metadata={
                'trend_signal': trend_signal,
                'momentum_signal': momentum_signal,
                'volatility_signal': volatility_signal,
                'volume_signal': volume_signal
            },
            description=description
        )
    
    def _trend_signal(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate trend-based signal"""
        # Moving average alignment
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        ma_200 = prices.rolling(200).mean()
        
        current_price = prices.iloc[-1]
        
        # Score based on MA alignment
        score = 0
        if current_price > ma_20.iloc[-1]:
            score += 0.25
        if current_price > ma_50.iloc[-1]:
            score += 0.25
        if current_price > ma_200.iloc[-1]:
            score += 0.25
        if ma_20.iloc[-1] > ma_50.iloc[-1] > ma_200.iloc[-1]:
            score += 0.25  # Golden cross alignment
        
        # Convert to -1 to 1 scale
        signal = (score - 0.5) * 2
        
        # Confidence based on MA separation
        ma_spread = (ma_20.iloc[-1] - ma_200.iloc[-1]) / ma_200.iloc[-1]
        confidence = min(abs(ma_spread) * 10 + 0.5, 1.0)
        
        return {'signal': signal, 'confidence': confidence}
    
    def _momentum_signal(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate momentum-based signal"""
        # RSI
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        macd_hist = macd - signal_line
        
        # Combine RSI and MACD
        rsi_signal = (rsi.iloc[-1] - 50) / 50  # -1 to 1
        macd_signal = np.sign(macd_hist.iloc[-1]) * min(abs(macd_hist.iloc[-1]) / prices.iloc[-1] * 100, 1.0)
        
        combined = 0.6 * rsi_signal + 0.4 * macd_signal
        
        # Confidence
        confidence = 0.5 + 0.5 * (abs(rsi.iloc[-1] - 50) / 50)
        
        return {'signal': combined, 'confidence': min(confidence, 1.0)}
    
    def _volatility_signal(self, prices: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, float]:
        """Calculate volatility-based signal"""
        # Bollinger Bands
        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        
        # Position within bands
        current_price = prices.iloc[-1]
        band_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1] + 1e-10)
        
        # Signal: buy near lower band, sell near upper band (mean reversion)
        signal = 0.5 - band_position  # Invert: low position = bullish
        
        # ATR for volatility measure
        tr = pd.concat([
            high - low,
            (high - prices.shift(1)).abs(),
            (low - prices.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_percent = (atr.iloc[-1] / prices.iloc[-1]) * 100
        
        # Confidence higher when volatility is normal (not extreme)
        if 1 < atr_percent < 5:
            confidence = 0.8
        elif atr_percent < 1 or atr_percent > 10:
            confidence = 0.4
        else:
            confidence = 0.6
        
        return {'signal': signal, 'confidence': confidence}
    
    def _volume_signal(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, float]:
        """Calculate volume-based signal"""
        # Volume trend
        vol_ma_20 = volumes.rolling(20).mean()
        volume_ratio = volumes.iloc[-1] / vol_ma_20.iloc[-1]
        
        # On-Balance Volume
        obv = (np.sign(prices.diff()) * volumes).cumsum()
        obv_trend = obv.diff(20).iloc[-1]
        
        # Combine signals
        volume_confirmation = 1 if volume_ratio > 1.2 else (0 if volume_ratio < 0.8 else 0.5)
        obv_signal = np.sign(obv_trend)
        
        signal = 0.5 * volume_confirmation + 0.5 * obv_signal
        
        # Confidence based on volume clarity
        confidence = 0.5 + 0.3 * min(abs(volume_ratio - 1), 1)
        
        return {'signal': signal, 'confidence': min(confidence, 1.0)}
    
    def _generate_technical_description(
        self,
        trend: Dict,
        momentum: Dict,
        volatility: Dict,
        volume: Dict
    ) -> str:
        """Generate human-readable technical analysis description"""
        parts = []
        
        # Trend
        if trend['signal'] > 0.3:
            parts.append("strong uptrend")
        elif trend['signal'] > 0.1:
            parts.append("moderate uptrend")
        elif trend['signal'] < -0.3:
            parts.append("strong downtrend")
        elif trend['signal'] < -0.1:
            parts.append("moderate downtrend")
        else:
            parts.append("sideways movement")
        
        # Momentum
        if momentum['signal'] > 0.3:
            parts.append("positive momentum")
        elif momentum['signal'] < -0.3:
            parts.append("negative momentum")
        
        # Volatility
        if volatility['signal'] > 0.3:
            parts.append("near lower Bollinger Band (potential bounce)")
        elif volatility['signal'] < -0.3:
            parts.append("near upper Bollinger Band (potential pullback)")
        
        return f"Technical analysis shows {'; '.join(parts)}. "


class SentimentAnalyzer:
    """Analyze news and social media sentiment"""
    
    def __init__(self):
        # Simple keyword-based sentiment (can be replaced with LLM/NLP models)
        self.positive_keywords = [
            'beat', 'exceed', 'growth', 'profit', 'revenue', 'upgrade', 
            'bullish', 'buy', 'outperform', 'positive', 'strong', 'gain',
            'rally', 'surge', 'breakthrough', 'innovation', 'success'
        ]
        self.negative_keywords = [
            'miss', 'decline', 'loss', 'revenue warning', 'downgrade',
            'bearish', 'sell', 'underperform', 'negative', 'weak', 'drop',
            'crash', 'plunge', 'lawsuit', 'scandal', 'failure', 'risk'
        ]
    
    def analyze_news_sentiment(
        self,
        headlines: List[str],
        articles: Optional[List[str]] = None
    ) -> IndividualSignal:
        """Analyze news sentiment from headlines/articles"""
        
        if not headlines:
            return IndividualSignal(
                source=SignalSource.SENTIMENT_NEWS,
                signal_type=SignalType.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                description="No news data available"
            )
        
        # Analyze each headline
        sentiment_scores = []
        for headline in headlines:
            score = self._analyze_text_sentiment(headline)
            sentiment_scores.append(score)
        
        # Aggregate sentiment
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_std = np.std(sentiment_scores)
        
        # Determine signal type
        signal_type = SignalType.NEUTRAL
        if avg_sentiment > 0.2:
            signal_type = SignalType.BULLISH
        elif avg_sentiment < -0.2:
            signal_type = SignalType.BEARISH
        
        # Confidence based on consistency and sample size
        sample_size_factor = min(len(headlines) / 10, 1.0)
        consistency_factor = 1.0 - min(sentiment_std, 1.0)
        confidence = (sample_size_factor * 0.6 + consistency_factor * 0.4)
        
        strength = min(abs(avg_sentiment), 1.0)
        
        description = f"Analyzed {len(headlines)} news headlines. "
        if avg_sentiment > 0.3:
            description += "Overwhelmingly positive sentiment detected."
        elif avg_sentiment > 0.1:
            description += "Moderately positive sentiment."
        elif avg_sentiment < -0.3:
            description += "Overwhelmingly negative sentiment detected."
        elif avg_sentiment < -0.1:
            description += "Moderately negative sentiment."
        else:
            description += "Mixed/neutral sentiment."
        
        return IndividualSignal(
            source=SignalSource.SENTIMENT_NEWS,
            signal_type=signal_type,
            strength=strength,
            confidence=min(confidence, 1.0),
            timestamp=datetime.now(),
            metadata={
                'avg_sentiment': avg_sentiment,
                'num_headlines': len(headlines),
                'sentiment_distribution': sentiment_scores
            },
            description=description
        )
    
    def analyze_social_sentiment(
        self,
        posts: List[Dict[str, str]],
        platform: str = "twitter"
    ) -> IndividualSignal:
        """Analyze social media sentiment"""
        
        if not posts:
            return IndividualSignal(
                source=SignalSource.SENTIMENT_SOCIAL,
                signal_type=SignalType.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                description="No social media data available"
            )
        
        # Extract text from posts
        texts = [post.get('text', '') for post in posts]
        
        # Analyze sentiment
        sentiment_scores = [self._analyze_text_sentiment(text) for text in texts]
        
        # Calculate metrics
        avg_sentiment = np.mean(sentiment_scores)
        total_posts = len(posts)
        
        # Viral coefficient (how much engagement)
        engagements = [post.get('likes', 0) + post.get('shares', 0) for post in posts]
        avg_engagement = np.mean(engagements) if engagements else 0
        
        # Determine signal
        signal_type = SignalType.NEUTRAL
        if avg_sentiment > 0.25:
            signal_type = SignalType.BULLISH
        elif avg_sentiment < -0.25:
            signal_type = SignalType.BEARISH
        
        # Confidence
        volume_factor = min(total_posts / 100, 1.0)
        engagement_factor = min(avg_engagement / 100, 1.0)
        consistency_factor = 1.0 - np.std(sentiment_scores)
        
        confidence = (
            volume_factor * 0.4 +
            engagement_factor * 0.3 +
            consistency_factor * 0.3
        )
        
        strength = min(abs(avg_sentiment), 1.0)
        
        description = f"Analyzed {total_posts} social media posts from {platform}. "
        description += f"Average sentiment: {avg_sentiment:.2f}"
        
        return IndividualSignal(
            source=SignalSource.SENTIMENT_SOCIAL,
            signal_type=signal_type,
            strength=strength,
            confidence=min(confidence, 1.0),
            timestamp=datetime.now(),
            metadata={
                'platform': platform,
                'avg_sentiment': avg_sentiment,
                'total_posts': total_posts,
                'avg_engagement': avg_engagement
            },
            description=description
        )
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis (-1 to 1)"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Normalize to -1 to 1
        sentiment = (positive_count - negative_count) / total
        
        return sentiment


class MacroSignalGenerator:
    """Generate signals from macroeconomic indicators"""
    
    def __init__(self):
        self.indicator_impact = {
            'interest_rates': -0.3,      # Rising rates typically negative for stocks
            'inflation': -0.2,           # High inflation uncertainty
            'gdp_growth': 0.25,          # GDP growth positive
            'unemployment': -0.15,       # High unemployment negative
            'pmi': 0.2,                  # PMI > 50 expansionary
            'consumer_confidence': 0.15, # Consumer spending driver
            'yield_curve': 0.15          # Normal curve positive, inverted negative
        }
    
    def generate_signal(
        self,
        macro_data: Dict[str, Any]
    ) -> IndividualSignal:
        """Generate signal from macroeconomic indicators"""
        
        if not macro_data:
            return IndividualSignal(
                source=SignalSource.MACRO_ECONOMIC,
                signal_type=SignalType.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                description="No macroeconomic data available"
            )
        
        # Calculate weighted macro signal
        weighted_signal = 0
        total_weight = 0
        
        for indicator, value in macro_data.items():
            if indicator in self.indicator_impact:
                # Normalize value (assuming most are percentages or indices)
                normalized_value = self._normalize_macro_value(indicator, value)
                
                # Apply impact weight
                impact = self.indicator_impact[indicator]
                weighted_signal += normalized_value * impact
                total_weight += abs(impact)
        
        # Normalize final signal
        if total_weight > 0:
            final_signal = weighted_signal / total_weight
        else:
            final_signal = 0
        
        # Determine signal type
        signal_type = SignalType.NEUTRAL
        if final_signal > 0.2:
            signal_type = SignalType.BULLISH
        elif final_signal < -0.2:
            signal_type = SignalType.BEARISH
        
        # Confidence based on data availability
        data_coverage = len(macro_data) / len(self.indicator_impact)
        confidence = min(data_coverage * 0.8 + 0.2, 1.0)
        
        strength = min(abs(final_signal), 1.0)
        
        description = f"Analyzed {len(macro_data)} macroeconomic indicators. "
        if final_signal > 0.3:
            description += "Favorable macro environment."
        elif final_signal > 0.1:
            description += "Slightly positive macro conditions."
        elif final_signal < -0.3:
            description += "Challenging macro environment."
        elif final_signal < -0.1:
            description += "Slightly negative macro conditions."
        else:
            description += "Mixed macro signals."
        
        return IndividualSignal(
            source=SignalSource.MACRO_ECONOMIC,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={
                'indicators_analyzed': list(macro_data.keys()),
                'macro_signal': final_signal
            },
            description=description
        )
    
    def _normalize_macro_value(self, indicator: str, value: Any) -> float:
        """Normalize macro values to -1 to 1 scale"""
        try:
            value_float = float(value)
            
            if indicator == 'interest_rates':
                # Rates above 5% restrictive, below 2% accommodative
                return max(-1, min(1, (5 - value_float) / 3))
            elif indicator == 'inflation':
                # 2% target, above 4% bad, below 1% deflationary concern
                if value_float > 4:
                    return -1
                elif value_float < 1:
                    return -0.5
                else:
                    return (3 - value_float) / 2
            elif indicator == 'gdp_growth':
                # >3% strong, <0% recession
                return max(-1, min(1, value_float / 3))
            elif indicator == 'unemployment':
                # <4% good, >8% bad
                return max(-1, min(1, (6 - value_float) / 2))
            elif indicator == 'pmi':
                # >50 expansion, <45 contraction
                return max(-1, min(1, (value_float - 50) / 5))
            elif indicator == 'consumer_confidence':
                # Normalize around 100 (neutral)
                return max(-1, min(1, (value_float - 100) / 20))
            elif indicator == 'yield_curve':
                # 10y-2y spread: >0 normal, <0 inverted (recession signal)
                return max(-1, min(1, value_float))
            else:
                return 0
        except (ValueError, TypeError):
            return 0


class SignalFusionEngine:
    """Fuse multiple signals into unified trading signal"""
    
    def __init__(self):
        # Default source weights (can be adapted based on regime)
        self.default_weights = {
            SignalSource.TECHNICAL: 0.35,
            SignalSource.SENTIMENT_NEWS: 0.20,
            SignalSource.SENTIMENT_SOCIAL: 0.10,
            SignalSource.MACRO_ECONOMIC: 0.25,
            SignalSource.FUNDAMENTAL: 0.10
        }
        
        # Regime-adaptive weights
        self.regime_weight_adjustments = {
            'high_volatility': {
                SignalSource.TECHNICAL: -0.10,      # Reduce tech reliance
                SignalSource.MACRO_ECONOMIC: +0.15  # Increase macro focus
            },
            'low_volatility': {
                SignalSource.TECHNICAL: +0.10,      # Increase tech signals
                SignalSource.SENTIMENT_SOCIAL: +0.05
            },
            'bull_market': {
                SignalSource.SENTIMENT_NEWS: +0.10, # Momentum driven by news
            },
            'bear_market': {
                SignalSource.MACRO_ECONOMIC: +0.15, # Fundamentals matter more
            },
            'crash': {
                SignalSource.TECHNICAL: -0.15,
                SignalSource.MACRO_ECONOMIC: +0.20  # Flight to safety
            }
        }
    
    def fuse_signals(
        self,
        individual_signals: List[IndividualSignal],
        regime_context: Optional[str] = None
    ) -> IntegratedSignal:
        """Fuse multiple individual signals into integrated signal"""
        
        if not individual_signals:
            raise ValueError("At least one individual signal required")
        
        # Adjust weights based on regime
        weights = self.default_weights.copy()
        if regime_context and regime_context in self.regime_weight_adjustments:
            adjustments = self.regime_weight_adjustments[regime_context]
            for source, adjustment in adjustments.items():
                if source in weights:
                    weights[source] = max(0.05, min(0.6, weights[source] + adjustment))
        
        # Normalize weights
        total_weight = sum(weights.get(sig.source, 0) for sig in individual_signals)
        if total_weight > 0:
            normalized_weights = {
                source.value: weight / total_weight
                for source, weight in weights.items()
            }
        else:
            normalized_weights = {source.value: 1.0/len(individual_signals) for source in weights}
        
        # Calculate weighted signal
        weighted_signal = 0
        weighted_confidence = 0
        
        for signal in individual_signals:
            weight = normalized_weights.get(signal.source.value, 0)
            signal_direction = 0
            if signal.signal_type == SignalType.BULLISH:
                signal_direction = 1
            elif signal.signal_type == SignalType.BEARISH:
                signal_direction = -1
            
            weighted_signal += signal_direction * signal.strength * weight
            weighted_confidence += signal.confidence * weight
        
        # Determine integrated signal type
        signal_type = SignalType.NEUTRAL
        if weighted_signal > 0.15:
            signal_type = SignalType.BULLISH
        elif weighted_signal < -0.15:
            signal_type = SignalType.BEARISH
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement(individual_signals)
        
        # Boost confidence if sources agree (PREVENT NaN WITH FALLBACK)
        if weighted_confidence > 0 and not np.isnan(weighted_confidence):
            final_confidence = weighted_confidence * (0.7 + 0.3 * agreement_score)
        else:
            # Fallback: use average signal confidence or default to low confidence
            avg_confidence = np.mean([s.confidence for s in individual_signals])
            final_confidence = avg_confidence * (0.5 + 0.3 * agreement_score)
            logger.warning(f"⚠️ Weighted confidence was invalid, using fallback: {final_confidence:.1%}")
        
        # Ensure confidence is valid (not NaN, not infinite, bounded 0-1)
        if np.isnan(final_confidence) or np.isinf(final_confidence):
            final_confidence = 0.5  # Default neutral confidence
            logger.warning("⚠️ Final confidence was invalid, using default 50%")
        
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Generate explanation
        explanation = self._generate_explanation(
            individual_signals, signal_type, weighted_signal, agreement_score
        )
        
        # Generate action recommendation
        action_rec = self._generate_action_recommendation(
            signal_type, weighted_signal, final_confidence
        )
        
        return IntegratedSignal(
            signal_type=signal_type,
            overall_strength=min(abs(weighted_signal), 1.0),
            overall_confidence=min(final_confidence, 1.0),
            contributing_signals=individual_signals,
            source_weights=normalized_weights,
            agreement_score=agreement_score,
            timestamp=datetime.now(),
            symbols=[],  # Can be set externally
            regime_context=regime_context,
            explanation=explanation,
            action_recommendation=action_rec
        )
    
    def _calculate_agreement(self, signals: List[IndividualSignal]) -> float:
        """Calculate how much individual signals agree with each other"""
        if len(signals) < 2:
            return 1.0
        
        # Convert signals to numeric values
        signal_values = []
        for sig in signals:
            if sig.signal_type == SignalType.BULLISH:
                signal_values.append(1)
            elif sig.signal_type == SignalType.BEARISH:
                signal_values.append(-1)
            else:
                signal_values.append(0)
        
        # Calculate variance
        variance = np.var(signal_values)
        
        # Convert to agreement score (0 variance = perfect agreement)
        agreement = 1.0 - min(variance / 2, 1.0)
        
        return agreement
    
    def _generate_explanation(
        self,
        signals: List[IndividualSignal],
        signal_type: SignalType,
        weighted_signal: float,
        agreement_score: float
    ) -> str:
        """Generate human-readable explanation"""
        parts = []
        
        # Overall direction
        if signal_type == SignalType.BULLISH:
            parts.append(f"BULLISH signal (strength: {abs(weighted_signal):.1%})")
        elif signal_type == SignalType.BEARISH:
            parts.append(f"BEARISH signal (strength: {abs(weighted_signal):.1%})")
        else:
            parts.append("NEUTRAL signal")
        
        # Source contributions
        bullish_sources = [s.source.value.replace('_', ' ') for s in signals if s.signal_type == SignalType.BULLISH]
        bearish_sources = [s.source.value.replace('_', ' ') for s in signals if s.signal_type == SignalType.BEARISH]
        
        if bullish_sources:
            parts.append(f"Bullish drivers: {', '.join(bullish_sources)}")
        if bearish_sources:
            parts.append(f"Bearish drivers: {', '.join(bearish_sources)}")
        
        # Agreement level
        if agreement_score > 0.8:
            parts.append("Strong consensus across sources")
        elif agreement_score > 0.5:
            parts.append("Moderate agreement among sources")
        else:
            parts.append("Mixed signals from different sources")
        
        return ". ".join(parts) + "."
    
    def _generate_action_recommendation(
        self,
        signal_type: SignalType,
        strength: float,
        confidence: float
    ) -> str:
        """Generate actionable recommendation"""
        
        if signal_type == SignalType.NEUTRAL:
            return "HOLD: No clear directional signal. Maintain current positions."
        
        action_verb = "BUY" if signal_type == SignalType.BULLISH else "SELL"
        
        if strength > 0.6 and confidence > 0.7:
            conviction = "STRONG"
            size_rec = "Consider larger-than-normal position size"
        elif strength > 0.4 and confidence > 0.5:
            conviction = "MODERATE"
            size_rec = "Use standard position sizing"
        else:
            conviction = "WEAK"
            size_rec = "Reduce position size due to uncertainty"
        
        return (
            f"{conviction} {action_verb}: {size_rec}. "
            f"Confidence: {confidence:.1%}, Strength: {strength:.1%}"
        )


class MultiSourceSignalIntegrator:
    """
    Main class integrating all signal sources
    
    Usage:
        integrator = MultiSourceSignalIntegrator()
        
        # Add data sources
        integrator.add_price_data(prices, volumes)
        integrator.add_news_data(headlines)
        integrator.add_macro_data(macro_indicators)
        
        # Generate integrated signal
        signal = integrator.generate_integrated_signal()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize signal generators
        self.technical_generator = TechnicalSignalGenerator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.macro_generator = MacroSignalGenerator()
        self.fusion_engine = SignalFusionEngine()
        
        # Storage for input data
        self.price_data: Optional[pd.Series] = None
        self.volume_data: Optional[pd.Series] = None
        self.news_data: List[str] = []
        self.social_data: List[Dict[str, str]] = []
        self.macro_data: Dict[str, Any] = {}
        
        # Regime context (optional)
        self.regime_context: Optional[str] = None
        
        logger.info("✅ Multi-source signal integrator initialized")
    
    def add_price_data(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None
    ):
        """Add price and volume data"""
        self.price_data = prices
        self.volume_data = volumes
        self.high_data = high
        self.low_data = low
        logger.debug(f"Added price data: {len(prices)} periods")
    
    def add_news_data(self, headlines: List[str], articles: Optional[List[str]] = None):
        """Add news headlines and articles"""
        self.news_data = headlines
        logger.debug(f"Added {len(headlines)} news headlines")
    
    def add_social_data(self, posts: List[Dict[str, str]], platform: str = "twitter"):
        """Add social media posts"""
        self.social_data = posts
        logger.debug(f"Added {len(posts)} social media posts from {platform}")
    
    def add_macro_data(self, macro_indicators: Dict[str, Any]):
        """Add macroeconomic indicators"""
        self.macro_data = macro_indicators
        logger.debug(f"Added {len(macro_indicators)} macro indicators")
    
    def set_regime_context(self, regime: str):
        """Set current market regime for adaptive weighting"""
        self.regime_context = regime
        logger.debug(f"Set regime context: {regime}")
    
    def generate_integrated_signal(
        self,
        symbols: Optional[List[str]] = None
    ) -> IntegratedSignal:
        """Generate integrated signal from all available sources"""
        
        if self.price_data is None:
            raise ValueError("Price data required. Call add_price_data() first.")
        
        individual_signals = []
        
        # 1. Generate technical signal
        try:
            tech_signal = self.technical_generator.generate_signal(
                self.price_data,
                self.volume_data,
                self.high_data,
                self.low_data
            )
            individual_signals.append(tech_signal)
            logger.debug(f"Generated technical signal: {tech_signal.signal_type.value}")
        except Exception as e:
            logger.warning(f"Technical signal generation failed: {e}")
        
        # 2. Generate news sentiment signal
        if self.news_data:
            try:
                news_signal = self.sentiment_analyzer.analyze_news_sentiment(self.news_data)
                individual_signals.append(news_signal)
                logger.debug(f"Generated news sentiment signal: {news_signal.signal_type.value}")
            except Exception as e:
                logger.warning(f"News sentiment analysis failed: {e}")
        
        # 3. Generate social sentiment signal
        if self.social_data:
            try:
                social_signal = self.sentiment_analyzer.analyze_social_sentiment(
                    self.social_data,
                    platform="twitter"
                )
                individual_signals.append(social_signal)
                logger.debug(f"Generated social sentiment signal: {social_signal.signal_type.value}")
            except Exception as e:
                logger.warning(f"Social sentiment analysis failed: {e}")
        
        # 4. Generate macro signal
        if self.macro_data:
            try:
                macro_signal = self.macro_generator.generate_signal(self.macro_data)
                individual_signals.append(macro_signal)
                logger.debug(f"Generated macro signal: {macro_signal.signal_type.value}")
            except Exception as e:
                logger.warning(f"Macro signal generation failed: {e}")
        
        if not individual_signals:
            raise ValueError("No signals generated. Check input data.")
        
        # 5. Fuse all signals
        integrated_signal = self.fusion_engine.fuse_signals(
            individual_signals,
            self.regime_context
        )
        
        # Add symbols if provided
        if symbols:
            integrated_signal.symbols = symbols
        
        logger.info(
            f"🎯 Generated integrated signal: {integrated_signal.signal_type.value} "
            f"(strength: {integrated_signal.overall_strength:.1%}, "
            f"confidence: {integrated_signal.overall_confidence:.1%})"
        )
        
        return integrated_signal
    
    def clear_data(self):
        """Clear all stored data"""
        self.price_data = None
        self.volume_data = None
        self.news_data = []
        self.social_data = []
        self.macro_data = {}
        self.regime_context = None
        logger.debug("Cleared all data")


def create_signal_integrator(config: Optional[Dict[str, Any]] = None) -> MultiSourceSignalIntegrator:
    """Factory function to create signal integrator"""
    return MultiSourceSignalIntegrator(config)
