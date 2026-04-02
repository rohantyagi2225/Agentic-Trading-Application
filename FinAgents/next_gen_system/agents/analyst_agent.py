"""
AnalystAgent - Multimodal Financial Analysis Agent

This agent processes multiple data modalities including:
- Time-series data (prices, volumes)
- Text data (news, sentiment, filings)
- Chart patterns (technical analysis)

Key Features:
------------
- Sentiment analysis from news
- Macroeconomic indicator analysis
- Event impact modeling
- Chart pattern recognition
- Cross-modal data fusion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """Comprehensive analysis report"""
    symbol: str
    overall_sentiment: float
    technical_score: float
    fundamental_score: float
    macro_impact: float
    event_impacts: List[Dict[str, Any]]
    chart_patterns: List[Dict[str, Any]]
    key_insights: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class AnalystAgent:
    """
    Domain-specialized analyst agent with multimodal processing capabilities
    
    Combines technical analysis, sentiment analysis, and macroeconomic
    factors to provide comprehensive market insights.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize the AnalystAgent
        
        Args:
            agent_id: Unique identifier
        """
        self.agent_id = agent_id
        
        # News sentiment cache
        self.sentiment_cache: Dict[str, List[float]] = {}
        
        # Event database
        self.event_database: List[Dict[str, Any]] = []
        
        # Pattern recognition models
        self.chart_patterns_db = self._initialize_pattern_database()
        
        logger.info(f"AnalystAgent {agent_id} initialized")
    
    def analyze_symbol(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        news_headlines: Optional[List[str]] = None,
        macro_indicators: Optional[Dict[str, float]] = None,
    ) -> AnalysisReport:
        """
        Perform comprehensive multimodal analysis
        
        Args:
            symbol: Stock symbol to analyze
            price_data: OHLCV data
            news_headlines: Recent news headlines
            macro_indicators: Macroeconomic indicators
            
        Returns:
            Comprehensive analysis report
        """
        # Technical analysis
        technical_score, technical_insights = self._technical_analysis(price_data)
        
        # Sentiment analysis
        overall_sentiment, sentiment_insights = self._analyze_sentiment(symbol, news_headlines)
        
        # Macro impact
        macro_impact, macro_insights = self._analyze_macro_impact(macro_indicators)
        
        # Chart pattern detection
        chart_patterns = self._detect_chart_patterns(price_data)
        
        # Event impact
        event_impacts = self._assess_event_impacts(symbol)
        
        # Combine scores
        fundamental_score = (overall_sentiment + 0.5) * 5  # Scale to 0-10
        
        # Generate insights
        all_insights = technical_insights + sentiment_insights + macro_insights
        
        # Calculate confidence
        data_sources = sum([
            1 if news_headlines else 0,
            1 if macro_indicators else 0,
            1 if len(price_data) > 20 else 0,
        ])
        confidence = min(data_sources / 3.0, 1.0)
        
        return AnalysisReport(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            macro_impact=macro_impact,
            event_impacts=event_impacts,
            chart_patterns=chart_patterns,
            key_insights=all_insights,
            confidence=confidence,
        )
    
    def _technical_analysis(
        self,
        price_data: pd.DataFrame,
    ) -> Tuple[float, List[str]]:
        """Perform technical analysis on price data"""
        insights = []
        score = 5.0  # Neutral starting point
        
        if len(price_data) < 10:
            return 5.0, ["Insufficient data for technical analysis"]
        
        closes = price_data["close"]
        highs = price_data["high"]
        lows = price_data["low"]
        volumes = price_data.get("volume", pd.Series([1] * len(closes)))
        
        # Moving averages
        ma_20 = closes.rolling(20).mean().iloc[-1]
        ma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else ma_20
        current_price = closes.iloc[-1]
        
        if current_price > ma_20 > ma_50:
            score += 1.5
            insights.append("Bullish moving average alignment")
        elif current_price < ma_20 < ma_50:
            score -= 1.5
            insights.append("Bearish moving average alignment")
        
        # RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if current_rsi > 70:
            score -= 1.0
            insights.append(f"Overbought conditions (RSI: {current_rsi:.1f})")
        elif current_rsi < 30:
            score += 1.0
            insights.append(f"Oversold conditions (RSI: {current_rsi:.1f})")
        
        # Volume analysis
        avg_volume = volumes.rolling(20).mean().iloc[-1]
        current_volume = volumes.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 2:
            insights.append(f"Unusually high volume ({volume_ratio:.1f}x average)")
            score += 0.5 if current_price > closes.iloc[-2] else -0.5
        elif volume_ratio < 0.5:
            insights.append(f"Low volume ({volume_ratio:.1f}x average)")
        
        # Volatility
        returns = closes.pct_change()
        volatility = returns.std()
        if volatility > 0.03:
            insights.append(f"High volatility ({volatility*100:.1f}% daily)")
        
        # Normalize score to 0-10 scale
        score = max(0, min(10, score))
        
        return score, insights
    
    def _analyze_sentiment(
        self,
        symbol: str,
        headlines: Optional[List[str]],
    ) -> Tuple[float, List[str]]:
        """Analyze news sentiment"""
        if not headlines:
            return 0.5, ["No news sentiment available"]
        
        sentiment_scores = []
        insights = []
        
        positive_words = [
            "surge", "jump", "rally", "gain", "beat", "outperform", 
            "upgrade", "bullish", "buy", "strong", "growth", "record"
        ]
        negative_words = [
            "plunge", "drop", "fall", "decline", "miss", "underperform",
            "downgrade", "bearish", "sell", "weak", "loss", "crash"
        ]
        
        for headline in headlines[:10]:  # Analyze up to 10 headlines
            headline_lower = headline.lower()
            
            positive_count = sum(1 for word in positive_words if word in headline_lower)
            negative_count = sum(1 for word in negative_words if word in headline_lower)
            
            if positive_count > negative_count:
                sentiment_scores.append(0.8)
            elif negative_count > positive_count:
                sentiment_scores.append(0.2)
            else:
                sentiment_scores.append(0.5)
        
        overall_sentiment = np.mean(sentiment_scores)
        
        if overall_sentiment > 0.6:
            insights.append(f"Positive news sentiment ({overall_sentiment:.2f})")
        elif overall_sentiment < 0.4:
            insights.append(f"Negative news sentiment ({overall_sentiment:.2f})")
        else:
            insights.append(f"Neutral news sentiment ({overall_sentiment:.2f})")
        
        # Cache for future reference
        self.sentiment_cache[symbol] = sentiment_scores
        
        return overall_sentiment, insights
    
    def _analyze_macro_impact(
        self,
        indicators: Optional[Dict[str, float]],
    ) -> Tuple[float, List[str]]:
        """Analyze macroeconomic impact"""
        if not indicators:
            return 0.0, ["No macroeconomic data available"]
        
        insights = []
        impact_score = 0.0
        
        # Interest rate impact
        if "interest_rate_change" in indicators:
            rate_change = indicators["interest_rate_change"]
            if rate_change > 0.25:
                impact_score -= 0.5
                insights.append(f"Rising interest rates (+{rate_change}%) - headwind for growth")
            elif rate_change < -0.25:
                impact_score += 0.5
                insights.append(f"Falling interest rates ({rate_change}%) - tailwind for growth")
        
        # Inflation impact
        if "inflation_rate" in indicators:
            inflation = indicators["inflation_rate"]
            if inflation > 3:
                impact_score -= 0.3
                insights.append(f"High inflation ({inflation}%) - economic concern")
            elif inflation < 1:
                impact_score += 0.2
                insights.append(f"Low inflation ({inflation}%) - accommodative environment")
        
        # GDP growth impact
        if "gdp_growth" in indicators:
            gdp = indicators["gdp_growth"]
            if gdp > 2:
                impact_score += 0.4
                insights.append(f"Strong GDP growth ({gdp}%) - positive for earnings")
            elif gdp < 0:
                impact_score -= 0.6
                insights.append(f"Economic contraction ({gdp}%) - recession risk")
        
        # VIX impact
        if "vix_level" in indicators:
            vix = indicators["vix_level"]
            if vix > 30:
                impact_score -= 0.4
                insights.append(f"High market fear (VIX: {vix})")
            elif vix < 15:
                impact_score += 0.2
                insights.append(f"Low market volatility (VIX: {vix}) - complacency")
        
        return impact_score, insights
    
    def _detect_chart_patterns(
        self,
        price_data: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Detect chart patterns in price data"""
        patterns = []
        
        if len(price_data) < 30:
            return patterns
        
        highs = price_data["high"]
        lows = price_data["low"]
        closes = price_data["close"]
        
        # Simple pattern detection logic
        
        # Double top detection
        if self._detect_double_top(highs, closes):
            patterns.append({
                "type": "double_top",
                "signal": "bearish",
                "reliability": 0.7,
                "timeframe": "medium-term",
            })
        
        # Double bottom detection
        if self._detect_double_bottom(lows, closes):
            patterns.append({
                "type": "double_bottom",
                "signal": "bullish",
                "reliability": 0.7,
                "timeframe": "medium-term",
            })
        
        # Trend detection
        trend = self._detect_trend(closes)
        if trend != "sideways":
            patterns.append({
                "type": f"{trend}_trend",
                "signal": trend,
                "reliability": 0.6,
                "timeframe": "long-term",
            })
        
        return patterns
    
    def _detect_double_top(self, highs: pd.Series, closes: pd.Series) -> bool:
        """Detect double top pattern"""
        # Simplified detection logic
        recent_highs = highs.tail(30)
        if len(recent_highs) < 30:
            return False
        
        peak1 = recent_highs.iloc[10:15].max()
        peak2 = recent_highs.iloc[20:25].max()
        trough = recent_highs.iloc[15:20].min()
        
        # Check if peaks are similar and separated by trough
        return abs(peak1 - peak2) / peak1 < 0.03 and trough < peak1 * 0.95
    
    def _detect_double_bottom(self, lows: pd.Series, closes: pd.Series) -> bool:
        """Detect double bottom pattern"""
        recent_lows = lows.tail(30)
        if len(recent_lows) < 30:
            return False
        
        bottom1 = recent_lows.iloc[10:15].min()
        bottom2 = recent_lows.iloc[20:25].min()
        peak = recent_lows.iloc[15:20].max()
        
        # Check if bottoms are similar and separated by peak
        return abs(bottom1 - bottom2) / bottom1 < 0.03 and peak > bottom1 * 1.05
    
    def _detect_trend(self, closes: pd.Series) -> str:
        """Detect trend direction"""
        if len(closes) < 50:
            return "sideways"
        
        ma_20 = closes.rolling(20).mean()
        ma_50 = closes.rolling(50).mean()
        
        current = closes.iloc[-1]
        ma20 = ma_20.iloc[-1]
        ma50 = ma_50.iloc[-1]
        
        if current > ma20 > ma50:
            return "uptrend"
        elif current < ma20 < ma50:
            return "downtrend"
        else:
            return "sideways"
    
    def _assess_event_impacts(self, symbol: str) -> List[Dict[str, Any]]:
        """Assess impact of known events"""
        impacts = []
        
        # Filter events for this symbol
        symbol_events = [e for e in self.event_database if e.get("symbol") == symbol]
        
        for event in symbol_events:
            impact = {
                "event_type": event.get("type"),
                "date": event.get("date"),
                "expected_impact": event.get("impact", "neutral"),
                "description": event.get("description", ""),
            }
            impacts.append(impact)
        
        return impacts
    
    def _initialize_pattern_database(self) -> Dict[str, Any]:
        """Initialize chart pattern recognition database"""
        return {
            "reversal_patterns": [
                "head_and_shoulders", "double_top", "double_bottom",
                "triple_top", "triple_bottom", "rising_wedge", "falling_wedge"
            ],
            "continuation_patterns": [
                "bull_flag", "bear_flag", "pennant", "symmetrical_triangle",
                "ascending_triangle", "descending_triangle", "cup_and_handle"
            ],
            "pattern_reliabilities": {
                "head_and_shoulders": 0.75,
                "double_top": 0.70,
                "double_bottom": 0.70,
                "bull_flag": 0.65,
                "bear_flag": 0.65,
            }
        }
    
    def add_event(
        self,
        symbol: str,
        event_type: str,
        date: str,
        impact: str = "neutral",
        description: str = "",
    ):
        """Add event to database"""
        self.event_database.append({
            "symbol": symbol,
            "type": event_type,
            "date": date,
            "impact": impact,
            "description": description,
        })
        logger.info(f"Added event for {symbol}: {event_type} on {date}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "sentiment_symbols_tracked": list(self.sentiment_cache.keys()),
            "events_in_database": len(self.event_database),
            "patterns_in_database": len(self.chart_patterns_db.get("reversal_patterns", [])) + 
                                   len(self.chart_patterns_db.get("continuation_patterns", [])),
        }
