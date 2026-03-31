"""
Synthetic Data Generation Module

Provides synthetic dataset generation for research and model training,
including chart-text pairs, report summaries, and labeled training data.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import random

import pandas as pd
import numpy as np


@dataclass
class ChartTextPair:
    """Represents a price pattern with corresponding news narrative."""
    price_pattern: str
    pattern_features: dict
    news_narrative: str
    sentiment: float
    expected_direction: str  # BUY/SELL/HOLD


@dataclass
class ReportSummaryPair:
    """Represents a full report with its summary."""
    full_report: str
    summary: str
    key_metrics: dict
    sentiment: float


class SyntheticDataGenerator:
    """
    Generates synthetic datasets for research and training.
    
    Creates chart-text pairs, report summaries, and labeled training data
    for machine learning models.
    """
    
    # Pattern detection parameters
    PATTERNS = {
        'double_top': {
            'description': 'Price reaches similar high twice with decline in between',
            'direction': 'SELL',
            'sentiment_range': (-0.8, -0.3)
        },
        'double_bottom': {
            'description': 'Price reaches similar low twice with rally in between',
            'direction': 'BUY',
            'sentiment_range': (0.3, 0.8)
        },
        'head_shoulders': {
            'description': 'Three peaks with middle highest, followed by decline',
            'direction': 'SELL',
            'sentiment_range': (-0.9, -0.4)
        },
        'inverse_head_shoulders': {
            'description': 'Three troughs with middle lowest, followed by rally',
            'direction': 'BUY',
            'sentiment_range': (0.4, 0.9)
        },
        'ascending_triangle': {
            'description': 'Flat top with rising bottom, bullish breakout expected',
            'direction': 'BUY',
            'sentiment_range': (0.3, 0.7)
        },
        'descending_triangle': {
            'description': 'Flat bottom with declining top, bearish breakdown expected',
            'direction': 'SELL',
            'sentiment_range': (-0.7, -0.3)
        },
        'breakout': {
            'description': 'Price breaks above resistance with volume',
            'direction': 'BUY',
            'sentiment_range': (0.5, 0.9)
        },
        'breakdown': {
            'description': 'Price breaks below support with volume',
            'direction': 'SELL',
            'sentiment_range': (-0.9, -0.5)
        },
        'bull_flag': {
            'description': 'Sharp rally followed by consolidation, continuation expected',
            'direction': 'BUY',
            'sentiment_range': (0.4, 0.8)
        },
        'bear_flag': {
            'description': 'Sharp decline followed by consolidation, continuation expected',
            'direction': 'SELL',
            'sentiment_range': (-0.8, -0.4)
        },
        'consolidation': {
            'description': 'Price moving sideways in a range',
            'direction': 'HOLD',
            'sentiment_range': (-0.2, 0.2)
        },
        'trend_up': {
            'description': 'Sustained upward price movement',
            'direction': 'BUY',
            'sentiment_range': (0.3, 0.7)
        },
        'trend_down': {
            'description': 'Sustained downward price movement',
            'direction': 'SELL',
            'sentiment_range': (-0.7, -0.3)
        }
    }
    
    # News narrative templates by pattern
    NARRATIVE_TEMPLATES = {
        'double_top': [
            "{symbol} fails to break above {resistance} for the second time, signaling potential reversal.",
            "Double top formation confirmed for {symbol} as selling pressure mounts at {resistance}.",
            "Resistance at {resistance} proves too strong for {symbol}, technical breakdown likely.",
        ],
        'double_bottom': [
            "{symbol} finds strong support at {support} for the second time, bullish reversal forming.",
            "Double bottom pattern emerges for {symbol} as buyers defend {support} level.",
            "{symbol} bounces off {support} twice, signaling potential trend change.",
        ],
        'head_shoulders': [
            "{symbol} completes head and shoulders pattern, targeting {target} on breakdown.",
            "Classic head and shoulders top visible on {symbol} charts, sellers in control.",
            "{symbol} right shoulder forming; failure to hold {support} confirms bearish pattern.",
        ],
        'inverse_head_shoulders': [
            "{symbol} forms inverse head and shoulders, breakout above {resistance} targets {target}.",
            "Bullish reversal pattern confirmed for {symbol} with inverse head and shoulders completion.",
            "{symbol} breaks neckline of inverse head and shoulders, strong upside expected.",
        ],
        'ascending_triangle': [
            "{symbol} coils in ascending triangle pattern, breakout above {resistance} imminent.",
            "Higher lows pattern for {symbol} suggests accumulation ahead of {resistance} test.",
            "Ascending triangle on {symbol} points to {target} target on confirmed breakout.",
        ],
        'descending_triangle': [
            "{symbol} trapped in descending triangle, breakdown below {support} targets {target}.",
            "Lower highs pattern for {symbol} indicates distribution ahead of {support} test.",
            "Descending triangle breakdown likely for {symbol} as bears control price action.",
        ],
        'breakout': [
            "{symbol} breaks out above {resistance} on heavy volume, momentum building.",
            "Major resistance cleared for {symbol} at {resistance}, new uptrend confirmed.",
            "{symbol} breakout signals institutional accumulation; {target} next target.",
        ],
        'breakdown': [
            "{symbol} breaks below critical support at {support}, accelerating lower.",
            "Support failure at {support} for {symbol} triggers stop-loss selling.",
            "{symbol} breaks down on volume; {target} becomes next downside target.",
        ],
        'bull_flag': [
            "{symbol} consolidates in bull flag after sharp rally, continuation higher expected.",
            "Bullish flag pattern for {symbol} suggests pause before next leg up to {target}.",
            "{symbol} flag formation near highs indicates healthy consolidation.",
        ],
        'bear_flag': [
            "{symbol} forms bear flag after steep decline, further downside to {target} likely.",
            "Bearish consolidation pattern for {symbol} signals continuation of downtrend.",
            "{symbol} bear flag breakdown targets {target} as selling pressure resumes.",
        ],
        'consolidation': [
            "{symbol} trades in tight range between {support} and {resistance}, awaiting catalyst.",
            "Consolidation continues for {symbol} as market digests recent moves.",
            "{symbol} range-bound; breakout above {resistance} or below {support} will set direction.",
        ],
        'trend_up': [
            "{symbol} maintains uptrend with series of higher highs and higher lows.",
            "Bullish momentum persists for {symbol}, dips being bought aggressively.",
            "{symbol} trend remains up; support at {support} holds on pullbacks.",
        ],
        'trend_down': [
            "{symbol} downtrend intact with lower highs and lower lows pattern.",
            "Bearish momentum continues for {symbol}, rallies sold into at {resistance}.",
            "{symbol} remains in downtrend; resistance at {resistance} caps recovery attempts.",
        ]
    }
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            random_seed: Optional seed for reproducible generation
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def generate_chart_text_pairs(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        count: int = 100
    ) -> list[ChartTextPair]:
        """
        Generate chart-text pairs from price data.
        
        Detects patterns in price data and generates matching news narratives
        that would explain the pattern.
        
        Args:
            symbol: Stock symbol
            prices_df: DataFrame with OHLCV data
            count: Number of pairs to generate
            
        Returns:
            List of ChartTextPair objects
        """
        if len(prices_df) < 30:
            raise ValueError("Price data must have at least 30 periods for pattern detection")
        
        pairs = []
        patterns = list(self.PATTERNS.keys())
        
        for _ in range(count):
            # Randomly select a pattern
            pattern_name = random.choice(patterns)
            pattern_info = self.PATTERNS[pattern_name]
            
            # Generate pattern features
            features = self._generate_pattern_features(prices_df, pattern_name)
            
            # Generate narrative
            narrative = self._generate_narrative(symbol, pattern_name, features)
            
            # Generate sentiment within pattern's range
            sentiment = random.uniform(*pattern_info['sentiment_range'])
            
            pairs.append(ChartTextPair(
                price_pattern=pattern_name,
                pattern_features=features,
                news_narrative=narrative,
                sentiment=sentiment,
                expected_direction=pattern_info['direction']
            ))
        
        return pairs
    
    def _generate_pattern_features(self, df: pd.DataFrame, pattern: str) -> dict:
        """Generate realistic features for a pattern."""
        # Get price statistics
        close = df['Close'].iloc[-20:]
        high = df['High'].iloc[-20:]
        low = df['Low'].iloc[-20:]
        volume = df['Volume'].iloc[-20:]
        
        current_price = close.iloc[-1]
        price_range = high.max() - low.min()
        
        features = {
            'current_price': round(current_price, 2),
            'periods': 20,
            'volatility': round(price_range / current_price * 100, 2),
            'volume_trend': round(volume.iloc[-5:].mean() / volume.iloc[:5].mean() - 1, 2),
        }
        
        # Pattern-specific features
        if pattern in ['double_top', 'head_shoulders', 'ascending_triangle', 'breakout']:
            features['resistance'] = round(high.max(), 2)
            features['target'] = round(current_price + price_range * 0.5, 2)
        
        if pattern in ['double_bottom', 'inverse_head_shoulders', 'descending_triangle', 'breakdown']:
            features['support'] = round(low.min(), 2)
            features['target'] = round(current_price - price_range * 0.5, 2)
        
        if pattern in ['consolidation']:
            features['support'] = round(low.min(), 2)
            features['resistance'] = round(high.max(), 2)
        
        return features
    
    def _generate_narrative(self, symbol: str, pattern: str, features: dict) -> str:
        """Generate a news narrative for a pattern."""
        templates = self.NARRATIVE_TEMPLATES.get(pattern, ["{symbol} shows interesting price action."])
        template = random.choice(templates)
        
        # Fill in template variables
        params = {'symbol': symbol}
        params.update(features)
        
        try:
            return template.format(**params)
        except KeyError:
            # Fallback if some features missing
            return template.replace('{symbol}', symbol)
    
    def generate_report_summary_pairs(self, count: int = 50) -> list[ReportSummaryPair]:
        """
        Generate synthetic earnings report and summary pairs.
        
        Args:
            count: Number of pairs to generate
            
        Returns:
            List of ReportSummaryPair objects
        """
        pairs = []
        
        companies = [
            ('AAPL', 'Apple Inc.', 'Technology'),
            ('MSFT', 'Microsoft Corp.', 'Technology'),
            ('GOOGL', 'Alphabet Inc.', 'Technology'),
            ('AMZN', 'Amazon.com Inc.', 'Consumer'),
            ('TSLA', 'Tesla Inc.', 'Automotive'),
            ('JPM', 'JPMorgan Chase', 'Financials'),
            ('JNJ', 'Johnson & Johnson', 'Healthcare'),
            ('V', 'Visa Inc.', 'Financials'),
            ('NVDA', 'NVIDIA Corp.', 'Technology'),
            ('XOM', 'Exxon Mobil', 'Energy'),
        ]
        
        quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
        
        for _ in range(count):
            symbol, company_name, sector = random.choice(companies)
            quarter = random.choice(quarters)
            
            # Generate financial metrics
            actual_eps = round(random.uniform(0.5, 5.0), 2)
            surprise_pct = round(random.uniform(-15, 20), 1)
            expected_eps = round(actual_eps / (1 + surprise_pct/100), 2)
            
            revenue = round(random.uniform(5, 100), 1)
            revenue_growth = round(random.uniform(-10, 30), 1)
            
            # Determine sentiment based on surprise
            if surprise_pct > 5:
                sentiment = random.uniform(0.4, 0.9)
                tone = 'positive'
            elif surprise_pct < -5:
                sentiment = random.uniform(-0.9, -0.4)
                tone = 'negative'
            else:
                sentiment = random.uniform(-0.2, 0.2)
                tone = 'neutral'
            
            # Generate full report
            full_report = self._generate_full_report(
                symbol, company_name, sector, quarter,
                actual_eps, expected_eps, surprise_pct,
                revenue, revenue_growth, tone
            )
            
            # Generate summary
            summary = self._generate_summary(
                symbol, company_name, quarter,
                actual_eps, surprise_pct, revenue, revenue_growth, tone
            )
            
            key_metrics = {
                'eps_actual': actual_eps,
                'eps_expected': expected_eps,
                'surprise_pct': surprise_pct,
                'revenue_billions': revenue,
                'revenue_growth_pct': revenue_growth,
                'sector': sector
            }
            
            pairs.append(ReportSummaryPair(
                full_report=full_report,
                summary=summary,
                key_metrics=key_metrics,
                sentiment=sentiment
            ))
        
        return pairs
    
    def _generate_full_report(
        self, symbol: str, company_name: str, sector: str, quarter: str,
        actual_eps: float, expected_eps: float, surprise_pct: float,
        revenue: float, revenue_growth: float, tone: str
    ) -> str:
        """Generate a full earnings report."""
        segments = [
            f"{company_name} ({symbol}) reported {quarter} earnings results.",
            f"Earnings per share came in at ${actual_eps:.2f} ",
        ]
        
        if surprise_pct > 0:
            segments[-1] += f"vs. ${expected_eps:.2f} expected, beating estimates by {surprise_pct:.1f}%."
        else:
            segments[-1] += f"vs. ${expected_eps:.2f} expected, missing estimates by {abs(surprise_pct):.1f}%."
        
        segments.append(f" Revenue totaled ${revenue:.1f}B, representing {revenue_growth:.1f}% year-over-year growth.")
        
        if tone == 'positive':
            segments.append(
                f" Gross margin expanded to {random.uniform(35, 55):.1f}% while operating margin improved to {random.uniform(15, 30):.1f}%. "
                f"Management raised full-year guidance citing strong demand trends. "
                f"The company announced a ${random.randint(10, 50)} billion share buyback program."
            )
        elif tone == 'negative':
            segments.append(
                f" Gross margin contracted to {random.uniform(25, 40):.1f}% while operating margin declined to {random.uniform(8, 18):.1f}%. "
                f"Management lowered full-year guidance citing macro headwinds. "
                f"The company announced cost reduction initiatives to improve profitability."
            )
        else:
            segments.append(
                f" Gross margin remained stable at {random.uniform(30, 45):.1f}% with operating margin at {random.uniform(12, 25):.1f}%. "
                f"Management maintained full-year guidance. "
                f"The company continues to invest in growth initiatives while maintaining capital discipline."
            )
        
        segments.append(
            f" Segment performance: {sector} operations showed {random.choice(['strength', 'mixed results', 'challenges'])}. "
            f"Free cash flow was ${random.uniform(2, 20):.1f}B. "
            f"The balance sheet remains {random.choice(['strong', 'solid', 'adequate'])} with ${random.uniform(10, 200):.1f}B in cash."
        )
        
        return "".join(segments)
    
    def _generate_summary(
        self, symbol: str, company_name: str, quarter: str,
        actual_eps: float, surprise_pct: float, revenue: float,
        revenue_growth: float, tone: str
    ) -> str:
        """Generate an executive summary."""
        if tone == 'positive':
            return (
                f"{company_name} ({symbol}) delivered strong {quarter} results with EPS of ${actual_eps:.2f}, "
                f"beating estimates by {surprise_pct:.1f}%. Revenue grew {revenue_growth:.1f}% to ${revenue:.1f}B. "
                f"Margins expanded and guidance was raised. Positive outlook supported by robust demand."
            )
        elif tone == 'negative':
            return (
                f"{company_name} ({symbol}) reported disappointing {quarter} results with EPS of ${actual_eps:.2f}, "
                f"missing estimates by {abs(surprise_pct):.1f}%. Revenue growth of {revenue_growth:.1f}% to ${revenue:.1f}B was below expectations. "
                f"Margins contracted and guidance was lowered. Challenging environment cited by management."
            )
        else:
            return (
                f"{company_name} ({symbol}) reported {quarter} EPS of ${actual_eps:.2f}, in line with estimates. "
                f"Revenue of ${revenue:.1f}B grew {revenue_growth:.1f}% year-over-year. "
                f"Results were mixed with stable margins. Guidance maintained as company navigates uncertain environment."
            )
    
    def generate_labeled_training_data(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        lookforward: int = 5
    ) -> pd.DataFrame:
        """
        Create labeled training dataset from price data.
        
        Creates features (technical indicators) and labels (future return direction).
        Labels: 1 (price up >1%), -1 (price down >1%), 0 (flat)
        
        Args:
            symbol: Stock symbol
            prices_df: DataFrame with OHLCV data
            lookforward: Number of periods to look forward for label
            
        Returns:
            DataFrame with features and labels
        """
        if len(prices_df) < lookforward + 20:
            raise ValueError(f"Price data must have at least {lookforward + 20} periods")
        
        df = prices_df.copy()
        
        # Calculate technical indicators
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_atr(df)
        df = self._calculate_obv(df)
        
        # Calculate momentum features
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['volatility_14d'] = df['returns_1d'].rolling(14).std() * np.sqrt(252)
        df['volume_change'] = df['Volume'].pct_change()
        
        # Calculate future return and label
        df['future_return'] = df['Close'].shift(-lookforward) / df['Close'] - 1
        df['label'] = df['future_return'].apply(self._classify_return)
        
        # Select feature columns
        feature_cols = [
            'RSI_14', 'MACD', 'MACD_signal', 'BB_position',
            'ATR_14', 'OBV', 'returns_1d', 'returns_5d',
            'volatility_14d', 'volume_change'
        ]
        
        # Create result dataframe
        result = df[feature_cols + ['label', 'future_return']].copy()
        result = result.dropna()
        result['symbol'] = symbol
        
        return result
    
    def _classify_return(self, ret: float) -> int:
        """Classify return into label."""
        if pd.isna(ret):
            return 0
        if ret > 0.01:
            return 1
        elif ret < -0.01:
            return -1
        else:
            return 0
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator."""
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        df['BB_upper'] = sma + (std * 2)
        df['BB_lower'] = sma - (std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR_14'] = true_range.rolling(period).mean()
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        return df
