"""
Feature Engineering Module

Provides feature extraction pipeline for research-grade financial data analysis.
Includes technical indicators, statistical features, cross-asset features, and sentiment features.
"""

from typing import Optional
import warnings

import pandas as pd
import numpy as np

from .data_sources import NewsItem


class FeatureEngineer:
    """
    Feature extraction pipeline for financial data.
    
    Computes technical indicators, statistical features, cross-asset features,
    and sentiment features from market data and news items.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators from OHLCV data.
        
        Adds columns: RSI_14, MACD, MACD_signal, BB_upper, BB_lower, 
        BB_position, ATR_14, OBV
        
        All computed using pure pandas/numpy (no TA-Lib dependency).
        
        Args:
            df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with added technical indicator columns
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < 26:
            raise ValueError("DataFrame must have at least 26 rows for technical indicator calculation")
        
        result = df.copy()
        
        # RSI (Relative Strength Index)
        result = self._add_rsi(result, period=14)
        
        # MACD (Moving Average Convergence Divergence)
        result = self._add_macd(result)
        
        # Bollinger Bands
        result = self._add_bollinger_bands(result, period=20)
        
        # ATR (Average True Range)
        result = self._add_atr(result, period=14)
        
        # OBV (On-Balance Volume)
        result = self._add_obv(result)
        
        return result
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator to DataFrame."""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator to DataFrame."""
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands to DataFrame."""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        df['BB_upper'] = sma + (std * 2)
        df['BB_lower'] = sma - (std * 2)
        # Position within bands (0 = lower, 1 = upper, 0.5 = middle)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add ATR indicator to DataFrame."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = true_range.rolling(window=period).mean()
        return df
    
    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add OBV indicator to DataFrame."""
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
    
    def compute_statistical_features(
        self,
        df: pd.DataFrame,
        windows: list[int] = None
    ) -> pd.DataFrame:
        """
        Compute statistical features for price data.
        
        For each window: rolling_mean, rolling_std, rolling_skew, 
        rolling_kurtosis, z_score
        
        Args:
            df: DataFrame with price data
            windows: List of window sizes. Defaults to [5, 10, 20, 60]
            
        Returns:
            DataFrame with added statistical feature columns
        """
        if windows is None:
            windows = [5, 10, 20, 60]
        
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")
        
        result = df.copy()
        
        for window in windows:
            if len(df) < window:
                warnings.warn(f"DataFrame has fewer rows ({len(df)}) than window size ({window})")
                continue
            
            prefix = f"{window}d"
            rolling = df['Close'].rolling(window=window)
            
            result[f'{prefix}_mean'] = rolling.mean()
            result[f'{prefix}_std'] = rolling.std()
            result[f'{prefix}_skew'] = rolling.skew()
            result[f'{prefix}_kurt'] = rolling.kurt()
            
            # Z-score: how many std devs current price is from rolling mean
            result[f'{prefix}_zscore'] = (df['Close'] - rolling.mean()) / rolling.std()
        
        return result
    
    def compute_cross_asset_features(
        self,
        dfs: dict[str, pd.DataFrame],
        window: int = 20
    ) -> pd.DataFrame:
        """
        Compute cross-asset features from multiple symbol DataFrames.
        
        Calculates pairwise correlations, relative strength, and beta to market.
        
        Args:
            dfs: Dictionary mapping symbol names to DataFrames with 'Close' column
            window: Rolling window for correlation and beta calculation
            
        Returns:
            DataFrame with cross-asset features indexed by date
        """
        if len(dfs) < 2:
            raise ValueError("At least 2 symbols required for cross-asset features")
        
        # Extract close prices
        prices = {}
        for symbol, df in dfs.items():
            if 'Close' not in df.columns:
                raise ValueError(f"DataFrame for {symbol} missing 'Close' column")
            prices[symbol] = df['Close']
        
        # Create aligned price DataFrame
        price_df = pd.DataFrame(prices)
        
        # Calculate returns
        returns = price_df.pct_change().dropna()
        
        # Use first symbol as market proxy if SPY not present
        market_symbol = 'SPY' if 'SPY' in returns.columns else returns.columns[0]
        market_returns = returns[market_symbol]
        
        result_data = {}
        
        # Calculate pairwise correlations
        symbols = list(returns.columns)
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr_col = f'corr_{sym1}_{sym2}'
                result_data[corr_col] = returns[sym1].rolling(window).corr(returns[sym2])
        
        # Calculate relative strength (price ratio normalized)
        for sym in symbols:
            if sym != market_symbol:
                rs_col = f'rs_{sym}_vs_{market_symbol}'
                result_data[rs_col] = (price_df[sym] / price_df[market_symbol]).rolling(window).mean()
        
        # Calculate beta to market
        for sym in symbols:
            if sym != market_symbol:
                beta_col = f'beta_{sym}'
                # Beta = Cov(sym, market) / Var(market)
                cov = returns[sym].rolling(window).cov(market_returns)
                var = market_returns.rolling(window).var()
                result_data[beta_col] = cov / var
        
        result = pd.DataFrame(result_data, index=price_df.index)
        return result
    
    def compute_sentiment_features(
        self,
        news_items: list[NewsItem],
        window_days: int = 5
    ) -> pd.DataFrame:
        """
        Compute sentiment features from news items.
        
        Calculates rolling average sentiment, sentiment momentum, news volume,
        and sentiment dispersion.
        
        Args:
            news_items: List of NewsItem objects
            window_days: Rolling window in days for sentiment calculations
            
        Returns:
            DataFrame with sentiment features indexed by date
        """
        if not news_items:
            raise ValueError("news_items list cannot be empty")
        
        # Convert to DataFrame
        news_df = pd.DataFrame([
            {
                'date': item.timestamp.date(),
                'sentiment': item.sentiment_label,
                'event_type': item.event_type
            }
            for item in news_items
        ])
        
        # Group by date
        daily = news_df.groupby('date').agg({
            'sentiment': ['mean', 'std', 'count'],
            'event_type': 'count'
        })
        daily.columns = ['sentiment_mean', 'sentiment_std', 'news_count', 'event_count']
        daily = daily.fillna(0)
        
        # Create date range index
        date_range = pd.date_range(
            start=daily.index.min(),
            end=daily.index.max(),
            freq='D'
        )
        
        # Reindex to fill missing dates
        daily = daily.reindex(date_range.date, fill_value=0)
        daily.index = pd.to_datetime(daily.index)
        
        # Calculate rolling features
        result = pd.DataFrame(index=daily.index)
        
        # Rolling average sentiment
        result['sentiment_avg'] = daily['sentiment_mean'].rolling(window=window_days, min_periods=1).mean()
        
        # Sentiment momentum (change in sentiment)
        result['sentiment_momentum'] = result['sentiment_avg'].diff(window_days)
        
        # News volume
        result['news_volume'] = daily['news_count'].rolling(window=window_days, min_periods=1).sum()
        
        # Sentiment dispersion (std of sentiment)
        result['sentiment_dispersion'] = daily['sentiment_mean'].rolling(window=window_days, min_periods=1).std()
        
        # Sentiment trend (positive/negative ratio)
        positive = (daily['sentiment_mean'] > 0).rolling(window=window_days, min_periods=1).sum()
        negative = (daily['sentiment_mean'] < 0).rolling(window=window_days, min_periods=1).sum()
        result['sentiment_ratio'] = positive / (negative + 1)  # +1 to avoid division by zero
        
        return result
    
    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        news_items: Optional[list[NewsItem]] = None,
        other_symbols_dfs: Optional[dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Combine all feature types into a single feature matrix.
        
        Args:
            df: Primary DataFrame with OHLCV data
            news_items: Optional list of NewsItem objects for sentiment features
            other_symbols_dfs: Optional dict of other symbol DataFrames for cross-asset features
            
        Returns:
            Combined feature matrix DataFrame
        """
        # Start with technical indicators
        result = self.compute_technical_indicators(df)
        
        # Add statistical features
        result = self.compute_statistical_features(result)
        
        # Add sentiment features if provided
        if news_items is not None:
            sentiment_df = self.compute_sentiment_features(news_items)
            # Align indices
            sentiment_df.index = pd.to_datetime(sentiment_df.index)
            result.index = pd.to_datetime(result.index)
            result = result.join(sentiment_df, how='left')
            result[sentiment_df.columns] = result[sentiment_df.columns].fillna(method='ffill')
        
        # Add cross-asset features if provided
        if other_symbols_dfs is not None:
            # Include primary symbol for cross-asset calculation
            all_symbols = {**other_symbols_dfs}
            # Find symbol name from df if possible, otherwise use 'PRIMARY'
            primary_symbol = 'PRIMARY'
            cross_asset_df = self.compute_cross_asset_features(all_symbols, window=20)
            # Align indices
            cross_asset_df.index = pd.to_datetime(cross_asset_df.index)
            result = result.join(cross_asset_df, how='left')
            result[cross_asset_df.columns] = result[cross_asset_df.columns].fillna(method='ffill')
        
        return result
