"""Time Series Encoder for OHLCV Feature Extraction.

This module provides comprehensive technical indicator extraction from
OHLCV (Open, High, Low, Close, Volume) price data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesFeatures:
    """Container for extracted time series features.

    Attributes
    ----------
    features : np.ndarray
        2D feature matrix where each row is a timestep and columns are features.
    feature_names : List[str]
        Names of each feature column in order.
    timestamp : datetime
        Timestamp associated with the most recent data point.
    metadata : dict
        Additional metadata about feature extraction.
    """

    features: np.ndarray
    feature_names: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesEncoder:
    """Extracts technical indicators and features from OHLCV price data.

    This encoder computes a comprehensive set of technical indicators including
    price features, moving averages, momentum indicators, volatility measures,
    volume features, and pattern detection.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with the following options:
        - indicator_windows: List[int] - Windows for moving averages (default [5, 10, 20, 50])
        - normalize: bool - Whether to apply z-score normalization (default True)

    Example
    -------
    >>> encoder = TimeSeriesEncoder(config={"indicator_windows": [5, 10, 20]})
    >>> features = encoder.encode(prices_df)
    >>> print(features.features.shape)  # (timesteps, ~25-30 features)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TimeSeriesEncoder.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. See class docstring for options.
        """
        self.config = config or {}
        self.indicator_windows = self.config.get("indicator_windows", [5, 10, 20, 50])
        self.normalize = self.config.get("normalize", True)

        # Build feature names list
        self._feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        """Build the list of feature names based on configuration."""
        names = [
            # Price features
            "returns",
            "log_returns",
            "high_low_range",
            "close_to_open",
        ]

        # Moving averages for each window
        for w in self.indicator_windows:
            names.extend([f"sma_{w}", f"ema_{w}"])

        # Momentum indicators for each window
        for w in self.indicator_windows:
            names.append(f"roc_{w}")
        names.append("rsi_14")

        # Volatility for each window
        for w in self.indicator_windows:
            names.append(f"rolling_std_{w}")
        names.extend(["atr_14", "bb_position"])

        # Volume features
        names.extend(["volume_sma_ratio", "obv_slope"])

        # Pattern features
        names.extend(["higher_highs", "lower_lows", "trend_strength"])

        return names

    def encode(self, prices_df: pd.DataFrame) -> TimeSeriesFeatures:
        """Extract features from OHLCV price data.

        Parameters
        ----------
        prices_df : pd.DataFrame
            DataFrame with columns [open, high, low, close, volume] (case-insensitive).
            May optionally include 'timestamp' column.

        Returns
        -------
        TimeSeriesFeatures
            Container with extracted features, feature names, timestamp, and metadata.

        Raises
        ------
        ValueError
            If required columns are missing from the input DataFrame.
        """
        # Normalize column names
        df = self._normalize_columns(prices_df.copy())

        # Validate required columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Extract arrays
        open_prices = df["open"].values.astype(float)
        high_prices = df["high"].values.astype(float)
        low_prices = df["low"].values.astype(float)
        close_prices = df["close"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        n = len(close_prices)
        max_window = max(self.indicator_windows)

        if n < max_window + 1:
            # Return empty features if insufficient data
            return TimeSeriesFeatures(
                features=np.array([]).reshape(0, len(self._feature_names)),
                feature_names=self._feature_names,
                timestamp=self._get_timestamp(df),
                metadata={"warning": f"Insufficient data: {n} rows, need {max_window + 1}"},
            )

        # Initialize feature matrix
        features = np.zeros((n, len(self._feature_names)))
        col_idx = 0

        # 1. Price features
        returns = np.zeros(n)
        returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
        features[:, col_idx] = returns
        col_idx += 1

        log_returns = np.zeros(n)
        log_returns[1:] = np.log(close_prices[1:] / close_prices[:-1])
        features[:, col_idx] = log_returns
        col_idx += 1

        high_low_range = (high_prices - low_prices) / close_prices
        features[:, col_idx] = high_low_range
        col_idx += 1

        close_to_open = (close_prices - open_prices) / open_prices
        features[:, col_idx] = close_to_open
        col_idx += 1

        # 2. Moving averages for each window
        for w in self.indicator_windows:
            sma = self._sma(close_prices, w)
            features[:, col_idx] = sma
            col_idx += 1

            ema = self._ema(close_prices, w)
            features[:, col_idx] = ema
            col_idx += 1

        # 3. Momentum indicators
        for w in self.indicator_windows:
            roc = self._roc(close_prices, w)
            features[:, col_idx] = roc
            col_idx += 1

        rsi = self._rsi(close_prices, 14)
        features[:, col_idx] = rsi
        col_idx += 1

        # 4. Volatility
        for w in self.indicator_windows:
            rolling_std = self._rolling_std(returns, w)
            features[:, col_idx] = rolling_std
            col_idx += 1

        atr = self._atr(high_prices, low_prices, close_prices, 14)
        features[:, col_idx] = atr
        col_idx += 1

        bb_position = self._bollinger_position(close_prices, 20)
        features[:, col_idx] = bb_position
        col_idx += 1

        # 5. Volume features
        volume_sma = self._sma(volumes, 20)
        volume_sma_ratio = np.where(volume_sma > 0, volumes / volume_sma, 1.0)
        features[:, col_idx] = volume_sma_ratio
        col_idx += 1

        obv_slope = self._obv_slope(close_prices, volumes, 5)
        features[:, col_idx] = obv_slope
        col_idx += 1

        # 6. Pattern features
        higher_highs = self._higher_highs(high_prices, 5)
        features[:, col_idx] = higher_highs
        col_idx += 1

        lower_lows = self._lower_lows(low_prices, 5)
        features[:, col_idx] = lower_lows
        col_idx += 1

        trend_strength = self._trend_strength(close_prices, 20)
        features[:, col_idx] = trend_strength
        col_idx += 1

        # Apply normalization if configured
        if self.normalize:
            features = self.normalize_features(features)

        # Build metadata
        metadata = {
            "num_timesteps": n,
            "indicator_windows": self.indicator_windows,
            "normalized": self.normalize,
            "feature_count": len(self._feature_names),
        }

        return TimeSeriesFeatures(
            features=features,
            feature_names=self._feature_names,
            timestamp=self._get_timestamp(df),
            metadata=metadata,
        )

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization per feature column.

        Parameters
        ----------
        features : np.ndarray
            2D feature matrix to normalize.

        Returns
        -------
        np.ndarray
            Normalized feature matrix with zero mean and unit variance per column.
        """
        if features.size == 0:
            return features

        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)

        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)

        return (features - mean) / std

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df.columns = [col.lower().strip() for col in df.columns]
        return df

    def _get_timestamp(self, df: pd.DataFrame) -> datetime:
        """Extract timestamp from DataFrame or return current time."""
        if "timestamp" in df.columns:
            try:
                ts = df["timestamp"].iloc[-1]
                if isinstance(ts, datetime):
                    return ts
                elif isinstance(ts, str):
                    return pd.to_datetime(ts).to_pydatetime()
            except (ValueError, TypeError):
                pass
        return datetime.now()

    def _sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        result = np.zeros_like(prices)
        if len(prices) < window:
            return result

        cumsum = np.cumsum(prices, dtype=float)
        cumsum = np.insert(cumsum, 0, 0)
        result[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
        result[: window - 1] = np.nan
        return np.nan_to_num(result, nan=0.0)

    def _ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        result = np.zeros_like(prices)
        if len(prices) < 1:
            return result

        alpha = 2.0 / (window + 1)
        result[0] = prices[0]
        for i in range(1, len(prices)):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
        return result

    def _roc(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Rate of Change."""
        result = np.zeros_like(prices)
        if len(prices) <= window:
            return result

        result[window:] = (prices[window:] - prices[:-window]) / prices[:-window] * 100
        return result

    def _rsi(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Relative Strength Index."""
        result = np.zeros_like(prices)
        if len(prices) < window + 1:
            return result

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))

        # Initial averages
        avg_gain[window] = np.mean(gains[:window])
        avg_loss[window] = np.mean(losses[:window])

        # Smoothed averages
        for i in range(window + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gains[i - 1]) / window
            avg_loss[i] = (avg_loss[i - 1] * (window - 1) + losses[i - 1]) / window

        # RSI calculation
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        result = 100 - (100 / (1 + rs))

        return result / 100  # Normalize to [0, 1]

    def _rolling_std(self, values: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        result = np.zeros_like(values)
        if len(values) < window:
            return result

        for i in range(window - 1, len(values)):
            result[i] = np.std(values[i - window + 1 : i + 1])
        return result

    def _atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Calculate Average True Range."""
        result = np.zeros_like(close)
        if len(close) < window + 1:
            return result

        # True Range calculation
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Average True Range
        result[window - 1] = np.mean(tr[:window])
        for i in range(window, len(close)):
            result[i] = (result[i - 1] * (window - 1) + tr[i]) / window

        # Normalize by close price
        return result / close

    def _bollinger_position(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate position within Bollinger Bands.

        Returns a value between -1 (at lower band) and 1 (at upper band).
        """
        result = np.zeros_like(prices)
        if len(prices) < window:
            return result

        sma = self._sma(prices, window)
        std = self._rolling_std(prices, window)

        # Position within bands
        upper = sma + 2 * std
        lower = sma - 2 * std
        band_width = upper - lower

        result = np.where(band_width > 0, (prices - lower) / band_width * 2 - 1, 0)
        return np.clip(result, -1, 1)

    def _obv_slope(
        self, close: np.ndarray, volume: np.ndarray, window: int
    ) -> np.ndarray:
        """Calculate On-Balance Volume slope."""
        result = np.zeros_like(close)
        if len(close) < window + 1:
            return result

        # Calculate OBV
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # Calculate slope
        for i in range(window, len(close)):
            x = np.arange(window)
            y = obv[i - window + 1 : i + 1]
            if np.std(y) > 0:
                result[i] = np.polyfit(x, y, 1)[0] / np.mean(np.abs(y))
            else:
                result[i] = 0

        return result

    def _higher_highs(self, high: np.ndarray, window: int) -> np.ndarray:
        """Detect higher highs pattern.

        Returns 1 if higher high detected, 0 otherwise.
        """
        result = np.zeros_like(high)
        if len(high) < window + 1:
            return result

        for i in range(window, len(high)):
            recent = high[i - window + 1 : i + 1]
            if all(recent[j] < recent[j + 1] for j in range(len(recent) - 1)):
                result[i] = 1
        return result

    def _lower_lows(self, low: np.ndarray, window: int) -> np.ndarray:
        """Detect lower lows pattern.

        Returns 1 if lower low detected, 0 otherwise.
        """
        result = np.zeros_like(low)
        if len(low) < window + 1:
            return result

        for i in range(window, len(low)):
            recent = low[i - window + 1 : i + 1]
            if all(recent[j] > recent[j + 1] for j in range(len(recent) - 1)):
                result[i] = 1
        return result

    def _trend_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend strength using linear regression slope.

        Returns a value between -1 (strong downtrend) and 1 (strong uptrend).
        """
        result = np.zeros_like(prices)
        if len(prices) < window:
            return result

        for i in range(window - 1, len(prices)):
            x = np.arange(window)
            y = prices[i - window + 1 : i + 1]

            # Linear regression
            slope, _ = np.polyfit(x, y, 1)

            # Normalize by price magnitude
            if np.mean(y) > 0:
                result[i] = np.clip(slope / np.mean(y) * 100, -1, 1)

        return result
