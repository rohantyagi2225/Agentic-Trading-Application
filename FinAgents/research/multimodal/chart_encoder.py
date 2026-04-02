"""Chart Encoder for Chart Pattern Detection.

This module provides pattern detection, support/resistance level identification,
and volume profile analysis from OHLCV price data using numerical methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PatternDetection:
    """Container for a detected chart pattern.

    Attributes
    ----------
    pattern_name : str
        Name of the detected pattern (e.g., 'double_top', 'breakout_up').
    confidence : float
        Confidence level of the detection in range [0, 1].
    start_idx : int
        Starting index of the pattern in the price array.
    end_idx : int
        Ending index of the pattern in the price array.
    direction : str
        Direction of expected move: 'bullish', 'bearish', or 'neutral'.
    expected_move_pct : float
        Expected percentage move magnitude if pattern completes.
    """

    pattern_name: str
    confidence: float
    start_idx: int
    end_idx: int
    direction: str
    expected_move_pct: float


@dataclass
class ChartFeatures:
    """Container for chart pattern features.

    Attributes
    ----------
    features : np.ndarray
        Chart pattern feature vector (~10 features).
    detected_patterns : List[PatternDetection]
        List of detected chart patterns.
    support_levels : List[float]
        Detected support price levels.
    resistance_levels : List[float]
        Detected resistance price levels.
    metadata : dict
        Additional metadata about feature extraction.
    """

    features: np.ndarray
    detected_patterns: List[PatternDetection] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChartEncoder:
    """Encodes chart patterns and technical levels from OHLCV price data.

    This encoder detects various chart patterns, identifies support/resistance
    levels, and computes volume profile features using numerical methods.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with the following options:
        - price_tolerance: float - Tolerance for price matching (default 0.02)
        - volume_threshold: float - Volume surge threshold (default 1.5)
        - support_resistance_window: int - Window for S/R detection (default 20)
        - num_support_levels: int - Number of S/R levels to detect (default 3)

    Example
    -------
    >>> encoder = ChartEncoder(config={"price_tolerance": 0.03})
    >>> features = encoder.encode(prices_df)
    >>> print(features.detected_patterns)  # List of PatternDetection
    """

    # Expected move percentages for patterns
    PATTERN_EXPECTED_MOVES = {
        "double_top": 0.05,
        "double_bottom": 0.05,
        "head_and_shoulders": 0.07,
        "inverse_head_and_shoulders": 0.07,
        "breakout_up": 0.04,
        "breakout_down": 0.04,
        "ascending_triangle": 0.05,
        "descending_triangle": 0.05,
        "bull_flag": 0.03,
        "bear_flag": 0.03,
        "channel_up": 0.02,
        "channel_down": 0.02,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ChartEncoder.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. See class docstring for options.
        """
        self.config = config or {}
        self.price_tolerance = self.config.get("price_tolerance", 0.02)
        self.volume_threshold = self.config.get("volume_threshold", 1.5)
        self.sr_window = self.config.get("support_resistance_window", 20)
        self.num_sr_levels = self.config.get("num_support_levels", 3)

    def encode(self, prices_df: pd.DataFrame) -> ChartFeatures:
        """Extract chart features from OHLCV price data.

        Parameters
        ----------
        prices_df : pd.DataFrame
            DataFrame with columns [open, high, low, close, volume] (case-insensitive).

        Returns
        -------
        ChartFeatures
            Container with features, detected patterns, and support/resistance levels.
        """
        # Normalize columns
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

        if n < 30:
            # Return minimal features if insufficient data
            return ChartFeatures(
                features=np.zeros(10),
                detected_patterns=[],
                support_levels=[],
                resistance_levels=[],
                metadata={"warning": f"Insufficient data: {n} rows, need at least 30"},
            )

        # Detect patterns
        patterns = self.detect_patterns(
            np.column_stack([open_prices, high_prices, low_prices, close_prices, volumes])
        )

        # Find support and resistance levels
        support_levels, resistance_levels = self.find_support_resistance(
            close_prices, window=self.sr_window, num_levels=self.num_sr_levels
        )

        # Compute volume profile
        volume_profile = self.encode_volume_profile(close_prices, volumes, num_bins=10)

        # Build feature vector
        features = self._build_features(
            close_prices=close_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            volumes=volumes,
            patterns=patterns,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            volume_profile=volume_profile,
        )

        metadata = {
            "num_patterns": len(patterns),
            "num_support": len(support_levels),
            "num_resistance": len(resistance_levels),
            "timestamp": self._get_timestamp(df),
        }

        return ChartFeatures(
            features=features,
            detected_patterns=patterns,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            metadata=metadata,
        )

    def detect_patterns(self, prices: np.ndarray) -> List[PatternDetection]:
        """Detect chart patterns from OHLCV price array.

        Parameters
        ----------
        prices : np.ndarray
            2D array with columns [open, high, low, close, volume].

        Returns
        -------
        List[PatternDetection]
            List of detected patterns with confidence and direction.
        """
        patterns = []
        n = len(prices)

        if n < 30:
            return patterns

        close = prices[:, 3]
        high = prices[:, 1]
        low = prices[:, 2]
        volume = prices[:, 4]

        # Detect various patterns
        patterns.extend(self._detect_double_top(close, high, low))
        patterns.extend(self._detect_double_bottom(close, high, low))
        patterns.extend(self._detect_head_and_shoulders(close, high, low))
        patterns.extend(self._detect_inverse_head_and_shoulders(close, high, low))
        patterns.extend(self._detect_breakout_up(close, high, low, volume))
        patterns.extend(self._detect_breakout_down(close, high, low, volume))
        patterns.extend(self._detect_ascending_triangle(close, high, low))
        patterns.extend(self._detect_descending_triangle(close, high, low))
        patterns.extend(self._detect_bull_flag(close, high, low))
        patterns.extend(self._detect_bear_flag(close, high, low))

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def find_support_resistance(
        self,
        prices: np.ndarray,
        window: int = 20,
        num_levels: int = 3,
    ) -> Tuple[List[float], List[float]]:
        """Find key support and resistance levels using local min/max clustering.

        Parameters
        ----------
        prices : np.ndarray
            1D array of close prices.
        window : int
            Window for finding local extrema.
        num_levels : int
            Number of levels to return for each.

        Returns
        -------
        Tuple[List[float], List[float]]
            Tuple of (support_levels, resistance_levels).
        """
        n = len(prices)
        if n < window * 2:
            return [], []

        # Find local minima (potential support) and maxima (potential resistance)
        local_mins = []
        local_maxs = []

        for i in range(window, n - window):
            # Check if local minimum
            if all(prices[i] <= prices[i - j] for j in range(1, window + 1)) and all(
                prices[i] <= prices[i + j] for j in range(1, window + 1)
            ):
                local_mins.append((i, prices[i]))

            # Check if local maximum
            if all(prices[i] >= prices[i - j] for j in range(1, window + 1)) and all(
                prices[i] >= prices[i + j] for j in range(1, window + 1)
            ):
                local_maxs.append((i, prices[i]))

        # Cluster similar levels
        support_levels = self._cluster_levels(local_mins, num_levels)
        resistance_levels = self._cluster_levels(local_maxs, num_levels)

        # Sort: support ascending, resistance descending
        support_levels.sort()
        resistance_levels.sort(reverse=True)

        # Filter: only include levels that haven't been broken
        current_price = prices[-1]
        support_levels = [s for s in support_levels if s < current_price]
        resistance_levels = [r for r in resistance_levels if r > current_price]

        return support_levels[:num_levels], resistance_levels[:num_levels]

    def encode_volume_profile(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        num_bins: int = 10,
    ) -> np.ndarray:
        """Compute volume profile across price levels.

        Parameters
        ----------
        prices : np.ndarray
            1D array of close prices.
        volumes : np.ndarray
            1D array of volume values.
        num_bins : int
            Number of price bins for the profile.

        Returns
        -------
        np.ndarray
            Normalized volume profile vector of length num_bins.
        """
        if len(prices) < num_bins:
            return np.zeros(num_bins)

        # Create price bins
        price_min, price_max = prices.min(), prices.max()
        if price_max == price_min:
            return np.ones(num_bins) / num_bins

        bin_edges = np.linspace(price_min, price_max, num_bins + 1)

        # Compute volume at each price level
        volume_profile = np.zeros(num_bins)
        for i in range(len(prices)):
            bin_idx = np.searchsorted(bin_edges[1:], prices[i])
            bin_idx = min(bin_idx, num_bins - 1)
            volume_profile[bin_idx] += volumes[i]

        # Normalize
        total_volume = volume_profile.sum()
        if total_volume > 0:
            volume_profile = volume_profile / total_volume

        return volume_profile

    def _build_features(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        volumes: np.ndarray,
        patterns: List[PatternDetection],
        support_levels: List[float],
        resistance_levels: List[float],
        volume_profile: np.ndarray,
    ) -> np.ndarray:
        """Build the chart feature vector."""
        current_price = close_prices[-1]

        # Count bullish and bearish patterns
        bullish_patterns = [p for p in patterns if p.direction == "bullish"]
        bearish_patterns = [p for p in patterns if p.direction == "bearish"]

        # Strongest pattern confidence
        max_confidence = max([p.confidence for p in patterns], default=0.0)

        # Distance to support and resistance
        distance_to_support = 0.0
        if support_levels:
            nearest_support = max(support_levels)
            distance_to_support = (current_price - nearest_support) / current_price

        distance_to_resistance = 0.0
        if resistance_levels:
            nearest_resistance = min(resistance_levels)
            distance_to_resistance = (nearest_resistance - current_price) / current_price

        # Volume profile skew (where is most volume concentrated)
        # Positive skew = volume at higher prices (selling pressure)
        # Negative skew = volume at lower prices (buying support)
        bin_weights = np.arange(len(volume_profile)) - len(volume_profile) / 2
        volume_skew = np.sum(volume_profile * bin_weights)

        # Trend angle (linear regression slope)
        x = np.arange(len(close_prices))
        slope, _ = np.polyfit(x, close_prices, 1)
        trend_angle = slope / np.mean(close_prices) * 100  # Percentage per bar
        trend_angle = np.clip(trend_angle, -1, 1)

        # Pattern agreement score (how many patterns agree in direction)
        if patterns:
            bullish_weight = sum(p.confidence for p in bullish_patterns)
            bearish_weight = sum(p.confidence for p in bearish_patterns)
            total_weight = bullish_weight + bearish_weight
            if total_weight > 0:
                agreement = abs(bullish_weight - bearish_weight) / total_weight
            else:
                agreement = 0.0
        else:
            agreement = 0.0

        features = np.array(
            [
                float(len(bullish_patterns)),
                float(len(bearish_patterns)),
                max_confidence,
                distance_to_support,
                distance_to_resistance,
                volume_skew,
                trend_angle,
                agreement,
                np.mean(volumes[-20:]) / np.mean(volumes) if len(volumes) >= 20 else 1.0,  # Volume ratio
                np.std(close_prices[-20:]) / np.mean(close_prices[-20:]) if len(close_prices) >= 20 else 0.0,  # Volatility
            ]
        )

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _detect_double_top(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect double top pattern (bearish reversal)."""
        patterns = []
        n = len(close)

        # Look for two peaks at similar levels with a trough between
        for i in range(20, n - 5):
            # Find local peaks
            window = 5
            if not self._is_local_max(high, i, window):
                continue

            # Look for second peak
            for j in range(i + 10, min(i + 30, n - 5)):
                if not self._is_local_max(high, j, window):
                    continue

                # Check if peaks are at similar level
                peak_diff = abs(high[i] - high[j]) / high[i]
                if peak_diff <= self.price_tolerance:
                    # Find trough between peaks
                    trough_slice_start = i + window
                    trough_slice_end = j - window
                    if trough_slice_end <= trough_slice_start:
                        continue  # Not enough space for trough
                    trough_idx = i + window + np.argmin(low[trough_slice_start:trough_slice_end])
                    trough = low[trough_idx]

                    # Check pattern validity
                    peak_avg = (high[i] + high[j]) / 2
                    trough_drop = (peak_avg - trough) / peak_avg

                    if trough_drop >= 0.02:  # At least 2% drop between peaks
                        confidence = min(0.9, 0.5 + (1 - peak_diff) * 10 + trough_drop * 5)
                        patterns.append(
                            PatternDetection(
                                pattern_name="double_top",
                                confidence=confidence,
                                start_idx=i - window,
                                end_idx=j + window,
                                direction="bearish",
                                expected_move_pct=self.PATTERN_EXPECTED_MOVES["double_top"],
                            )
                        )

        return patterns

    def _detect_double_bottom(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect double bottom pattern (bullish reversal)."""
        patterns = []
        n = len(close)

        for i in range(20, n - 5):
            window = 5
            if not self._is_local_min(low, i, window):
                continue

            for j in range(i + 10, min(i + 30, n - 5)):
                if not self._is_local_min(low, j, window):
                    continue

                trough_diff = abs(low[i] - low[j]) / low[i]
                if trough_diff <= self.price_tolerance:
                    peak_slice_start = i + window
                    peak_slice_end = j - window
                    if peak_slice_end <= peak_slice_start:
                        continue  # Not enough space for peak
                    peak_idx = i + window + np.argmax(high[peak_slice_start:peak_slice_end])
                    peak = high[peak_idx]

                    trough_avg = (low[i] + low[j]) / 2
                    peak_rise = (peak - trough_avg) / trough_avg

                    if peak_rise >= 0.02:
                        confidence = min(0.9, 0.5 + (1 - trough_diff) * 10 + peak_rise * 5)
                        patterns.append(
                            PatternDetection(
                                pattern_name="double_bottom",
                                confidence=confidence,
                                start_idx=i - window,
                                end_idx=j + window,
                                direction="bullish",
                                expected_move_pct=self.PATTERN_EXPECTED_MOVES["double_bottom"],
                            )
                        )

        return patterns

    def _detect_head_and_shoulders(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect head and shoulders pattern (bearish reversal)."""
        patterns = []
        n = len(close)

        for i in range(30, n - 10):
            window = 5

            # Find potential left shoulder
            if not self._is_local_max(high, i, window):
                continue

            # Find potential head (higher peak)
            for head_idx in range(i + 10, min(i + 25, n - 10)):
                if not self._is_local_max(high, head_idx, window):
                    continue

                if high[head_idx] <= high[i]:
                    continue

                # Find potential right shoulder (similar to left)
                for right_idx in range(head_idx + 10, min(head_idx + 25, n - 5)):
                    if not self._is_local_max(high, right_idx, window):
                        continue

                    shoulder_diff = abs(high[i] - high[right_idx]) / high[i]
                    if shoulder_diff <= self.price_tolerance * 1.5:
                        # Check neckline
                        nl_left_start, nl_left_end = i + window, head_idx - window
                        nl_right_start, nl_right_end = head_idx + window, right_idx - window
                        
                        if nl_left_end <= nl_left_start or nl_right_end <= nl_right_start:
                            continue  # Not enough space for neckline
                        
                        neckline_left = low[nl_left_start + np.argmin(low[nl_left_start:nl_left_end])]
                        neckline_right = low[nl_right_start + np.argmin(low[nl_right_start:nl_right_end])]
                        neckline_diff = abs(neckline_left - neckline_right) / neckline_left

                        if neckline_diff <= self.price_tolerance * 2:
                            confidence = min(
                                0.9,
                                0.4
                                + (1 - shoulder_diff) * 5
                                + (1 - neckline_diff) * 3,
                            )
                            patterns.append(
                                PatternDetection(
                                    pattern_name="head_and_shoulders",
                                    confidence=confidence,
                                    start_idx=i - window,
                                    end_idx=right_idx + window,
                                    direction="bearish",
                                    expected_move_pct=self.PATTERN_EXPECTED_MOVES[
                                        "head_and_shoulders"
                                    ],
                                )
                            )

        return patterns

    def _detect_inverse_head_and_shoulders(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect inverse head and shoulders pattern (bullish reversal)."""
        patterns = []
        n = len(close)

        for i in range(30, n - 10):
            window = 5

            if not self._is_local_min(low, i, window):
                continue

            for head_idx in range(i + 10, min(i + 25, n - 10)):
                if not self._is_local_min(low, head_idx, window):
                    continue

                if low[head_idx] >= low[i]:
                    continue

                for right_idx in range(head_idx + 10, min(head_idx + 25, n - 5)):
                    if not self._is_local_min(low, right_idx, window):
                        continue

                    shoulder_diff = abs(low[i] - low[right_idx]) / low[i]
                    if shoulder_diff <= self.price_tolerance * 1.5:
                        confidence = min(0.9, 0.4 + (1 - shoulder_diff) * 5)
                        patterns.append(
                            PatternDetection(
                                pattern_name="inverse_head_and_shoulders",
                                confidence=confidence,
                                start_idx=i - window,
                                end_idx=right_idx + window,
                                direction="bullish",
                                expected_move_pct=self.PATTERN_EXPECTED_MOVES[
                                    "inverse_head_and_shoulders"
                                ],
                            )
                        )

        return patterns

    def _detect_breakout_up(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray
    ) -> List[PatternDetection]:
        """Detect upward breakout pattern."""
        patterns = []
        n = len(close)

        if n < 30:
            return patterns

        # Find recent resistance
        recent_high = np.max(high[-30:-5])
        recent_volume_avg = np.mean(volume[-30:-5])

        # Check for breakout
        if high[-1] > recent_high * 1.01 and volume[-1] > recent_volume_avg * self.volume_threshold:
            confidence = min(0.9, 0.5 + (volume[-1] / recent_volume_avg - 1) * 0.3)
            patterns.append(
                PatternDetection(
                    pattern_name="breakout_up",
                    confidence=confidence,
                    start_idx=n - 30,
                    end_idx=n - 1,
                    direction="bullish",
                    expected_move_pct=self.PATTERN_EXPECTED_MOVES["breakout_up"],
                )
            )

        return patterns

    def _detect_breakout_down(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray
    ) -> List[PatternDetection]:
        """Detect downward breakout pattern."""
        patterns = []
        n = len(close)

        if n < 30:
            return patterns

        recent_low = np.min(low[-30:-5])
        recent_volume_avg = np.mean(volume[-30:-5])

        if low[-1] < recent_low * 0.99 and volume[-1] > recent_volume_avg * self.volume_threshold:
            confidence = min(0.9, 0.5 + (volume[-1] / recent_volume_avg - 1) * 0.3)
            patterns.append(
                PatternDetection(
                    pattern_name="breakout_down",
                    confidence=confidence,
                    start_idx=n - 30,
                    end_idx=n - 1,
                    direction="bearish",
                    expected_move_pct=self.PATTERN_EXPECTED_MOVES["breakout_down"],
                )
            )

        return patterns

    def _detect_ascending_triangle(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect ascending triangle pattern (bullish continuation)."""
        patterns = []
        n = len(close)

        if n < 30:
            return patterns

        # Look for flat resistance and rising support
        window = 20
        recent_highs = high[-window:]
        recent_lows = low[-window:]

        # Check for flat resistance
        high_std = np.std(recent_highs) / np.mean(recent_highs)
        if high_std > 0.02:
            return patterns

        # Check for rising lows
        low_slope = np.polyfit(np.arange(window), recent_lows, 1)[0]
        if low_slope <= 0:
            return patterns

        confidence = min(0.85, 0.5 + (1 - high_std * 20) + low_slope * 1000)
        patterns.append(
            PatternDetection(
                pattern_name="ascending_triangle",
                confidence=confidence,
                start_idx=n - window,
                end_idx=n - 1,
                direction="bullish",
                expected_move_pct=self.PATTERN_EXPECTED_MOVES["ascending_triangle"],
            )
        )

        return patterns

    def _detect_descending_triangle(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect descending triangle pattern (bearish continuation)."""
        patterns = []
        n = len(close)

        if n < 30:
            return patterns

        window = 20
        recent_highs = high[-window:]
        recent_lows = low[-window:]

        # Check for flat support
        low_std = np.std(recent_lows) / np.mean(recent_lows)
        if low_std > 0.02:
            return patterns

        # Check for declining highs
        high_slope = np.polyfit(np.arange(window), recent_highs, 1)[0]
        if high_slope >= 0:
            return patterns

        confidence = min(0.85, 0.5 + (1 - low_std * 20) + abs(high_slope) * 1000)
        patterns.append(
            PatternDetection(
                pattern_name="descending_triangle",
                confidence=confidence,
                start_idx=n - window,
                end_idx=n - 1,
                direction="bearish",
                expected_move_pct=self.PATTERN_EXPECTED_MOVES["descending_triangle"],
            )
        )

        return patterns

    def _detect_bull_flag(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect bull flag pattern (bullish continuation)."""
        patterns = []
        n = len(close)

        if n < 30:
            return patterns

        # Look for strong uptrend followed by consolidation
        pole_window = 10
        flag_window = 10

        if n < pole_window + flag_window:
            return patterns

        # Check for strong uptrend (pole)
        pole_close = close[-pole_window - flag_window : -flag_window]
        pole_return = (pole_close[-1] - pole_close[0]) / pole_close[0]

        if pole_return < 0.05:  # Need at least 5% rise for pole
            return patterns

        # Check for consolidation (flag)
        flag_close = close[-flag_window:]
        flag_return = (flag_close[-1] - flag_close[0]) / flag_close[0]

        if flag_return > -0.02:  # Need slight downward consolidation
            return patterns

        confidence = min(0.85, 0.5 + pole_return * 2)
        patterns.append(
            PatternDetection(
                pattern_name="bull_flag",
                confidence=confidence,
                start_idx=n - pole_window - flag_window,
                end_idx=n - 1,
                direction="bullish",
                expected_move_pct=self.PATTERN_EXPECTED_MOVES["bull_flag"],
            )
        )

        return patterns

    def _detect_bear_flag(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> List[PatternDetection]:
        """Detect bear flag pattern (bearish continuation)."""
        patterns = []
        n = len(close)

        if n < 30:
            return patterns

        pole_window = 10
        flag_window = 10

        if n < pole_window + flag_window:
            return patterns

        # Check for strong downtrend (pole)
        pole_close = close[-pole_window - flag_window : -flag_window]
        pole_return = (pole_close[-1] - pole_close[0]) / pole_close[0]

        if pole_return > -0.05:  # Need at least 5% drop for pole
            return patterns

        # Check for consolidation (flag)
        flag_close = close[-flag_window:]
        flag_return = (flag_close[-1] - flag_close[0]) / flag_close[0]

        if flag_return < 0.02:  # Need slight upward consolidation
            return patterns

        confidence = min(0.85, 0.5 + abs(pole_return) * 2)
        patterns.append(
            PatternDetection(
                pattern_name="bear_flag",
                confidence=confidence,
                start_idx=n - pole_window - flag_window,
                end_idx=n - 1,
                direction="bearish",
                expected_move_pct=self.PATTERN_EXPECTED_MOVES["bear_flag"],
            )
        )

        return patterns

    def _is_local_max(self, high: np.ndarray, idx: int, window: int) -> bool:
        """Check if index is a local maximum."""
        n = len(high)
        if idx < window or idx >= n - window:
            return False

        return all(high[idx] >= high[idx - j] for j in range(1, window + 1)) and all(
            high[idx] >= high[idx + j] for j in range(1, window + 1)
        )

    def _is_local_min(self, low: np.ndarray, idx: int, window: int) -> bool:
        """Check if index is a local minimum."""
        n = len(low)
        if idx < window or idx >= n - window:
            return False

        return all(low[idx] <= low[idx - j] for j in range(1, window + 1)) and all(
            low[idx] <= low[idx + j] for j in range(1, window + 1)
        )

    def _cluster_levels(
        self, points: List[Tuple[int, float]], num_levels: int
    ) -> List[float]:
        """Cluster similar price levels together."""
        if not points:
            return []

        levels = [p[1] for p in points]
        if len(levels) <= num_levels:
            return sorted(set(levels))

        # Simple clustering: group nearby levels
        levels.sort()
        clusters = [[levels[0]]]

        for level in levels[1:]:
            if abs(level - clusters[-1][-1]) / clusters[-1][-1] <= self.price_tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])

        # Take average of each cluster
        cluster_avgs = [np.mean(c) for c in clusters]

        # Return top num_levels by cluster size (more touches = stronger level)
        cluster_sizes = [(np.mean(c), len(c)) for c in clusters]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in cluster_sizes[:num_levels]]

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df.columns = [col.lower().strip() for col in df.columns]
        return df

    def _get_timestamp(self, df: pd.DataFrame) -> str:
        """Get timestamp from DataFrame."""
        if "timestamp" in df.columns:
            try:
                return str(df["timestamp"].iloc[-1])
            except (IndexError, KeyError):
                pass
        return datetime.now().isoformat()
