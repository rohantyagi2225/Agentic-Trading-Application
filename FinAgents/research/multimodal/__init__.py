"""Multimodal Intelligence Pipeline for Financial Analysis.

This module provides a comprehensive multimodal approach to financial market analysis
by combining time series encoding, text sentiment analysis, and chart pattern detection.

Key Components
--------------
TimeSeriesEncoder
    Extracts technical indicators and features from OHLCV price data.

TextEncoder
    Processes financial news and text for sentiment analysis and entity extraction.

ChartEncoder
    Detects chart patterns and support/resistance levels from price data.

FusionEngine
    Combines signals from multiple modalities into unified trading signals.

MultimodalAgent
    Full multimodal agent that integrates all encoders and fusion engine.

Example
-------
>>> from FinAgents.research.multimodal import (
...     TimeSeriesEncoder, TextEncoder, ChartEncoder,
...     FusionEngine, MultimodalAgent
... )
>>> 
>>> # Create encoders
>>> ts_encoder = TimeSeriesEncoder()
>>> text_encoder = TextEncoder()
>>> chart_encoder = ChartEncoder()
>>> 
>>> # Process data
>>> ts_features = ts_encoder.encode(prices_df)
>>> text_features = text_encoder.encode(news_items)
>>> chart_features = chart_encoder.encode(prices_df)
>>> 
>>> # Fuse signals
>>> fusion = FusionEngine()
>>> result = fusion.fuse(ts_features, text_features, chart_features)
"""

from FinAgents.research.multimodal.time_series_encoder import (
    TimeSeriesEncoder,
    TimeSeriesFeatures,
)
from FinAgents.research.multimodal.text_encoder import (
    TextEncoder,
    TextFeatures,
    SentimentResult,
    EntityResult,
)
from FinAgents.research.multimodal.chart_encoder import (
    ChartEncoder,
    ChartFeatures,
    PatternDetection,
)
from FinAgents.research.multimodal.fusion_engine import (
    FusionEngine,
    FusionResult,
)
from FinAgents.research.multimodal.multimodal_agent import (
    MultimodalAgent,
)

__all__ = [
    # Encoders
    "TimeSeriesEncoder",
    "TextEncoder",
    "ChartEncoder",
    "FusionEngine",
    # Agent
    "MultimodalAgent",
    # Dataclasses
    "TimeSeriesFeatures",
    "TextFeatures",
    "SentimentResult",
    "EntityResult",
    "ChartFeatures",
    "PatternDetection",
    "FusionResult",
]

__version__ = "1.0.0"
