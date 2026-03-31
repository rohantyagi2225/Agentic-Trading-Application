"""Text Encoder for Financial News and Text Processing.

This module provides sentiment analysis, entity extraction, and news impact
scoring for financial text data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SentimentResult:
    """Container for sentiment analysis results.

    Attributes
    ----------
    score : float
        Sentiment score in range [-1, 1] where -1 is very negative
        and 1 is very positive.
    magnitude : float
        Magnitude of sentiment in range [0, 1], indicating the strength
        of emotional content regardless of direction.
    positive_words : List[str]
        List of positive financial words found in the text.
    negative_words : List[str]
        List of negative financial words found in the text.
    financial_terms : List[str]
        List of financial terminology found in the text.
    """

    score: float
    magnitude: float
    positive_words: List[str] = field(default_factory=list)
    negative_words: List[str] = field(default_factory=list)
    financial_terms: List[str] = field(default_factory=list)


@dataclass
class EntityResult:
    """Container for entity extraction results.

    Attributes
    ----------
    tickers : List[str]
        List of stock ticker symbols found (e.g., 'AAPL', 'MSFT').
    events : List[str]
        List of financial events mentioned (e.g., 'earnings beat', 'FDA approval').
    institutions : List[str]
        List of financial institutions mentioned.
    amounts : List[str]
        List of monetary amounts found in the text.
    """

    tickers: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    institutions: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)


@dataclass
class TextFeatures:
    """Container for aggregated text features.

    Attributes
    ----------
    features : np.ndarray
        Aggregated text feature vector.
    per_item_features : List[dict]
        Per-item breakdown of features for each text item.
    sentiment_score : float
        Overall sentiment score in range [-1, 1].
    metadata : dict
        Additional metadata about feature extraction.
    """

    features: np.ndarray
    per_item_features: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextEncoder:
    """Encodes financial news and text into feature vectors.

    This encoder performs sentiment analysis using financial word lists,
    extracts entities (tickers, events, institutions, amounts), and computes
    temporal news impact scores.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with the following options:
        - positive_words: List[str] - Custom positive word list
        - negative_words: List[str] - Custom negative word list
        - decay_hours: float - Hours for temporal decay (default 24)

    Example
    -------
    >>> encoder = TextEncoder()
    >>> news_items = [
    ...     {"text": "AAPL surges on record earnings", "timestamp": "2024-01-15", "source": "reuters"}
    ... ]
    >>> features = encoder.encode(news_items)
    >>> print(features.sentiment_score)  # Positive sentiment
    """

    # Comprehensive financial sentiment word lists
    DEFAULT_POSITIVE_WORDS = {
        # Price movement
        "rally",
        "surge",
        "soar",
        "jump",
        "climb",
        "rise",
        "gain",
        "advance",
        "rallying",
        "surging",
        # Performance
        "beat",
        "outperform",
        "exceed",
        "surpass",
        "top",
        "beat estimates",
        "raised guidance",
        # Analyst actions
        "upgrade",
        "upgraded",
        "buy rating",
        "overweight",
        "strong buy",
        # Business outlook
        "bullish",
        "growth",
        "profit",
        "record",
        "breakthrough",
        "expansion",
        "success",
        "strong",
        # Dividends and returns
        "dividend",
        "buyback",
        "share repurchase",
        # Deals and partnerships
        "deal",
        "partnership",
        "acquisition",
        "merger",
        "strategic",
        # Market position
        "market leader",
        "dominant",
        "competitive advantage",
        "moat",
        # Technical
        "breakout",
        "support",
        "golden cross",
        "oversold",
        # General positive
        "opportunity",
        "potential",
        "promising",
        "optimistic",
        "confident",
        "recovery",
        "rebound",
    }

    DEFAULT_NEGATIVE_WORDS = {
        # Price movement
        "crash",
        "plunge",
        "plummet",
        "tumble",
        "slide",
        "fall",
        "drop",
        "decline",
        "sink",
        "crashing",
        "plunging",
        # Performance
        "miss",
        "underperform",
        "disappoint",
        "fall short",
        "miss estimates",
        "lowered guidance",
        "warn",
        "warning",
        # Analyst actions
        "downgrade",
        "downgraded",
        "sell rating",
        "underweight",
        "strong sell",
        # Business outlook
        "bearish",
        "loss",
        "recession",
        "default",
        "bankruptcy",
        "bankrupt",
        "layoffs",
        "layoff",
        "fired",
        "closure",
        "shutdown",
        # Legal and regulatory
        "lawsuit",
        "investigation",
        "probe",
        "fraud",
        "scandal",
        "fine",
        "penalty",
        "sanction",
        # Market position
        "struggle",
        "struggling",
        "weak",
        "concern",
        "risk",
        "threat",
        # Technical
        "breakdown",
        "resistance",
        "death cross",
        "overbought",
        # General negative
        "uncertain",
        "uncertainty",
        "volatility",
        "turmoil",
        "crisis",
        "pessimistic",
        "worried",
        "fear",
        "dump",
    }

    # Financial terminology for entity extraction
    FINANCIAL_TERMS = {
        "earnings",
        "revenue",
        "profit",
        "margin",
        "ebitda",
        "eps",
        "guidance",
        "forecast",
        "outlook",
        "dividend",
        "buyback",
        "fomc",
        "fed",
        "interest rate",
        "inflation",
        "gdp",
        "unemployment",
        "cpi",
        "pmi",
    }

    # Event phrases for entity extraction
    EVENT_PATTERNS = [
        r"earnings\s+(?:beat|miss|report|call)",
        r"(?:fda|ema)\s+(?:approval|rejection)",
        r"(?:product|drug)\s+(?:launch|approval|recall)",
        r"(?:merger|acquisition|deal)",
        r"(?:buyback|share\s+repurchase)",
        r"dividend\s+(?:increase|cut|suspend)",
        r"(?:upgrade|downgrade)",
        r"(?:guidance|forecast)\s+(?:raised|lowered)",
        r"(?:quarterly|annual)\s+results",
        r"ipos?",
        r"(?:stock|share)\s+(?:split|offering)",
    ]

    # Major financial institutions
    INSTITUTION_PATTERNS = [
        r"Goldman\s+Sachs",
        r"Morgan\s+Stanley",
        r"J\.?P\.?\s*Morgan",
        r"Bank\s+of\s+America",
        r"Citigroup",
        r"Wells\s+Fargo",
        r"BlackRock",
        r"Vanguard",
        r"Berkshire\s+Hathaway",
        r"Fidelity",
        r"Charles\s+Schwab",
        r"Raymond\s+James",
        r"Deutsche\s+Bank",
        r"Credit\s+Suisse",
        r"UBS",
        r"Barclays",
        r"HSBC",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TextEncoder.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. See class docstring for options.
        """
        self.config = config or {}
        self.decay_hours = self.config.get("decay_hours", 24)

        # Use custom word lists or defaults
        self.positive_words = set(
            self.config.get("positive_words", list(self.DEFAULT_POSITIVE_WORDS))
        )
        self.negative_words = set(
            self.config.get("negative_words", list(self.DEFAULT_NEGATIVE_WORDS))
        )

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for entity extraction."""
        # Ticker pattern: $TICKER or standalone uppercase 1-5 letter word
        self.ticker_pattern = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{1,5})\b")

        # Amount pattern: $X.XX or $X million/billion
        self.amount_pattern = re.compile(
            r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|m|b|t))?"
        )

        # Event patterns
        self.event_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.EVENT_PATTERNS
        ]

        # Institution patterns
        self.institution_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INSTITUTION_PATTERNS
        ]

    def encode(self, text_items: List[Dict[str, Any]]) -> TextFeatures:
        """Encode a list of text items into aggregated features.

        Parameters
        ----------
        text_items : List[dict]
            List of text items, each with keys:
            - text: str - The text content
            - timestamp: str or datetime - Timestamp of the text
            - source: str - Source of the text (e.g., 'reuters', 'bloomberg')

        Returns
        -------
        TextFeatures
            Container with aggregated features, per-item breakdown,
            and overall sentiment score.
        """
        if not text_items:
            return self._empty_features()

        per_item_features = []
        sentiments = []
        magnitudes = []
        entity_counts = []
        timestamps = []

        for item in text_items:
            text = item.get("text", "")
            if not text:
                continue

            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(text)
            sentiments.append(sentiment_result.score)
            magnitudes.append(sentiment_result.magnitude)

            # Extract entities
            entity_result = self.extract_entities(text)
            entity_count = (
                len(entity_result.tickers)
                + len(entity_result.events)
                + len(entity_result.institutions)
            )
            entity_counts.append(entity_count)

            # Parse timestamp
            ts = self._parse_timestamp(item.get("timestamp"))
            if ts:
                timestamps.append(ts)

            # Store per-item features
            per_item_features.append(
                {
                    "text_snippet": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment_score": sentiment_result.score,
                    "sentiment_magnitude": sentiment_result.magnitude,
                    "positive_words": sentiment_result.positive_words,
                    "negative_words": sentiment_result.negative_words,
                    "tickers": entity_result.tickers,
                    "events": entity_result.events,
                    "institutions": entity_result.institutions,
                    "amounts": entity_result.amounts,
                    "source": item.get("source", "unknown"),
                    "timestamp": item.get("timestamp"),
                }
            )

        if not sentiments:
            return self._empty_features()

        # Compute aggregated features
        current_time = max(timestamps) if timestamps else datetime.now()

        # Temporal impact score
        temporal_impact = self.compute_news_impact(text_items, current_time)

        # Build feature vector
        # [avg_sentiment, sentiment_std, max_sentiment, min_sentiment,
        #  news_count, avg_magnitude, entity_density, temporal_impact_score]
        features = np.array(
            [
                np.mean(sentiments),
                np.std(sentiments) if len(sentiments) > 1 else 0.0,
                np.max(sentiments),
                np.min(sentiments),
                float(len(sentiments)),
                np.mean(magnitudes),
                np.mean(entity_counts) if entity_counts else 0.0,
                temporal_impact,
            ]
        )

        metadata = {
            "num_items": len(text_items),
            "sources": list(set(item.get("source", "unknown") for item in text_items)),
            "time_range": self._get_time_range(timestamps),
        }

        return TextFeatures(
            features=features,
            per_item_features=per_item_features,
            sentiment_score=float(np.mean(sentiments)),
            metadata=metadata,
        )

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of financial text.

        Parameters
        ----------
        text : str
            The text to analyze.

        Returns
        -------
        SentimentResult
            Container with sentiment score, magnitude, and found words.
        """
        if not text:
            return SentimentResult(score=0.0, magnitude=0.0)

        text_lower = text.lower()
        words = set(re.findall(r"\b[a-z]+\b", text_lower))
        total_words = len(words)

        # Find positive and negative words
        found_positive = [w for w in words if w in self.positive_words]
        found_negative = [w for w in words if w in self.negative_words]

        # Also check for multi-word phrases
        for phrase in self.positive_words:
            if " " in phrase and phrase in text_lower:
                found_positive.append(phrase)
        for phrase in self.negative_words:
            if " " in phrase and phrase in text_lower:
                found_negative.append(phrase)

        pos_count = len(found_positive)
        neg_count = len(found_negative)

        # Calculate score: (pos - neg) / (pos + neg + 1)
        # Range: -1 (all negative) to 1 (all positive)
        score = (pos_count - neg_count) / (pos_count + neg_count + 1)

        # Magnitude: (pos + neg) / total_words
        # Range: 0 (no sentiment words) to 1 (all words are sentiment words)
        magnitude = (pos_count + neg_count) / max(total_words, 1)

        # Find financial terms
        found_terms = [t for t in self.FINANCIAL_TERMS if t in text_lower]

        return SentimentResult(
            score=float(np.clip(score, -1, 1)),
            magnitude=float(np.clip(magnitude, 0, 1)),
            positive_words=found_positive,
            negative_words=found_negative,
            financial_terms=list(found_terms),
        )

    def extract_entities(self, text: str) -> EntityResult:
        """Extract financial entities from text.

        Parameters
        ----------
        text : str
            The text to extract entities from.

        Returns
        -------
        EntityResult
            Container with extracted tickers, events, institutions, and amounts.
        """
        if not text:
            return EntityResult()

        # Extract tickers
        tickers = set()
        for match in self.ticker_pattern.finditer(text):
            ticker = match.group(1) or match.group(2)
            if ticker and len(ticker) >= 1:
                # Filter out common false positives
                if ticker not in {"A", "I", "THE", "AND", "FOR", "TO", "OF", "IN", "ON"}:
                    tickers.add(ticker)

        # Extract events
        events = set()
        for pattern in self.event_patterns:
            for match in pattern.finditer(text):
                events.add(match.group(0))

        # Extract institutions
        institutions = set()
        for pattern in self.institution_patterns:
            for match in pattern.finditer(text):
                institutions.add(match.group(0))

        # Extract amounts
        amounts = set()
        for match in self.amount_pattern.finditer(text):
            amounts.add(match.group(0))

        return EntityResult(
            tickers=list(tickers),
            events=list(events),
            institutions=list(institutions),
            amounts=list(amounts),
        )

    def compute_news_impact(
        self,
        text_items: List[Dict[str, Any]],
        current_time: datetime,
        decay_hours: Optional[float] = None,
    ) -> float:
        """Compute temporal news impact score.

        Recent news has higher weight, with exponential decay.

        Parameters
        ----------
        text_items : List[dict]
            List of text items with 'text' and 'timestamp' keys.
        current_time : datetime
            Current time for computing decay.
        decay_hours : float, optional
            Decay constant in hours. Default uses config value.

        Returns
        -------
        float
            Temporal impact score, weighted average of sentiments with decay.
        """
        if decay_hours is None:
            decay_hours = self.decay_hours

        if not text_items:
            return 0.0

        weighted_sentiments = []
        weights = []

        for item in text_items:
            text = item.get("text", "")
            ts = self._parse_timestamp(item.get("timestamp"))

            if not text or not ts:
                continue

            # Get sentiment
            sentiment = self.analyze_sentiment(text).score

            # Compute time difference in hours
            time_diff = (current_time - ts).total_seconds() / 3600
            if time_diff < 0:
                time_diff = 0  # Future timestamp, treat as now

            # Exponential decay weight
            weight = np.exp(-time_diff / decay_hours)

            weighted_sentiments.append(sentiment * weight)
            weights.append(weight)

        if not weights:
            return 0.0

        # Normalized weighted impact
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(weighted_sentiments) / total_weight
        return 0.0

    def _empty_features(self) -> TextFeatures:
        """Return empty TextFeatures for edge cases."""
        return TextFeatures(
            features=np.zeros(8),
            per_item_features=[],
            sentiment_score=0.0,
            metadata={"warning": "No valid text items provided"},
        )

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
            except ValueError:
                try:
                    # Try common formats
                    for fmt in [
                        "%Y-%m-%d",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%SZ",
                    ]:
                        try:
                            return datetime.strptime(ts, fmt)
                        except ValueError:
                            continue
                except Exception:
                    pass
        return None

    def _get_time_range(self, timestamps: List[datetime]) -> Dict[str, str]:
        """Get time range from list of timestamps."""
        if not timestamps:
            return {"start": "N/A", "end": "N/A"}

        return {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
        }
