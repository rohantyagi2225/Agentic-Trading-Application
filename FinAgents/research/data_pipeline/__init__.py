"""
Research Data Pipeline Module

This module provides research-grade data preprocessing and synthetic data generation
for the FinAgents trading system.
"""

from .data_sources import (
    DataSourceManager,
    SimulatedNewsGenerator,
    SyntheticReportGenerator,
    NewsItem,
    EarningsReport,
    MacroReport,
)

from .synthetic_data import (
    SyntheticDataGenerator,
    ChartTextPair,
    ReportSummaryPair,
)

from .feature_engineering import (
    FeatureEngineer,
)

from .preprocessor import (
    DataPreprocessor,
    DataQualityReport,
    ResearchDataset,
)

__all__ = [
    # Data Sources
    "DataSourceManager",
    "SimulatedNewsGenerator",
    "SyntheticReportGenerator",
    "NewsItem",
    "EarningsReport",
    "MacroReport",
    # Synthetic Data
    "SyntheticDataGenerator",
    "ChartTextPair",
    "ReportSummaryPair",
    # Feature Engineering
    "FeatureEngineer",
    # Preprocessing
    "DataPreprocessor",
    "DataQualityReport",
    "ResearchDataset",
]
