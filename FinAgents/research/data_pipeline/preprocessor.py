"""
Data Preprocessing Module

Provides research-grade data preprocessing including missing data handling,
normalization, temporal splitting, data quality validation, and end-to-end
dataset creation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import pandas as pd
import numpy as np

from .data_sources import DataSourceManager
from .feature_engineering import FeatureEngineer


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""
    total_rows: int
    missing_pct: dict
    outlier_count: int
    date_gaps: list
    duplicate_dates: list
    quality_score: float


@dataclass
class ResearchDataset:
    """Complete research dataset with all components."""
    features: pd.DataFrame
    labels: pd.Series
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    quality_report: DataQualityReport
    metadata: dict


class DataPreprocessor:
    """
    Data preprocessing pipeline for research datasets.
    
    Handles missing data, normalization, temporal splitting, and quality validation.
    Provides end-to-end dataset creation from raw data.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.normalization_params = {}
    
    def handle_missing_data(
        self,
        df: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Handle missing data in DataFrame.
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values:
                - 'forward_fill': Forward fill (ffill)
                - 'interpolate': Linear interpolation
                - 'drop': Drop rows with any missing values
                
        Returns:
            DataFrame with missing values handled
            
        Raises:
            ValueError: If invalid method specified
        """
        result = df.copy()
        
        # Report missing percentages
        missing_pct = (result.isnull().sum() / len(result) * 100).to_dict()
        missing_pct = {k: round(v, 2) for k, v in missing_pct.items() if v > 0}
        
        if missing_pct:
            print(f"Missing data detected: {missing_pct}")
        
        if method == 'forward_fill':
            result = result.ffill().bfill()  # Forward then backward fill
        elif method == 'interpolate':
            result = result.interpolate(method='linear').ffill().bfill()
        elif method == 'drop':
            result = result.dropna()
        else:
            raise ValueError(f"Invalid method: {method}. Choose from 'forward_fill', 'interpolate', 'drop'")
        
        return result
    
    def normalize(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        columns: Optional[list[str]] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize DataFrame columns.
        
        Args:
            df: Input DataFrame
            method: Normalization method:
                - 'zscore': Standard score (mean=0, std=1)
                - 'minmax': Min-max scaling to [0, 1]
                - 'robust': Robust scaling using median and IQR
            columns: List of columns to normalize. If None, normalizes all numeric columns.
            
        Returns:
            Tuple of (normalized DataFrame, normalization parameters dict)
            
        Raises:
            ValueError: If invalid method specified
        """
        result = df.copy()
        
        # Select columns to normalize
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        params = {}
        
        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame")
                continue
            
            col_data = df[col].dropna()
            
            if method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                if std != 0:
                    result[col] = (df[col] - mean) / std
                params[col] = {'method': 'zscore', 'mean': mean, 'std': std}
                
            elif method == 'minmax':
                min_val = col_data.min()
                max_val = col_data.max()
                range_val = max_val - min_val
                if range_val != 0:
                    result[col] = (df[col] - min_val) / range_val
                params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                
            elif method == 'robust':
                median = col_data.median()
                q75 = col_data.quantile(0.75)
                q25 = col_data.quantile(0.25)
                iqr = q75 - q25
                if iqr != 0:
                    result[col] = (df[col] - median) / iqr
                params[col] = {'method': 'robust', 'median': median, 'iqr': iqr}
                
            else:
                raise ValueError(f"Invalid method: {method}. Choose from 'zscore', 'minmax', 'robust'")
        
        self.normalization_params = params
        return result, params
    
    def inverse_normalize(
        self,
        df: pd.DataFrame,
        params: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Inverse normalize DataFrame using stored or provided parameters.
        
        Args:
            df: Normalized DataFrame
            params: Normalization parameters. If None, uses stored params from last normalize() call.
            
        Returns:
            DataFrame with original scale restored
        """
        if params is None:
            params = self.normalization_params
        
        if not params:
            raise ValueError("No normalization parameters available. Run normalize() first or provide params.")
        
        result = df.copy()
        
        for col, col_params in params.items():
            if col not in df.columns:
                continue
            
            method = col_params['method']
            
            if method == 'zscore':
                result[col] = df[col] * col_params['std'] + col_params['mean']
            elif method == 'minmax':
                result[col] = df[col] * (col_params['max'] - col_params['min']) + col_params['min']
            elif method == 'robust':
                result[col] = df[col] * col_params['iqr'] + col_params['median']
        
        return result
    
    def temporal_split(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train/val/test sets respecting time order.
        
        No future leakage: all training data comes before validation data,
        which comes before test data.
        
        Args:
            df: Input DataFrame (should be time-indexed)
            train_pct: Percentage for training set
            val_pct: Percentage for validation set
            test_pct: Percentage for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
            
        Raises:
            ValueError: If percentages don't sum to 1.0
        """
        if not np.isclose(train_pct + val_pct + test_pct, 1.0):
            raise ValueError(f"Percentages must sum to 1.0, got {train_pct + val_pct + test_pct}")
        
        n = len(df)
        train_end = int(n * train_pct)
        val_end = train_end + int(n * val_pct)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def validate_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate data quality and generate report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataQualityReport with quality metrics
        """
        total_rows = len(df)
        
        # Missing data percentage per column
        missing_pct = (df.isnull().sum() / total_rows * 100).to_dict()
        missing_pct = {k: round(v, 2) for k, v in missing_pct.items()}
        
        # Outlier detection (using IQR method on numeric columns)
        outlier_count = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                outlier_count += outliers
        
        # Date gaps (if datetime index)
        date_gaps = []
        if isinstance(df.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
                missing_dates = full_range.difference(df.index)
                date_gaps = [str(d) for d in missing_dates]
        
        # Duplicate dates
        duplicate_dates = []
        if isinstance(df.index, pd.DatetimeIndex):
            duplicates = df.index[df.index.duplicated()]
            duplicate_dates = [str(d) for d in duplicates.unique()]
        
        # Calculate quality score (0-1)
        quality_score = 1.0
        
        # Penalize missing data
        avg_missing = sum(missing_pct.values()) / len(missing_pct) if missing_pct else 0
        quality_score -= min(avg_missing / 100, 0.3)
        
        # Penalize outliers
        outlier_ratio = outlier_count / (total_rows * len(numeric_cols)) if numeric_cols.any() else 0
        quality_score -= min(outlier_ratio, 0.2)
        
        # Penalize date gaps
        if isinstance(df.index, pd.DatetimeIndex):
            expected_days = (df.index.max() - df.index.min()).days + 1
            gap_ratio = len(date_gaps) / expected_days if expected_days > 0 else 0
            quality_score -= min(gap_ratio, 0.2)
        
        # Penalize duplicates
        dup_ratio = len(duplicate_dates) / total_rows if total_rows > 0 else 0
        quality_score -= min(dup_ratio, 0.1)
        
        quality_score = max(0.0, min(1.0, round(quality_score, 2)))
        
        return DataQualityReport(
            total_rows=total_rows,
            missing_pct=missing_pct,
            outlier_count=outlier_count,
            date_gaps=date_gaps,
            duplicate_dates=duplicate_dates,
            quality_score=quality_score
        )
    
    def create_research_dataset(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        label_lookforward: int = 5,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2
    ) -> ResearchDataset:
        """
        Create a complete research dataset end-to-end.
        
        Pipeline: load -> feature engineer -> preprocess -> split -> validate
        
        Args:
            symbol: Stock symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            label_lookforward: Periods to look forward for label generation
            train_pct: Training set percentage
            val_pct: Validation set percentage
            test_pct: Test set percentage
            
        Returns:
            ResearchDataset with all components
        """
        # Step 1: Load data
        data_manager = DataSourceManager()
        raw_df = data_manager.load_historical_data(symbol, start_date, end_date)
        
        # Step 2: Feature engineering
        engineer = FeatureEngineer()
        features_df = engineer.compute_technical_indicators(raw_df)
        features_df = engineer.compute_statistical_features(features_df)
        
        # Step 3: Create labels (future return direction)
        features_df['future_return'] = features_df['Close'].shift(-label_lookforward) / features_df['Close'] - 1
        features_df['label'] = features_df['future_return'].apply(self._classify_return)
        
        # Step 4: Handle missing data
        features_df = self.handle_missing_data(features_df, method='forward_fill')
        
        # Step 5: Validate quality
        quality_report = self.validate_data_quality(features_df)
        
        # Step 6: Split into train/val/test
        train_df, val_df, test_df = self.temporal_split(
            features_df, train_pct, val_pct, test_pct
        )
        
        # Extract features and labels
        feature_cols = [c for c in features_df.columns if c not in ['label', 'future_return']]
        features = features_df[feature_cols]
        labels = features_df['label']
        
        # Create metadata
        metadata = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'label_lookforward': label_lookforward,
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'test_rows': len(test_df),
            'feature_count': len(feature_cols),
        }
        
        return ResearchDataset(
            features=features,
            labels=labels,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            quality_report=quality_report,
            metadata=metadata
        )
    
    def _classify_return(self, ret: float) -> int:
        """Classify return into label: 1 (up >1%), -1 (down >1%), 0 (flat)."""
        if pd.isna(ret):
            return 0
        if ret > 0.01:
            return 1
        elif ret < -0.01:
            return -1
        else:
            return 0
