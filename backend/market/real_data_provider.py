"""
Real Market Data Provider using Yahoo Finance

This module provides access to REAL historical market data instead of synthetic data.
Supports stocks, ETFs, indices, forex, and crypto.

Key Features:
- Historical daily data (OHLCV)
- Adjusted prices for splits/dividends
- Multiple asset classes
- Automatic data validation

Author: FinAgent Team
Version: 1.0.0
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

logger = logging.getLogger("MarketDataProvider")


class RealMarketDataFetcher:
    """
    Fetches real historical market data from Yahoo Finance
    
    Supported assets:
    - Stocks (AAPL, MSFT, TSLA)
    - ETFs (SPY, QQQ, IWM)
    - Indices (^GSPC, ^DJI, ^IXIC)
    - Forex (EURUSD=X, GBPUSD=X)
    - Crypto (BTC-USD, ETH-USD)
    """
    
    def __init__(self, default_period: str = "2y", default_interval: str = "1d"):
        """
        Initialize data fetcher
        
        Args:
            default_period: Default lookback period (e.g., "2y" = 2 years)
            default_interval: Default data interval ("1d", "1h", etc.)
        """
        self.default_period = default_period
        self.default_interval = default_interval
        
        logger.info(f"✅ Real market data fetcher initialized")
        logger.debug(f"   Default period: {default_period}")
        logger.debug(f"   Default interval: {default_interval}")
    
    def fetch(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol
        
        Args:
            symbol: Ticker symbol (e.g., "AAPL", "^GSPC", "BTC-USD")
            start_date: Start date (if None, uses period parameter)
            end_date: End date (defaults to today)
            period: Lookback period ("1y", "2y", "max") - used if start_date not provided
            interval: Data interval ("1d", "1h", "5m", etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        end_date = end_date or datetime.now()
        
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        
        try:
            if start_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval or self.default_interval)
            else:
                df = ticker.history(period=period or self.default_period, interval=interval or self.default_interval)
            
            if df.empty:
                raise ValueError(f"No data found for symbol '{symbol}'")
            
            # Validate data quality
            self._validate_data(df, symbol)
            
            logger.info(f"✅ Fetched {len(df)} records for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch(symbol, start_date, end_date, interval=interval)
                results[symbol] = df
                logger.debug(f"Fetched {symbol}: {len(df)} records")
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(results)}/{len(symbols)} symbols")
        
        return results
    
    def _validate_data(self, df: pd.DataFrame, symbol: str):
        """Validate data quality"""
        # Check for missing values
        if df.isnull().any().any():
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            logger.warning(f"{symbol}: {missing_pct:.1%} missing values")
        
        # Check for zero prices
        if 'Close' in df.columns and (df['Close'] == 0).any():
            logger.warning(f"{symbol}: Found zero prices - may indicate issues")
        
        # Check for negative volumes
        if 'Volume' in df.columns and (df['Volume'] < 0).any():
            raise ValueError(f"{symbol}: Negative volumes detected - data error")
        
        # Check for extreme returns (potential errors)
        if len(df) > 1:
            returns = df['Close'].pct_change()
            extreme_returns = (returns.abs() > 0.5).sum()
            if extreme_returns > 0:
                logger.warning(f"{symbol}: {extreme_returns} days with >50% moves - verify data")
    
    def get_sp500_constituents(self) -> List[str]:
        """Get list of S&P 500 constituents"""
        try:
            sp500 = yf.Ticker("^GSPC")
            # Get components from Wikipedia via yfinance
            import requests
            response = requests.get(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            tables = pd.read_html(response.text)
            df = tables[0]
            symbols = df['Symbol'].tolist()
            logger.info(f"Retrieved {len(symbols)} S&P 500 constituents")
            return symbols
        except Exception as e:
            logger.error(f"Failed to get S&P 500 constituents: {e}")
            return []
    
    def get_market_holidays(self, year: int = 2024) -> List[datetime]:
        """Get list of market holidays (NYSE closed dates)"""
        # NYSE holidays for common years
        holidays = {
            2024: [
                datetime(year, 1, 1),   # New Year's Day
                datetime(year, 1, 15),  # MLK Day
                datetime(year, 2, 19),  # Presidents Day
                datetime(year, 3, 29),  # Good Friday
                datetime(year, 5, 27),  # Memorial Day
                datetime(year, 6, 19),  # Juneteenth
                datetime(year, 7, 4),   # Independence Day
                datetime(year, 9, 2),   # Labor Day
                datetime(year, 11, 28), # Thanksgiving
                datetime(year, 12, 25), # Christmas
            ],
            2025: [
                datetime(year, 1, 1),   # New Year's Day
                datetime(year, 1, 20),  # MLK Day
                datetime(year, 2, 17),  # Presidents Day
                datetime(year, 4, 18),  # Good Friday
                datetime(year, 5, 26),  # Memorial Day
                datetime(year, 6, 19),  # Juneteenth
                datetime(year, 7, 4),   # Independence Day
                datetime(year, 9, 1),   # Labor Day
                datetime(year, 11, 27), # Thanksgiving
                datetime(year, 12, 25), # Christmas
            ]
        }
        
        return holidays.get(year, [])


def create_market_data_fetcher(**kwargs) -> RealMarketDataFetcher:
    """Factory function to create market data fetcher"""
    return RealMarketDataFetcher(**kwargs)


# Example usage
if __name__ == "__main__":
    fetcher = create_market_data_fetcher()
    
    # Fetch Apple stock data
    print("Fetching AAPL data...")
    aapl = fetcher.fetch("AAPL", period="1y")
    print(f"AAPL: {len(aapl)} days of data")
    print(aapl.tail())
    
    # Fetch S&P 500 data
    print("\nFetching S&P 500 data...")
    spy = fetcher.fetch("^GSPC", period="1y")
    print(f"S&P 500: {len(spy)} days of data")
    
    # Fetch Bitcoin data
    print("\nFetching Bitcoin data...")
    btc = fetcher.fetch("BTC-USD", period="1y")
    print(f"BTC: {len(btc)} days of data")
