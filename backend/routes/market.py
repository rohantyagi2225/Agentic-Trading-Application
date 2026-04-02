from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional, Dict, Any
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import time

logger = logging.getLogger("MarketRoutes")

# Symbol mapping for special Yahoo Finance symbols
SYMBOL_MAPPING = {
    "VIX": "^VIX",  # CBOE Volatility Index
    "DJI": "^DJI",  # Dow Jones Industrial Average
    "IXIC": "^IXIC",  # NASDAQ Composite
    "GSPC": "^GSPC",  # S&P 500 Index
    "RUT": "^RUT",  # Russell 2000 Index
    "FTSE": "^FTSE",  # FTSE 100 Index
    "N225": "^N225",  # Nikkei 225 Index
    "HSI": "^HSI",  # Hang Seng Index
    # Crypto spot pairs
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
}


def normalize_symbol(symbol: str) -> str:
    """Convert frontend symbol to Yahoo Finance compatible symbol"""
    return SYMBOL_MAPPING.get(symbol.upper(), symbol)

router = APIRouter(tags=["Market"])

PERIOD_MAP = {
    "1D": ("1d", "5m"),
    "1W": ("5d", "15m"),
    "1M": ("1mo", "1h"),
    "3M": ("3mo", "1d"),
    "1Y": ("1y", "1wk"),
}

POPULAR_SYMBOLS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Auto"},
    {"symbol": "META", "name": "Meta Platforms", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
    {"symbol": "BTC-USD", "name": "Bitcoin", "sector": "Crypto"},
    {"symbol": "ETH-USD", "name": "Ethereum", "sector": "Crypto"},
    {"symbol": "SPY", "name": "S&P 500 ETF", "sector": "ETF"},
]


def format_error_response(message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Standardized error response format"""
    response = {
        "status": "error",
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    if details:
        response["details"] = details
    return response


def format_success_response(data: Any) -> Dict[str, Any]:
    """Standardized success response format"""
    return {
        "status": "success",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/price/{symbol}")
async def get_price(symbol: str):
    """Get current price and basic stats for a symbol with comprehensive error handling"""
    try:
        original_symbol = symbol.upper()
        # Normalize symbol for Yahoo Finance (e.g., VIX -> ^VIX)
        yahoo_symbol = normalize_symbol(original_symbol)

        cache_key = f"{original_symbol}"
        now_ts = time.time()
        cached = _price_cache.get(cache_key)
        if cached:
            cached_data, cached_at = cached
            if (now_ts - cached_at) < _PRICE_TTL_SECONDS:
                return format_success_response(cached_data)

        logger.info(f"Fetching price for {original_symbol} (Yahoo: {yahoo_symbol})")

        ticker = yf.Ticker(yahoo_symbol)

        # Prefer fast_info for low-latency quote data
        fast_info = getattr(ticker, "fast_info", {}) or {}
        price = fast_info.get("last_price") or fast_info.get("last") or fast_info.get("regular_market_price")
        prev_close = fast_info.get("previous_close") or fast_info.get("previousClose")
        open_price = fast_info.get("open")
        day_high = fast_info.get("day_high") or fast_info.get("high")
        day_low = fast_info.get("day_low") or fast_info.get("low")
        volume = fast_info.get("volume")

        # Fallback to recent history if fast_info is incomplete
        hist = ticker.history(period="2d")
        if (price is None or prev_close is None) and hist is not None and not hist.empty:
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
            price = float(latest["Close"])
            prev_close = float(prev["Close"])
            open_price = float(latest.get("Open", price))
            day_high = float(latest.get("High", price))
            day_low = float(latest.get("Low", price))
            volume = int(latest.get("Volume", 0))

        if price is None or prev_close is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Symbol '{original_symbol}' not found or market is closed"
            )

        change = float(price) - float(prev_close)
        # Return change_pct as decimal (e.g., 0.0123 = 1.23%)
        change_pct = (change / float(prev_close)) if prev_close else 0.0

        # Build lightweight sparkline history (last 30 closes)
        spark = []
        hist_short = ticker.history(period="1mo", interval="1d")
        if hist_short is not None and not hist_short.empty:
            for ts, row in hist_short.tail(30).iterrows():
                spark.append({
                    "time": int(ts.timestamp()),
                    "close": round(float(row["Close"]), 4),
                })

        data = {
            "symbol": original_symbol,
            "price": round(float(price), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 6),
            "volume": int(volume or 0),
            "open": round(float(open_price or price), 2),
            "high": round(float(day_high or price), 2),
            "low": round(float(day_low or price), 2),
            "prev_close": round(float(prev_close), 2),
            "timestamp": datetime.utcnow().isoformat(),
            "history": spark,
        }

        _price_cache[cache_key] = (data, now_ts)
        
        logger.debug(f"Successfully fetched price for {original_symbol}: ${price:.2f}")
        return format_success_response(data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {str(e)}", exc_info=True)
        cached = _price_cache.get(symbol.upper())
        if cached:
            cached_data, _ = cached
            cached_data = {**cached_data, "source": "cache_fallback"}
            return format_success_response(cached_data)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch price data: {str(e)}"
        )


# Simple in-memory cache for OHLCV data
_ohlcv_cache = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes for OHLCV data

# Simple in-memory cache for price quotes
_price_cache = {}
_PRICE_TTL_SECONDS = 8  # short TTL to smooth bursts without going stale

@router.get("/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    period: str = Query("1M", enum=["1D", "1W", "1M", "3M", "1Y"]),
    interval: Optional[str] = Query(None, description="Override default interval")
):
    """Get OHLCV data with proper error handling and validation"""
    try:
        original_symbol = symbol.upper()
        # Normalize symbol for Yahoo Finance (e.g., VIX -> ^VIX)
        yahoo_symbol = normalize_symbol(original_symbol)
        
        cache_key = f"{original_symbol}:{period}:{interval or 'default'}"
        
        # Check cache first
        current_time = time.time()
        if cache_key in _ohlcv_cache:
            cached_data, cached_at = _ohlcv_cache[cache_key]
            if (current_time - cached_at) < _CACHE_TTL_SECONDS:
                logger.debug(f"Cache hit for {cache_key}")
                return format_success_response(cached_data)
            else:
                logger.debug(f"Cache expired for {cache_key}")
                del _ohlcv_cache[cache_key]
        
        logger.info(f"Fetching OHLCV for {original_symbol} (Yahoo: {yahoo_symbol}), period={period}")
        
        yf_period, default_interval = PERIOD_MAP.get(period, ("1mo", "1d"))
        use_interval = interval or default_interval
        
        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period=yf_period, interval=use_interval)
        
        if hist.empty:
            logger.warning(f"No OHLCV data found for {original_symbol}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data available for {original_symbol} with period {period}"
            )
        
        records = []
        for ts, row in hist.iterrows():
            records.append({
                "time": int(ts.timestamp()),
                "date": ts.strftime("%b %d, %Y"),
                "datetime": ts.isoformat(),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row.get("Volume", 0)),
            })
        
        data = {
            "symbol": symbol,
            "period": period,
            "interval": use_interval,
            "count": len(records),
            "first_date": records[0]["date"] if records else None,
            "last_date": records[-1]["date"] if records else None,
            "prices": records
        }
        
        # Cache the result
        _ohlcv_cache[cache_key] = (data, current_time)
        logger.debug(f"Cached {len(records)} OHLCV records for {cache_key}")
        
        logger.debug(f"Successfully fetched {len(records)} OHLCV records for {symbol}")
        return format_success_response(data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch OHLCV data: {str(e)}"
        )


@router.get("/info/{symbol}")
async def get_symbol_info(symbol: str):
    """Get detailed symbol information with comprehensive error handling"""
    try:
        original_symbol = symbol.upper()
        yahoo_symbol = normalize_symbol(original_symbol)
        logger.info(f"Fetching info for {original_symbol} (Yahoo: {yahoo_symbol})")
        
        ticker = yf.Ticker(yahoo_symbol)
        info = ticker.info or {}
        
        # Validate we got some data
        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No information found for symbol '{symbol}'"
            )
        
        hist = ticker.history(period="2d")
        price = float(hist.iloc[-1]["Close"]) if not hist.empty else None
        
        data = {
            "symbol": original_symbol,
            "name": info.get("longName") or info.get("shortName", symbol),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            "eps": round(info.get("trailingEps"), 2) if info.get("trailingEps") else None,
            "dividend_yield": round(info.get("dividendYield") * 100, 2) if info.get("dividendYield") else None,
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "description": (info.get("longBusinessSummary", "")[:500] 
                          if info.get("longBusinessSummary") else ""),
            "price": round(price, 2) if price else None,
            "website": info.get("website"),
            "employees": info.get("fullTimeEmployees"),
            "headquarters": info.get("city")
        }
        
        logger.debug(f"Successfully fetched info for {symbol}")
        return format_success_response(data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching info for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch symbol information: {str(e)}"
        )


@router.get("/search")
async def search_symbols(q: str = Query(..., min_length=1, max_length=10)):
    """Search for symbols with improved error handling"""
    try:
        query = q.upper().strip()
        logger.info(f"Searching for: {query}")
        
        results = [
            s for s in POPULAR_SYMBOLS
            if query in s["symbol"].upper() or query in s["name"].upper()
        ]
        
        # Try direct ticker lookup for exact matches
        if not results and 1 <= len(query) <= 6:
            try:
                ticker = yf.Ticker(query)
                info = ticker.info or {}
                if info.get("regularMarketPrice") or info.get("currentPrice"):
                    results.append({
                        "symbol": query,
                        "name": info.get("longName") or info.get("shortName", query),
                        "sector": info.get("sector", ""),
                    })
                    logger.debug(f"Found direct match for {query}")
            except Exception as e:
                logger.debug(f"Direct lookup failed for {query}: {e}")
        
        if not results:
            logger.info(f"No results found for query: {query}")
        
        return format_success_response({
            "query": query,
            "count": len(results),
            "results": results[:10]
        })
        
    except Exception as e:
        logger.error(f"Error searching symbols: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/popular")
async def get_popular():
    """Get list of popular symbols"""
    return format_success_response({
        "count": len(POPULAR_SYMBOLS),
        "symbols": POPULAR_SYMBOLS
    })
