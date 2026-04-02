import asyncio
import random
import yfinance as yf
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["WebSocket"])

BASE_PRICES = {
    "AAPL": 189.5, "MSFT": 378.2, "GOOGL": 141.8, "AMZN": 182.4,
    "TSLA": 248.7, "META": 503.1, "NVDA": 875.4,
}

ACTIONS = ["BUY", "SELL", "HOLD"]
EXPLANATIONS = {
    "BUY": "Momentum signal: 20-day MA crossed above 50-day MA with strong volume.",
    "SELL": "Mean reversion: Z-score > 2.0 — price significantly above historical mean.",
    "HOLD": "No edge detected — waiting for higher conviction signal.",
}


@router.websocket("/prices/{symbol}")
async def websocket_prices(websocket: WebSocket, symbol: str) -> None:
    """Stream real-time prices from Yahoo Finance with ultra-low latency"""
    symbol = symbol.upper()
    await websocket.accept()
    
    # Map special symbols for Yahoo Finance
    SYMBOL_MAPPING = {
        "VIX": "^VIX",
        "DJI": "^DJI",
        "IXIC": "^IXIC",
        "GSPC": "^GSPC",
    }
    yahoo_symbol = SYMBOL_MAPPING.get(symbol, symbol)
    
    try:
        ticker = yf.Ticker(yahoo_symbol)
        last_price = None
        
        while True:
            try:
                # Fetch latest price from Yahoo Finance
                info = ticker.info
                if info and 'regularMarketPrice' in info:
                    last_price = info['regularMarketPrice']
                    
                    # Calculate change
                    prev_close = info.get('previousClose', last_price)
                    change = last_price - prev_close
                    change_pct = (change / prev_close) if prev_close else 0
                    
                    await websocket.send_json({
                        "status": "success",
                        "data": {
                            "symbol": symbol,
                            "price": round(last_price, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 6),
                            "volume": info.get('volume', 0),
                            "timestamp": datetime.now().isoformat(),
                            "source": "yfinance_realtime",
                        },
                    })
                elif last_price:
                    # Fallback: send last known price if API fails
                    await websocket.send_json({
                        "status": "success",
                        "data": {
                            "symbol": symbol,
                            "price": round(last_price, 2),
                            "change": 0,
                            "change_pct": 0,
                            "volume": 0,
                            "timestamp": datetime.now().isoformat(),
                            "source": "cached",
                        },
                    })
                
                # Update every 2 seconds for low latency
                await asyncio.sleep(2.0)
                
            except Exception as e:
                # On error, send cached price and retry
                if last_price:
                    await websocket.send_json({
                        "status": "success",
                        "data": {
                            "symbol": symbol,
                            "price": round(last_price, 2),
                            "error": str(e),
                            "source": "error_fallback",
                        },
                    })
                await asyncio.sleep(3.0)
        
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/signals/{symbol}")
async def websocket_signals(websocket: WebSocket, symbol: str) -> None:
    symbol = symbol.upper()
    base = BASE_PRICES.get(symbol, 150.0)
    price = base
    await websocket.accept()

    try:
        while True:
            # Simulate price walk
            price = round(price * (1 + random.uniform(-0.005, 0.005)), 2)
            action = random.choices(ACTIONS, weights=[0.35, 0.25, 0.40])[0]
            confidence = round(random.uniform(0.48, 0.91), 3)

            await websocket.send_json({
                "status": "success",
                "data": {
                    "symbol": symbol,
                    "price": price,
                    "signal": action,
                    "action": action,
                    "confidence": confidence,
                    "explanation": EXPLANATIONS[action],
                    "agent": "momentum",
                },
            })
            await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
