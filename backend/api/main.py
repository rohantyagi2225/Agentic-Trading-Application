from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.portfolio import router as portfolio_router
from backend.routes.health import router as health_router
from backend.routes.agents import router as agents_router
from backend.routes.market import router as market_router
from backend.routes.ws_signals import router as ws_signals_router
from backend.routes.signals import router as signals_router


app = FastAPI(
    title="Agentic Trading API",
    version="1.0.0",
    description="Production-grade API for Agentic Trading System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(signals_router, prefix="/signals", tags=["Signals"])
app.include_router(portfolio_router, prefix="/portfolio", tags=["Portfolio"])
app.include_router(agents_router, prefix="/agents", tags=["Agents"])
app.include_router(market_router, prefix="/market", tags=["Market"])
app.include_router(health_router, tags=["Health"])

# WebSocket routes
app.include_router(ws_signals_router, prefix="/ws", tags=["WebSocket"])


@app.get("/")
def root():
    return {
        "message": "Agentic Trading API is running",
        "docs": "/docs",
        "health": "/health"
    }