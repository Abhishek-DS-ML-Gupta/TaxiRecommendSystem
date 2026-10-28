"""
main.py -- FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Then open: http://localhost:8000
"""
import sys, io
# Force UTF-8 stdout on Windows with line buffering so logs print in real-time
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api import recommend, network, stats
from app.core.startup import load_all

from dotenv import load_dotenv
load_dotenv()


# ─── Lifespan: load data once at startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60, flush=True)
    print("[TAXI] Taxi Route Recommender -- Starting Up", flush=True)
    print("=" * 60, flush=True)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    load_all(data_dir=data_dir, time_limit_tau=10.0)

    print("=" * 60, flush=True)
    print("[OK] Ready! Open http://localhost:8000 in your browser", flush=True)
    print("=" * 60, flush=True)
    yield
    print("Shutting down...", flush=True)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Taxi Route Recommender API",
    description=(
        "Real-time route recommendation for vacant taxi drivers. "
        "Based on the INFORMS JoC paper: "
        "\"A Cost-Effective Sequential Route Recommender System for Taxi Drivers\" "
        "(Liu, Teng, Chen, Xiong, 2023). "
        "Uses GCN-LSTM predictions + recursive route searching (RRS/SRRS)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow browser requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Routes ───────────────────────────────────────────────────────────────

app.include_router(recommend.router, prefix="/api", tags=["Recommendations"])
app.include_router(network.router, prefix="/api", tags=["Road Network"])
app.include_router(stats.router, prefix="/api", tags=["Model Statistics"])

@app.get("/api/config", tags=["Configuration"])
async def get_config():
    """Returns frontend configuration such as API keys from the .env file."""
    token = os.environ.get("MAPBOX_ACCESS_TOKEN", "")
    return {
        "mapbox_token": token,
        "mapbox_access_token": token
    }

# ─── Static Files (Frontend SPA) ──────────────────────────────────────────────

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Mount static assets (CSS, JS)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the main SPA."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/health")
async def health_check():
    from app.core.startup import app_state
    return {
        "status": "ok",
        "data_loaded": app_state["loaded"],
        "road_segments": len(app_state.get("pickup_probs", {})),
    }
