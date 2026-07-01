"""
HubMicro AI-Scribe — FastAPI application entry point.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

from app.database import create_tables
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router
from app.routes.projects import router as projects_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create SQLite tables on startup (safe to call every time — uses IF NOT EXISTS)
    create_tables()
    yield


app = FastAPI(
    title="HubMicro AI-Scribe",
    description=(
        "Standalone RAG service that ingests speaker-diarized meeting transcripts, "
        "transforms them into structured Markdown PRDs via Groq LLM, stores them in "
        "ChromaDB for semantic search, and exposes an MCP interface for IDE integration."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# Lightweight middleware to log incoming requests for debugging the ingest 405
@app.middleware("http")
async def log_requests(request, call_next):
    try:
        origin = request.headers.get("origin")
        content_type = request.headers.get("content-type")
        print(f"[REQ] {request.method} {request.url.path} Origin={origin} Content-Type={content_type}")
    except Exception:
        pass
    response = await call_next(request)
    try:
        print(f"[RESP] {request.method} {request.url.path} -> {response.status_code}")
    except Exception:
        pass
    return response

API_PREFIX = "/api/v1"
app.include_router(ingest_router, prefix=API_PREFIX, tags=["Ingestion"])
app.include_router(query_router, prefix=API_PREFIX, tags=["RAG Query"])
app.include_router(projects_router, prefix=API_PREFIX, tags=["Projects"])

ROOT_DIR = Path(__file__).resolve().parent.parent


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Frontend"])
def dashboard_page() -> HTMLResponse:
    dashboard_path = ROOT_DIR / "dashboard.html"
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/dashboard.html", response_class=HTMLResponse, tags=["Frontend"])
def dashboard_html_page() -> HTMLResponse:
    dashboard_path = ROOT_DIR / "dashboard.html"
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/healthz", tags=["Health"])
def healthz() -> dict:
    return {"status": "ok", "service": "hubmicro-ai-scribe", "version": "2.0.0"}
