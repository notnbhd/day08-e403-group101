from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .chat_service import run_chat
    from .memory import CONVERSATIONS
    from .schemas import ChatRequest, ChatResponse, HealthResponse, ResetRequest
except ImportError:
    from chat_service import run_chat
    from memory import CONVERSATIONS
    from schemas import ChatRequest, ChatResponse, HealthResponse, ResetRequest


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


app = FastAPI(
    title="Day 8 RAG Chatbot API",
    description="FastAPI web UI for RAG chatbot with citation, memory, and source documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(INDEX_FILE)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app="day8-rag-chatbot-fastapi")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    return run_chat(
        message=message,
        session_id=request.session_id,
        top_k=request.top_k,
        rewrite_followup=request.rewrite_followup,
        include_logs=request.include_logs,
    )


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict[str, str | bool | None]:
    CONVERSATIONS.reset(request.session_id)
    return {
        "ok": True,
        "session_id": request.session_id,
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "history": CONVERSATIONS.get_history(session_id),
    }


@app.get("/api/sessions")
def list_sessions() -> dict:
    return {
        "sessions": CONVERSATIONS.list_sessions(),
    }
