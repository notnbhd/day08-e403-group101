from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Conversation session id")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of source chunks")
    rewrite_followup: bool = Field(
        default=True,
        description="Rewrite follow-up question into a standalone retrieval query",
    )
    include_logs: bool = Field(default=True, description="Return pipeline logs")


class ResetRequest(BaseModel):
    session_id: str | None = None


class SourceDocument(BaseModel):
    index: int
    content: str
    preview: str
    score: float
    retrieval_source: str
    source_name: str
    doc_type: str
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    original_question: str
    standalone_question: str
    citations: list[str]
    sources: list[SourceDocument]
    suggested_questions: list[str] = Field(default_factory=list)
    retrieval_source: str
    elapsed_seconds: float
    logs: str = ""
    history_size: int


class HealthResponse(BaseModel):
    status: str
    app: str
