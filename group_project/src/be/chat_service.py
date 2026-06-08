from __future__ import annotations

import contextlib
import io
import random
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from .memory import CONVERSATIONS
    from .schemas import ChatResponse, SourceDocument
except ImportError:
    from memory import CONVERSATIONS
    from schemas import ChatResponse, SourceDocument


# Allow running from either:
#   cd group_project && uvicorn src.be.main:app --reload
# or:
#   cd group_project/src/be && uvicorn main:app --reload
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "data" / "standardized"

from src.config import get_openai_client, settings  # noqa: E402
from src.task10_generation import generate_with_citation  # noqa: E402


REWRITE_SYSTEM_PROMPT = """Bạn là query rewriter cho hệ thống RAG tiếng Việt.

Nhiệm vụ:
- Viết lại câu hỏi hiện tại thành một câu hỏi độc lập, đầy đủ ngữ cảnh.
- Dùng lịch sử hội thoại để thay thế các đại từ/mốc tham chiếu như: "người đó", "trường hợp này", "ở trên", "còn cái đó thì sao".
- Không trả lời câu hỏi.
- Không thêm thông tin mới ngoài lịch sử hội thoại.
- Giữ lại tên riêng, năm, điều luật, loại văn bản, sự kiện, chủ thể nếu có.
- Chỉ trả về đúng một câu hỏi đã viết lại.
"""


def _format_history_for_rewrite(history: list[dict[str, Any]], max_turns: int = 6) -> str:
    if not history:
        return "Không có lịch sử hội thoại."

    recent = history[-max_turns:]
    blocks: list[str] = []

    for idx, turn in enumerate(recent, start=1):
        user = (turn.get("user") or "").strip()
        assistant = (turn.get("assistant") or "").strip()
        standalone = (turn.get("standalone_question") or "").strip()

        if len(assistant) > 900:
            assistant = assistant[:900] + "..."

        blocks.append(
            f"Turn {idx}\n"
            f"User: {user}\n"
            f"Standalone query used: {standalone}\n"
            f"Assistant: {assistant}"
        )

    return "\n\n".join(blocks)


def rewrite_followup_question(
    question: str,
    history: list[dict[str, Any]],
    enabled: bool = True,
) -> str:
    """
    Convert a follow-up question into a standalone retrieval query.

    If there is no history, returns the original question.
    If rewrite fails, safely falls back to the original question.
    """
    question = question.strip()
    if not enabled or not history:
        return question

    try:
        client = get_openai_client()
        history_text = _format_history_for_rewrite(history)
        user_prompt = (
            f"Lịch sử hội thoại:\n{history_text}\n\n"
            f"Câu hỏi hiện tại:\n{question}\n\n"
            "Câu hỏi độc lập:"
        )

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ Follow-up rewrite failed: {type(exc).__name__}: {exc}")
        return question


def extract_citations(answer: str) -> list[str]:
    """
    Extract bracket-style citations from the answer.

    Example:
        [Luật Phòng chống ma tuý 2021, Điều 3]
        [VnExpress, 2024]
    """
    raw = re.findall(r"\[([^\[\]\n]{1,180})\]", answer or "")
    citations: list[str] = []
    seen: set[str] = set()

    for item in raw:
        value = item.strip()
        if value and value not in seen:
            citations.append(value)
            seen.add(value)

    return citations


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_sources(sources: list[dict[str, Any]] | None) -> list[SourceDocument]:
    normalized: list[SourceDocument] = []

    for idx, src in enumerate(sources or [], start=1):
        metadata = src.get("metadata") or {}
        content = src.get("content") or ""
        preview = content[:420].replace("\n", " ").strip()

        normalized.append(
            SourceDocument(
                index=idx,
                content=content,
                preview=preview,
                score=_safe_float(src.get("score")),
                retrieval_source=str(src.get("source") or "unknown"),
                source_name=str(metadata.get("source") or f"Source {idx}"),
                doc_type=str(metadata.get("type") or metadata.get("doc_type") or "unknown"),
                chunk_index=metadata.get("chunk_index"),
                metadata=metadata,
            )
        )

    return normalized


def _clean_question_text(value: str, max_len: int = 92) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .:-")
    if len(value) <= max_len:
        return value
    return value[:max_len].rsplit(" ", 1)[0].strip()


def _is_drug_related(value: str) -> bool:
    normalized = value.lower().replace("ma tuý", "ma túy")
    terms = (
        "ma túy",
        "ma túy",
        "chất ma",
        "chất ma",
        "cai nghiện",
        "nghiện ma",
        "thuốc phiện",
        "cần sa",
        "côca",
        "cocaine",
        "heroine",
        "ketamine",
        "methamphetamine",
    )
    return any(term in normalized for term in terms)


@lru_cache(maxsize=1)
def _dataset_question_seeds() -> tuple[dict[str, str], ...]:
    seeds: list[dict[str, str]] = []
    if not DATASET_DIR.exists():
        return ()

    for md_file in sorted(DATASET_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
        except OSError:
            continue

        doc_type = "legal" if "legal" in md_file.parts else "news"
        for match in re.finditer(r"(Điều\s+\d+\.\s+[^\n]{8,160})", content):
            window = content[match.start() : match.start() + 1200]
            title = _clean_question_text(match.group(1))
            if md_file.name.startswith("bo-luat-hinh-su") and not _is_drug_related(title):
                continue
            if doc_type == "legal" and not md_file.name.startswith("bo-luat-hinh-su") and not _is_drug_related(window):
                continue
            if title:
                seeds.append({"source": md_file.name, "type": doc_type, "topic": title})

        title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        if title_match and _is_drug_related(content[:1600]):
            title = _clean_question_text(title_match.group(1))
            if title:
                seeds.append({"source": md_file.name, "type": doc_type, "topic": title})

    return tuple(seeds)


def _source_question_seeds(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    seeds: list[dict[str, str]] = []
    for src in sources:
        content = src.get("content") or ""
        metadata = src.get("metadata") or {}
        source = str(metadata.get("source") or "nguồn đã dùng")
        doc_type = str(metadata.get("type") or metadata.get("doc_type") or "unknown")
        match = re.search(r"(Điều\s+\d+\.\s+.{8,160})", content)
        if match:
            seeds.append(
                {
                    "source": source,
                    "type": doc_type,
                    "topic": _clean_question_text(match.group(1)),
                }
            )
            continue

        sentence = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", content).strip())[0]
        sentence = _clean_question_text(sentence)
        if sentence:
            seeds.append({"source": source, "type": doc_type, "topic": sentence})

    return seeds


def generate_suggested_questions(
    sources: list[dict[str, Any]] | None,
    current_question: str,
    count: int = 3,
) -> list[str]:
    rng = random.SystemRandom()
    seeds = _source_question_seeds(sources or [])
    seeds.extend(rng.sample(list(_dataset_question_seeds()), k=min(12, len(_dataset_question_seeds()))))
    rng.shuffle(seeds)

    suggestions: list[str] = []
    seen: set[str] = {current_question.strip().lower()}
    for seed in seeds:
        topic = seed.get("topic", "")
        source = seed.get("source", "")
        if not topic:
            continue

        if seed.get("type") == "legal" or topic.startswith("Điều "):
            templates = [
                f"{topic} quy định những gì?",
                f"Giải thích {topic} theo văn bản {source}.",
                f"Các trường hợp áp dụng trong {topic} là gì?",
            ]
        else:
            templates = [
                f"Tóm tắt thông tin chính trong bài: {topic}.",
                f"Bài {source} nêu những sự kiện liên quan ma túy nào?",
                f"Các nhân vật hoặc vụ việc chính trong '{topic}' là gì?",
            ]

        question = rng.choice(templates)
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(question)
        if len(suggestions) >= count:
            break

    return suggestions


def run_chat(
    message: str,
    session_id: str | None = None,
    top_k: int = 5,
    rewrite_followup: bool = True,
    include_logs: bool = True,
) -> ChatResponse:
    sid = CONVERSATIONS.ensure_session(session_id)
    history = CONVERSATIONS.get_history(sid)

    log_buffer = io.StringIO()
    started = time.time()

    result: dict[str, Any] = {
        "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
        "sources": [],
        "retrieval_source": "none",
    }

    with contextlib.redirect_stdout(log_buffer):
        standalone_question = rewrite_followup_question(
            question=message,
            history=history,
            enabled=rewrite_followup,
        )

        print(f"▶ Session: {sid}")
        print(f"▶ Original question: {message!r}")
        print(f"▶ Standalone question: {standalone_question!r}")
        print(f"▶ top_k={top_k}")

        try:
            result = generate_with_citation(standalone_question, top_k=top_k)
            print("✓ Retrieval + generation completed.")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ Pipeline error: {type(exc).__name__}: {exc}")
            result = {
                "answer": f"Lỗi pipeline: {type(exc).__name__}: {exc}",
                "sources": [],
                "retrieval_source": "error",
            }

    elapsed = time.time() - started

    answer = result.get("answer") or ""
    raw_sources = result.get("sources") or []
    sources = normalize_sources(raw_sources)
    citations = extract_citations(answer)
    logs = log_buffer.getvalue().strip() if include_logs else ""
    suggested_questions = generate_suggested_questions(raw_sources, message)

    CONVERSATIONS.append_turn(
        session_id=sid,
        user_message=message,
        assistant_answer=answer,
        standalone_question=standalone_question,
        sources=raw_sources,
    )

    return ChatResponse(
        session_id=sid,
        answer=answer,
        original_question=message,
        standalone_question=standalone_question,
        citations=citations,
        sources=sources,
        suggested_questions=suggested_questions,
        retrieval_source=str(result.get("retrieval_source") or "none"),
        elapsed_seconds=round(elapsed, 3),
        logs=logs,
        history_size=len(CONVERSATIONS.get_history(sid)),
    )
