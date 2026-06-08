"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Cài đặt:
    pip install openai python-dotenv

File .env:
    OPENAI_API_KEY=your_api_key_here
    LLM_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import os
import re
from typing import Any

from _console import configure_utf8_output
from dotenv import load_dotenv

configure_utf8_output()

load_dotenv()


# =============================================================================
# ROBUST IMPORT
# =============================================================================

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# TOP_K = 5:
# - Đủ evidence để trả lời các câu hỏi legal/news có nhiều phần.
# - Không quá dài để tránh context nhiễu và lost-in-the-middle.
TOP_K = int(os.getenv("GENERATION_TOP_K", "5"))

# TOP_P = 1.0:
# - Giữ nucleus sampling trung tính.
# - Vì RAG cần factual, ta kiểm soát độ ổn định bằng TEMPERATURE thấp thay vì giảm cả top_p.
TOP_P = float(os.getenv("GENERATION_TOP_P", "1.0"))

# TEMPERATURE = 0.2:
# - RAG pháp luật/tin tức cần câu trả lời ổn định, ít sáng tạo.
# - 0.2 vẫn cho phép diễn đạt tự nhiên nhưng giảm suy diễn.
TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))

# Model mặc định theo skeleton trước đó. Có thể đổi qua .env.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Giới hạn context để tránh prompt quá dài.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1800"))
MIN_RETRIEVAL_CONFIDENCE = float(os.getenv("MIN_RETRIEVAL_CONFIDENCE", "0.15"))

# Nếu true: khi retrieval yếu/rỗng, không gọi LLM mà trả abstain ngay.
STRICT_EVIDENCE_CHECK = os.getenv("STRICT_EVIDENCE_CHECK", "true").lower() == "true"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý RAG trả lời bằng tiếng Việt, chỉ dựa trên CONTEXT được cung cấp.

Quy tắc bắt buộc:
- Chỉ dùng thông tin xuất hiện rõ trong CONTEXT.
- Mọi mệnh đề factual phải có citation ngay sau câu hoặc cụm tương ứng.
- Citation phải dùng đúng nhãn document trong CONTEXT, ví dụ: [D1] hoặc [D2].
- Không dùng kiến thức nền ngoài CONTEXT.
- Không đoán, không suy diễn vượt quá evidence.
- Nếu CONTEXT không đủ để xác minh, trả lời đúng câu: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
- Không hướng dẫn cách sử dụng, điều chế, mua bán, che giấu hoặc né tránh xử lý liên quan đến chất cấm.
- Nếu câu hỏi yêu cầu hướng dẫn nguy hiểm hoặc bất hợp pháp, chỉ trả lời ở mức thông tin pháp luật/an toàn và nêu rằng không thể hỗ trợ hướng dẫn đó.

Yêu cầu trình bày:
- Trả lời trực tiếp câu hỏi.
- Dùng đoạn văn rõ ràng hoặc bullet ngắn nếu có nhiều ý.
- Cuối câu trả lời thêm mục "Nguồn đã dùng" liệt kê các citation label đã sử dụng.
"""


# =============================================================================
# DOCUMENT REORDERING — tránh lost in the middle
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle".

    LLM thường chú ý tốt hơn ở đầu và cuối prompt.
    Strategy:
        - Chunk rank 1 đặt đầu.
        - Chunk rank 2 đặt cuối.
        - Các chunk còn lại đặt giữa theo thứ tự xen kẽ.

    Ví dụ input theo score:
        [1, 2, 3, 4, 5]

    Output:
        [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    # Đảm bảo input đang sorted theo score/confidence giảm dần.
    sorted_chunks = sorted(
        chunks,
        key=lambda item: float(item.get("confidence", item.get("score", 0.0)) or 0.0),
        reverse=True,
    )

    front = sorted_chunks[0::2]
    back = sorted_chunks[1::2]

    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def safe_get(metadata: dict, *keys: str, default: str = "N/A") -> str:
    """Lấy metadata theo nhiều key fallback."""
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def truncate_chunk(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """Cắt chunk quá dài để giữ prompt gọn."""
    text = str(text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[truncated]"


def make_source_label(index: int) -> str:
    """Tạo citation label ngắn."""
    return f"D{index}"


def infer_source_name(chunk: dict, index: int) -> str:
    """Tạo tên nguồn dễ đọc từ metadata."""
    metadata = chunk.get("metadata", {}) or {}

    title = safe_get(metadata, "title", default="")
    source_path = safe_get(metadata, "source_path", "source", "filename", default="")
    url = safe_get(metadata, "url", default="")

    if title and title != "N/A":
        return title

    if source_path and source_path != "N/A":
        return source_path

    if url and url != "N/A":
        return url

    return f"Source {index}"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label [D1], [D2], ... để LLM cite chính xác.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    if not chunks:
        return ""

    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        label = make_source_label(i)

        source_name = infer_source_name(chunk, i)
        doc_type = safe_get(metadata, "doc_type", "type", default="unknown")
        source_path = safe_get(metadata, "source_path", "source", "filename", default="N/A")
        url = safe_get(metadata, "url", default="N/A")
        chunk_index = safe_get(metadata, "chunk_index", default="N/A")
        retrieval_source = chunk.get("source") or safe_get(metadata, "retrieval_pipeline_source", default="unknown")
        score = float(chunk.get("score", 0.0) or 0.0)
        confidence = float(chunk.get("confidence", 0.0) or 0.0)

        content = truncate_chunk(chunk.get("content", ""))

        context_parts.append(
            f"[{label}]\n"
            f"Source name: {source_name}\n"
            f"Source path: {source_path}\n"
            f"URL: {url}\n"
            f"Document type: {doc_type}\n"
            f"Chunk index: {chunk_index}\n"
            f"Retrieval source: {retrieval_source}\n"
            f"Retrieval score: {score:.4f}\n"
            f"Retrieval confidence: {confidence:.4f}\n"
            f"Content:\n{content}\n"
        )

    return "\n---\n".join(context_parts)


def build_source_summary(chunks: list[dict]) -> list[dict]:
    """Tạo danh sách source trả về kèm answer."""
    sources = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}

        sources.append(
            {
                "label": make_source_label(i),
                "source_name": infer_source_name(chunk, i),
                "source_path": safe_get(metadata, "source_path", "source", "filename", default="N/A"),
                "url": safe_get(metadata, "url", default="N/A"),
                "doc_type": safe_get(metadata, "doc_type", "type", default="unknown"),
                "chunk_index": metadata.get("chunk_index"),
                "score": float(chunk.get("score", 0.0) or 0.0),
                "confidence": float(chunk.get("confidence", 0.0) or 0.0),
                "retrieval_source": chunk.get("source") or metadata.get("retrieval_pipeline_source", "unknown"),
                "content_preview": str(chunk.get("content", ""))[:250],
            }
        )

    return sources


# =============================================================================
# EVIDENCE CHECKING
# =============================================================================

def has_sufficient_evidence(chunks: list[dict]) -> bool:
    """
    Check đơn giản trước khi gọi LLM.

    Điều kiện:
        - Có ít nhất 1 chunk.
        - Chunk tốt nhất có confidence hoặc score trên ngưỡng thấp.
    """
    if not chunks:
        return False

    best = chunks[0]
    confidence = float(best.get("confidence", best.get("score", 0.0)) or 0.0)

    return confidence >= MIN_RETRIEVAL_CONFIDENCE


def abstain_response(query: str, chunks: list[dict] | None = None) -> dict:
    """Return chuẩn khi không đủ evidence."""
    chunks = chunks or []

    return {
        "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
        "sources": build_source_summary(chunks),
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "context": format_context(chunks) if chunks else "",
        "query": query,
        "model": None,
        "top_k": len(chunks),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "used_llm": False,
    }


# =============================================================================
# LLM CALL
# =============================================================================

def get_openai_client():
    """Khởi tạo OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Hãy thêm vào file .env:\n"
            "OPENAI_API_KEY=your_api_key_here"
        )

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Gọi OpenAI Chat Completions API.

    Chat Completions vẫn được hỗ trợ; code dùng API này để khớp skeleton bài học.
    """
    client = get_openai_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return answer.strip()


# =============================================================================
# POST-CHECK
# =============================================================================

def answer_has_citation(answer: str) -> bool:
    """Kiểm tra câu trả lời có citation dạng [D1], [D2]."""
    return bool(re.search(r"\[D\d+\]", answer or ""))


def enforce_citation_policy(answer: str, chunks: list[dict]) -> str:
    """
    Nếu model trả lời không citation, thay bằng abstain để tránh hallucination.
    """
    if not answer:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    if "Tôi không thể xác minh thông tin này từ nguồn hiện có" in answer:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    if not answer_has_citation(answer):
        return (
            "Tôi không thể xác minh thông tin này từ nguồn hiện có.\n\n"
            "Lý do: câu trả lời sinh ra không có citation bắt buộc từ context."
        )

    return answer


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    query = str(query or "").strip()

    if not query:
        raise ValueError("Query không được rỗng.")

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if STRICT_EVIDENCE_CHECK and not has_sufficient_evidence(chunks):
        return abstain_response(query, chunks)

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    if not context.strip():
        return abstain_response(query, chunks)

    # Step 4: Build prompt
    user_message = f"""CONTEXT:
{context}

---

QUESTION:
{query}

---

TASK:
Trả lời QUESTION bằng tiếng Việt, chỉ dựa trên CONTEXT.
Mỗi factual claim phải có citation dạng [D1], [D2].
Nếu CONTEXT không đủ evidence, trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
"""

    # Step 5: Call LLM
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        answer = enforce_citation_policy(answer, reordered)

    except Exception as exc:
        answer = (
            "Tôi không thể xác minh thông tin này từ nguồn hiện có.\n\n"
            f"Lý do kỹ thuật: {exc}"
        )

    # Step 6: Return
    return {
        "answer": answer,
        "sources": build_source_summary(reordered),
        "retrieval_source": reordered[0].get("source", "hybrid") if reordered else "none",
        "context": context,
        "query": query,
        "model": LLM_MODEL,
        "top_k": top_k,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "used_llm": True,
    }


# =============================================================================
# CLI
# =============================================================================

def print_sources(sources: list[dict]):
    """In source summary."""
    if not sources:
        print("[Sources: 0]")
        return

    print(f"\n[Sources: {len(sources)} chunks]")
    for source in sources:
        print(
            f"  {source['label']}: "
            f"{source.get('source_path')} | "
            f"type={source.get('doc_type')} | "
            f"score={source.get('score', 0.0):.4f} | "
            f"confidence={source.get('confidence', 0.0):.4f}"
        )


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma túy?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma túy 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)

        result = generate_with_citation(q)

        print(f"\nA: {result['answer']}")
        print_sources(result["sources"])
        print(f"\n[Retrieval via: {result['retrieval_source']} | model={result['model']}]")
