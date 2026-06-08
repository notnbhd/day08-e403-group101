"""
Task 10 — Generation Có Citation.

1. Retrieve (Task 9) → 2. Reorder tránh "lost in the middle" →
3. Format context có source → 4. Inject vào prompt → 5. Gọi LLM (OpenRouter
openai/gpt-4o-mini) → trả lời có citation, thiếu evidence thì từ chối.
"""

try:
    from .config import settings, get_openai_client
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # chạy trực tiếp: python src/task10_generation.py
    from config import settings, get_openai_client
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — lý do lựa chọn
# =============================================================================

# top_k=5: đủ evidence để trả lời nhưng không quá dài gây loãng / lost-in-middle.
TOP_K = 5
# top_p=0.9: nucleus sampling vừa phải — câu trả lời tự nhiên, không quá ngẫu nhiên.
TOP_P = 0.9
# temperature=0.3: RAG cần factual, bám context, hạn chế "bịa".
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp tránh "lost in the middle": quan trọng nhất ở ĐẦU và CUỐI,
    kém quan trọng ở GIỮA.

    Input (theo score giảm dần): [0, 1, 2, 3, 4]
    Output:                      [0, 2, 4, 3, 1]
    → chunk tốt nhất (0) đứng đầu, tốt nhì (1) đứng cuối, yếu nhất ở giữa.
    """
    if len(chunks) <= 2:
        return list(chunks)
    evens = chunks[0::2]          # 0, 2, 4, ... → phần đầu
    odds = chunks[1::2][::-1]     # ..., 3, 1   → phần cuối (đảo ngược)
    return evens + odds


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context có nhãn source để LLM cite được."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]
    for q in test_queries:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
