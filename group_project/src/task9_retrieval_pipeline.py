"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh (Weaviate native hybrid).

Logic:
    Query
      ├→ embed query (OpenRouter)
      ├→ Weaviate HYBRID (BM25 + dense, fuse nội bộ qua alpha)  → source="hybrid"
      ├→ Rerank (cohere/rerank-v3.5 qua OpenRouter)
      ├→ Nếu rỗng / top score < threshold → fallback PageIndex (source="pageindex")
      └→ Nếu cloud fallback lỗi/rỗng → fallback local markdown (source="local")

Vì sao dùng Weaviate hybrid thay vì tự RRF: Weaviate fuse BM25 + dense ngay
trong 1 query (alpha=0.5 = cân bằng), giảm round-trip và là 1 nguồn dữ liệu duy
nhất. (rerank_rrf vẫn được giữ ở Task 7 để tham khảo / so sánh.)
"""

import math
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

try:
    from .config import settings, connect_weaviate
    from .task5_semantic_search import embed_query
    from .task7_reranking import rerank
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:  # chạy trực tiếp: python src/task9_retrieval_pipeline.py
    from config import settings, connect_weaviate
    from task5_semantic_search import embed_query
    from task7_reranking import rerank
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3      # top score < ngưỡng → fallback PageIndex
DEFAULT_TOP_K = 5
HYBRID_ALPHA = 0.5         # 0 = thuần BM25, 1 = thuần vector, 0.5 = cân bằng
RERANK_METHOD = "cross_encoder"
LOCAL_CHUNK_SIZE = 900
LOCAL_CHUNK_OVERLAP = 120
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
STOPWORDS = {
    "a", "b", "c", "d", "đ", "e", "g", "h", "i", "k", "l", "m", "n",
    "ai", "bao", "bị", "bởi", "các", "cái", "cho", "có", "của", "đã",
    "đến", "để", "được", "gì", "hay", "khi", "không", "là", "làm",
    "lên", "luật", "mà", "một", "năm", "nào", "này", "như", "những",
    "ở", "phải", "qua", "quy", "ra", "sẽ", "sau", "số", "theo",
    "thì", "trong", "từ", "và", "vào", "về", "vì", "việc", "với",
    "định",
}


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    # Common spelling variants in the source/query text.
    return text.replace("ma tuý", "ma túy")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", _normalize_text(text), flags=re.UNICODE)


def _content_terms(tokens: list[str]) -> set[str]:
    return {token for token in tokens if len(token) > 1 and token not in STOPWORDS}


def _query_phrases(tokens: list[str], min_n: int = 2, max_n: int = 5) -> list[str]:
    content_tokens = [token for token in tokens if token not in STOPWORDS]
    phrases: list[str] = []
    for size in range(max_n, min_n - 1, -1):
        for start in range(0, len(content_tokens) - size + 1):
            phrases.append(" ".join(content_tokens[start : start + size]))
    return phrases


def _article_heading(text: str) -> str:
    match = re.match(r"^(điều\s+\d+\.)(.*?)(?=\s+\d+\.|$)", text)
    if not match:
        return ""
    return f"{match.group(1)}{match.group(2)}".strip()


def _article_body_after_heading(text: str, heading: str) -> str:
    if not heading:
        return text
    return text[len(heading) :].strip()


def _chunk_text(text: str, chunk_size: int = LOCAL_CHUNK_SIZE) -> list[dict]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    article_sections = [
        section.strip()
        for section in re.split(r"(?=Điều\s+\d+\.)", normalized)
        if section.strip()
    ]
    units = article_sections if len(article_sections) > 1 else [
        p.strip() for p in re.split(r"(?<=[.!?])\s+", normalized) if p.strip()
    ]

    chunks: list[str] = []
    current = ""

    for paragraph in units:
        if not current:
            current = paragraph
            continue
        if len(current) + len(paragraph) + 1 <= chunk_size:
            current = f"{current} {paragraph}"
            continue
        chunks.append(current)
        current = paragraph

    if current:
        chunks.append(current)

    expanded: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.3:
            expanded.append(
                {
                    "content": chunk,
                    "article_heading": _article_heading(_normalize_text(chunk)),
                    "is_continuation": False,
                }
            )
            continue

        heading_match = re.match(r"^(Điều\s+\d+\..{0,180}?)(?=\s+\d+\.)", chunk)
        heading = heading_match.group(1).strip() if heading_match else ""
        normalized_heading = _normalize_text(heading)
        start = 0
        while start < len(chunk):
            end = min(start + chunk_size, len(chunk))
            if end < len(chunk):
                boundary = chunk.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            part = chunk[start:end].strip()
            if heading and start > 0 and not part.startswith(heading):
                part = f"{heading} {part}"
            expanded.append(
                {
                    "content": part,
                    "article_heading": normalized_heading,
                    "is_continuation": start > 0,
                }
            )
            if end >= len(chunk):
                break
            start = max(end - LOCAL_CHUNK_OVERLAP, start + 1)

    return [chunk for chunk in expanded if chunk["content"]]


@lru_cache(maxsize=1)
def _local_index() -> tuple[dict, ...]:
    """Load markdown chunks from data/standardized for offline fallback search."""
    if not STANDARDIZED_DIR.exists():
        return ()

    chunks: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        doc_type = "legal" if "legal" in md_file.parts else "news"
        for idx, chunk in enumerate(_chunk_text(content)):
            chunk_text = chunk["content"]
            tokens = _tokenize(chunk_text)
            if not tokens:
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "source": md_file.name,
                        "type": doc_type,
                        "chunk_index": idx,
                    },
                    "article_heading": chunk.get("article_heading", ""),
                    "is_continuation": bool(chunk.get("is_continuation")),
                    "term_counts": Counter(tokens),
                    "length": len(tokens),
                }
            )

    return tuple(chunks)


def _local_search(query: str, top_k: int) -> list[dict]:
    """
    Lightweight BM25-style fallback over local markdown.

    This keeps the chatbot useful for demos when Weaviate is unreachable or the
    PageIndex key is missing/invalid. It does not replace cloud retrieval.
    """
    docs = _local_index()
    if not docs:
        return []

    query_terms = Counter(_tokenize(query))
    if not query_terms:
        return []

    avgdl = sum(doc["length"] for doc in docs) / len(docs)
    doc_freq = Counter()
    for doc in docs:
        for term in query_terms:
            if term in doc["term_counts"]:
                doc_freq[term] += 1

    scored: list[dict] = []
    k1 = 1.5
    b = 0.75
    query_token_list = list(query_terms)
    query_content_terms = _content_terms(query_token_list)
    query_phrases = _query_phrases(query_token_list)
    for doc in docs:
        score = 0.0
        length = doc["length"] or 1
        content_lower = _normalize_text(doc["content"])
        leading_text = content_lower[:320]
        for term, query_weight in query_terms.items():
            tf = doc["term_counts"].get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf + k1 * (1 - b + b * length / avgdl)
            score += query_weight * idf * (tf * (k1 + 1) / denom)

        heading = str(doc.get("article_heading") or _article_heading(content_lower))
        if heading:
            body_after_heading = _article_body_after_heading(content_lower, heading)
            heading_terms = _content_terms(_tokenize(heading))
            leading_terms = _content_terms(_tokenize(leading_text))
            heading_overlap = query_content_terms & heading_terms
            leading_overlap = query_content_terms & leading_terms

            # Generic legal-document signals: article headings and early text are
            # usually better evidence than later boilerplate or footnotes.
            score += 5.0
            score += 8.0 * len(heading_overlap)
            score += 2.5 * len(leading_overlap)

            phrase_hits = sum(1 for phrase in query_phrases if phrase in leading_text)
            score += 12.0 * phrase_hits

            if doc.get("is_continuation"):
                score -= 24.0
            elif re.match(r"^\d+\.", body_after_heading):
                score += 16.0
            elif body_after_heading:
                score -= 6.0

            if query_content_terms and not heading_overlap:
                score -= 32.0
                if len(leading_overlap) <= 1:
                    score -= 10.0

        if score <= 0:
            continue
        scored.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": dict(doc["metadata"]),
                "source": "local",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _hybrid_search(query: str, limit: int) -> list[dict]:
    """Weaviate native hybrid (BM25 + dense)."""
    from weaviate.classes.query import MetadataQuery

    query_vector = embed_query(query)
    client = connect_weaviate()
    try:
        collection = client.collections.get(settings.WEAVIATE_COLLECTION)
        res = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=HYBRID_ALPHA,
            limit=limit,
            return_metadata=MetadataQuery(score=True),
        )
        out = []
        for obj in res.objects:
            props = obj.properties
            score = obj.metadata.score
            out.append(
                {
                    "content": props.get("content", ""),
                    "score": float(score) if score is not None else 0.0,
                    "metadata": {
                        "source": props.get("source", ""),
                        "type": props.get("doc_type", ""),
                        "chunk_index": props.get("chunk_index", 0),
                    },
                    "source": "hybrid",
                }
            )
        return out
    finally:
        client.close()


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'hybrid'|'pageindex'}
    """
    # Step 1-2: hybrid fusion trong Weaviate.
    try:
        candidates = _hybrid_search(query, limit=top_k * 2)
    except Exception as e:
        print(f"  ⚠ Hybrid search lỗi: {e}")
        candidates = []

    # Step 3: rerank (giữ source="hybrid").
    final_results = candidates
    if use_reranking and candidates:
        try:
            reranked = rerank(query, candidates, top_k=top_k, method=RERANK_METHOD)
            for r in reranked:
                r["source"] = "hybrid"
            final_results = reranked
        except Exception as e:
            print(f"  ⚠ Rerank lỗi, dùng kết quả hybrid: {e}")
            final_results = candidates[:top_k]

    # Step 4: fallback PageIndex nếu không đủ tốt.
    top_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or top_score < score_threshold:
        print(
            f"  ⚠ Hybrid yếu (top={top_score:.3f} < {score_threshold}). "
            f"Fallback → PageIndex"
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as e:
            print(f"  ⚠ PageIndex fallback lỗi: {e}")

        local_fallback = _local_search(query, top_k=top_k)
        if local_fallback:
            print(f"  ✓ Local markdown fallback: {len(local_fallback)} chunks")
            return local_fallback
        print("  ⚠ Local markdown fallback không có kết quả")

    return final_results[:top_k]


if __name__ == "__main__":
    queries = sys.argv[1:] or ["Nhập câu hỏi cần kiểm thử retrieval"]
    for q in queries:
        print(f"\nQuery: {q}\n" + "-" * 60)
        for i, r in enumerate(retrieve(q, top_k=3), 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
