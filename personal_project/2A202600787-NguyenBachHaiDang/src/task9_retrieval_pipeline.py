"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh (Weaviate native hybrid).

Logic:
    Query
      ├→ embed query (OpenRouter)
      ├→ Weaviate HYBRID (BM25 + dense, fuse nội bộ qua alpha)  → source="hybrid"
      ├→ Rerank (cohere/rerank-v3.5 qua OpenRouter)
      └→ Nếu rỗng / top score < threshold → fallback PageIndex (source="pageindex")

Vì sao dùng Weaviate hybrid thay vì tự RRF: Weaviate fuse BM25 + dense ngay
trong 1 query (alpha=0.5 = cân bằng), giảm round-trip và là 1 nguồn dữ liệu duy
nhất. (rerank_rrf vẫn được giữ ở Task 7 để tham khảo / so sánh.)
"""

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

    return final_results[:top_k]


if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]
    for q in queries:
        print(f"\nQuery: {q}\n" + "-" * 60)
        for i, r in enumerate(retrieve(q, top_k=3), 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
