"""
Task 6 — Lexical Search Module (BM25 built-in của Weaviate).

Thay vì rank-bm25 in-memory, ta dùng BM25 native của Weaviate (`query.bm25`)
trên cùng collection đã index ở Task 4 → 1 nguồn dữ liệu duy nhất với dense search.

Cơ chế (giải thích cho demo → +5 bonus):
    Weaviate dùng BM25F trên inverted index:
      score(q,d) = Σ IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - IDF: từ hiếm trong corpus → quan trọng hơn.
    - TF với term-saturation (k1) + chuẩn hoá độ dài document (b).
    - BM25F = mở rộng BM25 cho nhiều field có trọng số khác nhau; chạy trực tiếp
      trên inverted index nên scale tốt hơn rank-bm25 tải toàn bộ corpus vào RAM.

    lexical_search(query, top_k) -> list[{'content', 'score', 'metadata'}]
"""

try:
    from .config import settings, connect_weaviate
except ImportError:  # chạy trực tiếp: python src/task6_lexical_search.py
    from config import settings, connect_weaviate


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Lexical retrieval bằng Weaviate BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by BM25 score descending.
    """
    from weaviate.classes.query import MetadataQuery

    client = connect_weaviate()
    try:
        collection = client.collections.get(settings.WEAVIATE_COLLECTION)
        res = collection.query.bm25(
            query=query,
            query_properties=["content"],
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
        )
        results = []
        for obj in res.objects:
            props = obj.properties
            score = obj.metadata.score
            results.append(
                {
                    "content": props.get("content", ""),
                    "score": float(score) if score is not None else 0.0,
                    "metadata": {
                        "source": props.get("source", ""),
                        "type": props.get("doc_type", ""),
                        "chunk_index": props.get("chunk_index", 0),
                    },
                }
            )
    finally:
        client.close()

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for r in lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
