"""
Task 5 — Semantic Search Module (dense retrieval).

Embed query bằng cùng model ở Task 4 (text-embedding-3-small qua OpenRouter),
rồi near_vector trên Weaviate Cloud.

    semantic_search(query, top_k) -> list[{'content', 'score', 'metadata'}]
"""

try:
    from .config import settings, get_openai_client, connect_weaviate
except ImportError:  # chạy trực tiếp: python src/task5_semantic_search.py
    from config import settings, get_openai_client, connect_weaviate


def embed_query(query: str) -> list[float]:
    """Embed query bằng cùng embedding model của Task 4."""
    client = get_openai_client()
    resp = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=[query],
        dimensions=settings.EMBEDDING_DIM,
    )
    return resp.data[0].embedding


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Dense retrieval bằng cosine similarity trên Weaviate.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    from weaviate.classes.query import MetadataQuery

    query_vector = embed_query(query)

    client = connect_weaviate()
    try:
        collection = client.collections.get(settings.WEAVIATE_COLLECTION)
        res = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        results = []
        for obj in res.objects:
            dist = obj.metadata.distance
            score = 1.0 - dist if dist is not None else 0.0
            props = obj.properties
            results.append(
                {
                    "content": props.get("content", ""),
                    "score": float(score),
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
    for r in semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
