"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import weaviate_test  # type: ignore[import]
from sentence_transformers import SentenceTransformer  # type: ignore[import]


def semantic_search(query, top_k=10):
    """Tìm kiếm ngữ nghĩa sử dụng vector similarity."""

    model = SentenceTransformer("BAAI/bge-m3")
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    client = weaviate_test.Client("http://localhost:8080")

    query_request = client.query.get(
        "DrugLawDocs",
        ["content", "source", "doc_type", "chunk_index"],
    )
    query_request = query_request.with_near_vector({"vector": query_embedding})
    query_request = query_request.with_additional(["distance"])
    query_request = query_request.with_limit(top_k)
    response = query_request.do()

    data = response.get("data", {})
    items = data.get("Get", {}).get("DrugLawDocs", []) or []
    results = []
    for item in items:
        additional = item.get("_additional", {})
        distance = additional.get("distance", 1.0)
        metadata = dict(
            source=item.get("source", ""),
            doc_type=item.get("doc_type", ""),
            chunk_index=item.get("chunk_index", -1),
        )
        results.append(
            {
                "content": item.get("content", ""),
                "score": 1 - distance,
                "metadata": metadata,
            }
        )
    return results

if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")