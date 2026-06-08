"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Cài đặt:
    pip install sentence-transformers weaviate-client

Điều kiện:
    - Đã chạy Task 4 để index dữ liệu vào Weaviate.
    - Weaviate local đang chạy ở port 8080 và gRPC 50051.
"""

import os
from functools import lru_cache
from typing import Any

from _console import configure_utf8_output

configure_utf8_output()


# =============================================================================
# CONFIGURATION — phải khớp với Task 4
# =============================================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

VECTOR_STORE = os.getenv("VECTOR_STORE", "weaviate")

# Trong file Task 4 đã dùng COLLECTION_NAME = "LegalNewsDocs".
# Nếu bạn đặt tên collection khác ở Task 4, sửa biến này cho khớp.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "LegalNewsDocs")

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))


# =============================================================================
# EMBEDDING
# =============================================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load embedding model một lần để tránh tải lại ở mỗi query."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(query: str) -> list[float]:
    """
    Embed query bằng cùng model đã dùng trong Task 4.

    normalize_embeddings=True vì Task 4 cũng normalize embedding khi index.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query không được rỗng.")

    model = get_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    actual_dim = int(embedding.shape[0])
    if actual_dim != EMBEDDING_DIM:
        print(
            f"⚠ EMBEDDING_DIM config={EMBEDDING_DIM}, "
            f"nhưng model trả về dim={actual_dim}."
        )

    return embedding.astype("float32").tolist()


# =============================================================================
# HELPERS
# =============================================================================

def validate_top_k(top_k: int) -> int:
    """Kiểm tra và giới hạn top_k."""
    if not isinstance(top_k, int):
        raise TypeError("top_k phải là int.")

    if top_k <= 0:
        raise ValueError("top_k phải > 0.")

    return min(top_k, 100)


def distance_to_similarity(distance: float | None) -> float:
    """
    Convert Weaviate distance sang similarity score.

    Với cosine distance:
        similarity ~= 1 - distance

    Score càng cao thì chunk càng liên quan.
    """
    if distance is None:
        return 0.0

    return float(1.0 - distance)


def clean_metadata(properties: dict[str, Any]) -> dict:
    """Chuẩn hóa metadata trả về cho dễ đọc."""
    metadata_keys = [
        "source",
        "source_path",
        "doc_type",
        "title",
        "chunk_id",
        "object_uuid",
        "chunk_index",
        "chunking_method",
        "char_count",
        "content_hash",
        "document_hash",
        "embedding_model",
        "embedding_dim",
        "url",
        "source_domain",
        "published_date",
        "date_crawled",
    ]

    metadata = {}

    for key in metadata_keys:
        if key in properties:
            metadata[key] = properties[key]

    return metadata


# =============================================================================
# SEMANTIC SEARCH — WEAVIATE
# =============================================================================

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }

    Output được sort theo score giảm dần.
    """
    if VECTOR_STORE != "weaviate":
        raise ValueError(
            f"File Task 5 này đang implement Weaviate. "
            f"VECTOR_STORE hiện tại: {VECTOR_STORE}"
        )

    import weaviate
    from weaviate.classes.query import MetadataQuery

    top_k = validate_top_k(top_k)
    query_embedding = embed_query(query)

    client = None

    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
        )

        if not client.is_ready():
            raise RuntimeError("Weaviate local chưa sẵn sàng.")

        if not client.collections.exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Không tìm thấy collection '{COLLECTION_NAME}'. "
                "Hãy chạy Task 4 trước để index dữ liệu."
            )

        collection = client.collections.use(COLLECTION_NAME)

        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        output = []

        for obj in results.objects:
            properties = dict(obj.properties or {})
            content = properties.get("content", "")

            distance = getattr(obj.metadata, "distance", None)
            score = distance_to_similarity(distance)

            output.append(
                {
                    "content": content,
                    "score": float(score),
                    "metadata": clean_metadata(properties),
                }
            )

        output.sort(key=lambda item: item["score"], reverse=True)
        return output

    finally:
        if client is not None:
            client.close()


# =============================================================================
# TEST CLI
# =============================================================================

def print_results(results: list[dict]):
    """In kết quả search ra terminal."""
    if not results:
        print("Không có kết quả.")
        return

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        content = result.get("content", "")

        print("=" * 80)
        print(f"Rank: {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {metadata.get('source_path') or metadata.get('source')}")
        print(f"Doc type: {metadata.get('doc_type')}")
        print(f"Chunk index: {metadata.get('chunk_index')}")
        print("-" * 80)
        print(content[:500].replace("\n", " "))
        print()


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    print_results(results)
