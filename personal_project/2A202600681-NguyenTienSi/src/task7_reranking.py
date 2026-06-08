"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker
    - MMR (Maximal Marginal Relevance)
    - RRF (Reciprocal Rank Fusion)

Lựa chọn chính trong project này:
    - Dùng RRF làm default vì:
        1. Không cần API key.
        2. Không cần tải model reranker lớn.
        3. Phù hợp để gộp semantic_search(Task 5) + lexical_search(Task 6).
        4. RRF chỉ dựa vào rank nên ổn định khi score của BM25 và dense search khác scale.

Cài đặt tối thiểu:
    pip install numpy

Nếu dùng cross_encoder:
    pip install sentence-transformers
"""

import hashlib
import os
import re
from functools import lru_cache
from typing import Any

from _console import configure_utf8_output

configure_utf8_output()


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_RERANK_METHOD = os.getenv("RERANK_METHOD", "rrf")

# Model nhỏ để test local. Có thể đổi bằng env:
#   set CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
# hoặc dùng model reranker multilingual khác nếu máy đủ tài nguyên.
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)

# RRF smoothing constant phổ biến: 60
RRF_K = int(os.getenv("RRF_K", "60"))

# MMR: lambda cao ưu tiên relevance; lambda thấp tăng diversity.
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))


# =============================================================================
# HELPERS
# =============================================================================

def validate_top_k(top_k: int, max_k: int | None = None) -> int:
    """Kiểm tra top_k."""
    if not isinstance(top_k, int):
        raise TypeError("top_k phải là int.")

    if top_k <= 0:
        raise ValueError("top_k phải > 0.")

    if max_k is not None:
        return min(top_k, max_k)

    return top_k


def normalize_score(value: Any) -> float:
    """Convert score về float an toàn."""
    try:
        return float(value)
    except Exception:
        return 0.0


def content_key(item: dict) -> str:
    """
    Tạo key ổn định để deduplicate candidates.

    Ưu tiên chunk_id nếu có. Nếu không có, dùng hash nội dung.
    """
    metadata = item.get("metadata", {}) or {}

    for key in ("chunk_id", "id", "content_hash"):
        value = metadata.get(key)
        if value:
            return str(value)

    content = item.get("content", "")
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Giới hạn text cho cross-encoder để tránh input quá dài."""
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính cosine similarity giữa 2 vector."""
    import numpy as np

    a = np.asarray(vec_a, dtype="float32")
    b = np.asarray(vec_b, dtype="float32")

    if a.size == 0 or b.size == 0:
        return 0.0

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize list score về [0, 1]."""
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [1.0 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """
    Gộp candidates trùng nhau.

    Nếu trùng content/chunk_id, giữ bản có score cao hơn.
    """
    best_by_key: dict[str, dict] = {}

    for item in candidates:
        key = content_key(item)
        current_score = normalize_score(item.get("score", 0.0))

        if key not in best_by_key:
            best_by_key[key] = item
            continue

        old_score = normalize_score(best_by_key[key].get("score", 0.0))
        if current_score > old_score:
            best_by_key[key] = item

    return list(best_by_key.values())


# =============================================================================
# CROSS-ENCODER RERANKING
# =============================================================================

@lru_cache(maxsize=1)
def get_cross_encoder_model():
    """
    Load CrossEncoder một lần.

    Cross-encoder nhận cặp [query, document] và trả relevance score.
    Chính xác hơn bi-encoder retrieval nhưng chậm hơn, nên chỉ dùng cho top candidates.
    """
    from sentence_transformers.cross_encoder import CrossEncoder

    print(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query không được rỗng.")

    if not candidates:
        return []

    candidates = deduplicate_candidates(candidates)
    top_k = validate_top_k(top_k, max_k=len(candidates))

    model = get_cross_encoder_model()

    pairs = [
        [query, truncate_text(candidate.get("content", ""))]
        for candidate in candidates
    ]

    scores = model.predict(pairs)
    scores = [normalize_score(score) for score in scores]

    reranked = []

    for candidate, rerank_score in zip(candidates, scores):
        item = candidate.copy()
        metadata = dict(item.get("metadata", {}) or {})

        item["original_score"] = normalize_score(candidate.get("score", 0.0))
        item["score"] = float(rerank_score)
        item["rerank_score"] = float(rerank_score)
        item["rerank_method"] = "cross_encoder"

        metadata["rerank_method"] = "cross_encoder"
        metadata["cross_encoder_model"] = CROSS_ENCODER_MODEL
        item["metadata"] = metadata

        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


# =============================================================================
# MMR RERANKING
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Công thức:
        MMR = λ * sim(query, doc) - (1 - λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {
            'content': str,
            'score': float,
            'embedding': list[float],
            'metadata': dict
        }
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not query_embedding:
        raise ValueError("query_embedding không được rỗng.")

    if not candidates:
        return []

    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param phải nằm trong [0, 1].")

    candidates = deduplicate_candidates(candidates)

    for candidate in candidates:
        if "embedding" not in candidate:
            raise ValueError(
                "MMR yêu cầu mỗi candidate có key 'embedding'. "
                "Nếu candidates lấy từ Weaviate không trả embedding, hãy dùng RRF hoặc cross_encoder."
            )

    top_k = validate_top_k(top_k, max_k=len(candidates))

    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    while remaining_indices and len(selected_indices) < top_k:
        best_idx = None
        best_mmr_score = float("-inf")
        best_relevance = 0.0
        best_diversity_penalty = 0.0

        for idx in remaining_indices:
            candidate_embedding = candidates[idx]["embedding"]

            relevance = cosine_sim(query_embedding, candidate_embedding)

            if not selected_indices:
                max_sim_to_selected = 0.0
            else:
                max_sim_to_selected = max(
                    cosine_sim(
                        candidate_embedding,
                        candidates[selected_idx]["embedding"],
                    )
                    for selected_idx in selected_indices
                )

            mmr_score = (
                lambda_param * relevance
                - (1.0 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
                best_relevance = relevance
                best_diversity_penalty = max_sim_to_selected

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        candidates[best_idx]["score"] = float(best_mmr_score)
        candidates[best_idx]["rerank_score"] = float(best_mmr_score)
        candidates[best_idx]["rerank_method"] = "mmr"
        candidates[best_idx]["mmr_relevance"] = float(best_relevance)
        candidates[best_idx]["mmr_diversity_penalty"] = float(best_diversity_penalty)

        metadata = dict(candidates[best_idx].get("metadata", {}) or {})
        metadata["rerank_method"] = "mmr"
        metadata["mmr_lambda"] = lambda_param
        candidates[best_idx]["metadata"] = metadata

    return [candidates[idx] for idx in selected_indices]


# =============================================================================
# RRF RERANKING
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Công thức:
        RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists.
            Ví dụ:
                [
                    semantic_results,  # từ Task 5
                    lexical_results,   # từ Task 6
                ]
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant. Default = 60.

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    if k <= 0:
        raise ValueError("k phải > 0.")

    top_k = validate_top_k(top_k)

    rrf_scores: dict[str, float] = {}
    item_map: dict[str, dict] = {}
    rank_sources: dict[str, list[dict]] = {}

    for list_index, ranked_list in enumerate(ranked_lists):
        if not ranked_list:
            continue

        # Đảm bảo mỗi list đầu vào đã sorted theo score giảm dần.
        sorted_list = sorted(
            ranked_list,
            key=lambda item: normalize_score(item.get("score", 0.0)),
            reverse=True,
        )

        seen_in_this_ranker = set()

        for rank, item in enumerate(sorted_list, start=1):
            key = content_key(item)

            # Không tính trùng 2 lần trong cùng 1 ranker.
            if key in seen_in_this_ranker:
                continue
            seen_in_this_ranker.add(key)

            contribution = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + contribution

            # Giữ item có original score cao nhất để preserve content/metadata tốt nhất.
            if key not in item_map:
                item_map[key] = item
            else:
                old_score = normalize_score(item_map[key].get("score", 0.0))
                new_score = normalize_score(item.get("score", 0.0))
                if new_score > old_score:
                    item_map[key] = item

            rank_sources.setdefault(key, []).append(
                {
                    "ranker_index": list_index,
                    "rank": rank,
                    "original_score": normalize_score(item.get("score", 0.0)),
                    "rrf_contribution": contribution,
                }
            )

    sorted_keys = sorted(
        rrf_scores.keys(),
        key=lambda key: rrf_scores[key],
        reverse=True,
    )

    results = []

    for key in sorted_keys[:top_k]:
        item = item_map[key].copy()
        metadata = dict(item.get("metadata", {}) or {})

        item["original_score"] = normalize_score(item.get("score", 0.0))
        item["score"] = float(rrf_scores[key])
        item["rerank_score"] = float(rrf_scores[key])
        item["rerank_method"] = "rrf"
        item["rank_sources"] = rank_sources.get(key, [])

        metadata["rerank_method"] = "rrf"
        metadata["rrf_k"] = k
        metadata["ranker_count"] = len(ranked_lists)
        item["metadata"] = metadata

        results.append(item)

    return results


# =============================================================================
# QUERY EMBEDDING HELPER FOR MMR
# =============================================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load embedding model giống Task 4/5 nếu cần MMR."""
    from sentence_transformers import SentenceTransformer

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    print(f"Loading embedding model for MMR: {embedding_model_name}")
    return SentenceTransformer(embedding_model_name)


def embed_query_for_mmr(query: str) -> list[float]:
    """Embed query để dùng với rerank_mmr."""
    query = query.strip()
    if not query:
        raise ValueError("Query không được rỗng.")

    model = get_embedding_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding.astype("float32").tolist()


# =============================================================================
# MAIN RERANK INTERFACE
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = DEFAULT_RERANK_METHOD,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates:
            - cross_encoder: list[dict]
            - mmr: list[dict] có embedding
            - rrf: list[list[dict]], ví dụ [semantic_results, lexical_results]
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" | "mmr" | "rrf"

    Returns:
        List of top_k reranked candidates.
    """
    method = method.lower().strip()

    if method == "cross_encoder":
        if candidates and isinstance(candidates[0], list):
            flat_candidates = []
            for ranked_list in candidates:
                flat_candidates.extend(ranked_list)
            candidates = flat_candidates

        return rerank_cross_encoder(query, candidates, top_k)

    if method == "mmr":
        if candidates and isinstance(candidates[0], list):
            raise ValueError("MMR cần list[dict], không phải list[list[dict]].")

        query_embedding = embed_query_for_mmr(query)
        return rerank_mmr(
            query_embedding=query_embedding,
            candidates=candidates,
            top_k=top_k,
            lambda_param=MMR_LAMBDA,
        )

    if method == "rrf":
        if not candidates:
            return []

        # Cho phép truyền list[dict]; khi đó coi như chỉ có 1 ranker.
        if candidates and isinstance(candidates[0], dict):
            ranked_lists = [candidates]
        else:
            ranked_lists = candidates

        return rerank_rrf(
            ranked_lists=ranked_lists,
            top_k=top_k,
            k=RRF_K,
        )

    raise ValueError(f"Unknown rerank method: {method}")


# =============================================================================
# OPTIONAL INTEGRATION HELPERS
# =============================================================================

def hybrid_rerank_with_rrf(query: str, top_k: int = 5, retrieve_k: int = 20) -> list[dict]:
    """
    Helper tích hợp Task 5 + Task 6:
        - semantic_search(query)
        - lexical_search(query)
        - rerank_rrf([semantic_results, lexical_results])

    Điều kiện:
        task5_semantic_search.py và task6_lexical_search.py nằm cùng thư mục.
    """
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search

    semantic_results = semantic_search(query, top_k=retrieve_k)
    lexical_results = lexical_search(query, top_k=retrieve_k)

    return rerank_rrf(
        ranked_lists=[semantic_results, lexical_results],
        top_k=top_k,
        k=RRF_K,
    )


def format_result(result: dict, max_chars: int = 300) -> str:
    """Format 1 result để in ra terminal."""
    metadata = result.get("metadata", {}) or {}
    content = result.get("content", "")
    content = re.sub(r"\s+", " ", content).strip()

    source = metadata.get("source_path") or metadata.get("source") or "unknown"
    doc_type = metadata.get("doc_type") or "unknown"
    chunk_index = metadata.get("chunk_index", "unknown")

    return (
        f"[{result.get('score', 0.0):.4f}] "
        f"source={source} | type={doc_type} | chunk={chunk_index}\n"
        f"{content[:max_chars]}..."
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    query = "hình phạt tàng trữ ma túy"

    dummy_semantic_results = [
        {
            "content": "Điều 249: Tội tàng trữ trái phép chất ma túy có thể bị phạt tù tùy khối lượng và tính chất hành vi.",
            "score": 0.88,
            "metadata": {"source": "bo_luat_hinh_su.md", "doc_type": "legal", "chunk_index": 1, "chunk_id": "a"},
        },
        {
            "content": "Nghị định quy định danh mục chất ma túy và tiền chất.",
            "score": 0.74,
            "metadata": {"source": "nghi_dinh_57.md", "doc_type": "legal", "chunk_index": 3, "chunk_id": "b"},
        },
        {
            "content": "Một bài báo đưa tin về nghệ sĩ liên quan đến ma túy.",
            "score": 0.68,
            "metadata": {"source": "article_01.md", "doc_type": "news", "chunk_index": 0, "chunk_id": "c"},
        },
    ]

    dummy_lexical_results = [
        {
            "content": "Điều 249: Tội tàng trữ trái phép chất ma túy có thể bị phạt tù tùy khối lượng và tính chất hành vi.",
            "score": 12.5,
            "metadata": {"source": "bo_luat_hinh_su.md", "doc_type": "legal", "chunk_index": 1, "chunk_id": "a"},
        },
        {
            "content": "Điều 248: Tội sản xuất trái phép chất ma túy.",
            "score": 9.2,
            "metadata": {"source": "bo_luat_hinh_su.md", "doc_type": "legal", "chunk_index": 2, "chunk_id": "d"},
        },
        {
            "content": "Luật phòng, chống ma túy quy định trách nhiệm phòng ngừa, phát hiện và xử lý hành vi liên quan.",
            "score": 7.1,
            "metadata": {"source": "luat_phong_chong_ma_tuy.md", "doc_type": "legal", "chunk_index": 5, "chunk_id": "e"},
        },
    ]

    print("=" * 80)
    print("Test RRF reranking")
    print("=" * 80)

    rrf_results = rerank(
        query=query,
        candidates=[dummy_semantic_results, dummy_lexical_results],
        top_k=3,
        method="rrf",
    )

    for result in rrf_results:
        print(format_result(result))
        print("-" * 80)

    print("\nGợi ý tích hợp thật sau khi Task 5 và Task 6 chạy ổn:")
    print("results = hybrid_rerank_with_rrf('hình phạt tàng trữ ma túy', top_k=5)")
