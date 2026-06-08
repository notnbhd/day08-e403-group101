"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""
import os
from typing import Optional

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def _cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import numpy as np  # type: ignore[import]

    vec1_np = np.array(vec1, dtype=float)
    vec2_np = np.array(vec2, dtype=float)
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


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
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY environment variable not set")

    import requests  # type: ignore[import]

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
    )
    response.raise_for_status()
    reranked = response.json()["results"]
    return [
        {**candidates[r["index"]], "score": r["relevance_score"]} for r in reranked
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_similarity(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0
            for sel_idx in selected:
                sim = _cosine_similarity(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results

# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")

if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
