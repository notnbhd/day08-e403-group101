"""
Task 7 — Reranking Module.

Mặc định: cross-encoder rerank qua OpenRouter (cohere/rerank-v3.5).
Kèm RRF (Reciprocal Rank Fusion) và MMR để dùng/giải thích khi cần.
"""

import math

import requests

try:
    from .config import settings, openrouter_headers
except ImportError:  # chạy trực tiếp: python src/task7_reranking.py
    from config import settings, openrouter_headers

RERANK_ENDPOINT = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/rerank"


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng cross-encoder cohere/rerank-v3.5 (qua OpenRouter).

    Args:
        query: Câu truy vấn.
        candidates: List of {'content': str, 'score': float, 'metadata': dict}.
        top_k: Số kết quả sau rerank.

    Returns:
        Top_k candidates, 'score' = relevance_score, sorted descending.
    """
    if not candidates:
        return []

    documents = [c["content"] for c in candidates]
    payload = {
        "model": settings.RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }
    resp = requests.post(
        RERANK_ENDPOINT, headers=openrouter_headers(), json=payload, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()

    reranked = []
    for item in data.get("results", []):
        idx = item["index"]
        score = item.get("relevance_score", item.get("score", 0.0))
        merged = {**candidates[idx], "score": float(score)}
        reranked.append(merged)

    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked[:top_k]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — cân bằng relevance và diversity.
    MMR = λ·sim(query, doc) − (1−λ)·max sim(doc, đã chọn).
    Yêu cầu mỗi candidate có key 'embedding'.
    """
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            relevance = _cosine_sim(query_embedding, emb)
            max_sim = max(
                (_cosine_sim(emb, candidates[s].get("embedding", [])) for s in selected),
                default=0.0,
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp nhiều ranked list.
    RRF(d) = Σ 1/(k + rank_r(d)). k=60 theo Cormack et al. 2009.
    Khử trùng lặp theo 'content'.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item)

    ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in ordered[:top_k]:
        item = dict(content_map[content])
        item["score"] = score
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Giao diện rerank thống nhất (mặc định cross-encoder cohere/rerank-v3.5)."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Gọi rerank_mmr trực tiếp với query_embedding")
    elif method == "rrf":
        raise NotImplementedError("Gọi rerank_rrf trực tiếp với nhiều ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
