"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả bằng RRF
    3. Rerank
    4. Nếu top result confidence < threshold → fallback sang PageIndex
    5. Return top_k results

Cài đặt tối thiểu:
    pip install numpy

Điều kiện:
    - Task 4 đã index dữ liệu.
    - Task 5 semantic search chạy được.
    - Task 6 lexical search chạy được.
    - Task 7 reranking chạy được.
    - Task 8 PageIndex chỉ bắt buộc nếu muốn fallback thật.
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from _console import configure_utf8_output

configure_utf8_output()


# =============================================================================
# ROBUST IMPORTS
# =============================================================================
# Khi chạy theo package:
#   python -m personal_project.xxx.src.task9_retrieval_pipeline
# thì relative import hoạt động.
#
# Khi chạy trực tiếp:
#   python src/task9_retrieval_pipeline.py
# thì fallback import hoạt động.

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf


def _load_pageindex_search():
    """
    Import PageIndex fallback linh hoạt vì file Task 8 có thể được đặt tên:
        - task8_pageindex_vectorless.py
        - task8_pageindex.py
    """
    try:
        from .task8_pageindex_vectorless import pageindex_search
        return pageindex_search
    except Exception:
        pass

    try:
        from task8_pageindex_vectorless import pageindex_search
        return pageindex_search
    except Exception:
        pass

    try:
        from .task8_pageindex import pageindex_search
        return pageindex_search
    except Exception:
        pass

    try:
        from task8_pageindex import pageindex_search
        return pageindex_search
    except Exception:
        return None


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dùng confidence score đã normalize để so threshold.
# Không dùng raw RRF score vì RRF thường chỉ khoảng 0.01–0.05.
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.30"))

DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Nên dùng "rrf" mặc định vì không cần tải cross-encoder.
# Nếu muốn rerank mạnh hơn:
#   set RERANK_METHOD=cross_encoder
RERANK_METHOD = os.getenv("RERANK_METHOD", "rrf")  # "cross_encoder" | "mmr" | "rrf"

RETRIEVE_MULTIPLIER = int(os.getenv("RETRIEVE_MULTIPLIER", "4"))
MAX_RETRIEVE_K = int(os.getenv("MAX_RETRIEVE_K", "40"))

ENABLE_PAGEINDEX_FALLBACK = (
    os.getenv("ENABLE_PAGEINDEX_FALLBACK", "true").lower() == "true"
)

VERBOSE = os.getenv("RETRIEVAL_VERBOSE", "true").lower() == "true"


# =============================================================================
# HELPERS
# =============================================================================

def log(message: str):
    """In log nếu VERBOSE=true."""
    if VERBOSE:
        print(message)


def validate_query(query: str) -> str:
    """Validate query đầu vào."""
    query = str(query or "").strip()
    if not query:
        raise ValueError("Query không được rỗng.")
    return query


def validate_top_k(top_k: int) -> int:
    """Validate top_k."""
    if not isinstance(top_k, int):
        raise TypeError("top_k phải là int.")
    if top_k <= 0:
        raise ValueError("top_k phải > 0.")
    return min(top_k, 50)


def safe_score(value: Any) -> float:
    """Convert score về float an toàn."""
    try:
        return float(value)
    except Exception:
        return 0.0


def normalize_scores(results: list[dict], score_key: str = "score") -> list[dict]:
    """
    Thêm normalized_score vào từng result theo min-max normalization.

    Lý do:
        - BM25 score, cosine score, RRF score, cross-encoder score khác thang đo.
        - Threshold fallback cần một score cùng scale [0, 1].
    """
    if not results:
        return results

    scores = [safe_score(item.get(score_key, 0.0)) for item in results]
    min_score = min(scores)
    max_score = max(scores)

    output = []

    for item, score in zip(results, scores):
        new_item = item.copy()

        if max_score == min_score:
            normalized = 1.0 if score > 0 else 0.0
        else:
            normalized = (score - min_score) / (max_score - min_score)

        new_item["normalized_score"] = float(normalized)
        output.append(new_item)

    return output


def mark_results_source(results: list[dict], source: str) -> list[dict]:
    """Gắn source field vào mỗi result."""
    marked = []

    for item in results:
        new_item = item.copy()
        metadata = dict(new_item.get("metadata", {}) or {})

        new_item["source"] = source
        metadata["retrieval_pipeline_source"] = source
        new_item["metadata"] = metadata

        marked.append(new_item)

    return marked


def get_best_confidence(results: list[dict]) -> float:
    """
    Lấy confidence để quyết định fallback.

    Ưu tiên:
        - confidence
        - normalized_score
        - score
    """
    if not results:
        return 0.0

    best = results[0]

    if "confidence" in best:
        return safe_score(best["confidence"])

    if "normalized_score" in best:
        return safe_score(best["normalized_score"])

    return safe_score(best.get("score", 0.0))


def enrich_confidence(results: list[dict]) -> list[dict]:
    """
    Tạo confidence score [0, 1] sau rerank.

    Công thức đơn giản:
        confidence = normalized_score
        + boost nhỏ nếu item xuất hiện từ nhiều rankers qua rank_sources

    Không thay score gốc; chỉ thêm confidence để fallback logic dễ kiểm soát.
    """
    if not results:
        return results

    results = normalize_scores(results, score_key="score")
    enriched = []

    for item in results:
        new_item = item.copy()
        metadata = dict(new_item.get("metadata", {}) or {})

        normalized = safe_score(new_item.get("normalized_score", 0.0))

        rank_sources = new_item.get("rank_sources", []) or []
        source_count = len(rank_sources)

        # Nếu một chunk được cả dense và BM25 đưa lên, tăng confidence nhẹ.
        multi_source_boost = 0.10 if source_count >= 2 else 0.0

        confidence = min(1.0, normalized + multi_source_boost)

        new_item["confidence"] = float(confidence)
        metadata["confidence"] = float(confidence)
        metadata["normalized_score"] = normalized
        metadata["rank_source_count"] = source_count

        new_item["metadata"] = metadata
        enriched.append(new_item)

    enriched.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return enriched


def deduplicate_results(results: list[dict]) -> list[dict]:
    """
    Deduplicate theo chunk_id nếu có, nếu không theo content.
    Giữ result có score cao hơn.
    """
    best_by_key: dict[str, dict] = {}

    for item in results:
        metadata = item.get("metadata", {}) or {}
        key = (
            metadata.get("chunk_id")
            or metadata.get("content_hash")
            or item.get("content", "")
        )

        if not key:
            continue

        current_score = safe_score(item.get("score", 0.0))

        if key not in best_by_key:
            best_by_key[key] = item
            continue

        old_score = safe_score(best_by_key[key].get("score", 0.0))
        if current_score > old_score:
            best_by_key[key] = item

    return list(best_by_key.values())


# =============================================================================
# RETRIEVAL STEPS
# =============================================================================

def run_semantic_and_lexical(query: str, retrieve_k: int) -> tuple[list[dict], list[dict]]:
    """
    Chạy semantic_search và lexical_search song song.

    Nếu một nhánh lỗi, pipeline vẫn dùng nhánh còn lại.
    """
    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    tasks = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks[executor.submit(semantic_search, query, retrieve_k)] = "semantic"
        tasks[executor.submit(lexical_search, query, retrieve_k)] = "lexical"

        for future in as_completed(tasks):
            task_name = tasks[future]

            try:
                results = future.result() or []

                if task_name == "semantic":
                    dense_results = mark_results_source(results, "semantic")
                    log(f"✓ Semantic results: {len(dense_results)}")

                elif task_name == "lexical":
                    sparse_results = mark_results_source(results, "lexical")
                    log(f"✓ Lexical results: {len(sparse_results)}")

            except Exception as exc:
                log(f"⚠ {task_name} search failed: {exc}")
                log(traceback.format_exc())

    return dense_results, sparse_results


def merge_results_rrf(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Merge semantic + lexical bằng RRF.

    RRF hợp với hybrid retrieval vì:
        - Không cần cùng score scale giữa dense và BM25.
        - Dựa trên thứ hạng trong từng retriever.
    """
    ranked_lists = []

    if dense_results:
        ranked_lists.append(dense_results)

    if sparse_results:
        ranked_lists.append(sparse_results)

    if not ranked_lists:
        return []

    merged = rerank_rrf(ranked_lists, top_k=top_k)
    merged = mark_results_source(merged, "hybrid")
    merged = deduplicate_results(merged)

    merged.sort(key=lambda item: safe_score(item.get("score", 0.0)), reverse=True)
    return merged[:top_k]


def rerank_results(
    query: str,
    merged_results: list[dict],
    top_k: int,
    use_reranking: bool,
) -> list[dict]:
    """
    Rerank merged results.

    Nếu cross_encoder lỗi hoặc thiếu dependency, fallback về RRF/no-op.
    """
    if not merged_results:
        return []

    if not use_reranking:
        final_results = merged_results[:top_k]
        return enrich_confidence(final_results)

    method = RERANK_METHOD.lower().strip()

    try:
        if method == "rrf":
            # Vì đã merge bằng RRF rồi, bước này chỉ cắt top_k và normalize confidence.
            final_results = merged_results[:top_k]

        elif method == "cross_encoder":
            final_results = rerank(
                query=query,
                candidates=merged_results,
                top_k=top_k,
                method="cross_encoder",
            )
            final_results = mark_results_source(final_results, "hybrid_cross_encoder")

        elif method == "mmr":
            # MMR cần embedding trong candidates. Nếu Task 5 không trả embedding,
            # Task 7 sẽ raise ValueError và pipeline fallback.
            final_results = rerank(
                query=query,
                candidates=merged_results,
                top_k=top_k,
                method="mmr",
            )
            final_results = mark_results_source(final_results, "hybrid_mmr")

        else:
            raise ValueError(f"Unknown RERANK_METHOD: {RERANK_METHOD}")

    except Exception as exc:
        log(f"⚠ Reranking failed with method={method}: {exc}")
        log("↷ Fallback to RRF merged results.")
        final_results = merged_results[:top_k]

    return enrich_confidence(final_results)


def fallback_pageindex(query: str, top_k: int) -> list[dict]:
    """
    Fallback sang PageIndex nếu hybrid retrieval yếu.

    Nếu PageIndex chưa cấu hình API key hoặc module không tồn tại, return [].
    """
    if not ENABLE_PAGEINDEX_FALLBACK:
        log("↷ PageIndex fallback disabled.")
        return []

    pageindex_search = _load_pageindex_search()

    if pageindex_search is None:
        log("⚠ Không tìm thấy module PageIndex fallback.")
        return []

    try:
        fallback_results = pageindex_search(query, top_k=top_k) or []
        fallback_results = mark_results_source(fallback_results, "pageindex")
        fallback_results = enrich_confidence(fallback_results)
        log(f"✓ PageIndex fallback results: {len(fallback_results)}")
        return fallback_results[:top_k]

    except Exception as exc:
        log(f"⚠ PageIndex fallback failed: {exc}")
        log(traceback.format_exc())
        return []


# =============================================================================
# PUBLIC API
# =============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge bằng RRF → merged_results
          ├→ Rerank / normalize confidence → final_results
          │
          └→ If best confidence < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng confidence tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'confidence': float,
            'metadata': dict,
            'source': str
        }
    """
    query = validate_query(query)
    top_k = validate_top_k(top_k)

    retrieve_k = min(max(top_k * RETRIEVE_MULTIPLIER, top_k), MAX_RETRIEVE_K)

    log("=" * 70)
    log("Task 9 Retrieval Pipeline")
    log(f"Query: {query}")
    log(f"top_k={top_k}, retrieve_k={retrieve_k}, threshold={score_threshold}")
    log(f"rerank_method={RERANK_METHOD}, use_reranking={use_reranking}")
    log("=" * 70)

    # Step 1: semantic + lexical song song
    dense_results, sparse_results = run_semantic_and_lexical(query, retrieve_k)

    # Step 2: merge bằng RRF
    merged_results = merge_results_rrf(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=retrieve_k,
    )
    log(f"✓ Merged results: {len(merged_results)}")

    # Step 3: rerank
    final_results = rerank_results(
        query=query,
        merged_results=merged_results,
        top_k=top_k,
        use_reranking=use_reranking,
    )
    log(f"✓ Final hybrid results: {len(final_results)}")

    # Step 4: threshold → fallback
    best_confidence = get_best_confidence(final_results)

    if not final_results or best_confidence < score_threshold:
        log(
            f"⚠ Hybrid confidence={best_confidence:.3f} "
            f"< threshold={score_threshold:.3f}. Fallback → PageIndex"
        )

        fallback_results = fallback_pageindex(query, top_k=top_k)

        if fallback_results:
            return fallback_results[:top_k]

        log("↷ PageIndex fallback unavailable/empty. Return hybrid results anyway.")

    return final_results[:top_k]


# =============================================================================
# CLI TEST
# =============================================================================

def print_results(results: list[dict]):
    """In kết quả retrieval ra terminal."""
    if not results:
        print("Không có kết quả.")
        return

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {}) or {}
        content = str(result.get("content", "")).replace("\n", " ")

        print(f"  {i}. score={result.get('score', 0.0):.4f} "
              f"confidence={result.get('confidence', 0.0):.4f} "
              f"source={result.get('source')}")
        print(f"     file={metadata.get('source_path') or metadata.get('source') or metadata.get('filename')}")
        print(f"     type={metadata.get('doc_type')}")
        print(f"     text={content[:160]}...")
        print()


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy năm 2024",
        "Luật phòng chống ma túy 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {q}")
        print("-" * 80)

        results = retrieve(q, top_k=3)
        print_results(results)
