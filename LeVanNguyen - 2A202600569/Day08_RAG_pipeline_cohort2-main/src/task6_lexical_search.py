"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""
from rank_bm25 import BM25Okapi
from pathlib import Path
import numpy as np

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
bm25 = None  # Khởi tạo biến toàn cục

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.
    """
    global bm25
    if bm25 is None:
        raise ValueError("BM25 index chưa được khởi tạo. Hãy gọi build_bm25_index trước.")
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                 "content": CORPUS[idx]["content"],
                 "score": float(scores[idx]),
                 "metadata": CORPUS[idx]["metadata"]
            })
    return results

if __name__ == "__main__":
    # Ví dụ corpus mẫu
    CORPUS = [
        {"content": "Điều 248 tàng trữ trái phép chất ma tuý", "metadata": {"id": 1}},
        {"content": "Điều 249 vận chuyển trái phép chất ma tuý", "metadata": {"id": 2}},
    ]
    bm25 = build_bm25_index(CORPUS)

    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")