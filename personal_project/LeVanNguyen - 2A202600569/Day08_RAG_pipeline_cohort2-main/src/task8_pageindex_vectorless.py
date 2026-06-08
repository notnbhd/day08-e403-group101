"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pageindex import PageIndexClient
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        pi.submit_document(file_path=str(md_file))
        print(f"  ✓ Uploaded: {md_file.name}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    documents = pi.list_documents(limit=top_k)

    if isinstance(documents, dict):
        for key in ("documents", "data", "items", "results"):
            if key in documents and isinstance(documents[key], list):
                documents = documents[key]
                break

    if not isinstance(documents, list):
        return []

    results = []
    for doc in documents[:top_k]:
        doc_id = None
        if isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("id")

        if not doc_id:
            continue

        retrieval = pi.submit_query(doc_id=doc_id, query=query)
        retrieval_id = retrieval.get("retrieval_id") or retrieval.get("id")
        if not retrieval_id:
            continue

        retrieval_result = pi.get_retrieval(retrieval_id)
        content = None
        score = 0.0

        if isinstance(retrieval_result, dict):
            content = retrieval_result.get("content") or retrieval_result.get("text") or str(retrieval_result)
            score = retrieval_result.get("score", 0.0) if isinstance(retrieval_result.get("score", 0.0), (int, float)) else 0.0
        else:
            content = str(retrieval_result)

        results.append({
            "content": content,
            "score": float(score),
            "metadata": {"doc_id": doc_id},
            "source": "pageindex"
        })

    return results[:top_k]

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")