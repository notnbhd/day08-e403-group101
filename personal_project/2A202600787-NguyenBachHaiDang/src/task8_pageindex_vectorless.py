"""
Task 8 — PageIndex Vectorless RAG.

PageIndex biến document thành cây cấu trúc (tree index) và retrieve bằng
reasoning thay vì vector similarity → vectorless RAG. Dùng làm fallback cho
hybrid search ở Task 9.

SDK: https://docs.pageindex.ai  (pip install pageindex)
    from pageindex import PageIndexClient
    pi = PageIndexClient(api_key=...)
    doc_id = pi.submit_document("file.pdf")["doc_id"]
    pi.get_document(doc_id)["status"] == "completed"
    rid = pi.submit_query(doc_id, query)["retrieval_id"]
    pi.get_retrieval(rid)  # -> {'status': 'completed', 'retrieval': [...]}

Chạy (upload + test):
    uv run python -m src.task8_pageindex_vectorless
"""

import json
import time
from pathlib import Path

try:
    from .config import settings
except ImportError:  # chạy trực tiếp: python src/task8_pageindex_vectorless.py
    from config import settings

LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_CACHE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

# Tương thích ngược: test cũ tham chiếu biến này.
PAGEINDEX_API_KEY = settings.PAGEINDEX_API_KEY


def _client():
    from pageindex import PageIndexClient

    if not settings.PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa set trong .env")
    return PageIndexClient(api_key=settings.PAGEINDEX_API_KEY)


def _api_error():
    """PageIndexAPIError (import an toàn dù SDK đổi vị trí)."""
    try:
        from pageindex import PageIndexAPIError
    except ImportError:
        from pageindex.client import PageIndexAPIError
    return PageIndexAPIError


def _load_cache() -> dict:
    if DOC_CACHE.exists():
        return json.loads(DOC_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict):
    DOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DOC_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents(timeout: int = 300) -> dict:
    """
    Upload các PDF pháp luật lên PageIndex, đợi xử lý xong, cache lại doc_id.
    Returns: {filename: doc_id}
    """
    pi = _client()
    cache = _load_cache()
    APIError = _api_error()

    # Upload file NHỎ NHẤT trước → chắc chắn lọt qua quota PageIndex free-tier
    # (Bộ luật Hình sự rất lớn, để cuối). Chỉ cần ≥1 doc là Task 8 chạy được.
    pdfs = sorted(LEGAL_DIR.glob("*.pdf"), key=lambda p: p.stat().st_size) if LEGAL_DIR.exists() else []
    if not pdfs:
        print("⚠ Không có PDF trong data/landing/legal/. Chạy Task 1 trước.")
        return cache

    for pdf in pdfs:
        if pdf.name in cache:
            print(f"  ↪ Đã upload: {pdf.name} → {cache[pdf.name]}")
            continue
        print(f"  ↑ Uploading: {pdf.name} ({pdf.stat().st_size:,} bytes)")
        try:
            res = pi.submit_document(str(pdf))
        except APIError as e:
            if "LimitReached" in str(e):
                print(f"  ⚠ Hết quota PageIndex, bỏ qua {pdf.name}.")
                continue
            raise
        doc_id = res["doc_id"] if isinstance(res, dict) else res
        cache[pdf.name] = doc_id
        _save_cache(cache)

    if not cache:
        print("⚠ Không upload được doc nào (quota?). Xoá bớt doc trên dashboard "
              "PageIndex rồi chạy lại, hoặc dùng PDF nhỏ hơn.")

    # Đợi tất cả xử lý xong.
    deadline = time.time() + timeout
    for name, doc_id in cache.items():
        while time.time() < deadline:
            try:
                status = pi.get_document(doc_id).get("status")
            except Exception:
                status = None
            if status == "completed":
                print(f"  ✓ Ready: {name}")
                break
            time.sleep(5)
    return cache


def _extract_nodes(result: dict) -> list[dict]:
    """Lấy danh sách node từ result get_retrieval (chịu được nhiều shape)."""
    if not isinstance(result, dict):
        return result if isinstance(result, list) else []
    for key in ("retrieval", "retrieved_nodes", "nodes", "sources", "results"):
        val = result.get(key)
        if isinstance(val, list):
            return val
    # đôi khi nằm lồng trong "result"
    nested = result.get("result")
    if isinstance(nested, dict):
        return _extract_nodes(nested)
    if isinstance(nested, list):
        return nested
    return []


def _node_text(node: dict) -> str:
    """PageIndex node có thể chứa text ở nhiều key khác nhau."""
    for key in ("relevant_contents", "contents", "text", "content", "title"):
        val = node.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list):  # relevant_contents đôi khi là list
            joined = "\n".join(str(x) for x in val if x)
            if joined.strip():
                return joined
    return ""


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex (fallback của hybrid search).

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}
    """
    pi = _client()
    cache = _load_cache()
    if not cache:
        cache = upload_documents()

    results: list[dict] = []
    for name, doc_id in cache.items():
        try:
            submit = pi.submit_query(doc_id, query)
            rid = submit["retrieval_id"] if isinstance(submit, dict) else submit

            # Poll kết quả.
            res = None
            for _ in range(30):
                res = pi.get_retrieval(rid)
                if not isinstance(res, dict) or res.get("status") == "completed":
                    break
                time.sleep(3)

            for rank, node in enumerate(_extract_nodes(res)):
                if not isinstance(node, dict):
                    continue
                content = _node_text(node)
                if not content:
                    continue
                # PageIndex là reasoning-based, có thể không trả score →
                # suy ra score giảm dần theo thứ hạng để giữ format sorted.
                score = node.get("relevance_score") or node.get("score")
                score = float(score) if score is not None else 1.0 / (rank + 1)
                results.append(
                    {
                        "content": content,
                        "score": float(score),
                        "metadata": {"source": name, "type": "legal",
                                     "node_id": node.get("node_id")},
                        "source": "pageindex",
                    }
                )
        except Exception as e:
            print(f"  ✗ PageIndex query lỗi ({name}): {e}")
            continue

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not settings.PAGEINDEX_API_KEY:
        print("⚠ Set PAGEINDEX_API_KEY trong .env (đăng ký: https://pageindex.ai/)")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        for r in pageindex_search("hình phạt sử dụng ma tuý", top_k=3):
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
