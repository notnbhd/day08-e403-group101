"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install -U pageindex python-dotenv requests

File .env:
    PAGEINDEX_API_KEY=your_api_key_here

Ghi chú implementation:
    - PageIndex SDK submit_document() dùng cho PDF và trả doc_id.
    - PageIndex REST /markdown/ dùng cho Markdown tree structure.
    - pageindex_search() ưu tiên legacy retrieval API với PDF doc_id.
    - Nếu chỉ có Markdown tree, search fallback dùng structural tree matching local
      trên tree do PageIndex API tạo, không dùng embedding/vector store.
"""

import json
import os
import re
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _console import configure_utf8_output
import requests
from dotenv import load_dotenv


configure_utf8_output()
load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).parent.parent

STANDARDIZED_DIR = BASE_DIR / "data" / "standardized"
LANDING_DIR = BASE_DIR / "data" / "landing"
PAGEINDEX_DIR = BASE_DIR / "data" / "pageindex"

MANIFEST_PATH = PAGEINDEX_DIR / "pageindex_manifest.json"

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_API_BASE = os.getenv("PAGEINDEX_API_BASE", "https://api.pageindex.ai").rstrip("/")

# Upload mode:
#   markdown_only: upload .md files to /markdown/
#   pdf_only: upload original .pdf files to /doc/
#   both: upload both
PAGEINDEX_UPLOAD_MODE = os.getenv("PAGEINDEX_UPLOAD_MODE", "both")

# Polling config for PDF processing and retrieval tasks
POLL_INTERVAL_SECONDS = float(os.getenv("PAGEINDEX_POLL_INTERVAL_SECONDS", "5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("PAGEINDEX_POLL_TIMEOUT_SECONDS", "300"))

# Legacy retrieval has optional thinking flag.
PAGEINDEX_RETRIEVAL_THINKING = os.getenv("PAGEINDEX_RETRIEVAL_THINKING", "false").lower() == "true"


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs():
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)


def require_api_key():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Thiếu PAGEINDEX_API_KEY. Hãy tạo file .env và thêm:\n"
            "PAGEINDEX_API_KEY=your_api_key_here"
        )


def file_fingerprint(path: Path) -> str:
    """
    Fingerprint dựa trên path + size + modified time.
    Dùng để tránh upload lại cùng một file nhiều lần.
    """
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "documents": [],
        }

    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict):
    ensure_dirs()
    manifest["updated_at"] = now_iso()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def request_json(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    files: dict | None = None,
    data: dict | None = None,
    json_body: dict | None = None,
    params: dict | None = None,
    timeout: int = 120,
) -> dict:
    """
    Thin wrapper cho requests với error message dễ debug.
    """
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        files=files,
        data=data,
        json=json_body,
        params=params,
        timeout=timeout,
    )

    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text}

    if not response.ok:
        raise RuntimeError(
            f"PageIndex API error {response.status_code} at {url}\n"
            f"Response: {json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    return payload


def find_existing_entry(manifest: dict, fingerprint: str) -> dict | None:
    for item in manifest.get("documents", []):
        if item.get("fingerprint") == fingerprint:
            return item
    return None


def iter_markdown_files() -> list[Path]:
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {STANDARDIZED_DIR}. Hãy chạy Task 3 trước."
        )

    return sorted(
        p for p in STANDARDIZED_DIR.rglob("*.md")
        if p.is_file() and p.name != "manifest_standardized.md"
    )


def iter_pdf_files() -> list[Path]:
    """
    PageIndex document processing upload chính thức dùng PDF.
    Ưu tiên PDF gốc từ data/landing/legal/.
    """
    if not LANDING_DIR.exists():
        return []

    return sorted(p for p in LANDING_DIR.rglob("*.pdf") if p.is_file())


# =============================================================================
# MARKDOWN TREE UPLOAD
# =============================================================================

def upload_markdown_to_pageindex(md_file: Path) -> dict:
    """
    Upload Markdown lên PageIndex Markdown Processing API.

    Endpoint này trả tree structure, không trả doc_id persistent.
    Vì vậy kết quả được lưu local vào manifest để dùng fallback vectorless search.
    """
    require_api_key()

    url = f"{PAGEINDEX_API_BASE}/markdown/"
    headers = {"api_key": PAGEINDEX_API_KEY}

    data = {
        "if_add_node_id": "yes",
        "if_add_node_summary": "yes",
        "if_add_node_text": "yes",
        "if_add_doc_description": "yes",
    }

    with md_file.open("rb") as f:
        result = request_json(
            "POST",
            url,
            headers=headers,
            files={"file": (md_file.name, f, "text/markdown")},
            data=data,
            timeout=180,
        )

    return result


def make_markdown_entry(md_file: Path, result: dict) -> dict:
    relative_path = md_file.relative_to(STANDARDIZED_DIR)
    doc_type = "legal" if "legal" in relative_path.parts else "news" if "news" in relative_path.parts else "unknown"

    return {
        "kind": "markdown_tree",
        "status": "completed" if result.get("success") else "failed",
        "filename": md_file.name,
        "source_path": str(relative_path).replace("\\", "/"),
        "absolute_path": str(md_file),
        "doc_type": doc_type,
        "fingerprint": file_fingerprint(md_file),
        "uploaded_at": now_iso(),
        "pageindex_doc_name": result.get("doc_name"),
        "structure": result.get("structure", []),
        "raw_response": result,
    }


# =============================================================================
# PDF UPLOAD + PROCESSING
# =============================================================================

def upload_pdf_to_pageindex(pdf_file: Path) -> dict:
    """
    Upload PDF bằng PageIndex Python SDK.
    SDK trả {"doc_id": "..."}.
    """
    require_api_key()

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    result = client.submit_document(str(pdf_file))

    if "doc_id" not in result:
        raise RuntimeError(f"Không nhận được doc_id từ PageIndex: {result}")

    return result


def get_tree_status(doc_id: str) -> dict:
    """
    Check PDF processing status và tree result.

    Dùng REST vì endpoint trả thêm retrieval_ready.
    """
    require_api_key()

    url = f"{PAGEINDEX_API_BASE}/doc/{doc_id}/"
    headers = {"api_key": PAGEINDEX_API_KEY}

    return request_json(
        "GET",
        url,
        headers=headers,
        params={"type": "tree", "summary": "true"},
        timeout=120,
    )


def wait_until_document_ready(doc_id: str) -> dict:
    """
    Poll PageIndex đến khi PDF tree completed hoặc timeout.
    """
    start = time.time()

    while True:
        result = get_tree_status(doc_id)
        status = result.get("status")
        retrieval_ready = bool(result.get("retrieval_ready"))

        if status == "completed":
            return result

        if status == "failed":
            raise RuntimeError(f"PageIndex processing failed for doc_id={doc_id}: {result}")

        elapsed = time.time() - start
        if elapsed > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Timeout chờ PageIndex processing doc_id={doc_id}. "
                f"Last status: {status}, retrieval_ready={retrieval_ready}"
            )

        print(f"  ... waiting doc_id={doc_id}, status={status}, retrieval_ready={retrieval_ready}")
        time.sleep(POLL_INTERVAL_SECONDS)


def make_pdf_entry(pdf_file: Path, submit_result: dict, tree_result: dict) -> dict:
    relative_path = pdf_file.relative_to(LANDING_DIR)
    doc_type = "legal" if "legal" in relative_path.parts else "news" if "news" in relative_path.parts else "unknown"

    return {
        "kind": "pdf_document",
        "status": tree_result.get("status", "unknown"),
        "retrieval_ready": bool(tree_result.get("retrieval_ready")),
        "filename": pdf_file.name,
        "source_path": str(relative_path).replace("\\", "/"),
        "absolute_path": str(pdf_file),
        "doc_type": doc_type,
        "fingerprint": file_fingerprint(pdf_file),
        "uploaded_at": now_iso(),
        "doc_id": submit_result["doc_id"],
        "structure": tree_result.get("result", []),
        "raw_submit_response": submit_result,
    }


# =============================================================================
# UPLOAD ALL DOCUMENTS
# =============================================================================

def upload_documents():
    """
    Upload documents lên PageIndex.

    - Markdown files:
        data/standardized/**/*.md → /markdown/
        Trả tree structure, lưu vào local manifest.

    - PDF files:
        data/landing/**/*.pdf → SDK submit_document()
        Trả doc_id, có thể dùng legacy retrieval API hoặc Chat API.

    Manifest:
        data/pageindex/pageindex_manifest.json
    """
    require_api_key()
    ensure_dirs()

    manifest = load_manifest()
    uploaded = 0
    skipped = 0
    failed = 0

    upload_markdown = PAGEINDEX_UPLOAD_MODE in {"markdown_only", "both"}
    upload_pdf = PAGEINDEX_UPLOAD_MODE in {"pdf_only", "both"}

    print(f"PAGEINDEX_UPLOAD_MODE={PAGEINDEX_UPLOAD_MODE}")

    if upload_markdown:
        md_files = iter_markdown_files()
        print(f"Found {len(md_files)} markdown files in {STANDARDIZED_DIR}")

        for md_file in md_files:
            fp = file_fingerprint(md_file)
            existing = find_existing_entry(manifest, fp)

            if existing and existing.get("status") == "completed":
                print(f"↷ Skip existing Markdown tree: {md_file.name}")
                skipped += 1
                continue

            try:
                print(f"Uploading Markdown: {md_file.name}")
                result = upload_markdown_to_pageindex(md_file)
                entry = make_markdown_entry(md_file, result)

                manifest["documents"].append(entry)
                save_manifest(manifest)

                print(f"  ✓ Markdown tree saved: {md_file.name}")
                uploaded += 1

            except Exception as exc:
                failed += 1
                print(f"  ✗ Failed Markdown: {md_file.name} | {exc}")

    if upload_pdf:
        pdf_files = iter_pdf_files()
        print(f"Found {len(pdf_files)} PDF files in {LANDING_DIR}")

        for pdf_file in pdf_files:
            fp = file_fingerprint(pdf_file)
            existing = find_existing_entry(manifest, fp)

            if existing and existing.get("kind") == "pdf_document" and existing.get("status") == "completed":
                print(f"↷ Skip existing PDF doc: {pdf_file.name}")
                skipped += 1
                continue

            try:
                print(f"Uploading PDF: {pdf_file.name}")
                submit_result = upload_pdf_to_pageindex(pdf_file)
                doc_id = submit_result["doc_id"]
                print(f"  doc_id={doc_id}")

                tree_result = wait_until_document_ready(doc_id)
                entry = make_pdf_entry(pdf_file, submit_result, tree_result)

                manifest["documents"].append(entry)
                save_manifest(manifest)

                print(f"  ✓ PDF uploaded and processed: {pdf_file.name}")
                uploaded += 1

            except Exception as exc:
                failed += 1
                print(f"  ✗ Failed PDF: {pdf_file.name} | {exc}")

    print("=" * 70)
    print(f"Uploaded: {uploaded}")
    print(f"Skipped:  {skipped}")
    print(f"Failed:   {failed}")
    print(f"Manifest: {MANIFEST_PATH}")


# =============================================================================
# PAGEINDEX RETRIEVAL API FOR PDF DOC_ID
# =============================================================================

def submit_retrieval(doc_id: str, query: str, thinking: bool = False) -> str:
    """
    Submit query vào legacy retrieval API.
    Trả retrieval_id.
    """
    require_api_key()

    url = f"{PAGEINDEX_API_BASE}/retrieval/"
    headers = {"api_key": PAGEINDEX_API_KEY}

    payload = {
        "doc_id": doc_id,
        "query": query,
        "thinking": thinking,
    }

    result = request_json(
        "POST",
        url,
        headers=headers,
        json_body=payload,
        timeout=120,
    )

    retrieval_id = result.get("retrieval_id")
    if not retrieval_id:
        raise RuntimeError(f"Không nhận được retrieval_id: {result}")

    return retrieval_id


def get_retrieval_result(retrieval_id: str) -> dict:
    require_api_key()

    url = f"{PAGEINDEX_API_BASE}/retrieval/{retrieval_id}/"
    headers = {"api_key": PAGEINDEX_API_KEY}

    return request_json("GET", url, headers=headers, timeout=120)


def wait_until_retrieval_ready(retrieval_id: str) -> dict:
    start = time.time()

    while True:
        result = get_retrieval_result(retrieval_id)
        status = result.get("status")

        if status == "completed":
            return result

        if status == "failed":
            raise RuntimeError(f"Retrieval failed: {result}")

        elapsed = time.time() - start
        if elapsed > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Timeout chờ retrieval_id={retrieval_id}. Last status={status}"
            )

        time.sleep(POLL_INTERVAL_SECONDS)


def parse_retrieval_result(result: dict, doc_entry: dict, base_score: float = 1.0) -> list[dict]:
    """
    Convert PageIndex retrieved_nodes thành format chung:
        content, score, metadata, source
    """
    output = []

    retrieved_nodes = result.get("retrieved_nodes", []) or []

    rank = 0
    for node in retrieved_nodes:
        title = node.get("title", "")
        node_id = node.get("node_id", "")
        relevant_contents = node.get("relevant_contents", []) or []

        for content_item in relevant_contents:
            rank += 1

            text = (
                content_item.get("relevant_content")
                or content_item.get("text")
                or content_item.get("markdown")
                or ""
            )

            if not text:
                continue

            score = base_score / rank

            output.append(
                {
                    "content": text,
                    "score": float(score),
                    "metadata": {
                        "retrieval_id": result.get("retrieval_id"),
                        "doc_id": result.get("doc_id") or doc_entry.get("doc_id"),
                        "filename": doc_entry.get("filename"),
                        "source_path": doc_entry.get("source_path"),
                        "doc_type": doc_entry.get("doc_type"),
                        "title": title,
                        "node_id": node_id,
                        "page_index": content_item.get("page_index"),
                        "retrieval_source": "pageindex_legacy_retrieval",
                    },
                    "source": "pageindex",
                }
            )

    return output


# =============================================================================
# LOCAL STRUCTURAL TREE FALLBACK FOR MARKDOWN TREE
# =============================================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("ma tuý", "ma túy")
    text = text.replace("ma tuy", "ma túy")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return re.findall(r"[a-zà-ỹđ0-9]+(?:[/-][a-zà-ỹđ0-9]+)*", text, flags=re.IGNORECASE)


def flatten_tree(
    nodes: list[dict],
    *,
    doc_entry: dict,
    parent_titles: list[str] | None = None,
) -> list[dict]:
    """
    Flatten hierarchical tree thành node list.
    """
    parent_titles = parent_titles or []
    flattened = []

    for node in nodes or []:
        title = str(node.get("title", "") or "")
        summary = str(node.get("summary", "") or "")
        text = str(node.get("text", "") or "")

        path_titles = parent_titles + ([title] if title else [])

        content_parts = []
        if title:
            content_parts.append(f"# {title}")
        if summary:
            content_parts.append(f"Summary: {summary}")
        if text:
            content_parts.append(text)

        content = "\n\n".join(content_parts).strip()

        if content:
            flattened.append(
                {
                    "content": content,
                    "metadata": {
                        "filename": doc_entry.get("filename"),
                        "source_path": doc_entry.get("source_path"),
                        "doc_type": doc_entry.get("doc_type"),
                        "pageindex_doc_name": doc_entry.get("pageindex_doc_name"),
                        "title": title,
                        "node_id": node.get("node_id"),
                        "line_num": node.get("line_num"),
                        "tree_path": " > ".join(path_titles),
                        "retrieval_source": "pageindex_markdown_tree_local",
                    },
                }
            )

        child_nodes = node.get("nodes", []) or []
        if child_nodes:
            flattened.extend(
                flatten_tree(
                    child_nodes,
                    doc_entry=doc_entry,
                    parent_titles=path_titles,
                )
            )

    return flattened


def score_tree_node(query: str, node_content: str) -> float:
    """
    Vectorless fallback scoring trên tree:
        - Không embedding.
        - Không vector store.
        - Dựa trên overlap token giữa query và node text/title/summary.
    """
    q_tokens = tokenize(query)
    d_tokens = tokenize(node_content)

    if not q_tokens or not d_tokens:
        return 0.0

    q_set = set(q_tokens)
    d_set = set(d_tokens)

    overlap = q_set.intersection(d_set)
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(q_set)

    # Boost exact phrase nhẹ.
    exact_boost = 0.2 if normalize_text(query) in normalize_text(node_content) else 0.0

    # Boost nếu từ xuất hiện ở title/header.
    header_text = "\n".join(
        line for line in node_content.splitlines()
        if line.strip().startswith("#")
    )
    header_tokens = set(tokenize(header_text))
    header_overlap = len(q_set.intersection(header_tokens)) / max(len(q_set), 1)

    return float(coverage + exact_boost + 0.3 * header_overlap)


def search_markdown_trees(query: str, manifest: dict, top_k: int) -> list[dict]:
    """
    Search fallback trên PageIndex Markdown tree đã upload/convert.
    """
    candidates = []

    for entry in manifest.get("documents", []):
        if entry.get("kind") != "markdown_tree" or entry.get("status") != "completed":
            continue

        nodes = flatten_tree(entry.get("structure", []), doc_entry=entry)

        for node in nodes:
            score = score_tree_node(query, node["content"])
            if score <= 0:
                continue

            candidates.append(
                {
                    "content": node["content"],
                    "score": score,
                    "metadata": node["metadata"],
                    "source": "pageindex",
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


# =============================================================================
# PUBLIC SEARCH FUNCTION
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }

    Cơ chế:
        1. Nếu có PDF doc_id đã processed: gọi PageIndex legacy retrieval API.
        2. Nếu có Markdown tree: fallback search trên tree structure do PageIndex API tạo.
        3. Merge và sort descending theo score.
    """
    require_api_key()

    query = query.strip()
    if not query:
        raise ValueError("Query không được rỗng.")

    if top_k <= 0:
        raise ValueError("top_k phải > 0.")

    manifest = load_manifest()
    results = []

    # 1. Retrieval API cho PDF docs có doc_id.
    pdf_entries = [
        item for item in manifest.get("documents", [])
        if item.get("kind") == "pdf_document"
        and item.get("status") == "completed"
        and item.get("retrieval_ready")
        and item.get("doc_id")
    ]

    for entry_index, entry in enumerate(pdf_entries, start=1):
        try:
            retrieval_id = submit_retrieval(
                doc_id=entry["doc_id"],
                query=query,
                thinking=PAGEINDEX_RETRIEVAL_THINKING,
            )
            retrieval_result = wait_until_retrieval_ready(retrieval_id)

            # Slight score offset so earlier docs do not all tie exactly.
            parsed = parse_retrieval_result(
                retrieval_result,
                doc_entry=entry,
                base_score=1.0 / entry_index,
            )
            results.extend(parsed)

        except Exception as exc:
            print(f"⚠ Retrieval failed for {entry.get('filename')}: {exc}")

    # 2. Markdown tree fallback.
    markdown_results = search_markdown_trees(query, manifest, top_k=top_k * 3)
    results.extend(markdown_results)

    # 3. Sort + deduplicate by content.
    dedup = {}
    for item in results:
        key = hashlib.sha1(item["content"].encode("utf-8")).hexdigest()
        if key not in dedup or item["score"] > dedup[key]["score"]:
            dedup[key] = item

    merged = list(dedup.values())
    merged.sort(key=lambda item: item["score"], reverse=True)

    return merged[:top_k]


# =============================================================================
# OPTIONAL: PAGEINDEX CHAT ANSWER
# =============================================================================

def pageindex_chat_answer(query: str, doc_ids: list[str] | None = None) -> str:
    """
    Dùng Chat API để lấy câu trả lời trực tiếp.
    Hàm này không thay thế pageindex_search(), nhưng hữu ích để demo.
    """
    require_api_key()

    url = f"{PAGEINDEX_API_BASE}/chat/completions"
    headers = {
        "api_key": PAGEINDEX_API_KEY,
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "temperature": 0.0,
        "enable_citations": True,
    }

    if doc_ids:
        payload["doc_id"] = doc_ids[0] if len(doc_ids) == 1 else doc_ids

    result = request_json(
        "POST",
        url,
        headers=headers,
        json_body=payload,
        timeout=180,
    )

    return result["choices"][0]["message"]["content"]


# =============================================================================
# CLI
# =============================================================================

def print_results(results: list[dict]):
    if not results:
        print("Không có kết quả.")
        return

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        content = re.sub(r"\s+", " ", result.get("content", "")).strip()

        print("=" * 80)
        print(f"Rank: {i}")
        print(f"Score: {result.get('score', 0.0):.4f}")
        print(f"Source: {result.get('source')}")
        print(f"File: {metadata.get('source_path') or metadata.get('filename')}")
        print(f"Doc type: {metadata.get('doc_type')}")
        print(f"Node/Page: {metadata.get('node_id') or metadata.get('page_index') or metadata.get('line_num')}")
        print("-" * 80)
        print(content[:700])
        print()


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký/lấy API key tại PageIndex Developer Dashboard.")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma túy", top_k=3)
        print_results(results)
