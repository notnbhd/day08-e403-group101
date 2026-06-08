"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25 local bằng rank-bm25.

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Cài đặt:
    pip install rank-bm25 langchain-text-splitters numpy

Nếu muốn dùng Weaviate BM25 built-in:
    pip install weaviate-client
    set LEXICAL_BACKEND=weaviate_bm25
"""

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from _console import configure_utf8_output

configure_utf8_output()


# =============================================================================
# CONFIGURATION — khớp với Task 4
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = BASE_DIR / "data" / "standardized"

# Mặc định dùng BM25 local bằng rank-bm25.
# Đổi thành "weaviate_bm25" nếu muốn demo Weaviate BM25 built-in.
LEXICAL_BACKEND = os.getenv("LEXICAL_BACKEND", "local_bm25")

# Chunking config nên khớp Task 4 để lexical search và semantic search
# cùng làm việc trên đơn vị chunk tương đương.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 80

# Weaviate config nếu dùng LEXICAL_BACKEND="weaviate_bm25"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "LegalNewsDocs")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))


# Global cache đơn giản cho skeleton ban đầu
CORPUS: list[dict] = []
BM25_INDEX = None


# =============================================================================
# TEXT NORMALIZATION / TOKENIZATION
# =============================================================================

VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "có", "cho", "các", "một", "những", "được", "trong",
    "với", "về", "theo", "tại", "từ", "này", "đó", "khi", "để", "hoặc",
    "thì", "mà", "như", "nếu", "ra", "vào", "đến", "trên", "dưới",
}


def strip_accents(text: str) -> str:
    """
    Bỏ dấu tiếng Việt để tăng khả năng match:
        'ma túy' / 'ma tuý' / 'ma tuy' đều có token phụ là 'ma', 'tuy'.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text


def normalize_text(text: str) -> str:
    """Chuẩn hóa unicode, lowercase, gom khoảng trắng."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """
    Tokenizer đơn giản cho tiếng Việt + văn bản luật.

    Giữ:
        - từ tiếng Việt có dấu
        - số điều luật: 248, 249
        - mã văn bản: 73/2021/qh14, 57/2022/nd-cp

    Đồng thời thêm token không dấu để tăng recall.
    """
    text = normalize_text(text)

    # Bắt token dạng chữ/số, giữ được mã 73/2021/qh14, 57/2022/nd-cp.
    raw_tokens = re.findall(
        r"[a-zà-ỹđ0-9]+(?:[/-][a-zà-ỹđ0-9]+)*",
        text,
        flags=re.IGNORECASE,
    )

    tokens: list[str] = []

    for token in raw_tokens:
        token = token.strip("_- /")

        if not token:
            continue

        if token in VIETNAMESE_STOPWORDS and len(token) <= 4:
            continue

        tokens.append(token)

        no_accent = strip_accents(token)
        if no_accent != token:
            tokens.append(no_accent)

    return tokens


# =============================================================================
# LOAD + CHUNK CORPUS
# =============================================================================

def clean_markdown(text: str) -> str:
    """Làm sạch nhẹ Markdown trước khi chunk."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def infer_doc_type(path: Path) -> str:
    """Suy luận loại tài liệu dựa vào thư mục con."""
    parts = {p.lower() for p in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def extract_title(markdown: str, fallback: str) -> str:
    """Lấy heading H1 đầu tiên làm title."""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("# ") and len(line) > 2:
            return line[2:].strip()
    return fallback


def extract_markdown_metadata(markdown: str) -> dict[str, str]:
    """
    Đọc metadata header do Task 3 tạo, ví dụ:
        - **source_file:** `abc.pdf`
        - **url:** https://...
    """
    metadata: dict[str, str] = {}
    pattern = re.compile(r"^\s*-\s+\*\*(.+?):\*\*\s*(.*)\s*$")

    for line in markdown.splitlines()[:120]:
        match = pattern.match(line)
        if not match:
            continue

        key = match.group(1).strip()
        value = match.group(2).strip().strip("`").strip()

        if key and value:
            safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", key).strip("_").lower()
            metadata[safe_key] = value

    return metadata


def load_markdown_documents() -> list[dict]:
    """
    Đọc toàn bộ Markdown files từ data/standardized/.

    Returns:
        List of {
            'content': str,
            'metadata': dict
        }
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục {STANDARDIZED_DIR}. "
            "Hãy chạy Task 3 trước."
        )

    documents: list[dict] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = clean_markdown(md_file.read_text(encoding="utf-8"))

        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        extra_metadata = extract_markdown_metadata(content)

        metadata = {
            "source": md_file.name,
            "source_path": str(relative_path).replace("\\", "/"),
            "absolute_path": str(md_file),
            "doc_type": infer_doc_type(relative_path),
            "title": extract_title(content, md_file.stem),
            "file_stem": md_file.stem,
            "file_extension": md_file.suffix.lower(),
            **extra_metadata,
        }

        documents.append(
            {
                "content": content,
                "metadata": metadata,
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents giống Task 4 bằng RecursiveCharacterTextSplitter.

    BM25 làm việc tốt hơn trên chunk vừa phải thay vì cả file rất dài,
    vì score không bị loãng và output dễ đưa vào RAG context.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks: list[dict] = []

    for doc in documents:
        splits = splitter.split_text(doc["content"])
        chunk_index = 0

        for split in splits:
            chunk_text = clean_markdown(split)

            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            metadata = {
                **doc["metadata"],
                "chunk_index": chunk_index,
                "chunking_method": "recursive",
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "char_count": len(chunk_text),
            }

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )

            chunk_index += 1

    return chunks


def load_corpus() -> list[dict]:
    """
    Load corpus từ data/standardized/ rồi chunk.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    documents = load_markdown_documents()
    corpus = chunk_documents(documents)

    if not corpus:
        raise RuntimeError(
            "Không tạo được corpus BM25. "
            "Kiểm tra data/standardized/ có file .md không."
        )

    return corpus


# =============================================================================
# LOCAL BM25 USING rank-bm25
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Corpus rỗng, không thể build BM25 index.")

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]

    # k1=1.5, b=0.75 là cấu hình phổ biến:
    # - k1 kiểm soát saturation của term frequency
    # - b kiểm soát length normalization
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=1.5,
        b=0.75,
    )

    return bm25


def get_local_bm25():
    """Lazy-load corpus và BM25 index một lần."""
    global CORPUS, BM25_INDEX

    if BM25_INDEX is None:
        print("Loading corpus from data/standardized/ ...")
        CORPUS = load_corpus()

        print(f"Building BM25 index for {len(CORPUS)} chunks ...")
        BM25_INDEX = build_bm25_index(CORPUS)

    return BM25_INDEX


def lexical_search_local(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng BM25 local.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    query = query.strip()
    if not query:
        raise ValueError("Query không được rỗng.")

    if top_k <= 0:
        raise ValueError("top_k phải > 0.")

    top_k = min(top_k, 100)

    bm25 = get_local_bm25()
    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict] = []

    for idx in top_indices:
        score = float(scores[idx])

        # BM25 score <= 0 thường là không match keyword đáng kể.
        if score <= 0:
            continue

        item = CORPUS[int(idx)]

        results.append(
            {
                "content": item["content"],
                "score": score,
                "metadata": item["metadata"],
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


# =============================================================================
# OPTIONAL: WEAVIATE BM25 BUILT-IN
# =============================================================================

@lru_cache(maxsize=1)
def get_weaviate_client():
    """Kết nối Weaviate local nếu dùng backend weaviate_bm25."""
    import weaviate

    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )

    if not client.is_ready():
        client.close()
        raise RuntimeError("Weaviate local chưa sẵn sàng.")

    return client


def lexical_search_weaviate(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm keyword bằng Weaviate BM25 built-in.

    Cơ chế:
        - Weaviate tokenizes query và property content
        - BM25/BM25F tính điểm dựa trên exact token matching
        - Phù hợp với mã điều luật, số văn bản, tên riêng, keyword chính xác
    """
    from weaviate.classes.query import MetadataQuery

    query = query.strip()
    if not query:
        raise ValueError("Query không được rỗng.")

    if top_k <= 0:
        raise ValueError("top_k phải > 0.")

    top_k = min(top_k, 100)

    client = get_weaviate_client()

    if not client.collections.exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Không tìm thấy collection '{COLLECTION_NAME}'. "
            "Hãy chạy Task 4 trước."
        )

    collection = client.collections.use(COLLECTION_NAME)

    response = collection.query.bm25(
        query=query,
        query_properties=["content"],
        limit=top_k,
        return_metadata=MetadataQuery(score=True),
    )

    results: list[dict] = []

    for obj in response.objects:
        properties = dict(obj.properties or {})
        content = properties.pop("content", "")

        score = getattr(obj.metadata, "score", None)
        if score is None:
            score = 0.0

        metadata = {
            "source": properties.get("source"),
            "source_path": properties.get("source_path"),
            "doc_type": properties.get("doc_type"),
            "title": properties.get("title"),
            "chunk_id": properties.get("chunk_id"),
            "chunk_index": properties.get("chunk_index"),
            "chunking_method": properties.get("chunking_method"),
            "char_count": properties.get("char_count"),
            "url": properties.get("url"),
            "source_domain": properties.get("source_domain"),
            "published_date": properties.get("published_date"),
            "date_crawled": properties.get("date_crawled"),
        }

        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": metadata,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


# =============================================================================
# PUBLIC API
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if LEXICAL_BACKEND == "local_bm25":
        return lexical_search_local(query=query, top_k=top_k)

    if LEXICAL_BACKEND == "weaviate_bm25":
        return lexical_search_weaviate(query=query, top_k=top_k)

    raise ValueError(
        f"LEXICAL_BACKEND không hợp lệ: {LEXICAL_BACKEND}. "
        "Chỉ hỗ trợ: 'local_bm25' hoặc 'weaviate_bm25'."
    )


# =============================================================================
# CLI TEST
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
        print(content[:600].replace("\n", " "))
        print()


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    print_results(results)
