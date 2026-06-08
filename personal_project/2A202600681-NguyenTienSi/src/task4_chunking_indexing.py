"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy
    3. Chọn 1 embedding model
    4. Index vào vector store

Cài đặt chính:
    pip install langchain-text-splitters sentence-transformers weaviate-client

Nếu dùng Chroma fallback:
    pip install chromadb

Nếu dùng FAISS fallback:
    pip install faiss-cpu numpy
"""

import hashlib
import json
import re
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _console import configure_utf8_output

configure_utf8_output()


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = BASE_DIR / "data" / "standardized"
INDEX_DIR = BASE_DIR / "data" / "indexed"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstores"


# =============================================================================
# CONFIGURATION — lựa chọn cho corpus tiếng Việt gồm văn bản luật + bài báo
# =============================================================================

# Chọn RecursiveCharacterTextSplitter:
# - An toàn, phổ biến, ít phụ thuộc format.
# - Phù hợp cả văn bản luật convert từ PDF lẫn bài báo JSON convert sang Markdown.
# - Ưu tiên cắt theo đoạn / dòng / câu trước khi fallback sang ký tự.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn BAAI/bge-m3:
# - Multilingual, tốt hơn all-MiniLM cho tiếng Việt.
# - 1024 chiều, cân bằng chất lượng retrieval và chi phí local.
# - Không cần API key như OpenAI embedding.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 16

# Chọn Weaviate:
# - Khuyến cáo cho bài RAG vì hỗ trợ dense search, BM25 và hybrid search.
# - Script vẫn có Chroma/FAISS fallback nếu bạn đổi VECTOR_STORE.
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"

COLLECTION_NAME = "LegalNewsDocs"
RESET_COLLECTION = True
MIN_CHUNK_CHARS = 80


# =============================================================================
# UTILITIES
# =============================================================================

def now_iso() -> str:
    """Trả về thời điểm hiện tại theo UTC ISO format."""
    return datetime.now(timezone.utc).isoformat()


def sha1_text(text: str) -> str:
    """Hash ổn định cho nội dung text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
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
    Đọc metadata header do Task 3 tạo, dạng:
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


def stable_chunk_id(source_path: str, chunk_index: int, content: str) -> str:
    """Tạo chunk_id ổn định theo source + index + hash nội dung."""
    raw = f"{source_path}|{chunk_index}|{sha1_text(content)}"
    return sha1_text(raw)


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Chuẩn hóa metadata để hợp lệ với vector stores.
    Giữ scalar: str, int, float, bool. Các kiểu khác convert sang JSON string.
    """
    normalized: dict[str, Any] = {}

    for key, value in metadata.items():
        safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(key)).strip("_").lower()

        if not safe_key:
            continue

        if value is None:
            normalized[safe_key] = ""
        elif isinstance(value, (str, int, float, bool)):
            normalized[safe_key] = value
        else:
            normalized[safe_key] = json.dumps(value, ensure_ascii=False)

    return normalized


def ensure_dirs():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOAD DOCUMENTS
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {
            'content': str,
            'metadata': {
                'source': str,
                'source_path': str,
                'doc_type': str,
                'title': str,
                ...
            }
        }
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy STANDARDIZED_DIR: {STANDARDIZED_DIR}")

    documents: list[dict] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = clean_text(md_file.read_text(encoding="utf-8"))

        if not content:
            print(f"↷ Skip empty file: {md_file}")
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
            "document_hash": sha1_text(content),
            **extra_metadata,
        }

        documents.append(
            {
                "content": content,
                "metadata": normalize_metadata(metadata),
            }
        )

    return documents


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_with_recursive(documents: list[dict]) -> list[dict]:
    """Chunk bằng RecursiveCharacterTextSplitter."""
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

        local_index = 0
        for split in splits:
            chunk_text = clean_text(split)

            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            chunk_id = stable_chunk_id(
                doc["metadata"]["source_path"],
                local_index,
                chunk_text,
            )

            metadata = {
                **doc["metadata"],
                "chunk_id": chunk_id,
                "object_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                "chunk_index": local_index,
                "chunking_method": CHUNKING_METHOD,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "char_count": len(chunk_text),
                "content_hash": sha1_text(chunk_text),
            }

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": normalize_metadata(metadata),
                }
            )

            local_index += 1

    return chunks


def chunk_with_markdown_header(documents: list[dict]) -> list[dict]:
    """
    Chunk theo Markdown headers, sau đó recursive nếu section quá dài.
    Hữu ích khi file Markdown có heading tốt.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    for doc in documents:
        section_docs = header_splitter.split_text(doc["content"])

        local_index = 0
        for section in section_docs:
            section_text = clean_text(section.page_content)
            section_metadata = normalize_metadata(getattr(section, "metadata", {}) or {})

            if not section_text:
                continue

            smaller_splits = recursive_splitter.split_text(section_text)

            for split in smaller_splits:
                chunk_text = clean_text(split)

                if len(chunk_text) < MIN_CHUNK_CHARS:
                    continue

                chunk_id = stable_chunk_id(
                    doc["metadata"]["source_path"],
                    local_index,
                    chunk_text,
                )

                metadata = {
                    **doc["metadata"],
                    **section_metadata,
                    "chunk_id": chunk_id,
                    "object_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                    "chunk_index": local_index,
                    "chunking_method": CHUNKING_METHOD,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "char_count": len(chunk_text),
                    "content_hash": sha1_text(chunk_text),
                }

                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": normalize_metadata(metadata),
                    }
                )

                local_index += 1

    return chunks


def chunk_with_semantic(documents: list[dict]) -> list[dict]:
    """
    Semantic chunking nâng cao.

    Cần cài thêm:
        pip install langchain-experimental langchain-community

    Lưu ý: semantic chunking chậm hơn vì dùng embedding trong lúc split.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "CHUNKING_METHOD='semantic' cần cài thêm: "
            "pip install langchain-experimental langchain-community"
        ) from exc

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter = SemanticChunker(embeddings)

    chunks: list[dict] = []

    for doc in documents:
        semantic_docs = splitter.create_documents([doc["content"]])

        local_index = 0
        for semantic_doc in semantic_docs:
            chunk_text = clean_text(semantic_doc.page_content)

            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            chunk_id = stable_chunk_id(
                doc["metadata"]["source_path"],
                local_index,
                chunk_text,
            )

            metadata = {
                **doc["metadata"],
                "chunk_id": chunk_id,
                "object_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                "chunk_index": local_index,
                "chunking_method": CHUNKING_METHOD,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "char_count": len(chunk_text),
                "content_hash": sha1_text(chunk_text),
            }

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": normalize_metadata(metadata),
                }
            )

            local_index += 1

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    if CHUNKING_METHOD == "recursive":
        return chunk_with_recursive(documents)

    if CHUNKING_METHOD == "markdown_header":
        return chunk_with_markdown_header(documents)

    if CHUNKING_METHOD == "semantic":
        return chunk_with_semantic(documents)

    raise ValueError(f"Unsupported CHUNKING_METHOD: {CHUNKING_METHOD}")


# =============================================================================
# EMBEDDING
# =============================================================================

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    actual_dim = int(embeddings.shape[1])
    if actual_dim != EMBEDDING_DIM:
        print(
            f"⚠ EMBEDDING_DIM config={EMBEDDING_DIM}, "
            f"nhưng model trả về dim={actual_dim}. Dùng dim thực tế."
        )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.astype("float32").tolist()
        chunk["metadata"]["embedding_model"] = EMBEDDING_MODEL
        chunk["metadata"]["embedding_dim"] = actual_dim

    return chunks


# =============================================================================
# VECTOR STORE — WEAVIATE
# =============================================================================

def create_weaviate_collection(client):
    """Tạo hoặc reset collection trong Weaviate."""
    from weaviate.classes.config import Configure, DataType, Property

    if client.collections.exists(COLLECTION_NAME):
        if RESET_COLLECTION:
            print(f"Deleting existing Weaviate collection: {COLLECTION_NAME}")
            client.collections.delete(COLLECTION_NAME)
        else:
            print(f"Using existing Weaviate collection: {COLLECTION_NAME}")
            return client.collections.use(COLLECTION_NAME)

    print(f"Creating Weaviate collection: {COLLECTION_NAME}")

    collection = client.collections.create(
        name=COLLECTION_NAME,
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(
            bm25_k1=1.5,
            bm25_b=0.75,
            index_property_length=True,
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="source_path", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="object_uuid", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="chunking_method", data_type=DataType.TEXT),
            Property(name="char_count", data_type=DataType.INT),
            Property(name="content_hash", data_type=DataType.TEXT),
            Property(name="document_hash", data_type=DataType.TEXT),
            Property(name="embedding_model", data_type=DataType.TEXT),
            Property(name="embedding_dim", data_type=DataType.INT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="source_domain", data_type=DataType.TEXT),
            Property(name="published_date", data_type=DataType.TEXT),
            Property(name="date_crawled", data_type=DataType.TEXT),
        ],
    )

    return collection


def index_to_weaviate(chunks: list[dict]):
    """Index chunks vào Weaviate local."""
    import weaviate

    client = None

    try:
        client = weaviate.connect_to_local()

        if not client.is_ready():
            raise RuntimeError("Weaviate local chưa sẵn sàng.")

        collection = create_weaviate_collection(client)

        print(f"Indexing {len(chunks)} chunks to Weaviate/{COLLECTION_NAME}")

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = normalize_metadata(chunk["metadata"])

                properties = {
                    "content": chunk["content"],
                    "source": metadata.get("source", ""),
                    "source_path": metadata.get("source_path", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "title": metadata.get("title", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "object_uuid": metadata.get("object_uuid", ""),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "chunking_method": metadata.get("chunking_method", ""),
                    "char_count": int(metadata.get("char_count", 0)),
                    "content_hash": metadata.get("content_hash", ""),
                    "document_hash": metadata.get("document_hash", ""),
                    "embedding_model": metadata.get("embedding_model", ""),
                    "embedding_dim": int(metadata.get("embedding_dim", EMBEDDING_DIM)),
                    "url": metadata.get("url", ""),
                    "source_domain": metadata.get("source_domain", ""),
                    "published_date": metadata.get("published_date", ""),
                    "date_crawled": metadata.get("date_crawled", ""),
                }

                batch.add_object(
                    properties=properties,
                    vector=chunk["embedding"],
                )

                if batch.number_errors > 20:
                    raise RuntimeError("Batch import có quá nhiều lỗi, dừng indexing.")

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"⚠ Weaviate failed objects: {len(failed_objects)}")
            for failed in failed_objects[:3]:
                print(f"  - {failed}")

        print(f"✓ Weaviate indexing done: {COLLECTION_NAME}")

    except Exception as exc:
        raise RuntimeError(
            "Không index được vào Weaviate local. "
            "Hãy kiểm tra Weaviate server đã chạy ở port 8080 và gRPC 50051. "
            "Nếu dùng Docker, cần expose cả 8080:8080 và 50051:50051."
        ) from exc

    finally:
        if client is not None:
            client.close()


# =============================================================================
# VECTOR STORE — CHROMADB FALLBACK
# =============================================================================

def index_to_chromadb(chunks: list[dict]):
    """Index chunks vào ChromaDB local."""
    import chromadb

    chroma_dir = VECTORSTORE_DIR / "chroma_legal_news"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))

    existing_names = []
    for col in client.list_collections():
        existing_names.append(col if isinstance(col, str) else col.name)

    if COLLECTION_NAME in existing_names and RESET_COLLECTION:
        print(f"Deleting existing Chroma collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        metadata = normalize_metadata(chunk["metadata"])
        ids.append(metadata["chunk_id"])
        documents.append(chunk["content"])
        embeddings.append(chunk["embedding"])
        metadatas.append(metadata)

    print(f"Indexing {len(chunks)} chunks to ChromaDB/{COLLECTION_NAME}")

    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"✓ ChromaDB indexing done: {chroma_dir}")


# =============================================================================
# VECTOR STORE — FAISS FALLBACK
# =============================================================================

def index_to_faiss(chunks: list[dict]):
    """Index chunks vào FAISS local."""
    import faiss
    import numpy as np

    faiss_dir = VECTORSTORE_DIR / "faiss_legal_news"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")

    # Vì embeddings đã normalize nên inner product tương đương cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(faiss_dir / "index.faiss"))

    with (faiss_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            row = {
                "faiss_index": i,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓ FAISS indexing done: {faiss_dir}")


# =============================================================================
# INDEX ROUTER
# =============================================================================

def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào vector store đã chọn."""
    if not chunks:
        raise ValueError("Không có chunk nào để index.")

    if VECTOR_STORE == "weaviate":
        index_to_weaviate(chunks)
        return

    if VECTOR_STORE == "chromadb":
        index_to_chromadb(chunks)
        return

    if VECTOR_STORE == "faiss":
        index_to_faiss(chunks)
        return

    raise ValueError(f"Unsupported VECTOR_STORE: {VECTOR_STORE}")


# =============================================================================
# MANIFEST
# =============================================================================

def save_index_manifest(documents: list[dict], chunks: list[dict]):
    """Lưu manifest để kiểm tra pipeline indexing."""
    ensure_dirs()

    manifest = {
        "created_at": now_iso(),
        "standardized_dir": str(STANDARDIZED_DIR),
        "vector_store": VECTOR_STORE,
        "collection_name": COLLECTION_NAME,
        "chunking": {
            "method": CHUNKING_METHOD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_chars": MIN_CHUNK_CHARS,
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "configured_dim": EMBEDDING_DIM,
            "actual_dim": chunks[0]["metadata"].get("embedding_dim") if chunks else None,
        },
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "documents": [
            {
                "source": doc["metadata"].get("source"),
                "source_path": doc["metadata"].get("source_path"),
                "doc_type": doc["metadata"].get("doc_type"),
                "title": doc["metadata"].get("title"),
                "document_hash": doc["metadata"].get("document_hash"),
            }
            for doc in documents
        ],
        "chunk_preview": [
            {
                "chunk_id": chunk["metadata"].get("chunk_id"),
                "source_path": chunk["metadata"].get("source_path"),
                "doc_type": chunk["metadata"].get("doc_type"),
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "char_count": chunk["metadata"].get("char_count"),
                "text_preview": chunk["content"][:200],
            }
            for chunk in chunks[:20]
        ],
    }

    manifest_path = INDEX_DIR / "manifest_indexing.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✓ Saved manifest: {manifest_path}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 70)
    print("Task 4: Chunking & Indexing")
    print(f"  Standardized dir: {STANDARDIZED_DIR}")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (configured dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 70)

    ensure_dirs()

    try:
        docs = load_documents()
        print(f"\n✓ Loaded {len(docs)} documents")

        if not docs:
            raise RuntimeError(
                "Không tìm thấy file .md nào trong data/standardized/. "
                "Hãy chạy Task 3 trước."
            )

        chunks = chunk_documents(docs)
        print(f"✓ Created {len(chunks)} chunks")

        if not chunks:
            raise RuntimeError("Chunking tạo ra 0 chunks. Kiểm tra nội dung Markdown.")

        chunks = embed_chunks(chunks)
        print(f"✓ Embedded {len(chunks)} chunks")

        index_to_vectorstore(chunks)
        print("✓ Indexed to vector store")

        save_index_manifest(docs, chunks)

        print("\n✓ Done!")

    except Exception:
        error_path = INDEX_DIR / "task4_error.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")

        print("\n✗ Pipeline failed.")
        print(f"  Error log: {error_path}")
        raise


if __name__ == "__main__":
    run_pipeline()
