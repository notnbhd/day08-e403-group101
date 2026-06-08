"""
Task 4 — Chunking & Indexing vào Vector Store (Weaviate Cloud).

Pipeline: load markdown → chunk → embed (OpenRouter) → index (Weaviate).

LỰA CHỌN & LÝ DO:
  • Chunking: RecursiveCharacterTextSplitter.
      - An toàn & robust cho văn bản pháp luật tiếng Việt (cắt theo \n\n → \n →
        câu → từ), giữ ranh giới ngữ nghĩa tốt hơn cắt cứng theo ký tự.
  • CHUNK_SIZE = 800, CHUNK_OVERLAP = 120.
      - 800 ký tự ≈ 1 điều/khoản luật, đủ ngữ cảnh để trả lời mà không quá dài
        gây loãng embedding; overlap 120 (~15%) để không mất ngữ cảnh ở mép chunk.
  • Embedding: openai/text-embedding-3-small qua OpenRouter (1536 chiều).
      - Chất lượng tốt, rẻ ($0.02/1M token), hỗ trợ đa ngôn ngữ (tiếng Việt OK).
  • Vector store: Weaviate Cloud (vectorizer=none → ta tự đưa vector vào),
      hỗ trợ hybrid (BM25 + dense) built-in dùng cho Task 6 & 9.

Chạy:
    uv run python -m src.task4_chunking_indexing
"""

from pathlib import Path

try:
    from .config import settings, get_openai_client, connect_weaviate
except ImportError:  # chạy trực tiếp: python src/task4_chunking_indexing.py
    from config import settings, get_openai_client, connect_weaviate

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = settings.EMBEDDING_MODEL       # openai/text-embedding-3-small
EMBEDDING_DIM = settings.EMBEDDING_DIM           # 1536
VECTOR_STORE = "weaviate"
COLLECTION_NAME = settings.WEAVIATE_COLLECTION   # DrugLawDocs

EMBED_BATCH = 64  # số chunk gửi mỗi request embeddings


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type},
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["content"])):
            if not chunk_text.strip():
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks qua OpenRouter (text-embedding-3-small, 1536 chiều).
    Mỗi chunk được thêm key 'embedding': list[float].
    """
    client = get_openai_client()
    texts = [c["content"] for c in chunks]

    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
        )
        for j, item in enumerate(resp.data):
            chunks[start + j]["embedding"] = item.embedding
        print(f"  embedded {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")
    return chunks


def _recreate_collection(client):
    """Tạo (hoặc tạo lại) collection trong Weaviate với vectorizer=none."""
    from weaviate.classes.config import Configure, Property, DataType

    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    return client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks (kèm vector) vào Weaviate Cloud."""
    client = connect_weaviate()
    try:
        collection = _recreate_collection(client)
        with collection.batch.dynamic() as batch:
            for c in chunks:
                meta = c["metadata"]
                batch.add_object(
                    properties={
                        "content": c["content"],
                        "source": meta.get("source", ""),
                        "doc_type": meta.get("type", ""),
                        "chunk_index": int(meta.get("chunk_index", 0)),
                    },
                    vector=c["embedding"],
                )
        failed = collection.batch.failed_objects
        if failed:
            print(f"  ⚠ {len(failed)} object lỗi khi insert (vd: {failed[0]})")
        print(f"  ✓ Collection '{COLLECTION_NAME}' có {len(collection)} objects")
    finally:
        client.close()


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} / collection={COLLECTION_NAME}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("⚠ Không có markdown nào. Chạy Task 1-3 trước.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    # Fail-fast: kiểm tra kết nối Weaviate TRƯỚC khi embedding (tránh tốn tiền
    # embedding rồi mới phát hiện không kết nối được vector store).
    print("… Kiểm tra kết nối Weaviate")
    _client = connect_weaviate()
    _client.close()
    print("✓ Weaviate OK")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
