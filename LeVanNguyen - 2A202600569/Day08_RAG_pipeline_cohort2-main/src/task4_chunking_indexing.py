"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Vì sao chọn 500? Cân bằng giữa độ dài ngữ cảnh (semantic coherence) và hiệu suất truy vấn. 500 tokens đủ để chứa ý chính của văn bản pháp lý mà không quá dài.
CHUNK_OVERLAP = 50      # Vì sao chọn 50? Tạo sự liên kết giữa các chunks, tránh mất thông tin quan trọng ở biên giới chunk.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic" — Recursive xử lý tốt các cấu trúc đa dạng (tiêu đề, đoạn văn, danh sách) mà không yêu cầu markup Markdown.

# Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt. BGE-M3 là mô hình state-of-the-art cho tiếng Việt và hỗ trợ 100+ ngôn ngữ. Khác với all-MiniLM (chỉ tiếng Anh), nó giữ được ý nghĩa ngữ pháp Việt.
EMBEDDING_DIM = 1024   # Phải khớp với chiều của BGE-M3 (1024D), cho phép capture thông tin đầy đủ so với mô hình nhỏ hơn.

# Chọn vector store
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss" — Weaviate hỗ trợ hybrid search (kết hợp dense + sparse retrieval) và có sẵn BM25, giúp cải thiện độ chính xác retrieval cho văn bản pháp lý với terminology đặc thù.

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE == "weaviate":
        import weaviate
        # Connect to local Weaviate instance
        client = weaviate.connect_to_local()
        class_name = "DrugLawDocs"
        try:
            collection = client.collections.get(class_name)
        except weaviate.exceptions.WeaviateCollectionNotFoundError:
            collection = client.collections.create(
                name=class_name,
                vectorizer_config=None,  # vì bạn tự cung cấp embedding
                properties=[
                    {"name": "content", "dataType": "text"},
                    {"name": "source", "dataType": "text"},
                    {"name": "doc_type", "dataType": "text"},
                    {"name": "chunk_index", "dataType": "int"},
                ],
            )

        # Insert chunks
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if embedding is None or not isinstance(embedding, (list, tuple)):
                raise ValueError("Chunk thiếu embedding hoặc sai định dạng")

            collection.data.insert(
                properties={
                    "content": chunk["content"],
                    "source": chunk["metadata"].get("source", ""),
                    "doc_type": chunk["metadata"].get("type", ""),
                    "chunk_index": chunk["metadata"].get("chunk_index", 0),
                },
                vector=embedding,
            )

    elif VECTOR_STORE == "chromadb":
        import chromadb
        # The modern, supported way to save data locally in Chroma
        client = chromadb.PersistentClient(path=".chromadb")
        collection = client.get_or_create_collection(name="DrugLawDocs")

        ids = [f"{chunk['metadata']['source']}-{chunk['metadata']['chunk_index']}" for chunk in chunks]
        metadatas = [
            {
                "source": chunk["metadata"]["source"],
                "doc_type": chunk["metadata"]["type"],
                "chunk_index": chunk["metadata"]["chunk_index"],
            }
            for chunk in chunks
        ]
        documents = [chunk["content"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    elif VECTOR_STORE == "faiss":
        import faiss
        import numpy as np
        import pickle

        dim = EMBEDDING_DIM
        vectors = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        with open("faiss_index.pkl", "wb") as f:
            pickle.dump({"index": index, "chunks": chunks}, f)

    else:
        raise ValueError(f"Unsupported VECTOR_STORE: {VECTOR_STORE}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")

if __name__ == "__main__":
    run_pipeline()