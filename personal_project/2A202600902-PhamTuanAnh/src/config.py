"""
Central configuration cho RAG pipeline.

Toàn bộ API keys và model names được nạp từ .env qua đây, nên bạn có thể
đổi key / đổi model mà KHÔNG cần sửa code trong các task.

Cách dùng:
    from src.config import settings, get_openai_client, connect_weaviate
    client = get_openai_client()
    print(settings.LLM_MODEL)
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Nạp .env một lần duy nhất khi import module này.
load_dotenv()


def _get(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # --- OpenRouter (embeddings + rerank + LLM, dùng chung 1 key) ---
    OPENROUTER_API_KEY: str = _get("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # --- Models (đổi tự do trong .env) ---
    LLM_MODEL: str = _get("LLM_MODEL", "openai/gpt-4o-mini")
    EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    EMBEDDING_DIM: int = int(_get("EMBEDDING_DIM", "1536"))
    RERANK_MODEL: str = _get("RERANK_MODEL", "cohere/rerank-v3.5")

    # --- Weaviate Cloud ---
    WEAVIATE_URL: str = _get("WEAVIATE_URL")
    WEAVIATE_API_KEY: str = _get("WEAVIATE_API_KEY")
    WEAVIATE_COLLECTION: str = _get("WEAVIATE_COLLECTION", "DrugLawDocs")

    # --- PageIndex (Task 8 - vectorless RAG) ---
    PAGEINDEX_API_KEY: str = _get("PAGEINDEX_API_KEY")


settings = Settings()


def get_openai_client():
    """
    OpenAI SDK client trỏ vào OpenRouter.
    Dùng cho chat completions (Task 10) và embeddings (Task 4).
    """
    from openai import OpenAI

    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY chưa được set. Copy .env.example -> .env và điền key."
        )
    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )


def openrouter_headers() -> dict:
    """Headers cho các REST call trực tiếp tới OpenRouter (vd: /rerank)."""
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY chưa được set. Copy .env.example -> .env và điền key."
        )
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def connect_weaviate():
    """
    Kết nối tới Weaviate Cloud.
    Nhớ client.close() sau khi dùng (hoặc dùng context manager).
    """
    import weaviate
    from weaviate.classes.init import Auth

    if not settings.WEAVIATE_URL or not settings.WEAVIATE_API_KEY:
        raise RuntimeError(
            "WEAVIATE_URL / WEAVIATE_API_KEY chưa được set trong .env."
        )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.WEAVIATE_URL,
        auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY),
    )
