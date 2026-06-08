"""
Lo-fi CHATBOT UI cho RAG pipeline (Day 8).

Giao diện chat tối giản phong cách terminal:
  • Gõ câu hỏi như chat → chạy full pipeline (hybrid → rerank → fallback
    PageIndex → LLM có citation). Lịch sử hội thoại được giữ lại.
  • LOG TRACKING: bắt toàn bộ stdout của pipeline cho TỪNG tin nhắn, hiển thị
    trong expander "🖥 log" ngay dưới câu trả lời.
  • SOURCE TRACKING: liệt kê nguồn được dùng cho từng câu trả lời (điểm số,
    file nguồn, loại legal/news, nguồn truy hồi hybrid vs pageindex).

Chạy:
    uv run streamlit run app.py
"""

import contextlib
import io
import time

import streamlit as st

from src.config import settings
from src.task10_generation import generate_with_citation

# =============================================================================
# PAGE CONFIG + LO-FI CSS
# =============================================================================

st.set_page_config(page_title="RAG · lo-fi chat", page_icon="📟", layout="centered")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: 'JetBrains Mono','Courier New',monospace; }
      .stApp { background: #0d0f12; color: #c8d0c8; }
      h1, h2, h3 { color: #9ece6a !important; letter-spacing: .5px; }
      .term {
        background: #05070a; border: 1px solid #2a2f3a; border-radius: 4px;
        padding: 10px 12px; color: #8fb36b; font-size: 12px; line-height: 1.5;
        white-space: pre-wrap; max-height: 300px; overflow-y: auto;
      }
      .badge {
        display: inline-block; padding: 1px 8px; border-radius: 10px;
        font-size: 11px; font-weight: 700; letter-spacing: .5px;
      }
      .b-hybrid    { background: #1f3a5f; color: #7aa2f7; border: 1px solid #3b5a8a; }
      .b-pageindex { background: #4a2f1f; color: #e0af68; border: 1px solid #7a5a3b; }
      .b-none      { background: #3a1f1f; color: #f7768e; border: 1px solid #8a3b3b; }
      .b-legal     { background: #1f3a2f; color: #9ece6a; border: 1px solid #3b8a5a; }
      .b-news      { background: #3a1f3a; color: #bb9af7; border: 1px solid #6a3b8a; }
      .src-card {
        background: #11141a; border: 1px solid #2a2f3a; border-radius: 4px;
        padding: 8px 10px; margin-bottom: 6px;
      }
      .meta { color: #6b7280; font-size: 12px; }
      .stChatInput textarea, .stChatInput input {
        background: #05070a !important; color: #c8d0c8 !important;
        font-family: monospace !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

ORIGIN_CLS = {"hybrid": "b-hybrid", "pageindex": "b-pageindex"}
TYPE_CLS = {"legal": "b-legal", "news": "b-news"}


def _badge(label: str, cls: str) -> str:
    return f"<span class='badge {cls}'>{label}</span>"


# =============================================================================
# SIDEBAR — config + connection status
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙ config")
    st.markdown(
        f"<div class='meta'>"
        f"LLM&nbsp;&nbsp;&nbsp;&nbsp;: <b>{settings.LLM_MODEL}</b><br>"
        f"embed&nbsp;&nbsp;: <b>{settings.EMBEDDING_MODEL}</b> ({settings.EMBEDDING_DIM}d)<br>"
        f"rerank : <b>{settings.RERANK_MODEL}</b><br>"
        f"vstore : <b>weaviate / {settings.WEAVIATE_COLLECTION}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🔌 keys")

    def _ok(flag: bool) -> str:
        return "🟢 set" if flag else "🔴 missing"

    st.markdown(
        f"<div class='meta'>"
        f"OpenRouter : {_ok(bool(settings.OPENROUTER_API_KEY))}<br>"
        f"Weaviate&nbsp;&nbsp;&nbsp;: {_ok(bool(settings.WEAVIATE_URL and settings.WEAVIATE_API_KEY))}<br>"
        f"PageIndex&nbsp;&nbsp;: {_ok(bool(settings.PAGEINDEX_API_KEY))}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
    if st.button("🗑 Xoá hội thoại"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Lo-fi RAG chatbot · drug-law + news")


# =============================================================================
# RENDER HELPERS
# =============================================================================

def render_sources(sources: list[dict]):
    """SOURCE TRACKING — expander liệt kê nguồn dùng cho câu trả lời."""
    if not sources:
        return
    with st.expander(f"📚 nguồn ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            meta = s.get("metadata", {}) or {}
            src_file = meta.get("source", "?")
            dtype = meta.get("type", "?")
            origin = s.get("source", "?")
            score = s.get("score", 0.0)
            o_cls = ORIGIN_CLS.get(origin, "b-none")
            t_cls = TYPE_CLS.get(dtype, "b-none")
            preview = (s.get("content", "") or "")[:240].replace("\n", " ")
            chunk_bit = (
                f" · chunk {meta.get('chunk_index')}" if "chunk_index" in meta else ""
            )
            st.markdown(
                f"<div class='src-card'>"
                f"<b>#{i}</b> · score <b>{score:.3f}</b> "
                f"{_badge(origin, o_cls)} {_badge(dtype, t_cls)}<br>"
                f"<span class='meta'>📄 {src_file}{chunk_bit}</span><br>"
                f"<span class='meta'>{preview}…</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_logs(logs: str):
    """LOG TRACKING — expander hiển thị stdout pipeline cho câu trả lời."""
    if not logs:
        return
    with st.expander("🖥 log pipeline"):
        st.markdown(f"<div class='term'>{logs}</div>", unsafe_allow_html=True)


def render_assistant(msg: dict):
    """Render 1 tin nhắn assistant đã lưu (answer + meta + source + log)."""
    if msg.get("error"):
        st.error(msg["error"])
    else:
        origin = msg.get("retrieval_source", "none")
        st.markdown(
            f"<div class='meta'>retrieval: "
            f"{_badge(origin, ORIGIN_CLS.get(origin, 'b-none'))} · "
            f"⏱ {msg.get('elapsed', 0):.1f}s · "
            f"{len(msg.get('sources', []))} nguồn</div>",
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])
    render_sources(msg.get("sources", []))
    render_logs(msg.get("logs", ""))


# =============================================================================
# CHAT STATE + HISTORY
# =============================================================================

st.markdown("# 📟 RAG chatbot")
st.markdown(
    "<div class='meta'>hybrid (BM25+dense) → rerank (cohere) → "
    "fallback PageIndex → LLM có citation</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay lịch sử.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)


# =============================================================================
# CHAT INPUT → PIPELINE
# =============================================================================

prompt = st.chat_input("Hỏi về ma tuý / pháp luật / nghệ sĩ…")

if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        log_buf = io.StringIO()
        t0 = time.time()
        result, error = None, None
        with st.spinner("retrieval + generation…"):
            # LOG TRACKING: bắt mọi print() của pipeline vào buffer.
            with contextlib.redirect_stdout(log_buf):
                try:
                    print(f"▶ Query: {prompt!r}")
                    print(f"▶ top_k={top_k}")
                    result = generate_with_citation(prompt, top_k=top_k)
                    print("✓ Hoàn tất.")
                except Exception as e:  # noqa: BLE001 — show mọi lỗi lên UI
                    error = e
                    print(f"✗ Lỗi: {type(e).__name__}: {e}")
        elapsed = time.time() - t0
        logs = log_buf.getvalue().strip()

        if error:
            msg = {
                "role": "assistant",
                "content": "",
                "error": f"{type(error).__name__}: {error}",
                "sources": [],
                "logs": logs,
                "elapsed": elapsed,
            }
        else:
            msg = {
                "role": "assistant",
                "content": result["answer"],
                "retrieval_source": result.get("retrieval_source", "none"),
                "sources": result.get("sources", []),
                "logs": logs,
                "elapsed": elapsed,
            }
        render_assistant(msg)
        st.session_state.messages.append(msg)
