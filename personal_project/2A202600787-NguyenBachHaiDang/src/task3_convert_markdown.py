"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

- Legal (PDF/DOCX)  → MarkItDown của Microsoft.
- News (JSON crawl)  → lấy content_markdown đã có sẵn + thêm metadata header.

Output giữ nguyên cấu trúc thư mục con (legal/, news/) trong data/standardized/.

Chạy:
    uv run python -m src.task3_convert_markdown
"""

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> int:
    """Convert PDF/DOCX trong data/landing/legal/ sang markdown bằng MarkItDown."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  (chưa có data/landing/legal/)")
        return 0

    md = MarkItDown()
    count = 0
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue
        print(f"Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            text = result.text_content or ""
        except Exception as e:
            print(f"  ✗ Lỗi convert {filepath.name}: {e}")
            continue
        if len(text) < 200:
            print(f"  ✗ Nội dung quá ngắn ({len(text)} chars), bỏ qua.")
            continue
        out = output_dir / f"{filepath.stem}.md"
        out.write_text(text, encoding="utf-8")
        print(f"  ✓ Saved: {out.name} ({len(text):,} chars)")
        count += 1
    return count


def convert_news_articles() -> int:
    """Convert JSON crawl trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  (chưa có data/landing/news/)")
        return 0

    count = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ✗ Lỗi đọc JSON {filepath.name}: {e}")
            continue

        header = (
            f"# {data.get('title', 'Unknown')}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        )
        content = header + (data.get("content_markdown", "") or "")
        if len(content) < 200:
            print(f"  ✗ Nội dung quá ngắn, bỏ qua.")
            continue
        out = output_dir / f"{filepath.stem}.md"
        out.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved: {out.name} ({len(content):,} chars)")
        count += 1
    return count


def convert_all():
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents (MarkItDown) ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles (JSON) ---")
    n_news = convert_news_articles()

    print(f"\n✓ Done! {n_legal} legal + {n_news} news → {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
