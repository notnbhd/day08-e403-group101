"""
Task 2 — Crawl bài báo về nghệ sĩ Việt liên quan tới ma tuý.

Dùng Crawl4AI để crawl ≥5 bài báo, lưu mỗi bài thành 1 file JSON trong
data/landing/news/ kèm metadata (url, title, date_crawled, content_markdown).

Chạy:
    uv run python -m src.task2_crawl_news
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Danh sách bài báo (VnExpress / Tuổi Trẻ / Thanh Niên / VOV / VietNamNet).
# Để dư vài URL để phòng trường hợp 1-2 link lỗi mà vẫn đủ ≥5 bài.
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-chau-viet-cuong-hau-toa-vi-nhet-toi-hai-chet-co-gai-20-tuoi-3890738.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-nhan-13-nam-tu-vi-nhet-toi-hai-chet-co-gai-3891028.html",
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
    "https://thanhnien.vn/chi-dan-huu-tin-va-loat-sao-viet-gay-on-ao-vi-dinh-toi-ma-tuy-185241110141122628.htm",
    "https://vnexpress.net/20-nam-hoat-dong-cua-miu-le-truoc-khi-bi-bat-qua-tang-dung-ma-tuy-5072922.html",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://tienphong.vn/nhieu-nghe-si-viet-bi-bat-vi-dinh-vao-ma-tuy-post1649760.tpo",
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _slug(url: str) -> str:
    """Tạo tên file an toàn từ URL."""
    tail = url.rstrip("/").split("/")[-1]
    tail = re.sub(r"\.(html?|htm|tpo|vov|aspx)$", "", tail)
    tail = re.sub(r"[^a-zA-Z0-9-]+", "-", tail).strip("-")
    return tail[:80] or "article"


def _extract_markdown(result) -> str:
    """Crawl4AI đổi kiểu result.markdown qua các version → chuẩn hoá về str."""
    md = getattr(result, "markdown", "") or ""
    if not isinstance(md, str):
        # MarkdownGenerationResult object
        md = getattr(md, "raw_markdown", None) or str(md)
    return md


async def crawl_article(crawler, url: str) -> dict | None:
    """Crawl 1 bài báo → dict metadata + content, hoặc None nếu lỗi."""
    try:
        result = await crawler.arun(url=url)
    except Exception as e:
        print(f"  ✗ Lỗi crawl {url}: {e}")
        return None

    if not getattr(result, "success", True):
        print(f"  ✗ Crawl thất bại: {url}")
        return None

    content = _extract_markdown(result)
    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") or metadata.get("og:title") or "Unknown"

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_all() -> int:
    """Crawl toàn bộ ARTICLE_URLS. Trả về số bài lưu thành công."""
    setup_directory()
    from crawl4ai import AsyncWebCrawler

    saved = 0
    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            article = await crawl_article(crawler, url)
            if not article or len(article["content_markdown"]) < 500:
                print("  ✗ Bỏ qua (nội dung quá ngắn / lỗi).")
                continue
            filepath = DATA_DIR / f"article_{i:02d}_{_slug(url)}.json"
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  ✓ Saved: {filepath.name} ({len(article['content_markdown']):,} chars)")
            saved += 1

    print(f"\nHoàn tất: {saved} bài báo (yêu cầu ≥5).")
    return saved


if __name__ == "__main__":
    asyncio.run(crawl_all())
