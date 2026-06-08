"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_URLS = [
    "https://kenh14.vn/son-ngoc-minh-su-dung-ma-tuy-voi-ai-215260521140622858.chn",
    "https://kenh14.vn/toan-canh-be-boi-ma-tuy-cua-miu-le-su-sup-do-cua-nghe-si-dang-co-moi-thu-trong-tay-215260517071721899.chn",
    "https://tuoitre.vn/ca-si-long-nhat-thua-nhan-da-nhieu-lan-dat-mua-ma-tuy-ve-su-dung-20260520161117184.htm",
    "https://plo.vn/ca-si-chi-dan-an-tay-va-nhung-nghe-si-danh-mat-su-nghiep-vi-ma-tuy-post819930.html",
    "https://vietnamnet.vn/loat-ca-si-dinh-chat-cam-ma-tuy-ton-tai-trong-mau-nuoc-tieu-bao-lau-2518453.html"
]

async def crawl_article(url: str) -> dict:
    """Crawl một bài báo từ URL và trả về metadata + content."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return {
            "url": url,
            "title": result.metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }
    raise NotImplementedError("Implement crawl_article")


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath}")

if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
