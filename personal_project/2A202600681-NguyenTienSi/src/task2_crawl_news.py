"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma túy.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install -U crawl4ai
    crawl4ai-setup

Ghi chú:
    - Chỉ crawl bài báo công khai để phục vụ bài toán RAG/phân tích văn bản.
    - Không dùng nội dung này để cổ xúy, hướng dẫn hoặc mô tả cách sử dụng chất cấm.
"""

import asyncio
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from _console import configure_utf8_output

configure_utf8_output()

BASE_DIR = Path(__file__).parent.parent
os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(BASE_DIR / ".cache"))

DATA_DIR = BASE_DIR / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


ARTICLE_URLS = [
    # Tuổi Trẻ
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",

    # Dân trí
    "https://dantri.com.vn/phap-luat/ca-si-miu-le-bi-bat-qua-tang-su-dung-ma-tuy-o-hai-phong-20260511185303149.htm",
    "https://dantri.com.vn/phap-luat/truy-to-ca-si-chi-dan-nguoi-mau-an-tay-20260402122649916.htm",
    "https://dantri.com.vn/van-hoa/nhung-nghe-si-viet-lao-dao-vi-dinh-vao-ma-tuy-20230424033137629.htm",

    # VietnamNet
    "https://vietnamnet.vn/chi-dan-an-tay-truc-phuong-la-nhung-mat-xich-cuoi-trong-duong-day-ma-tuy-2342032.html",

    # Thanh Niên
    "https://thanhnien.vn/chuyen-an-bi-so-vn10-truy-to-nguoi-mau-an-tay-ca-si-chi-dan-truc-phuong-185260402125551927.htm",

    # PLO
    "https://plo.vn/truy-to-ca-sy-chi-dan-nguoi-mau-an-tay-va-225-bi-can-vu-4-tiep-vien-hang-khong-bi-loi-dung-van-chuyen-ma-tuy-post902216.html",
]


def slugify(text: str, max_length: int = 80) -> str:
    """Tạo slug an toàn cho tên file, không cần thư viện ngoài."""
    text = text.lower().strip()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_length] or "article"


def get_domain(url: str) -> str:
    """Lấy domain từ URL."""
    return urlparse(url).netloc.replace("www.", "")


def url_hash(url: str, length: int = 10) -> str:
    """Tạo hash ngắn để tránh trùng tên file."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:length]


def extract_markdown(result: Any) -> str:
    """
    Crawl4AI có thể trả markdown dạng string hoặc object.
    Hàm này xử lý cả hai trường hợp để script ổn định hơn.
    """
    markdown = getattr(result, "markdown", "")

    if isinstance(markdown, str):
        return markdown.strip()

    for attr in ("fit_markdown", "raw_markdown", "markdown"):
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    cleaned_html = getattr(result, "cleaned_html", "")
    if isinstance(cleaned_html, str):
        return cleaned_html.strip()

    return ""


def extract_title(result: Any, content_markdown: str) -> str:
    """Lấy title từ metadata; nếu thiếu thì lấy heading đầu tiên trong markdown."""
    metadata = getattr(result, "metadata", {}) or {}

    if isinstance(metadata, dict):
        for key in ("title", "og:title", "twitter:title"):
            title = metadata.get(key)
            if isinstance(title, str) and title.strip():
                return title.strip()

    for line in content_markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()

    return "Unknown"


def extract_published_date(result: Any) -> str | None:
    """Cố gắng lấy ngày xuất bản từ metadata nếu website có cung cấp."""
    metadata = getattr(result, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return None

    candidate_keys = [
        "article:published_time",
        "published_time",
        "publish_date",
        "date",
        "dc.date",
        "pubdate",
    ]

    for key in candidate_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


async def crawl_article(url: str, crawler: Any) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "source_domain": str,
            "title": str,
            "published_date": str | None,
            "date_crawled": str,
            "content_markdown": str,
            "content": str,
            "success": bool,
            "error": str | None
        }
    """
    try:
        result = await crawler.arun(url=url)

        success = getattr(result, "success", True)
        if success is False:
            error_message = getattr(result, "error_message", "Unknown crawl error")
            raise RuntimeError(error_message)

        content_markdown = extract_markdown(result)
        title = extract_title(result, content_markdown)

        if not content_markdown:
            raise ValueError("Không trích xuất được content_markdown")

        return {
            "url": url,
            "source_domain": get_domain(url),
            "title": title,
            "published_date": extract_published_date(result),
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content_markdown,
            "content": content_markdown,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        return {
            "url": url,
            "source_domain": get_domain(url),
            "title": "Unknown",
            "published_date": None,
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": "",
            "content": "",
            "success": False,
            "error": str(exc),
        }


def build_filename(index: int, article: dict) -> str:
    """Tạo tên file rõ ràng từ domain + title + hash URL."""
    domain = slugify(article.get("source_domain", "unknown"), max_length=30)
    title = slugify(article.get("title") or "unknown", max_length=70)
    short_hash = url_hash(article["url"])
    return f"article_{index:02d}_{domain}_{title}_{short_hash}.json"


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    from crawl4ai import AsyncWebCrawler

    manifest = []

    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

            article = await crawl_article(url, crawler)
            filename = build_filename(i, article)
            filepath = DATA_DIR / filename

            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            manifest.append(
                {
                    "index": i,
                    "filename": filename,
                    "filepath": str(filepath),
                    "url": url,
                    "source_domain": article["source_domain"],
                    "title": article["title"],
                    "success": article["success"],
                    "error": article["error"],
                }
            )

            if article["success"]:
                print(f"  ✓ Saved: {filepath}")
            else:
                print(f"  ✗ Failed but saved error JSON: {filepath}")
                print(f"    Error: {article['error']}")

            # Nghỉ nhẹ để tránh gửi request quá dồn dập.
            await asyncio.sleep(1)

    manifest_path = DATA_DIR / "manifest_news_sources.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ Đã ghi manifest: {manifest_path}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy.")
    else:
        asyncio.run(crawl_all())
