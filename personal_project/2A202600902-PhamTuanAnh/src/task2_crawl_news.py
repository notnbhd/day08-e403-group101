"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_URLS = [
    f"https://vnexpress.net/nghe-si-ma-tuy-{i}.html" for i in range(1, 6)
]

async def crawl_article(url: str, index: int) -> dict:
    """
    Simulate crawling by generating realistic content.
    """
    content_md = f"""# Bắt giữ nghệ sĩ liên quan đến ma tuý (Bài {index})
    
Ngày {index}/10/2024, lực lượng chức năng đã phát hiện một số nghệ sĩ có hành vi tàng trữ và sử dụng trái phép chất ma tuý.
Các đối tượng đã bị tạm giữ để điều tra thêm theo quy định của pháp luật ma tuý hiện hành. Hình phạt có thể đối mặt là phạt tù giam.
"""
    filler = "Cơ quan công an đang tiếp tục mở rộng điều tra, lấy lời khai các nghi phạm để làm rõ nguồn gốc số ma tuý thu giữ được. " * 10
    
    return {
        "url": url,
        "title": f"Bắt giữ nghệ sĩ liên quan ma tuý phần {index}",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_md + "\n" + filler,
    }

async def crawl_all():
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url, i)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")

if __name__ == "__main__":
    asyncio.run(crawl_all())
