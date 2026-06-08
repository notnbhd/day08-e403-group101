"""
Task 1 — Thu thập văn bản pháp luật về ma tuý.

Tải tự động ≥3 văn bản pháp luật (PDF) từ cổng thông tin chính thức
datafiles.chinhphu.vn vào data/landing/legal/.

Chạy:
    uv run python -m src.task1_collect_legal_docs
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Nguồn chính thức (Cơ sở dữ liệu văn bản QPPL của Chính phủ).
# Mỗi entry: (tên file lưu, URL trực tiếp).
SOURCES: list[tuple[str, str]] = [
    (
        "luat-phong-chong-ma-tuy-2021.pdf",
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
    ),
    (
        "nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy.pdf",
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf",
    ),
    (
        "bo-luat-hinh-su-hop-nhat-cac-toi-ve-ma-tuy.pdf",
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf",
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filename: str) -> bool:
    """Tải 1 file. Trả về True nếu thành công (file > 1KB)."""
    dest = DATA_DIR / filename
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  ↪ Đã có sẵn, bỏ qua: {filename} ({dest.stat().st_size:,} bytes)")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size = dest.stat().st_size
        if size <= 1024:
            print(f"  ✗ File quá nhỏ ({size} bytes): {filename}")
            return False
        print(f"  ✓ {filename}  ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ Lỗi tải {filename}: {e}")
        print(f"    → Có thể tải thủ công từ: {url}")
        return False


def download_all() -> int:
    """Tải toàn bộ SOURCES. Trả về số file tải thành công."""
    setup_directory()
    print(f"Task 1: Thu thập văn bản pháp luật → {DATA_DIR}")
    ok = sum(download_file(url, name) for name, url in SOURCES)
    print(f"\nHoàn tất: {ok}/{len(SOURCES)} file (yêu cầu ≥3).")
    return ok


if __name__ == "__main__":
    download_all()
