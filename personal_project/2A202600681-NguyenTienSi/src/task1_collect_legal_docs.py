"""
Task 1 — Thu thập văn bản pháp luật về ma túy và các chất cấm.

Nguồn chính thống dùng trong script:
    - Cổng TTĐT Chính phủ: https://vanban.chinhphu.vn
    - File đính kèm chính thức: https://datafiles.chinhphu.vn

Ghi chú:
    - Luật Phòng, chống ma túy 2021 có số đúng là 73/2021/QH14.
    - Script chỉ tải văn bản pháp luật phục vụ học tập/nghiên cứu/RAG,
      không trích xuất hay hướng dẫn sử dụng chất cấm.
"""

from pathlib import Path
import json
import requests

from _console import configure_utf8_output

configure_utf8_output()


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "manifest_legal_sources.json"

LEGAL_DOCUMENTS = [
    {
        "title": "Luật Phòng, chống ma túy",
        "number": "73/2021/QH14",
        "year": 2021,
        "filename": "luat_phong_chong_ma_tuy_73_2021_QH14_2021.pdf",
        "source_page": "https://vanban.chinhphu.vn/?docid=204940&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
    },
    {
        "title": "Nghị định quy định chi tiết và hướng dẫn thi hành một số điều của Luật phòng, chống ma túy",
        "number": "105/2021/ND-CP",
        "year": 2021,
        "filename": "nghi_dinh_105_2021_ND_CP_2021.pdf",
        "source_page": "https://vanban.chinhphu.vn/?docid=204678&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf",
    },
    {
        "title": "Nghị định quy định các danh mục chất ma túy và tiền chất",
        "number": "57/2022/ND-CP",
        "year": 2022,
        "filename": "nghi_dinh_57_2022_ND_CP_2022.pdf",
        "source_page": "https://vanban.chinhphu.vn/?docid=206454&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/08/57-cp.signed.pdf",
    },
    {
        "title": "Bộ luật Hình sự",
        "number": "100/2015/QH13",
        "year": 2015,
        "filename": "bo_luat_hinh_su_100_2015_QH13_2015.pdf",
        "source_page": "https://vanban.chinhphu.vn/default.aspx?docid=183216&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2016/01/100.signed_01.pdf",
    },
    {
        "title": "Luật sửa đổi, bổ sung một số điều của Bộ luật Hình sự số 100/2015/QH13",
        "number": "12/2017/QH14",
        "year": 2017,
        "filename": "luat_sua_doi_bo_luat_hinh_su_12_2017_QH14_2017.pdf",
        "source_page": "https://vanban.chinhphu.vn/default.aspx?docid=190507&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2017/08/12.signed.pdf",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str, overwrite: bool = False) -> Path:
    """Tải 1 file PDF/DOCX về DATA_DIR."""
    filepath = DATA_DIR / filename

    if filepath.exists() and not overwrite:
        print(f"↷ Bỏ qua vì đã tồn tại: {filepath}")
        return filepath

    headers = {
        "User-Agent": "Mozilla/5.0 legal-document-downloader/1.0"
    }

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    content = response.content
    if len(content) < 1024:
        raise ValueError(f"File tải về quá nhỏ, có thể lỗi: {url}")

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        raise ValueError(f"File không đúng định dạng PDF: {url}")

    filepath.write_bytes(content)
    print(f"✓ Đã tải: {filepath} ({len(content):,} bytes)")
    return filepath


def write_manifest(downloaded_files: list[dict]):
    """Ghi manifest nguồn để phục vụ ingestion/RAG và kiểm chứng citation."""
    MANIFEST_PATH.write_text(
        json.dumps(downloaded_files, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ Đã ghi manifest: {MANIFEST_PATH}")


def download_all(overwrite: bool = False):
    """Tải toàn bộ văn bản pháp luật trong LEGAL_DOCUMENTS."""
    downloaded_files = []

    for doc in LEGAL_DOCUMENTS:
        filepath = download_file(
            url=doc["download_url"],
            filename=doc["filename"],
            overwrite=overwrite,
        )

        downloaded_files.append(
            {
                "title": doc["title"],
                "number": doc["number"],
                "year": doc["year"],
                "filename": doc["filename"],
                "local_path": str(filepath),
                "source_page": doc["source_page"],
                "download_url": doc["download_url"],
                "source_type": "official_government_portal",
            }
        )

    write_manifest(downloaded_files)
    print(f"✓ Hoàn tất. Tổng số file trong danh sách: {len(downloaded_files)}")


if __name__ == "__main__":
    setup_directory()
    download_all(overwrite=False)
