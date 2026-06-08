"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[all]"

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _console import configure_utf8_output
from markitdown import MarkItDown

configure_utf8_output()


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".doc"}
SUPPORTED_JSON_EXTENSION = ".json"


def now_iso() -> str:
    """Trả về thời điểm hiện tại theo UTC ISO format."""
    return datetime.now(timezone.utc).isoformat()


def setup_output_directory():
    """Tạo thư mục data/standardized/ nếu chưa có."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")


def get_output_path(input_path: Path) -> Path:
    """
    Tạo output path trong data/standardized/ và giữ nguyên cấu trúc thư mục.

    Ví dụ:
        data/landing/legal/a.pdf
        -> data/standardized/legal/a.md
    """
    relative_path = input_path.relative_to(LANDING_DIR)
    return (OUTPUT_DIR / relative_path).with_suffix(".md")


def ensure_parent_dir(path: Path):
    """Tạo thư mục cha cho output file nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)


def build_metadata_header(
    *,
    title: str | None,
    source_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """
    Tạo phần metadata header ở đầu file Markdown.
    Dùng dạng dễ đọc, không dùng YAML phức tạp để tránh lỗi ký tự đặc biệt.
    """
    lines = []

    if title:
        lines.append(f"# {title}")
    else:
        lines.append(f"# {source_path.stem}")

    lines.extend(
        [
            "",
            "## Metadata",
            "",
            f"- **source_file:** `{source_path.name}`",
            f"- **source_path:** `{source_path}`",
            f"- **converted_at:** `{now_iso()}`",
        ]
    )

    if extra_metadata:
        for key, value in extra_metadata.items():
            if value is None or value == "":
                continue
            lines.append(f"- **{key}:** {value}")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def extract_markitdown_text(result: Any) -> str:
    """
    Lấy text Markdown từ object kết quả của MarkItDown.

    Các version phổ biến dùng result.text_content.
    Hàm này có fallback để tránh lỗi nếu API thay đổi nhẹ.
    """
    text = getattr(result, "text_content", None)
    if isinstance(text, str):
        return text

    text = getattr(result, "markdown", None)
    if isinstance(text, str):
        return text

    return str(result)


def convert_document_file(filepath: Path, md: MarkItDown) -> Path:
    """Convert PDF/DOC/DOCX sang Markdown bằng MarkItDown."""
    output_path = get_output_path(filepath)
    ensure_parent_dir(output_path)

    print(f"Converting document: {filepath.relative_to(LANDING_DIR)}")

    result = md.convert(str(filepath))
    markdown_text = extract_markitdown_text(result).strip()

    header = build_metadata_header(
        title=filepath.stem,
        source_path=filepath,
        extra_metadata={
            "original_extension": filepath.suffix.lower(),
            "converter": "markitdown",
        },
    )

    output_path.write_text(header + markdown_text + "\n", encoding="utf-8")
    print(f"  ✓ Saved: {output_path}")
    return output_path


def load_json_file(filepath: Path) -> dict | list:
    """Đọc JSON file với encoding UTF-8."""
    return json.loads(filepath.read_text(encoding="utf-8"))


def json_article_to_markdown(data: dict, filepath: Path) -> str:
    """
    Convert JSON bài báo đã crawl ở Task 2 sang Markdown.

    Ưu tiên content_markdown.
    Nếu không có, fallback sang content.
    """
    title = data.get("title") or filepath.stem
    url = data.get("url", "N/A")
    source_domain = data.get("source_domain", "N/A")
    published_date = data.get("published_date", "N/A")
    date_crawled = data.get("date_crawled", "N/A")

    content = (
        data.get("content_markdown")
        or data.get("content")
        or data.get("markdown")
        or ""
    )

    header = build_metadata_header(
        title=title,
        source_path=filepath,
        extra_metadata={
            "url": url,
            "source_domain": source_domain,
            "published_date": published_date,
            "date_crawled": date_crawled,
            "original_extension": ".json",
            "converter": "json_article_converter",
        },
    )

    return header + str(content).strip() + "\n"


def generic_json_to_markdown(data: dict | list, filepath: Path) -> str:
    """
    Convert JSON không phải article sang Markdown.
    Dùng cho manifest_legal_sources.json, manifest_news_sources.json hoặc JSON bất kỳ.
    """
    title = filepath.stem

    header = build_metadata_header(
        title=title,
        source_path=filepath,
        extra_metadata={
            "original_extension": ".json",
            "converter": "generic_json_converter",
        },
    )

    body = [
        "## JSON Content",
        "",
        "```json",
        json.dumps(data, ensure_ascii=False, indent=2),
        "```",
        "",
    ]

    return header + "\n".join(body)


def is_article_json(data: Any) -> bool:
    """
    Nhận diện JSON bài báo từ Task 2.
    Điều kiện tối thiểu: là dict và có content_markdown/content hoặc url/title.
    """
    if not isinstance(data, dict):
        return False

    article_keys = {"url", "title", "date_crawled", "content_markdown", "content"}
    return len(article_keys.intersection(data.keys())) >= 2


def convert_json_file(filepath: Path) -> Path:
    """Convert JSON file sang Markdown."""
    output_path = get_output_path(filepath)
    ensure_parent_dir(output_path)

    print(f"Converting JSON: {filepath.relative_to(LANDING_DIR)}")

    data = load_json_file(filepath)

    if is_article_json(data):
        markdown_text = json_article_to_markdown(data, filepath)
    else:
        markdown_text = generic_json_to_markdown(data, filepath)

    output_path.write_text(markdown_text, encoding="utf-8")
    print(f"  ✓ Saved: {output_path}")
    return output_path


def iter_landing_files():
    """Scan toàn bộ file trong data/landing/."""
    if not LANDING_DIR.exists():
        print(f"⚠ LANDING_DIR không tồn tại: {LANDING_DIR}")
        return

    for filepath in LANDING_DIR.rglob("*"):
        if filepath.is_file():
            yield filepath


def should_convert(filepath: Path) -> bool:
    """Kiểm tra file có thuộc định dạng cần convert không."""
    suffix = filepath.suffix.lower()
    return suffix in SUPPORTED_DOC_EXTENSIONS or suffix == SUPPORTED_JSON_EXTENSION


def convert_all():
    """Convert toàn bộ PDF/DOCX/DOC/JSON trong data/landing/ sang Markdown."""
    print("=" * 60)
    print("Task 3: Convert landing files to Markdown")
    print("=" * 60)

    setup_output_directory()

    md = MarkItDown()

    converted = []
    failed = []
    skipped = []

    for filepath in iter_landing_files():
        if not should_convert(filepath):
            skipped.append(str(filepath))
            continue

        try:
            suffix = filepath.suffix.lower()

            if suffix in SUPPORTED_DOC_EXTENSIONS:
                output_path = convert_document_file(filepath, md)
            elif suffix == SUPPORTED_JSON_EXTENSION:
                output_path = convert_json_file(filepath)
            else:
                skipped.append(str(filepath))
                continue

            converted.append(
                {
                    "input": str(filepath),
                    "output": str(output_path),
                    "status": "success",
                }
            )

        except Exception as exc:
            error_message = str(exc)
            failed.append(
                {
                    "input": str(filepath),
                    "status": "failed",
                    "error": error_message,
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  ✗ Failed: {filepath}")
            print(f"    Error: {error_message}")

    manifest = {
        "converted_at": now_iso(),
        "landing_dir": str(LANDING_DIR),
        "output_dir": str(OUTPUT_DIR),
        "total_converted": len(converted),
        "total_failed": len(failed),
        "total_skipped": len(skipped),
        "converted": converted,
        "failed": failed,
        "skipped": skipped,
    }

    manifest_path = OUTPUT_DIR / "manifest_standardized.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Converted: {len(converted)}")
    print(f"✗ Failed:    {len(failed)}")
    print(f"↷ Skipped:   {len(skipped)}")
    print(f"✓ Manifest:  {manifest_path}")
    print(f"✓ Done. Output tại: {OUTPUT_DIR}")


def convert_legal_docs():
    """
    Convert riêng legal docs nếu cần chạy từng phần.
    Vẫn giữ để tương thích với skeleton ban đầu.
    """
    md = MarkItDown()
    legal_dir = LANDING_DIR / "legal"

    if not legal_dir.exists():
        print(f"⚠ Không tìm thấy thư mục legal: {legal_dir}")
        return

    for filepath in legal_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
            convert_document_file(filepath, md)


def convert_news_articles():
    """
    Convert riêng news articles nếu cần chạy từng phần.
    Vẫn giữ để tương thích với skeleton ban đầu.
    """
    news_dir = LANDING_DIR / "news"

    if not news_dir.exists():
        print(f"⚠ Không tìm thấy thư mục news: {news_dir}")
        return

    for filepath in news_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() == SUPPORTED_JSON_EXTENSION:
            convert_json_file(filepath)


if __name__ == "__main__":
    convert_all()
