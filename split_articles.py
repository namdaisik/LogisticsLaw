import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader, PdfWriter


ARTICLE_REGEX = re.compile(
    r"(?m)^\s*(제\s*\d+\s*조(?:의\s*\d+)*)\s*(?:\(([^)\n]*)\))?"
)
INVALID_FILENAME_CHARS = re.compile(r"[\\/:*?\"<>|]")


def list_pdf_files(target: Path) -> List[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    pdfs: List[Path] = []
    for path in target.rglob("*.pdf"):
        if path.is_file():
            pdfs.append(path)
    return sorted(pdfs)


def normalize_heading(raw: str, title: str | None) -> str:
    base = raw.replace(" ", "")
    if title:
        return f"{base}({title.strip()})"
    return base


def slugify_heading(heading: str, index: int) -> str:
    trimmed = heading.replace(" ", "")
    trimmed = trimmed.replace("(", "_").replace(")", "")
    trimmed = INVALID_FILENAME_CHARS.sub("_", trimmed)
    trimmed = trimmed.strip("._") or f"article_{index+1}"
    prefix = f"{index+1:03d}"
    return f"{prefix}_{trimmed[:120]}.pdf"


def extract_articles(reader: PdfReader) -> List[Tuple[int, str]]:
    articles: List[Tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text:
            continue
        for match in ARTICLE_REGEX.finditer(text):
            heading = normalize_heading(match.group(1), match.group(2))
            if not heading:
                continue
            articles.append((page_index, heading))
    return articles


def page_range_for_article(idx: int, articles: List[Tuple[int, str]], total_pages: int) -> Tuple[int, int]:
    start = articles[idx][0]
    if idx + 1 < len(articles):
        next_start = articles[idx + 1][0]
        if next_start <= start:
            return start, start
        return start, next_start - 1
    return start, total_pages - 1


def split_pdf(pdf_path: Path, output_root: Path, overwrite: bool = False) -> None:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        print(f"[ERROR] {pdf_path}: failed to open PDF ({exc})")
        return

    articles = extract_articles(reader)
    if not articles:
        print(f"[WARN] {pdf_path}: no article headings matched; skipping")
        return

    dest_dir = output_root / pdf_path.stem
    dest_dir.mkdir(parents=True, exist_ok=True)

    total_pages = len(reader.pages)
    for idx, (_, heading) in enumerate(articles):
        start, end = page_range_for_article(idx, articles, total_pages)
        writer = PdfWriter()
        for page_num in range(start, end + 1):
            writer.add_page(reader.pages[page_num])

        filename = slugify_heading(heading, idx)
        output_path = dest_dir / filename

        if output_path.exists() and not overwrite:
            print(f"[SKIP] {output_path} already exists")
            continue

        with open(output_path, "wb") as f:
            writer.write(f)

        page_count = end - start + 1
        print(f"[OK] {output_path} ({page_count} page{'s' if page_count > 1 else ''})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split logistics-law PDFs into per-article PDFs")
    parser.add_argument("target", nargs="?", default=".", help="PDF file or root directory to process")
    parser.add_argument("--output", default="article_splits", help="Output directory for generated PDFs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-article PDFs")
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    output_root = Path(args.output).resolve()

    pdfs = list_pdf_files(target_path)
    if not pdfs:
        print(f"No PDFs found under {target_path}")
        return

    print(f"Processing {len(pdfs)} PDF(s) from {target_path}")
    for pdf_path in pdfs:
        print(f"\n=== {pdf_path} ===")
        split_pdf(pdf_path, output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

