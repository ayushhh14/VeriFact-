from __future__ import annotations

from io import BytesIO
import re
import fitz


def extract_pdf_pages(pdf_bytes: bytes, source_name: str) -> list[dict]:
    """Extract text page-by-page, preserving source name and one-based page number."""
    pages = []
    with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            # Remove repeated whitespace while retaining paragraph boundaries.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if text:
                pages.append({
                    "source": source_name,
                    "page": page_number,
                    "text": text,
                })
    return pages


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 180,
    overlap: int = 35,
) -> list[dict]:
    """Create word-based chunks while retaining source/page provenance."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks = []
    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append({
                    "id": f'{page["source"]}::p{page["page"]}::c{len(chunks)}',
                    "source": page["source"],
                    "page": page["page"],
                    "text": text,
                })
            if end == len(words):
                break
            start = end - overlap
    return chunks
