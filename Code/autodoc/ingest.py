"""PDF ingestion: page-by-page text extraction with page numbers preserved.

Handles invalid, empty, encrypted and scanned (little/no text) PDFs without crashing.
OCR is OPTIONAL and only used if pytesseract + Pillow + the Tesseract program are installed.
"""
import hashlib
import os
import re
import shutil
from collections import Counter

import pymupdf

from .config import CFG

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\s+\u2022\s+")
HEAD_INLINE = re.compile(r"^(\d+(?:\.\d+){0,4})\s+([A-Z][^\n]{2,80})$")
HEAD_NUM_ONLY = re.compile(r"^\d+(?:\.\d+){0,4}$")


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _tesseract_cmd():
    found = shutil.which("tesseract")
    if found:
        return found
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(p):
            return p
    return None


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return _tesseract_cmd() is not None


def _ocr_page(page) -> str:
    import io
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd()
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _clean_lines(raw: str):
    raw = raw.replace("\x00", "")
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)  # de-hyphenate line-end splits (may merge real hyphens)
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _remove_repeated_lines(pages):
    """Drop running headers/footers: lines (digits normalised) repeated on most pages."""
    n = len(pages)
    if n < CFG["ingestion"]["repeat_line_min_pages"]:
        return 0
    counts = Counter()
    for p in pages:
        for key in {re.sub(r"\d+", "#", ln) for ln in p["lines"]}:
            counts[key] += 1
    limit = CFG["ingestion"]["repeat_line_fraction"] * n
    bad = {k for k, c in counts.items() if c >= limit}
    removed = 0
    for p in pages:
        keep = [ln for ln in p["lines"] if re.sub(r"\d+", "#", ln) not in bad]
        removed += len(p["lines"]) - len(keep)
        p["lines"] = keep
    return removed


def _headings(lines):
    """Return [(number, title, first_idx, last_idx)] for numbered heading lines."""
    heads, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if re.search(r"(\.\s?){3,}", ln):  # table-of-contents leader dots
            i += 1
            continue
        m = HEAD_INLINE.match(ln)
        if m and not ln.endswith("."):
            heads.append((m.group(1), m.group(2), i, i))
        elif HEAD_NUM_ONLY.match(ln) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if (nxt[:1].isupper() and len(nxt) <= 80 and not nxt.endswith(".")
                    and not re.search(r"(\.\s?){3,}", nxt)):
                heads.append((ln, nxt, i, i + 1))
                i += 1
        i += 1
    return heads


def split_sentences(text: str):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def ingest_pdf(name: str, data: bytes, use_ocr: bool = False) -> dict:
    """Return a dict describing the document. Never raises for bad input."""
    doc_info = {"name": name, "hash": file_hash(data), "status": "ok", "message": "",
                "page_count": 0, "pages": [], "low_text_pages": [], "ocr_pages": [],
                "headings": [], "removed_header_footer_lines": 0}
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        doc_info.update(status="invalid", message=f"Could not open as PDF: {exc}")
        return doc_info
    try:
        if doc.needs_pass:
            doc_info.update(status="encrypted", message="Password-protected PDF; not processed.")
            return doc_info
        doc_info["page_count"] = doc.page_count
        if doc.page_count == 0:
            doc_info.update(status="empty", message="PDF has no pages.")
            return doc_info
        low = CFG["ingestion"]["low_text_chars"]
        for i, page in enumerate(doc, start=1):
            try:
                raw = page.get_text("text")
            except Exception:
                raw = ""
            source = "text"
            if len(raw.strip()) < low:
                doc_info["low_text_pages"].append(i)
                source = "none"
                if use_ocr and ocr_available():
                    try:
                        raw = _ocr_page(page)
                        if len(raw.strip()) >= low:
                            source = "ocr"
                            doc_info["ocr_pages"].append(i)
                    except Exception:
                        pass
            doc_info["pages"].append({"page": i, "lines": _clean_lines(raw), "source": source})
    except Exception as exc:
        doc_info.update(status="invalid", message=f"Extraction failed: {exc}")
        return doc_info
    finally:
        doc.close()

    pages = doc_info["pages"]
    doc_info["removed_header_footer_lines"] = _remove_repeated_lines(pages)
    cur = ("", "Front matter")
    for p in pages:
        heads = [h for h in _headings(p["lines"]) if int(h[0].split(".")[0]) <= 40]
        if len(heads) > 12:  # looks like a table-of-contents page: do not treat as real headings
            heads = []
        head_at = {h[2]: h for h in heads}
        skip = {k for h in heads for k in range(h[2], h[3] + 1)}
        p["start_section"] = f"{cur[0]} {cur[1]}".strip()
        segments, buf = [], []

        def close(section):
            text = re.sub(r"\s+", " ", " ".join(buf)).strip()
            if text:
                segments.append({"section": section, "text": text, "is_heading": False})
            buf.clear()

        for idx, ln in enumerate(p["lines"]):
            if idx in head_at:
                close(f"{cur[0]} {cur[1]}".strip())
                h = head_at[idx]
                cur = (h[0], h[1])
                doc_info["headings"].append({"page": p["page"], "number": h[0], "title": h[1]})
                segments.append({"section": f"{cur[0]} {cur[1]}".strip(), "text": f"{h[0]} {h[1]}",
                                 "is_heading": True})
            elif idx not in skip:
                buf.append(ln)
        close(f"{cur[0]} {cur[1]}".strip())
        p["segments"] = segments
        p["end_section"] = f"{cur[0]} {cur[1]}".strip()
        p["text"] = re.sub(r"\s+", " ", " ".join(ln for ln in p["lines"])).strip()
        p["chars"] = len(p["text"])

    if all(p["chars"] == 0 for p in pages):
        doc_info.update(status="no_text",
                        message="No extractable text (likely scanned). Enable OCR if installed.")
    elif doc_info["low_text_pages"]:
        remaining = [p for p in doc_info["low_text_pages"] if p not in doc_info["ocr_pages"]]
        if remaining:
            doc_info.update(status="partial",
                            message=f"{len(remaining)} page(s) with little/no text (possibly scanned): "
                                    f"{remaining[:15]}")
    return doc_info


def sentences_of(doc_info: dict):
    """All sentences with page + section, in reading order (used by extraction/compare/checks).
    Headings are included as their own entries (is_heading=True)."""
    out = []
    for p in doc_info["pages"]:
        for seg in p.get("segments", []):
            if seg["is_heading"]:
                out.append({"doc": doc_info["name"], "page": p["page"], "section": seg["section"],
                            "text": seg["text"], "is_heading": True})
                continue
            for s in split_sentences(seg["text"]):
                out.append({"doc": doc_info["name"], "page": p["page"], "section": seg["section"],
                            "text": s, "is_heading": False})
    return out
