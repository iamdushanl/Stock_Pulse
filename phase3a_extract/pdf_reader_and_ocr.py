"""
Phase 3A Stage 3, 4 & 5 – Unified Vector Text + Windows Media OCR Reader
=======================================================================
Extracts spatial word bounding boxes (`WordToken`) in uniform 72-pt PDF coordinates
from both machine-readable vector PDFs (`pymupdf`) and scanned or font-corrupted
PDF pages (`winocr` Windows Media OCR at 220 DPI), with deterministic caching and
per-page extraction quality & confidence metadata.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import winocr
from PIL import Image

GARBLE_PATTERN = re.compile(
    r"(?:\b202[sSlIo]\b|\(\d+[,.\d]*[sIlOo\]]|[\d,]+[sIlOo]\d|\d+[,\.]\d+['\]\\]|[A-Za-z][\'\\`][A-Za-z])"
)


@dataclass
class WordToken:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    method: str  # "Text" or "OCR"
    conf: float


@dataclass
class PageData:
    page_num: int
    width: float
    height: float
    tokens: list[WordToken]
    method: str  # "Text" | "OCR"
    ocr_used: bool
    ocr_reason: str
    text_quality_score: float
    raw_text: str


async def _run_winocr_on_page(page: pymupdf.Page, dpi: int = 220) -> tuple[list[dict[str, Any]], float]:
    """Render page to grayscale/RGB PNG at `dpi` and run Windows Media OCR cleanly."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    scale = 72.0 / float(dpi)
    res = await winocr.recognize_pil(img, "en")

    ocr_tokens: list[dict[str, Any]] = []
    clean_words = 0
    total_words = 0
    for ln in res.lines:
        for w in ln.words:
            txt = str(w.text).strip()
            if not txt:
                continue
            r = w.bounding_rect
            total_words += 1
            if re.match(r"^[A-Za-z0-9(),.\-%/'&:+]+$", txt):
                clean_words += 1
            ocr_tokens.append({
                "x0": round(float(r.x) * scale, 2),
                "y0": round(float(r.y) * scale, 2),
                "x1": round(float(r.x + r.width) * scale, 2),
                "y1": round(float(r.y + r.height) * scale, 2),
                "text": txt,
            })
    clean_ratio = float(clean_words) / float(max(1, total_words))
    # Cap pure OCR page quality strictly below vector 1.0 so downstream review sees provenance honestly
    conf = round(max(0.45, min(0.88, 0.45 + 0.43 * clean_ratio)), 3)
    if total_words == 0:
        conf = 0.0
    return ocr_tokens, conf


def _needs_ocr_fallback(page: pymupdf.Page, raw_text: str, total_pages: int) -> tuple[bool, str]:
    """Decide deterministically whether a PDF page requires OCR fallback."""
    stripped = raw_text.strip()
    cid_cnt = stripped.count("(cid:")
    if cid_cnt >= 3:
        return True, f"CID_ENCODING_CORRUPTION ({cid_cnt} cid tokens)"

    page_area = max(1.0, float(page.rect.width * page.rect.height))
    large_img_area = 0.0
    for info in page.get_image_info(xrefs=False):
        bx = info.get("bbox")
        if bx and len(bx) == 4:
            large_img_area += max(0.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
    img_cov = large_img_area / page_area

    garb_cnt = len(GARBLE_PATTERN.findall(stripped))
    if garb_cnt >= 3 or (img_cov >= 0.50 and garb_cnt >= 2):
        return True, f"CORRUPTED_EMBEDDED_OCR_GLYPHS ({garb_cnt} garbled tokens, img_cov={img_cov:.2f})"

    if len(stripped) < 160 and img_cov >= 0.45:
        if page.number == 0 and total_pages > 2 and len(stripped) >= 20:
            return False, ""
        return True, f"SCANNED_IMAGE_PAGE (chars={len(stripped)}, img_cov={img_cov:.2f})"

    return False, ""


async def _extract_pdf_async(
    pdf_path: Path,
    file_sha256: str,
    ocr_cache_dir: Path | None = None,
    max_ocr_pages: int = 18,
    dpi: int = 300,
) -> tuple[list[PageData], dict[str, Any]]:
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    pages_out: list[PageData] = []
    ocr_pages_count = 0
    ocr_words_count = 0
    ocr_confs: list[float] = []

    if ocr_cache_dir is not None:
        ocr_cache_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(total_pages):
        page = doc[idx]
        pno = idx + 1
        rect = page.rect
        raw_text = page.get_text("text") or ""
        need_ocr, reason = _needs_ocr_fallback(page, raw_text, total_pages)

        if need_ocr and ocr_pages_count < max_ocr_pages:
            cache_file = (
                ocr_cache_dir / f"{file_sha256[:24]}_p{pno}_dpi{dpi}.json"
                if ocr_cache_dir is not None
                else None
            )
            if cache_file is not None and cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                raw_ocr_tokens = cached["tokens"]
                page_conf = float(cached["conf"])
            else:
                raw_ocr_tokens, page_conf = await _run_winocr_on_page(page, dpi=dpi)
                if cache_file is not None:
                    cache_file.write_text(
                        json.dumps({"conf": page_conf, "tokens": raw_ocr_tokens}),
                        encoding="utf-8",
                    )

            tokens = [
                WordToken(
                    x0=t["x0"],
                    y0=t["y0"],
                    x1=t["x1"],
                    y1=t["y1"],
                    text=t["text"],
                    method="OCR",
                    conf=page_conf,
                )
                for t in raw_ocr_tokens
            ]
            # Build simple reading-order text string from OCR tokens
            ocr_text = "\n".join(
                " ".join(t.text for t in row)
                for row in _group_tokens_into_rows(tokens, y_tol=5.5)
            )
            ocr_pages_count += 1
            ocr_words_count += len(tokens)
            ocr_confs.append(page_conf)
            pages_out.append(
                PageData(
                    page_num=pno,
                    width=float(rect.width),
                    height=float(rect.height),
                    tokens=tokens,
                    method="OCR",
                    ocr_used=True,
                    ocr_reason=reason,
                    text_quality_score=page_conf,
                    raw_text=ocr_text,
                )
            )
        else:
            raw_words = page.get_text("words") or []
            tokens = [
                WordToken(
                    x0=float(w[0]),
                    y0=float(w[1]),
                    x1=float(w[2]),
                    y1=float(w[3]),
                    text=str(w[4]).strip(),
                    method="Text",
                    conf=0.98,
                )
                for w in raw_words
                if str(w[4]).strip()
            ]
            pages_out.append(
                PageData(
                    page_num=pno,
                    width=float(rect.width),
                    height=float(rect.height),
                    tokens=tokens,
                    method="Text",
                    ocr_used=False,
                    ocr_reason="",
                    text_quality_score=0.98 if tokens else 0.0,
                    raw_text=raw_text,
                )
            )

    doc.close()
    diag = {
        "total_pages": total_pages,
        "ocr_pages_processed": ocr_pages_count,
        "ocr_words_extracted": ocr_words_count,
        "ocr_mean_confidence": round(sum(ocr_confs) / len(ocr_confs), 3) if ocr_confs else None,
    }
    return pages_out, diag


def _group_tokens_into_rows(tokens: list[WordToken], y_tol: float = 4.0) -> list[list[WordToken]]:
    """Cluster spatial WordTokens into horizontal rows ordered top-to-bottom, left-to-right."""
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: ((t.y0 + t.y1) * 0.5, t.x0))
    rows: list[list[WordToken]] = []
    cur_row: list[WordToken] = [ordered[0]]
    cur_y = (ordered[0].y0 + ordered[0].y1) * 0.5
    for tok in ordered[1:]:
        yc = (tok.y0 + tok.y1) * 0.5
        if abs(yc - cur_y) <= y_tol:
            cur_row.append(tok)
            cur_y = (cur_y * (len(cur_row) - 1) + yc) / len(cur_row)
        else:
            rows.append(sorted(cur_row, key=lambda t: t.x0))
            cur_row = [tok]
            cur_y = yc
    if cur_row:
        rows.append(sorted(cur_row, key=lambda t: t.x0))
    return rows


def extract_pdf_pages_tokens(
    pdf_path: Path,
    file_sha256: str,
    ocr_cache_dir: Path | None = None,
    max_ocr_pages: int = 18,
    dpi: int = 300,
) -> tuple[list[PageData], dict[str, Any]]:
    """Synchronous wrapper returning normalized spatial WordToken pages & OCR stats."""
    return asyncio.run(
        _extract_pdf_async(
            pdf_path=pdf_path,
            file_sha256=file_sha256,
            ocr_cache_dir=ocr_cache_dir,
            max_ocr_pages=max_ocr_pages,
            dpi=dpi,
        )
    )
