#!/usr/bin/env python3
"""
GeM bid constraint extractor + adversarial validator (local Transformers / OpenAI-compatible API; derived from v31 pipeline).

Qwen performs the semantic constraint extraction useful for supplier/product matching.
The final_constraints.csv output is a canonicalized, IndiaMART-facing semantic set.
Deterministic PDF/table parsing is used only as source-document ingestion/evidence; it is
not allowed to become the final semantic constraint representation. Qwen is given the largest
combined document context that fits safely in its context window, rather than many tiny calls.
Use --no_llm to run with zero API calls (deterministic evidence only).

    pip install pdfplumber openai tqdm pytesseract pymupdf pillow openpyxl requests
    # Tesseract is a separate system executable; install it too (Windows: UB Mannheim installer)
    # If Tesseract is installed but not on PATH (Windows): set TESSERACT_CMD to its full executable path
    export OPENAI_API_KEY=...  OPENAI_BASE_URL=https://openrouter.ai/api/v1   # Qwen via OpenRouter
    python extract_constraints_hybrid.py --input_dir ./bid_pdfs --output_dir ./out --no_llm
    python extract_constraints_hybrid.py --input_dir ./bid_pdfs --output_dir ./out --model qwen/qwen3.8-27b:free --concurrency 1

Resumable: bids with an existing schema-v30 JSON in output_dir are skipped. Linked documents are stored under <output>/<BID_ID>/linked_documents/.
"""
import argparse, asyncio, csv, json, os, re, sys, traceback, io, tempfile, hashlib, time
from urllib.parse import urljoin, urlparse
import requests
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Golden-parameter wording varies across GeM templates. Keep this deliberately
# conservative: we only treat a bid as having an explicit golden marker when
# the document itself links "golden" (or mandatory) parameters/specifications
# with bold/bolded formatting.
GOLDEN_NOTE_PATTERNS = [
    re.compile(r"\bbold(?:ed)?\b.{0,160}\bgolden\s+(?:parameters?|specifications?|specs?)\b", re.I | re.S),
    re.compile(r"\bgolden\s+(?:parameters?|specifications?|specs?)\b.{0,160}\bbold(?:ed)?\b", re.I | re.S),
    re.compile(r"\bbold(?:ed)?\b.{0,160}\bmandatory\s+(?:parameters?|specifications?|specs?)\b", re.I | re.S),
]
NONGOLDEN_NOTE_PATTERNS = [
    re.compile(r"\bnon[- ]golden\s+(?:parameters?|specifications?|specs?)\b.{0,120}\b(?:preferred|prefer|desirable)\b", re.I | re.S),
    re.compile(r"\b(?:preferred|prefer|desirable)\b.{0,120}\bnon[- ]golden\s+(?:parameters?|specifications?|specs?)\b", re.I | re.S),
]
UNIT_RE = re.compile(r"\(([^)]+)\)\s*$")


def clean(s):
    return re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip()


def snake(s):
    s = re.sub(r"\([^)]*\)", " ", s)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def to_num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


# Requires: pip install json-repair for robust Qwen JSON recovery.

# ----------------------------------------------------------------------------- OCR engine discovery

def resolve_tesseract(required: bool = False) -> str | None:
    """Find and verify Tesseract even when it is installed outside PATH (common on Windows)."""
    import shutil
    candidates = []
    for env_name in ("TESSERACT_CMD", "TESSERACT_PATH", "TESSERACT_EXE"):
        value = os.getenv(env_name)
        if value:
            candidates.append(value.strip().strip('"'))
    try:
        import pytesseract
        configured = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
        if configured and configured != "tesseract":
            candidates.append(str(configured).strip().strip('"'))
    except Exception:
        pytesseract = None
    located = shutil.which("tesseract")
    if located:
        candidates.append(located)
    # Typical Windows installs, including Chocolatey and per-user installs.
    program_roots = [os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)"), os.getenv("LOCALAPPDATA")]
    for root in filter(None, program_roots):
        candidates.extend([
            str(Path(root) / "Tesseract-OCR" / "tesseract.exe"),
            str(Path(root) / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
            str(Path(root) / "Tesseract-OCR" / "tesseract"),
        ])
    candidates.extend([
        "/usr/bin/tesseract", "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract",
        "/opt/local/bin/tesseract", "/snap/bin/tesseract",
    ])
    seen = set()
    errors = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        # Accept a bare executable name only if it is resolvable by PATH.
        executable = candidate if os.path.isfile(candidate) else shutil.which(candidate)
        if not executable or not os.path.isfile(executable):
            continue
        try:
            if pytesseract is None:
                import pytesseract as _pytesseract
            else:
                _pytesseract = pytesseract
            _pytesseract.pytesseract.tesseract_cmd = executable
            # Validate the actual binary now, instead of failing silently on every page.
            _pytesseract.get_tesseract_version()
            return executable
        except Exception as exc:
            errors.append(f"{executable}: {exc}")
    if required:
        detail = " | ".join(errors[:3])
        raise RuntimeError(
            "Tesseract OCR is required for this run but could not be executed. "
            "The script checked TESSERACT_CMD/TESSERACT_PATH/TESSERACT_EXE, PATH, and common install folders. "
            "Set TESSERACT_CMD to the full path of tesseract.exe (Windows) or the tesseract binary."
            + (f" Probe errors: {detail}" if detail else "")
        )
    return None


def configure_tesseract(required: bool = False) -> str | None:
    """Configure pytesseract consistently in the parent and worker processes."""
    executable = resolve_tesseract(required=required)
    if executable:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = executable
    return executable


# ----------------------------------------------------------------------------- PDF parsing
def is_bold_cell(page, bbox):
    try:
        chars = page.crop(bbox, strict=False).chars
    except Exception:
        return False
    chars = [c for c in chars if c["text"].strip()]
    return bool(chars) and sum("bold" in c["fontname"].lower() for c in chars) / len(chars) > 0.5


def parse_allowed(raw):
    """'4, 8, 16, 32 Or higher' -> (['4','8','16','32'], True); '*' -> (None, False, any=True)"""
    raw = clean(raw)
    if raw in ("", "*"):
        return None, False, True
    higher = bool(re.search(r"\bor higher\b", raw, re.I))
    raw = re.sub(r"\s*\bor higher\b", "", raw, flags=re.I)
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return vals, higher, False


def _annotation_uri(ann):
    if not isinstance(ann, dict):
        return None
    uri = ann.get("uri")
    if uri:
        return uri
    action = ann.get("A") or {}
    if isinstance(action, dict):
        return action.get("URI") or action.get("uri")
    return None


def _annotation_rect(ann):
    if not isinstance(ann, dict):
        return None
    rect = ann.get("Rect") or ann.get("rect")
    if rect and len(rect) >= 4:
        try:
            return tuple(float(x) for x in rect[:4])
        except Exception:
            return None
    return None


def _looks_like_spec_link(label: str, context: str, uri: str) -> bool:
    blob = clean(" ".join([label or "", context or "", uri or ""])).lower()
    spec_words = (
        "technical specification", "technical specifications", "technical spec",
        "tech spec", "specification document", "specification", "spec document",
        "spec sheet", "product specification", "product spec", "technical data",
        "datasheet", "data sheet", "boq", "bill of quantities", "line item"
    )
    if any(w in blob for w in spec_words):
        return True
    # "View File" is ambiguous, so require specification context in the nearby text/URL.
    if re.search(r"\bview\s+file\b", blob) and re.search(
        r"\b(tech(?:nical)?|spec(?:ification)?|datasheet|boq|line\s*item)\b", blob
    ):
        return True
    return False


def _discover_spec_links(page):
    links = []
    try:
        annots = page.annots or []
    except Exception:
        annots = []
    for ann in annots:
        uri = _annotation_uri(ann)
        if not uri:
            continue
        rect = _annotation_rect(ann)
        label = ""
        context = ""
        try:
            if rect:
                x0, y0, x1, y1 = rect
                # Expand around the annotation so a row such as
                # "Technical Specifications | Specification Document | View File"
                # is visible even when only "View File" is inside the link bbox.
                bbox = (
                    max(0, x0 - 220), max(0, y0 - 12),
                    min(page.width, x1 + 220), min(page.height, y1 + 12)
                )
                context = clean(page.crop(bbox, strict=False).extract_text() or "")
                label = clean(page.crop(rect, strict=False).extract_text() or "")
        except Exception:
            pass
        if _looks_like_spec_link(label, context, uri):
            links.append({
                "url": uri,
                "label": label,
                "context": context,
                "page": getattr(page, "page_number", None),
            })
    # Deduplicate by URL.
    seen, out = set(), []
    for x in links:
        u = x["url"]
        if u not in seen:
            seen.add(u)
            out.append(x)
    return out


def parse_pdf(pdf_path: str, discover_links: bool = True, source_document: str = None) -> dict:
    import pdfplumber

    out = {"header": {}, "spec_rows": [], "consignees": [], "emd_required": None, "epbg_required": None,
           "full_text": "", "has_golden_note": False, "has_nongolden_note": False, "ocr_pages": 0,
           "spec_links": [], "pdf_links": [], "source_document": source_document or Path(pdf_path).name}
    texts, group, in_spec, parent = [], None, False, None
    flag_rows = []

    page_quality = []
    with pdfplumber.open(pdf_path) as pdf:
        # Re-open PyMuPDF only once if installed; it often recovers text that
        # pdfplumber misses in complex technical PDFs.
        fallback_pdf = None
        try:
            # PyMuPDF renamed its import to `pymupdf`; keep `fitz` as a fallback
            # for older installations without emitting the deprecation warning.
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            fallback_pdf = fitz.open(pdf_path)
        except Exception:
            fallback_pdf = None

        for pno, page in enumerate(pdf.pages, 1):
            pdfplumber_text = page.extract_text(layout=False) or ""
            txt = pdfplumber_text
            native_chars = len(re.sub(r"\s+", "", txt))
            # Compare independent PDF text engines before invoking OCR. Technical
            # drawings/tables are often invisible to pdfplumber but readable to MuPDF.
            pymupdf_chars = 0
            fallback_text = ""
            if native_chars < 500 and fallback_pdf is not None:
                try:
                    fallback_text = fallback_pdf[pno - 1].get_text("text", sort=True) or ""
                    pymupdf_chars = len(re.sub(r"\s+", "", fallback_text))
                except Exception as e:
                    out.setdefault("text_engine_errors", []).append({"page": pno, "engine": "pymupdf_text", "error": repr(e)})
                if pymupdf_chars > native_chars:
                    txt, native_chars = fallback_text, pymupdf_chars
            quality = {"page": pno, "pdfplumber_chars": len(re.sub(r"\s+", "", pdfplumber_text)),
                       "native_chars": native_chars, "pymupdf_chars": pymupdf_chars,
                       "extracted_chars": native_chars, "ocr_attempted": False, "ocr_used": False, "ocr_chars": 0}
            # Explicitly locate Tesseract; pytesseract can import successfully while
            # its executable is absent from the subprocess PATH in spawned workers.
            # OCR pages with sparse text and any page in a globally low-yield PDF.
            try:
                page_count = len(pdf.pages)
                 # v30: OCR only pages with sparse native text. OCRing every page
                # added minutes without improving already-readable bid/SLA pages.
                # Image-only technical-spec pages still get OCR.
                needs_ocr = native_chars < 180
                if needs_ocr:
                    quality["ocr_attempted"] = True
                    import pytesseract
                    tess = configure_tesseract(required=True)
                    pytesseract.pytesseract.tesseract_cmd = tess
                    image = None
                    if fallback_pdf is not None:
                        try:
                            pix = fallback_pdf[pno - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                            from PIL import Image
                            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        except Exception as e:
                            quality["render_error"] = repr(e)
                    if image is None:
                        # Avoid ImageMagick/ghostscript dependency where possible.
                        image = page.to_image(resolution=300).original
                    ocr_txt = pytesseract.image_to_string(image, config="--psm 6", timeout=20) or ""
                    ocr_chars = len(re.sub(r"\s+", "", ocr_txt))
                    quality["ocr_used"] = True
                    quality["ocr_chars"] = ocr_chars
                    out["ocr_pages"] += 1
                    if ocr_chars > len(re.sub(r"\s+", "", txt)):
                        txt = ocr_txt
                        quality["extracted_chars"] = ocr_chars
            except Exception as e:
                out.setdefault("ocr_errors", []).append({"page": pno, "error": repr(e)})
            page_quality.append(quality)
            texts.append(f"--- Page {pno} ---\n{txt}")


            if discover_links:
                # Record every URI annotation, not only technical-spec links. Some GeM PDFs
                # expose the attachment only as a generic "View File" annotation.
                try:
                    annots = page.annots or []
                except Exception:
                    annots = []
                for ann in annots:
                    uri = _annotation_uri(ann)
                    if uri:
                        rect = _annotation_rect(ann)
                        context = ""
                        label = ""
                        try:
                            if rect:
                                x0, y0, x1, y1 = rect
                                bbox = (max(0, x0 - 220), max(0, y0 - 12),
                                        min(page.width, x1 + 220), min(page.height, y1 + 12))
                                context = clean(page.crop(bbox, strict=False).extract_text() or "")
                                label = clean(page.crop(rect, strict=False).extract_text() or "")
                        except Exception:
                            pass
                        out["pdf_links"].append({"url": uri, "label": label, "context": context, "page": pno})
                for link in _discover_spec_links(page):
                    link["page"] = pno
                    out["spec_links"].append(link)

            for tbl in page.find_tables():
                rows = tbl.extract()
                if not rows:
                    continue
                hdr = [clean(c) for c in rows[0]]
                if hdr and "bid details" in hdr[0].lower():
                    for r in rows[1:]:
                        if len(r) >= 2 and r[0]:
                            out["header"][clean(r[0])] = clean(r[1])
                    continue

                is_spec_hdr = len(hdr) == 4 and "specification name" in hdr[1].lower()
                if is_spec_hdr:
                    in_spec = True
                if in_spec and len(rows[0]) == 4 and (is_spec_hdr or "s.no" not in hdr[0].lower()):
                    for ri, r in enumerate(rows):
                        if is_spec_hdr and ri == 0:
                            continue
                        if len(r) != 4 or not clean(r[1]):
                            continue
                        if clean(r[0]):
                            group = clean(r[0])
                        cells = tbl.rows[ri].cells if ri < len(tbl.rows) else [None] * 4
                        bold = is_bold_cell(page, cells[1]) if cells[1] else False
                        name, val, allowed = clean(r[1]), clean(r[2]), r[3]
                        vals, higher, anyv = parse_allowed(allowed)
                        out["spec_rows"].append({
                            "page": pno, "group": group, "name": name, "value": val,
                            "allowed_raw": clean(allowed), "allowed_values": vals,
                            "allows_higher": higher, "allowed_any": anyv, "golden": bold,
                            "source_document": out["source_document"],
                            "source_kind": "main_bid",
                        })
                    continue

                if len(hdr) >= 5 and "s.no" in hdr[0].lower() and "consignee" in hdr[1].lower():
                    in_spec = False
                    for r in rows[1:]:
                        if len(r) >= 5 and clean(r[0]):
                            out["consignees"].append({"s_no": clean(r[0]), "consignee": clean(r[1]),
                                                      "address": clean(r[2]), "quantity": clean(r[3]),
                                                      "delivery_days": clean(r[4])})
                    continue

                if len(hdr) == 2 and "required" in hdr[0].lower():
                    flag_rows.append(hdr[1])

    out["full_text"] = "\n\n".join(texts)
    out["has_golden_note"] = any(p.search(out["full_text"]) for p in GOLDEN_NOTE_PATTERNS)
    out["has_nongolden_note"] = any(p.search(out["full_text"]) for p in NONGOLDEN_NOTE_PATTERNS)
    if flag_rows:
        out["emd_required"] = flag_rows[0]
        out["epbg_required"] = flag_rows[1] if len(flag_rows) > 1 else None
    # A bold specification is itself an explicit golden marker. The footnote is useful
    # confirmation, but should not be required if the PDF's formatting is recoverable.
    out["has_explicit_bold_golden"] = any(
        r["golden"] and r["value"] not in ("", "-", "NA", "N/A") for r in out["spec_rows"]
    )
    # Keep only unique links.
    seen = set()
    out["spec_links"] = [x for x in out["spec_links"]
                         if not (x["url"] in seen or seen.add(x["url"]))]
    seen_links, unique_links = set(), []
    for x in out.get("pdf_links", []):
        if x.get("url") and x["url"] not in seen_links:
            seen_links.add(x["url"]); unique_links.append(x)
    for url in re.findall(r"https?://[^\s<>\"']+", out.get("full_text", "")):
        url = url.rstrip(".,);]")
        if url not in seen_links:
            seen_links.add(url)
            unique_links.append({"url": url, "label": "text_url", "context": "", "page": None})
    out["pdf_links"] = unique_links
    out["page_quality"] = page_quality if "page_quality" in locals() else []
    out["native_text_chars"] = sum(x.get("native_chars", 0) for x in out["page_quality"])
    out["extracted_text_chars"] = len(re.sub(r"\s+", "", out.get("full_text", "")))
    out["page_count"] = len(out["page_quality"])
    out["ocr_attempted_pages"] = sum(1 for x in out["page_quality"] if x.get("ocr_attempted", False))
    out["ocr_successful_pages"] = sum(1 for x in out["page_quality"] if x.get("ocr_used", False))
    out["ingestion_warnings"] = []
    if out["page_count"] and out["extracted_text_chars"] < max(500, out["page_count"] * 40):
        out["ingestion_warnings"].append(
            f"low_text_yield:{out['extracted_text_chars']} nonspace chars across {out['page_count']} pages"
        )
    if out.get("ocr_errors"):
        out["ingestion_warnings"].append("one_or_more_ocr_pages_failed")
    return out


def free_text_sections(full_text: str) -> dict:
    """Pull the free-text parts: splitting, special terms (category ATC), buyer-specific ATC."""
    t = re.sub(r"--- Page \d+ ---\n?", "", full_text)
    t = re.sub(r"\n\s*\d+\s*/\s*\d+\s*(?=\n|$)", "\n", t)  # strip '6 / 7' page footers

    def between(start, ends):
        m = re.search(start, t, re.I)
        if not m:
            return ""
        rest = t[m.end():]
        e = min([x.start() for x in (re.search(p, rest, re.I) for p in ends) if x] or [len(rest)])
        return rest[:e].strip()

    return {
        "splitting": between(r"\bSplitting\b", [r"Special terms and conditions", r"Buyer Specific Additional"]),
        "special_terms": between(r"Special terms and conditions[^\n]*\n", [r"Buyer Specific Additional", r"This Bid is also governed"]),
        "buyer_atc": between(r"Buyer Specific Additional Terms and Conditions\s*\n", [r"This Bid is also governed", r"---Thank You---"]),
    }


def split_clauses(block: str) -> list:
    if not block:
        return []
    parts = re.split(r"(?:(?<=\s)|^)(?=\d{1,2}[\.\)]\s*(?:[A-Z(]|\d))", block)
    return [clean(p) for p in parts if len(clean(p)) > 15]


def extract_contacts(parsed: dict) -> list:
    """Extract explicit buyer/grievance/contact details without treating consignees as generic contacts."""
    text = parsed.get("full_text", "")
    contacts = []

    # Explicit email addresses anywhere in the bid.
    emails = list(dict.fromkeys(re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)))
    phone_re = r"(?:\+91[\s-]?)?[6-9]\d{9}\b"
    phones = list(dict.fromkeys(re.findall(phone_re, text)))

    # Label-driven extraction for common GeM contact fields.
    for m in re.finditer(
        r"(?P<label>contact\s+(?:person|details?)|grievance\s+redressal|buyer\s+contact|contact\s+email)"
        r"\s*[:\-]?\s*(?P<body>[^\n]{0,220})",
        text, re.I
    ):
        body = clean(m.group("body"))
        em = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", body, re.I)
        ph = re.search(phone_re, body)
        name = None
        nm = re.search(r"(?:contact\s+person|name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,80})", body, re.I)
        if nm:
            name = clean(nm.group(1))
        contacts.append({
            "contact_type": "grievance_contact" if "grievance" in m.group("label").lower() else "buyer_contact",
            "name": name,
            "email": em.group(0) if em else None,
            "phone": ph.group(0) if ph else None,
            "address": None,
            "raw_text": f"{m.group('label')}: {body}",
            "source_document": parsed.get("source_document"),
        })

    # If labels were not useful, preserve emails/phones as unassigned buyer contacts.
    if not contacts:
        for em in emails:
            contacts.append({
                "contact_type": "buyer_contact",
                "name": None, "email": em, "phone": None, "address": None,
                "raw_text": em, "source_document": parsed.get("source_document"),
            })
        for ph in phones:
            contacts.append({
                "contact_type": "buyer_contact",
                "name": None, "email": None, "phone": ph, "address": None,
                "raw_text": ph, "source_document": parsed.get("source_document"),
            })

    # Consignee/reporting officers are kept separately and are NOT promoted to buyer_contact.
    for c in parsed.get("consignees", []):
        if c.get("consignee"):
            contacts.append({
                "contact_type": "consignee_reporting_officer",
                "name": c["consignee"],
                "email": None,
                "phone": None,
                "address": c.get("address"),
                "raw_text": json.dumps(c, ensure_ascii=False),
                "source_document": parsed.get("source_document"),
            })
    return contacts


def extract_actual_bid_id(parsed: dict, pdf: Path) -> str:
    """Prefer the GeM bid number printed inside the PDF over the local filename."""
    candidates = []
    for k, v in (parsed.get("header") or {}).items():
        if "bid number" in str(k).lower() and v:
            candidates.append(str(v))
    candidates.append(parsed.get("full_text", "") or "")
    for blob in candidates:
        m = re.search(r"\bGEM/\d{4}/B/\d+\b", blob, re.I)
        if m:
            return m.group(0).upper()
    return pdf.stem


def safe_bid_folder_name(bid_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", bid_id).strip("_") or "unknown_bid"


def linked_filename(url: str, index: int) -> str:
    """Create a readable, collision-resistant filename for a linked document."""
    path = urlparse(url).path
    base = Path(path).name or f"linked_document_{index}"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    allowed = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".txt"}
    if Path(base).suffix.lower() not in allowed:
        base += ".bin"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    stem, ext = os.path.splitext(base)
    return f"{index:02d}_{stem[:100]}_{digest}{ext}"


def _download_linked_document(url: str, cache_dir: Path, filename: str | None = None) -> tuple[Path | None, str]:
    """Download a linked document while preserving its real format."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    target = cache_dir / (filename or f"{key}")
    try:
        if target.exists() and target.stat().st_size > 100:
            return target, ""
        r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.content
        ctype = (r.headers.get("content-type") or "").lower()

        # Prefer URL extension when the server returns a generic content type.
        ext = Path(urlparse(url).path).suffix.lower()
        allowed_exts = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".txt"}
        if ext not in allowed_exts and not any(x in ctype for x in (
            "pdf", "csv", "spreadsheet", "excel", "word", "officedocument", "text/plain"
        )):
            return None, ctype

        if ext == "":
            if "pdf" in ctype:
                target = target.with_suffix(".pdf")
            elif "csv" in ctype:
                target = target.with_suffix(".csv")
            elif "spreadsheet" in ctype or "excel" in ctype:
                target = target.with_suffix(".xlsx")
            elif "word" in ctype:
                target = target.with_suffix(".docx")
            else:
                target = target.with_suffix(".txt")
        target.write_bytes(data)
        return target, ctype
    except Exception:
        return None, ""


def _rows_to_spec_rows(rows, source_document: str, source_kind: str = "linked_document") -> list:
    """Turn simple parameter/value tables from linked office docs into spec_rows."""
    out = []
    for row_i, row in enumerate(rows, start=1):
        vals = [clean(str(x)) for x in row if x not in (None, "")]
        if len(vals) < 2:
            continue
        # Key/value form: Parameter | Value [| Allowed]
        name = vals[0]
        value = vals[1]
        allowed_raw = vals[2] if len(vals) >= 3 else value
        if len(name) > 160 or len(value) > 500:
            continue
        if re.search(r"^(parameter|specification|spec|attribute|item|description)$", name, re.I):
            continue
        allowed_values, allows_higher, allowed_any = parse_allowed(allowed_raw)
        out.append({
            "page": row_i,
            "group": None,
            "name": name,
            "value": value,
            "allowed_raw": allowed_raw,
            "allowed_values": allowed_values,
            "allows_higher": allows_higher,
            "allowed_any": allowed_any,
            "golden": False,
            "source_document": source_document,
            "source_kind": source_kind,
        })
    return out


def parse_technical_pdf_fast(pdf_path: str, source_document: str = None) -> dict:
    """Extract linked PDFs with PyMuPDF and concurrent OCR for sparse pages."""
    import fitz
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = {"header": {}, "spec_rows": [], "consignees": [], "emd_required": None,
           "epbg_required": None, "full_text": "", "has_golden_note": False,
           "has_nongolden_note": False, "ocr_pages": 0, "spec_links": [],
           "pdf_links": [], "source_document": source_document or Path(pdf_path).name}
    try:
        import pytesseract
        tess = configure_tesseract(required=False)
        if tess: pytesseract.pytesseract.tesseract_cmd = tess
    except Exception:
        pytesseract = None
        tess = None

    page_texts=[]; page_quality=[]; rendered={}; ocr_errors=[]
    with fitz.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf, 1):
            txt=page.get_text("text", sort=True) or ""
            native_chars=len(re.sub(r"\s+", "", txt))
            q={"page":pno,"native_chars":native_chars,"pymupdf_chars":native_chars,
               "extracted_chars":native_chars,"ocr_attempted":native_chars < 180,"ocr_used":False,"ocr_chars":0}
            page_texts.append(txt); page_quality.append(q)
            # OCR only sparse/image-only pages. Readable native text is retained.
            if native_chars < 180 and pytesseract and tess:
                try:
                    # Store compact PNG bytes, then OCR in a small worker pool. This
                    # avoids sequential multi-minute scans on long image-based specs.
                    pix=page.get_pixmap(matrix=fitz.Matrix(1.25,1.25), alpha=False)
                    rendered[pno-1]=pix.tobytes("png")
                except Exception as e:
                    ocr_errors.append({"page":pno,"error":f"render_failed: {e!r}"})

    def ocr_one(item):
        idx,png_bytes=item
        import pytesseract
        from PIL import Image
        im=Image.open(io.BytesIO(png_bytes)).convert("RGB")
        txt=pytesseract.image_to_string(im, config="--psm 6", timeout=12) or ""
        return idx,txt

    if rendered and pytesseract and tess:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(ocr_one,item):item[0] for item in rendered.items()}
            for fut in as_completed(futures):
                idx=futures[fut]
                try:
                    _,ocr_txt=fut.result()
                    ocr_chars=len(re.sub(r"\s+", "", ocr_txt))
                    q=page_quality[idx]; q["ocr_used"]=True; q["ocr_chars"]=ocr_chars; out["ocr_pages"]+=1
                    if ocr_chars > q["native_chars"]:
                        page_texts[idx]=ocr_txt; q["extracted_chars"]=ocr_chars
                except Exception as e:
                    ocr_errors.append({"page":idx+1,"error":repr(e)})

    out["full_text"]="\n\n".join(f"--- Page {i+1} ---\n{txt}" for i,txt in enumerate(page_texts))
    out["page_quality"]=page_quality; out["page_count"]=len(page_quality)
    out["ocr_attempted_pages"]=len(page_quality)
    out["ocr_successful_pages"]=sum(1 for q in page_quality if q.get("ocr_used"))
    out["native_text_chars"]=sum(q["native_chars"] for q in page_quality)
    out["extracted_text_chars"]=sum(q["extracted_chars"] for q in page_quality)
    out["ocr_errors"]=ocr_errors
    warnings=[]
    if out["page_count"] and out["extracted_text_chars"] < max(1500,out["page_count"]*80):
        warnings.append(f"low_text_yield:{out['extracted_text_chars']}_chars_for_{out['page_count']}_pages")
    if ocr_errors: warnings.append("one_or_more_ocr_pages_failed")
    if not tess and any(q["native_chars"]<250 for q in page_quality): warnings.append("tesseract_not_installed_or_not_on_path")
    out["ingestion_warnings"]=warnings; out["has_explicit_bold_golden"]=False
    return out


def parse_linked_document(path: Path, source_document: str, linked_from: str, context: str | None):
    """Parse common linked GeM document formats.

    PDF uses the full GeM table parser. Office/spreadsheet formats are preserved as
    text and simple key/value tables so they can supplement the main bid without
    replacing buyer_value.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        d = parse_technical_pdf_fast(str(path), source_document=source_document)
    elif ext in {".csv", ".txt"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if ext == ".csv":
            try:
                rows = list(csv.reader(io.StringIO(raw)))
            except Exception:
                rows = []
        else:
            rows = [re.split(r"\s{2,}|\t|\|", line) for line in raw.splitlines()]
        d = {
            "header": {},
            "spec_rows": _rows_to_spec_rows(rows, source_document),
            "consignees": [],
            "emd_required": None,
            "epbg_required": None,
            "full_text": raw,
            "has_golden_note": False,
            "has_nongolden_note": False,
            "ocr_pages": [],
            "spec_links": [],
            "pdf_links": [],
            "source_document": source_document,
            "has_explicit_bold_golden": False,
        }
    elif ext in {".xlsx", ".xls"}:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=False, data_only=True)
            rows = []
            bold_cells = set()
            for ws in wb.worksheets:
                for row in ws.iter_rows():
                    vals = [c.value for c in row]
                    rows.append(vals)
                    for c in row:
                        if c.value not in (None, "") and getattr(c.font, "bold", False):
                            bold_cells.add((ws.title, c.row, c.column))
            text = "\n".join(" | ".join(clean(str(v)) for v in row if v not in (None, "")) for row in rows)
            spec_rows = _rows_to_spec_rows(rows, source_document)
            # Mark a parameter row golden if the parameter or value cell was bold.
            for idx, sr in enumerate(spec_rows, start=1):
                # Conservative: preserve explicit document formatting only when a
                # corresponding first/second cell was bold in the workbook.
                for ws in wb.worksheets:
                    if (ws.title, idx, 1) in bold_cells or (ws.title, idx, 2) in bold_cells:
                        sr["golden"] = True
                        break
            d = {
                "header": {}, "spec_rows": spec_rows, "consignees": [],
                "emd_required": None, "epbg_required": None, "full_text": text,
                "has_golden_note": False, "has_nongolden_note": False,
                "ocr_pages": [], "spec_links": [], "pdf_links": [], "source_document": source_document,
                "has_explicit_bold_golden": any(x.get("golden") for x in spec_rows),
            }
        except Exception:
            # Keep the file even if its optional parser dependency is absent.
            raw = path.read_bytes().decode("utf-8", errors="replace")
            d = {
                "header": {}, "spec_rows": [], "consignees": [],
                "emd_required": None, "epbg_required": None, "full_text": raw,
                "has_golden_note": False, "has_nongolden_note": False,
                "ocr_pages": [], "spec_links": [], "pdf_links": [], "source_document": source_document,
                "has_explicit_bold_golden": False,
            }
    elif ext in {".docx", ".doc"}:
        try:
            from docx import Document
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            rows = []
            for table in doc.tables:
                for row in table.rows:
                    rows.append([cell.text for cell in row.cells])
            text = "\n".join(paragraphs + [" | ".join(r) for r in rows])
            d = {
                "header": {}, "spec_rows": _rows_to_spec_rows(rows, source_document),
                "consignees": [], "emd_required": None, "epbg_required": None,
                "full_text": text, "has_golden_note": False, "has_nongolden_note": False,
                "ocr_pages": [], "spec_links": [], "pdf_links": [], "source_document": source_document,
                "has_explicit_bold_golden": False,
            }
        except Exception:
            raw = path.read_bytes().decode("utf-8", errors="replace")
            d = {
                "header": {}, "spec_rows": [], "consignees": [],
                "emd_required": None, "epbg_required": None, "full_text": raw,
                "has_golden_note": False, "has_nongolden_note": False,
                "ocr_pages": [], "spec_links": [], "pdf_links": [], "source_document": source_document,
                "has_explicit_bold_golden": False,
            }
    else:
        raise ValueError(f"unsupported linked document format: {ext}")

    # Preserve raw linked-table evidence independently of the heuristic spec parser.
    # This prevents BOQ rows from being mistaken for semantic specifications downstream.
    raw_table_rows = []
    if ext in {".csv", ".txt"}:
        try:
            for i, row in enumerate(rows, 1):
                vals = [clean(str(x)) for x in row]
                if any(vals):
                    raw_table_rows.append({"row": i, "values": vals})
        except Exception:
            pass
    d["raw_table_evidence"] = raw_table_rows
    d["source_kind"] = "linked_document"
    d["linked_from"] = linked_from
    d["link_context"] = context
    d["local_path"] = str(Path("linked_documents") / path.name)
    return d


def _relevant_document_links(parsed: dict) -> list:
    """Return document-like links while excluding obvious portal/system assets."""
    links = []
    seen = set()
    for link in parsed.get("pdf_links", []):
        url = link.get("url")
        if not url or url in seen:
            continue
        blob = clean(" ".join([
            link.get("label") or "",
            link.get("context") or "",
            url
        ])).lower()

        # Only consider links that look like downloadable buyer documents.
        doc_ext = Path(urlparse(url).path).suffix.lower() in {
            ".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".txt"
        }
        keywords = (
            "technical specification", "specification", "datasheet",
            "boq", "bill of quantities", "buyer uploaded", "buyer specific",
            "atc", "additional terms", "additional terms and conditions",
            "view file", "document"
        )
        if link in parsed.get("spec_links", []) or doc_ext or any(k in blob for k in keywords):
            seen.add(url)
            links.append(link)
    return links


def load_linked_spec_documents(
    parsed: dict,
    base_url: str | None,
    bid_link_dir: Path,
) -> tuple[list, list]:
    """Download and parse linked technical/specification PDFs into this bid's folder.

    Layout:
        <output_dir>/<BID_ID>/linked_documents/
            01_<document>_<hash>.pdf
            02_<document>_<hash>.pdf
            manifest.json

    The linked document is supplementary; it cannot replace main-bid buyer_value.
    """
    bid_link_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    manifest = []

    for i, link in enumerate(_relevant_document_links(parsed), start=1):
        raw_url = link.get("url")
        if not raw_url:
            continue
        url = urljoin(base_url, raw_url) if base_url else raw_url
        filename = linked_filename(url, i)
        entry = {
            "index": i,
            "url": url,
            "context": link.get("context"),
            "local_filename": filename,
            "local_path": str(Path("linked_documents") / filename),
            "downloaded": False,
            "parsed": False,
            "error": None,
        }
        p, content_type = _download_linked_document(url, bid_link_dir, filename)
        if not p:
            entry["error"] = "download_failed_or_unsupported_format"
            entry["content_type"] = content_type
            manifest.append(entry)
            continue

        # _download_linked_document may correct the extension when the server
        # provides the format instead of the URL.
        entry["local_filename"] = p.name
        entry["local_path"] = str(Path("linked_documents") / p.name)
        entry["content_type"] = content_type
        entry["downloaded"] = True
        try:
            d = parse_linked_document(
                p, p.name, url, link.get("context")
            )
            docs.append(d)
            entry["parsed"] = True
        except Exception as e:
            entry["error"] = f"parse_failed: {e!r}"
        manifest.append(entry)

    (bid_link_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return docs, manifest


# -----------------------------------------------------------------------------
# Safe prompt templating: the JSON schemas inside the prompts contain literal
# braces, so str.format() is intentionally NOT used for LLM prompts.

def render_prompt(template: str, **values) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


class RateLimiter:
    """Paces outgoing requests to a minimum interval apart, globally across all
    concurrent callers. This is the primary fix for the 429 storm: concurrency=1
    alone only prevents *simultaneous* requests, it does nothing to stop firing
    the next request immediately after the previous one returns. A free-tier
    endpoint's per-minute quota is hit by request *rate*, not just overlap.
    """
    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, min_interval)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self):
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_call = time.monotonic()



# -----------------------------------------------------------------------------
# v11 SEMANTIC EXTRACTION
#
# Architecture:
#   1. Parse/download documents only to obtain complete document text and links.
#   2. Agent 1 extracts from the complete bundle (or chunks only when context
#      limits require it); there is no document-triage or model-repair stage.
#   3. Agent 2 performs one consolidated validation call for routed constraints.
#   4. Python performs deterministic parsing, deduplication, evidence attachment,
#      and validation bookkeeping without extra LLM prompts.
#
# There is deliberately NO deterministic table->constraint parser in this path.
# PDF/XLSX table extraction is used only as document-text extraction.
# -----------------------------------------------------------------------------

LLM_RETRACE_PROMPT = """You are the semantic requirement reconstruction model for a buyer procurement request.

The source happens to come from a GeM bid, but treat it as a description of what a
commercial/private buyer wants from a product and/or supplier. Do not turn generic
government procurement bureaucracy into marketplace requirements.

You are given the COMPLETE CURRENT DOCUMENT SET for ONE bid. Reconstruct the entire
buyer requirement representation from scratch using ALL documents currently available.

This is a retrace, not an incremental patch:
- Do not assume an earlier representation was correct.
- Re-read all current documents.
- Add requirements newly revealed by later documents.
- Correct earlier interpretations when later documents clarify them.
- Preserve relationships between values that belong to the same physical/configured item.

MOST IMPORTANT REPRESENTATION RULE:
Never flatten separate physical products, equipment instances, configurations,
locations, BOQ line items, service packages, or requirement groups into independent
allowed-value lists.

BAD:
  capacity = [8T, 8T, 3T, 5T]
  span     = [9.75m, 6.5m, 3m, 5.55m]
because this destroys which span belongs to which capacity.

GOOD:
  CRANE_01: capacity=8T, span=9.75m, lift=4.5m, bay_length=56.5m
  CRANE_02: capacity=8T, span=6.5m, lift=13m, bay_length=17.7m
  CRANE_03: capacity=3T, span=3m, lift=11m, bay_length=17.7m
  CRANE_04: capacity=5T, span=5.55m, lift=15m, bay_length=37.44m

Use allowed_values ONLY when the source explicitly gives alternatives for the SAME
requirement/group. If two values belong to two different equipment instances,
they must be represented in two requirement groups.

A bid can contain multiple products/components/services. DO NOT force the entire bid
into one product identity.

Requirement groups are mandatory whenever the source contains multiple related
instances. Examples:
- CRANE_01, CRANE_02, ...
- PUMP_01, PUMP_02, ...
- ITEM_01, ITEM_02, ...
- MANDATORY_SPARES
- INSTALLATION
- COMMISSIONING
- TRAINING
- WARRANTY
Use stable descriptive IDs within this response. Do not merge groups merely because
they share the same attribute names.

IMPORTANT BOQ COMPONENT / ACCESSORY RULE:
A component may have its own BOQ line (for example 2.1.2(b) bus bar, 2.1.2(c) rail,
2.1.2(d) VVVF drives, 2.1.2(e) commissioning spares) while belonging to the crane
package described by parent line 2.1.2(a). Do NOT discard such a requirement merely
because its source line_item_id differs from the parent equipment line. Assign the
constraint to the correct parent requirement_group_id (CRANE_02), use the component's
actual source line_item_id (e.g. 2.1.2(b)), and describe it as a component/inclusion
constraint. Keep service lines such as supervision/erection/commissioning as their own
service/transaction groups when they are genuinely separate deliverables. Never copy
an accessory to another crane group just because the same accessory appears nearby.

Return ONE valid JSON object only. No markdown. No comments. No <think>.

Schema:
{
  "product_name": "string or null",
  "product_category": "string or null",
  "indiamart_search_query": "string or null",

  "requirement_groups": [
    {
      "requirement_group_id": "string",
      "requirement_group_type": "product|component|spares|service|supplier|transaction|document|other",
      "requirement_description": "short description",
      "line_item_id": "source line/BOQ identifier or null",
      "constraints": [
        {
          "category": "PRODUCT|TECHNICAL|QUALITY|CAPACITY|DELIVERY|COMMERCIAL|CERTIFICATION|SUPPLIER|GEOGRAPHY|DOCUMENT|OTHER",
          "scope": "product|supplier|transaction",
          "constraint_type": "product_spec|certification|supplier_requirement|delivery|quantity|commercial|quality|capacity|geography|document_requirement|other",
          "attribute": "snake_case",
          "spec_name": "original attribute name or null",
          "operator": "=|>=|<=|>|<|in|required|prohibited|conditional|other",
          "buyer_value": "required value or null",
          "allowed_values": ["alternative values only for THIS SAME requirement group"] ,
          "allows_higher": true,
          "unit": "unit or null",
          "condition": {"attribute":"...","operator":"=","value":"..."} or null,
          "golden_param": false,
          "status": "specified|conditional|unspecified|not_applicable",
          "source_document": "exact document label",
          "source_document_id": "MAIN_BID|DOC_01|DOC_02|...",
          "raw_text": "short exact source excerpt, normally <= 240 characters",
          "confidence": 0.0
        }
      ]
    }
  ],

  "contacts": [
    {"contact_type":"buyer_contact|grievance_contact|other","name":null,"email":null,
     "phone":null,"address":null,"raw_text":"..."}
  ],

  "office_name": "string or null",
  "golden_candidates": ["attribute names"],
  "linked_document_mentions": [
    {"label":"...","document_type":"technical_specification|boq|atc|other","raw_text":"..."}
  ]
}

SEARCH QUERY RULES (INDIAMART):

Generate `indiamart_search_query` as a practical product-discovery query for finding
matching products and suppliers on IndiaMART. This field is not a summary of the bid
and must not be a dump of every extracted constraint.

- Start with the actual product name/category and its common commercial name.
- Prioritize explicitly marked golden parameters because they are strong matching
  signals, but do NOT rely on golden_param alone: include other explicit, highly
  discriminative specifications even when golden_param=false.
- Prefer a small set of identifying details, such as material/grade, model or part
  number, dimensions, capacity, size, colour, finish, or a technical standard, when
  those details materially distinguish the requested item. Select only details that
  apply to the same product/configuration.
- For multiple distinct products, BOQ items, or configurations, do not combine their
  incompatible specifications into one query. Use the primary purchasable product
  for this single query; do not invent a query for a configuration that is not clear.
- Exclude generic GeM/bid administration, buyer identity/contact details, bid number,
  submission instructions, payment clauses, delivery deadlines, and other transaction
  terms unless a term directly helps identify the product being searched for.
- Do not add inferred brands, model numbers, certifications, materials, dimensions,
  or performance values that are not supported by the documents. Do not include a
  certification merely because it appears somewhere in the bid if it is unrelated to
  the product's identity.
- Use concise keyword-style phrasing suitable for a marketplace search, not a full
  sentence. Avoid vague filler such as "best quality", "required", or "as per tender".
- If the documents do not identify a searchable product with reasonable confidence,
  return null rather than fabricate a query.
- Keep `product_name`, `product_category`, and `indiamart_search_query` consistent.



EXTRACTION RULES:

1. Extract all substantive evidence-backed buyer/product/supplier/transaction
   requirements. Do not return only representative examples.

2. Preserve multi-instance structure. If a table has rows/configurations describing
   different physical items, make separate requirement groups. Keep correlated
   attributes together inside the same group.

3. Never use an allowed_values list as a substitute for multiple physical instances.
   Alternatives for one requirement are allowed_values; separate instances are groups.

4. If a BOQ has line items, preserve the line item relationship. If the same product
   appears with different configurations, each configuration gets its own group.

5. Include mandatory spares/components/services as their own groups when they are
   independently procurable or operationally meaningful.

6. Set golden_param=true ONLY when the source explicitly marks THIS requirement as golden/mandatory or an unmistakable source-format marker establishes that it is golden/mandatory. Otherwise set false. Never infer golden status from importance, numeric magnitude, wording such as "required", or model judgment. If uncertain, false.

7. Distinguish:
   PRODUCT = physical product characteristics
   SUPPLIER = manufacturer/OEM/status/capability requirements
   TRANSACTION = delivery/payment/MOQ/commercial terms
   SERVICE = installation/commissioning/training/supervision/service scope

9. Preserve exact numbers, ranges, units, grades, standards, model/part numbers,
   quantities, durations, percentages and conditions. Normalize only attribute names
   and operators.

10. Main bid wording is authoritative when documents conflict. Linked technical
    documents provide additional technical detail and may clarify configuration,
    but do not silently claim that linked-only details were stated in the main bid.

11. For every constraint, cite the specific document containing the evidence.
    If several documents support it, use the most specific decisive document.

12. Do not convert generic GeM portal mechanics, buyer contacts, grievance contacts,
    bank details, bid submission mechanics, auction settings, internal buyer metadata,
    or generic government policy into IndiaMART-facing constraints unless they directly
    determine supplier/product eligibility.

13. Do not invent missing values. Use null/unknown.
    raw_text must be a short exact excerpt, normally <= 240 characters. Do not concatenate multiple source passages.
    Every numeric/product specification must be grounded in the cited source excerpt; do not reinterpret project, plant, building, capacity, budget, or schedule context as a product specification unless the text explicitly applies it to the product/group.

14. Do not merge duplicate constraints when their requirement groups differ.

15. Requirement group IDs must remain stable during this response and must describe
    the semantic entity, not merely the document number.

16. The final representation must be internally consistent: every product constraint
    belongs to a requirement group; correlated specifications for one physical item
    must share the same group ID.

**Certificate and Yes/No Field Handling**

When extracting certificate-related requirements, distinguish between a certificate being mandatory, not mandatory, unavailable, or merely mentioned in a form.

- Do not infer that a certificate is required simply because a field contains the words "certificate", "required", or "No".
- Interpret values such as "Yes", "No", blank fields, asterisks, and form defaults in the context of the surrounding source text.
- A source entry such as "Type Test Certificate - * No" must not automatically be converted into a mandatory certificate requirement. If the meaning is ambiguous, preserve the exact source text and mark the requirement as `UNCLEAR`.
- Mark a certificate as mandatory only when the source explicitly supports that interpretation.
- Do not invent or infer certificate obligations to fill gaps. When uncertain, retain the extracted field but flag its status as unresolved.

CURRENT DOCUMENT SET:
{source_bundle}
"""

# One-document mode: every LLM call receives exactly one document.
LLM_SINGLE_DOCUMENT_PROMPT = LLM_RETRACE_PROMPT
LLM_SINGLE_DOCUMENT_PROMPT = LLM_SINGLE_DOCUMENT_PROMPT.replace(
    "You are given the COMPLETE CURRENT DOCUMENT SET for ONE bid. Reconstruct the entire "
    "buyer requirement representation from scratch using ALL documents currently available.",
    "You are given EXACTLY ONE DOCUMENT. Extract every substantive buyer/product/supplier/"
    "transaction/service requirement contained in this document. Do not use information "
    "from any other document."
)
LLM_SINGLE_DOCUMENT_PROMPT = LLM_SINGLE_DOCUMENT_PROMPT.replace(
    """This is a retrace, not an incremental patch:
- Do not assume an earlier representation was correct.
- Re-read all current documents.
- Add requirements newly revealed by later documents.
- Correct earlier interpretations when later documents clarify them.
- Preserve relationships between values that belong to the same physical/configured item.""",
    """This is an isolated document extraction:
- Extract all requirements present in this document.
- Do not infer missing requirements from other documents.
- Preserve relationships between values that belong to the same physical/configured item.
- Keep the document's own source identity on every constraint."""
)



# -----------------------------------------------------------------------------
# v23 CHUNKED EXTRACTION
# If the complete document set exceeds --llm_context_chars, split it into
# overlapping, document-aware chunks. If it fits, preserve the original v23
# single full-bundle call. No source text is discarded.
# -----------------------------------------------------------------------------
LLM_CHUNK_PROMPT = """You are the semantic requirement extraction model for a buyer procurement request.

The source happens to come from a GeM bid, but treat it as a description of what a
commercial/private buyer wants from a product and/or supplier. Do not turn generic
government procurement bureaucracy into marketplace requirements.

You are processing ONE CHUNK of the bid document set. Extract every substantive
buyer/product/supplier/transaction/service requirement that is actually present in
this chunk. The chunk may begin or end in the middle of a document/table. Use the
provided document ID and source document exactly. Do not invent context from chunks
that you cannot see.

IMPORTANT CHUNK RULES:
- This is one extraction pass among several chunks of the SAME bid.
- Extract requirements fully supported by this chunk.
- Preserve line-item/group relationships visible in this chunk.
- If a specification is repeated because of chunk overlap, output it normally; the
  deterministic merger will deduplicate identical constraints later.
- Do not infer missing values from another chunk.
- source_document_id MUST be the DOCUMENT_ID shown above the source text.
- raw_text must be a short exact excerpt from this chunk, normally <= 240 chars.
- golden_param=true ONLY when this chunk explicitly marks THIS requirement as
  golden/mandatory or an unmistakable source-format marker establishes that it is
  golden/mandatory. Otherwise false.
- Distinguish PRODUCT (physical characteristics), SUPPLIER (manufacturer/OEM/status/
  capability), TRANSACTION (delivery/payment/MOQ/commercial terms), and SERVICE
  (installation/commissioning/training/supervision/service scope).
- Do not convert generic GeM portal mechanics, contacts, bid submission mechanics,
  auction settings, internal metadata, or generic government policy into
  IndiaMART-facing requirements unless they directly determine product/supplier
  eligibility.
- Preserve exact numbers, ranges, units, grades, standards, model/part numbers,
  quantities, durations, percentages and conditions.
- TECHNICAL TABLE COMPLETENESS IS REQUIRED: inspect every row of every technical/specification table in the supplied text. Emit a separate constraint for each explicit product attribute/value, including size, cores, material, shape, insulation, sheath, standards, certifications, marking, test reports, and packaging/length where present. Do not stop after extracting the first row.
- For tables with both “Values” and “Bid Requirement (Allowed Values)”, preserve the distinction: the buyer requirement column controls the requirement; the Values column is the catalog/category value. Capture allowed alternatives only as allowed_values for that same attribute.
- Give each product a descriptive group ID based on the actual product (e.g. CABLE_01 for cable, not CRANE_01).
- Every constraint must carry a short exact raw_text excerpt from its source. Never return a constraint with no evidence excerpt when source text is available.
- Do not reinterpret project/plant/building/capacity/budget/schedule context as a
  product specification unless the text explicitly applies it to the product/group.
- Do not merge distinct physical instances into one group merely because they share
  an attribute.
- Generate `indiamart_search_query` as a concise marketplace product-discovery query:
  product name/category first, then explicitly marked golden parameters, then other
  explicit and highly discriminative specifications even if they are not golden.
  Use only specs belonging to the same product/configuration. Exclude generic GeM/bid
  administration and transaction terms unless they directly identify the product.
  Never invent brands, models, standards, dimensions, or values. For a chunk that does
  not contain enough product identity to form a reliable query, return null for this
  field; do not construct a query from unrelated items.

Return JSON only in this exact top-level shape:
{
  "schema_version": 30,
  "product_name": null,
  "product_category": null,
  "indiamart_search_query": null,
  "requirement_groups": [],
  "contacts": [],
  "office_name": null,
  "golden_candidates": [],
  "linked_document_mentions": []
}

Each requirement group:
{
  "requirement_group_id": "...",
  "requirement_group_type": "product|component|spares|service|supplier|transaction|document|other",
  "requirement_description": "...",
  "line_item_id": null,
  "source_document_id": "MAIN_BID|DOC_01|DOC_02",
  "constraints": []
}

Each constraint:
{
  "category": "PRODUCT|TECHNICAL|QUALITY|CAPACITY|DELIVERY|COMMERCIAL|CERTIFICATION|SUPPLIER|GEOGRAPHY|DOCUMENT|OTHER",
  "scope": "product|supplier|transaction",
  "constraint_type": "...",
  "attribute": "...",
  "spec_name": null,
  "operator": "=|>=|<=|>|<|in|required|prohibited|conditional|other",
  "buyer_value": null,
  "allowed_values": null,
  "allows_higher": false,
  "unit": null,
  "condition": null,
  "status": "specified|conditional|unspecified|not_applicable",
  "golden_param": false,
  "raw_text": "...",
  "confidence": 0.0
}

CHUNK METADATA:
{chunk_metadata}

CHUNK TEXT:
{source_bundle}
"""


LLM_MISSING_SPEC_RECOVERY_PROMPT = """You are recovering technical specifications omitted from an earlier extraction.
Return only constraints supported by the supplied source rows. Do not repeat constraints already extracted.
For each row, preserve the distinction between the catalog “Values” and buyer “Bid Requirement (Allowed Values)” columns. The buyer requirement column is the requirement; preserve alternatives only for that same attribute.
Use the product's actual descriptive group ID (for example CABLE_01 for a cable), never an unrelated product label. Include exact evidence in raw_text, source_document_id, attribute, buyer_value, unit, allowed_values, status, and golden_param only when explicitly supported.
Return JSON only in this shape: {"schema_version":30,"product_name":null,"product_category":null,"indiamart_search_query":null,"requirement_groups":[{"requirement_group_id":"CABLE_01","requirement_group_type":"product","requirement_description":"Submersible cable","line_item_id":null,"source_document_id":"MAIN_BID","constraints":[]}],"contacts":[],"office_name":null,"golden_candidates":[],"linked_document_mentions":[]}

ALREADY EXTRACTED CONSTRAINTS (do not repeat):
{existing_constraints}

MISSING SOURCE SPECIFICATION ROWS:
{missing_rows}
"""


# -----------------------------------------------------------------------------
# v17 ADVERSARIAL VALIDATION
# Agent 1 extracts. Agent 2 receives the complete document set + JSON1 and audits
# every constraint. Agent 2 is a critic, not a second free-form extractor.
# -----------------------------------------------------------------------------
LLM_VALIDATOR_PROMPT = """You are an adversarial validator for a buyer procurement constraint dataset.

You are given a SMALL BATCH of constraints from JSON1 plus TARGETED EVIDENCE SNIPPETS retrieved from the bid documents.
Do not re-extract the bid. Audit only the supplied constraints.

For each input constraint return exactly one:
- KEEP: supported as written.
- CORRECT: supported in substance but one or more semantic fields are wrong. Provide the corrected constraint.
- REJECT: unsupported, invented, or incidental context.
- KEEP: evidence is explicit or no affirmative contradiction was found. Do not use a review state.

Critical checks:
- project/plant context mistaken for product specification
- wrong attribute/value/unit/range/standard/model/quantity
- product vs supplier vs transaction vs service scope
- quantity mistaken for capacity
- delivery confused with warranty/lifetime
- supplier/OEM requirement confused with product specification
- certification/document requirement confused with physical product
- multiple physical instances flattened into allowed_values
- correlated attributes assigned to the wrong group
- group identity, line item, equipment type, and product-vs-service distinction
- golden_param incorrectly set
- generic GeM portal mechanics or government administration incorrectly treated as IndiaMART-facing requirements

GOLDEN PARAMETER RULE:
- golden_param is the ONLY decisive priority flag.
- true ONLY when the supplied evidence explicitly marks the requirement as golden/mandatory, or an unmistakable source-format marker establishes that it is a mandatory/golden parameter.
- false otherwise. Do not use confidence, hard/soft, negotiability, or another parallel priority flag.
- Do not infer golden=true merely because a product specification is important or sounds mandatory.

Evidence rules:
- KEEP and CORRECT require at least one exact short evidence excerpt with source_document_id.
- REJECT requires affirmative contradictory/unsupported evidence. A missing targeted snippet is NOT evidence that the constraint is false.
- If evidence retrieval is incomplete or ambiguous, KEEP the original constraint unless the available evidence affirmatively contradicts it. Do not reject merely because evidence is missing.
- Never manufacture quotations.
- Use only the supplied evidence snippets.
- Do not add new constraints.
- GROUP BINDING IS MANDATORY: validate each constraint against its own requirement_group_id,
  requirement_group_type, requirement_description, and line_item_id. These fields are part of
  the input constraint and must not be ignored in favor of generic bid-title/raw_text text.
- For a BOQ with multiple physical items, do not transfer a specification between different
  physical equipment groups. However, accessory/component rows may have their own line_item_id
  and still belong to the parent equipment package (e.g. crane 2.1.2(a) plus bus bar 2.1.2(b),
  rail 2.1.2(c), drives or spares). A different component line_item_id is NOT by itself a
  contradiction. Check the component description and parent package context. Preserve the actual
  component line_item_id and parent requirement_group_id. REJECT only if evidence affirmatively
  shows the component belongs to a different equipment group/package; otherwise KEEP/CORRECT or
  KEEP unless there is affirmative contradictory evidence.
- If the value is supported but the scope/category/type is wrong (for example supervision,
  installation, commissioning, or service visits classified as product), return CORRECT and
  fix the semantic fields. Do not REJECT a supported value solely because it was misclassified.
- Do not use a generic title or a different group as affirmative evidence against a specific
  line-item constraint.

SCOPE ONTOLOGY (SEMANTIC, NOT DOCUMENT-POSITION BASED):
- Decide scope from what the requirement means, not where it appears in a BOQ/table/section.
- local content, Make in India, Class 1/2 Local Supplier qualification, OEM/manufacturer/authorized manufacturer status, supplier experience, turnover, credentials and supplier eligibility are SUPPLIER scope even when printed under a product line or maintenance section.
- Product material, dimensions, ratings, capacity, performance, model, configuration and physical characteristics are PRODUCT scope.
- Delivery, payment, warranty, site visit, installation/commissioning and other execution/commercial obligations are TRANSACTION scope unless the text clearly makes them a supplier qualification.
- Project/plant/station capacity must not be converted into equipment capacity unless the source explicitly states that the equipment itself has that capacity.
- Do not change scope merely because the surrounding document section or line item is a product.

Return JSON only, with one decision for every supplied constraint ID.

Schema:
{
  "schema_version": 1,
  "validation_agent": "targeted_adversarial_validator",
  "validation_status": "ok|failed",
  "summary": {"input_constraints": 0, "keep": 0, "correct": 0, "reject": 0},
  "validated_constraints": [
    {
      "constraint_id": "exact input ID",
      "decision": "KEEP|CORRECT|REJECT",
      "original_constraint": {},
      "validated_constraint": {},
      "evidence": [{"source_document_id":"MAIN_BID|DOC_01|...","source_document":"...","raw_text":"short exact excerpt","support_type":"explicit|unambiguous"}],
      "reason": "short evidence-based explanation",
      "validation_confidence": 0.0
    }
  ]
}

TARGETED EVIDENCE:
{evidence_bundle}

JSON1 BATCH:
{extracted_json}
"""


def render_prompt(template: str, **values) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, min_interval)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self):
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            remaining = self.min_interval - (now - self._last_call)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_call = time.monotonic()



class APIDailyBudget:
    """Persistent per-output-dir API attempt budget.

    The free endpoint can enforce a small daily request quota. We count every
    actual HTTP attempt (including retries) and persist the count by UTC day so
    a crash/restart cannot accidentally burn the remaining quota.
    """
    def __init__(self, path: Path, limit: int):
        self.path = Path(path)
        self.limit = max(1, int(limit))
        self._lock = asyncio.Lock()
        self.date = time.strftime("%Y-%m-%d", time.gmtime())
        self.count = 0
        self._load()

    def _load(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("date") == self.date:
                self.count = int(d.get("count", 0))
        except Exception:
            self.count = 0

    async def acquire(self):
        async with self._lock:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if today != self.date:
                self.date, self.count = today, 0
            if self.count >= self.limit:
                raise RuntimeError(
                    f"API daily request budget exhausted: {self.count}/{self.limit}. "
                    f"State: {self.path}"
                )
            self.count += 1
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps({"date": self.date, "count": self.count, "limit": self.limit}, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path)
            return self.count


async def call_llm(client, model, sem, prompt, max_tokens, think_mode,
                   rate_limiter=None, api_budget=None, retries=2):
    """
    Make an API call with explicit retry timing.

    Retry delays are ALWAYS multiples of 8 seconds:
        8, 16, 24, 32, ... seconds
    A server Retry-After value is respected only after rounding UP to the
    next 8-second multiple.

    IMPORTANT: authentication is configured when the client is created in
    amain(); this function never silently falls back to another provider.
    """
    extra = {}
    if think_mode == "openrouter":
        extra["extra_body"] = {"reasoning": {"enabled": False}}
    elif think_mode == "chat_template":
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    last = None

    for attempt in range(1, retries + 1):
        try:
            async with sem:
                if api_budget:
                    await api_budget.acquire()
                if rate_limiter:
                    await rate_limiter.wait()

                r = await client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    **extra,
                )

            # OpenRouter can return a successful HTTP response with empty content
            # (for example, a reasoning-only response or an interrupted generation).
            # Log the response metadata instead of silently treating it as valid text.
            if not getattr(r, "choices", None):
                raise RuntimeError(
                    "LLM response contained no choices; "
                    f"model={getattr(r, 'model', None)!r}, "
                    f"usage={getattr(r, 'usage', None)!r}, "
                    f"response={str(r)[:1800]}"
                )
            choice = r.choices[0]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if isinstance(content, list):
                # Some compatible endpoints return content blocks instead of a string.
                content = "\n".join(
                    str(block.get("text", "")) if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            if not isinstance(content, str) or not content.strip():
                reasoning = getattr(message, "reasoning", None) if message is not None else None
                refusal = getattr(message, "refusal", None) if message is not None else None
                detail = (
                    f"OpenRouter returned empty message.content; requested_model={model!r}; "
                    f"served_model={getattr(r, 'model', None)!r}; "
                    f"finish_reason={getattr(choice, 'finish_reason', None)!r}; "
                    f"reasoning_chars={len(reasoning) if isinstance(reasoning, str) else None}; "
                    f"refusal={str(refusal)[:300]!r}; usage={getattr(r, 'usage', None)!r}; "
                    f"message={str(message)[:1200]}"
                )
                print("  [EMPTY RESPONSE] " + detail, file=sys.stderr)
                raise RuntimeError(detail)
            return content

        except Exception as e:
            last = e
            code = getattr(e, "status_code", None)

            # Surface the provider's actual response body. HTTP 400 is generally
            # a deterministic request/payload error, so retrying the same payload
            # wastes quota and can trigger recursive batch retries upstream.
            response_body = getattr(e, "body", None)
            if response_body is None:
                response = getattr(e, "response", None)
                try:
                    response_body = response.text if response is not None else None
                except Exception:
                    response_body = None
            detail = str(response_body or e).replace("\n", " ")[:1800]
            print(f"  [API failure] HTTP={code} model={model!r} attempt={attempt}/{retries} detail={detail}", file=sys.stderr)

            # Authentication failures and malformed/unsupported requests should
            # fail fast. Only transient throttling/server/network errors retry.
            if code in (400, 401, 403, 404, 422):
                raise RuntimeError(
                    f"Non-retryable API request failure (HTTP {code}); model={model!r}; detail={detail}"
                ) from e

            if attempt >= retries:
                raise

            retry_after = None
            hdrs = getattr(getattr(e, "response", None), "headers", None)
            if hdrs:
                try:
                    retry_after = float(hdrs.get("retry-after"))
                except (TypeError, ValueError):
                    pass

            # Every retry delay is an 8-second multiple.
            exponential = 8 * attempt
            wait = exponential

            if retry_after is not None:
                wait = max(wait, int(retry_after + 7.999) // 8 * 8)

            # Keep the backoff bounded, but still an 8-second multiple.
            wait = min(180, wait)
            wait = max(8, (wait // 8) * 8)

            print(
                f"  [API error {code}; retry {attempt + 1}/{retries} in {wait}s]",
                file=sys.stderr,
            )
            await asyncio.sleep(wait)

    raise last



LLM_JSON_REPAIR_PROMPT = """Repair the malformed JSON below into one valid JSON object.
Preserve the original data and values exactly; do not add, remove, infer, summarize, or semantically correct fields. Fix serialization syntax only. Return only the repaired JSON object, with no markdown fences or commentary.

MALFORMED_JSON:
{malformed_json}
"""


class JSONRepairFailed(RuntimeError):
    """A model response remained invalid after its single JSON-repair attempt."""


def parse_llm_json(text: str):
    """
    Parse Qwen JSON defensively:
      1. strict json.loads()
      2. deterministic local json-repair

    json-repair only repairs serialization syntax. It has no source documents
    and therefore cannot invent semantic requirements.
    """
    if not text:
        raise ValueError("empty LLM response")

    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    s = text.find("{")
    e = text.rfind("}")

    if s < 0:
        raise ValueError("no JSON object in LLM output")
    if e < s:
        raise ValueError(
            "LLM output appears truncated before JSON object closed"
        )

    candidate = text[s:e + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_error:
        pass

    try:
        from json_repair import repair_json

        repaired = repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict):
            return repaired

        raise ValueError(
            f"json-repair returned {type(repaired).__name__}, expected dict"
        )
    except Exception as repair_error:
        raise ValueError(
            f"invalid JSON after local repair: {first_error}; "
            f"json-repair: {repair_error}"
        ) from first_error


async def parse_or_repair_llm_json(
    client, model, sem, raw, max_tokens, think_mode,
    rate_limiter=None, api_budget=None,
):
    """Parse strictly first; call the JSON-repair prompt only for invalid JSON.

    If the one repair attempt is still invalid, raise JSONRepairFailed so the
    current bid is marked failed and processing continues with the next bid.
    """
    try:
        return parse_llm_json(raw)
    except Exception as parse_error:
        if client is None:
            raise JSONRepairFailed(f"Invalid JSON and no LLM client available for repair: {parse_error}") from parse_error
        repair_prompt = render_prompt(LLM_JSON_REPAIR_PROMPT, malformed_json=str(raw)[:100000])
        try:
            repaired_raw = await call_llm(
                client, model, sem, repair_prompt, max_tokens, think_mode,
                rate_limiter=rate_limiter, api_budget=api_budget, retries=1,
            )
            return parse_llm_json(repaired_raw)
        except Exception as repair_error:
            raise JSONRepairFailed(
                f"JSON invalid; single repair attempt failed. Original={parse_error!r}; repair={repair_error!r}"
            ) from repair_error


TECHNICAL_DOC_RE = re.compile(
    r"(?:technical|tech)[ _-]*(?:spec|specification|requirement|data[ _-]*sheet)|"
    r"bid[ _-]*specification|product[ _-]*(?:spec|datasheet)|buyer[ _-]*spec|"
    r"technical[ _-]*bid|specification[ _-]*doc|boq|bill[ _-]*of[ _-]*quantit|drawing",
    re.I,
)
ADMIN_DOC_RE = re.compile(
    r"(?:^|[ _.-])(?:sla(?:fds)?|terms?[ _-]*(?:and|&|n)?[ _-]*conditions?|general[ _-]*conditions?|"
    r"agreement|contract|undertaking|declaration|integrity[ _-]*pact|payment[ _-]*terms?|"
    r"penalt(?:y|ies)|service[ _-]*level)(?:[ _.-]|$)", re.I
)


def build_document_sections(parsed: dict, linked_docs: list) -> list:
    """One complete, labelled text section per document. No semantic table parser."""
    docs = [{
        "document_id": "MAIN_BID",
        "source_document": parsed.get("source_document") or "main_bid.pdf",
        "source_kind": "main_bid",
        "text": parsed.get("full_text", "") or "",
    }]
    for i, d in enumerate(linked_docs, 1):
        docs.append({
            "document_id": f"DOC_{i:02d}",
            "source_document": d.get("source_document") or f"linked_document_{i}",
            "source_kind": "linked_document",
            "text": d.get("full_text", "") or "",
        })

    sections = []
    for d in docs:
        text = d["text"].strip()
        if not text:
            continue
        sections.append(
            f"===== DOCUMENT_ID: {d['document_id']} =====\n"
            f"SOURCE_KIND: {d['source_kind']}\n"
            f"SOURCE_DOCUMENT: {d['source_document']}\n"
            f"===== DOCUMENT_TEXT_START =====\n"
            f"{text}\n"
            f"===== DOCUMENT_TEXT_END ====="
        )
    return sections


def build_retrace_bundle(sections: list, max_chars: int) -> str:
    """Original v23 behavior: use one complete bundle when it fits."""
    bundle = "\n\n".join(sections)
    if len(bundle) > max_chars:
        raise ValueError(
            f"current bid document set is {len(bundle):,} chars, above "
            f"--llm_context_chars={max_chars:,}"
        )
    return bundle


def _section_parts(documents: list, max_chars: int, overlap_chars: int) -> list:
    """Split documents without dropping text. Splits happen only inside a document."""
    parts = []
    overhead = 1200
    payload_limit = max(1000, max_chars - overhead)
    overlap_chars = max(0, min(overlap_chars, payload_limit // 4))

    for d in documents:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        prefix = (
            f"===== DOCUMENT_ID: {d['document_id']} =====\n"
            f"SOURCE_KIND: {d.get('source_kind','unknown')}\n"
            f"SOURCE_DOCUMENT: {d.get('source_document','')}\n"
            f"===== DOCUMENT_TEXT_START =====\n"
        )
        suffix = "\n===== DOCUMENT_TEXT_END ====="
        limit = max(1000, payload_limit - len(prefix) - len(suffix))

        if len(text) <= limit:
            parts.append({
                "document_id": d["document_id"],
                "source_document": d.get("source_document"),
                "text": prefix + text + suffix,
                "start": 0,
                "end": len(text),
                "total": len(text),
            })
            continue

        step = max(1, limit - overlap_chars)
        start = 0
        part_no = 1
        while start < len(text):
            end = min(len(text), start + limit)
            chunk_text = text[start:end]
            parts.append({
                "document_id": d["document_id"],
                "source_document": d.get("source_document"),
                "text": prefix +
                        f"[DOCUMENT_CHUNK_START={start} END={end} TOTAL={len(text)}]\n" +
                        chunk_text + suffix,
                "start": start,
                "end": end,
                "total": len(text),
                "part_no": part_no,
            })
            if end >= len(text):
                break
            start += step
            part_no += 1
    return parts


def build_extraction_chunks(documents: list, max_chars: int, overlap_chars: int = 5000) -> list:
    """Pack complete small documents together; split only oversized documents.

    Every character of every document is represented in at least one chunk.
    Oversized document pieces overlap so boundary-spanning tables/specifications
    are visible to both adjacent calls.
    """
    parts = _section_parts(documents, max_chars, overlap_chars)
    chunks = []
    current = []
    current_len = 0

    for part in parts:
        plen = len(part["text"])
        # Oversized-document pieces are kept as their own calls. This makes the
        # test behavior obvious and avoids mixing unrelated document fragments.
        is_split_piece = part["total"] > (max_chars - 1200)
        if is_split_piece:
            if current:
                chunks.append(current)
                current, current_len = [], 0
            chunks.append([part])
            continue

        if current and current_len + plen + 2 > max_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(part)
        current_len += plen + (2 if current_len else 0)

    if current:
        chunks.append(current)

    out = []
    for i, group in enumerate(chunks, 1):
        text = "\n\n".join(x["text"] for x in group)
        out.append({
            "chunk_id": f"CHUNK_{i:03d}",
            "text": text,
            "chars": len(text),
            "documents": [x["document_id"] for x in group],
            "parts": [
                {
                    "document_id": x["document_id"],
                    "start": x["start"],
                    "end": x["end"],
                    "total": x["total"],
                    "part_no": x.get("part_no", 1),
                }
                for x in group
            ],
        })
    return out


def normalize_raw_text(value):
    value = clean(str(value or ""))
    if len(value) <= 300:
        return value
    return value[:297].rstrip() + "..."


def normalize_group_constraint(c: dict, group: dict) -> dict:
    allowed = c.get("allowed_values")
    if isinstance(allowed, str):
        allowed = [x.strip() for x in allowed.split(",") if x.strip()]
    if not isinstance(allowed, list):
        allowed = None

    scope = c.get("scope", "product")
    if scope not in ("product", "supplier", "transaction"):
        scope = "product"

    category = str(c.get("category") or c.get("constraint_category") or "OTHER").upper()
    valid_categories = {"PRODUCT","TECHNICAL","QUALITY","CAPACITY","DELIVERY","COMMERCIAL","CERTIFICATION","SUPPLIER","GEOGRAPHY","DOCUMENT","OTHER"}
    if category not in valid_categories:
        category = "OTHER"

    golden = c.get("golden_param", False)
    if isinstance(golden, str):
        golden = golden.lower().strip() in ("true","yes","1")
    else:
        golden = bool(golden)

    return {
        "constraint_id": c.get("constraint_id"),
        "bid_id": c.get("bid_id"),
        "requirement_group_id": c.get("requirement_group_id") or group.get("requirement_group_id"),
        "requirement_group_type": c.get("requirement_group_type") or group.get("requirement_group_type"),
        "requirement_description": c.get("requirement_description") or group.get("requirement_description"),
        "line_item_id": c.get("line_item_id", group.get("line_item_id")),
        "constraint_category": category,
        "scope": scope,
        "constraint_type": c.get("constraint_type") or "other",
        "attribute": snake(c.get("attribute") or c.get("spec_name") or "requirement"),
        "spec_name": c.get("spec_name"),
        "operator": c.get("operator") or "=",
        "buyer_value": c.get("buyer_value"),
        "allowed_values": allowed,
        "allows_higher": bool(c.get("allows_higher", False)),
        "unit": c.get("unit"),
        "condition": c.get("condition"),
        "status": c.get("status") or "specified",
        "golden_param": golden,
        "source_document": c.get("source_document"),
        "source_document_id": c.get("source_document_id"),
        "raw_text": normalize_raw_text(c.get("raw_text")),
        "confidence": c.get("confidence", 0.0),
        "indiamart_matchable": c.get("indiamart_matchable", True),
        "source": c.get("source", "qwen"),
    }


def flatten_qwen_groups(obj: dict) -> list:
    """Flatten grouped Qwen output WITHOUT losing requirement_group identity."""
    out = []
    for group in obj.get("requirement_groups", []) or []:
        if not isinstance(group, dict):
            continue
        constraints = group.get("constraints") or []
        for c in constraints:
            if isinstance(c, dict):
                x = normalize_group_constraint(c, group)
                if x["attribute"]:
                    out.append(x)
    return out


def validate_representation(constraints: list) -> list:
    """
    Mechanical sanity checks. These DO NOT attempt semantic repair.
    They only flag structures that deserve another look.
    """
    flags = []
    groups = {}
    for c in constraints:
        gid = c.get("requirement_group_id")
        groups.setdefault(gid, []).append(c)

    # Same group + same attribute with many values can be legitimate alternatives,
    # but is worth recording so the JSON audit makes the issue visible.
    for gid, items in groups.items():
        attrs = {}
        for c in items:
            attrs.setdefault(c.get("attribute"), []).append(c)
        for attr, vals in attrs.items():
            concrete = [x for x in vals if x.get("buyer_value") not in (None, "")]
            if len(concrete) > 1:
                flags.append(
                    f"representation_check: group={gid} attribute={attr} has "
                    f"{len(concrete)} concrete values; inspect whether these are "
                    f"alternatives or separate instances"
                )

    # Detect a classic flattening signature: the same attribute has multiple
    # concrete values but there is only one requirement group for the product.
    product_groups = {
        c.get("requirement_group_id")
        for c in constraints
        if c.get("scope") == "product"
    }
    for attr in ("capacity", "crane_capacity", "span", "crane_span",
                 "lift", "crane_lift", "bay_length", "crane_bay_length"):
        vals = {
            json.dumps(c.get("buyer_value"), sort_keys=True, ensure_ascii=False)
            for c in constraints
            if c.get("attribute") == attr and c.get("buyer_value") not in (None, "")
        }
        if len(vals) > 1 and len(product_groups) <= 1:
            flags.append(
                f"representation_check: {attr} has {len(vals)} values but only "
                f"{len(product_groups)} product requirement group(s)"
            )
    return flags


def canonicalize_final_constraints(constraints: list) -> list:
    """Mechanical final cleanup only; Qwen's grouped representation remains authoritative."""
    out, seen = [], set()
    for c in constraints:
        if c.get("status") in ("unspecified", "not_applicable"):
            continue
        if not c.get("attribute") or not c.get("raw_text"):
            continue

        key = (
            c.get("requirement_group_id"),
            c.get("line_item_id"),
            c.get("constraint_category"),
            c.get("scope"),
            c.get("attribute"),
            c.get("operator"),
            json.dumps(c.get("buyer_value"), sort_keys=True, ensure_ascii=False),
            json.dumps(c.get("allowed_values"), sort_keys=True, ensure_ascii=False),
            json.dumps(c.get("condition"), sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)

        x = dict(c)
        cat = x.get("constraint_category")
        if cat in ("PRODUCT", "TECHNICAL", "QUALITY", "CAPACITY"):
            x["matchability"] = "product"
        elif cat in ("SUPPLIER", "CERTIFICATION", "DOCUMENT"):
            x["matchability"] = "supplier"
        elif cat in ("DELIVERY", "COMMERCIAL", "GEOGRAPHY"):
            x["matchability"] = "transaction"
        else:
            x["matchability"] = "unknown"
        x["indiamart_matchable"] = x["matchability"] != "unknown"
        out.append(x)
    return out


def process_bid_documents(parsed, linked_docs, client, sem, args, rate_limiter):
    """Unused sync wrapper retained only for readability; async implementation below."""
    raise NotImplementedError



def normalize_validation_decision(item: dict) -> str:
    d=str(item.get("decision") or "KEEP").upper().strip()
    if d == "NEEDS_REVIEW" or d not in {"KEEP", "CORRECT", "REJECT"}:
        return "KEEP"
    return d


def validate_validator_object(obj: dict, inputs: list) -> list:
    """Strict structural validation for ONE validator batch."""
    errors = []
    items = obj.get("validated_constraints")
    if not isinstance(items, list):
        return ["validator output has no validated_constraints list"]

    expected = [str(c.get("constraint_id")) for c in inputs]
    got = [str(x.get("constraint_id")) for x in items if isinstance(x, dict)]
    if len(items) != len(inputs):
        errors.append(f"validator returned {len(items)} decisions for {len(inputs)} input constraints")

    from collections import Counter
    dupes = sorted(cid for cid, n in Counter(got).items() if n > 1)
    if dupes:
        errors.append("validator duplicated constraint IDs: " + ", ".join(dupes[:20]))

    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if missing:
        errors.append("validator omitted constraint IDs: " + ", ".join(missing[:20]))
    if extra:
        errors.append("validator returned unknown constraint IDs: " + ", ".join(extra[:20]))

    by_id = {str(x.get("constraint_id")): x for x in items if isinstance(x, dict)}
    for cid in expected:
        x = by_id.get(cid)
        if not x:
            continue
        d = normalize_validation_decision(x)
        evidence = x.get("evidence")
        if d in {"KEEP", "CORRECT"} and (not isinstance(evidence, list) or not evidence):
            errors.append(f"{cid}: {d} requires documentary evidence")
        if d == "CORRECT" and not isinstance(x.get("validated_constraint"), dict):
            errors.append(f"{cid}: CORRECT requires validated_constraint")
        if d not in {"KEEP", "CORRECT", "REJECT"}:
            errors.append(f"{cid}: invalid decision")
    return errors


def _validator_failure_audit(inputs: list, reason: str) -> list:
    return [{
        "constraint_id": str(c.get("constraint_id")),
        "decision": "KEEP",
        "original_constraint": c,
        "validated_constraint": None,
        "evidence": [],
        "reason": reason,
        "validation_confidence": 0.0,
    } for c in inputs]


def _evidence_terms(c: dict) -> list[str]:
    terms=[]
    for k in ("raw_text","spec_name","attribute","buyer_value","requirement_group_id","requirement_description","line_item_id"):
        v=c.get(k)
        if isinstance(v, (list,dict)):
            continue
        v=clean(str(v or ""))
        if not v or len(v)<3:
            continue
        terms.append(v)
        if k == "attribute":
            terms.extend([x for x in re.split(r"[_\s]+", v) if len(x)>=3])
    vals=c.get("allowed_values") or []
    if isinstance(vals,list):
        terms.extend(clean(str(x)) for x in vals[:4] if len(clean(str(x)))>=3)
    # longest/highest-signal first
    uniq=[]
    for t in sorted(set(terms), key=len, reverse=True):
        if t and t not in uniq:
            uniq.append(t)
    return uniq[:8]


def attach_exact_evidence_anchors(constraints: list, documents: list, context_window: int = 1200):
    """Deterministically anchor each Qwen constraint's raw_text into its source document.

    Agent 1 already supplies source_document_id + raw_text. We locate that exact excerpt
    in the parsed source text so Agent 2 sees the authoritative passage first. Retrieval
    is only a fallback/supplement when the excerpt cannot be located verbatim.
    """
    by_id={d.get("document_id"): d for d in documents}
    for c in constraints:
        doc=by_id.get(c.get("source_document_id"))
        raw=clean(str(c.get("raw_text") or ""))
        anchor=None
        if doc and raw:
            text=doc.get("text") or ""
            pos=text.lower().find(raw.lower())
            if pos>=0:
                a=max(0,pos-context_window); b=min(len(text),pos+len(raw)+context_window)
                anchor={"document_id":doc.get("document_id"),"source_document":doc.get("source_document"),
                        "start":pos,"end":pos+len(raw),"context_start":a,"context_end":b,
                        "raw_text":text[pos:pos+len(raw)],"anchor_method":"exact"}
            else:
                # Keep an explicit normalized anchor when PDF line wraps differ from
                # the model's excerpt; never fabricate raw offsets into the source.
                norm_text=re.sub(r"\s+"," ",text).strip()
                norm_raw=re.sub(r"\s+"," ",raw).strip()
                npos=norm_text.lower().find(norm_raw.lower())
                if npos>=0:
                    anchor={"document_id":doc.get("document_id"),"source_document":doc.get("source_document"),
                            "start":None,"end":None,"context_start":None,"context_end":None,
                            "normalized_start":npos,"normalized_end":npos+len(norm_raw),
                            "raw_text":norm_raw,"anchor_method":"whitespace_normalized"}
        c["evidence_anchor"]=anchor
    return constraints

def build_targeted_evidence(constraints: list, documents: list, window: int = 900, max_chars: int = 70000) -> str:
    """Retrieve only local document windows relevant to the validator batch."""
    chunks=[]
    seen=set()
    for c in constraints:
        cid=str(c.get("constraint_id"))
        terms=_evidence_terms(c)
        found=0
        # Group-aware retrieval: a specific BOQ line item is stronger evidence
        # than a generic GeM title or another crane's matching attribute value.
        line_item = clean(str(c.get("line_item_id") or ""))
        group_specific_found = False
        if line_item:
            # Include the exact row and sibling component rows from the same BOQ family
            # (e.g. 2.1.2(a) through 2.1.2(e)). These rows can describe one package.
            family = re.match(r"^(\d+(?:\.\d+)*\s*\([a-zA-Z0-9]+\))", line_item)
            family_prefix = re.sub(r"\s*\([a-zA-Z0-9]+\)$", "", family.group(1)) if family else None
            for doc in documents:
                text = doc.get("text") or ""
                low = text.casefold()
                pos = low.find(line_item.casefold())
                if pos >= 0:
                    a=max(0,pos-window); b=min(len(text),pos+max(window,len(line_item)+window))
                    key=(doc.get("document_id"),a,b)
                    if key not in seen:
                        seen.add(key)
                        chunks.append(
                            f"===== CONSTRAINT_ID: {cid} =====\nDOCUMENT_ID: {doc['document_id']}\nSOURCE_DOCUMENT: {doc['source_document']}\nEXACT LINE-ITEM EVIDENCE (prioritized):\n{text[a:b].strip()}\n"
                        )
                    found += 1
                    group_specific_found = True
                if family_prefix:
                    # Gather nearby numbered sibling rows; bounded to avoid flooding context.
                    for m in list(re.finditer(re.escape(family_prefix) + r"\s*\([a-zA-Z0-9]+\)", text, flags=re.I))[:8]:
                        a=max(0,m.start()-window//2); b=min(len(text),m.end()+window)
                        key=(doc.get("document_id"),a,b)
                        if key not in seen:
                            seen.add(key)
                            chunks.append(
                                f"===== CONSTRAINT_ID: {cid} =====\nDOCUMENT_ID: {doc['document_id']}\nSOURCE_DOCUMENT: {doc['source_document']}\nSIBLING BOQ COMPONENT CONTEXT (may belong to same parent package; do not treat as contradiction solely due to row id):\n{text[a:b].strip()}\n"
                            )
                        found += 1
                if group_specific_found:
                    break
        anchor=c.get("evidence_anchor") or {}
        # If the specific line item was found, do not let an unrelated generic
        # anchor become the validator's primary evidence for this constraint.
        if anchor.get("document_id") and not group_specific_found:
            doc=next((d for d in documents if d.get("document_id")==anchor.get("document_id")),None)
            if doc:
                text=doc.get("text") or ""
                # Normalized anchors intentionally have null offsets because their
                # positions refer to whitespace-collapsed text, not original PDF text.
                # Never cast those nulls to int or use them as source-text offsets.
                start = anchor.get("start")
                end = anchor.get("end")
                context_start = anchor.get("context_start")
                context_end = anchor.get("context_end")
                has_original_offsets = (
                    isinstance(start, int) and not isinstance(start, bool)
                    and isinstance(end, int) and not isinstance(end, bool)
                    and 0 <= start <= end <= len(text)
                )
                if isinstance(context_start, int) and not isinstance(context_start, bool):
                    a = context_start
                elif has_original_offsets:
                    a = max(0, start - window)
                else:
                    a = 0
                if isinstance(context_end, int) and not isinstance(context_end, bool):
                    b = context_end
                elif has_original_offsets:
                    b = min(len(text), end + window)
                else:
                    b = min(len(text), max(window * 2, 2000))
                a=max(0,min(len(text),a)); b=max(a,min(len(text),b))
                exact=text[start:end] if has_original_offsets else ""
                chunks.append(
                    f"===== CONSTRAINT_ID: {cid} =====\nDOCUMENT_ID: {doc['document_id']}\nSOURCE_DOCUMENT: {doc['source_document']}\nEXACT_SOURCE_EXCERPT:\n{exact}\nCONTEXT_AROUND_EXCERPT:\n{text[a:b].strip()}\n"
                )
                found=1
        for doc in ([] if group_specific_found else documents):
            text=(doc.get("text") or "")
            low=text.lower()
            positions=[]
            for term in terms:
                p=low.find(term.lower())
                if p>=0:
                    positions.append(p)
            # Prefer raw_text exact hit; otherwise attribute/value hit.
            for p in sorted(set(positions))[:3]:
                a=max(0,p-window); b=min(len(text),p+max(window, len(terms[0]) if terms else 0))
                key=(doc["document_id"],a,b)
                if key in seen: continue
                seen.add(key)
                snippet=text[a:b].strip()
                chunks.append(
                    f"===== CONSTRAINT_ID: {cid} =====\nDOCUMENT_ID: {doc['document_id']}\nSOURCE_DOCUMENT: {doc['source_document']}\nEVIDENCE:\n{snippet}\n"
                )
                found+=1
                if found>=2: break
            if found>=2: break
        if found==0:
            chunks.append(f"===== CONSTRAINT_ID: {cid} =====\nNO TARGETED EVIDENCE FOUND\n")
    bundle="\n".join(chunks)
    if len(bundle)>max_chars:
        # Keep per-constraint coverage; trim each snippet instead of dropping constraints.
        compact=[]
        for block in chunks:
            compact.append(block[:max(700, max_chars//max(1,len(chunks)))])
        bundle="\n".join(compact)[:max_chars]
    return bundle


async def run_validator_batch(client, model, sem, evidence_bundle, inputs, max_tokens, think_mode,
                             rate_limiter=None, api_budget=None, retries=5):
    prompt=render_prompt(LLM_VALIDATOR_PROMPT,
        evidence_bundle=evidence_bundle,
        extracted_json=json.dumps(inputs,ensure_ascii=False,separators=(",",":")))
    raw=await call_llm(client,model,sem,prompt,max_tokens,think_mode,
                       rate_limiter=rate_limiter,api_budget=api_budget,retries=retries)
    obj=await parse_or_repair_llm_json(
        client,model,sem,raw,max_tokens,think_mode,
        rate_limiter=rate_limiter,api_budget=api_budget,
    )
    errors=validate_validator_object(obj,inputs)
    if errors:
        raise ValueError("validator structural validation failed: "+" | ".join(errors))
    return obj


async def validate_in_batches(client, model, sem, documents, inputs, batch_size, min_batch_size,
                              max_tokens, think_mode, rate_limiter=None, api_budget=None, retries=5,
                              evidence_chars=70000, evidence_window=900):
    if not inputs:
        return {"schema_version":30,"validation_agent":"targeted_adversarial_validator","validation_status":"ok",
                "summary":{"input_constraints":0,"keep":0,"correct":0,"reject":0,},
                "validated_constraints":[]},[]
    all_items=[]; batch_records=[]; had_failure=False
    async def validate_slice(items,label):
        nonlocal had_failure
        evidence=build_targeted_evidence(items,documents,window=evidence_window,max_chars=evidence_chars)
        try:
            obj=await run_validator_batch(client,model,sem,evidence,items,max_tokens,think_mode,
                                          rate_limiter=rate_limiter,api_budget=api_budget,retries=retries)
            batch_records.append({"batch":label,"input_count":len(items),"status":"ok","evidence_chars":len(evidence)})
            all_items.extend(obj.get("validated_constraints",[])); return
        except JSONRepairFailed:
            raise
        except Exception as first_error:
            # v30: never recursively split a failed Agent-2 batch. Recursive
            # splitting was the source of dozens of extra calls on free endpoints.
            # Fail open: preserve the Agent-1 constraints and record the failure.
            reason=f"validator batch failed; original constraints retained without retry/splitting: {first_error!r}"
            batch_records.append({"batch":label,"input_count":len(items),"status":"failed_open_keep","reason":reason})
            all_items.extend(_validator_failure_audit(items,reason))
    for i in range(0,len(inputs),batch_size):
        await validate_slice(inputs[i:i+batch_size],f"B{i//batch_size+1:03d}")
    expected={str(c.get("constraint_id")) for c in inputs}; by_id={}
    for item in all_items:
        cid=str(item.get("constraint_id"))
        if cid in expected and cid not in by_id: by_id[cid]=item
    for c in inputs:
        cid=str(c.get("constraint_id"))
        if cid not in by_id:
            had_failure=True; by_id[cid]=_validator_failure_audit([c],"Validator produced no decision.")[0]
    ordered=[by_id[str(c.get("constraint_id"))] for c in inputs]
    counts={"input_constraints":len(inputs),"keep":0,"correct":0,"reject":0,}
    for x in ordered:
        d=normalize_validation_decision(x); counts[{"KEEP":"keep","CORRECT":"correct","REJECT":"reject"}[d]]+=1
    # Failed-open Agent-2 batches are auditable but are not fatal: originals survive.
    status="ok"
    obj={"schema_version":30,"validation_agent":"targeted_adversarial_validator","validation_status":status,
         "summary":counts,"validated_constraints":ordered,"batches":batch_records}
    return obj,ordered



# Deterministic post-validation quality controls. These labels do not delete the
# source constraint: they separate semantic audit data from IndiaMART matching data.
ADMIN_ONLY_PATTERNS = [
    re.compile(r"\breporting officer\b", re.I),
    re.compile(r"\btechnical clarification(?:s)?\b.{0,100}\b(?:hours?|days?|within|respond|reply)\b", re.I),
    re.compile(r"\b(?:upload|uploaded)\b.{0,100}\b(?:excel|spreadsheet|xls|xlsx)\b", re.I),
    re.compile(r"\bevaluation method\b", re.I),
    re.compile(r"\bbid opening\b|\btechnical bid opening\b|\bfinancial bid opening\b", re.I),
    re.compile(r"\bbid (?:start|end|submission|validity) date\b", re.I),
    re.compile(r"\b(reverse auction|bid to ra|bid-to-ra|e-reverse auction)\b", re.I),
]
LOCAL_CONTENT_RE = re.compile(
    r"\b(local content|make in india|class\s*[i1]\s*local supplier|class\s*[i2]\s*local supplier|"
    r"local supplier|minimum local content|local-content)\b", re.I
)
DELIVERY_DEST_RE = re.compile(
    r"\b(consignee|ship[- ]?to|deliver(?:y)? address|delivery location|destination address)\b", re.I
)

def apply_matching_quality_labels(constraints):
    """Attach scope/relevance labels without deleting source evidence or audit records."""
    out = []
    for original in constraints:
        c = dict(original)
        attr = snake(str(c.get("attribute") or ""))
        raw = clean(str(c.get("raw_text") or ""))
        blob = " ".join([
            raw, str(c.get("spec_name") or ""), str(c.get("attribute") or ""),
            str(c.get("buyer_value") or ""), str(c.get("constraint_type") or "")
        ])
        if LOCAL_CONTENT_RE.search(blob) or attr in {
            "local_content", "make_in_india", "local_supplier",
            "class_1_local_supplier", "class_2_local_supplier"
        }:
            c["scope"] = "supplier"
            c["constraint_category"] = "SUPPLIER"
            c["constraint_type"] = "supplier_requirement"
            c["matching_relevance"] = "supplier_eligibility"
        elif DELIVERY_DEST_RE.search(blob):
            c["scope"] = "transaction"
            c["constraint_category"] = "DELIVERY"
            c["constraint_type"] = "delivery"
            c["matching_relevance"] = "delivery_feasibility"
        elif any(p.search(blob) for p in ADMIN_ONLY_PATTERNS):
            c["scope"] = c.get("scope") if c.get("scope") in {"supplier", "transaction"} else "transaction"
            c["matching_relevance"] = "exclude_from_product_matching"
            c["administrative_only"] = True
        elif c.get("constraint_category") in {"PRODUCT", "TECHNICAL", "QUALITY", "CAPACITY"}:
            c["matching_relevance"] = "product_matching"
        elif c.get("constraint_category") in {"DELIVERY", "COMMERCIAL", "GEOGRAPHY"}:
            c["matching_relevance"] = "procurement_context"
        elif c.get("constraint_category") in {"SUPPLIER", "CERTIFICATION", "DOCUMENT"}:
            c["matching_relevance"] = "supplier_eligibility"
        else:
            c["matching_relevance"] = "review"
        c["indiamart_matchable"] = c["matching_relevance"] != "exclude_from_product_matching"
        # Provenance remains tied to the extracted source; no fabricated source IDs.
        anchor = c.get("evidence_anchor")
        if isinstance(anchor, dict) and anchor.get("document_id"):
            c["evidence_status"] = "anchored"
        else:
            c["evidence_status"] = "anchor_missing"
        out.append(c)
    return out


def apply_validation(inputs, validation_obj):
    by_id={str(x.get("constraint_id")):x for x in validation_obj.get("validated_constraints",[]) if isinstance(x,dict)}
    final=[]; audit=[]
    for original in inputs:
        cid=str(original.get("constraint_id")); item=by_id.get(cid)
        if not item:
            item={"constraint_id":cid,"decision":"KEEP","original_constraint":original,"validated_constraint":original,"evidence":[],"reason":"Validator omitted decision; original constraint retained.","validation_confidence":0.0}
        d=normalize_validation_decision(item)
        item=dict(item)
        item["decision"]=d
        evidence=item.get("evidence") or []
        has_evidence=any(
            isinstance(e,dict)
            and clean(str(e.get("raw_text") or ""))
            and clean(str(e.get("raw_text") or "")).casefold() != "no targeted evidence found"
            for e in evidence
        )
        # Missing/retrieval-failed evidence is UNKNOWN, not a contradiction.
        # In particular, never apply a proposed correction or rejection without evidence.
        if d in {"KEEP", "CORRECT", "REJECT"} and not has_evidence:
            original_decision = d
            d="KEEP"
            item["decision"]=d
            item["validated_constraint"] = original
            item["reason"]=(str(item.get("reason") or "").strip() +
                f" Evidence was incomplete; original constraint retained (proposed decision: {original_decision}; no affirmative contradiction established).").strip()
        audit.append(item)
        if d in {"KEEP", "NEEDS_REVIEW"}: final.append(dict(original))
        elif d=="CORRECT" and isinstance(item.get("validated_constraint"),dict):
            proposed=dict(item["validated_constraint"])
            for k in ("constraint_category","category"): 
                if proposed.get(k): proposed["constraint_category"]=proposed.get(k); break
            for k in ("constraint_id","bid_id","requirement_group_id","requirement_group_type","requirement_description","line_item_id","source_document_id","source_document"):
                if not proposed.get(k): proposed[k]=original.get(k)
            proposed["constraint_id"]=cid
            proposed["source"]="qwen_validator_corrected"
            final.append(normalize_group_constraint(proposed,proposed))
    return final,audit


# -----------------------------------------------------------------------------
# v20 RISK ROUTER
# Agent 1 is the complete semantic extractor. Agent 2 is expensive and is only
# used for constraints whose interpretation is plausibly ambiguous/high-impact.
# Low-risk, explicitly grounded constraints are mechanically auto-kept.
# -----------------------------------------------------------------------------
HIGH_RISK_ATTRIBUTES = {
    "capacity", "power", "rating", "load", "lifting_capacity", "crane_capacity",
    "span", "lift", "bay_length", "dimension", "dimensions", "weight",
    "quantity", "moq", "minimum_order_quantity", "delivery", "delivery_days",
    "lead_time", "warranty", "experience", "years_experience", "manufacturer",
    "oem", "brand", "make", "model", "certification", "certificate",
    "local_content", "country_of_origin", "geography", "location",
    "eligibility", "turnover", "financial_capacity", "performance_security",
    "payment", "emd", "epbg", "service_life", "life", "availability",
}
RISK_THRESHOLD = 3

HIGH_RISK_WORDS = (
    "plant", "project", "station", "building", "unit", "mw", "mva", "eligib",
    "manufacturer", "oem", "experience", "certificate", "certification",
    "local content", "make in india", "class 1", "class 2", "warranty",
    "guarantee", "delivery", "commission", "payment", "turnover", "security",
    "mandatory", "golden", "preferred", "alternative", "or", "either",
)

def constraint_risk_score(c: dict) -> tuple[int, list[str]]:
    score=0; reasons=[]
    attr=clean(str(c.get("attribute") or "")).lower()
    spec=clean(str(c.get("spec_name") or "")).lower()
    raw=clean(str(c.get("raw_text") or "")).lower()
    scope=c.get("scope")
    category=c.get("constraint_category")
    if attr in HIGH_RISK_ATTRIBUTES:
        score += 3; reasons.append("high_risk_attribute")
    if any(w in (raw+" "+spec) for w in HIGH_RISK_WORDS):
        score += 2; reasons.append("high_risk_evidence_word")
    if scope in ("supplier", "transaction"):
        score += 2; reasons.append("supplier_or_transaction_scope")
    if category in ("CAPACITY", "CERTIFICATION", "SUPPLIER", "DELIVERY", "COMMERCIAL", "GEOGRAPHY"):
        score += 2; reasons.append("high_impact_category")
    if c.get("condition") is not None:
        score += 2; reasons.append("conditional")
    if c.get("allowed_values") and len(c.get("allowed_values") or []) > 1:
        score += 1; reasons.append("alternatives")
    if c.get("operator") in ("conditional", "required", "prohibited"):
        score += 1; reasons.append("non_simple_operator")
    # Project/plant-vs-product confusion is one of the known failure modes.
    if re.search(r"\b(project|plant|station|building|unit)\b", raw) and c.get("scope") == "product":
        score += 4; reasons.append("possible_context_vs_product_confusion")
    return score, reasons

def route_constraints(constraints: list, representation_flags: list) -> tuple[list, list, dict]:
    risky=[]; safe=[]; audit={}
    # If representation integrity is suspect, validate the whole affected bid rather
    # than trusting deterministic routing. This protects multi-instance groups.
    force_all = bool(representation_flags)
    for c in constraints:
        score,reasons=constraint_risk_score(c)
        if force_all:
            score=max(score,3); reasons=list(dict.fromkeys(reasons+["representation_flags"]))
        cid=str(c.get("constraint_id"))
        audit[cid]={"risk_score":score,"risk_reasons":reasons,"validation_route":"agent2" if score>=RISK_THRESHOLD else "auto_keep"}
        (risky if score>=RISK_THRESHOLD else safe).append(c)
    return risky,safe,audit

def apply_scope_ontology_guard(constraints: list) -> list:
    """v23 deterministic semantic scope guard. Document placement never decides scope."""
    supplier_attrs = {
        "local_content", "make_in_india", "local_supplier", "class_1_local_supplier",
        "class_2_local_supplier", "manufacturer", "oem", "authorized_manufacturer",
        "authorized_oem", "supplier_experience", "experience", "years_experience",
        "supplier_turnover", "turnover", "supplier_credentials", "supplier_eligibility",
    }
    transaction_attrs = {
        "delivery", "delivery_days", "lead_time", "payment", "payment_terms",
        "site_visit", "installation", "commissioning", "training",
    }
    out=[]
    for c in constraints:
        x=dict(c)
        attr=snake(str(x.get("attribute") or "")).lower()
        raw=clean(str(x.get("raw_text") or "")).lower()
        if attr in supplier_attrs or any(k in raw for k in (
            "local content", "class 1 local supplier", "class 2 local supplier",
            "make in india", "oem", "authorized manufacturer", "supplier experience",
        )):
            x["scope"]="supplier"
            if attr in {"local_content", "make_in_india", "local_supplier", "class_1_local_supplier", "class_2_local_supplier"}:
                x["constraint_category"]="SUPPLIER"
        elif attr in transaction_attrs:
            x["scope"]="transaction"
        out.append(x)
    return out


async def process_bid(pool, client, sem, pdf: Path, args, out_dir: Path,
                     rate_limiter=None, api_budget=None):
    """v30 bounded API calls + robust Tesseract discovery + group-aware validation: ingest all documents once -> full bundle when it fits, otherwise overlapping document-aware Agent-1 chunks -> targeted Agent-2 validation."""
    bid_id=pdf.stem
    try:
        loop=asyncio.get_running_loop()
        parsed=await loop.run_in_executor(pool,parse_pdf,str(pdf))
        bid_id=extract_actual_bid_id(parsed,pdf); safe_bid_id=safe_bid_folder_name(bid_id)
        out_path=out_dir/f"{safe_bid_id}.json"; bid_folder=out_dir/safe_bid_id; linked_dir=bid_folder/"linked_documents"
        linked_dir.mkdir(parents=True,exist_ok=True)
        if out_path.exists() and not args.force:
            try:
                old=json.loads(out_path.read_text(encoding="utf-8"))
                if old.get("schema_version")==30 and old.get("architecture")=="two_agent_bounded_api_validation_v30" and "final_constraints" in old:
                    return bid_id,"skipped",None
            except Exception: pass
        linked_docs=[]; linked_manifest=[]; queued_links=[]; seen_urls=set()
        for link in (_relevant_document_links(parsed) if not args.no_linked_docs else []):
            raw_url=link.get("url")
            if not raw_url: continue
            url=urljoin(args.base_url,raw_url) if args.base_url else raw_url
            if url not in seen_urls:
                seen_urls.add(url); queued_links.append((url,link))
        link_index=1; failed_documents=[]; all_flags=[]
        while queued_links:
            url,link=queued_links.pop(0); filename=linked_filename(url,link_index)
            entry={"index":link_index,"url":url,"context":link.get("context"),"local_filename":filename,
                   "local_path":str(Path("linked_documents")/filename),"downloaded":False,"parsed":False,"error":None}
            p,content_type=await loop.run_in_executor(pool,_download_linked_document,url,linked_dir,filename)
            entry["content_type"]=content_type
            if not p:
                entry["error"]="download_failed_or_unsupported_format"; linked_manifest.append(entry); failed_documents.append(url); all_flags.append(f"linked document download failed: {url}"); link_index+=1; continue
            entry["local_filename"]=p.name; entry["local_path"]=str(Path("linked_documents")/p.name); entry["downloaded"]=True
            try:
                d=await loop.run_in_executor(pool,parse_linked_document,p,p.name,url,link.get("context"))
            except Exception as e:
                entry["error"]=f"parse_failed: {e!r}"; linked_manifest.append(entry); failed_documents.append(url); all_flags.append(f"linked document parse failed: {p.name}"); link_index+=1; continue
            entry["parsed"]=True; linked_manifest.append(entry); linked_docs.append(d)
            for child in (_relevant_document_links(d) if not args.no_linked_docs else []):
                child_raw=child.get("url")
                if child_raw:
                    child_url=urljoin(url,child_raw)
                    if child_url not in seen_urls:
                        seen_urls.add(child_url); queued_links.append((child_url,child))
            link_index+=1
        documents=[{"document_id":"MAIN_BID","source_kind":"main_bid","source_document":parsed.get("source_document") or pdf.name,"text":parsed.get("full_text","") or "",
                     "page_count":parsed.get("page_count"),"ocr_pages":parsed.get("ocr_pages",0),"ocr_attempted_pages":parsed.get("ocr_attempted_pages",0),"ocr_successful_pages":parsed.get("ocr_successful_pages",0),
                     "native_text_chars":parsed.get("native_text_chars"),"extracted_text_chars":parsed.get("extracted_text_chars"),
                     "ingestion_warnings":parsed.get("ingestion_warnings",[]),"ocr_errors":parsed.get("ocr_errors",[]),"page_quality":parsed.get("page_quality",[]),"spec_rows":parsed.get("spec_rows",[])}]
        for i,d in enumerate(linked_docs,1):
            documents.append({"document_id":f"DOC_{i:02d}","source_kind":"linked_document","source_document":d.get("source_document") or f"linked_document_{i}","text":d.get("full_text","") or "","linked_from":d.get("linked_from"),"link_context":d.get("link_context"),"local_path":d.get("local_path"),
                "page_count":d.get("page_count"),"ocr_pages":d.get("ocr_pages",0),"ocr_attempted_pages":d.get("ocr_attempted_pages",0),"ocr_successful_pages":d.get("ocr_successful_pages",0),"native_text_chars":d.get("native_text_chars"),
                "extracted_text_chars":d.get("extracted_text_chars"),"ingestion_warnings":d.get("ingestion_warnings",[]),"ocr_errors":d.get("ocr_errors",[]),"page_quality":d.get("page_quality",[]),"spec_rows":d.get("spec_rows",[])})
        if not client: raise RuntimeError("v25 requires the LLM for final semantic constraints. Use --no_llm only for ingestion/testing.")
        all_documents = documents
        # Prompt budget policy: no separate document-triage LLM call. Include all
        # successfully parsed documents and let Agent 1 classify relevant content.
        documents = all_documents
        triage_audit = [{"document_id": d.get("document_id"),
                         "source_document": d.get("source_document"),
                         "decision": "INCLUDE", "reason": "LLM triage removed; Agent 1 handles relevance",
                         "method": "no_llm_triage"} for d in documents]
        # Quality-gate only documents selected for semantic extraction. Full ingestion
        # diagnostics and triage decisions for every document remain in the manifest.
        # If a technical PDF is badly truncated, continue extracting what is available.
        # extracting what is available but mark the bid for review and preserve the warning.
        for d in documents:
            text_chars=len(re.sub(r"\s+", "", d.get("text", "") or ""))
            page_count=d.get("page_count")
            if page_count and text_chars < max(500, int(page_count) * 40):
                warning=f"low_text_yield:{text_chars}_chars_for_{page_count}_pages"
                if warning not in d.setdefault("ingestion_warnings", []):
                    d["ingestion_warnings"].append(warning)
        # Build the logged bundle from the exact post-triage document set. Do not
        # accidentally count or pass skipped linked documents to downstream stages.
        sections=[]
        for d in documents:
            text=(d.get("text") or "").strip()
            if not text: continue
            sections.append(f"===== DOCUMENT_ID: {d.get('document_id')} =====\nSOURCE_DOCUMENT: {d.get('source_document')}\n===== DOCUMENT_TEXT_START =====\n{text}\n===== DOCUMENT_TEXT_END =====")
        bundle="\n\n".join(sections)
        chunks = build_extraction_chunks(documents, args.llm_context_chars, args.chunk_overlap_chars)
        print(f"  [AGENT1] {bid_id} | {len(documents)} docs | {len(bundle):,} included chars | {len(chunks)} extraction chunk(s)",file=sys.stderr)

        # If the whole bundle fits, preserve v23 exactly: one full-bundle call.
        # If it does not, use document-aware overlapping chunks. Nothing is discarded.
        chunk_objects=[]
        chunk_failures=[]
        for chunk in chunks:
            meta=json.dumps({
                "chunk_id": chunk["chunk_id"],
                "documents": chunk["documents"],
                "parts": chunk["parts"],
                "total_bid_documents": len(documents),
            }, ensure_ascii=False, separators=(",", ":"))
            if len(chunks)==1 and len(bundle)<=args.llm_context_chars:
                prompt=render_prompt(LLM_RETRACE_PROMPT,source_bundle=bundle)
            else:
                prompt=render_prompt(LLM_CHUNK_PROMPT,source_bundle=chunk["text"],chunk_metadata=meta)
            print(f"    [AGENT1 {chunk['chunk_id']}] {chunk['chars']:,} chars | docs={','.join(chunk['documents'])}",file=sys.stderr)
            try:
                raw=await call_llm(client,args.model,sem,prompt,args.max_new_tokens,args.no_think_mode,rate_limiter=rate_limiter,api_budget=api_budget)
                obj=await parse_or_repair_llm_json(
                    client,args.model,sem,raw,args.max_new_tokens,args.no_think_mode,
                    rate_limiter=rate_limiter,api_budget=api_budget,
                )
                chunk_objects.append({"chunk":chunk,"object":obj})
            except JSONRepairFailed:
                raise
            except Exception as e:
                chunk_failures.append({"chunk_id":chunk["chunk_id"],"error":repr(e),"documents":chunk["documents"]})
                print(f"    [CHUNK FAILED] {bid_id} {chunk['chunk_id']}: {e!r}",file=sys.stderr)

        if not chunk_objects:
            raise RuntimeError(f"Agent-1 extraction failed for every chunk: {chunk_failures}")

        raw_constraints=[]
        metadata_objects=[]
        for co in chunk_objects:
            obj=co["object"]
            metadata_objects.append(obj)
            chunk=co["chunk"]
            chunk_doc_ids=set(chunk["documents"])
            for c in flatten_qwen_groups(obj):
                # If Qwen omitted the source document, recover it when the chunk
                # contains exactly one source document. Otherwise leave the v23
                # fallback behavior to MAIN_BID rather than inventing a source.
                if not c.get("source_document_id") and len(chunk_doc_ids)==1:
                    c["source_document_id"]=next(iter(chunk_doc_ids))
                c["_extraction_chunk_id"]=chunk["chunk_id"]
                c["_chunk_parts"]=chunk["parts"]
                raw_constraints.append(c)

        # Deterministic technical-table completeness audit. If parsed table rows were
        # missed by Agent 1, make exactly one targeted recovery call for those rows.
        expected_rows = []
        for d in documents:
            for row in d.get("spec_rows", []) or []:
                name = clean(str(row.get("name") or ""))
                value = clean(str(row.get("value") or ""))
                allowed = clean(str(row.get("allowed_raw") or ""))
                if name and (value or allowed) and value not in {"-", "—", "NA", "N/A"}:
                    expected_rows.append({"source_document_id": d.get("document_id"), "source_document": d.get("source_document"),
                        "group": row.get("group"), "attribute": name, "catalog_value": value,
                        "buyer_requirement_allowed_values": allowed, "golden_param": bool(row.get("golden")),
                        "raw_text": f"{name}: Values={value}; Bid Requirement (Allowed Values)={allowed}"[:500]})
        def norm_spec(v):
            return re.sub(r"[^a-z0-9]+", "", str(v or "").casefold())
        def row_is_covered(row, constraints):
            row_name = norm_spec(row.get("attribute"))
            for c in constraints:
                names = {norm_spec(c.get("attribute")), norm_spec(c.get("spec_name"))}
                raw = norm_spec(c.get("raw_text"))
                if row_name and (row_name in names or (len(row_name) >= 5 and row_name in raw)):
                    return True
            return False
        missing_rows = [r for r in expected_rows if not row_is_covered(r, raw_constraints)]
        completeness_audit = {"parsed_spec_rows": len(expected_rows), "initial_constraints": len(raw_constraints),
                              "missing_spec_rows_before_recovery": len(missing_rows), "recovery_attempted": False, "recovered_constraints": 0}
        if missing_rows and not args.no_llm and client:
            # Bound the recovery payload: one compact call with only missing table rows.
            completeness_audit["recovery_attempted"] = True
            print(f"  [COVERAGE] {bid_id} | {len(missing_rows)}/{len(expected_rows)} parsed spec rows missing; one targeted recovery call", file=sys.stderr)
            recovery_prompt = render_prompt(LLM_MISSING_SPEC_RECOVERY_PROMPT,
                existing_constraints=json.dumps(raw_constraints, ensure_ascii=False)[:18000],
                missing_rows=json.dumps(missing_rows, ensure_ascii=False)[:24000])
            try:
                recovery_raw = await call_llm(client, args.model, sem, recovery_prompt,
                    min(args.max_new_tokens, 8192), args.no_think_mode, rate_limiter=rate_limiter, api_budget=api_budget)
                recovery_obj = await parse_or_repair_llm_json(client, args.model, sem, recovery_raw,
                    min(args.max_new_tokens, 8192), args.no_think_mode, rate_limiter=rate_limiter, api_budget=api_budget)
                recovered = flatten_qwen_groups(recovery_obj)
                recovered = [c for c in recovered if not row_is_covered({"attribute": c.get("spec_name") or c.get("attribute")}, raw_constraints)]
                for c in recovered:
                    c["_extraction_chunk_id"] = "TARGETED_SPEC_RECOVERY"
                    raw_constraints.append(c)
                completeness_audit["recovered_constraints"] = len(recovered)
                completeness_audit["missing_spec_rows_after_recovery"] = sum(1 for r in expected_rows if not row_is_covered(r, raw_constraints))
            except Exception as e:
                completeness_audit["recovery_error"] = repr(e)
                completeness_audit["missing_spec_rows_after_recovery"] = len(missing_rows)
        else:
            completeness_audit["missing_spec_rows_after_recovery"] = len(missing_rows)

        # Use the first metadata-bearing extraction for bid-level fields, preferring
        # non-null values across chunks rather than requiring one chunk to contain all metadata.
        obj={
            "schema_version":30,
            "product_name":next((x.get("product_name") for x in metadata_objects if x.get("product_name")),None),
            "product_category":next((x.get("product_category") for x in metadata_objects if x.get("product_category")),None),
            "indiamart_search_query":next((x.get("indiamart_search_query") for x in metadata_objects if x.get("indiamart_search_query")),None),
            "office_name":next((x.get("office_name") for x in metadata_objects if x.get("office_name")),None),
        }
        # Make chunk failures auditable; successful chunks still contribute.
        failed_documents.extend(chunk_failures)
        for c in raw_constraints:
            gid=c.get("requirement_group_id") or "UNSPECIFIED_GROUP"
            c["requirement_group_id"]=gid
            valid_doc_ids = {d["document_id"] for d in documents}
            source_id = c.get("source_document_id")
            if source_id not in valid_doc_ids:
                chunk_id = c.get("_extraction_chunk_id")
                chunk_meta = next((x for x in chunks if x.get("chunk_id") == chunk_id), None)
                chunk_doc_ids = (chunk_meta or {}).get("documents", [])
                if len(chunk_doc_ids) == 1 and chunk_doc_ids[0] in valid_doc_ids:
                    source_id = chunk_doc_ids[0]
                else:
                    raw = clean(str(c.get("raw_text") or "")).casefold()
                    matches = []
                    if len(raw) >= 12:
                        for candidate in documents:
                            source_text = clean(str(candidate.get("text") or "")).casefold()
                            if raw in source_text:
                                matches.append(candidate["document_id"])
                            else:
                                norm_source = re.sub(r"\s+", " ", source_text)
                                norm_raw = re.sub(r"\s+", " ", raw)
                                if norm_raw in norm_source:
                                    matches.append(candidate["document_id"])
                    source_id = matches[0] if len(set(matches)) == 1 else None
            c["source_document_id"] = source_id
            if not c.get("source_document") and source_id:
                c["source_document"] = next((d["source_document"] for d in documents if d["document_id"] == source_id), None)
        extraction_constraints=canonicalize_final_constraints(raw_constraints)
        # Qwen's group prose is audit-only. Replace it in the canonical representation
        # with a deterministic label so unsupported narrative cannot become dataset truth.
        for c in extraction_constraints:
            gid=str(c.get("requirement_group_id") or "UNSPECIFIED_GROUP")
            gtype=str(c.get("requirement_group_type") or "other")
            c["requirement_description"]=f"{gtype}:{gid}"
        attach_exact_evidence_anchors(extraction_constraints,documents)
        # Force the single decisive priority field.
        for c in extraction_constraints:
            g=c.get("golden_param",False)
            c["golden_param"]=bool(g) if not isinstance(g,str) else g.lower().strip() in ("true","yes","1")
            for k in ("hard_soft","negotiable","likely_hard_constraint"):
                c.pop(k,None)
            key=json.dumps({"bid_id":bid_id,"group":c.get("requirement_group_id"),"category":c.get("constraint_category"),"scope":c.get("scope"),"attribute":c.get("attribute"),"operator":c.get("operator"),"buyer_value":c.get("buyer_value"),"allowed_values":c.get("allowed_values"),"condition":c.get("condition")},sort_keys=True,ensure_ascii=False)
            c["constraint_id"]=f"{bid_id}_C_{hashlib.sha1(key.encode()).hexdigest()[:12]}"; c["bid_id"]=bid_id
        representation_flags=validate_representation(extraction_constraints)
        all_flags.extend(representation_flags)
        validation_obj=None; validation_audit=[]; validation_status="not_run"
        global RISK_THRESHOLD
        RISK_THRESHOLD = max(1, int(args.validator_risk_threshold))
        risky_constraints, safe_constraints, risk_audit = route_constraints(extraction_constraints, representation_flags)
        if args.no_validator:
            validation_status="disabled"; final_constraints=extraction_constraints
            validation_audit=[{"constraint_id":str(c.get("constraint_id")),"decision":"KEEP","route":"disabled_validator"} for c in extraction_constraints]
        elif not risky_constraints:
            validation_status="ok"; final_constraints=extraction_constraints
            validation_audit=[{"constraint_id":str(c.get("constraint_id")),"decision":"KEEP","route":"auto_keep","reason":"low-risk explicit constraint"} for c in extraction_constraints]
        else:
            print(f"  [AGENT2] {bid_id} | one validation call for {len(risky_constraints)}/{len(extraction_constraints)} high-risk constraints",file=sys.stderr)
            validation_obj,agent2_audit=await validate_in_batches(client,args.model,sem,documents,risky_constraints,
                batch_size=max(1, len(risky_constraints)),min_batch_size=max(1, len(risky_constraints)),
                max_tokens=args.validator_max_new_tokens or args.max_new_tokens,think_mode=args.no_think_mode,
                rate_limiter=rate_limiter,api_budget=api_budget,retries=1,
                evidence_chars=args.validator_evidence_chars,evidence_window=args.validator_evidence_window)
            validated_high,_=apply_validation(risky_constraints,validation_obj)
            validated_by_id={str(c.get("constraint_id")):c for c in validated_high}
            # Safe constraints bypass Agent 2; high-risk constraints are replaced only
            # Agent 2 has no review state; ambiguous or unverified items default to KEEP.
            final_constraints=list(safe_constraints)+list(validated_high)
            validation_audit=[]
            by_id={str(x.get("constraint_id")):x for x in agent2_audit}
            for c in extraction_constraints:
                cid=str(c.get("constraint_id"))
                if cid in by_id:
                    x=dict(by_id[cid]); x["route"]="agent2"; validation_audit.append(x)
                else:
                    validation_audit.append({"constraint_id":cid,"decision":"KEEP","route":"auto_keep","reason":"low-risk explicit constraint"})
            validation_status=validation_obj.get("validation_status") or "ok"
            validation_status = "ok"  # Agent-2 failures retain original constraints in v30.
        final_constraints=canonicalize_final_constraints(final_constraints)
        final_constraints=apply_scope_ontology_guard(final_constraints)
        final_constraints=apply_matching_quality_labels(final_constraints)
        for c in final_constraints:
            gid=str(c.get("requirement_group_id") or "UNSPECIFIED_GROUP")
            gtype=str(c.get("requirement_group_type") or "other")
            c["requirement_description"]=f"{gtype}:{gid}"
        attach_exact_evidence_anchors(final_constraints,documents)
        for c in final_constraints:
            c.pop("hard_soft",None); c.pop("negotiable",None); c.pop("likely_hard_constraint",None)
            c["golden_param"]=bool(c.get("golden_param",False)) if not isinstance(c.get("golden_param"),str) else c.get("golden_param").lower().strip() in ("true","yes","1")
        main_obj=obj
        contacts=extract_contacts(parsed)
        office_name=parsed.get("header",{}).get("Office Name") or main_obj.get("office_name")
        product_name=main_obj.get("product_name"); product_category=main_obj.get("product_category"); search_query=main_obj.get("indiamart_search_query")
        ingestion_warnings = [f"{d['document_id']}:{w}" for d in documents for w in d.get("ingestion_warnings", [])]
        # A named technical/specification PDF with very low yield is a hard review gate.
        # Keep the text for traceability, but never report a clean success.
        for d in documents:
            name = str(d.get("source_document") or "")
            text_chars = len(re.sub(r"\s+", "", d.get("text", "") or ""))
            if d.get("document_id") != "MAIN_BID" and TECHNICAL_DOC_RE.search(name) and d.get("page_count") and text_chars < max(1500, int(d["page_count"]) * 80):
                gate = f"{d['document_id']}:critical_technical_document_ingestion_incomplete"
                if gate not in ingestion_warnings:
                    ingestion_warnings.append(gate)
        all_flags.extend(ingestion_warnings)
        status="failed" if failed_documents else "ok"
        (bid_folder/"extraction.json").write_text(json.dumps({"schema_version":30,"agent":"qwen_constraint_extractor","bid_id":bid_id,"constraints":extraction_constraints},indent=2,ensure_ascii=False),encoding="utf-8")
        (bid_folder/"chunk_extractions.json").write_text(json.dumps({
            "schema_version":30,
            "bid_id":bid_id,
            "architecture":"two_agent_bounded_api_validation_v31",
            "total_bundle_chars":len(bundle),
            "chunk_limit_chars":args.llm_context_chars,
            "chunk_overlap_chars":args.chunk_overlap_chars,
            "chunks":[{
                "chunk_id":co["chunk"]["chunk_id"],
                "chars":co["chunk"]["chars"],
                "documents":co["chunk"]["documents"],
                "parts":co["chunk"]["parts"],
                "n_constraints":len(flatten_qwen_groups(co["object"]))
            } for co in chunk_objects],
            "failed_chunks":chunk_failures
        },indent=2,ensure_ascii=False),encoding="utf-8")
        (bid_folder/"validation.json").write_text(json.dumps(validation_obj or {"schema_version":30,"validation_agent":"targeted_adversarial_validator","validation_status":validation_status,"summary":{},"validated_constraints":validation_audit},indent=2,ensure_ascii=False),encoding="utf-8")
        (bid_folder/"document_extractions.json").write_text(json.dumps({
            "schema_version":30,
            "bid_id":bid_id,
            "documents":[{
                "document_id":d["document_id"],
                "source_document":d["source_document"],
                "chars":len(d.get("text","") or ""),
                "nonspace_chars":len(re.sub(r"\s+","",d.get("text","") or "")),
                "page_count":d.get("page_count"),
                "ocr_pages":d.get("ocr_pages",0),
                "native_text_chars":d.get("native_text_chars"),
                "extracted_text_chars":d.get("extracted_text_chars",len(re.sub(r"\s+","",d.get("text","") or ""))),
                "ocr_errors":d.get("ocr_errors",[]),
                "page_quality":d.get("page_quality",[]),
                "ingestion_warnings":d.get("ingestion_warnings",[])
            } | {"triage_decision": d.get("triage_decision", "INCLUDE"), "triage_reason": d.get("triage_reason", "")}
              for d in all_documents],
            "document_triage":triage_audit,
            "agent1_bundle_chars":len(bundle),
            "agent1_approx_input_tokens":max(1,len(bundle)//4),
            "agent1_extraction_chunks":len(chunks),
            "agent1_chunk_chars_limit":args.llm_context_chars,
            "agent1_chunk_overlap_chars":args.chunk_overlap_chars,
            "chunk_manifest":[{"chunk_id":x["chunk_id"],"chars":x["chars"],"documents":x["documents"],"parts":x["parts"]} for x in chunks],
            "ingestion_warnings":[f"{d['document_id']}:{w}" for d in documents for w in d.get("ingestion_warnings",[])]
        },indent=2,ensure_ascii=False),encoding="utf-8")
        result={"schema_version":30,"architecture":"two_agent_bounded_api_validation_v30","bid_id":bid_id,"source_pdf":pdf.name,"source_pdf_path":str(pdf),"header":parsed.get("header",{}),"product_category":product_category,"product_name":product_name,"indiamart_search_query":search_query,"office_name":office_name,"contacts":contacts,
                "linked_spec_documents":[{"document_id":f"DOC_{i:02d}","source_document":d.get("source_document"),"local_path":d.get("local_path")} for i,d in enumerate(linked_docs,1)],
                "n_linked_spec_documents":len(linked_docs),"n_documents_total":len(all_documents),"n_documents_extracted":len(documents),"document_triage":triage_audit,"agent1_bundle_chars":len(bundle),"agent1_approx_input_tokens":max(1,len(bundle)//4),"agent1_extraction_chunks":len(chunks),"agent1_chunk_chars_limit":args.llm_context_chars,"agent1_chunk_overlap_chars":args.chunk_overlap_chars,"n_final_constraints":len(final_constraints),"completeness_audit":completeness_audit,"representation_flags":all_flags,"validation_status":validation_status,"validation_summary":(validation_obj or {}).get("summary",{}),"risk_summary":{"total":len(extraction_constraints),"agent2":len(risky_constraints),"auto_keep":len(safe_constraints)},"risk_audit":risk_audit,"final_constraints":final_constraints,"linked_manifest":linked_manifest,"failed_documents":failed_documents}
        out_path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        return bid_id,status,None
    except Exception as e:
        return bid_id,"failed",{"bid_id":bid_id,"error":repr(e),"traceback":traceback.format_exc()}

def write_csvs(out_dir: Path):
    final_rows, matching_rows, bids = [], [], []

    for p in sorted(out_dir.glob("*.json")):
        if p.name == "failures.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("schema_version") != 30:
            continue

        bids.append({
            "bid_id": d.get("bid_id", ""),
            "source_pdf": d.get("source_pdf", ""),
            "product_category": d.get("product_category", ""),
            "product_name": d.get("product_name", ""),
            "indiamart_search_query": d.get("indiamart_search_query", ""),
            "office_name": d.get("office_name", ""),
            "contacts": json.dumps(d.get("contacts", []), ensure_ascii=False),
            "n_linked_spec_documents": d.get("n_linked_spec_documents", 0),
            "n_retrace_stages": d.get("n_retrace_stages", 0),
            "n_final_constraints": d.get("n_final_constraints", 0),
            "representation_flags": " | ".join(d.get("representation_flags", [])),
        })

        for c in d.get("final_constraints", []):
            x = dict(c)
            x["bid_id"] = d.get("bid_id", "")
            for k, v in list(x.items()):
                if isinstance(v, (list, dict)):
                    x[k] = json.dumps(v, ensure_ascii=False)
            final_rows.append(x)
            if c.get("matching_relevance") != "exclude_from_product_matching":
                matching_rows.append(dict(x))

    def dump_csv(filename, rows):
        if not rows:
            return
        fields = sorted({k for r in rows for k in r})
        with open(out_dir / filename, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # There is intentionally one semantic CSV in v16.
    dump_csv("final_constraints.csv", final_rows)
    dump_csv("bid_constraints.csv", final_rows)
    dump_csv("matching_constraints.csv", matching_rows)

    if bids:
        fields = list(bids[0])
        with open(out_dir / "bids.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(bids)

    return len(final_rows)



# ---- Local Transformers backend (single-GPU inference) -----------------------
from types import SimpleNamespace

class LocalQwenCompletions:
    def __init__(self, backend):
        self.backend = backend

    async def create(self, *, model=None, max_tokens=4096, temperature=0, messages=None,
                     response_format=None, **kwargs):
        # This pipeline sends one user message per call. Run model inference in a
        # worker thread so the asyncio event loop remains responsive.
        import asyncio
        prompt = "\n\n".join(str(m.get("content", "")) for m in (messages or []))
        content = await asyncio.to_thread(
            self.backend.generate, prompt, int(max_tokens or 4096), float(temperature or 0)
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

class LocalQwenChat:
    def __init__(self, backend):
        self.completions = LocalQwenCompletions(backend)

class LocalQwenClient:
    """Small AsyncOpenAI-compatible shim so pipeline prompt/validation code is reused."""
    def __init__(self, model_id, max_memory_gib=None, max_input_tokens=None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.lock = __import__('threading').Lock()
        print(f"[LLM] loading local model {model_id} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
        kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": False}
        if max_memory_gib:
            kwargs["max_memory"] = {0: f"{max_memory_gib}GiB", "cpu": "64GiB"}
        # Recommended model repo includes its own bitsandbytes 4-bit quantization config.
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.chat = LocalQwenChat(self)

    def generate(self, prompt, max_new_tokens, temperature=0):
        import torch
        with self.lock:
            messages = [{"role": "user", "content": prompt}]
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[rendered], return_tensors="pt")
            # Keep input tensors on the model's first device; device_map handles sharding.
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if self.max_input_tokens and inputs.get("input_ids") is not None:
                n = inputs["input_ids"].shape[-1]
                if n > self.max_input_tokens:
                    raise RuntimeError(f"Prompt is {n} tokens, above --local_max_input_tokens={self.max_input_tokens}. Reduce --llm_context_chars.")
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95
            with torch.inference_mode():
                output = self.model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[-1]
            generated = output[0][input_len:]
            text = self.processor.decode(generated, skip_special_tokens=True)
            return text.strip()

    async def close(self):
        return None


async def amain(args):
    # Fail before downloading/extracting a dataset if OCR cannot actually run.
    # This prevents another misleading one-constraint output caused by a missing PATH entry.
    configure_tesseract(required=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdfs = [Path(p) for p in args.pdf]
    else:
        if not args.input_dir:
            sys.exit("Provide either --pdf (repeatable) or --input_dir")
        pdfs = sorted(args.input_dir.glob("*.pdf"))
        if args.limit:
            pdfs = pdfs[:args.limit]

    missing = [str(p) for p in pdfs if not p.exists()]
    if missing:
        sys.exit("PDF(s) not found: " + ", ".join(missing))
    if not pdfs:
        sys.exit(f"No PDFs in {args.input_dir}")

    client = None
    api_provider = None

    if not args.no_llm:
        if args.backend == "local":
            client = LocalQwenClient(
                model_id=args.local_model,
                max_memory_gib=args.local_max_memory_gib,
                max_input_tokens=args.local_max_input_tokens,
            )
            api_provider = "local-transformers"
            # GPU inference is serialized; concurrent generate calls can exhaust VRAM.
            args.concurrency = 1
            args.min_request_interval = 0
        else:
            from openai import AsyncOpenAI
            base_url = (
                args.base_url
                or os.getenv("OPENAI_BASE_URL")
                or ("https://openrouter.ai/api/v1" if os.getenv("OPENAI_API_KEY") else "")
            )
            base_url_l = base_url.lower()
            if args.api_key_env:
                api_key = os.getenv(args.api_key_env)
                if not api_key:
                    raise RuntimeError(f"Environment variable {args.api_key_env!r} is not set.")
            elif "openrouter.ai" in base_url_l:
                api_key = args.api_key or os.getenv("OPENAI_API_KEY")
                api_provider = "openrouter"
                if not api_key:
                    raise RuntimeError("OpenRouter selected but no key was found. Set OPENAI_API_KEY or pass --api_key.")
            else:
                api_key = args.api_key or os.getenv("OPENAI_API_KEY")
                api_provider = "openai-compatible"
                if not api_key:
                    raise RuntimeError("No API key found. Set OPENAI_API_KEY or pass --api_key.")
            client_kwargs = {"api_key": api_key, "timeout": 300, "max_retries": 0}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = AsyncOpenAI(**client_kwargs)
            print(f"[LLM] provider={api_provider} base_url={base_url or 'default'} model={args.model}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    rate_limiter = RateLimiter(args.min_request_interval) if client else None
    api_budget = APIDailyBudget(args.api_budget_file or (args.output_dir / "api_usage.json"), args.max_api_requests) if client else None
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        async def guarded(p):
            return await process_bid(
                pool, client, sem, p, args, args.output_dir, rate_limiter=rate_limiter, api_budget=api_budget
            )

        tasks = [asyncio.create_task(guarded(p)) for p in pdfs]
        try:
            from tqdm.asyncio import tqdm_asyncio
            it = tqdm_asyncio.as_completed(tasks, total=len(tasks))
        except ImportError:
            it = asyncio.as_completed(tasks)

        for fut in it:
            bid_id, status, fail = await fut
            counts[status] += 1
            if fail:
                failures.append(fail)
                print(f"[{bid_id}] FAILED: {fail['error']}", file=sys.stderr)
                if fail.get("traceback"):
                    print(fail["traceback"], file=sys.stderr)

    n_final = write_csvs(args.output_dir)
    print(
        f"\nDone: {counts}. final_constraints={n_final} -> "
        f"final_constraints.csv, bids.csv",
        file=sys.stderr,
    )
    if failures:
        (args.output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )


def main():
    ap = argparse.ArgumentParser(
        description="GeM bid semantic extractor v30: bounded API calls, two-agent chunking, conditional JSON repair, selective OCR"
    )
    ap.add_argument("--input_dir", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--pdf", action="append", default=[],
        help="Process an exact PDF; repeat --pdf for a small test set."
    )
    ap.add_argument("--backend", choices=["api", "local"], default="api",
                    help="Inference backend: OpenAI-compatible API or local Transformers model.")
    ap.add_argument("--local_model", default="greghavens/Qwen3.8-27B-bnb-4bit",
                    help="Hugging Face model ID or local model directory for --backend local.")
    ap.add_argument("--local_max_memory_gib", type=int, default=32,
                    help="Maximum VRAM allocated to model weights; leave headroom on a 36 GB card.")
    ap.add_argument("--local_max_input_tokens", type=int, default=40000,
                    help="Reject prompts exceeding this token count rather than risking OOM.")
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--no_linked_docs", action="store_true")
    ap.add_argument("--no_validator", action="store_true", help="Skip Agent 2 validation (debug only).")
    ap.add_argument("--validator_max_new_tokens", type=int, default=0, help="Validator output tokens; 0 uses --max_new_tokens.")
    ap.add_argument("--validator_retries", type=int, default=0, help="Legacy option; Agent 2 now makes one call without model retries.")
    ap.add_argument("--validator_evidence_chars", type=int, default=70000, help="Max targeted evidence chars per Agent-2 batch.")
    ap.add_argument("--validator_evidence_window", type=int, default=900, help="Characters around each evidence hit.")
    ap.add_argument("--validator_batch_size", type=int, default=100000, help="Legacy option; Agent 2 now validates all routed constraints in one call.")
    ap.add_argument("--validator_min_batch_size", type=int, default=100000, help="Legacy option; Agent 2 now validates all routed constraints in one call.")
    ap.add_argument("--validator_risk_threshold", type=int, default=3, help="Agent-2 risk score threshold; constraints below it are auto-kept.")
    ap.add_argument("--force", action="store_true", help="Reprocess bids even if a an output JSON already exists.")
    ap.add_argument(
        "--base_url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI-compatible API base URL. For Qwen on OpenRouter use https://openrouter.ai/api/v1",
    )
    ap.add_argument(
        "--api_key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key. Prefer OPENAI_API_KEY for OpenRouter or OPENAI_API_KEY for OpenAI.",
    )
    ap.add_argument(
        "--api_key_env",
        default=None,
        help="Environment variable containing the API key. Overrides automatic provider detection.",
    )
    ap.add_argument("--model", default="qwen/qwen3.8-27b:free")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min_request_interval", type=float, default=8.0,
                    help="Minimum seconds between API requests. Use 8+ for rate-limited/free endpoints.")
    ap.add_argument("--max_api_requests", type=int, default=45,
                help="Hard cap on HTTP API attempts per UTC day for this output directory. Default 45 leaves headroom under a 50/day quota.")
    ap.add_argument("--api_budget_file", type=Path, default=None,
                help="Optional persistent API budget state file. Defaults to <output_dir>/api_usage.json.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--max_new_tokens", type=int, default=12288,
        help="Maximum Qwen output tokens for each complete retrace."
    )
    ap.add_argument(
        "--llm_context_chars", type=int, default=120000,
        help="Maximum chars per Agent-1 extraction call. If the complete bid exceeds this, it is split into document-aware overlapping chunks; no document text is discarded."
    )
    ap.add_argument(
        "--chunk_overlap_chars", type=int, default=5000,
        help="Overlap between adjacent chunks of an oversized document. Helps preserve tables/specifications crossing chunk boundaries."
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--no_think_mode",
        choices=["openrouter", "chat_template", "none"],
        default="openrouter",
    )
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
