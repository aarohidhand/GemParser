#!/usr/bin/env python3
"""
GeM bid constraint extractor + adversarial constraint validator.

Qwen performs the semantic constraint extraction useful for supplier/product matching.
The final_constraints.csv output is a canonicalized, IndiaMART-facing semantic set.
Deterministic PDF/table parsing is used only as source-document ingestion/evidence; it is
not allowed to become the final semantic constraint representation. Qwen is given the largest
combined document context that fits safely in its context window, rather than many tiny calls.
Use --no_llm to run with zero API calls (deterministic evidence only).

    pip install -r requirements.txt
    # First run downloads Qwen3.8-27B from Hugging Face and caches it locally.
    python extract_constraints_hybrid.py --input_dir ./bid_pdfs --output_dir ./out --concurrency 1
    python extract_constraints_hybrid.py --input_dir ./bid_pdfs --output_dir ./out --limit 1 --concurrency 1

Resumable: bids with an existing schema-v19 JSON in output_dir are skipped. Linked documents are stored under <output>/<BID_ID>/linked_documents/.
"""
import argparse, asyncio, csv, json, os, re, sys, traceback, io, tempfile, hashlib, time
from urllib.parse import urljoin, urlparse
import requests
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Local Transformers inference for Qwen3.8-27B.
# The model is loaded once in the main process and reused for Agent 1 + Agent 2.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            txt = page.extract_text() or ""
            if len(txt.strip()) < 20:
                try:
                    import pytesseract
                    txt = pytesseract.image_to_string(page.to_image(resolution=300).original)
                    out["ocr_pages"] += 1
                except Exception:
                    txt = ""
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


def parse_linked_document(path: Path, source_document: str, linked_from: str, context: str | None):
    """Parse common linked GeM document formats.

    PDF uses the full GeM table parser. Office/spreadsheet formats are preserved as
    text and simple key/value tables so they can supplement the main bid without
    replacing buyer_value.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        d = parse_pdf(str(path), discover_links=False, source_document=source_document)
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
LLM_COMPACT_RETRY_PROMPT = """Reconstruct the complete buyer requirement representation
from the CURRENT DOCUMENT SET below.

This is a retry because the previous response was not valid JSON.

Rules:
- output ONE JSON object only
- use compact JSON
- no markdown
- no <think>
- no explanation
- preserve every substantive requirement
- preserve correlations between specifications belonging to the same physical item
- NEVER flatten separate physical/configuration instances into allowed_values
- raw_text <= 240 characters
- do not invent values

Top-level schema:
{
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

CURRENT DOCUMENT SET:
{source_bundle}
"""

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
#   2. Give Qwen the COMPLETE current document set for a bid.
#   3. After each newly discovered linked document, RE-TRACE the entire current
#      document set with Qwen. The model must reconstruct the complete
#      requirement representation, not patch/append isolated constraints.
#   4. The final pass over all successfully parsed documents is the sole source
#      of final semantic constraints.
#   5. Python performs only mechanical validation/deduplication/ID assignment.
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
- NEEDS_REVIEW: evidence is ambiguous/conflicting.

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
- If the relevant evidence is unavailable, ambiguous, or retrieval failed, return NEEDS_REVIEW.
- Never manufacture quotations.
- Use only the supplied evidence snippets.
- Do not add new constraints.

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
  "validation_status": "ok|needs_review|failed",
  "summary": {"input_constraints": 0, "keep": 0, "correct": 0, "reject": 0, "needs_review": 0},
  "validated_constraints": [
    {
      "constraint_id": "exact input ID",
      "decision": "KEEP|CORRECT|REJECT|NEEDS_REVIEW",
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

LLM_VALIDATOR_REPAIR_PROMPT = """Repair the following validator output into ONE valid JSON object.
Do NOT add, remove, or semantically alter validation decisions. Preserve every constraint_id and
KEEP/CORRECT/REJECT/NEEDS_REVIEW decision. Do not invent evidence or source text. Output JSON only.
Required fields: schema_version, validation_agent, validation_status, summary, validated_constraints.
Each decision requires constraint_id, decision, original_constraint, validated_constraint, evidence, reason, validation_confidence.

MODEL OUTPUT:
{text}
"""

LLM_REPAIR_PROMPT = """Repair the following model output into ONE valid JSON object for the
requirement-group procurement extraction schema.

Do NOT add facts or change values. Preserve requirement groups and their relationships.
Do NOT flatten separate physical/configuration instances into allowed_values.
The top-level object must contain product_name, product_category, indiamart_search_query,
requirement_groups, contacts, office_name, golden_candidates, and linked_document_mentions.

Each requirement group must contain:
requirement_group_id, requirement_group_type, requirement_description, line_item_id, constraints.

Each constraint must contain:
category, scope, constraint_type, attribute, spec_name, operator, buyer_value,
allowed_values, allows_higher, unit, condition, status, golden_param,
source_document, source_document_id, raw_text, confidence.

Output JSON only.

MODEL OUTPUT:
{text}
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
                   rate_limiter=None, api_budget=None, retries=1):
    """
    Local Qwen3.8-27B inference through Transformers.

    The old implementation used AsyncOpenAI/OpenRouter.  This version ignores
    the client/HTTP arguments and calls a locally loaded Hugging Face model.

    Qwen3.8-27B is quantized to 4-bit NF4 so it fits on a 36-GB GPU.
    Inference is serialized because one model instance is shared on the GPU.
    """
    if client is None:
        raise RuntimeError("Local Qwen model is not loaded.")

    # Serialize generation. Multiple simultaneous generate() calls on one
    # 36-GB GPU are not useful here and can cause OOMs.
    async with sem:
        def _generate():
            tokenizer = client["tokenizer"]
            model_obj = client["model"]

            messages = [{"role": "user", "content": prompt}]

            # Qwen3.8 supports enable_thinking=False through its chat template.
            # This is important here because the pipeline expects JSON rather
            # than a long reasoning trace.
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
            except TypeError:
                # Compatibility fallback for tokenizer versions that do not
                # expose enable_thinking as a direct keyword.
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                input_ids = tokenizer(text, return_tensors="pt").input_ids

            device = next(model_obj.parameters()).device
            input_ids = input_ids.to(device)

            with torch.inference_mode():
                output_ids = model_obj.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            generated = output_ids[0, input_ids.shape[-1]:]
            return tokenizer.decode(generated, skip_special_tokens=True)

        last = None
        for attempt in range(retries):
            try:
                return await asyncio.to_thread(_generate)
            except RuntimeError as e:
                last = e
                # Free temporary activations/cache before a retry.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt + 1 < retries:
                    await asyncio.sleep(2)
                else:
                    raise
        raise last



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
    client,
    model,
    sem,
    raw,
    max_tokens,
    think_mode,
    rate_limiter=None,
    api_budget=None,
):
    """Parse JSON with local syntax repair only.

    We intentionally do not spend another API request trying to repair malformed
    JSON. If the response is malformed, the caller performs one fresh semantic
    reconstruction over the COMPLETE document set. This avoids wasting the free
    endpoint quota on a second prompt that contains the model's own broken output.
    """
    return parse_llm_json(raw)


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
    """
    A retrace must see the whole current document set. If the set exceeds the
    configured context budget, fail rather than silently splitting it into
    independent semantic calls. Splitting would recreate the representation bug.
    """
    bundle = "\n\n".join(sections)
    if len(bundle) > max_chars:
        raise ValueError(
            f"current bid document set is {len(bundle):,} chars, above "
            f"--llm_context_chars={max_chars:,}; refusing to split semantic context"
        )
    return bundle


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
    d=str(item.get("decision") or "NEEDS_REVIEW").upper().strip()
    return d if d in {"KEEP","CORRECT","REJECT","NEEDS_REVIEW"} else "NEEDS_REVIEW"


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
        if d not in {"KEEP", "CORRECT", "REJECT", "NEEDS_REVIEW"}:
            errors.append(f"{cid}: invalid decision")
    return errors


def _validator_failure_audit(inputs: list, reason: str) -> list:
    return [{
        "constraint_id": str(c.get("constraint_id")),
        "decision": "NEEDS_REVIEW",
        "original_constraint": c,
        "validated_constraint": None,
        "evidence": [],
        "reason": reason,
        "validation_confidence": 0.0,
    } for c in inputs]


def _evidence_terms(c: dict) -> list[str]:
    terms=[]
    for k in ("raw_text","spec_name","attribute","buyer_value"):
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
                        "raw_text":text[pos:pos+len(raw)]}
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
        anchor=c.get("evidence_anchor") or {}
        if anchor.get("document_id"):
            doc=next((d for d in documents if d.get("document_id")==anchor.get("document_id")),None)
            if doc:
                text=doc.get("text") or ""
                a=int(anchor.get("context_start", max(0,int(anchor.get("start",0))-window)))
                b=int(anchor.get("context_end", min(len(text),int(anchor.get("end",0))+window)))
                a=max(0,a); b=min(len(text),b)
                exact=text[int(anchor.get("start",0)):int(anchor.get("end",0))]
                chunks.append(
                    f"===== CONSTRAINT_ID: {cid} =====\nDOCUMENT_ID: {doc['document_id']}\nSOURCE_DOCUMENT: {doc['source_document']}\nEXACT_SOURCE_EXCERPT:\n{exact}\nCONTEXT_AROUND_EXCERPT:\n{text[a:b].strip()}\n"
                )
                found=1
        for doc in documents:
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
    obj=parse_llm_json(raw)
    errors=validate_validator_object(obj,inputs)
    if errors:
        raise ValueError("validator structural validation failed: "+" | ".join(errors))
    return obj


async def validate_in_batches(client, model, sem, documents, inputs, batch_size, min_batch_size,
                              max_tokens, think_mode, rate_limiter=None, api_budget=None, retries=5,
                              evidence_chars=70000, evidence_window=900):
    if not inputs:
        return {"schema_version":21,"validation_agent":"targeted_adversarial_validator","validation_status":"ok",
                "summary":{"input_constraints":0,"keep":0,"correct":0,"reject":0,"needs_review":0},
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
        except Exception as first_error:
            if len(items)>min_batch_size:
                mid=len(items)//2
                batch_records.append({"batch":label,"input_count":len(items),"status":"split","reason":repr(first_error)})
                await validate_slice(items[:mid],label+"a")
                await validate_slice(items[mid:],label+"b")
                return
            had_failure=True
            reason=f"validator batch failed at minimum batch size {len(items)}: {first_error!r}"
            batch_records.append({"batch":label,"input_count":len(items),"status":"failed","reason":reason})
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
    counts={"input_constraints":len(inputs),"keep":0,"correct":0,"reject":0,"needs_review":0}
    for x in ordered:
        d=normalize_validation_decision(x); counts[{"KEEP":"keep","CORRECT":"correct","REJECT":"reject","NEEDS_REVIEW":"needs_review"}[d]]+=1
    status="needs_review" if had_failure or counts["needs_review"] else "ok"
    obj={"schema_version":21,"validation_agent":"targeted_adversarial_validator","validation_status":status,
         "summary":counts,"validated_constraints":ordered,"batches":batch_records}
    return obj,ordered


def apply_validation(inputs, validation_obj):
    by_id={str(x.get("constraint_id")):x for x in validation_obj.get("validated_constraints",[]) if isinstance(x,dict)}
    final=[]; audit=[]
    for original in inputs:
        cid=str(original.get("constraint_id")); item=by_id.get(cid)
        if not item:
            item={"constraint_id":cid,"decision":"NEEDS_REVIEW","original_constraint":original,"validated_constraint":None,"evidence":[],"reason":"Validator omitted decision.","validation_confidence":0.0}
        d=normalize_validation_decision(item)
        item=dict(item)
        item["decision"]=d
        evidence=item.get("evidence") or []
        has_evidence=any(isinstance(e,dict) and clean(str(e.get("raw_text") or "")) and clean(str(e.get("raw_text") or "")) != "NO TARGETED EVIDENCE FOUND" for e in evidence)
        # v21: missing retrieval evidence cannot be treated as a contradiction.
        if d in {"KEEP", "CORRECT", "REJECT"} and not has_evidence:
            d="NEEDS_REVIEW"
            item["decision"]=d
            item["reason"]=(str(item.get("reason") or "").strip() + " Evidence was unavailable; routed to NEEDS_REVIEW rather than treating retrieval failure as contradiction.").strip()
        audit.append(item)
        if d=="KEEP": final.append(dict(original))
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
    """v21 deterministic semantic scope guard. Document placement never decides scope."""
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
    """v20: ingest all documents once -> ONE full-bundle Agent-1 extraction -> targeted Agent-2 validation."""
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
                if old.get("schema_version")==21 and old.get("architecture")=="full_document_extraction_plus_exact_anchored_targeted_adversarial_validation_v21" and "final_constraints" in old:
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
        documents=[{"document_id":"MAIN_BID","source_kind":"main_bid","source_document":parsed.get("source_document") or pdf.name,"text":parsed.get("full_text","") or ""}]
        for i,d in enumerate(linked_docs,1):
            documents.append({"document_id":f"DOC_{i:02d}","source_kind":"linked_document","source_document":d.get("source_document") or f"linked_document_{i}","text":d.get("full_text","") or "","linked_from":d.get("linked_from"),"link_context":d.get("link_context"),"local_path":d.get("local_path")})
        if not client: raise RuntimeError("Local Qwen model is required for final semantic constraints. Use --no_llm only for ingestion/testing.")
        sections=build_document_sections(parsed,linked_docs)
        bundle=build_retrace_bundle(sections,args.llm_context_chars)
        print(f"  [AGENT1] {bid_id} | {len(documents)} docs | {len(bundle):,} chars (~{max(1,len(bundle)//4):,} input tokens)",file=sys.stderr)
        prompt=render_prompt(LLM_RETRACE_PROMPT,source_bundle=bundle)
        try:
            raw=await call_llm(client,args.model,sem,prompt,args.max_new_tokens,args.no_think_mode,rate_limiter=rate_limiter,api_budget=api_budget)
            try: obj=parse_llm_json(raw)
            except Exception as e1:
                print(f"  [JSON retry] {bid_id}: {e1}",file=sys.stderr)
                raw2=await call_llm(client,args.model,sem,render_prompt(LLM_COMPACT_RETRY_PROMPT,source_bundle=bundle),args.max_new_tokens,args.no_think_mode,rate_limiter=rate_limiter,api_budget=api_budget)
                obj=parse_llm_json(raw2)
        except Exception as e:
            raise RuntimeError(f"Agent-1 full-bundle extraction failed: {e!r}") from e
        raw_constraints=flatten_qwen_groups(obj)
        for c in raw_constraints:
            gid=c.get("requirement_group_id") or "UNSPECIFIED_GROUP"
            c["requirement_group_id"]=gid
            c["source_document_id"]=c.get("source_document_id") or "MAIN_BID"
            if not any(c["source_document_id"]==d["document_id"] for d in documents): c["source_document_id"]="MAIN_BID"
            if not c.get("source_document"): c["source_document"]=next((d["source_document"] for d in documents if d["document_id"]==c["source_document_id"]),parsed.get("source_document"))
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
            print(f"  [AGENT2] {bid_id} | {len(risky_constraints)}/{len(extraction_constraints)} high-risk constraints | targeted evidence | batch={args.validator_batch_size}",file=sys.stderr)
            validation_obj,agent2_audit=await validate_in_batches(client,args.model,sem,documents,risky_constraints,
                batch_size=args.validator_batch_size,min_batch_size=args.validator_min_batch_size,
                max_tokens=args.validator_max_new_tokens or args.max_new_tokens,think_mode=args.no_think_mode,
                rate_limiter=rate_limiter,api_budget=api_budget,retries=args.validator_retries,
                evidence_chars=args.validator_evidence_chars,evidence_window=args.validator_evidence_window)
            validated_high,_=apply_validation(risky_constraints,validation_obj)
            validated_by_id={str(c.get("constraint_id")):c for c in validated_high}
            # Safe constraints bypass Agent 2; high-risk constraints are replaced only
            # when Agent 2 KEEP/CORRECTs them. NEEDS_REVIEW constraints are excluded.
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
            if any(normalize_validation_decision(x)=="NEEDS_REVIEW" for x in validation_audit): validation_status="needs_review"
        final_constraints=canonicalize_final_constraints(final_constraints)
        final_constraints=apply_scope_ontology_guard(final_constraints)
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
        status="needs_review" if (all_flags or failed_documents or validation_status in ("failed","needs_review")) else "ok"
        (bid_folder/"extraction.json").write_text(json.dumps({"schema_version":21,"agent":"qwen_constraint_extractor","bid_id":bid_id,"constraints":extraction_constraints},indent=2,ensure_ascii=False),encoding="utf-8")
        (bid_folder/"validation.json").write_text(json.dumps(validation_obj or {"schema_version":21,"validation_agent":"targeted_adversarial_validator","validation_status":validation_status,"summary":{},"validated_constraints":validation_audit},indent=2,ensure_ascii=False),encoding="utf-8")
        (bid_folder/"document_extractions.json").write_text(json.dumps({"schema_version":21,"bid_id":bid_id,"documents":[{"document_id":d["document_id"],"source_document":d["source_document"],"chars":len(d.get("text","") or "")} for d in documents],"agent1_bundle_chars":len(bundle),"agent1_approx_input_tokens":max(1,len(bundle)//4)},indent=2,ensure_ascii=False),encoding="utf-8")
        result={"schema_version":21,"architecture":"full_document_extraction_plus_exact_anchored_targeted_adversarial_validation_v21","bid_id":bid_id,"source_pdf":pdf.name,"source_pdf_path":str(pdf),"header":parsed.get("header",{}),"product_category":product_category,"product_name":product_name,"indiamart_search_query":search_query,"office_name":office_name,"contacts":contacts,
                "linked_spec_documents":[{"document_id":f"DOC_{i:02d}","source_document":d.get("source_document"),"local_path":d.get("local_path")} for i,d in enumerate(linked_docs,1)],
                "n_linked_spec_documents":len(linked_docs),"n_documents_extracted":len(documents),"agent1_bundle_chars":len(bundle),"agent1_approx_input_tokens":max(1,len(bundle)//4),"n_final_constraints":len(final_constraints),"representation_flags":all_flags,"validation_status":validation_status,"validation_summary":(validation_obj or {}).get("summary",{}),"risk_summary":{"total":len(extraction_constraints),"agent2":len(risky_constraints),"auto_keep":len(safe_constraints)},"risk_audit":risk_audit,"final_constraints":final_constraints,"linked_manifest":linked_manifest,"failed_documents":failed_documents}
        out_path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        return bid_id,status,None
    except Exception as e:
        return bid_id,"failed",{"bid_id":bid_id,"error":repr(e),"traceback":traceback.format_exc()}

def write_csvs(out_dir: Path):
    final_rows, bids = [], []

    for p in sorted(out_dir.glob("*.json")):
        if p.name == "failures.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
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

    if bids:
        fields = list(bids[0])
        with open(out_dir / "bids.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(bids)

    return len(final_rows)


async def amain(args):
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

    if not args.no_llm:
        print(
            f"[LLM] loading local Transformers model={args.model} "
            f"quantization=4bit device=cuda",
            file=sys.stderr,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not detected. This local Qwen3.8-27B configuration "
                "requires an NVIDIA CUDA GPU."
            )

        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU] {gpu_name} | VRAM={total_gb:.1f} GB", file=sys.stderr)

        if total_gb < 34:
            print(
                "[WARN] This configuration targets a 36-GB class GPU. "
                "Your available VRAM is below ~34 GB; reduce context/output "
                "limits if you hit CUDA OOM.",
                file=sys.stderr,
            )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=True,
        )

        model_obj = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model_obj.eval()

        client = {
            "model": model_obj,
            "tokenizer": tokenizer,
        }

        print(
            f"[LLM] loaded {args.model} in 4-bit; "
            f"allocated={torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB",
            file=sys.stderr,
        )


    sem = asyncio.Semaphore(args.concurrency)
    rate_limiter = None
    api_budget = None
    counts = {"ok": 0, "needs_review": 0, "skipped": 0, "failed": 0}
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
        description="GeM bid semantic extractor v21: one full-bundle extraction + targeted adversarial validation"
    )
    ap.add_argument("--input_dir", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--pdf", action="append", default=[],
        help="Process an exact PDF; repeat --pdf for a small test set."
    )
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--no_linked_docs", action="store_true")
    ap.add_argument("--no_validator", action="store_true", help="Skip Agent 2 validation (debug only).")
    ap.add_argument("--validator_max_new_tokens", type=int, default=0, help="Validator output tokens; 0 uses --max_new_tokens.")
    ap.add_argument("--validator_retries", type=int, default=5, help="API retries per validator call.")
    ap.add_argument("--validator_evidence_chars", type=int, default=70000, help="Max targeted evidence chars per Agent-2 batch.")
    ap.add_argument("--validator_evidence_window", type=int, default=900, help="Characters around each evidence hit.")
    ap.add_argument("--validator_batch_size", type=int, default=20, help="Agent-2 constraints per targeted-evidence validation call.")
    ap.add_argument("--validator_min_batch_size", type=int, default=5, help="Minimum batch size before a validator failure becomes NEEDS_REVIEW.")
    ap.add_argument("--validator_risk_threshold", type=int, default=3, help="Agent-2 risk score threshold; constraints below it are auto-kept.")
    ap.add_argument("--force", action="store_true", help="Reprocess bids even if a valid v19 JSON already exists.")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--concurrency", type=int, default=1, help="Keep at 1 for a single 36-GB GPU.")
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
        "--llm_context_chars", type=int, default=750000,
        help="Maximum chars for the COMPLETE current bid document bundle. Agent 1 sees the whole bundle once; it is never semantically split."
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--no_think_mode",
        choices=["openrouter", "chat_template", "none"],
        default="chat_template",
        help="Local Transformers mode uses the Qwen chat template with thinking disabled.",
    )
    asyncio.run(amain(ap.parse_args()))

if __name__ == "__main__":
    main()