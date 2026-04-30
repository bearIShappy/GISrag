# import os
# import re
# import json
# from pathlib import Path
# from dotenv import load_dotenv
# from docling.document_converter import DocumentConverter
# from gliner2 import GLiNER2

# # ─────────────────────────────────────────────
# # 1. ENVIRONMENT & PATHS
# # ─────────────────────────────────────────────

# load_dotenv()

# INPUT_DIR  = Path(os.getenv("INPUT_FOLDER",            "./data/input"))
# OUTPUT_DIR = Path(os.getenv("EXTRACTED_OUTPUT_FOLDER", "./data/output"))
# MODEL_ID   = os.getenv("PARSING_MODEL_NAME", "fastino/gliner2-large-v1")
# DEVICE     = os.getenv("DEVICE", "cpu")   # Use "cuda" for RTX 3060

# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # ─────────────────────────────────────────────
# # 2. MODEL INITIALISATION
# # ─────────────────────────────────────────────

# print("--- Initializing Extraction Pipeline ---")
# print(f"Loading {MODEL_ID} on {DEVICE}...")

# converter = DocumentConverter()
# extractor = GLiNER2.from_pretrained(MODEL_ID)
# if DEVICE == "cuda":
#     extractor.to("cuda")

# # ─────────────────────────────────────────────
# # 3. EXTRACTION SCHEMA
# # ─────────────────────────────────────────────

# extraction_schema = {
#     "document_info": [
#         "date::str::The date mentioned in the document, journey date, or report date",
#         "time::str::Specific time or military time format (e.g. 1400 hrs)",
#         "place::str::The specific location, city, site, or waypoint name",
#         "latitude::str::A decimal latitude coordinate such as 40.7128 N or -74.0060",
#         "longitude::str::A decimal longitude coordinate such as 40.7128 N or -74.0060",
#         "role::str::Designation of persons mentioned (e.g. Lead Surveyor, Coordinator)",
#         "summary::str::A brief executive summary of the activity or event described",
#     ]
# }

# MIN_CHUNK_LENGTH = 80


# # ─────────────────────────────────────────────
# # 4. CHUNKING  —  three-tier strategy
# #
# # Tier 1  chunk_by_entry()   splits on date-header patterns
# #         → keeps each observation in its own context window so
# #           GLiNER2 cannot bleed entities across records.
# #           This is the root fix for coord/place mismatches seen
# #           in the India expedition and env-log PDFs.
# #
# # Tier 2  get_page_chunks()  falls back to Docling page boundaries
# #         → good for PDFs where entries span full pages.
# #
# # Tier 3  get_markdown_chunks() / whole-doc fallback
# #         → last resort when neither pattern fires.
# # ─────────────────────────────────────────────

# # Matches structured date headers in common formats:
# #   "Date: March 12, 2024"
# #   "[April 11, 2026 | 10:30 AM]"
# #   "Inspection Date: April 12, 2026"
# #   "Recorded on: October 05, 2024"
# #   "Date: April 2026"          ← month-year only
# DATE_HEADER = re.compile(
#     r'(?:'
#     r'(?:Date|Inspection Date|Recorded on)\s*:\s*'          # labelled date
#     r'|'
#     r'\['                                                    # bracket-prefixed log entry
#     r')'
#     r'(?:\w+\s+\d{1,2},?\s+\d{4}'                          # "April 11, 2026"
#     r'|\w+\s+\d{4}'                                         # "April 2026"
#     r'|\d{4}-\d{2}-\d{2}'                                   # ISO date
#     r')',
#     re.IGNORECASE
# )


# def chunk_by_entry(full_text: str) -> list[str]:
#     """
#     Tier-1 chunker: split text on date-header boundaries.

#     Each chunk spans from one date header to the next (exclusive),
#     so all coordinates and place names extracted from a chunk
#     belong to the same observation event.

#     Falls back to [full_text] when no headers are found so
#     subsequent tiers can take over.
#     """
#     splits = list(DATE_HEADER.finditer(full_text))
#     if not splits:
#         return []   # signal: no date headers found, try next tier

#     chunks: list[str] = []
#     for i, match in enumerate(splits):
#         start = match.start()
#         end = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
#         chunk = full_text[start:end].strip()
#         if len(chunk) >= MIN_CHUNK_LENGTH:
#             chunks.append(chunk)

#     # ── Orphan-coordinate merge ───────────────────────────────────────────
#     # After splitting, a chunk may contain Coordinates: but no Place: line.
#     # This happens when a page break fell between the Place and Coordinates
#     # fields and the form-feed was not fully normalised.
#     # Solution: merge such an "orphan" chunk back onto the previous chunk so
#     # GLiNER2 sees both the place name and the coordinates in one context window.
#     merged: list[str] = []
#     for chunk in chunks:
#         has_coords = bool(re.search(r'Coordinates\s*:', chunk, re.IGNORECASE))
#         has_place  = bool(re.search(r'(?:^|\n)Place\s*:', chunk, re.IGNORECASE))
#         if has_coords and not has_place and merged:
#             # Stitch onto the previous chunk (a blank line separates them visually)
#             merged[-1] = merged[-1] + "\n\n" + chunk
#         else:
#             merged.append(chunk)
#     return merged


# def get_page_chunks(docling_result) -> list[str]:
#     """
#     Tier-2 chunker: one chunk per physical PDF page via Docling.
#     Used when date-header splitting yields nothing (e.g. single-
#     page PDFs or PDFs without labelled date lines).
#     """
#     chunks: list[str] = []
#     try:
#         for page in docling_result.document.pages.values():
#             page_texts = []
#             for item, _ in docling_result.document.iterate_items(page_no=page.page_no):
#                 text = getattr(item, "text", None)
#                 if text:
#                     page_texts.append(text.strip())
#             page_chunk = "\n".join(page_texts).strip()
#             if len(page_chunk) >= MIN_CHUNK_LENGTH:
#                 chunks.append(page_chunk)
#     except Exception:
#         pass
#     return chunks


# def get_markdown_chunks(markdown_text: str) -> list[str]:
#     """
#     Tier-3 chunker: split exported markdown on double newlines,
#     merging short paragraphs until each chunk exceeds MIN_CHUNK_LENGTH.
#     """
#     raw_chunks = [c.strip() for c in markdown_text.split("\n\n") if c.strip()]
#     merged: list[str] = []
#     buffer = ""
#     for chunk in raw_chunks:
#         buffer = (buffer + "\n\n" + chunk).strip() if buffer else chunk
#         if len(buffer) >= MIN_CHUNK_LENGTH:
#             merged.append(buffer)
#             buffer = ""
#     if buffer:
#         merged.append(buffer)
#     return [c for c in merged if len(c) >= MIN_CHUNK_LENGTH]


# def chunk_document(docling_result, markdown_text: str) -> list[str]:
#     """
#     Orchestrate the three-tier chunking strategy.

#     Priority:
#       1. Date-header entry splitting  (most accurate, entry-scoped)
#       2. Page-level splitting         (good for page-per-entry PDFs)
#       3. Paragraph merging            (generic fallback)
#       4. Whole document               (last resort)
#     """
#     # Tier 1 — entry-level (preferred)
#     chunks = chunk_by_entry(markdown_text)
#     if chunks:
#         return chunks

#     # Tier 2 — page-level
#     chunks = get_page_chunks(docling_result)
#     if len(chunks) >= 2:
#         return chunks

#     # Tier 3 — paragraph merge
#     chunks = get_markdown_chunks(markdown_text)
#     if chunks:
#         return chunks

#     # Tier 4 — whole document
#     return [markdown_text.strip()]


# # ─────────────────────────────────────────────
# # 5. DEDUPLICATION
# # ─────────────────────────────────────────────

# def dedup_records(records: list[dict]) -> list[dict]:
#     """
#     Remove duplicate records produced by overlapping chunks.
#     Composite key: (date, place, latitude).
#     Records where all three are null/empty are also dropped.
#     """
#     seen: set[tuple] = set()
#     unique: list[dict] = []
#     for rec in records:
#         key = (
#             str(rec.get("date")      or "").strip().lower(),
#             str(rec.get("place")     or "").strip().lower(),
#             str(rec.get("latitude")  or "").strip().lower(),
#         )
#         if all(k in ("", "none") for k in key):
#             continue
#         if key not in seen:
#             seen.add(key)
#             unique.append(rec)
#     return unique


# # ─────────────────────────────────────────────
# # 6. MAIN PROCESSING LOOP
# # ─────────────────────────────────────────────

# def process_documents():
#     valid_extensions = ('.pdf', '.docx', '.xlsx', '.csv')
#     files = [f for f in INPUT_DIR.glob("*.*") if f.suffix.lower() in valid_extensions]

#     if not files:
#         print(f"No files found in {INPUT_DIR}. Please add some files.")
#         return

#     for file_path in files:
#         print(f"\n[Processing] {file_path.name}")

#         try:
#             # STEP 1 — Layout-aware parsing via Docling
#             result        = converter.convert(file_path)
#             markdown_text = result.document.export_to_markdown()

#             # STEP 2 — Chunk (entry-level preferred, page/para fallback)
#             # Strip PDF page-break form-feeds (\x0c) so entries that span a
#             # physical page boundary (e.g. Place: on p.1, Coordinates: on p.2)
#             # are kept in a single chunk and GLiNER2 sees both fields together.
#             markdown_text = markdown_text.replace("\x0c", "\n")
#             chunks = chunk_document(result, markdown_text)
#             print(f"  -> Split into {len(chunks)} chunk(s)")

#             # STEP 3 — Extract per chunk with GLiNER2
#             all_records: list[dict] = []
#             for i, chunk in enumerate(chunks):
#                 print(f"  -> Extracting chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
#                 try:
#                     extracted = extractor.extract_json(chunk, extraction_schema)
#                     records   = extracted.get("document_info", [])
#                     if isinstance(records, dict):
#                         records = [records]
#                     all_records.extend(records)
#                 except Exception as e:
#                     print(f"     [WARN] Chunk {i+1} extraction failed: {e}")

#             # STEP 4 — Deduplicate
#             before        = len(all_records)
#             all_records   = dedup_records(all_records)
#             after         = len(all_records)
#             if before != after:
#                 print(f"  -> Deduped: {before} → {after} records")

#             print(f"  -> Total records extracted: {after}")

#             # STEP 5 — Save (format unchanged so cleaner.py reads it as-is)
#             output_file = OUTPUT_DIR / f"{file_path.stem}_data.json"
#             with open(output_file, "w", encoding="utf-8") as f:
#                 json.dump({"document_info": all_records}, f, indent=4, ensure_ascii=False)

#             print(f"--- Successfully extracted to: {output_file.name} ---")

#         except Exception as e:
#             print(f"!!! Error processing {file_path.name}: {e}")


# if __name__ == "__main__":
#     process_documents()
#     print("\n--- Pipeline Execution Complete ---")
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from gliner2 import GLiNER2

# ─────────────────────────────────────────────
# 1. ENVIRONMENT & PATHS
# ─────────────────────────────────────────────

load_dotenv()

INPUT_DIR  = Path(os.getenv("INPUT_FOLDER",            "./data/input"))
OUTPUT_DIR = Path(os.getenv("EXTRACTED_OUTPUT_FOLDER", "./data/output"))
MODEL_ID   = os.getenv("PARSING_MODEL_NAME", "fastino/gliner2-large-v1")
DEVICE     = os.getenv("DEVICE", "cpu")   # Use "cuda" for RTX 3060

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 2. MODEL INITIALISATION
# ─────────────────────────────────────────────

print("--- Initializing Extraction Pipeline ---")
print(f"Loading {MODEL_ID} on {DEVICE}...")

converter = DocumentConverter()
extractor = GLiNER2.from_pretrained(MODEL_ID)
if DEVICE == "cuda":
    extractor.to("cuda")

# ─────────────────────────────────────────────
# 3. EXTRACTION SCHEMA
# ─────────────────────────────────────────────

extraction_schema = {
    "document_info": [
        "date::str::The date mentioned in the document, journey date, or report date",
        "time::str::Specific time or military time format (e.g. 1400 hrs)",
        "place::str::The specific location, city, site, or waypoint name",
        "latitude::str::A decimal latitude coordinate WITH its hemisphere letter attached "
                        "(e.g. '25.3176 N', '7.5407 S'). Always include N or S.",
        "longitude::str::A decimal longitude coordinate WITH its hemisphere letter attached "
                         "(e.g. '82.9739 E', '74.0060 W'). Always include E or W.",
        "role::str::Designation of persons mentioned (e.g. Lead Surveyor, Coordinator)",
        "summary::str::A brief executive summary of the activity or event described",
    ]
}

MIN_CHUNK_LENGTH = 80

# ─────────────────────────────────────────────
# 4. EXCEL ROW-LABELLING  (new — Issue #1)
#
# Excel files lose hemisphere context when Docling flattens them to
# markdown: a cell containing "51.9225" carries no sign information
# because the column header "Latitude (N)" is stripped.
#
# Solution: convert each data row to a key: value block so GLiNER2
# sees "Latitude (N): 51.9225" and can emit "51.9225 N" reliably.
# ─────────────────────────────────────────────

EXCEL_EXTENSIONS = ('.xlsx', '.xls', '.xlsm')

# Maps common Excel column-header patterns to coordinate hints that
# GLiNER2 can pick up in free text.
_LAT_HINT = re.compile(r'lat', re.IGNORECASE)
_LON_HINT = re.compile(r'lo[n|g]', re.IGNORECASE)
_NORTH_HINT = re.compile(r'\bN\b|north', re.IGNORECASE)
_SOUTH_HINT = re.compile(r'\bS\b|south', re.IGNORECASE)
_EAST_HINT  = re.compile(r'\bE\b|east',  re.IGNORECASE)
_WEST_HINT  = re.compile(r'\bW\b|west',  re.IGNORECASE)


def _infer_hemisphere(column_name: str, is_lat: bool) -> str:
    """
    Return the hemisphere letter implied by a column header, or ""
    if it cannot be determined.

    Examples:
      "Latitude (N)"  → "N"
      "Lon_W"         → "W"
      "longitude"     → ""   (ambiguous — leave for cleaner to handle)
    """
    if is_lat:
        if _SOUTH_HINT.search(column_name):
            return "S"
        if _NORTH_HINT.search(column_name):
            return "N"
    else:
        if _WEST_HINT.search(column_name):
            return "W"
        if _EAST_HINT.search(column_name):
            return "E"
    return ""


def _append_hemisphere(value: str, hemisphere: str) -> str:
    """
    Append a hemisphere letter to a numeric coordinate string if it is
    not already present and the hemisphere is known.

    "51.9225"   + "N"  →  "51.9225 N"
    "51.9225 N" + "N"  →  "51.9225 N"   (no duplication)
    "-74.006"   + ""   →  "-74.006"     (keep existing sign)
    """
    if not hemisphere:
        return value
    # Already has a direction letter
    if re.search(r'[NSEWnsew]', value):
        return value
    # Has an explicit negative sign — direction is unambiguous; do not append
    if value.strip().startswith('-'):
        return value
    return f"{value.strip()} {hemisphere}"


def excel_to_labeled_chunks(file_path: Path) -> list[str] | None:
    """
    Convert each row of an Excel workbook to a labeled text block,
    injecting hemisphere suffixes into coordinate values so that
    GLiNER2 (and cleaner.py) always see explicit N/S/E/W.

    Returns None if the file is not an Excel format (caller falls
    through to Docling).

    Strategy per column:
      • Detect lat/lon columns by header keyword.
      • Derive hemisphere from header (e.g. "Lat_N", "Longitude (W)").
      • Append hemisphere letter to the cell value.
      • Emit every row as:
            Sheet: <sheet_name>
            <col1>: <val1>
            Latitude: <val> N       ← hemisphere always explicit
            Longitude: <val> E
            ...
    """
    if file_path.suffix.lower() not in EXCEL_EXTENSIONS:
        return None

    try:
        import pandas as pd
    except ImportError:
        print("  [WARN] pandas not installed — Excel processing unavailable.")
        return None

    chunks: list[str] = []

    try:
        xl = pd.ExcelFile(file_path)
    except Exception as e:
        print(f"  [WARN] Could not open Excel file {file_path.name}: {e}")
        return None

    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name).fillna("")
        except Exception as e:
            print(f"  [WARN] Could not parse sheet '{sheet_name}': {e}")
            continue

        # Pre-compute column metadata once per sheet
        col_meta: dict[str, dict] = {}
        for col in df.columns:
            col_str = str(col)
            is_lat = bool(_LAT_HINT.search(col_str))
            is_lon = bool(_LON_HINT.search(col_str) and not is_lat)
            hemisphere = ""
            if is_lat or is_lon:
                hemisphere = _infer_hemisphere(col_str, is_lat=is_lat)
            col_meta[col_str] = {
                "is_lat": is_lat,
                "is_lon": is_lon,
                "hemisphere": hemisphere,
            }

        for _, row in df.iterrows():
            lines = [f"Sheet: {sheet_name}"]
            for col in df.columns:
                raw_val = str(row[col]).strip()
                if not raw_val or raw_val in ("nan", "None", ""):
                    continue

                col_str  = str(col)
                meta     = col_meta[col_str]
                label    = col_str

                if meta["is_lat"] or meta["is_lon"]:
                    # Always emit the coordinate with its hemisphere attached
                    display_val = _append_hemisphere(raw_val, meta["hemisphere"])
                    # Use a standardised label so GLiNER2's prompt matches
                    label = "Latitude" if meta["is_lat"] else "Longitude"
                    lines.append(f"{label}: {display_val}")
                else:
                    lines.append(f"{col_str}: {raw_val}")

            chunk = "\n".join(lines).strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)

    return chunks if chunks else None


# ─────────────────────────────────────────────
# 5. CHUNKING  —  three-tier strategy
#
# Tier 1  chunk_by_entry()   splits on date-header patterns
# Tier 2  get_page_chunks()  falls back to Docling page boundaries
# Tier 3  get_markdown_chunks() / whole-doc fallback
# ─────────────────────────────────────────────

DATE_HEADER = re.compile(
    r'(?:'
    r'(?:Date|Inspection Date|Recorded on)\s*:\s*'
    r'|'
    r'\['
    r')'
    r'(?:\w+\s+\d{1,2},?\s+\d{4}'
    r'|\w+\s+\d{4}'
    r'|\d{4}-\d{2}-\d{2}'
    r')',
    re.IGNORECASE
)


def _ensure_hemisphere_in_chunk(chunk: str) -> str:
    """
    Post-process a text chunk to append explicit N/S/E/W hemisphere
    letters to any bare decimal coordinate that follows a Latitude:/
    Longitude: label but has no direction indicator yet.

    This is the last-resort normalisation for PDFs and DOCX files where
    the document author wrote coordinates without hemisphere letters.
    We use the *label context* (Latitude vs Longitude) and the *numeric
    sign* to infer direction:

      Latitude: 34.5822   → Latitude: 34.5822 N   (positive → N)
      Latitude: -7.5407   → Latitude: -7.5407 S   (negative → S)
      Longitude: 82.9739  → Longitude: 82.9739 E  (positive → E)
      Longitude: -74.006  → Longitude: -74.006 W  (negative → W)

    Coordinates that already carry a letter (N/S/E/W) or that sit
    outside a labeled context are left unchanged.
    """
    def _replace_coord(m: re.Match) -> str:
        label  = m.group(1)   # "Latitude" or "Longitude"
        value  = m.group(2)   # e.g. "34.5822" or "-7.5407"
        rest   = m.group(3)   # anything on the same token after the digits

        # Already has a hemisphere letter anywhere in rest
        if re.search(r'[NSEWnsew]', rest):
            return m.group(0)

        try:
            fval = float(value)
        except ValueError:
            return m.group(0)

        is_lat = bool(re.match(r'lat', label, re.IGNORECASE))
        if is_lat:
            hemi = "N" if fval >= 0 else "S"
        else:
            hemi = "E" if fval >= 0 else "W"

        return f"{label}: {value} {hemi}{rest}"

    # Match "Latitude: 34.5822" or "Longitude: -74.006"
    pattern = re.compile(
        r'(Lat(?:itude)?|Lon(?:gitude)?)\s*:\s*(-?\d+(?:\.\d+)?)([^\n]*)',
        re.IGNORECASE
    )
    return pattern.sub(_replace_coord, chunk)


def chunk_by_entry(full_text: str) -> list[str]:
    splits = list(DATE_HEADER.finditer(full_text))
    if not splits:
        return []

    chunks: list[str] = []
    for i, match in enumerate(splits):
        start = match.start()
        end   = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
        chunk = full_text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)

    # Orphan-coordinate merge (page-break between Place: and Coordinates:)
    merged: list[str] = []
    for chunk in chunks:
        has_coords = bool(re.search(r'Coordinates\s*:', chunk, re.IGNORECASE))
        has_place  = bool(re.search(r'(?:^|\n)Place\s*:', chunk, re.IGNORECASE))
        if has_coords and not has_place and merged:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    return merged


def get_page_chunks(docling_result) -> list[str]:
    chunks: list[str] = []
    try:
        for page in docling_result.document.pages.values():
            page_texts = []
            for item, _ in docling_result.document.iterate_items(page_no=page.page_no):
                text = getattr(item, "text", None)
                if text:
                    page_texts.append(text.strip())
            page_chunk = "\n".join(page_texts).strip()
            if len(page_chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(page_chunk)
    except Exception:
        pass
    return chunks


def get_markdown_chunks(markdown_text: str) -> list[str]:
    raw_chunks = [c.strip() for c in markdown_text.split("\n\n") if c.strip()]
    merged: list[str] = []
    buffer = ""
    for chunk in raw_chunks:
        buffer = (buffer + "\n\n" + chunk).strip() if buffer else chunk
        if len(buffer) >= MIN_CHUNK_LENGTH:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return [c for c in merged if len(c) >= MIN_CHUNK_LENGTH]


def chunk_document(docling_result, markdown_text: str) -> list[str]:
    # Tier 1 — entry-level (preferred)
    chunks = chunk_by_entry(markdown_text)
    if chunks:
        return chunks

    # Tier 2 — page-level
    chunks = get_page_chunks(docling_result)
    if len(chunks) >= 2:
        return chunks

    # Tier 3 — paragraph merge
    chunks = get_markdown_chunks(markdown_text)
    if chunks:
        return chunks

    # Tier 4 — whole document
    return [markdown_text.strip()]


# ─────────────────────────────────────────────
# 6. DEDUPLICATION
# ─────────────────────────────────────────────

def dedup_records(records: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for rec in records:
        key = (
            str(rec.get("date")      or "").strip().lower(),
            str(rec.get("place")     or "").strip().lower(),
            str(rec.get("latitude")  or "").strip().lower(),
        )
        if all(k in ("", "none") for k in key):
            continue
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


# ─────────────────────────────────────────────
# 7. MAIN PROCESSING LOOP
# ─────────────────────────────────────────────

def process_documents():
    valid_extensions = ('.pdf', '.docx', '.xlsx', '.xls', '.xlsm', '.csv')
    files = [f for f in INPUT_DIR.glob("*.*") if f.suffix.lower() in valid_extensions]

    if not files:
        print(f"No files found in {INPUT_DIR}. Please add some files.")
        return

    for file_path in files:
        print(f"\n[Processing] {file_path.name}")

        try:
            # ── STEP 1 — Chunking ────────────────────────────────────────────
            # Excel: convert rows to labeled key-value blocks BEFORE Docling
            # so hemisphere context from column headers is preserved.
            excel_chunks = excel_to_labeled_chunks(file_path)
            if excel_chunks is not None:
                # Apply hemisphere normalisation to each row-chunk too
                chunks = [_ensure_hemisphere_in_chunk(c) for c in excel_chunks
                          if len(c) >= MIN_CHUNK_LENGTH]
                print(f"  -> Excel mode: {len(chunks)} row-chunk(s)")
            else:
                # Non-Excel path: Docling → markdown → tier-chunking
                result        = converter.convert(file_path)
                markdown_text = result.document.export_to_markdown()
                # Strip form-feeds so Place:/Coordinates: pairs split across
                # pages land in the same chunk
                markdown_text = markdown_text.replace("\x0c", "\n")
                chunks        = chunk_document(result, markdown_text)
                # Inject N/S/E/W where labels exist but letters are absent
                chunks        = [_ensure_hemisphere_in_chunk(c) for c in chunks]
                print(f"  -> Split into {len(chunks)} chunk(s)")

            # ── STEP 2 — Extract per chunk with GLiNER2 ──────────────────────
            all_records: list[dict] = []
            for i, chunk in enumerate(chunks):
                print(f"  -> Extracting chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                try:
                    extracted = extractor.extract_json(chunk, extraction_schema)
                    records   = extracted.get("document_info", [])
                    if isinstance(records, dict):
                        records = [records]
                    all_records.extend(records)
                except Exception as e:
                    print(f"     [WARN] Chunk {i+1} extraction failed: {e}")

            # ── STEP 3 — Deduplicate ──────────────────────────────────────────
            before      = len(all_records)
            all_records = dedup_records(all_records)
            after       = len(all_records)
            if before != after:
                print(f"  -> Deduped: {before} → {after} records")

            print(f"  -> Total records extracted: {after}")

            # ── STEP 4 — Save ─────────────────────────────────────────────────
            output_file = OUTPUT_DIR / f"{file_path.stem}_data.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"document_info": all_records}, f, indent=4, ensure_ascii=False)

            print(f"--- Successfully extracted to: {output_file.name} ---")

        except Exception as e:
            print(f"!!! Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    process_documents()
    print("\n--- Pipeline Execution Complete ---")