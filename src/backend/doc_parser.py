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
#         "latitude::str::A decimal latitude coordinate WITH its hemisphere letter attached "
#                         "(e.g. '25.3176 N', '7.5407 S'). Always include N or S.",
#         "longitude::str::A decimal longitude coordinate WITH its hemisphere letter attached "
#                          "(e.g. '82.9739 E', '74.0060 W'). Always include E or W.",
#         "role::str::Designation of persons mentioned (e.g. Lead Surveyor, Coordinator)",
#         "summary::str::A brief executive summary of the activity or event described",
#     ]
# }

# MIN_CHUNK_LENGTH = 80

# # ─────────────────────────────────────────────
# # 4. EXCEL ROW-LABELLING  (new — Issue #1)
# #
# # Excel files lose hemisphere context when Docling flattens them to
# # markdown: a cell containing "51.9225" carries no sign information
# # because the column header "Latitude (N)" is stripped.
# #
# # Solution: convert each data row to a key: value block so GLiNER2
# # sees "Latitude (N): 51.9225" and can emit "51.9225 N" reliably.
# # ─────────────────────────────────────────────

# EXCEL_EXTENSIONS = ('.xlsx', '.xls', '.xlsm')

# # Maps common Excel column-header patterns to coordinate hints that
# # GLiNER2 can pick up in free text.
# _LAT_HINT = re.compile(r'lat', re.IGNORECASE)
# _LON_HINT = re.compile(r'lo[n|g]', re.IGNORECASE)
# _NORTH_HINT = re.compile(r'\bN\b|north', re.IGNORECASE)
# _SOUTH_HINT = re.compile(r'\bS\b|south', re.IGNORECASE)
# _EAST_HINT  = re.compile(r'\bE\b|east',  re.IGNORECASE)
# _WEST_HINT  = re.compile(r'\bW\b|west',  re.IGNORECASE)


# def _infer_hemisphere(column_name: str, is_lat: bool) -> str:
#     """
#     Return the hemisphere letter implied by a column header, or ""
#     if it cannot be determined.

#     Examples:
#       "Latitude (N)"  → "N"
#       "Lon_W"         → "W"
#       "longitude"     → ""   (ambiguous — leave for cleaner to handle)
#     """
#     if is_lat:
#         if _SOUTH_HINT.search(column_name):
#             return "S"
#         if _NORTH_HINT.search(column_name):
#             return "N"
#     else:
#         if _WEST_HINT.search(column_name):
#             return "W"
#         if _EAST_HINT.search(column_name):
#             return "E"
#     return ""


# def _append_hemisphere(value: str, hemisphere: str) -> str:
#     """
#     Append a hemisphere letter to a numeric coordinate string if it is
#     not already present and the hemisphere is known.

#     "51.9225"   + "N"  →  "51.9225 N"
#     "51.9225 N" + "N"  →  "51.9225 N"   (no duplication)
#     "-74.006"   + ""   →  "-74.006"     (keep existing sign)
#     """
#     if not hemisphere:
#         return value
#     # Already has a direction letter
#     if re.search(r'[NSEWnsew]', value):
#         return value
#     # Has an explicit negative sign — direction is unambiguous; do not append
#     if value.strip().startswith('-'):
#         return value
#     return f"{value.strip()} {hemisphere}"


# def excel_to_labeled_chunks(file_path: Path) -> list[str] | None:
#     """
#     Convert each row of an Excel workbook to a labeled text block,
#     injecting hemisphere suffixes into coordinate values so that
#     GLiNER2 (and cleaner.py) always see explicit N/S/E/W.

#     Returns None if the file is not an Excel format (caller falls
#     through to Docling).

#     Strategy per column:
#       • Detect lat/lon columns by header keyword.
#       • Derive hemisphere from header (e.g. "Lat_N", "Longitude (W)").
#       • Append hemisphere letter to the cell value.
#       • Emit every row as:
#             Sheet: <sheet_name>
#             <col1>: <val1>
#             Latitude: <val> N       ← hemisphere always explicit
#             Longitude: <val> E
#             ...
#     """
#     if file_path.suffix.lower() not in EXCEL_EXTENSIONS:
#         return None

#     try:
#         import pandas as pd
#     except ImportError:
#         print("  [WARN] pandas not installed — Excel processing unavailable.")
#         return None

#     chunks: list[str] = []

#     try:
#         xl = pd.ExcelFile(file_path)
#     except Exception as e:
#         print(f"  [WARN] Could not open Excel file {file_path.name}: {e}")
#         return None

#     for sheet_name in xl.sheet_names:
#         try:
#             df = xl.parse(sheet_name).fillna("")
#         except Exception as e:
#             print(f"  [WARN] Could not parse sheet '{sheet_name}': {e}")
#             continue

#         # Pre-compute column metadata once per sheet
#         col_meta: dict[str, dict] = {}
#         for col in df.columns:
#             col_str = str(col)
#             is_lat = bool(_LAT_HINT.search(col_str))
#             is_lon = bool(_LON_HINT.search(col_str) and not is_lat)
#             hemisphere = ""
#             if is_lat or is_lon:
#                 hemisphere = _infer_hemisphere(col_str, is_lat=is_lat)
#             col_meta[col_str] = {
#                 "is_lat": is_lat,
#                 "is_lon": is_lon,
#                 "hemisphere": hemisphere,
#             }

#         for _, row in df.iterrows():
#             lines = [f"Sheet: {sheet_name}"]
#             for col in df.columns:
#                 raw_val = str(row[col]).strip()
#                 if not raw_val or raw_val in ("nan", "None", ""):
#                     continue

#                 col_str  = str(col)
#                 meta     = col_meta[col_str]
#                 label    = col_str

#                 if meta["is_lat"] or meta["is_lon"]:
#                     # Always emit the coordinate with its hemisphere attached
#                     display_val = _append_hemisphere(raw_val, meta["hemisphere"])
#                     # Use a standardised label so GLiNER2's prompt matches
#                     label = "Latitude" if meta["is_lat"] else "Longitude"
#                     lines.append(f"{label}: {display_val}")
#                 else:
#                     lines.append(f"{col_str}: {raw_val}")

#             chunk = "\n".join(lines).strip()
#             if len(chunk) >= MIN_CHUNK_LENGTH:
#                 chunks.append(chunk)

#     return chunks if chunks else None


# # ─────────────────────────────────────────────
# # 5. CHUNKING  —  three-tier strategy
# #
# # Tier 1  chunk_by_entry()   splits on date-header patterns
# # Tier 2  get_page_chunks()  falls back to Docling page boundaries
# # Tier 3  get_markdown_chunks() / whole-doc fallback
# # ─────────────────────────────────────────────

# DATE_HEADER = re.compile(
#     r'(?:'
#     r'(?:Date|Inspection Date|Recorded on)\s*:\s*'
#     r'|'
#     r'\['
#     r')'
#     r'(?:\w+\s+\d{1,2},?\s+\d{4}'
#     r'|\w+\s+\d{4}'
#     r'|\d{4}-\d{2}-\d{2}'
#     r')',
#     re.IGNORECASE
# )


# def _ensure_hemisphere_in_chunk(chunk: str) -> str:
#     """
#     Post-process a text chunk to append explicit N/S/E/W hemisphere
#     letters to any bare decimal coordinate that follows a Latitude:/
#     Longitude: label but has no direction indicator yet.

#     This is the last-resort normalisation for PDFs and DOCX files where
#     the document author wrote coordinates without hemisphere letters.
#     We use the *label context* (Latitude vs Longitude) and the *numeric
#     sign* to infer direction:

#       Latitude: 34.5822   → Latitude: 34.5822 N   (positive → N)
#       Latitude: -7.5407   → Latitude: -7.5407 S   (negative → S)
#       Longitude: 82.9739  → Longitude: 82.9739 E  (positive → E)
#       Longitude: -74.006  → Longitude: -74.006 W  (negative → W)

#     Coordinates that already carry a letter (N/S/E/W) or that sit
#     outside a labeled context are left unchanged.
#     """
#     def _replace_coord(m: re.Match) -> str:
#         label  = m.group(1)   # "Latitude" or "Longitude"
#         value  = m.group(2)   # e.g. "34.5822" or "-7.5407"
#         rest   = m.group(3)   # anything on the same token after the digits

#         # Already has a hemisphere letter anywhere in rest
#         if re.search(r'[NSEWnsew]', rest):
#             return m.group(0)

#         try:
#             fval = float(value)
#         except ValueError:
#             return m.group(0)

#         is_lat = bool(re.match(r'lat', label, re.IGNORECASE))
#         if is_lat:
#             hemi = "N" if fval >= 0 else "S"
#         else:
#             hemi = "E" if fval >= 0 else "W"

#         return f"{label}: {value} {hemi}{rest}"

#     # Match "Latitude: 34.5822" or "Longitude: -74.006"
#     pattern = re.compile(
#         r'(Lat(?:itude)?|Lon(?:gitude)?)\s*:\s*(-?\d+(?:\.\d+)?)([^\n]*)',
#         re.IGNORECASE
#     )
#     return pattern.sub(_replace_coord, chunk)


# def chunk_by_entry(full_text: str) -> list[str]:
#     splits = list(DATE_HEADER.finditer(full_text))
#     if not splits:
#         return []

#     chunks: list[str] = []
#     for i, match in enumerate(splits):
#         start = match.start()
#         end   = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
#         chunk = full_text[start:end].strip()
#         if len(chunk) >= MIN_CHUNK_LENGTH:
#             chunks.append(chunk)

#     # Orphan-coordinate merge (page-break between Place: and Coordinates:)
#     merged: list[str] = []
#     for chunk in chunks:
#         has_coords = bool(re.search(r'Coordinates\s*:', chunk, re.IGNORECASE))
#         has_place  = bool(re.search(r'(?:^|\n)Place\s*:', chunk, re.IGNORECASE))
#         if has_coords and not has_place and merged:
#             merged[-1] = merged[-1] + "\n\n" + chunk
#         else:
#             merged.append(chunk)
#     return merged


# def get_page_chunks(docling_result) -> list[str]:
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
# # 6. DEDUPLICATION
# # ─────────────────────────────────────────────

# def dedup_records(records: list[dict]) -> list[dict]:
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
# # 7. MAIN PROCESSING LOOP
# # ─────────────────────────────────────────────

# def process_documents():
#     valid_extensions = ('.pdf', '.docx', '.xlsx', '.xls', '.xlsm', '.csv')
#     files = [f for f in INPUT_DIR.glob("*.*") if f.suffix.lower() in valid_extensions]

#     if not files:
#         print(f"No files found in {INPUT_DIR}. Please add some files.")
#         return

#     for file_path in files:
#         print(f"\n[Processing] {file_path.name}")

#         try:
#             # ── STEP 1 — Chunking ────────────────────────────────────────────
#             # Excel: convert rows to labeled key-value blocks BEFORE Docling
#             # so hemisphere context from column headers is preserved.
#             excel_chunks = excel_to_labeled_chunks(file_path)
#             if excel_chunks is not None:
#                 # Apply hemisphere normalisation to each row-chunk too
#                 chunks = [_ensure_hemisphere_in_chunk(c) for c in excel_chunks
#                           if len(c) >= MIN_CHUNK_LENGTH]
#                 print(f"  -> Excel mode: {len(chunks)} row-chunk(s)")
#             else:
#                 # Non-Excel path: Docling → markdown → tier-chunking
#                 result        = converter.convert(file_path)
#                 markdown_text = result.document.export_to_markdown()
#                 # Strip form-feeds so Place:/Coordinates: pairs split across
#                 # pages land in the same chunk
#                 markdown_text = markdown_text.replace("\x0c", "\n")
#                 chunks        = chunk_document(result, markdown_text)
#                 # Inject N/S/E/W where labels exist but letters are absent
#                 chunks        = [_ensure_hemisphere_in_chunk(c) for c in chunks]
#                 print(f"  -> Split into {len(chunks)} chunk(s)")

#             # ── STEP 2 — Extract per chunk with GLiNER2 ──────────────────────
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

#             # ── STEP 3 — Deduplicate ──────────────────────────────────────────
#             before      = len(all_records)
#             all_records = dedup_records(all_records)
#             after       = len(all_records)
#             if before != after:
#                 print(f"  -> Deduped: {before} → {after} records")

#             print(f"  -> Total records extracted: {after}")

#             # ── STEP 4 — Save ─────────────────────────────────────────────────
#             output_file = OUTPUT_DIR / f"{file_path.stem}_data.json"
#             with open(output_file, "w", encoding="utf-8") as f:
#                 json.dump({"document_info": all_records}, f, indent=4, ensure_ascii=False)

#             print(f"--- Successfully extracted to: {output_file.name} ---")

#         except Exception as e:
#             print(f"!!! Error processing {file_path.name}: {e}")


# if __name__ == "__main__":
#     process_documents()
#     print("\n--- Pipeline Execution Complete ---")


# # =============================================================================
# # NEW: INCIDENT SUMMARY EXTRACTOR
# # Added to doc_parser.py without modifying any existing function.
# #
# # Goal: produce a compact, 3–5 line structured summary from raw text
# #       that can be stored in the 'summary' field or passed to the LLM
# #       as a pre-processed context snippet.
# #
# # Design constraints (as specified):
# #   • No heavy NLP dependencies  (only stdlib re, which is already imported)
# #   • Works on any free-text chunk produced by the existing chunkers
# #   • Returns a human-readable string, NOT a dict, so it can be dropped
# #     directly into the 'summary' column of cleaned_records
# # =============================================================================

# # ── Regex patterns used by extract_incident_summary()  # ADDED ──────────────

# # ISO date: 2024-03-15
# _ISO_DATE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')

# # Written date: "March 15, 2024" / "15 March 2024" / "April 2026"
# _WRITTEN_DATE = re.compile(
#     r'\b(?:(\d{1,2})\s+)?'
#     r'(January|February|March|April|May|June|July|August|September|October|November|December)'
#     r'(?:\s+(\d{1,2}),?)?\s+(\d{4})\b',
#     re.IGNORECASE,
# )

# # Labelled date header: "Date: March 15, 2024" / "Inspection Date: 2024-03-15"
# _LABELLED_DATE = re.compile(
#     r'(?:Date|Inspection Date|Recorded on)\s*:\s*([^\n]{5,30})',
#     re.IGNORECASE,
# )

# # Time: "14:30", "1430 hrs", "2:00 PM"
# _TIME_PATTERN = re.compile(
#     r'\b(\d{1,2}:\d{2}(?:\s?[AP]M)?|\d{3,4}\s?hrs?)\b',
#     re.IGNORECASE,
# )

# # Labelled place: "Place: Leh" / "Location: Port of Rotterdam"
# _LABELLED_PLACE = re.compile(
#     r'(?:Place|Location|Site|Waypoint)\s*:\s*([^\n]{3,80})',
#     re.IGNORECASE,
# )

# # Labelled coordinates
# _LABELLED_LAT = re.compile(
#     r'Lat(?:itude)?\s*:\s*(-?\d+(?:\.\d+)?(?:\s*[NSns])?)',
#     re.IGNORECASE,
# )
# _LABELLED_LON = re.compile(
#     r'Lon(?:gitude)?\s*:\s*(-?\d+(?:\.\d+)?(?:\s*[EWew])?)',
#     re.IGNORECASE,
# )

# # Labelled role / personnel
# _LABELLED_ROLE = re.compile(
#     r'(?:Role|Designation|Personnel|Contact|Surveyor|Coordinator)\s*:\s*([^\n]{3,60})',
#     re.IGNORECASE,
# )

# # Labelled summary / event
# _LABELLED_SUMMARY = re.compile(
#     r'(?:Summary|Event|Activity|Incident|Description|Notes?)\s*:\s*([^\n]{5,200})',
#     re.IGNORECASE,
# )

# # Noise patterns to strip before summarising
# _NOISE = re.compile(
#     r'[\r\t]|[ ]{2,}|[-=]{3,}',  # carriage returns, tabs, multi-space, dividers
#     re.MULTILINE,
# )


# def extract_incident_summary(text: str) -> str:  # NEW
#     """
#     Extract a concise, incident-focused 3–5 line structured summary
#     from a raw text chunk.  # NEW

#     Strategy (no external NLP required):
#       1. Strip noise (extra whitespace, dividers, carriage returns).
#       2. Extract labelled fields via regex (Date, Place, Time, Role, Summary).
#       3. Fall back to pattern-matched values when labels are absent.
#       4. Compose a deterministic, human-readable summary string.

#     Parameters
#     ----------
#     text : str
#         Raw text from any tier-chunker or excel_to_labeled_chunks().

#     Returns
#     -------
#     str
#         3–5 line summary in the form::

#             Date    : 2024-03-15
#             Location: Port of Rotterdam (lat 51.9225 N, lon 4.4792 E)
#             Time    : 14:30
#             Role    : Port Logistics Supervisor
#             Event   : Survey of container terminal berth allocation completed.

#         Returns an empty string if no useful fields could be extracted.

#     Notes
#     -----
#     • This function is ADDITIVE — it does not modify any existing pipeline step.
#     • Callers can pass the result to cleaner.py or store it in cleaned_records.summary.
#     • For production use, consider replacing the regex heuristics with a lightweight
#       NER model (e.g. spaCy sm) without changing this function's signature.
#     """  # NEW

#     if not text or not text.strip():  # ADDED
#         return ""

#     # ── Step 1: normalise whitespace / noise  # ADDED ───────────────────────
#     clean = _NOISE.sub(" ", text).strip()

#     # ── Step 2: extract fields  # ADDED ─────────────────────────────────────

#     # Date (prefer labelled, then ISO, then written)
#     date_str = ""
#     m = _LABELLED_DATE.search(clean)
#     if m:
#         date_str = m.group(1).strip()
#     else:
#         m = _ISO_DATE.search(clean)
#         if m:
#             date_str = m.group(1)
#         else:
#             m = _WRITTEN_DATE.search(clean)
#             if m:
#                 day   = m.group(1) or m.group(3) or ""
#                 month = m.group(2)
#                 year  = m.group(4)
#                 date_str = f"{day} {month} {year}".strip()

#     # Time  # ADDED
#     time_str = ""
#     m = _TIME_PATTERN.search(clean)
#     if m:
#         time_str = m.group(1).strip()

#     # Location  # ADDED
#     place_str = ""
#     m = _LABELLED_PLACE.search(clean)
#     if m:
#         place_str = m.group(1).strip()

#     # Coordinates  # ADDED
#     lat_str = lon_str = ""
#     m = _LABELLED_LAT.search(clean)
#     if m:
#         lat_str = m.group(1).strip()
#     m = _LABELLED_LON.search(clean)
#     if m:
#         lon_str = m.group(1).strip()

#     coord_str = ""
#     if lat_str and lon_str:
#         coord_str = f"(lat {lat_str}, lon {lon_str})"
#     elif lat_str:
#         coord_str = f"(lat {lat_str})"
#     elif lon_str:
#         coord_str = f"(lon {lon_str})"

#     # Role / personnel  # ADDED
#     role_str = ""
#     m = _LABELLED_ROLE.search(clean)
#     if m:
#         role_str = m.group(1).strip()

#     # Event / activity summary  # ADDED
#     event_str = ""
#     m = _LABELLED_SUMMARY.search(clean)
#     if m:
#         event_str = m.group(1).strip()
#     else:
#         # Fallback: use first substantive sentence (>= 20 chars, not a label line)
#         for sentence in re.split(r'(?<=[.!?])\s+', clean):
#             sentence = sentence.strip()
#             if (
#                 len(sentence) >= 20
#                 and not re.match(r'^[A-Za-z ]+\s*:', sentence)  # skip "Label: value" lines
#             ):
#                 event_str = sentence[:200]
#                 break

#     # ── Step 3: compose the summary  # ADDED ────────────────────────────────
#     lines: list[str] = []

#     if date_str:
#         lines.append(f"Date    : {date_str}")
#     if time_str:
#         lines.append(f"Time    : {time_str}")

#     loc_line = place_str
#     if coord_str:
#         loc_line = f"{place_str} {coord_str}".strip() if place_str else coord_str
#     if loc_line:
#         lines.append(f"Location: {loc_line}")

#     if role_str:
#         lines.append(f"Role    : {role_str}")
#     if event_str:
#         lines.append(f"Event   : {event_str}")

#     # Return empty string if nothing useful was extracted  # ADDED
#     return "\n".join(lines) if lines else ""
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
        # FIX: More directive hint with concrete examples matching log/conversation format
        "summary::str::The key activity, observation, or decision described in this log entry. "
                       "Examples: 'Riverbank mapping completed at 25.3176 N', "
                       "'Team debated route to Nubra Valley due to weather at Khardung La', "
                       "'Multispectral camera setup at Hunder sand dunes'. "
                       "Extract from Discussion sections or the main body of the log entry.",
    ]
}

MIN_CHUNK_LENGTH = 80

# ─────────────────────────────────────────────
# 4. EXCEL ROW-LABELLING
#
# Excel files lose hemisphere context when Docling flattens them to
# markdown: a cell containing "51.9225" carries no sign information
# because the column header "Latitude (N)" is stripped.
#
# Solution: convert each data row to a key: value block so GLiNER2
# sees "Latitude (N): 51.9225" and can emit "51.9225 N" reliably.
# ─────────────────────────────────────────────

EXCEL_EXTENSIONS = ('.xlsx', '.xls', '.xlsm')

_LAT_HINT   = re.compile(r'lat', re.IGNORECASE)
_LON_HINT   = re.compile(r'lo[n|g]', re.IGNORECASE)
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
    if re.search(r'[NSEWnsew]', value):
        return value
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

                col_str = str(col)
                meta    = col_meta[col_str]
                label   = col_str

                if meta["is_lat"] or meta["is_lon"]:
                    display_val = _append_hemisphere(raw_val, meta["hemisphere"])
                    label = "Latitude" if meta["is_lat"] else "Longitude"
                    lines.append(f"{label}: {display_val}")
                else:
                    lines.append(f"{col_str}: {raw_val}")

            chunk = "\n".join(lines).strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)

    return chunks if chunks else None


# ─────────────────────────────────────────────
# 5. CHUNKING  —  four-tier strategy
#
# Tier 0  chunk_by_site()       splits on "Site:" headers (inspection reports)
# Tier 1  chunk_by_entry()      splits on date-header patterns
# Tier 2  get_page_chunks()     falls back to Docling page boundaries
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
    """
    def _replace_coord(m: re.Match) -> str:
        label = m.group(1)
        value = m.group(2)
        rest  = m.group(3)

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

    pattern = re.compile(
        r'(Lat(?:itude)?|Lon(?:gitude)?)\s*:\s*(-?\d+(?:\.\d+)?)([^\n]*)',
        re.IGNORECASE
    )
    return pattern.sub(_replace_coord, chunk)


def _normalize_discussion_label(chunk: str) -> str:
    """
    Rename 'Discussion:' to 'Summary:' so GLiNER2's schema hint matches
    the Discussion sections present in conversation/log documents.
    """
    return re.sub(r'\bDiscussion\s*:', 'Summary:', chunk, flags=re.IGNORECASE)


# Matches "Site:" at the start of a line — used by chunk_by_site()
SITE_HEADER = re.compile(r'(?:^|\n)(Site\s*:\s*)', re.IGNORECASE)


def chunk_by_site(full_text: str) -> list[str]:
    """
    Tier 0 splitter for ASI-style inspection reports and any document
    that uses 'Site: <name>' as a section header.

    Splits the document at every 'Site:' header so each site gets its
    own chunk — preventing coordinates from one site being attributed
    to the place name of the next site (the cross-contamination bug).

    Returns [] if fewer than 2 'Site:' headers are found, so the caller
    falls through to chunk_by_entry() for other document formats.
    """
    splits = list(SITE_HEADER.finditer(full_text))
    if len(splits) < 2:
        return []

    chunks: list[str] = []
    for i, match in enumerate(splits):
        # Start at the 'S' of 'Site:' (group 1 start), not the newline
        start = match.start(1)
        end   = splits[i + 1].start(1) if i + 1 < len(splits) else len(full_text)
        chunk = full_text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
    return chunks


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
    # Tier 0 — site-header split (ASI / inspection report format)
    chunks = chunk_by_site(markdown_text)
    if chunks:
        return chunks

    # Tier 1 — date-header split (conversation logs, field journals)
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
# 7. INCIDENT SUMMARY EXTRACTOR
#
# Produces a compact 3–5 line structured summary from raw text.
# Used as a post-processing fallback when GLiNER2 returns null summaries.
#
# MUST be defined here — ABOVE process_documents() — so the name is
# resolved when process_documents() calls it at runtime.
# ─────────────────────────────────────────────

# ISO date: 2024-03-15
_ISO_DATE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')

# Written date: "March 15, 2024" / "15 March 2024" / "April 2026"
_WRITTEN_DATE = re.compile(
    r'\b(?:(\d{1,2})\s+)?'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+(\d{1,2}),?)?\s+(\d{4})\b',
    re.IGNORECASE,
)

# Labelled date header: "Date: March 15, 2024" / "Inspection Date: 2024-03-15"
# Stops at pipe (|), parenthesis, or timezone abbreviations so that
# "Inspection Date: April 12, 2026 | Time: 07:15 IST" does not bleed
# "| Time:" into the captured date string.
_LABELLED_DATE = re.compile(
    r'(?:Date|Inspection Date|Recorded on)\s*:\s*([^\n|(\[]{5,30}?)(?:\s*[\|(].*)?$',
    re.IGNORECASE | re.MULTILINE,
)

# Time: "14:30", "1430 hrs", "2:00 PM"
_TIME_PATTERN = re.compile(
    r'\b(\d{1,2}:\d{2}(?:\s?[AP]M)?|\d{3,4}\s?hrs?)\b',
    re.IGNORECASE,
)

# Labelled place: "Place: Leh" / "Location: Port of Rotterdam"
_LABELLED_PLACE = re.compile(
    r'(?:Place|Location|Site|Waypoint)\s*:\s*([^\n]{3,80})',
    re.IGNORECASE,
)

# Labelled coordinates
_LABELLED_LAT = re.compile(
    r'Lat(?:itude)?\s*:\s*(-?\d+(?:\.\d+)?(?:\s*[NSns])?)',
    re.IGNORECASE,
)
_LABELLED_LON = re.compile(
    r'Lon(?:gitude)?\s*:\s*(-?\d+(?:\.\d+)?(?:\s*[EWew])?)',
    re.IGNORECASE,
)

# Labelled role / personnel
_LABELLED_ROLE = re.compile(
    r'(?:Role|Designation|Personnel|Contact|Surveyor|Coordinator)\s*:\s*([^\n]{3,60})',
    re.IGNORECASE,
)

# Labelled summary / event (includes Summary: produced by _normalize_discussion_label)
_LABELLED_SUMMARY = re.compile(
    r'(?:Summary|Event|Activity|Incident|Description|Notes?)\s*:\s*([^\n]{5,200})',
    re.IGNORECASE,
)

# Noise patterns to strip before summarising
_NOISE = re.compile(
    r'[\r\t]|[ ]{2,}|[-=]{3,}',
    re.MULTILINE,
)


def extract_incident_summary(text: str) -> str:
    """
    Extract a concise, incident-focused 3–5 line structured summary
    from a raw text chunk.

    Strategy (no external NLP required):
      1. Strip noise (extra whitespace, dividers, carriage returns).
      2. Extract labelled fields via regex (Date, Place, Time, Role, Summary).
      3. Fall back to pattern-matched values when labels are absent.
      4. Compose a deterministic, human-readable summary string.

    Parameters
    ----------
    text : str
        Raw text from any tier-chunker or excel_to_labeled_chunks().

    Returns
    -------
    str
        3–5 line summary in the form::

            Date    : 2024-03-15
            Location: Port of Rotterdam (lat 51.9225 N, lon 4.4792 E)
            Time    : 14:30
            Role    : Port Logistics Supervisor
            Event   : Survey of container terminal berth allocation completed.

        Returns an empty string if no useful fields could be extracted.
    """
    if not text or not text.strip():
        return ""

    # Step 1: normalise whitespace / noise
    clean = _NOISE.sub(" ", text).strip()

    # Step 2: extract fields

    # Date (prefer labelled, then ISO, then written)
    date_str = ""
    m = _LABELLED_DATE.search(clean)
    if m:
        date_str = m.group(1).strip()
    else:
        m = _ISO_DATE.search(clean)
        if m:
            date_str = m.group(1)
        else:
            m = _WRITTEN_DATE.search(clean)
            if m:
                day      = m.group(1) or m.group(3) or ""
                month    = m.group(2)
                year     = m.group(4)
                date_str = f"{day} {month} {year}".strip()

    # Time
    time_str = ""
    m = _TIME_PATTERN.search(clean)
    if m:
        time_str = m.group(1).strip()

    # Location
    place_str = ""
    m = _LABELLED_PLACE.search(clean)
    if m:
        place_str = m.group(1).strip()

    # Coordinates
    lat_str = lon_str = ""
    m = _LABELLED_LAT.search(clean)
    if m:
        lat_str = m.group(1).strip()
    m = _LABELLED_LON.search(clean)
    if m:
        lon_str = m.group(1).strip()

    coord_str = ""
    if lat_str and lon_str:
        coord_str = f"(lat {lat_str}, lon {lon_str})"
    elif lat_str:
        coord_str = f"(lat {lat_str})"
    elif lon_str:
        coord_str = f"(lon {lon_str})"

    # Role / personnel
    role_str = ""
    m = _LABELLED_ROLE.search(clean)
    if m:
        role_str = m.group(1).strip()

    # Event / activity summary
    event_str = ""
    m = _LABELLED_SUMMARY.search(clean)
    if m:
        event_str = m.group(1).strip()
    else:
        # Fallback: collect ALL substantive sentences (>= 20 chars, not label lines)
        # and join them — captures full paragraphs like ASI inspection body text
        substantive: list[str] = []
        for sentence in re.split(r'(?<=[.!?])\s+', clean):
            sentence = sentence.strip()
            if (
                len(sentence) >= 20
                and not re.match(r'^[A-Za-z ]{1,40}\s*:', sentence)  # skip "Label: value" lines
            ):
                substantive.append(sentence)
        event_str = " ".join(substantive)[:400]  # cap at 400 chars

    # Step 3: compose the summary
    # Date and Time are already top-level record fields — omit them here
    # to avoid duplication in the stored summary string.
    lines: list[str] = []

    loc_line = place_str
    if coord_str:
        loc_line = f"{place_str} {coord_str}".strip() if place_str else coord_str
    if loc_line:
        lines.append(f"Location: {loc_line}")

    if role_str:
        lines.append(f"Role    : {role_str}")
    if event_str:
        lines.append(f"Event   : {event_str}")

    return "\n".join(lines) if lines else ""


# ─────────────────────────────────────────────
# 8. MAIN PROCESSING LOOP
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
            excel_chunks = excel_to_labeled_chunks(file_path)
            if excel_chunks is not None:
                chunks = [
                    _normalize_discussion_label(_ensure_hemisphere_in_chunk(c))
                    for c in excel_chunks
                    if len(c) >= MIN_CHUNK_LENGTH
                ]
                print(f"  -> Excel mode: {len(chunks)} row-chunk(s)")
            else:
                result        = converter.convert(file_path)
                markdown_text = result.document.export_to_markdown()
                markdown_text = markdown_text.replace("\x0c", "\n")
                chunks        = chunk_document(result, markdown_text)
                chunks        = [
                    _normalize_discussion_label(_ensure_hemisphere_in_chunk(c))
                    for c in chunks
                ]
                print(f"  -> Split into {len(chunks)} chunk(s)")

            # ── STEP 2 — Extract per chunk with GLiNER2 ──────────────────────
            all_records: list[dict] = []
            for i, chunk in enumerate(chunks):
                print(f"  -> Extracting chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")

                # GLiNER2 extraction — isolated so a failure here doesn't
                # prevent the summary fallback from running
                records: list[dict] = []
                try:
                    extracted = extractor.extract_json(chunk, extraction_schema)
                    records   = extracted.get("document_info", [])
                    if isinstance(records, dict):
                        records = [records]
                except Exception as e:
                    print(f"     [WARN] GLiNER2 extraction failed on chunk {i+1}: {e}")

                # Summary fallback — runs independently of GLiNER2 result
                for rec in records:
                    if not rec.get("summary"):
                        try:
                            rec["summary"] = extract_incident_summary(chunk) or None
                        except Exception as e:
                            print(f"     [WARN] Summary fallback failed on chunk {i+1}: {e}")

                all_records.extend(records)

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