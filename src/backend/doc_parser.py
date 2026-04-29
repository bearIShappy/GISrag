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
MODEL_ID   = os.getenv("MODEL_NAME", "fastino/gliner2-large-v1")
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
        "latitude::str::A decimal latitude coordinate such as 40.7128 N or -74.0060",
        "longitude::str::A decimal longitude coordinate such as 40.7128 N or -74.0060",
        "role::str::Designation of persons mentioned (e.g. Lead Surveyor, Coordinator)",
        "summary::str::A brief executive summary of the activity or event described",
    ]
}

MIN_CHUNK_LENGTH = 80


# ─────────────────────────────────────────────
# 4. CHUNKING  —  three-tier strategy
#
# Tier 1  chunk_by_entry()   splits on date-header patterns
#         → keeps each observation in its own context window so
#           GLiNER2 cannot bleed entities across records.
#           This is the root fix for coord/place mismatches seen
#           in the India expedition and env-log PDFs.
#
# Tier 2  get_page_chunks()  falls back to Docling page boundaries
#         → good for PDFs where entries span full pages.
#
# Tier 3  get_markdown_chunks() / whole-doc fallback
#         → last resort when neither pattern fires.
# ─────────────────────────────────────────────

# Matches structured date headers in common formats:
#   "Date: March 12, 2024"
#   "[April 11, 2026 | 10:30 AM]"
#   "Inspection Date: April 12, 2026"
#   "Recorded on: October 05, 2024"
#   "Date: April 2026"          ← month-year only
DATE_HEADER = re.compile(
    r'(?:'
    r'(?:Date|Inspection Date|Recorded on)\s*:\s*'          # labelled date
    r'|'
    r'\['                                                    # bracket-prefixed log entry
    r')'
    r'(?:\w+\s+\d{1,2},?\s+\d{4}'                          # "April 11, 2026"
    r'|\w+\s+\d{4}'                                         # "April 2026"
    r'|\d{4}-\d{2}-\d{2}'                                   # ISO date
    r')',
    re.IGNORECASE
)


def chunk_by_entry(full_text: str) -> list[str]:
    """
    Tier-1 chunker: split text on date-header boundaries.

    Each chunk spans from one date header to the next (exclusive),
    so all coordinates and place names extracted from a chunk
    belong to the same observation event.

    Falls back to [full_text] when no headers are found so
    subsequent tiers can take over.
    """
    splits = list(DATE_HEADER.finditer(full_text))
    if not splits:
        return []   # signal: no date headers found, try next tier

    chunks: list[str] = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
        chunk = full_text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
    return chunks


def get_page_chunks(docling_result) -> list[str]:
    """
    Tier-2 chunker: one chunk per physical PDF page via Docling.
    Used when date-header splitting yields nothing (e.g. single-
    page PDFs or PDFs without labelled date lines).
    """
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
    """
    Tier-3 chunker: split exported markdown on double newlines,
    merging short paragraphs until each chunk exceeds MIN_CHUNK_LENGTH.
    """
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
    """
    Orchestrate the three-tier chunking strategy.

    Priority:
      1. Date-header entry splitting  (most accurate, entry-scoped)
      2. Page-level splitting         (good for page-per-entry PDFs)
      3. Paragraph merging            (generic fallback)
      4. Whole document               (last resort)
    """
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
# 5. DEDUPLICATION
# ─────────────────────────────────────────────

def dedup_records(records: list[dict]) -> list[dict]:
    """
    Remove duplicate records produced by overlapping chunks.
    Composite key: (date, place, latitude).
    Records where all three are null/empty are also dropped.
    """
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
# 6. MAIN PROCESSING LOOP
# ─────────────────────────────────────────────

def process_documents():
    valid_extensions = ('.pdf', '.docx', '.xlsx', '.csv')
    files = [f for f in INPUT_DIR.glob("*.*") if f.suffix.lower() in valid_extensions]

    if not files:
        print(f"No files found in {INPUT_DIR}. Please add some files.")
        return

    for file_path in files:
        print(f"\n[Processing] {file_path.name}")

        try:
            # STEP 1 — Layout-aware parsing via Docling
            result        = converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()

            # STEP 2 — Chunk (entry-level preferred, page/para fallback)
            chunks = chunk_document(result, markdown_text)
            print(f"  -> Split into {len(chunks)} chunk(s)")

            # STEP 3 — Extract per chunk with GLiNER2
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

            # STEP 4 — Deduplicate
            before        = len(all_records)
            all_records   = dedup_records(all_records)
            after         = len(all_records)
            if before != after:
                print(f"  -> Deduped: {before} → {after} records")

            print(f"  -> Total records extracted: {after}")

            # STEP 5 — Save (format unchanged so cleaner.py reads it as-is)
            output_file = OUTPUT_DIR / f"{file_path.stem}_data.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"document_info": all_records}, f, indent=4, ensure_ascii=False)

            print(f"--- Successfully extracted to: {output_file.name} ---")

        except Exception as e:
            print(f"!!! Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    process_documents()
    print("\n--- Pipeline Execution Complete ---")