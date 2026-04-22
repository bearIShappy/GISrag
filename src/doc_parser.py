import os
import json
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from gliner2 import GLiNER2

# 1. Setup Environments and Paths
load_dotenv()

INPUT_DIR = Path(os.getenv("INPUT_FOLDER", "./data/input"))
OUTPUT_DIR = Path(os.getenv("EXTRACTED_OUTPUT_FOLDER", "./data/output"))
MODEL_ID = os.getenv("MODEL_NAME", "fastino/gliner2-large-v1")
DEVICE = os.getenv("DEVICE", "cpu")  # Use "cuda" for your RTX 3060

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Initialize Models
print(f"--- Initializing Extraction Pipeline ---")
print(f"Loading {MODEL_ID} on {DEVICE}...")

converter = DocumentConverter()

extractor = GLiNER2.from_pretrained(MODEL_ID)
if DEVICE == "cuda":
    extractor.to("cuda")

# 3. Schema — unchanged from your original
extraction_schema = {
    "document_info": [
        "date::str::The date mentioned in the document, journey date, or report date",
        "time::str::Specific time or military time format (e.g. 1400 hrs)",
        "place::str::The specific location, city, site, or waypoint name",
        "latitude::str::A decimal latitude coordinate such as 40.7128 N or -74.0060",
        "longitude::str::A decimal longitude coordinate such as 40.7128 N or -74.0060",
        "role::str::Designation of persons mentioned (e.g. Lead Surveyor, Coordinator)",
        "summary::str::A brief executive summary of the activity or event described"
    ]
}

# Minimum characters a chunk must have to be worth extracting from
MIN_CHUNK_LENGTH = 80


# ─────────────────────────────────────────────────────────────
# CHUNKING STRATEGIES
# Docling does NOT add "---" between pages. Instead we use the
# document object's page-level iteration, then fall back to
# paragraph splitting on the exported markdown.
# ─────────────────────────────────────────────────────────────

def get_page_chunks(docling_result) -> list:
    """
    Primary strategy: use Docling's page iterator to get one text
    chunk per physical page. Each page in your PDFs is a
    self-contained event/section.
    """
    chunks = []
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
        pass  # fall through to markdown chunking below
    return chunks


def get_markdown_chunks(markdown_text: str) -> list:
    """
    Fallback strategy: split exported markdown on double newlines
    (paragraph boundaries). Groups short consecutive paragraphs
    together so each chunk has enough context for GLiNER2.
    """
    raw_chunks = [c.strip() for c in markdown_text.split("\n\n") if c.strip()]

    merged = []
    buffer = ""
    for chunk in raw_chunks:
        buffer = (buffer + "\n\n" + chunk).strip() if buffer else chunk
        if len(buffer) >= MIN_CHUNK_LENGTH:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)

    return [c for c in merged if len(c) >= MIN_CHUNK_LENGTH]


def chunk_document(docling_result, markdown_text: str) -> list:
    """
    Try page-level first (preferred). Fall back to paragraph merge.
    Always returns at least one chunk so extraction never silently
    produces zero records.
    """
    chunks = get_page_chunks(docling_result)

    if len(chunks) >= 2:
        return chunks

    chunks = get_markdown_chunks(markdown_text)

    if chunks:
        return chunks

    # Last resort: whole document as one chunk
    return [markdown_text.strip()]


# ─────────────────────────────────────────────────────────────
# DEDUPLICATION
# GLiNER2 can return the same value from adjacent chunks.
# Deduplicate on (date, place, latitude) as a composite key.
# ─────────────────────────────────────────────────────────────

def dedup_records(records: list) -> list:
    seen = set()
    unique = []
    for rec in records:
        key = (
            str(rec.get("date") or "").strip().lower(),
            str(rec.get("place") or "").strip().lower(),
            str(rec.get("latitude") or "").strip().lower(),
        )
        # Skip if all three key fields are null/empty
        if all(k in ("", "none") for k in key):
            continue
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


# ─────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────

def process_documents():
    valid_extensions = ('.pdf', '.docx', '.xlsx', '.csv')
    files = [f for f in INPUT_DIR.glob("*.*") if f.suffix.lower() in valid_extensions]

    if not files:
        print(f"No files found in {INPUT_DIR}. Please add some files.")
        return

    for file_path in files:
        print(f"\n[Processing] {file_path.name}")

        try:
            # STEP 1: Layout-Aware Parsing (unchanged)
            result = converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()

            # STEP 2: Chunk the document before extraction
            chunks = chunk_document(result, markdown_text)
            print(f"  -> Split into {len(chunks)} chunk(s)")

            # STEP 3: Extract per chunk, then merge
            all_records = []
            for i, chunk in enumerate(chunks):
                print(f"  -> Extracting chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                try:
                    extracted = extractor.extract_json(chunk, extraction_schema)
                    records = extracted.get("document_info", [])
                    # GLiNER2 sometimes returns a dict instead of a list
                    if isinstance(records, dict):
                        records = [records]
                    all_records.extend(records)
                except Exception as e:
                    print(f"     [WARN] Chunk {i+1} extraction failed: {e}")
                    continue

            # STEP 4: Deduplicate
            before = len(all_records)
            all_records = dedup_records(all_records)
            after = len(all_records)
            if before != after:
                print(f"  -> Deduped: {before} -> {after} records")

            print(f"  -> Total records extracted: {after}")

            # STEP 5: Save — same format as before so cleaner.py reads it unchanged
            output_file = OUTPUT_DIR / f"{file_path.stem}_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"document_info": all_records}, f, indent=4,
                          ensure_ascii=False)

            print(f"--- Successfully extracted to: {output_file.name} ---")

        except Exception as e:
            print(f"!!! Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    process_documents()
    print("\n--- Pipeline Execution Complete ---")