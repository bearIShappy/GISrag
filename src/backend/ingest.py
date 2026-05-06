"""
ingest.py — Stage 3: Embedding Generation + PostgreSQL/pgvector Ingestion
---------------------------------------------------------------------------
Input:  cleaned_records.json  (output of cleaner.py)
        Located at: settings.cleaned_records_path
        OR  data/output/cleaned/cleaned_records.json  (env fallback)

Output: Embeddings stored in PostgreSQL via pgvector for semantic search

Full field schema preserved from cleaner.py:
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ place               │ Raw place name extracted by GLiNER2              │
  │ geocoded_place      │ Nominatim/Photon reverse-geocoded label          │
  │ latitude            │ Signed float (N=+, S=−)                         │
  │ longitude           │ Signed float (E=+, W=−)                         │
  │ date                │ ISO 8601 string or period (e.g. "2024-04")       │
  │ date_type           │ event | period | historical | unparseable | None │
  │ date_raw            │ Original date string from doc_parser             │
  │ time                │ Normalised HH:MM[:SS] or None                    │
  │ role                │ Person designation from GLiNER2                  │
  │ summary             │ Executive summary from GLiNER2                   │
  │ geocode_status      │ exact | forward | reverse | reverse_coord_       │
  │                     │   fallback | failed | partial | mismatch         │
  │ source_file         │ Origin *_data.json filename                      │
  │ sanity_note         │ Optional coord-correction note from cleaner      │
  └─────────────────────┴──────────────────────────────────────────────────┘

Pipeline:
  1. Load cleaned_records.json
  2. Filter/warn on geocode_status (skip mismatch; warn partial/failed)
  3. Generate embeddings with SentenceTransformer (batched)
     Embed priority: summary → place/geocoded_place + date + role (fallback)
  4. Attach stable document ID (sha256 of source_file|date|place|latitude)
  5. Upsert records + embeddings into PostgreSQL via db_store
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

from .db_store import (
    ensure_schema,
    insert_records,
    close_pool,
)

from src.config.settings import settings

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Records with these geocode_status values are silently dropped.
# "mismatch" means the extractor produced contradictory lat/lon+place pairs
# that cleaner.py could not resolve — ingesting them would pollute the index.
SKIP_STATUSES = {"mismatch"}

# Records with these statuses are ingested but flagged in the log.
# Coordinates exist but confidence is reduced (partial fix, failed reverse, etc.)
WARN_STATUSES = {"partial", "reverse_failed", "failed", "forward", "reverse_coord_fallback"}

EMBED_BATCH_SIZE: int = settings.text_embedding.BATCH_SIZE
EMBED_MODEL_PATH: str = settings.text_embedding.MODEL_PATH

# Fallback cleaned_records path mirrors cleaner.py OUTPUT_DIR convention:
#   data/output/cleaned/cleaned_records.json
_DEFAULT_CLEANED_PATH = Path(
    os.getenv("CLEANED_OUTPUT_FOLDER", "./data/output/cleaned")
) / "cleaned_records.json"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_document_id(record: Dict[str, Any]) -> str:
    """
    Deterministic 16-char document ID derived from stable identity fields.

    Uses source_file + date + place + latitude — the same fields cleaner.py
    uses for deduplication — so re-ingesting the same cleaned_records.json
    always produces the same IDs (enabling safe upserts).
    """
    key = "|".join([
        str(record.get("source_file") or ""),
        str(record.get("date")        or ""),
        str(record.get("place")       or ""),
        str(record.get("latitude")    or ""),
    ])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_embed_text(record: Dict[str, Any]) -> str:
    """
    Build the text that will be encoded into a vector embedding.

    Priority order (matches the richness of semantic content):
      1. summary          — GLiNER2 executive summary; best semantic signal
      2. place + geocoded_place + date + role + source_file
                          — structured fallback when summary is absent
         • Both place variants are included so the vector captures the
           raw GLiNER2 extraction AND the Nominatim-resolved label.
         • date_raw is preferred over the ISO date so temporal phrasing
           from the original document is preserved in the embedding.
      3. "unknown record" — last resort; prevents encoding an empty string.
    """
    # Tier 1 — summary
    summary = (record.get("summary") or "").strip()
    if summary:
        return summary

    # Tier 2 — structured fields
    parts: List[Optional[str]] = [
        record.get("place"),
        record.get("geocoded_place"),   # reverse/forward geocoded label
        record.get("date_raw") or record.get("date"),
        record.get("role"),
        record.get("source_file"),
    ]
    fallback = " ".join(str(p) for p in parts if p)
    return fallback.strip() or "unknown record"


def _log_geocode_warnings(record: Dict[str, Any]) -> None:
    """Emit a structured warning for low-confidence geocode records."""
    logger.warning(
        "Low-confidence coords (geocode_status=%s): "
        "place=%r  geocoded_place=%r  source=%s",
        record.get("geocode_status"),
        record.get("place"),
        record.get("geocoded_place"),
        record.get("source_file"),
    )


def validate_and_filter(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enforce geocode_status filtering rules from cleaner.py before embedding.

    - SKIP_STATUSES  (mismatch):  record is dropped; logged as WARNING.
    - WARN_STATUSES  (partial, reverse_failed, failed, forward,
                      reverse_coord_fallback):
                     record is kept; low-confidence coords are flagged.
    - Records missing both place AND geocoded_place AND coords are also
      dropped — they carry no locatable information.

    Returns the accepted subset.
    """
    accepted: List[Dict[str, Any]] = []
    skipped = 0

    for rec in records:
        status = rec.get("geocode_status", "exact")

        # Hard skip — unresolvable by cleaner
        if status in SKIP_STATUSES:
            logger.warning(
                "Skipping record (geocode_status=%s): "
                "place=%r  date=%r  source=%s",
                status,
                rec.get("place"),
                rec.get("date"),
                rec.get("source_file"),
            )
            skipped += 1
            continue

        # Soft warn — usable but degraded coordinates
        if status in WARN_STATUSES:
            _log_geocode_warnings(rec)

        # Drop records with no locatable information at all.
        # (cleaner.py tags these with drop_reason="no_place_no_coords" and
        #  routes them to quarantine.json, but we guard here defensively.)
        has_place  = bool(rec.get("place") or rec.get("geocoded_place"))
        has_coords = rec.get("latitude") is not None and rec.get("longitude") is not None
        if not has_place and not has_coords:
            logger.warning(
                "Dropping unlocatable record (no place, no coords): source=%s",
                rec.get("source_file"),
            )
            skipped += 1
            continue

        accepted.append(rec)

    if skipped:
        logger.info(
            "Filtered out %d record(s) before embedding "
            "(status in %s or no locatable info).",
            skipped, SKIP_STATUSES,
        )

    return accepted


def prepare_batch(
    records: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """
    Filter, batch-encode, and enrich records with a stable id + embedding.

    Steps:
      1. validate_and_filter  — drops mismatch/unlocatable records
      2. build_embed_text     — constructs text per record (summary-first)
      3. model.encode (batched) — generates float32 vectors
      4. Attach id + embedding to each accepted record dict

    All original cleaner.py fields are preserved unchanged.
    """
    accepted = validate_and_filter(records)

    if not accepted:
        return []

    texts = [build_embed_text(r) for r in accepted]

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i : i + EMBED_BATCH_SIZE]
        embeddings  = model.encode(batch_texts, show_progress_bar=True)

        for j, emb in enumerate(embeddings):
            idx = i + j
            accepted[idx] = {
                **accepted[idx],
                "id":        make_document_id(accepted[idx]),
                "embedding": emb.tolist(),
            }

        logger.info(
            "Embedded records %d–%d / %d",
            i + 1,
            min(i + EMBED_BATCH_SIZE, len(accepted)),
            len(accepted),
        )

    return accepted


def _build_summary_stats(batch: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Compute per-status counts and field-coverage stats for the ingestion
    summary printout — mirrors the statistics cleaner.py prints.
    """
    stats: Dict[str, int] = {
        "total":          len(batch),
        "with_coords":    sum(1 for r in batch if r.get("latitude") and r.get("longitude")),
        "with_place":     sum(1 for r in batch if r.get("place")),
        "with_geocoded":  sum(1 for r in batch if r.get("geocoded_place")),
        "with_date":      sum(1 for r in batch if r.get("date")),
        "with_summary":   sum(1 for r in batch if r.get("summary")),
        "with_role":      sum(1 for r in batch if r.get("role")),
        "sanity_fixed":   sum(1 for r in batch if r.get("sanity_note")),
        # geocode_status breakdown
        "exact":                  sum(1 for r in batch if r.get("geocode_status") == "exact"),
        "forward":                sum(1 for r in batch if r.get("geocode_status") == "forward"),
        "reverse":                sum(1 for r in batch if r.get("geocode_status") == "reverse"),
        "reverse_coord_fallback": sum(1 for r in batch if r.get("geocode_status") == "reverse_coord_fallback"),
        "partial":                sum(1 for r in batch if r.get("geocode_status") == "partial"),
        "failed":                 sum(1 for r in batch if r.get("geocode_status") == "failed"),
    }
    return stats


# ─────────────────────────────────────────────
# PostgreSQL + pgvector INGESTOR
# ─────────────────────────────────────────────

class PgVectorIngestor:
    """
    Ingests cleaned_records.json into PostgreSQL with pgvector embeddings.

    All fields produced by cleaner.py are stored verbatim in PostgreSQL;
    nothing is silently dropped.  The embedding is generated from the
    richest available text signal (summary → structured fallback).
    """

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", EMBED_MODEL_PATH)
        self.model = SentenceTransformer(EMBED_MODEL_PATH)

    def setup_schema(self) -> None:
        """Create tables, indexes, and pgvector extension if absent."""
        ensure_schema()
        logger.info("PostgreSQL + pgvector schema verified.")

    def ingest_records(self, records: List[Dict[str, Any]]) -> None:
        """
        Generate embeddings for all accepted records and upsert into PostgreSQL.

        Accepts the full list from cleaned_records.json (including any records
        that cleaner.py did not quarantine).  Filtering on geocode_status and
        field completeness is handled inside prepare_batch().
        """
        if not records:
            logger.warning("ingest_records called with empty list — nothing to do.")
            return

        batch = prepare_batch(records, self.model)
        if not batch:
            logger.warning(
                "All records were filtered out after validation. "
                "Check geocode_status distribution in cleaned_records.json."
            )
            return

        db_summary = insert_records(batch)
        field_stats = _build_summary_stats(batch)

        logger.info(
            "Ingested %d record(s)  (inserted=%d  updated=%d  errors=%d).",
            len(batch),
            db_summary["inserted"],
            db_summary["updated"],
            db_summary["errors"],
        )

        # ── Pretty print (mirrors cleaner.py output style) ──────────────────
        w = 42
        print(f"\n  {'─'*w}")
        print(f"  pgvector Ingestion Summary")
        print(f"  {'─'*w}")
        print(f"  Records processed        : {field_stats['total']}")
        print(f"  ✓ Inserted               : {db_summary['inserted']}")
        print(f"  ↻ Updated                : {db_summary['updated']}")
        print(f"  ✗ Errors                 : {db_summary['errors']}")
        print(f"  {'─'*w}")
        print(f"  Field coverage")
        print(f"    With coordinates       : {field_stats['with_coords']}/{field_stats['total']}")
        print(f"    With place name        : {field_stats['with_place']}/{field_stats['total']}")
        print(f"    With geocoded_place    : {field_stats['with_geocoded']}/{field_stats['total']}")
        print(f"    With date              : {field_stats['with_date']}/{field_stats['total']}")
        print(f"    With summary (→ embed) : {field_stats['with_summary']}/{field_stats['total']}")
        print(f"    With role              : {field_stats['with_role']}/{field_stats['total']}")
        print(f"    Sanity-fixed coords    : {field_stats['sanity_fixed']}")
        print(f"  {'─'*w}")
        print(f"  Geocode status breakdown")
        print(f"    exact                  : {field_stats['exact']}")
        print(f"    forward                : {field_stats['forward']}")
        print(f"    reverse                : {field_stats['reverse']}")
        print(f"    reverse_coord_fallback : {field_stats['reverse_coord_fallback']}")
        print(f"    partial                : {field_stats['partial']}")
        print(f"    failed                 : {field_stats['failed']}")
        print(f"  {'─'*w}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    """
    Resolve cleaned_records.json path, load, validate, embed, and ingest.

    Path resolution order (mirrors cleaner.py OUTPUT_DIR convention):
      1. settings.cleaned_records_path          (from src/config/settings.py)
      2. data/output/cleaned/cleaned_records.json  (CLEANED_OUTPUT_FOLDER env)
      3. cleaned_records.json                   (cwd fallback)
    """
    candidates = [
        str(settings.cleaned_records_path),
        str(_DEFAULT_CLEANED_PATH),
        "cleaned_records.json",
    ]

    path: Optional[str] = None
    for candidate in candidates:
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        logger.error(
            "cleaned_records.json not found at any of:\n  %s\n"
            "Run cleaner.py first.",
            "\n  ".join(candidates),
        )
        return

    try:
        logger.info("Loading records from: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a JSON array at top level, got {type(records).__name__}. "
                "Ensure cleaner.py wrote a flat list (not {{\"document_info\": [...]}})"
            )

        logger.info("Loaded %d record(s).", len(records))

        ingestor = PgVectorIngestor()
        ingestor.setup_schema()
        ingestor.ingest_records(records)

        logger.info("Ingestion process completed successfully.")

    except FileNotFoundError:
        logger.error(
            "File disappeared between discovery and open: %s", path
        )
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse cleaned_records.json: %s\n"
            "The file may be truncated — re-run cleaner.py.", e
        )
    except ValueError as e:
        logger.error("Unexpected format in cleaned_records.json: %s", e)
    except ConnectionError as e:
        logger.error(
            "Could not connect to PostgreSQL. Make sure Docker is running:\n"
            "  docker compose -f Docker/docker-compose.yml up -d\n"
            "Error: %s", e,
        )
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
    finally:
        close_pool()


if __name__ == "__main__":
    main()