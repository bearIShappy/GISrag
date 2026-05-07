"""
db_store.py — Stage 2b: PostgreSQL + pgvector Persistence
-----------------------------------------------------------
Insert cleaned records produced by cleaner.py into PostgreSQL and
store vector embeddings via the pgvector extension for semantic search.

Architecture:
  PostgreSQL (static fields)   — metadata, keyword search, geo queries
  pgvector   (embedding column) — cosine-similarity semantic search

Design goals:
  • Modular:   exposes store_cleaned_records(records) — one call from main.py
  • Resilient: connection retries with exponential back-off
  • Idempotent: UPSERT (ON CONFLICT) so re-runs never duplicate rows
  • Schema:    two tables
      – documents       unique source files
      – cleaned_records  one row per observation, FK → documents,
                          includes a 'vector' embedding column for pgvector
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv

import psycopg2
import psycopg2.extras
from psycopg2 import pool, OperationalError, DatabaseError

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. CONFIGURATION  (from .env)
# ─────────────────────────────────────────────

PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = int(os.getenv("PG_PORT",  "5432"))
PG_DB       = os.getenv("PG_DB",       "gisrag_db")
PG_USER     = os.getenv("PG_USER",     "gisrag")
PG_PASSWORD = os.getenv("PG_PASSWORD", "gisrag_secret")

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))

CLEANED_OUTPUT_DIR = Path(os.getenv("CLEANED_OUTPUT_FOLDER", "./data/output/cleaned"))
CLEANED_FILE       = CLEANED_OUTPUT_DIR / "cleaned_records.json"

# Connection-pool limits
_POOL_MIN = 1
_POOL_MAX = 5

# Retry policy
MAX_RETRIES    = 5
INITIAL_DELAY  = 1        # seconds
BACKOFF_FACTOR = 2


# ─────────────────────────────────────────────
# 2. CONNECTION POOL  (with retry)
# ─────────────────────────────────────────────

_connection_pool: pool.SimpleConnectionPool | None = None


def _create_pool() -> pool.SimpleConnectionPool:
    """
    Create or return the singleton connection pool.
    Retries with exponential back-off if PostgreSQL is not yet ready
    (common when the Docker container is still starting).
    """
    global _connection_pool
    if _connection_pool and not _connection_pool.closed:
        return _connection_pool

    delay = INITIAL_DELAY
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _connection_pool = pool.SimpleConnectionPool(
                _POOL_MIN,
                _POOL_MAX,
                host=PG_HOST,
                port=PG_PORT,
                dbname=PG_DB,
                user=PG_USER,
                password=PG_PASSWORD,
                connect_timeout=10,
            )
            logger.info(
                "PostgreSQL connection pool created  "
                f"(host={PG_HOST}:{PG_PORT}, db={PG_DB})"
            )
            return _connection_pool

        except OperationalError as exc:
            last_err = exc
            logger.warning(
                f"[Attempt {attempt}/{MAX_RETRIES}] "
                f"Cannot connect to PostgreSQL: {exc}  — retrying in {delay}s"
            )
            time.sleep(delay)
            delay *= BACKOFF_FACTOR

    raise ConnectionError(
        f"Failed to connect to PostgreSQL after {MAX_RETRIES} attempts: {last_err}"
    )


def close_pool() -> None:
    """Close every connection in the pool (call at shutdown)."""
    global _connection_pool
    if _connection_pool and not _connection_pool.closed:
        _connection_pool.closeall()
        logger.info("PostgreSQL connection pool closed.")
        _connection_pool = None


@contextmanager
def get_connection():
    """
    Context manager that checks out a connection from the pool,
    yields it, and guarantees it is returned afterward.
    """
    p = _create_pool()
    conn = p.getconn()
    try:
        yield conn
    finally:
        p.putconn(conn)


# ─────────────────────────────────────────────
# 3. SCHEMA CREATION  (pgvector-enabled)
# ─────────────────────────────────────────────

_SCHEMA_SQL = f"""
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Source documents (one row per parsed file)
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL       PRIMARY KEY,
    source_file   VARCHAR(512) NOT NULL UNIQUE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Cleaned observation records  (static fields + embedding)
CREATE TABLE IF NOT EXISTS cleaned_records (
    id               SERIAL        PRIMARY KEY,
    document_id      INTEGER       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Stable hash for idempotent upserts
    record_hash      VARCHAR(64)   NOT NULL UNIQUE,

    -- Location  (keyword search + geo queries)
    place            TEXT,
    geocoded_place   TEXT,
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    geocode_status   VARCHAR(64),

    -- Temporal  (keyword / range search)
    date             VARCHAR(32),
    date_type        VARCHAR(32),
    date_raw         TEXT,
    time             VARCHAR(64),

    -- Metadata  (keyword search)
    role             TEXT,
    summary          TEXT,
    sanity_note      TEXT,
    source_file      VARCHAR(512)  NOT NULL,

    -- Semantic search  (pgvector)
    embedding        vector({VECTOR_DIMENSION}),

    -- Housekeeping
    inserted_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- Indexes for keyword / metadata search
CREATE INDEX IF NOT EXISTS idx_records_source   ON cleaned_records (source_file);
CREATE INDEX IF NOT EXISTS idx_records_place    ON cleaned_records USING gin (to_tsvector('english', coalesce(place, '')));
CREATE INDEX IF NOT EXISTS idx_records_summary  ON cleaned_records USING gin (to_tsvector('english', coalesce(summary, '')));
CREATE INDEX IF NOT EXISTS idx_records_date     ON cleaned_records (date);
CREATE INDEX IF NOT EXISTS idx_records_role     ON cleaned_records (role);
CREATE INDEX IF NOT EXISTS idx_records_geocode  ON cleaned_records (geocode_status);
CREATE INDEX IF NOT EXISTS idx_records_coords   ON cleaned_records (latitude, longitude);

-- HNSW index for fast approximate nearest-neighbor search on embeddings
CREATE INDEX IF NOT EXISTS idx_records_embedding ON cleaned_records
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


def ensure_schema() -> None:
    """Create pgvector extension, tables, and indexes if they do not exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
    logger.info("PostgreSQL + pgvector schema verified / created.")


# ─────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────

def make_record_hash(rec: dict) -> str:
    """
    Deterministic hash from stable fields for idempotent upserts.
    """
    key = "|".join([
        str(rec.get("source_file") or ""),
        str(rec.get("date")        or ""),
        str(rec.get("place")       or ""),
        str(rec.get("latitude")    or ""),
    ])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _ensure_document(cur, source_file: str) -> int:
    """
    Return the documents.id for *source_file*, inserting a new row
    if it does not yet exist.
    """
    cur.execute(
        """
        INSERT INTO documents (source_file)
        VALUES (%s)
        ON CONFLICT (source_file) DO UPDATE SET source_file = EXCLUDED.source_file
        RETURNING id
        """,
        (source_file,),
    )
    return cur.fetchone()[0]


# ─────────────────────────────────────────────
# 5. DATA INSERTION  (upsert with optional embedding)
# ─────────────────────────────────────────────

def _upsert_record(cur, doc_id: int, rec: dict) -> bool:
    """
    Insert or update a single cleaned_record.
    Returns True if a new row was inserted, False on update.
    """
    embedding = rec.get("embedding")
    # Convert list → pgvector literal string, or None
    embedding_str = None
    if embedding is not None:
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    record_hash = make_record_hash(rec)

    cur.execute(
        """
        INSERT INTO cleaned_records (
            document_id, record_hash,
            place, geocoded_place, latitude, longitude, geocode_status,
            date, date_type, date_raw, time,
            role, summary, sanity_note, source_file,
            embedding
        )
        VALUES (
            %(doc_id)s, %(record_hash)s,
            %(place)s, %(geocoded_place)s, %(latitude)s, %(longitude)s, %(geocode_status)s,
            %(date)s, %(date_type)s, %(date_raw)s, %(time)s,
            %(role)s, %(summary)s, %(sanity_note)s, %(source_file)s,
            %(embedding)s::vector
        )
        ON CONFLICT (record_hash)
        DO UPDATE SET
            geocoded_place = EXCLUDED.geocoded_place,
            longitude      = EXCLUDED.longitude,
            geocode_status = EXCLUDED.geocode_status,
            date_type      = EXCLUDED.date_type,
            date_raw       = EXCLUDED.date_raw,
            time           = EXCLUDED.time,
            role           = EXCLUDED.role,
            summary        = EXCLUDED.summary,
            sanity_note    = EXCLUDED.sanity_note,
            embedding      = EXCLUDED.embedding
        RETURNING (xmax = 0) AS was_insert
        """,
        {
            "doc_id":         doc_id,
            "record_hash":    record_hash,
            "place":          rec.get("place"),
            "geocoded_place": rec.get("geocoded_place"),
            "latitude":       rec.get("latitude"),
            "longitude":      rec.get("longitude"),
            "geocode_status": rec.get("geocode_status"),
            "date":           rec.get("date"),
            "date_type":      rec.get("date_type"),
            "date_raw":       rec.get("date_raw"),
            "time":           rec.get("time"),
            "role":           rec.get("role"),
            "summary":        rec.get("summary"),
            "sanity_note":    rec.get("sanity_note"),
            "source_file":    rec.get("source_file", ""),
            "embedding":      embedding_str,
        },
    )
    row = cur.fetchone()
    return row[0] if row else True  # was_insert: True = fresh insert


def insert_records(records: list[dict]) -> dict:
    """
    Batch-insert a list of cleaned record dicts into PostgreSQL.

    Returns a summary dict:
      {"inserted": int, "updated": int, "errors": int}
    """
    inserted = updated = errors = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            doc_cache: dict[str, int] = {}

            for rec in records:
                src = rec.get("source_file", "unknown")
                try:
                    if src not in doc_cache:
                        doc_cache[src] = _ensure_document(cur, src)
                    doc_id = doc_cache[src]

                    was_insert = _upsert_record(cur, doc_id, rec)
                    if was_insert:
                        inserted += 1
                    else:
                        updated += 1

                except DatabaseError as exc:
                    conn.rollback()
                    errors += 1
                    logger.warning(f"DB error inserting record from {src}: {exc}")

        conn.commit()

    summary = {"inserted": inserted, "updated": updated, "errors": errors}
    logger.info(f"PostgreSQL insert summary: {summary}")
    return summary


# ─────────────────────────────────────────────
# 6. SEMANTIC SEARCH  (pgvector cosine similarity)
# ─────────────────────────────────────────────

def semantic_search(
    query_embedding: list[float],
    limit: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Find the *limit* nearest records by cosine similarity.

    Parameters
    ----------
    query_embedding : list[float]
        The embedding vector for the search query.
    limit : int
        Maximum number of results.
    min_score : float
        Minimum cosine similarity (0–1). Records below this are excluded.

    Returns
    -------
    list[dict]
        Each dict contains all record fields plus a 'similarity' score.
    """
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    sql = """
    SELECT
        cr.id,
        cr.place,
        cr.geocoded_place,
        cr.latitude,
        cr.longitude,
        cr.date,
        cr.date_type,
        cr.time,
        cr.role,
        cr.summary,
        cr.source_file,
        cr.geocode_status,
        1 - (cr.embedding <=> %(emb)s::vector) AS similarity
    FROM cleaned_records cr
    WHERE cr.embedding IS NOT NULL
    ORDER BY cr.embedding <=> %(emb)s::vector
    LIMIT %(limit)s
    """

    results: list[dict] = []
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"emb": emb_str, "limit": limit})
            for row in cur.fetchall():
                if row["similarity"] >= min_score:
                    results.append(dict(row))

    return results


# ────────────────────────────────────────────
# 6b. HYBRID SEARCH  # NEW
#     Phase 1: structured SQL filter (location / date / category)
#     Phase 2: pgvector cosine-similarity on the filtered CTE
#
#     This makes the two-phase flow explicit and traceable so callers
#     can log the filter hit count separately from the vector rank.
# ────────────────────────────────────────────

def hybrid_search(  # NEW
    query_embedding: list[float],
    *,
    place:    str | None = None,
    date:     str | None = None,
    category: str | None = None,   # maps to the 'role' column (event category / type)
    top_k:    int = 5,
    min_score: float = 0.0,
) -> dict:
    """
    Two-phase Hybrid Search.  # NEW

    Phase 1 — SQL structured filter
    --------------------------------
    Builds a CTE (`filtered`) that applies optional WHERE predicates on
    location, date, and category (role) columns.  These are fast index
    scans on the existing b-tree / GIN indexes.

    Phase 2 — pgvector semantic re-rank
    ------------------------------------
    Runs cosine-distance ordering ONLY on the rows that survived the
    filter.  Because pgvector operates on the CTE output (not the full
    table) this is significantly faster when filters are selective.

    Parameters
    ----------
    query_embedding : list[float]   Query vector from embed_query().
    place           : str | None    ILIKE pattern on place / geocoded_place.
    date            : str | None    Exact or prefix match on the date column.
    category        : str | None    ILIKE pattern on role (event category).
    top_k           : int           Maximum rows to return after re-ranking.
    min_score       : float         Cosine similarity floor (0–1).

    Returns
    -------
    dict with keys:
        'filter_count' : int         — rows matched by the SQL filter
        'results'      : list[dict]  — top_k rows sorted by similarity desc
    """
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    # ── Phase 1: build the SQL filter predicate ──
    filter_clauses: list[str] = ["embedding IS NOT NULL"]
    params: dict[str, Any] = {"emb": emb_str, "top_k": top_k}

    if place:  # ADDED
        filter_clauses.append(
            "(place ILIKE %(place_pat)s OR geocoded_place ILIKE %(place_pat)s)"
        )
        params["place_pat"] = f"%{place}%"

    if date:  # ADDED
        filter_clauses.append("(date = %(date_val)s OR date LIKE %(date_pre)s)")
        params["date_val"] = date
        params["date_pre"] = f"{date}%"

    if category:  # ADDED — category maps to the 'role' column
        filter_clauses.append("role ILIKE %(cat_pat)s")
        params["cat_pat"] = f"%{category}%"

    where_sql = " AND ".join(filter_clauses)

    # ── Phase 2: CTE + vector re-rank on the filtered subset ──
    sql = f"""
    WITH filtered AS (
        -- Phase 1: SQL structured filter  # NEW
        SELECT
            id, place, geocoded_place, latitude, longitude,
            date, date_type, time, role, summary, source_file,
            geocode_status, embedding
        FROM cleaned_records
        WHERE {where_sql}
    ),
    counted AS (
        SELECT COUNT(*) AS total FROM filtered
    )
    -- Phase 2: pgvector cosine re-rank on filtered CTE  # NEW
    SELECT
        f.id,
        f.place,
        f.geocoded_place,
        f.latitude,
        f.longitude,
        f.date,
        f.date_type,
        f.time,
        f.role,
        f.summary,
        f.source_file,
        f.geocode_status,
        1 - (f.embedding <=> %(emb)s::vector) AS similarity,
        c.total                                              AS filter_count
    FROM filtered f, counted c
    ORDER BY f.embedding <=> %(emb)s::vector
    LIMIT %(top_k)s
    """

    results: list[dict] = []
    filter_count = 0

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                row_dict = dict(row)
                filter_count = row_dict.pop("filter_count", 0)  # extract once
                if row_dict["similarity"] >= min_score:
                    results.append(row_dict)

    logger.info(
        "hybrid_search: filter_count=%d  returned=%d  (top_k=%d, min_score=%.2f)",
        filter_count, len(results), top_k, min_score,
    )
    return {"filter_count": filter_count, "results": results}


# ────────────────────────────────────────────
# 6c. CATEGORY / ROLE SEARCH  # ADDED
# ────────────────────────────────────────────

def search_by_category(category: str, limit: int = 20) -> list[dict]:  # ADDED
    """
    Case-insensitive ILIKE filter on the 'role' column, which stores the
    event category / designation (e.g. 'Surveyor', 'Fire Incident', 'Flood').

    This mirrors search_by_place / search_by_date for category-based filtering.
    """  # ADDED
    sql = """
    SELECT * FROM cleaned_records
    WHERE role ILIKE %(pattern)s
    ORDER BY date NULLS LAST
    LIMIT %(limit)s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"pattern": f"%{category}%", "limit": limit})
            return [dict(row) for row in cur.fetchall()]


# ────────────────────────────────────────────
# 6d. RECORD LOOKUP BY ID  # ADDED
# ────────────────────────────────────────────

def get_record_by_id(record_id: int) -> dict | None:  # ADDED
    """
    Fetch a single cleaned_record by its primary key.
    Returns None if the ID does not exist.
    Useful for context enrichment after a hybrid_search() call.
    """  # ADDED
    sql = "SELECT * FROM cleaned_records WHERE id = %(id)s"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"id": record_id})
            row = cur.fetchone()
            return dict(row) if row else None



# ─────────────────────────────────────────────
# 7. KEYWORD SEARCH  (PostgreSQL full-text search)
# ─────────────────────────────────────────────

def keyword_search(
    query: str,
    field: str = "summary",
    limit: int = 10,
) -> list[dict]:
    """
    Full-text keyword search on a text column (place, summary, etc.).

    Uses PostgreSQL ts_vector / ts_query with english dictionary.
    """
    allowed_fields = {"place", "summary", "geocoded_place", "role"}
    if field not in allowed_fields:
        raise ValueError(f"field must be one of {allowed_fields}")

    sql = f"""
    SELECT
        cr.id,
        cr.place,
        cr.geocoded_place,
        cr.latitude,
        cr.longitude,
        cr.date,
        cr.date_type,
        cr.time,
        cr.role,
        cr.summary,
        cr.source_file,
        cr.geocode_status,
        ts_rank(to_tsvector('english', coalesce(cr.{field}, '')),
                plainto_tsquery('english', %(query)s)) AS rank
    FROM cleaned_records cr
    WHERE to_tsvector('english', coalesce(cr.{field}, ''))
          @@ plainto_tsquery('english', %(query)s)
    ORDER BY rank DESC
    LIMIT %(limit)s
    """

    results: list[dict] = []
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"query": query, "limit": limit})
            results = [dict(row) for row in cur.fetchall()]

    return results


# ─────────────────────────────────────────────
# 8. METADATA LOOKUPS  (exact / filter queries)
# ─────────────────────────────────────────────

def search_by_place(place: str, limit: int = 20) -> list[dict]:
    """Case-insensitive ILIKE search on place and geocoded_place."""
    sql = """
    SELECT * FROM cleaned_records
    WHERE place ILIKE %(pattern)s
       OR geocoded_place ILIKE %(pattern)s
    ORDER BY date NULLS LAST
    LIMIT %(limit)s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"pattern": f"%{place}%", "limit": limit})
            return [dict(row) for row in cur.fetchall()]


def search_by_date(date_str: str, limit: int = 20) -> list[dict]:
    """Exact or prefix match on the date field."""
    sql = """
    SELECT * FROM cleaned_records
    WHERE date = %(date)s OR date LIKE %(prefix)s
    ORDER BY date
    LIMIT %(limit)s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"date": date_str, "prefix": f"{date_str}%", "limit": limit})
            return [dict(row) for row in cur.fetchall()]


def search_by_role(role: str, limit: int = 20) -> list[dict]:
    """Case-insensitive search on role."""
    sql = """
    SELECT * FROM cleaned_records
    WHERE role ILIKE %(pattern)s
    ORDER BY date NULLS LAST
    LIMIT %(limit)s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"pattern": f"%{role}%", "limit": limit})
            return [dict(row) for row in cur.fetchall()]


# ─────────────────────────────────────────────
# 9. HIGH-LEVEL API  (called by main.py / cleaner.py)
# ─────────────────────────────────────────────

def store_cleaned_records(records: list[dict] | None = None) -> dict:
    """
    Top-level entry point for the pipeline.

    If *records* is provided, inserts them directly.
    Otherwise, reads from the cleaned_records.json file on disk.

    Returns the insert summary dict.
    """
    if records is None:
        if not CLEANED_FILE.exists():
            logger.error(f"Cleaned records file not found: {CLEANED_FILE}")
            return {"inserted": 0, "updated": 0, "errors": 0}

        logger.info(f"Loading cleaned records from {CLEANED_FILE}")
        with open(CLEANED_FILE, encoding="utf-8") as f:
            records = json.load(f)

    if not records:
        logger.info("No records to insert.")
        return {"inserted": 0, "updated": 0, "errors": 0}

    logger.info(f"Storing {len(records)} cleaned record(s) into PostgreSQL…")

    ensure_schema()
    summary = insert_records(records)

    print(f"\n  {'-'*40}")
    print(f"  PostgreSQL Storage Summary")
    print(f"  {'-'*40}")
    print(f"  + Inserted   : {summary['inserted']}")
    print(f"  ~ Updated    : {summary['updated']}")
    print(f"  x Errors     : {summary['errors']}")
    print(f"  {'-'*40}")

    return summary


# ─────────────────────────────────────────────
# 10. STANDALONE ENTRY POINT
# ─────────────────────────────────────────────

def main():
    """Run as a standalone script: read cleaned_records.json → insert into PG."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    print("=" * 55)
    print("  Stage 2b: PostgreSQL + pgvector Persistence")
    print("=" * 55)

    try:
        summary = store_cleaned_records()
        if summary["errors"]:
            print(f"\n  ⚠  {summary['errors']} record(s) failed — check logs.")
    except ConnectionError as exc:
        print(f"\n  ✗ Could not connect to PostgreSQL:\n    {exc}")
        print("    Make sure the Docker container is running:")
        print("      docker compose -f Docker/docker-compose.yml up -d")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
