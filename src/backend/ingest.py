# """
# ingest.py — Stage 2: Graph + Vector Ingestion into Neo4j
# ---------------------------------------------------------
# Input:  cleaned_records.json  (output of cleaner.py)
# Output: Neo4j graph with Document, Location, Date, Role nodes
#         + vector embeddings on Document for semantic search

# Changes from the original vs the cleaner.py pipeline
# ─────────────────────────────────────────────────────
# 1. FIELD NAMES FIXED
#    cleaner writes 'latitude'/'longitude'; old code read 'lat'/'lon' → KeyError.
#    Now reads the correct field names.

# 2. STABLE DOCUMENT ID GENERATED
#    cleaner never writes an 'id' field; old MERGE on record.id=None collapsed
#    all documents onto a single node. ID is now a deterministic hash of
#    (source_file + date + place + latitude) so re-runs are idempotent.

# 3. NULL SUMMARY GUARD
#    model.encode(None) raises TypeError and crashes the whole batch.
#    Embedding is built from a fallback text when summary is null:
#    "{place} {date} {role}" — always a non-empty string.

# 4. NULL ROLE / NULL DATE — NO ORPHAN NODES
#    MERGE (r:Role {role: null}) and MERGE (dt:Date {date: null}) create
#    garbage nodes that pollute every future query. Relationships are now
#    only created when the value is non-null (FOREACH trick).

# 5. GEOCODE_STATUS CONFIDENCE FILTER
#    Records with geocode_status "mismatch" are skipped — they belong in
#    quarantine.json, not the graph. Records with partial/uncertain status
#    are ingested but flagged on the Location node so queries can filter
#    by confidence.

# 6. ALL CLEANER FIELDS STORED
#    New fields written to the graph:
#      Document  : time, date_type, date_raw, geocode_status,
#                  place_coord_validated, place_coord_distance_km
#      Location  : geocoded_place, geocode_status
#    Nothing from the pipeline is silently discarded.

# 7. BATCHED EMBEDDING + PROGRESS LOGGING
#    Embeddings are generated in configurable batches (default 32) with
#    per-batch progress so long runs don't look frozen.
# """
# import os
# import hashlib
# import json
# import logging
# import os
# from typing import List, Dict, Any

# from neo4j import GraphDatabase
# from sentence_transformers import SentenceTransformer


# from .config import Config

# # ─────────────────────────────────────────────
# # LOGGING
# # ─────────────────────────────────────────────

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # ─────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────

# # geocode_status values whose records must NOT enter the graph
# SKIP_STATUSES = {"mismatch"}

# # geocode_status values where coords exist but confidence is reduced
# WARN_STATUSES = {"partial", "reverse_failed", "failed", "forward"}

# # EMBED_DIM sourced from settings.text_embedding.VECTOR_DIM
# EMBED_BATCH_SIZE = os.getenv("EMBED_BATCH_SIZE")
# EMBED_MODEL_PATH = settings.text_embedding.MODEL_PATH

# # ─────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────

# def make_document_id(record: Dict[str, Any]) -> str:
#     """
#     Deterministic document ID from stable fields.

#     Using a hash means:
#       - Re-running ingest on the same data is fully idempotent (MERGE no-ops).
#       - No two records from different source files / dates / places collide,
#         even when summary is null.
#     """
#     key = "|".join([
#         str(record.get("source_file") or ""),
#         str(record.get("date")        or ""),
#         str(record.get("place")       or ""),
#         str(record.get("latitude")    or ""),
#     ])
#     return hashlib.sha256(key.encode()).hexdigest()[:16]


# def build_embed_text(record: Dict[str, Any]) -> str:
#     """
#     Construct the text to embed for a record.

#     Priority:
#       1. summary  (richest semantic content)
#       2. Fallback: place / geocoded_place / date / role concatenated —
#          so the embedding is never built from an empty string.
#     """
#     if record.get("summary"):
#         return record["summary"]

#     parts = [
#         record.get("place") or record.get("geocoded_place"),
#         record.get("date_raw") or record.get("date"),
#         record.get("role"),
#         record.get("source_file"),
#     ]
#     fallback = " ".join(str(p) for p in parts if p)
#     return fallback if fallback.strip() else "unknown record"


# def prepare_batch(
#     records: List[Dict[str, Any]],
#     model: SentenceTransformer,
# ) -> List[Dict[str, Any]]:
#     """
#     Filter, enrich each record with a stable id and a vector embedding.

#     Skips records whose geocode_status is in SKIP_STATUSES (mismatch).
#     Logs a warning for WARN_STATUSES records that are included with
#     reduced-confidence coordinates.
#     """
#     accepted: List[Dict[str, Any]] = []
#     skipped = 0

#     for rec in records:
#         status = rec.get("geocode_status", "exact")

#         if status in SKIP_STATUSES:
#             logger.warning(
#                 "Skipping record (geocode_status=%s): place=%s date=%s source=%s",
#                 status, rec.get("place"), rec.get("date"), rec.get("source_file"),
#             )
#             skipped += 1
#             continue

#         if status in WARN_STATUSES:
#             logger.warning(
#                 "Low-confidence coords (geocode_status=%s): place=%s source=%s",
#                 status,
#                 rec.get("place") or rec.get("geocoded_place"),
#                 rec.get("source_file"),
#             )

#         accepted.append(rec)

#     if skipped:
#         logger.info("Skipped %d record(s) with status in %s.", skipped, SKIP_STATUSES)

#     # Batch-encode all embed texts at once (much faster than one-by-one)
#     texts = [build_embed_text(r) for r in accepted]
#     for i in range(0, len(texts), EMBED_BATCH_SIZE):
#         batch_texts = texts[i : i + EMBED_BATCH_SIZE]
#         embeddings = model.encode(batch_texts, show_progress_bar=False)
#         for j, emb in enumerate(embeddings):
#             idx = i + j
#             accepted[idx] = {
#                 **accepted[idx],
#                 "id":        make_document_id(accepted[idx]),
#                 "embedding": emb.tolist(),
#             }
#         logger.info(
#             "Embedded records %d-%d / %d",
#             i + 1, min(i + EMBED_BATCH_SIZE, len(accepted)), len(accepted),
#         )

#     return accepted


# # ─────────────────────────────────────────────
# # NEO4J INGESTOR
# # ─────────────────────────────────────────────

# class Neo4jIngestor:
#     """
#     Ingests cleaned records into Neo4j with graph relationships and
#     vector embeddings.
#     """

#     def __init__(self):
#         self.driver = GraphDatabase.driver(
#             Config.NEO4J_URI,
#             auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
#         )
#         self.model = SentenceTransformer(EMBED_MODEL_PATH)

#     def close(self):
#         self.driver.close()

#     # ── Schema setup ──────────────────────────────────────────────────────

#     def create_constraints(self):
#         """Uniqueness constraints and lookup indexes."""
#         queries = [
#             # Document uniqueness — stable hash id prevents duplicate nodes
#             "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
#             # Lookup indexes
#             "CREATE INDEX role_name      IF NOT EXISTS FOR (r:Role)     ON (r.role)",
#             "CREATE INDEX location_place IF NOT EXISTS FOR (l:Location) ON (l.place)",
#             "CREATE INDEX date_value     IF NOT EXISTS FOR (dt:Date)    ON (dt.date)",
#         ]
#         with self.driver.session() as session:
#             for query in queries:
#                 session.run(query)
#         logger.info("Constraints and indexes created.")

#     def create_vector_index(self):
#         """Vector index for semantic similarity search on Document.embedding."""
#         query = f"""
#         CREATE VECTOR INDEX {settings.neo4j.VECTOR_INDEX_NAME} IF NOT EXISTS
#         FOR (d:Document) ON (d.embedding)
#         OPTIONS {{
#           indexConfig: {{
#             `vector.dimensions`:          {settings.neo4j.VECTOR_DIMENSION},
#             `vector.similarity_function`: '{settings.neo4j.SIMILARITY_FUNCTION}'
#           }}
#         }}
#         """
#         with self.driver.session() as session:
#             session.run(query)
#         logger.info("Vector index '%s' created.", settings.neo4j.VECTOR_INDEX_NAME)

#     # ── Ingestion ─────────────────────────────────────────────────────────

#     def ingest_records(self, records: List[Dict[str, Any]]):
#         """
#         Prepare embeddings then write to Neo4j in one UNWIND batch.

#         Graph schema written:
#           (:Document)-[:LOCATED_AT]->(:Location)
#           (:Document)-[:HAPPENED_ON]->(:Date)     <- only when date non-null
#           (:Document)-[:HAS_ROLE]->(:Role)         <- only when role non-null

#         All cleaner.py fields are preserved on the nodes they belong to.
#         Nothing is silently dropped.
#         """
#         if not records:
#             logger.warning("ingest_records called with empty list — nothing to do.")
#             return

#         batch = prepare_batch(records, self.model)
#         if not batch:
#             logger.warning("All records were filtered out. Nothing ingested.")
#             return

#         # FOREACH (x IN CASE WHEN cond THEN [1] ELSE [] END | ...)
#         # is the idiomatic Neo4j way to conditionally execute a MERGE
#         # inside an UNWIND batch without breaking the pipeline.
#         query = """
#         UNWIND $batch AS rec

#         // ── Document node ────────────────────────────────────────────
#         MERGE (d:Document {id: rec.id})
#         SET d.summary                 = rec.summary,
#             d.source_file             = rec.source_file,
#             d.time                    = rec.time,
#             d.date_type               = rec.date_type,
#             d.date_raw                = rec.date_raw,
#             d.geocode_status          = rec.geocode_status,
#             d.place_coord_validated   = rec.place_coord_validated,
#             d.place_coord_distance_km = rec.place_coord_distance_km,
#             d.embedding               = rec.embedding

#         // ── Location node ─────────────────────────────────────────────
#         // Use extracted place name when available; fall back to the
#         // reverse-geocoded label. Never MERGE on null — use a sentinel.
#         WITH d, rec,
#              coalesce(rec.place, rec.geocoded_place, '__unknown__') AS loc_key
#         MERGE (l:Location {place: loc_key})
#         SET l.latitude       = rec.latitude,
#             l.longitude      = rec.longitude,
#             l.geocoded_place = rec.geocoded_place,
#             l.geocode_status = rec.geocode_status
#         MERGE (d)-[:LOCATED_AT]->(l)

#         // ── Date node — only when date is non-null ────────────────────
#         WITH d, rec
#         FOREACH (x IN CASE WHEN rec.date IS NOT NULL THEN [1] ELSE [] END |
#             MERGE (dt:Date {date: rec.date})
#             SET dt.date_type = rec.date_type
#             MERGE (d)-[:HAPPENED_ON]->(dt)
#         )

#         // ── Role node — only when role is non-null ────────────────────
#         WITH d, rec
#         FOREACH (x IN CASE WHEN rec.role IS NOT NULL THEN [1] ELSE [] END |
#             MERGE (r:Role {role: rec.role})
#             MERGE (d)-[:HAS_ROLE]->(r)
#         )
#         """

#         with self.driver.session() as session:
#             session.run(query, batch=batch)

#         logger.info("Ingested %d document(s) into Neo4j.", len(batch))


# # ─────────────────────────────────────────────
# # ENTRY POINT
# # ─────────────────────────────────────────────

# def main():
#     path = Config.CLEANED_RECORDS_PATH
#     try:
#         if not os.path.exists(path):
#             path = "cleaned_records.json"

#         logger.info("Loading records from: %s", path)
#         with open(path, "r", encoding="utf-8") as f:
#             records = json.load(f)
#         logger.info("Loaded %d record(s).", len(records))

#         ingestor = Neo4jIngestor()
#         ingestor.create_constraints()
#         ingestor.create_vector_index()
#         ingestor.ingest_records(records)
#         ingestor.close()

#         logger.info("Ingestion process completed successfully.")

#     except FileNotFoundError:
#         logger.error(
#             "cleaned_records.json not found at '%s'. Run cleaner.py first.", path
#         )
#     except json.JSONDecodeError as e:
#         logger.error("Failed to parse cleaned_records.json: %s", e)
#     except Exception as e:
#         logger.exception("Ingestion failed: %s", e)


# if __name__ == "__main__":
#     main()
"""
ingest.py — Stage 2: Graph + Vector Ingestion into Neo4j
---------------------------------------------------------
Input:  cleaned_records.json  (output of cleaner.py)
Output: Neo4j graph with Document, Location, Date, Role nodes
        + vector embeddings on Document for semantic search
"""
import os
import hashlib
import json
import logging
from typing import List, Dict, Any

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ADDED: Load environment variables so os.getenv actually works
from dotenv import load_dotenv
load_dotenv()

from src.config.settings import settings

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS & ENV VARIABLES
# ─────────────────────────────────────────────

# geocode_status values whose records must NOT enter the graph
SKIP_STATUSES = {"mismatch"}

# geocode_status values where coords exist but confidence is reduced
WARN_STATUSES = {"partial", "reverse_failed", "failed", "forward"}

# ADDED FALLBACKS: This prevents the Pooling TypeError if your .env is missing
# EMBED_DIM sourced from settings.text_embedding.VECTOR_DIM
EMBED_BATCH_SIZE = settings.text_embedding.BATCH_SIZE
EMBED_MODEL_PATH = settings.text_embedding.MODEL_PATH

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_document_id(record: Dict[str, Any]) -> str:
    """
    Deterministic document ID from stable fields.
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
    Construct the text to embed for a record.
    """
    if record.get("summary"):
        return record["summary"]

    parts = [
        record.get("place") or record.get("geocoded_place"),
        record.get("date_raw") or record.get("date"),
        record.get("role"),
        record.get("source_file"),
    ]
    fallback = " ".join(str(p) for p in parts if p)
    return fallback if fallback.strip() else "unknown record"


def prepare_batch(
    records: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """
    Filter, enrich each record with a stable id and a vector embedding.
    """
    accepted: List[Dict[str, Any]] = []
    skipped = 0

    for rec in records:
        status = rec.get("geocode_status", "exact")

        if status in SKIP_STATUSES:
            logger.warning(
                "Skipping record (geocode_status=%s): place=%s date=%s source=%s",
                status, rec.get("place"), rec.get("date"), rec.get("source_file"),
            )
            skipped += 1
            continue

        if status in WARN_STATUSES:
            logger.warning(
                "Low-confidence coords (geocode_status=%s): place=%s source=%s",
                status,
                rec.get("place") or rec.get("geocoded_place"),
                rec.get("source_file"),
            )

        accepted.append(rec)

    if skipped:
        logger.info("Skipped %d record(s) with status in %s.", skipped, SKIP_STATUSES)

    # Batch-encode all embed texts at once
    texts = [build_embed_text(r) for r in accepted]
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i : i + EMBED_BATCH_SIZE]
        
        # This is where sentence-transformers generates the vectors
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        
        for j, emb in enumerate(embeddings):
            idx = i + j
            accepted[idx] = {
                **accepted[idx],
                "id":        make_document_id(accepted[idx]),
                "embedding": emb.tolist(),
            }
        logger.info(
            "Embedded records %d-%d / %d",
            i + 1, min(i + EMBED_BATCH_SIZE, len(accepted)), len(accepted),
        )

    return accepted


# ─────────────────────────────────────────────
# NEO4J INGESTOR
# ─────────────────────────────────────────────

class Neo4jIngestor:
    """
    Ingests cleaned records into Neo4j with graph relationships and
    vector embeddings.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j.URI,
            auth=(settings.neo4j.USER, settings.neo4j.PASSWORD),
        )
        # Initializes the MX-Embed model via sentence-transformers
        logger.info(f"Loading embedding model: {EMBED_MODEL_PATH}")
        self.model = SentenceTransformer(EMBED_MODEL_PATH)

    def close(self):
        self.driver.close()

    # ── Schema setup ──────────────────────────────────────────────────────

    def create_constraints(self):
        """Uniqueness constraints and lookup indexes."""
        queries = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX role_name      IF NOT EXISTS FOR (r:Role)     ON (r.role)",
            "CREATE INDEX location_place IF NOT EXISTS FOR (l:Location) ON (l.place)",
            "CREATE INDEX date_value     IF NOT EXISTS FOR (dt:Date)    ON (dt.date)",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
        logger.info("Constraints and indexes created.")

    def create_vector_index(self):
        """Vector index for semantic similarity search on Document.embedding."""
        query = f"""
        CREATE VECTOR INDEX {settings.neo4j.VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (d:Document) ON (d.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`:          {settings.neo4j.VECTOR_DIMENSION},
            `vector.similarity_function`: '{settings.neo4j.SIMILARITY_FUNCTION}'
          }}
        }}
        """
        with self.driver.session() as session:
            session.run(query)
        logger.info("Vector index '%s' created.", settings.neo4j.VECTOR_INDEX_NAME)

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest_records(self, records: List[Dict[str, Any]]):
        """
        Prepare embeddings then write to Neo4j in one UNWIND batch.
        """
        if not records:
            logger.warning("ingest_records called with empty list — nothing to do.")
            return

        batch = prepare_batch(records, self.model)
        if not batch:
            logger.warning("All records were filtered out. Nothing ingested.")
            return

        query = """
        UNWIND $batch AS rec

        // ── Document node ────────────────────────────────────────────
        MERGE (d:Document {id: rec.id})
        SET d.summary                 = rec.summary,
            d.source_file             = rec.source_file,
            d.time                    = rec.time,
            d.date_type               = rec.date_type,
            d.date_raw                = rec.date_raw,
            d.geocode_status          = rec.geocode_status,
            d.place_coord_validated   = rec.place_coord_validated,
            d.place_coord_distance_km = rec.place_coord_distance_km,
            d.embedding               = rec.embedding

        // ── Location node ─────────────────────────────────────────────
        WITH d, rec,
             coalesce(rec.place, rec.geocoded_place, '__unknown__') AS loc_key
        MERGE (l:Location {place: loc_key})
        SET l.latitude       = rec.latitude,
            l.longitude      = rec.longitude,
            l.geocoded_place = rec.geocoded_place,
            l.geocode_status = rec.geocode_status
        MERGE (d)-[:LOCATED_AT]->(l)

        // ── Date node — only when date is non-null ────────────────────
        WITH d, rec
        FOREACH (x IN CASE WHEN rec.date IS NOT NULL THEN [1] ELSE [] END |
            MERGE (dt:Date {date: rec.date})
            SET dt.date_type = rec.date_type
            MERGE (d)-[:HAPPENED_ON]->(dt)
        )

        // ── Role node — only when role is non-null ────────────────────
        WITH d, rec
        FOREACH (x IN CASE WHEN rec.role IS NOT NULL THEN [1] ELSE [] END |
            MERGE (r:Role {role: rec.role})
            MERGE (d)-[:HAS_ROLE]->(r)
        )
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        logger.info("Ingested %d document(s) into Neo4j.", len(batch))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    path = settings.cleaned_records_path
    try:
        if not os.path.exists(path):
            path = "cleaned_records.json"

        logger.info("Loading records from: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        logger.info("Loaded %d record(s).", len(records))

        ingestor = Neo4jIngestor()
        ingestor.create_constraints()
        ingestor.create_vector_index()
        ingestor.ingest_records(records)
        ingestor.close()

        logger.info("Ingestion process completed successfully.")

    except FileNotFoundError:
        logger.error(
            "cleaned_records.json not found at '%s'. Run cleaner.py first.", path
        )
    except json.JSONDecodeError as e:
        logger.error("Failed to parse cleaned_records.json: %s", e)
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)


if __name__ == "__main__":
    main()