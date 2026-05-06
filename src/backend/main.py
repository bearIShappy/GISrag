"""
main.py -- GISrag Pipeline Orchestrator
-----------------------------------------
Runs the full ingestion + RAG pipeline in sequence.

Stages:
  1. doc_parser   -> Parse PDFs via Docling + GLiNER2
  2. cleaner      -> Normalize, geocode, sanity-check
  3. db_store     -> Upsert cleaned records into PostgreSQL
  4. ingest       -> Generate embeddings -> store via pgvector
  5. rag_pipeline -> Query interface (optional interactive mode)
"""

import sys
import logging
from .doc_parser import process_documents
from .cleaner import main as run_cleaner
from .db_store import store_cleaned_records, close_pool as close_pg
from .ingest import main as run_ingest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline():
    """
    Runs the full GISrag ingestion pipeline:
      1. Document Parsing    (Docling + GLiNER2)
      2. Data Cleaning       (Geocoding + Normalization)
      3. PostgreSQL Storage  (Cleaned records -> DB)
      4. Embedding Ingestion (SentenceTransformer -> pgvector)
    """
    logger.info("Starting Full GISrag Pipeline...")

    try:
        # Step 1: Parse Documents
        logger.info("Step 1/4: Parsing documents...")
        process_documents()

        # Step 2: Clean Data
        logger.info("Step 2/4: Cleaning and normalizing data...")
        run_cleaner()

        # Step 3: Store cleaned records in PostgreSQL
        logger.info("Step 3/4: Storing cleaned records in PostgreSQL...")
        store_cleaned_records()

        # Step 4: Generate embeddings and ingest into pgvector
        logger.info("Step 4/4: Embedding + pgvector ingestion...")
        run_ingest()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        close_pg()


def run_query():
    """
    Launch the interactive RAG query CLI.
    Useful after ingestion is complete.
    """
    from .rag_pipeline import main as rag_main
    rag_main()


if __name__ == "__main__":
    # Support:  python -m src.backend.main          -> full pipeline
    #           python -m src.backend.main query     -> interactive RAG
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        run_query()
    else:
        run_pipeline()