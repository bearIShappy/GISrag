import logging
from .doc_parser import process_documents
from .cleaner import main as run_cleaner
from .ingest import main as run_ingest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Runs the full GraphRAG ingestion pipeline:
    1. Document Parsing (Docling + GLiNER2)
    2. Data Cleaning & Normalization (Geocoding)
    3. Neo4j Ingestion (Embeddings + Graph)
    """
    logger.info("Starting Full GraphRAG Pipeline...")
    
    try:
        # Step 1: Parse Documents
        logger.info("Step 1: Parsing documents...")
        process_documents()
        
        # Step 2: Clean Data
        logger.info("Step 2: Cleaning and normalizing data...")
        run_cleaner()
        
        # Step 3: Ingest into Neo4j
        logger.info("Step 3: Ingesting records into Neo4j...")
        run_ingest()
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    run_pipeline()