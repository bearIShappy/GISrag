import json
import logging
import os
from typing import List, Dict, Any
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jIngestor:
    """
    Ingests cleaned records into Neo4j with graph relationships and vector embeddings.
    """
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        self.model = SentenceTransformer(Config.EMBED_MODEL_PATH)

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """
        Creates uniqueness constraints and indexes for the graph.
        """
        queries = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX role_name IF NOT EXISTS FOR (r:Role) ON (r.role)",
            "CREATE INDEX location_place IF NOT EXISTS FOR (l:Location) ON (l.place)",
            "CREATE INDEX date_value IF NOT EXISTS FOR (dt:Date) ON (dt.date)"
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
        logger.info("Constraints and indexes created.")

    def create_vector_index(self):
        """
        Creates a vector index for semantic search on Document nodes.
        """
        query = f"""
        CREATE VECTOR INDEX {Config.VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (d:Document) ON (d.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {Config.VECTOR_DIMENSION},
            `vector.similarity_function`: '{Config.SIMILARITY_FUNCTION}'
          }}
        }}
        """
        with self.driver.session() as session:
            session.run(query)
        logger.info(f"Vector index '{Config.VECTOR_INDEX_NAME}' created.")

    def ingest_records(self, records: List[Dict[str, Any]]):
        """
        Batch inserts records into Neo4j using MERGE to avoid duplicates.
        """
        query = """
        UNWIND $batch AS record
        MERGE (d:Document {id: record.id})
        SET d.summary = record.summary,
            d.source_file = record.source_file,
            d.embedding = record.embedding

        MERGE (r:Role {role: record.role})
        MERGE (d)-[:HAS_ROLE]->(r)

        MERGE (l:Location {place: record.place})
        SET l.lat = record.lat,
            l.lon = record.lon
        MERGE (d)-[:LOCATED_AT]->(l)

        MERGE (dt:Date {date: record.date})
        MERGE (d)-[:HAPPENED_ON]->(dt)
        """
        
        # Prepare batch with embeddings
        batch = []
        for record in records:
            embedding = self.model.encode(record['summary']).tolist()
            record_with_embedding = record.copy()
            record_with_embedding['embedding'] = embedding
            batch.append(record_with_embedding)

        with self.driver.session() as session:
            session.run(query, batch=batch)
        logger.info(f"Ingested {len(records)} records into Neo4j.")

def main():
    try:
        path = Config.CLEANED_RECORDS_PATH
        if not os.path.exists(path):
            # Fallback to current directory if defined path doesn't exist
            path = "cleaned_records.json"
            
        with open(path, 'r') as f:
            records = json.load(f)
        
        ingestor = Neo4jIngestor()
        ingestor.create_constraints()
        ingestor.create_vector_index()
        ingestor.ingest_records(records)
        ingestor.close()
        logger.info("Ingestion process completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")

if __name__ == "__main__":
    main()
