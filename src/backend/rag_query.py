import json
import logging
import requests
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from .config import Config
from .graph_query import GraphQuerier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGQuerySystem:
    """
    Hybrid GraphRAG system combining Neo4j graph traversal and semantic search.
    """
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        self.embed_model = Config.EMBED_MODEL_PATH
        self.graph_querier = GraphQuerier()

    def close(self):
        self.driver.close()
        self.graph_querier.close()

    def extract_entities(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extracts entities (place, date, role) from the user query using a lightweight LLM call or simple regex.
        For this implementation, we'll use a prompt to Ollama to extract entities in JSON.
        """
        prompt = f"""
        Extract entities from the following user query: "{query}"
        Return a JSON object with keys: "place", "date", "role". 
        If an entity is not mentioned, set its value to null.
        Example output: {{"place": "Delhi", "date": "2024-03-14", "role": "Officer"}}
        Output raw JSON only.
        """
        try:
            response = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": Config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            return json.loads(response.json()['response'])
        except Exception as e:
            logger.warning(f"Entity extraction failed: {str(e)}. Proceeding without entities.")
            return {"place": None, "date": None, "role": None}

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search using Neo4j vector index.
        """
        query_embedding = self.embed_model.encode(query).tolist()
        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        MATCH (node)-[:LOCATED_AT]->(l:Location)
        MATCH (node)-[:HAPPENED_ON]->(dt:Date)
        MATCH (node)-[:HAS_ROLE]->(r:Role)
        RETURN node.id AS id, node.summary AS summary, l.place AS place, l.lat AS lat, l.lon AS lon, dt.date AS date, r.role AS role, score
        """
        with self.driver.session() as session:
            result = session.run(cypher, {
                "index_name": Config.VECTOR_INDEX_NAME,
                "limit": limit,
                "embedding": query_embedding
            })
            return [record.data() for record in result]

    def query_rag(self, user_query: str) -> Dict[str, Any]:
        """
        Main RAG flow: extract entities, retrieve context, generate answer.
        """
        # 1. Extract entities
        entities = self.extract_entities(user_query)
        logger.info(f"Extracted entities: {entities}")

        # 2. Retrieve Graph Context (Hybrid: Graph Traversal + Semantic Search)
        graph_context = self.graph_querier.search_hybrid(
            place=entities.get("place"),
            date_str=entities.get("date"),
            role=entities.get("role")
        )
        
        semantic_context = self.semantic_search(user_query)
        
        # Combine and deduplicate context
        combined_context = {item['id']: item for item in graph_context + semantic_context}.values()
        
        if not combined_context:
            return {"answer": "I couldn't find any relevant records in the graph.", "map_points": []}

        # 3. Build LLM Context
        context_str = "\n".join([
            f"- Document {c['id']}: {c['summary']} at {c['place']} on {c['date']} (Role: {c['role']})"
            for c in combined_context
        ])

        system_prompt = """
        You are a GraphRAG assistant. Use the provided graph context to answer the user query.
        - Use ONLY the provided graph context.
        - Preserve relationships and chronology.
        - Never hallucinate facts.
        - Include all relevant locations in your response.
        - Always return your response in JSON format with 'answer' and 'map_points'.
        """

        prompt = f"""
        User Query: {user_query}

        Graph Context:
        {context_str}

        Return a JSON response in this format:
        {{
            "answer": "Detailed answer explaining the findings...",
            "map_points": [
                {{
                    "lat": float,
                    "lon": float,
                    "place": "string",
                    "summary": "string",
                    "date": "string"
                }}
            ]
        }}
        """

        # 4. Generate Answer via Ollama
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": False,
                "format": "json"
            }
        )
        
        try:
            return json.loads(response.json()['response'])
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return {"answer": "Error generating answer.", "map_points": []}

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"your query\"")
        return

    query = " ".join(sys.argv[1:])
    rag = RAGQuerySystem()
    result = rag.query_rag(query)
    print(json.dumps(result, indent=2))
    rag.close()

if __name__ == "__main__":
    main()
