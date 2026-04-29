import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphQuerier:
    """
    Handles Cypher queries for searching the Neo4j graph.
    """
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def _execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query and returns the results.
        """
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def search_place(self, place: str) -> List[Dict[str, Any]]:
        """
        Searches for documents located at a specific place.
        """
        query = """
        MATCH (l:Location {place: $place})<-[:LOCATED_AT]-(d:Document)
        MATCH (d)-[:HAS_ROLE]->(r:Role)
        MATCH (d)-[:HAPPENED_ON]->(dt:Date)
        RETURN d.id AS id, d.summary AS summary, l.place AS place, l.lat AS lat, l.lon AS lon, dt.date AS date, r.role AS role
        """
        return self._execute_query(query, {"place": place})

    def search_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Searches for documents that happened on a specific date.
        """
        query = """
        MATCH (dt:Date {date: $date})<-[:HAPPENED_ON]-(d:Document)
        MATCH (d)-[:LOCATED_AT]->(l:Location)
        MATCH (d)-[:HAS_ROLE]->(r:Role)
        RETURN d.id AS id, d.summary AS summary, l.place AS place, l.lat AS lat, l.lon AS lon, dt.date AS date, r.role AS role
        """
        return self._execute_query(query, {"date": date})

    def search_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Searches for documents involving a specific role.
        """
        query = """
        MATCH (r:Role {role: $role})<-[:HAS_ROLE]-(d:Document)
        MATCH (d)-[:LOCATED_AT]->(l:Location)
        MATCH (d)-[:HAPPENED_ON]->(dt:Date)
        RETURN d.id AS id, d.summary AS summary, l.place AS place, l.lat AS lat, l.lon AS lon, dt.date AS date, r.role AS role
        """
        return self._execute_query(query, {"role": role})

    def search_summary(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Searches for documents with a keyword in their summary.
        """
        query = """
        MATCH (d:Document)
        WHERE d.summary CONTAINS $keyword
        MATCH (d)-[:LOCATED_AT]->(l:Location)
        MATCH (d)-[:HAPPENED_ON]->(dt:Date)
        MATCH (d)-[:HAS_ROLE]->(r:Role)
        RETURN d.id AS id, d.summary AS summary, l.place AS place, l.lat AS lat, l.lon AS lon, dt.date AS date, r.role AS role
        """
        return self._execute_query(query, {"keyword": keyword})

    def search_hybrid(self, place: str = None, date_str: str = None, role: str = None) -> List[Dict[str, Any]]:
        """
        Search by combining multiple entities (Location + Date + Role).
        """
        conditions = []
        params = {}
        
        if place:
            conditions.append("(d)-[:LOCATED_AT]->(:Location {place: $place})")
            params["place"] = place
        if date_str:
            conditions.append("(d)-[:HAPPENED_ON]->(:Date {date: $date})")
            params["date"] = date_str
        if role:
            conditions.append("(d)-[:HAS_ROLE]->(:Role {role: $role})")
            params["role"] = role

        if not conditions:
            return []

        match_clause = "MATCH (d:Document), " + ", ".join(conditions)
        query = f"""
        {match_clause}
        MATCH (d)-[:LOCATED_AT]->(l:Location)
        MATCH (d)-[:HAPPENED_ON]->(dt:Date)
        MATCH (d)-[:HAS_ROLE]->(r:Role)
        RETURN d.id AS id, d.summary AS summary, l.place AS place, l.lat AS lat, l.lon AS lon, dt.date AS date, r.role AS role
        """
        return self._execute_query(query, params)
