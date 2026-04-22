from src.doc_parser import DocParser
from src.geocoder import Geocoder
from src.neo4j_writer import Neo4jWriter

# 1. Extract from PDF
parser = DocParser()
output, extraction_result = parser.parse("docs\travel_itinerary_journal.pdf")

# 2. Geocode the places
geocoder = Geocoder(backend="nominatim")
geocoded_places = geocoder.enrich(extraction_result)

# 3. Write to Neo4j
writer = Neo4jWriter(uri="bolt://localhost:7687", user="neo4j", password="password")
writer.write(extraction_result, geocoded_places)

# 4. Now check stats
print(writer.stats())
writer.close()