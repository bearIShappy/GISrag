# GIS RAG Pipeline

Extract geospatial entities from survey and field documents, store them in a Neo4j knowledge graph, and answer natural language questions with an LLM that renders answers on a real interactive map.

---

## What It Does

Feed it a PDF or DOCX survey report. It extracts every coordinate, place name, date, and person from the document, geocodes the place names to lat/lon, builds a relationship graph in Neo4j, embeds everything for semantic search, and lets you ask questions like:

> *"Where did the survey team go in April 2026?"*

The answer comes back as text **plus** an interactive Leaflet map with pins, popups, and route lines drawn between locations.

---

## Project Structure

```
gis-rag/
├── src/
│   └── backend/
│       ├── __init__.py
│       ├── doc_parser.py            # Stages 1 + 2 — text extraction + entity extraction
│       ├── graph_builder.py         # GraphRAG-style NetworkX graph (GraphML export)
│       ├── stage3_geocoder.py       # Stage 3 — resolve place names → lat/lon
│       ├── stage4_neo4j_writer.py   # Stage 4 — write entities + relationships to Neo4j
│       ├── stage5_embedder.py       # Stage 5 — embed nodes, store in Neo4j vector index
│       ├── stage6_rag_query.py      # Stage 6 — RAG query: vector search + graph + LLM
│       └── stage7_8_map_renderer.py # Stages 7+8 — structured output → Folium/Leaflet map
├── docs/                            # Put your input PDF/DOCX files here
├── output_map.html                  # Generated map (auto-opened in browser)
├── parsed_output.json               # Extracted entities (intermediate output)
├── graph.graphml                    # GraphML export (open in Gephi/yEd)
├── geocode_cache.json               # Geocoding cache (avoids re-hitting API)
└── README.md
```

---

## Tech Stack

| Layer | Library / Tool | Purpose |
|---|---|---|
| **Document parsing** | `unstructured[pdf,docx]` | PDF + DOCX text extraction, multi-column, tables |
| **OCR fallback** | `pytesseract` + `Pillow` + `PyMuPDF` | Scanned / image-only PDFs |
| **NER** | `spaCy` (`en_core_web_sm`) | Extract place names, persons (GPE, LOC, PERSON) |
| **Coordinate extraction** | `regex` (built-in) | Parse lat/lon in multiple formats from raw text |
| **Date parsing** | `python-dateutil` | Robust date extraction from free text |
| **Geocoding** | `geopy` (Nominatim / Google Maps) | Resolve place names → lat/lon |
| **Graph database** | `Neo4j 5` + `neo4j` Python driver | Store entities, relationships, spatial + vector index |
| **Graph export** | `networkx` | GraphML export for Gephi/yEd visualisation |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local 384-dim embeddings (or OpenAI `text-embedding-3-small`) |
| **Vector search** | Neo4j native vector index | Cosine similarity search over embedded nodes |
| **LLM** | `anthropic` (Claude) or `openai` (GPT-4o) | Answer questions, return structured JSON output |
| **Map rendering** | `folium` (Leaflet.js) | Interactive HTML map with markers, popups, routes |
| **OCR system deps** | `tesseract-ocr`, `poppler-utils` | Required by pytesseract for scanned PDFs |

---

## Installation

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd gis-rag
python -m venv venvGIS
# Windows
venvGIS\Scripts\activate
# Linux / macOS
source venvGIS/bin/activate
```

### 2. Install Python dependencies

```bash
pip install "unstructured[pdf,docx]" pytesseract pillow pymupdf
pip install spacy networkx python-docx python-dateutil
pip install geopy neo4j sentence-transformers folium
pip install anthropic          # if using Claude
pip install openai             # if using OpenAI

python -m spacy download en_core_web_sm
```

### 3. Install system dependencies (for OCR on scanned PDFs)

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows — download installers:
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler:   https://github.com/oschwartz10612/poppler-windows
```

### 4. Start Neo4j

```bash
docker run \
  --name neo4j-gis \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

Neo4j browser available at `http://localhost:7474`

### 5. Set API keys

```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here
set OPENAI_API_KEY=your_key_here         # optional

# Linux / macOS
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here      # optional
```

---

## Pipeline Stages

### Stage 1+2 — Document parsing + entity extraction (`doc_parser.py`)

Reads PDF or DOCX files and extracts three entity types:

- **Coordinates** — regex patterns covering `25.3176° N, 82.9739° E`, `Lat: 25.31, Lon: 82.97`, ISO decimal, and more
- **Places + Persons** — spaCy NER labels `GPE`, `LOC`, `FAC`, `PERSON`
- **Dates** — python-dateutil handles `April 07, 2026`, `2026-04-07`, `07/04/2026`

Outputs `parsed_output.json` and `graph.graphml`.

```bash
# Single file
python -m src.backend.doc_parser --input docs/survey_report.pdf

# Scanned PDF (force OCR on every page)
python -m src.backend.doc_parser --input docs/scanned.pdf --force-ocr

# Batch process a folder
python -m src.backend.doc_parser --input docs/
```

---

### Stage 3 — Geocoding (`stage3_geocoder.py`)

Resolves NER-found place names (e.g. "Varanasi", "Kaziranga") to lat/lon using Nominatim (free, OpenStreetMap) or Google Maps API. Results are cached in `geocode_cache.json` — delete it to force fresh lookups.

```bash
python -m src.backend.stage3_geocoder
```

```python
from src.backend.stage3_geocoder import Geocoder

geocoder = Geocoder(backend="nominatim", country_bias="IN")
result = geocoder.lookup("Kaziranga National Park")
# → GeocodedPlace(lat=26.58, lon=93.17, display_name="Kaziranga, Assam, India")

# Switch to Google for production
geocoder = Geocoder(backend="google", google_api_key="YOUR_KEY")
```

---

### Stage 4 — Neo4j graph store (`stage4_neo4j_writer.py`)

Writes all entities and relationships into Neo4j using `MERGE` (idempotent — safe to re-run). Creates a native spatial `point()` property on every GeoPoint for proximity queries.

**Node labels:** `Document` · `GeoPoint` · `Place` · `Date` · `Person`

**Relationship types:**

| Relationship | Meaning |
|---|---|
| `CONTAINS_LOCATION` | Document → GeoPoint |
| `MENTIONS_DATE` | Document → Date |
| `NEXT_LOCATION` | GeoPoint → GeoPoint (page order) |
| `VISITED_ON` | GeoPoint → Date (same page) |
| `OBSERVED_AT` | Person → GeoPoint (same page) |
| `ROUTE_TO` | Place → Place ("from X to Y" in text) |

```bash
python -m src.backend.stage4_neo4j_writer
```

```python
from src.backend.stage4_neo4j_writer import Neo4jWriter

writer = Neo4jWriter(uri="bolt://localhost:7687", user="neo4j", password="password")
writer.write(extraction_result, geocoded_places)

# Proximity query — find all points within 100km of Varanasi
nearby = writer.find_nearby(25.3176, 82.9739, radius_km=100)
```

---

### Stage 5 — Embeddings (`stage5_embedder.py`)

Embeds each GeoPoint node using a rich natural language context string (name + coordinates + nearby dates + persons + raw text snippet from the document). Stores the vector back into Neo4j's native vector index for semantic search in Stage 6.

Default model: `all-MiniLM-L6-v2` (384-dim, runs fully local, no API needed).

```bash
python -m src.backend.stage5_embedder
```

```python
from src.backend.stage5_embedder import Embedder

embedder = Embedder(backend="minilm")       # local, free
# embedder = Embedder(backend="openai")     # text-embedding-3-small

embedder.embed_all(writer)

# Semantic search
results = embedder.search(writer, "survey sites near the river delta", top_k=5)
```

---

### Stage 6 — RAG query layer (`stage6_rag_query.py`)

The core question-answering engine. For each query it:

1. Embeds the question → vector search → top-k matching GeoPoint nodes
2. Walks 1–2 hops in the graph to collect related dates, persons, routes
3. Packs the subgraph as structured context into the LLM prompt
4. Returns the LLM answer + a `map_points` JSON array ready for Stage 8

```python
from src.backend.stage6_rag_query import RAGQueryEngine

engine = RAGQueryEngine(writer, embedder, llm_backend="claude")
response = engine.query("Where did the survey team go in April 2026?")

print(response["answer"])
# → "The survey team visited three sites in April 2026: Varanasi (April 7)..."

print(response["map_points"])
# → [{"name": "Varanasi", "lat": 25.3176, "lon": 82.9739, "date": "2026-04-07", ...}]
```

For advanced queries you can write Cypher directly:

```python
engine.cypher_search(
    "MATCH (g:GeoPoint)-[:VISITED_ON]->(d:Date {iso: '2026-04-07'}) "
    "RETURN g.name, g.lat, g.lon"
)
```

---

### Stages 7+8 — Map rendering (`stage7_8_map_renderer.py`)

Takes the Stage 6 JSON response and renders a self-contained interactive HTML map using Folium (Leaflet.js). Features:

- Colour-coded markers by location type with rich popups (name, date, coordinates, summary)
- Dashed route polylines with directional arrows
- Marker clustering for dense point sets
- Floating LLM answer panel in the top-left corner
- Auto-opens in browser on save

```python
from src.backend.stage7_8_map_renderer import MapRenderer

renderer = MapRenderer(tile_provider="CartoDB positron")
renderer.render(response, output_path="output_map.html")
```

---

## Full Pipeline — Single Command

```bash
python -m src.backend.map_renderer ^
  --input docs/survey_report.pdf ^
  --question "Where did the survey team go in April 2026?" ^
  --output output_map.html ^
  --llm claude ^
  --neo4j-uri bolt://localhost:7687 ^
  --neo4j-user neo4j ^
  --neo4j-password password
```

Or from Python:

```python
from src.backend.stage7_8_map_renderer import run_full_pipeline

run_full_pipeline(
    input_file="docs/survey_report.pdf",
    question="Where did the survey team go in April 2026?",
    output_map="output_map.html",
    llm_backend="claude",
)
```

---

## Example Queries

```python
engine.query("What locations were surveyed in this document?")
engine.query("Which team members visited sites in Rajasthan?")
engine.query("Show me the route taken between April 5 and April 10.")
engine.query("Are there any survey points within 50km of the river?")
engine.query("Who was the lead surveyor at the Leh site?")
```

---

## Supported Input Formats

| Format | Notes |
|---|---|
| `.pdf` | Digital PDFs (fast) and scanned PDFs via OCR (`--force-ocr`) |
| `.docx` | Word documents, paragraph-by-paragraph extraction |
| `.txt` / `.md` | Plain text, treated as single page |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | If using Claude | Anthropic API key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key (embeddings or LLM) |
| `GOOGLE_API_KEY` | Optional | Google Maps geocoding (production geocoder) |

---

## Troubleshooting

**`unstructured` import error** — make sure you installed the extras:
```bash
pip install "unstructured[pdf,docx]"
```

**spaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**Neo4j connection refused** — check Docker container is running:
```bash
docker ps
docker start neo4j-gis   # if stopped
```

**Nominatim rate limit (429)** — Nominatim enforces 1 request/second. The geocoder already adds a 1.1s delay. If you're processing many documents, switch to the Google backend or increase `rate_limit_delay`.

**Vector index not found** — run Stage 4 first to create the schema, then Stage 5 to populate embeddings before running Stage 6 queries.

**OCR produces garbled text** — try increasing DPI in `_ocr_page_fitz` (change `fitz.Matrix(2.0, 2.0)` to `fitz.Matrix(3.0, 3.0)`) or ensure `tesseract-ocr` is properly installed.