"""
A geospatial document intelligence pipeline
— it reads PDFs/DOCX files and extracts where, when, and who from them, then builds a knowledge graph of their relationships.
— this tool automatically pulls out every GPS coordinate, date, and person mentioned, then maps how they connect.

Real world use cases include:
An archaeological survey report, a field inspection PDF, or a travel log.

The 3-Layer Architecture
Layer 1 — Text Extraction (TextExtractor)
unstructured — handles messy real-world docs, multi-column layouts, and scanned PDFs via OCR (pytesseract)

Layer 2 — Entity Extraction (EntityExtractor)
Three complementary strategies run in parallel:
Regex for coordinates — catches formats like 25.3176° N, 82.9739° E or Lat: 25.31, Lon: 82.97
spaCy NER for place names (GPE, LOC) and persons (PERSON) — contextual, not just pattern matching
python-dateutil for dates — handles April 07, 2026, ISO dates, 07/04/2026, etc.

Layer 3 — Graph Building (GraphBuilder, GraphRAG-style)
Builds a directed knowledge graph with typed nodes and edges:
Edge TypeMeaningCONTAINS_LOCATIONDocument → CoordinateNEXT_LOCATIONCoord A → Coord B (page order = travel order)VISITED_ONCoordinate ↔ Date (same page = same visit)OBSERVED_ATPerson → Coordinate (same page)ROUTE_TOPlace → Place (regex detects "from X to Y" phrases)
The graph exports to GraphML, which tools like Gephi, Neo4j, or yEd can visualize and query.

One Subtle Design Choice Worth Noting
The _enrich_coords_with_places method links place names to coordinates by character proximity in the raw text — if a coordinate appears within 500 characters of a place name, they get associated. Simple but effective for structured reports.

Supports:
  - Digital PDFs, scanned PDFs (via OCR fallback), DOCX, plain text
  - Messy/real-world documents via `unstructured` library
  - Entity extraction: coordinates, place names, dates, people
  - Relationship inference: who-was-where-when, place-to-place routes, temporal chains
  - Output: JSON (entities + graph), GraphML, optional Neo4j export

Requirements (install all):
    pip install "unstructured[pdf,docx]" pytesseract pillow
    pip install spacy networkx python-docx pymupdf
    pip install python-dateutil
    python -m spacy download en_core_web_sm

For scanned PDFs (OCR):
    sudo apt-get install tesseract-ocr poppler-utils   # Linux
    brew install tesseract poppler                      # macOS

Usage:
    python doc_parser.py --input file.pdf
    python doc_parser.py --input file.pdf --output results.json --graph graph.graphml
    python doc_parser.py --input file.pdf --force-ocr        # always OCR (scanned docs)
    python doc_parser.py --input /path/to/folder/            # batch process folder
"""

import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict, field
from src.backend.graph_builder import GraphBuilder

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class GeoPoint:
    lat: float
    lon: float
    place_name: Optional[str] = None
    raw_text: Optional[str] = None     # surrounding text snippet
    page: Optional[int] = None
    source_file: Optional[str] = None


@dataclass
class DateMention:
    raw: str                           # original string e.g. "April 07, 2026"
    parsed: Optional[str] = None      # ISO format: "2026-04-07"
    page: Optional[int] = None
    context: Optional[str] = None     # surrounding sentence


@dataclass
class Person:
    name: str
    role: Optional[str] = None        # e.g. "Lead Surveyor"
    page: Optional[int] = None


@dataclass
class ExtractionResult:
    source_file: str
    geo_points: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    persons: list = field(default_factory=list)
    raw_text_by_page: dict = field(default_factory=dict)
    relationships: list = field(default_factory=list)


# ─────────────────────────────────────────────
# TEXT EXTRACTION LAYER
# ─────────────────────────────────────────────

class TextExtractor:
    """
    Extracts raw text from PDF/DOCX using unstructured (primary) with OCR fallback.
    Returns dict: {page_number: text_string}

    For scanned PDFs, unstructured auto-invokes pytesseract OCR when it detects
    no extractable text. Set force_ocr=True to always OCR every page.
    """

    def __init__(self, force_ocr: bool = False):
        self.force_ocr = force_ocr

    def extract(self, filepath: str) -> dict:
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".docx":
            return self._extract_docx(filepath)
        elif suffix == ".pdf":
            return self._extract_pdf_unstructured(filepath)
        elif suffix in (".txt", ".md"):
            return {0: path.read_text(encoding="utf-8", errors="replace")}
        else:
            log.warning(f"Unknown file type: {suffix}. Attempting raw text read.")
            return {0: path.read_text(encoding="utf-8", errors="replace")}

    def _extract_pdf_unstructured(self, filepath: str) -> dict:
        """
        Uses `unstructured` — best for messy/real-world PDFs.
        Handles: multi-column, headers/footers, tables, rotated text, scanned (OCR).

        strategy options:
          "auto"     — detects whether to use fast or OCR (recommended default)
          "hi_res"   — uses detectron2 for layout detection, best for messy/complex
          "ocr_only" — always OCR (use with force_ocr=True)
          "fast"     — digital text only, fastest
        """
        try:
            from unstructured.partition.pdf import partition_pdf

            strategy = "ocr_only" if self.force_ocr else "auto"
            elements = partition_pdf(
                filename=filepath,
                strategy=strategy,
                languages=["eng"],          # add "hin" for Hindi documents
                include_page_breaks=True,
            )

            pages = {}
            current_page = 0
            for el in elements:
                page_num = getattr(el.metadata, "page_number", current_page) or current_page
                if hasattr(el, "text") and el.text:
                    pages.setdefault(page_num, []).append(el.text)

            return {p: "\n".join(chunks) for p, chunks in pages.items()}

        except ImportError:
            log.error(
                "unstructured not installed. Run: pip install \"unstructured[pdf,docx]\"\n"
                "For OCR support also install: pytesseract pillow\n"
                "And system deps: sudo apt-get install tesseract-ocr poppler-utils"
            )
            return {}

    def _ocr_page_fitz(self, filepath: str, page_index: int) -> str:
        """OCR a single page using PyMuPDF + pytesseract (used as OCR fallback)."""
        try:
            import fitz
            import pytesseract
            from PIL import Image
            import io

            doc = fitz.open(filepath)
            page = doc[page_index]
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom ≈ 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang="eng")
        except Exception as e:
            log.warning(f"OCR fallback failed for page {page_index}: {e}")
            return ""

    def _extract_docx(self, filepath: str) -> dict:
        """Extract text from DOCX paragraph by paragraph."""
        try:
            from docx import Document
            doc = Document(filepath)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return {1: full_text}
        except ImportError:
            log.error("python-docx not installed. Run: pip install python-docx")
            return {}


# ─────────────────────────────────────────────
# ENTITY EXTRACTION LAYER
# ─────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts geospatial coordinates, place names, dates, and persons from text.

    Strategy:
      - Regex for coordinates: high recall, works on messy/noisy text
      - spaCy NER for place names and persons: contextual, handles variations
      - python-dateutil for robust date parsing
    """

    # Covers most real-world coordinate formats:
    # "25.3176° N, 82.9739° E"  |  "25.3176 N, 82.9739 E"
    # "34.1526° N, 77.5771° E"  |  "Lat: 25.31, Lon: 82.97"
    COORD_PATTERNS = [
        # Decimal with cardinal directions: 25.3176° N, 82.9739° E
        re.compile(
            r"(?P<lat_val>[-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*(?P<lat_dir>[NS])"
            r"\s*[,/\s]\s*"
            r"(?P<lon_val>[-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*(?P<lon_dir>[EW])",
            re.IGNORECASE,
        ),
        # Labeled: Lat: 25.31, Lon: 82.97 or latitude=25.31 longitude=82.97
        re.compile(
            r"lat(?:itude)?[\s:=]+(?P<lat_val>[-+]?\d{1,3}(?:\.\d+)?)"
            r"\s*[,\s]+\s*lon(?:gitude)?[\s:=]+(?P<lon_val>[-+]?\d{1,3}(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ]

    def __init__(self):
        self.nlp = self._load_spacy()

    def _load_spacy(self):
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            log.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm\n"
                "Falling back to regex-only (no contextual place/person NER)."
            )
            return None

    def extract_all(self, pages: dict, source_file: str) -> ExtractionResult:
        result = ExtractionResult(source_file=source_file)
        result.raw_text_by_page = pages

        for page_num, text in pages.items():
            coords = self._extract_coordinates(text, page_num)
            result.geo_points.extend(coords)

            dates = self._extract_dates(text, page_num)
            result.dates.extend(dates)

            if self.nlp:
                places, persons = self._extract_ner(text, page_num)
                self._enrich_coords_with_places(coords, places, text)
                result.persons.extend(persons)

        result.geo_points = self._deduplicate_coords(result.geo_points)
        result.dates = self._deduplicate_dates(result.dates)

        return result

    def _extract_coordinates(self, text: str, page: int) -> list:
        points = []
        seen = set()

        for pattern in self.COORD_PATTERNS:
            for m in pattern.finditer(text):
                try:
                    lat = float(m.group("lat_val"))
                    lon = float(m.group("lon_val"))

                    if "lat_dir" in m.groupdict() and m.group("lat_dir"):
                        if m.group("lat_dir").upper() == "S":
                            lat = -abs(lat)
                    if "lon_dir" in m.groupdict() and m.group("lon_dir"):
                        if m.group("lon_dir").upper() == "W":
                            lon = -abs(lon)

                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        continue

                    key = (round(lat, 4), round(lon, 4))
                    if key in seen:
                        continue
                    seen.add(key)

                    start = max(0, m.start() - 100)
                    end = min(len(text), m.end() + 100)
                    context = text[start:end].strip().replace("\n", " ")

                    points.append(GeoPoint(
                        lat=round(lat, 6),
                        lon=round(lon, 6),
                        raw_text=context,
                        page=page,
                    ))
                except (ValueError, IndexError):
                    continue

        return points

    def _extract_dates(self, text: str, page: int) -> list:
        dates = []
        try:
            from dateutil import parser as dateparser
        except ImportError:
            log.warning("python-dateutil not installed. Skipping date parsing.")
            return dates

        date_patterns = [
            # April 07, 2026 or [April 07, 2026]
            r"\[?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2}[,\s]+\d{4}\]?",
            # ISO and numeric: 2026-04-07, 07/04/2026, 07-04-2026
            r"\d{4}[-/]\d{2}[-/]\d{2}",
            r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",
        ]

        seen_dates = set()
        for pattern in date_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                raw = m.group(0).strip("[] \t\n")
                if raw in seen_dates:
                    continue
                seen_dates.add(raw)

                parsed_iso = None
                try:
                    dt = dateparser.parse(raw, dayfirst=False)
                    if dt:
                        parsed_iso = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                context = text[start:end].strip().replace("\n", " ")

                dates.append(DateMention(
                    raw=raw,
                    parsed=parsed_iso,
                    page=page,
                    context=context,
                ))

        return dates

    def _extract_ner(self, text: str, page: int):
        """Use spaCy to extract GPE/LOC (places) and PERSON entities."""
        places, persons = [], []
        if not self.nlp:
            return places, persons

        chunk_size = 100_000
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = self.nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC", "FAC"):
                    places.append({
                        "name": ent.text.strip(),
                        "label": ent.label_,
                        "page": page,
                        "start_char": i + ent.start_char,
                    })
                elif ent.label_ == "PERSON":
                    role = self._extract_role(text, i + ent.start_char)
                    persons.append(Person(
                        name=ent.text.strip(),
                        role=role,
                        page=page,
                    ))

        return places, persons

    def _extract_role(self, text: str, pos: int) -> Optional[str]:
        """Look backwards from a person mention to find a role like 'Lead Surveyor'."""
        window = text[max(0, pos - 60):pos]
        m = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s*\(", window)
        if m:
            return m.group(1)
        return None

    def _enrich_coords_with_places(self, coords: list, places: list, text: str):
        """Assign nearest place name to each coordinate based on character proximity."""
        if not places:
            return

        for coord in coords:
            if coord.raw_text and not coord.place_name:
                coord_pos = text.find(coord.raw_text[:40]) if coord.raw_text else -1
                best_match = None
                best_distance = float("inf")

                for p in places:
                    dist = abs(p.get("start_char", 0) - coord_pos)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = p["name"]

                if best_match and best_distance < 500:
                    coord.place_name = best_match

    def _deduplicate_coords(self, coords: list) -> list:
        seen = {}
        for c in coords:
            key = (round(c.lat, 3), round(c.lon, 3))
            if key not in seen:
                seen[key] = c
            elif c.place_name and not seen[key].place_name:
                seen[key].place_name = c.place_name
        return list(seen.values())

    def _deduplicate_dates(self, dates: list) -> list:
        seen = set()
        unique = []
        for d in dates:
            key = d.parsed or d.raw
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique


# ─────────────────────────────────────────────
# RELATIONSHIP / GRAPH BUILDER (GraphRAG-style)
# ─────────────────────────────────────────────

class GraphBuilder:
    """
    Builds a directed knowledge graph from extracted entities.

    Node types:   GeoPoint | Date | Person | Document | Place
    Edge types:
      CONTAINS_LOCATION  — Document → GeoPoint
      MENTIONS_DATE      — Document → Date
      NEXT_LOCATION      — GeoPoint → GeoPoint  (temporal order by page)
      VISITED_ON         — GeoPoint → Date       (co-occurrence on same page)
      OBSERVED_AT        — Person → GeoPoint     (person mentioned on same page as coord)
      ROUTE_TO           — Place → Place         (explicit "from X to Y" in text)

    Export: GraphML (compatible with Gephi, Neo4j, yEd, etc.)
    """

    ROUTE_PATTERN = re.compile(
        r"(?:from|departing?|leaving?)\s+([A-Z][a-zA-Z\s,]+?)\s+"
        r"(?:to|toward|towards|arriving?)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|\n|$)",
        re.IGNORECASE,
    )

    def __init__(self):
        self.G = nx.DiGraph()

    def build(self, result: ExtractionResult) -> nx.DiGraph:
        G = self.G
        source = result.source_file

        G.add_node(f"doc:{source}", type="Document", label=Path(source).name)

        for gp in result.geo_points:
            node_id = f"geo:{gp.lat},{gp.lon}"
            label = gp.place_name or f"({gp.lat}, {gp.lon})"
            G.add_node(node_id,
                       type="GeoPoint",
                       lat=gp.lat,
                       lon=gp.lon,
                       place_name=gp.place_name or "",
                       label=label,
                       page=gp.page or 0)
            G.add_edge(f"doc:{source}", node_id, relation="CONTAINS_LOCATION")

        for dt in result.dates:
            node_id = f"date:{dt.parsed or dt.raw}"
            G.add_node(node_id, type="Date", raw=dt.raw, iso=dt.parsed or "", label=dt.raw)
            G.add_edge(f"doc:{source}", node_id, relation="MENTIONS_DATE")

        for person in result.persons:
            node_id = f"person:{person.name}"
            G.add_node(node_id, type="Person", label=person.name, role=person.role or "")

        # NEXT_LOCATION: sequential chain of GeoPoints (by page order)
        sorted_geo = sorted(result.geo_points, key=lambda g: (g.page or 0))
        for i in range(len(sorted_geo) - 1):
            a, b = sorted_geo[i], sorted_geo[i + 1]
            G.add_edge(
                f"geo:{a.lat},{a.lon}",
                f"geo:{b.lat},{b.lon}",
                relation="NEXT_LOCATION",
                page_a=a.page,
                page_b=b.page,
            )

        # VISITED_ON: GeoPoint ↔ Date (same page co-occurrence)
        for gp in result.geo_points:
            for dt in result.dates:
                if gp.page == dt.page:
                    G.add_edge(
                        f"geo:{gp.lat},{gp.lon}",
                        f"date:{dt.parsed or dt.raw}",
                        relation="VISITED_ON",
                    )

        # ROUTE_TO: explicit textual route patterns ("from X to Y")
        for page_num, text in result.raw_text_by_page.items():
            for m in self.ROUTE_PATTERN.finditer(text):
                origin = m.group(1).strip()
                dest = m.group(2).strip()
                o_node = self._find_geo_node(G, origin) or f"place:{origin}"
                d_node = self._find_geo_node(G, dest) or f"place:{dest}"
                if not G.has_node(o_node):
                    G.add_node(o_node, type="Place", label=origin)
                if not G.has_node(d_node):
                    G.add_node(d_node, type="Place", label=dest)
                G.add_edge(o_node, d_node, relation="ROUTE_TO", page=page_num)

        # OBSERVED_AT: Person → GeoPoint (same page)
        for person in result.persons:
            for gp in result.geo_points:
                if person.page == gp.page:
                    G.add_edge(
                        f"person:{person.name}",
                        f"geo:{gp.lat},{gp.lon}",
                        relation="OBSERVED_AT",
                    )

        return G

    def _find_geo_node(self, G: nx.DiGraph, place_name: str) -> Optional[str]:
        place_lower = place_name.lower().strip()
        for node, data in G.nodes(data=True):
            if data.get("type") == "GeoPoint":
                pname = (data.get("place_name") or "").lower()
                if pname and (pname in place_lower or place_lower in pname):
                    return node
        return None

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n, **data} for n, data in self.G.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data}
                      for u, v, data in self.G.edges(data=True)],
        }


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class DocParser:
    """
    End-to-end pipeline:
      1. Extract text (with OCR fallback for scanned docs)
      2. Extract entities: coordinates, dates, persons, places
      3. Build GraphRAG-style relationship graph
      4. Export JSON (entities + graph) and GraphML
    """

    def __init__(self, force_ocr: bool = False):
        self.extractor = TextExtractor(force_ocr=force_ocr)
        self.entity_extractor = EntityExtractor()

    def parse(self, filepath: str) -> tuple:
        log.info(f"Parsing: {filepath}")

        pages = self.extractor.extract(filepath)
        if not pages:
            log.error("No text extracted. Check file and parser settings.")
            return {}, None

        total_chars = sum(len(t) for t in pages.values())
        log.info(f"Extracted {len(pages)} pages, {total_chars} characters.")

        result = self.entity_extractor.extract_all(pages, filepath)
        log.info(
            f"Found: {len(result.geo_points)} coordinates, "
            f"{len(result.dates)} dates, "
            f"{len(result.persons)} persons."
        )

        builder = GraphBuilder()
        G = builder.build(result)
        log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

        timespan = self._calculate_timespan(result.dates)
        unique_places = len({gp.place_name for gp in result.geo_points if gp.place_name})

        output = {
            "source_file": filepath,
            "parsed_at": datetime.now().isoformat(),
            "summary": {
                "pages": len(pages),
                "geo_points": len(result.geo_points),
                "dates": len(result.dates),
                "places": unique_places,
                "persons": len(result.persons),
                "graph_nodes": G.number_of_nodes(),
                "graph_edges": G.number_of_edges(),
                "timespan": timespan,
            },
            "entities": {
                "geo_points": [asdict(gp) for gp in result.geo_points],
                "dates": [asdict(d) for d in result.dates],
                "persons": [asdict(p) for p in result.persons],
            },
            "graph": builder.to_dict(),
        }

        return output, G

    def _calculate_timespan(self, dates: list) -> str:
        parsed_dates = [d.parsed for d in dates if d.parsed]
        if not parsed_dates:
            return "N/A"

        try:
            timestamps = [datetime.fromisoformat(dt) for dt in parsed_dates]
            start = min(timestamps).date().isoformat()
            end = max(timestamps).date().isoformat()
            return start if start == end else f"{start} → {end}"
        except ValueError:
            return "N/A"

    def parse_folder(self, folder: str) -> list:
        folder_path = Path(folder)
        supported = {".pdf", ".docx", ".txt", ".md"}
        files = [f for f in folder_path.iterdir() if f.suffix.lower() in supported]
        log.info(f"Found {len(files)} files to parse in {folder}")
        results = []
        for f in files:
            try:
                out, _ = self.parse(str(f))
                if out:
                    results.append(out)
            except Exception as e:
                log.error(f"Failed to parse {f}: {e}")
        return results


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract geo-entities and relationships from PDF/DOCX files (GraphRAG-ready)."
    )
    ap.add_argument(
        "--input",
        default=r"docs",
        help="Path to a file or folder (default: docs)",
    )
    ap.add_argument("--output", default="parsed_output.json", help="JSON output path")
    ap.add_argument("--graph", default="graph.graphml", help="GraphML output path")
    ap.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on every page (use for scanned/image-only PDFs)",
    )
    args = ap.parse_args()

    parser = DocParser(force_ocr=args.force_ocr)
    input_path = Path(args.input)

    if input_path.is_dir():
        results = parser.parse_folder(str(input_path))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"Saved {len(results)} results → {args.output}")
    else:
        output, G = parser.parse(str(input_path))
        if not output:
            return

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info(f"Saved JSON → {args.output}")

        if G is not None:
            nx.write_graphml(G, args.graph)
            log.info(f"Saved graph → {args.graph}")

        # Terminal summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        s = output["summary"]
        print(f"  Pages parsed  : {s['pages']}")
        print(f"  Geo points    : {s['geo_points']}")
        print(f"  Dates         : {s['dates']}")
        print(f"  Timespan      : {s.get('timespan', 'N/A')}")
        print(f"  Places found  : {s.get('unique_places', 0)}")
        print(f"  Persons       : {s['persons']}")
        print(f"  Graph nodes   : {s['graph_nodes']}")
        print(f"  Graph edges   : {s['graph_edges']}")
        print("-" * 60)
        print(f"  Source file   : {output['source_file']}")

        print("\nGEO POINTS:")
        for gp in output["entities"]["geo_points"]:
            name = gp.get("place_name") or "unnamed"
            print(f"  ({gp['lat']}, {gp['lon']})  →  {name}  [page {gp['page']}]")

        print("\nDATES:")
        for d in output["entities"]["dates"]:
            print(f"  {d['raw']}  →  {d['parsed']}  [page {d['page']}]")

        print("\nPERSONS:")
        for p in output["entities"]["persons"]:
            role = p.get("role") or "—"
            print(f"  {p['name']}  ({role})  [page {p['page']}]")
        print("=" * 60)


if __name__ == "__main__":
    main()