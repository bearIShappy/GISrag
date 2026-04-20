"""
graph_builder.py
================
GraphRAG-style relationship graph builder.
Import and use via:
    from graph_builder import GraphBuilder
— self-contained, only needs re, pathlib, typing, and networkx. The type hint on build() was loosened from ExtractionResult to just result (no import needed) since Python's duck typing handles it fine — it just accesses .geo_points, .dates, etc.
"""

import re
from pathlib import Path
from typing import Optional

import networkx as nx


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

    def build(self, result) -> nx.DiGraph:
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
    
if __name__ == "__main__":
    print("GraphBuilder loaded OK.")
    gb = GraphBuilder()
    print(f"Empty graph nodes: {gb.G.number_of_nodes()}")