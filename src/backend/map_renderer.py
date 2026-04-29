import folium
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def render_map(data: Dict[str, Any], output_file: str = "output_map.html"):
    """
    Renders a Folium map based on the RAG query output.
    """
    points = data.get("map_points", [])
    if not points:
        logger.warning("No map points found in data.")
        return

    # Sort points by date for the path
    try:
        points.sort(key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d"))
    except Exception as e:
        logger.warning(f"Failed to sort points by date: {str(e)}")

    # Initialize map at the first point
    start_coords = [points[0]['lat'], points[0]['lon']]
    m = folium.Map(location=start_coords, zoom_start=6, tiles="CartoDB positron")

    path_coords = []
    for point in points:
        lat, lon = point['lat'], point['lon']
        path_coords.append([lat, lon])
        
        # Create popup content
        popup_text = f"""
        <b>Place:</b> {point['place']}<br>
        <b>Date:</b> {point['date']}<br>
        <b>Summary:</b> {point['summary']}
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=point['place'],
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Draw a line connecting the points if multiple points exist
    if len(path_coords) > 1:
        folium.PolyLine(
            path_coords,
            color="red",
            weight=2.5,
            opacity=0.8,
            tooltip="Event Chronology"
        ).add_to(m)

    m.save(output_file)
    logger.info(f"Map saved to {output_file}")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python map_renderer.py path_to_rag_output.json")
        return

    try:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        render_map(data)
    except Exception as e:
        logger.error(f"Failed to render map: {str(e)}")

if __name__ == "__main__":
    main()
