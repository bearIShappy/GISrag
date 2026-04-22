"""
cleaner.py — Stage 1: Data Cleaning & Normalization
-----------------------------------------------------
Input:  Raw _data.json files from doc_parser2.py (GLiNER2 output)
Output: cleaned_records.json — a flat list of complete, normalized records

What it does:
  1. Loads all *_data.json files from OUTPUT_DIR
  2. Normalizes lat/lon strings → float  ("25.3176° N" → 25.3176)
  3. Standardizes date strings → ISO format ("April 11, 2026" → "2026-04-11")
  4. Forward-geocodes missing coords from place name  (Nominatim / OSM)
  5. Reverse-geocodes missing place name from coords
  6. Drops records that are completely unresolvable (no place AND no coords)
  7. Tags each record with source_file for traceability
  8. Writes cleaned_records.json to OUTPUT_DIR
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import dateparser
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

load_dotenv()

INPUT_DIR  = Path(os.getenv("EXTRACTED_OUTPUT_FOLDER", "./data/output/extracted"))   # GLiNER2 output dir
OUTPUT_DIR = Path(os.getenv("CLEANED_OUTPUT_FOLDER", "./data/output/cleaned"))     # Cleaned output dir
CLEANED_FILE = OUTPUT_DIR / "cleaned_records.json"

# Nominatim requires a unique user-agent string
geolocator = Nominatim(user_agent="rag-geo-pipeline-v1", timeout=10)


# ─────────────────────────────────────────────
# 1. COORDINATE NORMALIZATION
# ─────────────────────────────────────────────

def parse_coordinate(raw: str | None, is_longitude: bool = False) -> float | None:
    """
    Convert any lat/lon string to a signed float.

    Handles:
      "25.3176° N"  → 25.3176
      "82.9739° E"  → 82.9739
      "34.5822 N"   → 34.5822
      "-74.0060"    → -74.0060
      "40.7128 S"   → -40.7128  (southern hemisphere → negative)
      "74.0060 W"   → -74.0060  (western hemisphere → negative)
    """
    if raw is None:
        return None

    raw = str(raw).strip()

    # Extract numeric part
    num_match = re.search(r"[-+]?\d+\.?\d*", raw)
    if not num_match:
        return None

    value = float(num_match.group())

    # Apply hemisphere sign
    upper = raw.upper()
    if "S" in upper or "W" in upper:
        value = -value

    # Basic sanity check
    if is_longitude and not (-180 <= value <= 180):
        return None
    if not is_longitude and not (-90 <= value <= 90):
        return None

    return round(value, 6)


# ─────────────────────────────────────────────
# 2. DATE NORMALIZATION
# ─────────────────────────────────────────────

def normalize_date(raw: str | None) -> str | None:
    """
    Convert any human date string to ISO 8601 (YYYY-MM-DD).
    Falls back to year-month (YYYY-MM) if exact day is unavailable.

    Examples:
      "April 11, 2026"  → "2026-04-11"
      "April 2026"      → "2026-04"   (no day available)
      "11/04/2026"      → "2026-04-11"
    """
    if raw is None:
        return None

    raw = str(raw).strip()

    # Month-year only pattern e.g. "April 2026"
    month_year = re.match(
        r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{4})$",
        raw, re.IGNORECASE
    )
    if month_year:
        parsed = dateparser.parse(raw)
        if parsed:
            return parsed.strftime("%Y-%m")

    parsed = dateparser.parse(raw, settings={"RETURN_AS_TIMEZONE_AWARE": False})
    if parsed:
        return parsed.strftime("%Y-%m-%d")

    return raw   # Return original if unparseable (better than None)


# ─────────────────────────────────────────────
# 3. GEOCODING HELPERS
# ─────────────────────────────────────────────

def forward_geocode(place: str) -> tuple[float, float] | tuple[None, None]:
    """Place name → (lat, lon). Returns (None, None) on failure."""
    try:
        time.sleep(1.1)   # Nominatim rate limit: 1 req/sec
        location = geolocator.geocode(place)
        if location:
            return round(location.latitude, 6), round(location.longitude, 6)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"    [GEOCODE WARN] Forward geocode failed for '{place}': {e}")
    return None, None


def reverse_geocode(lat: float, lon: float) -> str | None:
    """(lat, lon) → human-readable place name. Returns None on failure."""
    try:
        time.sleep(1.1)
        location = geolocator.reverse((lat, lon), language="en")
        if location:
            # Return city + country for brevity
            addr = location.raw.get("address", {})
            parts = [
                addr.get("city") or addr.get("town") or addr.get("village"),
                addr.get("state"),
                addr.get("country"),
            ]
            return ", ".join(p for p in parts if p)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"    [GEOCODE WARN] Reverse geocode failed for ({lat},{lon}): {e}")
    return None


# ─────────────────────────────────────────────
# 4. RECORD CLEANING
# ─────────────────────────────────────────────

def clean_record(raw: dict, source_file: str) -> dict | None:
    """
    Clean and enrich a single GLiNER2-extracted record.
    Returns None if the record is unresolvable (no place, no coords).
    """

    # --- Coordinates ---
    lat = parse_coordinate(raw.get("latitude"), is_longitude=False)
    lon = parse_coordinate(raw.get("longitude"), is_longitude=True)
    place = str(raw.get("place")).strip() if raw.get("place") else None

    # Forward geocode if coords missing but place exists
    if (lat is None or lon is None) and place:
        print(f"    [GEOCODE] Forward geocoding: '{place}'")
        lat, lon = forward_geocode(place)

    # Reverse geocode if place missing but coords exist
    if place is None and lat is not None and lon is not None:
        print(f"    [GEOCODE] Reverse geocoding: ({lat}, {lon})")
        place = reverse_geocode(lat, lon)

    # Drop if still unresolvable
    if place is None and (lat is None or lon is None):
        print(f"    [SKIP] Unresolvable record (no place, no coords) in {source_file}")
        return None

    # --- Date ---
    date_norm = normalize_date(raw.get("date"))

    # --- Assemble clean record ---
    return {
        "place":       place,
        "latitude":    lat,
        "longitude":   lon,
        "date":        date_norm,
        "date_raw":    raw.get("date"),        # Keep original for reference
        "time":        raw.get("time"),
        "role":        raw.get("role"),
        "summary":     raw.get("summary"),
        "source_file": source_file,
    }


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def load_and_clean_all() -> list[dict]:
    json_files = sorted(INPUT_DIR.glob("*_data.json"))

    if not json_files:
        print(f"No *_data.json files found in {INPUT_DIR}")
        return []

    all_cleaned = []

    for json_path in json_files:
        print(f"\n[Loading] {json_path.name}")
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  !!! JSON parse error: {e}")
            continue

        # GLiNER2 output wraps records under "document_info"
        records = data.get("document_info", [])
        if not isinstance(records, list):
            records = [records]

        print(f"  Found {len(records)} raw record(s)")

        for i, raw_record in enumerate(records):
            print(f"  → Cleaning record {i+1}/{len(records)}...")
            cleaned = clean_record(raw_record, source_file=json_path.name)
            if cleaned:
                all_cleaned.append(cleaned)

    return all_cleaned


def main():
    print("=" * 55)
    print("  Stage 1: Data Cleaning & Normalization")
    print("=" * 55)

    cleaned = load_and_clean_all()

    if not cleaned:
        print("\n[!] No valid records produced. Check your input JSON files.")
        return

    # ── Summary stats ──
    total         = len(cleaned)
    with_coords   = sum(1 for r in cleaned if r["latitude"] and r["longitude"])
    with_place    = sum(1 for r in cleaned if r["place"])
    with_date     = sum(1 for r in cleaned if r["date"])
    with_summary  = sum(1 for r in cleaned if r["summary"])

    print(f"\n{'─'*40}")
    print(f"  ✓ Total clean records  : {total}")
    print(f"  ✓ With coordinates     : {with_coords}/{total}")
    print(f"  ✓ With place name      : {with_place}/{total}")
    print(f"  ✓ With date            : {with_date}/{total}")
    print(f"  ✓ With summary         : {with_summary}/{total}")
    print(f"{'─'*40}")

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLEANED_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved → {CLEANED_FILE}")
    print("\n  Next step: run ingest.py to embed & store in Qdrant")


if __name__ == "__main__":
    main()