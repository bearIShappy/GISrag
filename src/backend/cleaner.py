"""
cleaner.py — Stage 1: Data Cleaning & Normalization
-----------------------------------------------------
Input:  Raw *_data.json files from doc_parser.py (GLiNER2 output)
Output: cleaned_records.json  — valid, normalised records
        quarantine.json        — unresolvable records with drop_reason

What it does:
  1. Loads all *_data.json files from INPUT_DIR
  2. Normalises lat/lon strings → signed float  (hemisphere-aware)
  3. Standardises date strings → ISO 8601, with date_type tagging
  4. Cleans time strings — rejects garbage values like "."
  5. Forward-geocodes missing coords from place name  (Nominatim/OSM)
  6. Reverse-geocodes missing place name from coords
       → stored in geocoded_place; never overwrites extracted place
  7. Quarantines (not silently drops) unresolvable records
  8. Tags every record with source_file + extraction metadata
  9. Writes cleaned_records.json and quarantine.json to OUTPUT_DIR
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import date as date_cls, datetime
from dotenv import load_dotenv

import dateparser
from dateutil import parser as dateutil_parser
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

load_dotenv()

INPUT_DIR    = Path(os.getenv("EXTRACTED_OUTPUT_FOLDER", "./data/output/extracted"))
OUTPUT_DIR   = Path(os.getenv("CLEANED_OUTPUT_FOLDER",   "./data/output/cleaned"))
CLEANED_FILE = OUTPUT_DIR / "cleaned_records.json"
QUARANTINE_FILE = OUTPUT_DIR / "quarantine.json"

geolocator = Nominatim(user_agent="rag-geo-pipeline-v1", timeout=10)

# Accumulates records that cannot be resolved (written at the end)
QUARANTINE: list[dict] = []


# ─────────────────────────────────────────────
# 1. COORDINATE NORMALISATION  (hemisphere-aware)
# ─────────────────────────────────────────────

def parse_coordinate(raw: str | float | None, is_longitude: bool = False) -> float | None:
    """
    Convert any lat/lon string to a signed float.

    Reads the N/S/E/W direction BEFORE stripping it so the sign is
    never lost.  The old implementation used re.search(r'[-+]?\\d+')
    which found the numeric part correctly but then applied the sign
    check on the full raw string — a string like "7.5407S" still had
    "S" in it, so the sign was applied.  The real failure mode was
    strings like "34.5822 N" where the regex grabbed "34" not "34.5822"
    because it stopped at the space.  The rewrite below is unambiguous.

    Examples
    --------
    "25.3176° N"  →  25.3176
    "7.5407° S"   →  -7.5407    ← was broken (sign lost in some paths)
    "74.0060 W"   →  -74.0060
    "34.5822 N"   →  34.5822
    "-74.0060"    →  -74.0060
    None          →  None
    """
    if raw is None:
        return None

    s = str(raw).strip()

    # 1. Capture direction letter before any stripping
    direction: str | None = None
    dir_match = re.search(r'[NSEWnsew]', s)
    if dir_match:
        direction = dir_match.group().upper()
        # Remove the direction letter from the string
        s = s[:dir_match.start()] + s[dir_match.end():]

    # 2. Strip degree symbols and whitespace, then parse float
    s = re.sub(r'[°\s]', '', s)
    try:
        value = float(s)
    except ValueError:
        return None

    # 3. Apply hemisphere sign
    if direction in ('S', 'W'):
        value = -abs(value)
    elif direction in ('N', 'E'):
        value = abs(value)
    # No direction letter: trust the sign already in the number

    # 4. Sanity bounds
    if is_longitude and not (-180 <= value <= 180):
        return None
    if not is_longitude and not (-90 <= value <= 90):
        return None

    return round(value, 6)


# ─────────────────────────────────────────────
# 2. DATE NORMALISATION  (with type tagging)
# ─────────────────────────────────────────────

def normalize_date(raw: str | None) -> dict:
    """
    Return a dict with keys:
      date      – ISO 8601 string or None
      date_type – "event" | "period" | "historical" | "unparseable" | None

    The old normalize_date() returned a bare string and passed
    garbage like "13th-century" straight through because dateparser
    can (sometimes) parse ordinals.  This version classifies first.
    """
    if not raw:
        return {"date": None, "date_type": None}

    s = str(raw).strip()

    # Historical architectural / era reference  e.g. "13th-century"
    if re.search(r'\d+(st|nd|rd|th)[\s\-]century', s, re.IGNORECASE):
        return {"date": None, "date_type": "historical"}

    # Quarter or month-year period  e.g. "Q1", "Q2 2026", "April 2026"
    if re.match(r'^Q[1-4](\s+\d{4})?$', s, re.IGNORECASE):
        return {"date": s, "date_type": "period"}
    month_year = re.match(
        r'^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+(\d{4})$', s, re.IGNORECASE
    )
    if month_year:
        parsed = dateparser.parse(s)
        if parsed:
            return {"date": parsed.strftime("%Y-%m"), "date_type": "period"}

    # Relative "today"
    if s.lower() == "today":
        return {"date": date_cls.today().isoformat(), "date_type": "event"}

    # Standard ISO / natural language
    try:
        d = dateutil_parser.parse(s, fuzzy=True)
        return {"date": d.date().isoformat(), "date_type": "event"}
    except Exception:
        pass

    # dateparser as final fallback
    parsed = dateparser.parse(s, settings={"RETURN_AS_TIMEZONE_AWARE": False})
    if parsed:
        return {"date": parsed.strftime("%Y-%m-%d"), "date_type": "event"}

    return {"date": None, "date_type": "unparseable", "date_raw_original": s}


# ─────────────────────────────────────────────
# 3. TIME VALIDATION
# ─────────────────────────────────────────────

def clean_time(value: str | None) -> str | None:
    """
    Return the time string only if it looks like an actual time.
    Rejects garbage values like ".", single letters, bare timezone
    abbreviations, etc.

    Accepts:
      "07:15 IST"                  → "07:15 IST"
      "14:20 PM"                   → "14:20 PM"
      "2024-01-10T09:30:00Z"       → "09:30:00"   (extracts time part)
      "03:22:11 UTC"               → "03:22:11 UTC"

    Rejects:
      "."   →  None
      "IST" →  None   (bare timezone with no time component)
    """
    if not value:
        return None
    s = str(value).strip()
    # ISO datetime  e.g. "2024-01-10T09:30:00Z"
    if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', s):
        return s.split('T')[1].rstrip('Z')
    # Must start with HH:MM to be a valid time
    if re.match(r'^\d{1,2}:\d{2}', s):
        return s
    return None


# ─────────────────────────────────────────────
# 4. GEOCODING HELPERS
# ─────────────────────────────────────────────

def forward_geocode(place: str) -> tuple[float, float] | tuple[None, None]:
    """Place name → (lat, lon).  Returns (None, None) on failure."""
    try:
        time.sleep(1.1)   # Nominatim rate limit: 1 req/sec
        location = geolocator.geocode(place)
        if location:
            return round(location.latitude, 6), round(location.longitude, 6)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"    [GEOCODE WARN] Forward geocode failed for '{place}': {e}")
    return None, None


def reverse_geocode(lat: float, lon: float) -> str | None:
    """
    (lat, lon) → human-readable label.

    IMPORTANT: the result is stored in geocoded_place, NEVER in place.
    Reverse geocoding returns the nearest road/suburb, which is often
    wrong for remote observation sites (e.g. 7.6775N/98.7651E → Phi Phi
    Islands, but Nominatim returns "Ao Nang").  Keep the original
    extracted place as the authoritative name.
    """
    try:
        time.sleep(1.1)
        location = geolocator.reverse((lat, lon), language="en")
        if location:
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
# 5. RECORD CLEANING  (main per-record logic)
# ─────────────────────────────────────────────

def clean_record(raw: dict, source_file: str) -> dict:
    """
    Clean and enrich a single GLiNER2-extracted record.

    Returns a record dict always — unresolvable records carry
    drop_reason and are routed to QUARANTINE by the caller; they are
    never silently discarded.

    Key changes vs the old version
    ───────────────────────────────
    • Uses parse_coordinate() (hemisphere-aware, fixed)
    • Uses normalize_date() which returns a dict with date_type
    • Uses clean_time() to reject garbage time strings
    • Does NOT overwrite place with reverse-geocode result —
      stores it in geocoded_place instead
    • Attaches geocode_status for downstream confidence filtering
    """
    geocode_status = "exact"   # assume coords came from document

    # ── Coordinates ──────────────────────────────────────────
    lat = parse_coordinate(raw.get("latitude"),  is_longitude=False)
    lon = parse_coordinate(raw.get("longitude"), is_longitude=True)
    place = str(raw.get("place")).strip() if raw.get("place") else None

    # Forward-geocode only when BOTH coords are missing
    if lat is None and lon is None and place:
        print(f"    [GEOCODE] Forward geocoding: '{place}'")
        lat, lon = forward_geocode(place)
        geocode_status = "forward" if lat is not None else "failed"

    # Partial coords (one of lat/lon is None but not both) — flag but keep
    elif (lat is None) != (lon is None):
        print(f"    [WARN] Partial coords in {source_file}: lat={lat}, lon={lon}, place={place}")
        geocode_status = "partial"

    # Reverse-geocode: only fills geocoded_place, never place
    geocoded_place: str | None = None
    if lat is not None and lon is not None:
        if place is None:
            print(f"    [GEOCODE] Reverse geocoding: ({lat}, {lon})")
            geocoded_place = reverse_geocode(lat, lon)
            geocode_status = "reverse" if geocoded_place else "failed"
        # Always store reverse label as a secondary field when we have coords
        # (handy for display even when place is known)
        elif geocode_status == "exact":
            geocoded_place = None   # skip to avoid unnecessary API calls

    # ── Date ──────────────────────────────────────────────────
    date_info = normalize_date(raw.get("date"))

    # ── Time ──────────────────────────────────────────────────
    cleaned_time = clean_time(raw.get("time"))

    # ── Assemble ─────────────────────────────────────────────
    record = {
        "place":          place,
        "geocoded_place": geocoded_place,   # Nominatim label (secondary, never authoritative)
        "latitude":       lat,
        "longitude":      lon,
        "date":           date_info["date"],
        "date_type":      date_info["date_type"],    # "event"|"period"|"historical"|"unparseable"
        "date_raw":       raw.get("date"),
        "time":           cleaned_time,
        "role":           raw.get("role"),
        "summary":        raw.get("summary"),
        "geocode_status": geocode_status,            # "exact"|"forward"|"reverse"|"partial"|"failed"
        "source_file":    source_file,
    }

    # Mark unresolvable records so the caller can quarantine them
    if place is None and geocoded_place is None and (lat is None or lon is None):
        record["drop_reason"] = "no_place_no_coords"

    return record


# ─────────────────────────────────────────────
# 6. QUARANTINE ROUTER
# ─────────────────────────────────────────────

def route_record(record: dict, cleaned: list[dict]) -> None:
    """
    Send the record to the cleaned list or to QUARANTINE.
    Never silently drops.
    """
    if "drop_reason" in record:
        QUARANTINE.append(record)
        print(f"    [QUARANTINE] {record.get('source_file')} — {record['drop_reason']}")
    else:
        cleaned.append(record)


# ─────────────────────────────────────────────
# 7. MAIN LOAD & CLEAN LOOP
# ─────────────────────────────────────────────

def load_and_clean_all() -> list[dict]:
    json_files = sorted(INPUT_DIR.glob("*_data.json"))

    if not json_files:
        print(f"No *_data.json files found in {INPUT_DIR}")
        return []

    all_cleaned: list[dict] = []

    for json_path in json_files:
        print(f"\n[Loading] {json_path.name}")
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  !!! JSON parse error: {e}")
            continue

        records = data.get("document_info", [])
        if isinstance(records, dict):
            records = [records]

        print(f"  Found {len(records)} raw record(s)")

        for i, raw_record in enumerate(records):
            print(f"  → Cleaning record {i+1}/{len(records)}...")
            cleaned = clean_record(raw_record, source_file=json_path.name)
            route_record(cleaned, all_cleaned)

    return all_cleaned


def main():
    print("=" * 55)
    print("  Stage 1: Data Cleaning & Normalization")
    print("=" * 55)

    cleaned = load_and_clean_all()

    # ── Summary stats ──────────────────────────────────────
    total        = len(cleaned)
    quarantined  = len(QUARANTINE)
    with_coords  = sum(1 for r in cleaned if r["latitude"] and r["longitude"])
    with_place   = sum(1 for r in cleaned if r["place"])
    with_date    = sum(1 for r in cleaned if r["date"])
    exact        = sum(1 for r in cleaned if r["geocode_status"] == "exact")
    forward_geo  = sum(1 for r in cleaned if r["geocode_status"] == "forward")
    reverse_geo  = sum(1 for r in cleaned if r["geocode_status"] == "reverse")

    print(f"\n{'─'*40}")
    print(f"  ✓ Clean records        : {total}")
    print(f"  ⚠ Quarantined          : {quarantined}")
    print(f"  ✓ With coordinates     : {with_coords}/{total}")
    print(f"  ✓ With place name      : {with_place}/{total}")
    print(f"  ✓ With date            : {with_date}/{total}")
    print(f"  · Geocode — exact      : {exact}")
    print(f"  · Geocode — forward    : {forward_geo}")
    print(f"  · Geocode — reverse    : {reverse_geo}")
    print(f"{'─'*40}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save cleaned records ───────────────────────────────
    if cleaned:
        with open(CLEANED_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {CLEANED_FILE}")
    else:
        print("\n[!] No valid records produced. Check your input JSON files.")

    # ── Save quarantine ────────────────────────────────────
    if QUARANTINE:
        with open(QUARANTINE_FILE, "w", encoding="utf-8") as f:
            json.dump(QUARANTINE, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {QUARANTINE_FILE}  ({quarantined} record(s) need review)")

    print("\n  Next step: run ingest.py to embed & store in DB")


if __name__ == "__main__":
    main()