"""
cleaner.py — Stage 1: Data Cleaning & Normalization
-----------------------------------------------------
Input:  Raw *_data.json files from doc_parser.py (GLiNER2 output)
Output: cleaned_records.json  — valid, normalised records
        quarantine.json        — unresolvable records with drop_reason

What it does:
  1. Loads all *_data.json files from INPUT_DIR
  2. Normalises lat/lon strings → signed float  (hemisphere-aware)
       • Reads N/S/E/W BEFORE stripping so sign is never lost
       • Handles bare numerics by inferring hemisphere from value sign
       • assert_signed_coords() gate re-checks sign from raw before Nominatim
       • Guarantees: N→+lat  S→−lat  E→+lon  W→−lon  reaching geocoder
  3. Standardises date strings → ISO 8601, with date_type tagging
  4. Cleans time strings — handles ISO datetimes, HH:MM, Excel fractional
     day floats, and seconds-style ("SS") values; rejects garbage
  5. Forward-geocodes missing coords from place name  (Nominatim/OSM)
  5. Reverse-geocodes missing place name from coords using a four-tier
     fallback ladder: Nominatim zoom [18→10→6→3] → Photon → BigDataCloud
     → coordinate-string.  geocoded_place is NEVER null when lat+lon exist.
     geocode_status values: exact | forward | reverse | reverse_coord_fallback | failed | partial
  7. Sanity-checks extracted coords (lat==lon bleed, wrong-hemisphere)
  8. Quarantines (not silently drops) unresolvable records
  9. Tags every record with source_file + extraction metadata
 10. Writes cleaned_records.json and quarantine.json to OUTPUT_DIR
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
from geopy.exc import (GeocoderTimedOut, GeocoderServiceError,
                       GeocoderInsufficientPrivileges, GeocoderUnavailable,
                       GeopyError)

load_dotenv()

INPUT_DIR    = Path(os.getenv("EXTRACTED_OUTPUT_FOLDER", "./data/output/extracted"))
OUTPUT_DIR   = Path(os.getenv("CLEANED_OUTPUT_FOLDER",   "./data/output/cleaned"))
CLEANED_FILE    = OUTPUT_DIR / "cleaned_records.json"
QUARANTINE_FILE = OUTPUT_DIR / "quarantine.json"

geolocator = Nominatim(user_agent="rag-geo-pipeline-v1", timeout=10)

QUARANTINE: list[dict] = []


# ─────────────────────────────────────────────
# 1. COORDINATE NORMALISATION  (hemisphere-aware)
# ─────────────────────────────────────────────

def parse_coordinate(raw: str | float | None, is_longitude: bool = False) -> float | None:
    """
    Convert any lat/lon string to a signed float, preserving hemisphere.

    The function reads the N/S/E/W direction letter FIRST, before any
    stripping, so the sign is never lost — even for values like
    "34.5822 N", "7.5407 S", "74.0060 W", or "-13.1631".

    Also handles:
      • Degree symbols:  "25.3176° N"
      • DMS shorthand:   "25°19'3.36\"N"  (common in older field logs)
      • Bare negatives:  "-74.0060"  (sign is the hemisphere)
      • Bare positives:  "34.5822"   (no hemisphere, sign = +)

    Returns None for unparseable or out-of-bounds values.
    """
    if raw is None:
        return None

    s = str(raw).strip()

    # ── DMS  "25°19'3.36\"N"  or  "25 19 3.36 N" ──────────────────────
    dms_match = re.match(
        r"""^(-?\d+)[°\s]\s*(\d+)['\s]\s*(\d+(?:\.\d+)?)[\"'\s]*([NSEWnsew]?)""",
        s
    )
    if dms_match:
        deg, mins, secs, direction = dms_match.groups()
        value = abs(float(deg)) + float(mins) / 60 + float(secs) / 3600
        direction = direction.upper()
        if direction in ('S', 'W'):
            value = -value
        elif not direction and float(deg) < 0:
            value = -value
        return _bounds_check(round(value, 6), is_longitude)

    # ── Standard decimal with optional hemisphere ───────────────────────

    # 1. Capture direction letter before any stripping
    direction: str | None = None
    dir_match = re.search(r'[NSEWnsew]', s)
    if dir_match:
        direction = dir_match.group().upper()
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
    # No direction letter → trust the existing sign in the number

    return _bounds_check(round(value, 6), is_longitude)


def _bounds_check(value: float, is_longitude: bool) -> float | None:
    """Return value if within valid geographic bounds, else None."""
    if is_longitude and not (-180 <= value <= 180):
        return None
    if not is_longitude and not (-90 <= value <= 90):
        return None
    return value


# ─────────────────────────────────────────────
# 2. COORDINATE SANITY CHECK  (post-parse)
# ─────────────────────────────────────────────

# Place keywords that strongly imply the Western hemisphere.
# Used to detect when a positive longitude is clearly wrong.
_WEST_KEYWORDS = (
    'peru', 'bolivia', 'chile', 'brazil', 'colombia', 'ecuador',
    'argentina', 'mexico', 'canada', 'united states', 'usa',
    'aguas calientes', 'machu picchu', 'salt flats', 'uyuni',
    'cusco', 'cuzco', 'lima', 'bogota', 'santiago', 'buenos aires',
    'caracas', 'quito', 'la paz',
)

# Place keywords that strongly imply the Southern hemisphere.
# IMPORTANT: only include places that are unambiguously south of the equator.
# Sipadan Island (Malaysia) is at 4.1° N — do NOT include it here.
# Indonesia spans both hemispheres — too ambiguous to include.
_SOUTH_KEYWORDS = (
    'south africa', 'cape town', 'johannesburg', 'australia', 'sydney',
    'melbourne', 'new zealand', 'auckland', 'argentina', 'chile',
    'bolivia', 'peru', 'brazil', 'mount merapi', 'salt flats', 'uyuni',
)


def _sanity_check_coords(
    lat: float | None,
    lon: float | None,
    place: str | None,
) -> tuple[float | None, float | None, str]:
    """
    Catch the two most common GLiNER2 coordinate bleed errors and
    hemisphere-sign errors that survive parse_coordinate():

    Error 1 — lat == lon
      The extractor copied the same value into both fields (e.g. Leh
      had lat=34.5822, lon=34.5822 when the real lon is ~77.58).
      Fix: null out longitude; cleaner will forward-geocode from place.

    Error 2 — wrong-sign longitude for clearly Western places
      GLiNER2 sometimes strips the "W" when parsing tables; the result
      is a positive longitude for a South-American or North-American site.
      Fix: flip sign when place keyword implies Western hemisphere.

    Error 3 — wrong-sign latitude for clearly Southern places
      Same issue for S hemisphere sites.
      Fix: flip sign when place keyword implies Southern hemisphere.

    Returns (corrected_lat, corrected_lon, note_string).
    """
    note = ""

    if lat is None or lon is None:
        return lat, lon, note

    place_lower = (place or "").lower()

    # Error 1: identical lat and lon (copy-paste bleed)
    if lat == lon:
        lon  = None
        note = "lon_nulled:lat_eq_lon"
        return lat, lon, note

    # Error 2: positive longitude for a known-Western location
    if lon > 0 and any(kw in place_lower for kw in _WEST_KEYWORDS):
        lon  = -abs(lon)
        note = "lon_sign_flipped:west_keyword"

    # Error 3: positive latitude for a known-Southern location
    if lat > 0 and any(kw in place_lower for kw in _SOUTH_KEYWORDS):
        lat  = -abs(lat)
        note = (note + " lat_sign_flipped:south_keyword").strip()

    return lat, lon, note


# ─────────────────────────────────────────────
# 3. SIGNED-FLOAT GATE  (pre-geocoding assertion)
# ─────────────────────────────────────────────

def assert_signed_coords(
    lat: float | None,
    lon: float | None,
    raw_lat: str | None,
    raw_lon: str | None,
    source_file: str,
) -> tuple[float | None, float | None]:
    """
    Final gate called IMMEDIATELY before any geocoding (forward or reverse).

    Guarantees that whatever floats reach Nominatim obey the convention:
        N → positive latitude     e.g.  4.1147
        S → negative latitude     e.g. -4.1147
        E → positive longitude    e.g. 118.6287
        W → negative longitude    e.g. -118.6287

    If parse_coordinate() and _sanity_check_coords() have done their job
    correctly the values should already be signed.  This function is a
    defensive double-check: it re-reads the hemisphere letter directly
    from the original raw strings and compares the expected sign against
    the parsed float.  If there is a mismatch it corrects and logs it so
    the problem is never silently passed to Nominatim.

    It also logs the final (lat, lon) that will be sent to the geocoder
    so every geocoding call is fully traceable in the console output.

    Returns (lat, lon) — corrected if necessary, otherwise unchanged.
    """
    def _expected_sign(raw: str | None, negative_letters: tuple) -> int | None:
        """
        Return +1, -1, or None (unknown) based on the hemisphere letter
        found in the raw string.
        negative_letters = ('S',) for latitude, ('W',) for longitude.
        """
        if not raw:
            return None
        m = re.search(r'[NSEWnsew]', str(raw))
        if not m:
            return None
        letter = m.group().upper()
        return -1 if letter in negative_letters else +1

    corrected = False

    if lat is not None:
        exp = _expected_sign(raw_lat, ('S',))
        if exp is not None:
            actual_sign = -1 if lat < 0 else +1
            if actual_sign != exp:
                old_lat = lat
                lat = exp * abs(lat)
                print(
                    f"    [SIGN-GATE] {source_file} | "
                    f"lat sign mismatch: raw='{raw_lat}' expected={'S' if exp==-1 else 'N'} "
                    f"got {old_lat} → corrected to {lat}"
                )
                corrected = True

    if lon is not None:
        exp = _expected_sign(raw_lon, ('W',))
        if exp is not None:
            actual_sign = -1 if lon < 0 else +1
            if actual_sign != exp:
                old_lon = lon
                lon = exp * abs(lon)
                print(
                    f"    [SIGN-GATE] {source_file} | "
                    f"lon sign mismatch: raw='{raw_lon}' expected={'W' if exp==-1 else 'E'} "
                    f"got {old_lon} → corrected to {lon}"
                )
                corrected = True

    # Always log the coordinates that are about to enter the geocoder
    lat_str = f"{lat:+.6f}" if lat is not None else "None"
    lon_str = f"{lon:+.6f}" if lon is not None else "None"
    tag     = " [corrected]" if corrected else ""
    print(f"    [PRE-GEOCODE]{tag} lat={lat_str}  lon={lon_str}  src='{raw_lat}' / '{raw_lon}'")

    return lat, lon


# ─────────────────────────────────────────────
# 4. DATE NORMALISATION  (with type tagging)
# ─────────────────────────────────────────────

def normalize_date(raw: str | None) -> dict:
    """
    Return a dict with keys:
      date      – ISO 8601 string or None
      date_type – "event" | "period" | "historical" | "unparseable" | None
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
# 4. TIME NORMALISATION
# ─────────────────────────────────────────────

def clean_time(value: str | None) -> str | None:
    """
    Normalise and validate a time string extracted from a document.

    Accepts and normalises:
      "07:15 IST"                  → "07:15 IST"
      "14:20 PM"                   → "14:20 PM"
      "2024-01-10T09:30:00Z"       → "09:30:00"   (ISO datetime)
      "2024-01-10 09:30:00"        → "09:30:00"   (ISO with space separator)
      "03:22:11 UTC"               → "03:22:11 UTC"
      "0.625"                      → "15:00:00"   (Excel fractional day)
      "45"                         → "00:00:45"   (bare seconds "SS")
      "930"  / "0930"              → "09:30:00"   (military HHMM without colon)

    Rejects:
      "."    →  None
      "IST"  →  None   (bare timezone with no time component)
      "N"    →  None   (single letter — direction bleed from coordinates)
    """
    if not value:
        return None
    s = str(value).strip()

    # ── ISO datetime  "2024-01-10T09:30:00Z"  or  "2024-01-10 09:30:00" ──
    iso_match = re.match(r'\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2}(?::\d{2})?)', s)
    if iso_match:
        return iso_match.group(1)

    # ── Standard HH:MM[:SS] with optional timezone / AM/PM suffix ─────────
    if re.match(r'^\d{1,2}:\d{2}', s):
        return s

    # ── Military HHMM without colon  "0930", "1400" ───────────────────────
    mil_match = re.match(r'^(\d{2})(\d{2})$', s)
    if mil_match:
        hh, mm = int(mil_match.group(1)), int(mil_match.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    # ── Excel fractional day  "0.625"  → 15:00:00 ─────────────────────────
    try:
        frac = float(s)
        if 0.0 <= frac < 1.0:
            total_sec = int(round(frac * 86400))
            h, rem    = divmod(total_sec, 3600)
            m, sec    = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"
    except ValueError:
        pass

    # ── Bare seconds "SS"  "45" → "00:00:45" ─────────────────────────────
    ss_match = re.match(r'^(\d{1,2})$', s)
    if ss_match:
        sec = int(ss_match.group(1))
        if 0 <= sec <= 59:
            return f"00:00:{sec:02d}"

    return None


# ─────────────────────────────────────────────
# 5. GEOCODING HELPERS
# ─────────────────────────────────────────────

def forward_geocode(place: str) -> tuple[float, float] | tuple[None, None]:
    """Place name → (lat, lon).  Returns (None, None) on failure."""
    try:
        time.sleep(1.1)
        location = geolocator.geocode(place)
        if location:
            return round(location.latitude, 6), round(location.longitude, 6)
    except GeopyError as e:
        print(f"    [GEOCODE WARN] Forward geocode failed for '{place}': {e}")
    return None, None


def _extract_label_from_nominatim(raw: dict, lat: float, lon: float, zoom: int) -> str | None:
    """
    Pull the best human-readable label out of a single Nominatim response dict.

    Tries structured address fields first (natural features, populated
    places, admin regions), then falls back to display_name.
    Returns None if nothing usable is found at this zoom level.
    """
    addr = raw.get("address", {})

    # Priority 1 — named natural / marine features
    natural_label = (
        addr.get("natural")
        or addr.get("water")
        or addr.get("sea")
        or addr.get("ocean")
        or addr.get("reef")
        or addr.get("bay")
        or addr.get("archipelago")
        or addr.get("island")
        or addr.get("peninsula")
        or addr.get("strait")
        or addr.get("marine")
    )
    # Priority 2 — populated place
    place_label = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("suburb")
        or addr.get("municipality")
        or addr.get("county")
        or addr.get("district")
    )
    # Priority 3 — administrative region
    admin_label = (
        addr.get("state")
        or addr.get("region")
        or addr.get("province")
    )
    country = addr.get("country")

    primary = natural_label or place_label or admin_label
    if primary:
        parts = [p for p in (primary,
                              admin_label if primary != admin_label else None,
                              country) if p]
        label = ", ".join(parts)
        print(f"    [REVERSE-GEO] ({lat}, {lon}) zoom={zoom} addr → '{label}'")
        return label

    # Structured address empty — try display_name
    display = raw.get("display_name", "").strip()
    if display:
        tokens = [t.strip() for t in display.split(",") if t.strip()]
        if tokens:
            label = ", ".join(tokens[:3])
            print(f"    [REVERSE-GEO] ({lat}, {lon}) zoom={zoom} display_name → '{label}'")
            return label

    return None


def _ocean_label(lat: float, lon: float) -> str | None:
    """
    Last-resort label for open-ocean coordinates where Nominatim returns
    nothing.  Uses the free BigDataCloud reverse-geocode-client endpoint
    (no API key required) which explicitly names seas and oceans.

    Falls back to a human-readable coordinate string if the API is
    unreachable, so geocoded_place is never null just because the point
    is in the ocean.
    """
    import urllib.request as _urlreq

    url = (
        f"https://api.bigdatacloud.net/data/reverse-geocode-client"
        f"?latitude={lat}&longitude={lon}&localityLanguage=en"
    )
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "rag-geo-pipeline/1.0"})
        with _urlreq.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        # BigDataCloud response structure for ocean coords:
        #   continent, countryName, principalSubdivision, city, locality
        #   For open ocean: countryName="" but continent and ocean name present
        parts = []
        locality     = data.get("locality") or data.get("city") or ""
        subdivision  = data.get("principalSubdivision") or ""
        country_name = data.get("countryName") or ""
        continent    = data.get("continent") or ""
        # Also check the 'localityInfo' block for ocean/sea names
        ocean_name = ""
        for block in data.get("localityInfo", {}).get("informative", []):
            if block.get("order", 99) <= 2:   # order 1-2 = ocean / sea level
                ocean_name = block.get("name", "")
                break

        for part in (ocean_name, locality, subdivision, country_name, continent):
            if part and part not in parts:
                parts.append(part)

        if parts:
            label = ", ".join(parts[:3])
            print(f"    [REVERSE-GEO] ({lat}, {lon}) BigDataCloud → '{label}'")
            return label

    except Exception as e:
        print(f"    [REVERSE-GEO] BigDataCloud unavailable for ({lat},{lon}): {e}")

    # Absolute last resort: human-readable coordinate string
    lat_str = f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.4f}{'E' if lon >= 0 else 'W'}"
    label   = f"{lat_str}, {lon_str}"
    print(f"    [REVERSE-GEO] ({lat}, {lon}) coord-string fallback → '{label}'")
    return label


def reverse_geocode(lat: float, lon: float) -> str | None:
    """
    (lat, lon) → human-readable label stored in geocoded_place.

    When place is null this is the ONLY location label the record will
    ever have — it must never return None for a resolvable coordinate.

    Strategy (four tiers)
    ─────────────────────
    1. Nominatim zoom ladder [18, 10, 6, 3]
         zoom 18 = street level  → often empty for ocean / wilderness
         zoom 10 = city / large area
         zoom  6 = country / region
         zoom  3 = continent
       Each zoom tries structured address fields first, then display_name.

    2. Photon (OpenStreetMap-based, different backend)
       Better at returning named water bodies that Nominatim misses.

    3. BigDataCloud free API
       Explicitly names oceans, seas, and offshore areas.
       No API key required.

    4. Coordinate string fallback  "15.1234S, 145.8765E"
       Guarantees geocoded_place is never null — the caller and any
       downstream map tool can always display something.

    The result is stored in geocoded_place, NEVER in place.
    """
    from geopy.geocoders import Photon

    ZOOM_LADDER = [18, 10, 6, 3]

    # ── Tier 1: Nominatim zoom ladder ────────────────────────────────────
    for zoom in ZOOM_LADDER:
        try:
            time.sleep(1.1)
            location = geolocator.reverse((lat, lon), language="en", zoom=zoom)
            if not location:
                continue
            label = _extract_label_from_nominatim(location.raw, lat, lon, zoom)
            if label:
                return label
        except GeopyError as e:
            print(f"    [GEOCODE WARN] Nominatim zoom={zoom} failed ({lat},{lon}): {e}")
            continue

    print(f"    [REVERSE-GEO] Nominatim exhausted for ({lat},{lon}) — trying Photon")

    # ── Tier 2: Photon ────────────────────────────────────────────────────
    try:
        photon = Photon(user_agent="rag-geo-pipeline/1.0", timeout=10)
        time.sleep(1.1)
        loc = photon.reverse((lat, lon), exactly_one=True)
        if loc:
            # Photon returns a formatted address string in loc.address
            display = (loc.address or "").strip()
            if display:
                tokens = [t.strip() for t in display.split(",") if t.strip()]
                label  = ", ".join(tokens[:3])
                print(f"    [REVERSE-GEO] ({lat}, {lon}) Photon → '{label}'")
                return label
    except Exception as e:
        print(f"    [REVERSE-GEO] Photon failed for ({lat},{lon}): {e}")

    # ── Tier 3 & 4: BigDataCloud / coord-string ───────────────────────────
    return _ocean_label(lat, lon)


# ─────────────────────────────────────────────
# 6. RECORD CLEANING  (main per-record logic)
# ─────────────────────────────────────────────

def clean_record(raw: dict, source_file: str) -> dict:
    """
    Clean and enrich a single GLiNER2-extracted record.

    Returns a record dict always — unresolvable records carry
    drop_reason and are routed to QUARANTINE by the caller; they are
    never silently discarded.

    Coordinate pipeline
    ───────────────────
    1. parse_coordinate()       — reads N/S/E/W first, applies sign, handles DMS
    2. _sanity_check_coords()   — fixes lat==lon bleed and keyword-based sign errors
    3. assert_signed_coords()   — re-verifies sign from raw strings; logs pre-geocode values
                                   N→+lat  S→−lat  E→+lon  W→−lon  (guaranteed before Nominatim)
    4. forward_geocode()        — fills coords when both are missing
    5. reverse_geocode()        — fills geocoded_place when place is missing
    """
    geocode_status = "exact"

    # ── Coordinates ──────────────────────────────────────────────────────
    lat   = parse_coordinate(raw.get("latitude"),  is_longitude=False)
    lon   = parse_coordinate(raw.get("longitude"), is_longitude=True)
    place = str(raw.get("place")).strip() if raw.get("place") else None

    # Sanity-check after parsing (fixes bleed + hemisphere sign errors)
    lat, lon, sanity_note = _sanity_check_coords(lat, lon, place)
    if sanity_note:
        print(f"    [SANITY] {source_file}: {sanity_note} | place={place}")
        if "lon_nulled" in sanity_note:
            geocode_status = "partial"

    # ── SIGNED-FLOAT GATE ────────────────────────────────────────────────
    # Final check: re-verify hemisphere sign from raw strings before any
    # Nominatim call.  Logs the exact (lat, lon) entering the geocoder.
    # Rule:  N → +lat  |  S → −lat  |  E → +lon  |  W → −lon
    lat, lon = assert_signed_coords(
        lat, lon,
        raw_lat=raw.get("latitude"),
        raw_lon=raw.get("longitude"),
        source_file=source_file,
    )

    # Forward-geocode when BOTH coords are missing
    if lat is None and lon is None and place:
        print(f"    [GEOCODE] Forward geocoding: '{place}'")
        lat, lon = forward_geocode(place)
        geocode_status = "forward" if lat is not None else "failed"

    # Partial coords (one of lat/lon is None but not both) — flag but keep
    elif (lat is None) != (lon is None):
        print(f"    [WARN] Partial coords in {source_file}: lat={lat}, lon={lon}, place={place}")
        geocode_status = "partial"

    # ── Reverse-geocode when place is null ───────────────────────────────
    # When the doc_parser extracted no place name, geocoded_place is the
    # only location label this record will ever have.  We MUST populate it
    # from the signed (lat, lon) that just passed through the gate above.
    # geocoded_place is also attempted when a sanity fix nulled longitude
    # (partial status) and forward-geocode recovered both coords.
    geocoded_place: str | None = None
    if lat is not None and lon is not None:
        if place is None:
            print(f"    [GEOCODE] place=null — reverse geocoding ({lat}, {lon})")
            geocoded_place = reverse_geocode(lat, lon)
            if geocoded_place:
                # Distinguish a real named label from the coord-string fallback
                # The coord-string fallback looks like "15.1234S, 145.8765E"
                _is_coord_fallback = bool(
                    re.match(r'^\d+\.\d+[NS],\s*\d+\.\d+[EW]$', geocoded_place)
                )
                geocode_status = "reverse_coord_fallback" if _is_coord_fallback else "reverse"
                print(f"    [GEOCODE] geocoded_place='{geocoded_place}' status={geocode_status}")
            else:
                geocode_status = "reverse_failed"
                print(f"    [GEOCODE] reverse geocode returned nothing for ({lat}, {lon})")
        elif geocode_status == "exact":
            geocoded_place = None   # place already known, skip API call

    # ── Date ─────────────────────────────────────────────────────────────
    date_info = normalize_date(raw.get("date"))

    # ── Time ─────────────────────────────────────────────────────────────
    cleaned_time = clean_time(raw.get("time"))

    # ── Assemble ──────────────────────────────────────────────────────────
    record = {
        "place":          place,
        "geocoded_place": geocoded_place,
        "latitude":       lat,
        "longitude":      lon,
        "date":           date_info["date"],
        "date_type":      date_info["date_type"],
        "date_raw":       raw.get("date"),
        "time":           cleaned_time,
        "role":           raw.get("role"),
        "summary":        raw.get("summary"),
        "geocode_status": geocode_status,
        "source_file":    source_file,
    }

    if sanity_note:
        record["sanity_note"] = sanity_note

    # Mark unresolvable records
    if place is None and geocoded_place is None and (lat is None or lon is None):
        record["drop_reason"] = "no_place_no_coords"

    return record


# ─────────────────────────────────────────────
# 7. QUARANTINE ROUTER
# ─────────────────────────────────────────────

def route_record(record: dict, cleaned: list[dict]) -> None:
    """Send the record to the cleaned list or to QUARANTINE."""
    if "drop_reason" in record:
        QUARANTINE.append(record)
        print(f"    [QUARANTINE] {record.get('source_file')} — {record['drop_reason']}")
    else:
        cleaned.append(record)


# ─────────────────────────────────────────────
# 8. MAIN LOAD & CLEAN LOOP
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

    total        = len(cleaned)
    quarantined  = len(QUARANTINE)
    with_coords  = sum(1 for r in cleaned if r["latitude"] and r["longitude"])
    with_place   = sum(1 for r in cleaned if r["place"])
    with_date    = sum(1 for r in cleaned if r["date"])
    exact        = sum(1 for r in cleaned if r["geocode_status"] == "exact")
    forward_geo  = sum(1 for r in cleaned if r["geocode_status"] == "forward")
    reverse_geo  = sum(1 for r in cleaned if r["geocode_status"] == "reverse")
    sanity_fixed = sum(1 for r in cleaned if r.get("sanity_note"))

    print(f"\n{'─'*40}")
    print(f"  ✓ Clean records        : {total}")
    print(f"  ⚠ Quarantined          : {quarantined}")
    print(f"  ✓ With coordinates     : {with_coords}/{total}")
    print(f"  ✓ With place name      : {with_place}/{total}")
    print(f"  ✓ With date            : {with_date}/{total}")
    print(f"  · Geocode — exact      : {exact}")
    print(f"  · Geocode — forward    : {forward_geo}")
    print(f"  · Geocode — reverse    : {reverse_geo}")
    print(f"  · Sanity-fixed coords  : {sanity_fixed}")
    print(f"{'─'*40}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if cleaned:
        with open(CLEANED_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {CLEANED_FILE}")
    else:
        print("\n[!] No valid records produced. Check your input JSON files.")

    if QUARANTINE:
        with open(QUARANTINE_FILE, "w", encoding="utf-8") as f:
            json.dump(QUARANTINE, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {QUARANTINE_FILE}  ({quarantined} record(s) need review)")

    print("\n  Next step: run ingest.py to embed & store in DB")


if __name__ == "__main__":
    main()