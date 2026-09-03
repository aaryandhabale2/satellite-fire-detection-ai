"""
src/fetch_firms_archive.py
--------------------------
Downloads up to 1 full year of historical NASA FIRMS hotspot data for India
by chunking the date range into 5-day windows and calling the FIRMS Area CSV
API with the optional DATE parameter:

    /api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAY_RANGE}/{DATE}

This is the SAME endpoint used by fetch_firms_data.py — no extra credentials
needed.  The DATE parameter is the only addition; it makes the API return
data starting from that date instead of "most recent".

Sources fetched (default):
  - VIIRS_SNPP_NRT  (375 m, will be discontinued Nov 1 2026 — also fetches
                     VIIRS_NOAA20_NRT as replacement if env flag set)
  - MODIS_NRT       (1 km, Terra + Aqua)

Output files
------------
  data/archive_chunks/          -- one CSV per (source, month) chunk
  data/firms_archive_raw.csv    -- merged, de-duplicated full dataset

Environment variables
---------------------
  FIRMS_MAP_KEY          -- required NASA FIRMS MAP_KEY
  FIRMS_ARCHIVE_START    -- YYYY-MM-DD start date  (default: 1 year ago)
  FIRMS_ARCHIVE_END      -- YYYY-MM-DD end date    (default: yesterday)
  FIRMS_ARCHIVE_SOURCES  -- comma-separated sources
                            (default: VIIRS_SNPP_NRT,MODIS_NRT)
  FIRMS_INCLUDE_NOAA20   -- "true" to also fetch VIIRS_NOAA20_NRT

Usage
-----
    python src/fetch_firms_archive.py
"""

import io
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import requests
import scipy.spatial
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAP_KEY = os.getenv("FIRMS_MAP_KEY", "").strip()

# Date range defaults: 1 year of data ending yesterday
_today     = date.today()
_yesterday = _today - timedelta(days=1)
_one_year_ago = _today - timedelta(days=365)

ARCHIVE_START_STR = os.getenv("FIRMS_ARCHIVE_START", _one_year_ago.strftime("%Y-%m-%d")).strip()
ARCHIVE_END_STR   = os.getenv("FIRMS_ARCHIVE_END",   _yesterday.strftime("%Y-%m-%d")).strip()

# Sources to fetch
_DEFAULT_ARCHIVE_SOURCES = ["VIIRS_SNPP_NRT", "MODIS_NRT"]
_SOURCES_ENV = os.getenv("FIRMS_ARCHIVE_SOURCES", "").strip()
ARCHIVE_SOURCES = (
    [s.strip() for s in _SOURCES_ENV.split(",") if s.strip()]
    if _SOURCES_ENV
    else list(_DEFAULT_ARCHIVE_SOURCES)
)

# Optionally add NOAA-20 as VIIRS S-NPP replacement
INCLUDE_NOAA20 = os.getenv("FIRMS_INCLUDE_NOAA20", "false").strip().lower() in ("true", "1", "yes")
if INCLUDE_NOAA20 and "VIIRS_NOAA20_NRT" not in ARCHIVE_SOURCES:
    ARCHIVE_SOURCES.append("VIIRS_NOAA20_NRT")

# FIRMS API settings
FIRMS_URL     = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
INDIA_BBOX    = "68.1,7.9,97.4,35.5"
CHUNK_DAYS    = 5       # days per API request (max allowed by FIRMS)
REQUEST_DELAY = 1.5     # seconds between requests (polite rate-limiting)
REQUEST_TIMEOUT = 90    # seconds per HTTP request

# NRT sources only retain a rolling ~5-month window.
# Anything older must use Standard Processing (SP) sources.
# This cutoff determines which variant to use per chunk automatically.
NRT_CUTOFF_DAYS = 150   # chunks older than this use SP sources

# Map: NRT source  →  SP equivalent (used for older date chunks)
NRT_TO_SP = {
    "VIIRS_SNPP_NRT"  : "VIIRS_SNPP_SP",
    "MODIS_NRT"       : "MODIS_SP",
    "VIIRS_NOAA20_NRT": "VIIRS_NOAA20_SP",
}

# Coincidence detection (same logic as fetch_firms_data.py)
COINCIDENT_RADIUS_M   = 1000.0
COINCIDENT_TIME_HOURS = 6.0
# Skip coincident detection above this size — cKDTree query_pairs is O(n²)
# and hangs on 800K+ rows. co_confirmed=False is set as default instead.
COINCIDENT_SKIP_THRESHOLD = 100_000

# Paths
ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT / "data"
CHUNK_DIR    = DATA_DIR / "archive_chunks"
OUTPUT_FILE  = DATA_DIR / "firms_archive_raw.csv"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config() -> Tuple[date, date]:
    """Validate env config and return (start_date, end_date)."""
    if not MAP_KEY:
        print(
            "\n[ERROR] FIRMS_MAP_KEY is not set.\n"
            "  Steps to fix:\n"
            "    1. Copy .env.example to .env\n"
            "    2. Open .env and set:  FIRMS_MAP_KEY=your_actual_key\n"
            "    3. Get a free key at:  https://firms.modaps.eosdis.nasa.gov/api/map_key\n"
        )
        sys.exit(1)

    try:
        start = date.fromisoformat(ARCHIVE_START_STR)
    except ValueError:
        print(f"[ERROR] FIRMS_ARCHIVE_START is not a valid date: {ARCHIVE_START_STR!r}")
        sys.exit(1)

    try:
        end = date.fromisoformat(ARCHIVE_END_STR)
    except ValueError:
        print(f"[ERROR] FIRMS_ARCHIVE_END is not a valid date: {ARCHIVE_END_STR!r}")
        sys.exit(1)

    if end <= start:
        print(f"[ERROR] FIRMS_ARCHIVE_END ({end}) must be after FIRMS_ARCHIVE_START ({start})")
        sys.exit(1)

    return start, end


# ---------------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------------

def generate_date_chunks(start: date, end: date, chunk_days: int = CHUNK_DAYS) -> List[Tuple[date, int]]:
    """
    Split [start, end] into chunks of at most chunk_days each.

    Returns a list of (chunk_start_date, effective_days) tuples where
    effective_days <= chunk_days and the window never extends past end.
    """
    chunks = []
    current = start
    while current < end:
        days_remaining = (end - current).days
        effective = min(chunk_days, days_remaining)
        chunks.append((current, effective))
        current += timedelta(days=effective)
    return chunks


# ---------------------------------------------------------------------------
# Single chunk fetch
# ---------------------------------------------------------------------------

def fetch_chunk(source: str, chunk_start: date, chunk_days: int, chunk_index: int, total_chunks: int) -> pd.DataFrame:
    """
    Download one 5-day chunk from FIRMS for the given source and start date.

    Automatically switches to the SP (Standard Processing) source when
    chunk_start is older than NRT_CUTOFF_DAYS, because NRT endpoints only
    keep a rolling ~5-month window.

    Returns an empty DataFrame on any error (loop continues safely).
    """
    # Auto-select SP vs NRT based on age of the chunk
    days_ago = (date.today() - chunk_start).days
    if days_ago > NRT_CUTOFF_DAYS and source in NRT_TO_SP:
        effective_source = NRT_TO_SP[source]
    else:
        effective_source = source

    url = f"{FIRMS_URL}/{MAP_KEY}/{effective_source}/{INDIA_BBOX}/{chunk_days}/{chunk_start.isoformat()}"
    safe_url = url.replace(MAP_KEY, "MAP_KEY_HIDDEN")

    print(
        f"  [{chunk_index+1:03d}/{total_chunks:03d}] {effective_source}  {chunk_start}  ({chunk_days}d)  {safe_url}",
        flush=True,
    )

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        print(f"    [WARN] Network error: {exc} — skipping chunk.", flush=True)
        return pd.DataFrame()

    if response.status_code != 200:
        print(f"    [WARN] HTTP {response.status_code} — skipping chunk.", flush=True)
        return pd.DataFrame()

    body = response.text.strip()
    if not body:
        return pd.DataFrame()

    if body.lower().startswith("invalid") or "map_key" in body.lower() or "error" in body[:100].lower():
        print(f"    [WARN] API error message: {body[:200]}", flush=True)
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(body))
    except Exception as exc:
        print(f"    [WARN] CSV parse error: {exc} — skipping chunk.", flush=True)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Standardise satellite_source column
    if "VIIRS_SNPP" in source:
        df["satellite_source"] = "VIIRS_SNPP"
        if "instrument" not in df.columns:
            df["instrument"] = "VIIRS"
    elif "VIIRS_NOAA20" in source:
        df["satellite_source"] = "VIIRS_NOAA20"
        if "instrument" not in df.columns:
            df["instrument"] = "VIIRS"
    elif "MODIS" in source:
        df["satellite_source"] = "MODIS"
        if "instrument" not in df.columns:
            df["instrument"] = "MODIS"
    else:
        df["satellite_source"] = source

    # Harmonise brightness column names across sensors
    if "bright_ti4" in df.columns and "brightness" not in df.columns:
        df["brightness"] = df["bright_ti4"]
    elif "brightness" in df.columns and "bright_ti4" not in df.columns:
        df["bright_ti4"] = df["brightness"]

    return df


# ---------------------------------------------------------------------------
# Coincident detection (ported from fetch_firms_data.py — keep in sync)
# ---------------------------------------------------------------------------

def detect_coincident_hotspots(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Tag rows where VIIRS and MODIS both detected the same hotspot within
    ~1 km and 6 hours.  Preserves both records; adds co_confirmed column.

    NOTE: Skipped for datasets > COINCIDENT_SKIP_THRESHOLD rows because
    cKDTree.query_pairs() is O(n²) in memory and hangs on 800K+ rows.
    co_confirmed is set to False for all rows in that case.
    """
    if len(df) == 0:
        df["co_confirmed"] = False
        return df, 0

    if len(df) > COINCIDENT_SKIP_THRESHOLD:
        print(f"  [INFO] Dataset has {len(df):,} rows — skipping coincident detection "
              f"(too large for KDTree query_pairs). co_confirmed=False for all rows.", flush=True)
        df = df.copy()
        df["co_confirmed"] = False
        return df, 0

    df = df.copy().reset_index(drop=True)
    df["co_confirmed"] = False

    lat_rad = np.radians(df["latitude"].values)
    lon_rad = np.radians(df["longitude"].values)
    x = 6_371_000.0 * lon_rad * np.cos(lat_rad)
    y = 6_371_000.0 * lat_rad
    tree = scipy.spatial.cKDTree(np.column_stack([x, y]))
    candidate_pairs = tree.query_pairs(r=COINCIDENT_RADIUS_M)

    def _parse_time_min(val) -> float:
        try:
            s = str(int(val)).zfill(4)
            return int(s[:2]) * 60.0 + int(s[2:])
        except Exception:
            return 0.0

    times_min = df["acq_time"].apply(_parse_time_min).values
    dates     = df["acq_date"].astype(str).values
    sources   = df["satellite_source"].values

    confirmed: Set[int] = set()
    for i, j in candidate_pairs:
        if sources[i] != sources[j] and dates[i] == dates[j]:
            if abs(times_min[i] - times_min[j]) / 60.0 <= COINCIDENT_TIME_HOURS:
                confirmed.add(i)
                confirmed.add(j)

    for idx in confirmed:
        df.loc[idx, "co_confirmed"] = True

    return df, len(confirmed)


# ---------------------------------------------------------------------------
# Chunk saving (crash-safe: each month saved independently)
# ---------------------------------------------------------------------------

def save_chunk_to_disk(df: pd.DataFrame, source: str, chunk_start: date) -> Path:
    """Save a single chunk to archive_chunks/ using a stable filename."""
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    # Use month-level grouping for filenames (multiple 5-day chunks may share a month)
    month_str   = chunk_start.strftime("%Y-%m")
    safe_source = source.replace("_NRT", "").replace("_", "-")
    filename    = CHUNK_DIR / f"{safe_source}_{month_str}_{chunk_start.isoformat()}.csv"
    df.to_csv(filename, index=False)
    return filename


def load_existing_chunks() -> pd.DataFrame:
    """Load all previously-saved chunk CSVs (for resume-after-crash support)."""
    csvs = list(CHUNK_DIR.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()
    dfs = []
    for f in csvs:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  NASA FIRMS — 1-YEAR HISTORICAL ARCHIVE FETCH")
    print("=" * 70)

    start_date, end_date = validate_config()
    total_days = (end_date - start_date).days

    print(f"\n  Date range  : {start_date}  →  {end_date}  ({total_days} days)")
    print(f"  Sources     : {', '.join(ARCHIVE_SOURCES)}")
    print(f"  Chunk size  : {CHUNK_DAYS} days per request")
    print(f"  India bbox  : {INDIA_BBOX}")
    print(f"  Output      : {OUTPUT_FILE}")
    print()

    # Generate all (source, chunk_start, chunk_days) jobs
    date_chunks = generate_date_chunks(start_date, end_date, CHUNK_DAYS)
    all_jobs: List[Tuple[str, date, int]] = [
        (src, cs, cd)
        for src in ARCHIVE_SOURCES
        for cs, cd in date_chunks
    ]
    total_jobs = len(all_jobs)
    print(f"  Total API requests planned : {total_jobs}  "
          f"({len(date_chunks)} chunks × {len(ARCHIVE_SOURCES)} source(s))\n")

    # Resume: skip chunks already on disk
    already_done: set = set()
    if CHUNK_DIR.exists():
        for f in CHUNK_DIR.glob("*.csv"):
            # Filename encodes source and start date — use stem as key
            already_done.add(f.stem)

    # Fetch loop
    all_dfs: List[pd.DataFrame] = []
    skipped_count = 0
    fetched_rows  = 0

    for idx, (source, chunk_start, chunk_days) in enumerate(all_jobs):
        safe_source = source.replace("_NRT", "").replace("_", "-")
        chunk_key   = f"{safe_source}_{chunk_start.strftime('%Y-%m')}_{chunk_start.isoformat()}"

        if chunk_key in already_done:
            print(f"  [{idx+1:03d}/{total_jobs:03d}] SKIPPED (already fetched): {chunk_key}", flush=True)
            skipped_count += 1
            continue

        df = fetch_chunk(source, chunk_start, chunk_days, idx, total_jobs)

        if not df.empty:
            save_chunk_to_disk(df, source, chunk_start)
            all_dfs.append(df)
            fetched_rows += len(df)
            print(f"    → {len(df):,} rows  (running total: {fetched_rows:,})", flush=True)
        else:
            print(f"    → 0 rows (no fires detected for this chunk)", flush=True)

        time.sleep(REQUEST_DELAY)

    print(f"\n  Fetch complete. {fetched_rows:,} new rows fetched.")
    if skipped_count:
        print(f"  {skipped_count} chunks skipped (already on disk — loaded from cache).")

    # ── Load ALL chunks (new + previously cached) ──────────────────────────
    print("\n  Loading all chunks from disk for final merge ...", flush=True)
    combined = load_existing_chunks()

    if combined.empty:
        print("\n[ERROR] No data found in any chunk — check your MAP_KEY and date range.")
        sys.exit(1)

    print(f"  {len(combined):,} total rows before de-duplication.")

    # ── Remove exact duplicates (same lat/lon/date/acq_time/source) ──────
    before_dedup = len(combined)
    dedup_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time", "satellite_source"]
                  if c in combined.columns]
    combined.drop_duplicates(subset=dedup_cols, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    after_dedup = len(combined)
    print(f"  {before_dedup - after_dedup:,} exact duplicates removed → {after_dedup:,} unique rows.")

    # ── Multi-sensor coincident detection ─────────────────────────────────
    print("\n  Running multi-sensor coincident detection ...", flush=True)
    combined, co_count = detect_coincident_hotspots(combined)
    print(f"  {co_count:,} co-confirmed hotspot detections (flagged, both records kept).")

    # ── Save final merged file ─────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    actual_start = pd.to_datetime(combined["acq_date"]).min().date()
    actual_end   = pd.to_datetime(combined["acq_date"]).max().date()

    print("\n" + "=" * 70)
    print("  PHASE 1 RESULTS — firms_archive_raw.csv")
    print("=" * 70)
    print(f"  Total rows              : {len(combined):,}")
    print(f"  Date range (actual)     : {actual_start}  →  {actual_end}")
    print(f"  Days covered            : {(actual_end - actual_start).days}")
    print()
    print("  Breakdown by satellite source:")
    for src, cnt in combined["satellite_source"].value_counts().items():
        pct = cnt / len(combined) * 100
        print(f"    {src:<20}: {cnt:>8,}  ({pct:.1f}%)")
    print()
    print("  Monthly distribution (top months by hotspot count):")
    combined["_month"] = pd.to_datetime(combined["acq_date"]).dt.to_period("M")
    monthly = combined.groupby("_month").size().sort_values(ascending=False)
    for period, count in monthly.head(6).items():
        print(f"    {str(period):<12}: {count:>8,} hotspots")
    combined.drop(columns=["_month"], inplace=True, errors="ignore")
    print()
    print(f"  Saved to: {OUTPUT_FILE}")
    print("=" * 70)
    print()
    print("[PHASE 1 COMPLETE] Review the row count and date range above.")
    print("                   If it looks correct, proceed to Phase 2:")
    print("                   python src/fetch_osm_landuse_bulk.py")
    print()


if __name__ == "__main__":
    main()
