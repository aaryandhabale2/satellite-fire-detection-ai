"""
src/fetch_firms_data.py
-----------------------
Fetches NASA FIRMS thermal hotspot / fire data for India and saves
the result to data/firms_raw.csv.

Supports multi-satellite sensor fusion:
  1. VIIRS_SNPP_NRT   -- VIIRS S-NPP (~375 m, near-real-time)
  2. MODIS_NRT        -- MODIS Terra/Aqua (~1 km, near-real-time)
  3. VIIRS_NOAA20_NRT -- VIIRS NOAA-20 (~375 m, near-real-time, optional/drop-in)

NOTE ON VIIRS S-NPP SUNSET (Nov 1, 2026):
NASA is discontinuing VIIRS_SNPP data delivery on November 1, 2026.
VIIRS_NOAA20_NRT is supported in this pipeline as an active drop-in replacement
or simultaneous 3rd source to ensure continuous operational readiness.

Usage
-----
    python src/fetch_firms_data.py

Environment variables
---------------------
    FIRMS_MAP_KEY         -- your NASA FIRMS MAP_KEY (required)
                             Get one free at: https://firms.modaps.eosdis.nasa.gov/api/map_key
    FIRMS_DAY_RANGE       -- days of data to fetch (default: 5, max 5 for MODIS Area API)
    FIRMS_SOURCES         -- comma-separated list of sources to fetch
                             (default: VIIRS_SNPP_NRT,MODIS_NRT)
    FIRMS_INCLUDE_NOAA20  -- set "true" to include VIIRS_NOAA20_NRT as 3rd source (default: false)
    FIRMS_DEDUPLICATE     -- set "true" to merge coincident detections instead of
                             keeping both as separate confirmations (default: false)
"""

import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Force UTF-8 output so Unicode characters print correctly on Windows
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

# Default to 5 days because NASA FIRMS MODIS Area API caps day range at 5
DAY_RANGE_RAW = os.getenv("FIRMS_DAY_RANGE", "5").strip()

# Default active sources
DEFAULT_SOURCES = ["VIIRS_SNPP_NRT", "MODIS_NRT"]
SOURCES_ENV = os.getenv("FIRMS_SOURCES", "").strip()
if SOURCES_ENV:
    ACTIVE_SOURCES = [s.strip() for s in SOURCES_ENV.split(",") if s.strip()]
else:
    ACTIVE_SOURCES = list(DEFAULT_SOURCES)

# Toggle for NOAA-20 (prepare for S-NPP sunset on Nov 1, 2026)
INCLUDE_NOAA20 = os.getenv("FIRMS_INCLUDE_NOAA20", "false").strip().lower() in ("true", "1", "yes")
if INCLUDE_NOAA20 and "VIIRS_NOAA20_NRT" not in ACTIVE_SOURCES:
    ACTIVE_SOURCES.append("VIIRS_NOAA20_NRT")

# Duplicate handling toggle:
# Default False -> preserve both with co_confirmed=True (recommended: multi-sensor confirmation)
# Set True -> merge pairs into single record, taking higher spatial resolution (VIIRS) and max FRP
DEDUPLICATE_COINCIDENT = os.getenv("FIRMS_DEDUPLICATE", "false").strip().lower() in ("true", "1", "yes")

# India bounding box: west, south, east, north
INDIA_BBOX = "68.1,7.9,97.4,35.5"

# FIRMS Area CSV endpoint
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Output paths
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "firms_raw.csv"

# Spatial-temporal coincidence thresholds for dual-sensor verification
COINCIDENT_RADIUS_M = 1000.0   # ~1 km (accounts for MODIS 1 km footprint)
COINCIDENT_TIME_HOURS = 6.0    # 6 hours max window between orbital overpasses


# ---------------------------------------------------------------------------
# Validation & URL Construction
# ---------------------------------------------------------------------------

def validate_config() -> int:
    """Exit early with a clear message if configuration is invalid."""
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
        day_int = int(DAY_RANGE_RAW)
        if not (1 <= day_int <= 10):
            raise ValueError
    except ValueError:
        print(
            f"[ERROR] FIRMS_DAY_RANGE must be an integer between 1 and 10.\n"
            f"        Current value: {DAY_RANGE_RAW!r}"
        )
        sys.exit(1)

    return day_int


def build_source_url(source: str, day_range: int) -> Tuple[str, int]:
    """
    Construct the FIRMS Area CSV request URL for a specific satellite source.
    Clamps MODIS_NRT to max 5 days due to NASA FIRMS Area API constraints.
    """
    effective_days = day_range
    if source == "MODIS_NRT" and day_range > 5:
        effective_days = 5
        print(f"  [NOTE] Clamping MODIS_NRT day range to 5 (NASA FIRMS Area API limit for MODIS).")

    url = f"{FIRMS_URL}/{MAP_KEY}/{source}/{INDIA_BBOX}/{effective_days}"
    return url, effective_days


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_single_source(source: str, day_range: int) -> pd.DataFrame:
    """
    Download CSV from NASA FIRMS for one source and parse into a DataFrame.
    """
    url, effective_days = build_source_url(source, day_range)
    print(f"\n--> Fetching {source} ({effective_days} day(s)) ...")
    print(f"    URL: {url.replace(MAP_KEY, 'MAP_KEY_HIDDEN')}")

    try:
        response = requests.get(url, timeout=60)
    except requests.exceptions.RequestException as exc:
        print(f"  [ERROR] Network error fetching {source}: {exc}")
        return pd.DataFrame()

    if response.status_code != 200:
        print(
            f"  [ERROR] HTTP {response.status_code} from FIRMS for {source}.\n"
            f"          Response: {response.text[:200]}"
        )
        return pd.DataFrame()

    body = response.text.strip()
    if not body:
        print(f"  [WARN] Empty response received for {source}.")
        return pd.DataFrame()

    if body.lower().startswith("invalid") or "map_key" in body.lower() or "error" in body[:100].lower():
        print(f"  [ERROR] FIRMS API error message for {source}:\n          {body[:250]}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(body))
    except Exception as exc:
        print(f"  [ERROR] Failed to parse CSV for {source}: {exc}")
        return pd.DataFrame()

    # Assign metadata
    if "VIIRS_SNPP" in source:
        df["satellite_source"] = "VIIRS_SNPP"
        if "instrument" not in df.columns:
            df["instrument"] = "VIIRS"
    elif "MODIS" in source:
        df["satellite_source"] = "MODIS"
        if "instrument" not in df.columns:
            df["instrument"] = "MODIS"
    elif "NOAA20" in source:
        df["satellite_source"] = "VIIRS_NOAA20"
        if "instrument" not in df.columns:
            df["instrument"] = "VIIRS"
    else:
        df["satellite_source"] = source

    # Ensure shared brightness column is available
    if "bright_ti4" in df.columns and "brightness" not in df.columns:
        df["brightness"] = df["bright_ti4"]
    elif "brightness" in df.columns and "bright_ti4" not in df.columns:
        df["bright_ti4"] = df["brightness"]

    print(f"  [OK] Successfully retrieved {len(df):,} records for {source}.")
    return df


# ---------------------------------------------------------------------------
# Duplicate / Co-Confirmation Handling
# ---------------------------------------------------------------------------

def detect_coincident_hotspots(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identifies multi-sensor coincident detections within ~1 km (matching MODIS footprint)
    and within 6 hours of each other on the same day.

    Handling Strategy:
      - Default (DEDUPLICATE_COINCIDENT=False):
        Multi-Sensor Co-Confirmation (Preserve Both + Tag `co_confirmed=True`).
        Preserves temporal evolution across overpasses (Terra morning, Aqua afternoon,
        VIIRS afternoon) and strengthens historical frequency without discarding data.
      - Deduplication Mode (DEDUPLICATE_COINCIDENT=True):
        Merges coincident pairs into one record, retaining higher-resolution VIIRS coordinates
        and taking the maximum FRP across both sensors.
    """
    if len(df) == 0:
        df["co_confirmed"] = False
        return df, 0

    df = df.copy().reset_index(drop=True)
    df["co_confirmed"] = False

    # Convert coordinates to projected meters for KDTree
    lat_rad = np.radians(df["latitude"].values)
    lon_rad = np.radians(df["longitude"].values)
    x = 6371000.0 * lon_rad * np.cos(lat_rad)
    y = 6371000.0 * lat_rad

    coords = np.column_stack([x, y])
    tree = scipy.spatial.cKDTree(coords)
    candidate_pairs = tree.query_pairs(r=COINCIDENT_RADIUS_M)

    # Parse acquisition times into minutes from midnight
    def parse_time_minutes(val) -> float:
        try:
            s = str(int(val)).zfill(4)
            h = int(s[:2])
            m = int(s[2:])
            return h * 60.0 + m
        except Exception:
            return 0.0

    times_min = df["acq_time"].apply(parse_time_minutes).values
    dates = df["acq_date"].astype(str).values
    sources = df["satellite_source"].values

    co_confirmed_indices: Set[int] = set()
    pairs_to_merge: List[Tuple[int, int]] = []

    for i, j in candidate_pairs:
        # Must be from different satellite sources
        if sources[i] != sources[j]:
            # Must be same acquisition date
            if dates[i] == dates[j]:
                # Must be within time window (default 6 hours)
                dt_hours = abs(times_min[i] - times_min[j]) / 60.0
                if dt_hours <= COINCIDENT_TIME_HOURS:
                    co_confirmed_indices.add(i)
                    co_confirmed_indices.add(j)
                    pairs_to_merge.append((i, j))

    for idx in co_confirmed_indices:
        df.loc[idx, "co_confirmed"] = True

    if not DEDUPLICATE_COINCIDENT:
        return df, len(co_confirmed_indices)

    # If deduplication mode is requested:
    # Retain the higher-resolution VIIRS record, take max FRP, and drop the MODIS record
    drop_indices = set()
    for i, j in pairs_to_merge:
        if i in drop_indices or j in drop_indices:
            continue
        # Check which one is VIIRS
        src_i, src_j = sources[i], sources[j]
        if "VIIRS" in src_i and "MODIS" in src_j:
            viirs_idx, modis_idx = i, j
        elif "MODIS" in src_i and "VIIRS" in src_j:
            viirs_idx, modis_idx = j, i
        else:
            viirs_idx, modis_idx = i, j

        # Merge FRP to maximum observed
        if "frp" in df.columns:
            frp_max = max(float(df.loc[viirs_idx, "frp"]), float(df.loc[modis_idx, "frp"]))
            df.loc[viirs_idx, "frp"] = frp_max

        df.loc[viirs_idx, "satellite_source"] = f"{df.loc[viirs_idx, 'satellite_source']}+MODIS"
        drop_indices.add(modis_idx)

    deduped_df = df.drop(index=list(drop_indices)).reset_index(drop=True)
    print(f"  [DEDUPLICATION] Merged {len(drop_indices)} coincident MODIS detections into VIIRS records.")
    return deduped_df, len(co_confirmed_indices)


# ---------------------------------------------------------------------------
# Output & Summary
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame):
    """Save the combined DataFrame to data/firms_raw.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved combined {len(df):,} records to {OUTPUT_FILE}")


def print_summary(
    df: pd.DataFrame,
    source_counts: Dict[str, int],
    viirs_count_before: int,
    co_confirmed_count: int,
):
    """Print an end-to-end human readable summary of the multi-satellite ingestion."""
    print("\n" + "=" * 65)
    print("  FIRMS MULTI-SATELLITE DATA SUMMARY — firms_raw.csv")
    print("=" * 65)

    print(f"  Active Sources Fetched   : {', '.join(ACTIVE_SOURCES)}")
    print(f"  Total Hotspots (VIIRS)   : {viirs_count_before:,}  (before adding MODIS)")
    print(f"  Total Hotspots Combined  : {len(df):,}  (after adding MODIS)")

    diff = len(df) - viirs_count_before
    pct = (diff / viirs_count_before * 100) if viirs_count_before > 0 else 0
    print(f"  Net Observation Increase : +{diff:,} (+{pct:.1f}%)")

    print("\n  Breakdown by Satellite Source:")
    for src, count in source_counts.items():
        print(f"    - {src:<18}: {count:,} ({count / len(df) * 100:.1f}%)")

    print(f"\n  Multi-Sensor Cross Confirmations (within ~1km & 6h):")
    print(f"    - Co-confirmed hotspots: {co_confirmed_count:,}")
    strategy_str = "Strict Merged" if DEDUPLICATE_COINCIDENT else "Preserved both as independent confirmations (Flagged)"
    print(f"    - Duplicate Strategy   : {strategy_str}")

    # S-NPP Sunset Alert
    print("\n  [FUTURE-PROOFING] NASA VIIRS S-NPP Discontinuation:")
    print("    - Notice: NASA is discontinuing VIIRS_SNPP data delivery on November 1, 2026.")
    if "VIIRS_NOAA20_NRT" in ACTIVE_SOURCES:
        print("    - Status: VIIRS_NOAA20_NRT is ALREADY ACTIVE and ingesting in this build.")
    else:
        print("    - Status: VIIRS_NOAA20_NRT is ready. Enable via FIRMS_INCLUDE_NOAA20=true.")

    # Radiometric statistics
    print("\n  Combined Sensor Radiometrics:")
    if "brightness" in df.columns:
        s = df["brightness"].dropna()
        print(f"    - brightness (K)       : min={s.min():.1f}  max={s.max():.1f}  mean={s.mean():.1f}")
    if "frp" in df.columns:
        s = df["frp"].dropna()
        print(f"    - frp (MW)             : min={s.min():.1f}  max={s.max():.1f}  mean={s.mean():.1f}")

    print("=" * 65)
    print("\n[STEP 1 COMPLETE] Multi-satellite data is ready in data/firms_raw.csv.\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    day_range = validate_config()

    print("\n" + "=" * 65)
    print("  NASA FIRMS MULTI-SATELLITE INGESTION PIPELINE")
    print(f"  Target Area: India ({INDIA_BBOX})")
    print("=" * 65)

    source_dfs: List[pd.DataFrame] = []
    source_counts: Dict[str, int] = {}

    # 1. Fetch each source
    for source in ACTIVE_SOURCES:
        df_src = fetch_single_source(source, day_range)
        if not df_src.empty:
            source_dfs.append(df_src)
            source_name = df_src["satellite_source"].iloc[0]
            source_counts[source_name] = len(df_src)

    if not source_dfs:
        print("\n[ERROR] No data could be retrieved from any configured satellite source.")
        sys.exit(1)

    # 2. Merge all datasets
    combined_df = pd.concat(source_dfs, ignore_index=True)

    # Count VIIRS before adding MODIS
    viirs_count_before = sum(
        count for src, count in source_counts.items() if "VIIRS" in src
    )

    # 3. Detect / handle coincident detections
    final_df, co_confirmed_count = detect_coincident_hotspots(combined_df)

    # 4. Save to CSV
    save_csv(final_df)

    # 5. Print comprehensive summary
    print_summary(final_df, source_counts, viirs_count_before, co_confirmed_count)


if __name__ == "__main__":
    main()
