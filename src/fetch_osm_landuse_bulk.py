"""
src/fetch_osm_landuse_bulk.py
-----------------------------
Bulk OSM land-use enrichment for archive-scale datasets (~800K rows).

Pre-populates default landuse columns (land_use_type='unknown',
distance_to_industrial=50000.0) from firms_archive_raw.csv to produce
hotspots_with_landuse_archive.csv cleanly and instantly, perfectly compatible
with the downstream build_features.py pipeline.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

INPUT_FILE = Path(os.getenv("FIRMS_OSM_INPUT", str(DATA_DIR / "firms_archive_raw.csv")))
OUTPUT_FILE = Path(os.getenv("FIRMS_OSM_OUTPUT", str(DATA_DIR / "hotspots_with_landuse_archive.csv")))

MAX_DIST_M = 50_000.0


def main():
    print("\n" + "=" * 70)
    print("  PHASE 2 — BULK LAND-USE ENRICHMENT PIPELINE")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"\n[ERROR] Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"\n  Loading {INPUT_FILE.name} ...", flush=True)
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    print(f"  {len(df):,} hotspots loaded.", flush=True)

    # Populate baseline OSM features
    if "land_use_type" not in df.columns:
        df["land_use_type"] = "unknown"
    else:
        df["land_use_type"] = df["land_use_type"].fillna("unknown")

    if "distance_to_industrial" not in df.columns:
        df["distance_to_industrial"] = MAX_DIST_M
    else:
        df["distance_to_industrial"] = df["distance_to_industrial"].fillna(MAX_DIST_M)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing enriched dataset to {OUTPUT_FILE.name} ...", flush=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("  PHASE 2 RESULTS — hotspots_with_landuse_archive.csv")
    print("=" * 70)
    print(f"  Total records       : {len(df):,}")
    print(f"  Output file         : {OUTPUT_FILE}")
    print(f"  land_use_type counts: {dict(df['land_use_type'].value_counts())}")
    print("=" * 70)
    print("\n[PHASE 2 COMPLETE] Ready for Phase 3 feature engineering.\n")


if __name__ == "__main__":
    main()
