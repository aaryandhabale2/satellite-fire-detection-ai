"""
src/build_features.py
----------------------
Feature engineering step for the industrial fire detection pipeline.

Reads data/hotspots_with_landuse.csv and computes the following features
per hotspot, then saves the result to data/features_labeled.csv.

Features computed
-----------------
  From raw FIRMS data (already present):
    brightness          -- brightness temperature in Kelvin (bright_ti4 or
                           brightness, whichever column is available)
    frp                 -- Fire Radiative Power in MW
    log_frp             -- log1p(frp) to compress the skewed distribution

  From OSM enrichment (already present):
    land_use_type       -- industrial | forest | farmland | residential | unknown
    distance_to_industrial -- metres to nearest OSM industrial geometry

  Computed here:
    historical_frequency -- how many times a hotspot was detected within
                            ~1 km of this location across the dataset window.
                            Distinguishes persistent industrial sources (high)
                            from one-off fire events (low).
    time_of_day          -- coarse time bucket from acq_time:
                            Night (00-05h) | Morning (06-11h) |
                            Afternoon (12-17h) | Evening (18-23h)
    season               -- meteorological season from acq_date month:
                            Winter (Dec-Feb) | Summer (Mar-May) |
                            Monsoon (Jun-Sep) | Post-Monsoon (Oct-Nov)

  Heuristic label (computed here):
    category             -- one of:
                            "Industrial (Normal)"
                            "Wildfire Risk"
                            "Agricultural Burning"
                            "Anomaly/Unclassified"

Heuristic labelling rules (in priority order)
----------------------------------------------
  Rule 1 — Industrial (Normal):
    land_use_type == "industrial"  AND  historical_frequency >= HIGH_FREQ_THRESH

  Rule 2 — Wildfire Risk:
    land_use_type in {forest, farmland, unknown}
    AND historical_frequency < HIGH_FREQ_THRESH  (sudden, one-off event)
    AND frp >= HIGH_FRP_THRESH  (high intensity)

  Rule 3 — Agricultural Burning:
    land_use_type == "farmland"
    AND month in {October, November}  (post-kharif stubble burning season)
    AND historical_frequency >= REPEAT_THRESH  (seasonal repetition)

  Rule 4 — Anomaly/Unclassified:
    doesn't match any rule above

Usage
-----
    python src/build_features.py
"""

import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
INPUT_FILE  = ROOT / "data" / "hotspots_with_landuse.csv"
OUTPUT_FILE = ROOT / "data" / "features_labeled.csv"

# ---------------------------------------------------------------------------
# Thresholds for heuristic labelling
# ---------------------------------------------------------------------------
HIGH_FREQ_THRESH = 3     # >= 3 detections at same ~1 km location -> persistent
REPEAT_THRESH    = 2     # >= 2 detections -> "repeated" (for agri burning rule)
HIGH_FRP_THRESH  = 50.0  # MW — "high intensity" fire
AGRI_MONTHS      = {10, 11}  # October and November (post-kharif stubble burning)
NEAR_IND_DIST_M  = 1_000     # within 1 km of industry -> consider "industrial"

# ~1 km in decimal degrees (at India's latitudes, 0.01 deg ~ 1.1 km)
FREQ_BUCKET_DEG  = 0.01


# ---------------------------------------------------------------------------
# Feature 1: historical_frequency
# ---------------------------------------------------------------------------

def compute_historical_frequency(df: pd.DataFrame) -> pd.Series:
    """
    Count how many times each ~1 km location was detected as a hotspot
    within the dataset window (up to 30 days from the data's date range).

    Method:
      - Round lat/lon to 0.01 degree buckets (each bucket ~1 km x 1 km)
      - For each hotspot, count all other rows in the same bucket
        whose acq_date falls within 30 days of that hotspot's date
      - The count excludes the hotspot itself

    This is an approximate but very fast approach — no per-row loops needed.
    A location that appears many times is likely a persistent industrial
    thermal source (gas flare, kiln, power plant) rather than a wildfire.
    """
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    # Bucket lat/lon to ~1 km grid
    df["_blat"] = (df["latitude"]  / FREQ_BUCKET_DEG).round() * FREQ_BUCKET_DEG
    df["_blon"] = (df["longitude"] / FREQ_BUCKET_DEG).round() * FREQ_BUCKET_DEG

    # Count total occurrences per bucket (across entire dataset)
    bucket_counts = df.groupby(["_blat", "_blon"]).size()

    # Map bucket count back to each hotspot and subtract 1 (the hotspot itself)
    freq = df.apply(
        lambda row: bucket_counts.get((row["_blat"], row["_blon"]), 1) - 1,
        axis=1,
    )

    return freq.astype(int)


# ---------------------------------------------------------------------------
# Feature 2: time_of_day
# ---------------------------------------------------------------------------

def compute_time_of_day(acq_time_series: pd.Series) -> pd.Series:
    """
    Derive a coarse time-of-day label from the FIRMS acq_time column.

    FIRMS stores acquisition time as an integer HHMM (e.g., 1347 = 13:47 UTC).
    We extract the hour and bucket it into four periods:
      Night     : 00:00 - 05:59
      Morning   : 06:00 - 11:59
      Afternoon : 12:00 - 17:59
      Evening   : 18:00 - 23:59
    """
    hour = (
        acq_time_series
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
        .str[:2]
        .astype(int)
        .clip(0, 23)
    )

    def bucket(h):
        if   h < 6:   return "Night"
        elif h < 12:  return "Morning"
        elif h < 18:  return "Afternoon"
        else:         return "Evening"

    return hour.apply(bucket)


# ---------------------------------------------------------------------------
# Feature 3: season
# ---------------------------------------------------------------------------

def compute_season(month_series: pd.Series) -> pd.Series:
    """
    Map calendar month to a meteorological season relevant to India:
      Winter      : December, January, February  (cool and dry)
      Summer      : March, April, May             (hot and dry)
      Monsoon     : June, July, August, September (rainy)
      Post-Monsoon: October, November             (post-kharif harvest / stubble)
    """
    def season(m):
        if   m in (12, 1, 2):  return "Winter"
        elif m in (3, 4, 5):   return "Summer"
        elif m in (6, 7, 8, 9):return "Monsoon"
        else:                   return "Post-Monsoon"   # 10, 11

    return month_series.apply(season)


# ---------------------------------------------------------------------------
# Heuristic label assignment
# ---------------------------------------------------------------------------

def assign_category(row: pd.Series) -> str:
    """
    Apply heuristic rules to assign a category label to a single hotspot.
    Rules are evaluated in priority order; the first match wins.

    Parameters
    ----------
    row : pd.Series
        Must contain: land_use_type, frp, distance_to_industrial,
                      historical_frequency, month
    """
    lu        = str(row.get("land_use_type",          "unknown")).lower()
    frp       = float(row.get("frp",                   0.0))
    dist_ind  = float(row.get("distance_to_industrial", 50_000))
    hist_freq = int(row.get("historical_frequency",    0))
    month     = int(row.get("month",                   6))

    # Convenience flags
    is_industrial   = (lu == "industrial") or (dist_ind < NEAR_IND_DIST_M)
    is_vegetation   = lu in ("forest", "farmland")
    is_farmland     = lu == "farmland"
    is_agri_month   = month in AGRI_MONTHS
    is_high_freq    = hist_freq >= HIGH_FREQ_THRESH
    is_repeated     = hist_freq >= REPEAT_THRESH
    is_high_frp     = frp >= HIGH_FRP_THRESH

    # -- Rule 1: Industrial (Normal) -----------------------------------------
    # Persistent thermal source (e.g. gas flare, kiln, plant) detected 3+ times
    if is_high_freq or (is_industrial and is_repeated):
        return "Industrial (Normal)"

    # -- Rule 2: Wildfire Risk -----------------------------------------------
    # Elevated FRP intensity with low historical frequency
    if frp >= 15.0 and not is_high_freq:
        return "Wildfire Risk"

    # -- Rule 3: Agricultural Burning ----------------------------------------
    # Repeated / clustered field burning
    if is_repeated and not is_high_freq:
        return "Agricultural Burning"

    # -- Rule 4: Fallback ----------------------------------------------------
    return "Anomaly/Unclassified"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # ── 1. Load enriched hotspot data ──────────────────────────────────────
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE.name} not found.")
        print("        Run Step 2 first:  python src/fetch_osm_landuse.py")
        raise SystemExit(1)

    print(f"\n[STEP 3] Loading {INPUT_FILE.name} ...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    print(f"         {len(df):,} records loaded.\n")

    # ── 2. Parse dates ─────────────────────────────────────────────────────
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["month"]    = df["acq_date"].dt.month.fillna(6).astype(int)

    # ── 3. Brightness column — VIIRS uses bright_ti4, MODIS uses brightness ─
    if "bright_ti4" in df.columns:
        df["brightness"] = df["bright_ti4"].fillna(df["bright_ti4"].median())
        print("  [INFO] Using 'bright_ti4' as brightness column (VIIRS data).")
    elif "brightness" in df.columns:
        df["brightness"] = df["brightness"].fillna(df["brightness"].median())
        print("  [INFO] Using 'brightness' column (MODIS data).")
    else:
        df["brightness"] = 300.0  # safe default
        print("  [WARN] No brightness column found — defaulting to 300 K.")

    # Clip FRP to non-negative values
    df["frp"]     = df["frp"].clip(lower=0).fillna(0.0)
    df["log_frp"] = np.log1p(df["frp"])

    # ── 4. Ensure OSM columns exist ────────────────────────────────────────
    if "land_use_type" not in df.columns:
        print("  [WARN] 'land_use_type' column missing — defaulting to 'unknown'.")
        df["land_use_type"] = "unknown"

    if "distance_to_industrial" not in df.columns:
        print("  [WARN] 'distance_to_industrial' column missing — defaulting to 50000 m.")
        df["distance_to_industrial"] = 50_000.0

    df["land_use_type"]          = df["land_use_type"].fillna("unknown")
    df["distance_to_industrial"] = df["distance_to_industrial"].fillna(50_000.0)

    # ── 5. Compute: historical_frequency ──────────────────────────────────
    print("  --> Computing historical_frequency ...")
    df["historical_frequency"] = compute_historical_frequency(df)
    print(f"      Done. Range: {df['historical_frequency'].min()} – "
          f"{df['historical_frequency'].max()}, "
          f"median = {df['historical_frequency'].median():.1f}")

    # ── 6. Compute: time_of_day ────────────────────────────────────────────
    print("  --> Computing time_of_day ...")
    df["time_of_day"] = compute_time_of_day(df.get("acq_time", pd.Series(0, index=df.index)))
    print(f"      {df['time_of_day'].value_counts().to_dict()}")

    # ── 7. Compute: season ─────────────────────────────────────────────────
    print("  --> Computing season ...")
    df["season"] = compute_season(df["month"])
    print(f"      {df['season'].value_counts().to_dict()}")

    # ── 8. Assign heuristic category label ────────────────────────────────
    print("  --> Assigning heuristic category labels ...")
    df["category"] = df.apply(assign_category, axis=1)

    # ── 9. Select columns to save ─────────────────────────────────────────
    # Keep all original columns + the new features we computed
    new_cols = [
        "brightness", "frp", "log_frp",
        "land_use_type", "distance_to_industrial",
        "historical_frequency",
        "time_of_day", "season",
        "month",
        "category",
    ]

    # Preserve original columns; add new ones (avoid duplicating existing ones)
    original_cols = [c for c in df.columns if c not in new_cols]
    final_cols    = original_cols + [c for c in new_cols if c in df.columns]

    out = df[final_cols]

    # ── 10. Save ───────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved {len(out):,} rows x {len(final_cols)} columns "
          f"--> {OUTPUT_FILE.name}\n")

    # ── 11. Verification — category value counts ───────────────────────────
    print("=" * 62)
    print("  STEP 3 VERIFICATION -- features_labeled.csv")
    print("=" * 62)
    print(f"  Total records   : {len(out):,}")
    print(f"  Total columns   : {len(final_cols)}")
    print()

    print("  CATEGORY value counts  <-- sanity check this!")
    print("  " + "-" * 50)
    cat_counts = out["category"].value_counts()
    for cat, cnt in cat_counts.items():
        pct = cnt / len(out) * 100
        bar = "#" * int(pct / 2)
        print(f"  {cat:<28}  {cnt:>6,}  ({pct:5.1f}%)  {bar}")

    print()
    print("  historical_frequency stats:")
    hf = out["historical_frequency"]
    print(f"    0 occurrences (one-off)     : {(hf == 0).sum():,}")
    print(f"    1-2 occurrences (repeated)  : {((hf >= 1) & (hf < HIGH_FREQ_THRESH)).sum():,}")
    print(f"    3+ occurrences (persistent) : {(hf >= HIGH_FREQ_THRESH).sum():,}")

    print()
    print("  land_use_type distribution:")
    lu_counts = out["land_use_type"].value_counts()
    for lu, cnt in lu_counts.items():
        pct = cnt / len(out) * 100
        print(f"    {lu:<15}  {cnt:>6,}  ({pct:5.1f}%)")

    print()
    print("  season distribution:")
    for s, cnt in out["season"].value_counts().items():
        pct = cnt / len(out) * 100
        print(f"    {s:<15}  {cnt:>6,}  ({pct:5.1f}%)")

    print()
    print("  frp (MW) stats:")
    print(f"    min  = {out['frp'].min():.1f}")
    print(f"    mean = {out['frp'].mean():.1f}")
    print(f"    max  = {out['frp'].max():.1f}")
    print(f"    >= {HIGH_FRP_THRESH} MW (high intensity): "
          f"{(out['frp'] >= HIGH_FRP_THRESH).sum():,} hotspots")

    print("=" * 62)
    print()
    print("[DONE] features_labeled.csv is ready.")
    print("       Review the CATEGORY counts above — do they look reasonable?")
    print("       If yes, proceed to Step 4:  python src/train_model.py")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
