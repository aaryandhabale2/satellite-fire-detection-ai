"""
src/fetch_firms_data.py
-----------------------
Fetches NASA FIRMS thermal hotspot / fire data for India and saves
the result to data/firms_raw.csv.

The script uses the FIRMS Area API with India's bounding box.
NASA does not offer a publicly accessible country-code CSV endpoint
for direct download, so we use the bounding box area endpoint which
covers all of India (west=68.1, south=7.9, east=97.4, north=35.5).

Usage
-----
    # 1. Copy .env.example to .env and fill in your key
    # 2. Run:
    python src/fetch_firms_data.py

Environment variables
---------------------
    FIRMS_MAP_KEY   -- your NASA FIRMS MAP_KEY (required)
                       Get one free at:
                       https://firms.modaps.eosdis.nasa.gov/api/map_key
    FIRMS_SOURCE    -- satellite source (default: VIIRS_SNPP_NRT)
    FIRMS_DAY_RANGE -- days of data to fetch, 1-10 (default: 7)

Available FIRMS sources
-----------------------
    VIIRS_SNPP_NRT    -- VIIRS S-NPP,   ~375 m, near-real-time
    VIIRS_NOAA20_NRT  -- VIIRS NOAA-20, ~375 m, near-real-time
    VIIRS_NOAA21_NRT  -- VIIRS NOAA-21, ~375 m, near-real-time
    MODIS_NRT         -- MODIS Terra/Aqua, ~1 km, near-real-time
"""

import io
import os
import sys

# Force UTF-8 output so Unicode characters print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env if it exists
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (read from environment, with sensible defaults)
# ---------------------------------------------------------------------------
MAP_KEY   = os.getenv("FIRMS_MAP_KEY",   "").strip()
SOURCE    = os.getenv("FIRMS_SOURCE",    "VIIRS_SNPP_NRT")
DAY_RANGE = os.getenv("FIRMS_DAY_RANGE", "7")

# India bounding box: west, south, east, north
INDIA_BBOX = "68.1,7.9,97.4,35.5"

# FIRMS Area CSV endpoint
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Output path
ROOT        = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "firms_raw.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_config():
    """Exit early with a clear message if configuration is invalid."""
    if not MAP_KEY:
        print(
            "\n[ERROR] FIRMS_MAP_KEY is not set.\n"
            "  Steps to fix:\n"
            "    1. Copy .env.example to .env\n"
            "    2. Open .env and set:  FIRMS_MAP_KEY=your_actual_key\n"
            "    3. Get a free key at:  "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key\n"
        )
        sys.exit(1)

    try:
        day_int = int(DAY_RANGE)
        if not (1 <= day_int <= 10):
            raise ValueError
    except ValueError:
        print(
            f"[ERROR] FIRMS_DAY_RANGE must be an integer between 1 and 10.\n"
            f"        Current value: {DAY_RANGE!r}"
        )
        sys.exit(1)


def build_url() -> str:
    """Construct the FIRMS Area CSV request URL."""
    return f"{FIRMS_URL}/{MAP_KEY}/{SOURCE}/{INDIA_BBOX}/{DAY_RANGE}"


def fetch_firms(url: str) -> pd.DataFrame:
    """
    Download the CSV from NASA FIRMS and parse it into a DataFrame.
    Handles network errors, HTTP errors, and invalid/empty responses.
    """
    print("\n[STEP 1] Fetching NASA FIRMS data for India ...")
    print(f"         Source    : {SOURCE}")
    print(f"         Day range : {DAY_RANGE} day(s)")
    print(f"         Bbox      : {INDIA_BBOX}  (India)")
    print(f"         URL       : {url}\n")

    # --- Network request -------------------------------------------------------
    try:
        response = requests.get(url, timeout=60)
    except requests.exceptions.SSLError as exc:
        print(f"[ERROR] SSL certificate error connecting to NASA FIRMS.\n  {exc}")
        print("  --> Try again later or check your network/proxy settings.")
        sys.exit(1)
    except requests.exceptions.ConnectionError as exc:
        print(
            "[ERROR] Cannot connect to NASA FIRMS server.\n"
            "  Possible causes:\n"
            "    • No internet connection\n"
            "    • NASA FIRMS API is temporarily down\n"
            "    • A firewall is blocking the request\n"
            f"  Detail: {exc}"
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(
            "[ERROR] Request timed out after 60 s.\n"
            "  NASA FIRMS may be slow or overloaded. Try again in a few minutes."
        )
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Unexpected network error: {exc}")
        sys.exit(1)

    # --- HTTP status check -----------------------------------------------------
    if response.status_code != 200:
        print(
            f"[ERROR] Unexpected HTTP {response.status_code} from FIRMS API.\n"
            f"        Response: {response.text[:400]}"
        )
        sys.exit(1)

    # --- Body sanity check (FIRMS returns 200 even for auth errors) ------------
    body = response.text.strip()

    if not body:
        print("[ERROR] FIRMS returned an empty response. Check your API key and quota.")
        sys.exit(1)

    if body.lower().startswith("invalid") or "MAP_KEY" in body or "error" in body[:100].lower():
        print(
            f"[ERROR] FIRMS API returned an error message:\n"
            f"        {body[:400]}\n"
            f"  --> Verify your FIRMS_MAP_KEY is valid and active."
        )
        sys.exit(1)

    # --- Parse CSV -------------------------------------------------------------
    try:
        df = pd.read_csv(io.StringIO(body))
    except Exception as exc:
        print(f"[ERROR] Could not parse the FIRMS response as CSV.\n  {exc}")
        print(f"  Raw response (first 300 chars):\n  {body[:300]}")
        sys.exit(1)

    return df


def save_csv(df: pd.DataFrame):
    """Save the DataFrame to data/firms_raw.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {len(df):,} records to {OUTPUT_FILE}\n")


def print_summary(df: pd.DataFrame):
    """Print a human-readable summary so the user can verify the output."""
    print("=" * 60)
    print("  FIRMS DATA SUMMARY — firms_raw.csv")
    print("=" * 60)

    print(f"  Total records  : {len(df):,}")
    print(f"  Columns ({len(df.columns)})   : {list(df.columns)}\n")

    # Date range
    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
        min_d = df["acq_date"].min().date()
        max_d = df["acq_date"].max().date()
        print(f"  Date range     : {min_d}  to  {max_d}")
    else:
        print("  Date range     : (acq_date column not found)")

    # Brightness temperature (VIIRS uses bright_ti4; MODIS uses brightness)
    for col in ("bright_ti4", "brightness"):
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col:<14} : min={s.min():.1f}  max={s.max():.1f}  mean={s.mean():.1f} K")

    # Fire Radiative Power
    if "frp" in df.columns:
        s = df["frp"].dropna()
        print(f"  frp (MW)       : min={s.min():.1f}  max={s.max():.1f}  mean={s.mean():.1f}")

    # Confidence breakdown
    if "confidence" in df.columns:
        counts = df["confidence"].value_counts().to_dict()
        print(f"  confidence     : {counts}")

    # Day / night split
    if "daynight" in df.columns:
        counts = df["daynight"].value_counts().to_dict()
        print(f"  daynight       : {counts}")

    print("=" * 60)
    print()
    print("[DONE] firms_raw.csv is ready in data/")
    print("       Verify the summary above looks correct before")
    print("       moving on to Step 2 (OSM land-use enrichment).")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    validate_config()
    url = build_url()
    df  = fetch_firms(url)
    save_csv(df)
    print_summary(df)


if __name__ == "__main__":
    main()
