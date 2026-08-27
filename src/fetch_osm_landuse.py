"""
src/fetch_osm_landuse.py
------------------------
Enriches NASA FIRMS hotspot data with OSM land-use information.

For each hotspot in data/firms_raw.csv this script:
  1. Determines the land-use type at/near that point
     (industrial | forest | farmland | residential | unknown)
  2. Calculates the straight-line distance in metres to the nearest
     OSM-tagged industrial geometry
  3. Merges this info back into the hotspot data
  4. Saves the result to data/hotspots_with_landuse.csv

Outputs
-------
  data/hotspots_with_landuse.csv
"""

import concurrent.futures
import math
import sys
import time
import traceback
import warnings
from pathlib import Path

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
INPUT_FILE  = ROOT / "data" / "firms_raw.csv"
OUTPUT_FILE = ROOT / "data" / "hotspots_with_landuse.csv"
CACHE_DIR   = ROOT / "data" / "osm_cache"

# ---------------------------------------------------------------------------
# OSMnx settings
# ---------------------------------------------------------------------------
ox.settings.log_console      = False
ox.settings.use_cache        = True
ox.settings.cache_folder     = str(CACHE_DIR)
ox.settings.requests_timeout = 8   # fast socket timeout

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLUSTER_DEG   = 0.5     # group hotspots within ~55 km
POINT_BUFFER  = 0.03    # ~3.3 km buffer around points
METRIC_CRS    = "EPSG:3857"
MAX_DIST_M    = 50_000

TARGET_TAGS = {
    "landuse": ["industrial", "forest", "farmland", "residential", "commercial", "quarry", "military"]
}

CATEGORY_MAP = {
    "industrial"   : "industrial",
    "quarry"       : "industrial",
    "military"     : "industrial",
    "factory"      : "industrial",
    "works"        : "industrial",
    "forest"       : "forest",
    "wood"         : "forest",
    "farmland"     : "farmland",
    "farmyard"     : "farmland",
    "orchard"      : "farmland",
    "vineyard"     : "farmland",
    "meadow"       : "farmland",
    "residential"  : "residential",
    "commercial"   : "residential",
    "retail"       : "residential",
    "village_green": "residential",
}

LANDUSE_PRIORITY = ["industrial", "forest", "farmland", "residential"]


def compute_cluster_bbox(points_df: pd.DataFrame) -> tuple:
    west  = float(points_df["longitude"].min() - POINT_BUFFER)
    south = float(points_df["latitude"].min()  - POINT_BUFFER)
    east  = float(points_df["longitude"].max() + POINT_BUFFER)
    north = float(points_df["latitude"].max()  + POINT_BUFFER)
    return west, south, east, north


def process_single_cluster(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Process a single cluster of hotspots against OSM."""
    res_df = cluster_df.copy()
    res_df["land_use_type"]          = "unknown"
    res_df["distance_to_industrial"] = float(MAX_DIST_M)

    west, south, east, north = compute_cluster_bbox(cluster_df)

    pts_gdf = gpd.GeoDataFrame(
        cluster_df,
        geometry=gpd.points_from_xy(cluster_df["longitude"], cluster_df["latitude"]),
        crs="EPSG:4326",
    )

    try:
        raw_gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=TARGET_TAGS,
        )
    except Exception:
        raw_gdf = None

    if raw_gdf is None or raw_gdf.empty or "geometry" not in raw_gdf.columns:
        return res_df

    # 1. Landuse classification
    try:
        polys = raw_gdf[raw_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if not polys.empty:
            def norm_lu(row):
                raw_lu = row.get("landuse", None)
                if pd.notna(raw_lu) and raw_lu:
                    return CATEGORY_MAP.get(str(raw_lu).lower(), "unknown")
                return "unknown"
            
            polys["land_use_type"] = polys.apply(norm_lu, axis=1)
            joined = gpd.sjoin(pts_gdf[["geometry"]], polys[["geometry", "land_use_type"]], how="left", predicate="within")
            
            if "land_use_type" in joined.columns:
                def best_cat(grp):
                    cats = grp["land_use_type"].dropna().tolist()
                    for p in LANDUSE_PRIORITY:
                        if p in cats:
                            return p
                    non_u = [c for c in cats if c != "unknown"]
                    return non_u[0] if non_u else "unknown"
                
                lu_series = joined.groupby(joined.index).apply(best_cat)
                res_df["land_use_type"] = lu_series.reindex(res_df.index, fill_value="unknown").values
    except Exception:
        pass

    # 2. Industrial distance
    try:
        ind_geoms = raw_gdf[
            (raw_gdf.get("landuse", "").isin(["industrial", "quarry"]))
        ]
        if not ind_geoms.empty:
            pts_m = pts_gdf.to_crs(METRIC_CRS)
            ind_m = ind_geoms[["geometry"]].to_crs(METRIC_CRS)
            nearest = gpd.sjoin_nearest(pts_m[["geometry"]], ind_m, how="left", distance_col="dist_m")
            nearest = nearest[~nearest.index.duplicated(keep="first")]
            dist_vals = nearest["dist_m"].reindex(res_df.index).fillna(MAX_DIST_M).values
            res_df["distance_to_industrial"] = dist_vals
    except Exception:
        pass

    return res_df


def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE.name} not found.")
        print("        Run Step 1 first:  python src/fetch_firms_data.py")
        sys.exit(1)

    print(f"\n[STEP 2] Loading {INPUT_FILE.name} ...", flush=True)
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    print(f"         {len(df):,} hotspots loaded.", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df["_cluster_lat"] = (df["latitude"]  / CLUSTER_DEG).apply(math.floor) * CLUSTER_DEG
    df["_cluster_lon"] = (df["longitude"] / CLUSTER_DEG).apply(math.floor) * CLUSTER_DEG

    cluster_groups = [group for _, group in df.groupby(["_cluster_lat", "_cluster_lon"])]
    print(f"         Processing {len(cluster_groups)} spatial clusters with parallel workers ...\n", flush=True)

    processed_dfs = []
    done_count = 0
    total_pts = len(df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_single_cluster, grp): grp for grp in cluster_groups}
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                processed_dfs.append(res)
                done_count += len(res)
                pct = (done_count / total_pts) * 100
                print(f"  [PROGRESS] {done_count:,}/{total_pts:,} hotspots processed ({pct:.1f}%)", flush=True)
            except Exception as e:
                orig = futures[future]
                orig["land_use_type"] = "unknown"
                orig["distance_to_industrial"] = float(MAX_DIST_M)
                processed_dfs.append(orig)
                done_count += len(orig)

    enriched_df = pd.concat(processed_dfs, ignore_index=True)
    enriched_df.drop(columns=["_cluster_lat", "_cluster_lon"], inplace=True, errors="ignore")
    enriched_df["distance_to_industrial"] = enriched_df["distance_to_industrial"].fillna(MAX_DIST_M)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 62, flush=True)
    print("  STEP 2 VERIFICATION — hotspots_with_landuse.csv", flush=True)
    print("=" * 62, flush=True)
    print(f"  Total records      : {len(enriched_df):,}", flush=True)
    print(f"  Columns            : {list(enriched_df.columns)}", flush=True)
    print("\n  land_use_type distribution:", flush=True)
    lu_counts = enriched_df["land_use_type"].value_counts()
    for lu, cnt in lu_counts.items():
        pct = cnt / len(enriched_df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {lu:<15}  {cnt:>6,}  ({pct:5.1f}%)  {bar}", flush=True)

    print("\n  distance_to_industrial (metres):", flush=True)
    dist_col = enriched_df["distance_to_industrial"]
    at_sentinel = (dist_col >= MAX_DIST_M).sum()
    has_real    = len(dist_col) - at_sentinel
    print(f"    Records with real distance : {has_real:,}", flush=True)
    print(f"    Records at sentinel (50 km): {at_sentinel:,}", flush=True)
    if has_real > 0:
        real = dist_col[dist_col < MAX_DIST_M]
        print(f"    Min distance : {real.min():,.0f} m", flush=True)
        print(f"    Median dist  : {real.median():,.0f} m", flush=True)
    print("=" * 62 + "\n", flush=True)
    print(f"[OK] Saved {len(enriched_df):,} enriched hotspots --> {OUTPUT_FILE}\n", flush=True)


if __name__ == "__main__":
    main()
