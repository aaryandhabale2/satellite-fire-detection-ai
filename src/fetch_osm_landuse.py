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

Strategy — Spatial Tile Batching
---------------------------------
Querying OSMnx once per hotspot would mean thousands of slow API calls.
Instead we use a 1 degree x 1 degree grid tile strategy:
  - Assign every hotspot to a ~110 km x 110 km tile
  - Query OSMnx ONCE per occupied tile (cached on disk after first run)
  - Run fast in-memory spatial joins for all hotspots in that tile
  - Re-runs over already-cached tiles are instant

Outputs
-------
  data/hotspots_with_landuse.csv
    All original FIRMS columns plus:
      land_use_type        -- OSM land-use category at the hotspot
      distance_to_industrial -- metres to nearest industrial OSM geometry

Usage
-----
    python src/fetch_osm_landuse.py
"""

import math
import sys
import time
import traceback
import warnings
from pathlib import Path

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
# OSMnx global settings
# ---------------------------------------------------------------------------
ox.settings.log_console  = False
ox.settings.use_cache    = True
ox.settings.cache_folder = str(CACHE_DIR)
ox.settings.timeout      = 180   # seconds per OSMnx request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_DEG      = 1.0     # degrees per tile side (~110 km)
BBOX_BUFFER   = 0.08    # extra degrees around each tile to avoid edge misses

METRIC_CRS    = "EPSG:3857"   # Web Mercator — for metre-accurate distances
MAX_DIST_M    = 50_000        # sentinel value (50 km) when no industry found

PROGRESS_EVERY = 50           # print a progress line every N hotspots

# OSM tags to fetch land-use polygons
LANDUSE_TAGS = {"landuse": True}

# OSM tags to fetch industrial geometries
INDUSTRIAL_TAGS = {
    "landuse": ["industrial", "quarry"],
    "man_made": ["works", "factory", "industrial_area"],
}

# How we map OSM landuse tag values to our 5 categories
CATEGORY_MAP = {
    "industrial"  : "industrial",
    "quarry"      : "industrial",
    "military"    : "industrial",
    "factory"     : "industrial",
    "works"       : "industrial",
    "forest"      : "forest",
    "wood"        : "forest",
    "farmland"    : "farmland",
    "farmyard"    : "farmland",
    "orchard"     : "farmland",
    "vineyard"    : "farmland",
    "meadow"      : "farmland",
    "residential" : "residential",
    "commercial"  : "residential",
    "retail"      : "residential",
    "village_green": "residential",
}

# Priority when a hotspot falls inside multiple overlapping polygons
LANDUSE_PRIORITY = [
    "industrial", "forest", "farmland", "residential"
]

# Retry delays in seconds for transient OSMnx/network errors
RETRY_DELAYS = [15, 45, 90]


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

def grid_origin(value: float, step: float = GRID_DEG) -> float:
    """Floor a coordinate value to the nearest grid origin."""
    return math.floor(value / step) * step


def tile_bbox(lat0: float, lon0: float) -> tuple:
    """
    Return (west, south, east, north) for osmnx features_from_bbox.
    Adds a small buffer around the tile so hotspots near edges are covered.
    """
    west  = lon0            - BBOX_BUFFER
    south = lat0            - BBOX_BUFFER
    east  = lon0 + GRID_DEG + BBOX_BUFFER
    north = lat0 + GRID_DEG + BBOX_BUFFER
    return west, south, east, north


# ---------------------------------------------------------------------------
# OSMnx query with retry / back-off
# ---------------------------------------------------------------------------

def query_osm_features(west: float, south: float,
                       east: float, north: float,
                       tags: dict,
                       label: str = "") -> gpd.GeoDataFrame:
    """
    Fetch OSM features within a bounding box.

    Retries up to len(RETRY_DELAYS) times on transient errors.
    Returns an empty GeoDataFrame if the tile has no matching features
    or if all retries fail.
    """
    delays = [0] + RETRY_DELAYS   # first attempt has no wait

    for attempt, wait in enumerate(delays):
        if wait > 0:
            print(f"      [RETRY {attempt}/{len(RETRY_DELAYS)}] "
                  f"Waiting {wait}s before re-querying OSM ({label}) ...")
            time.sleep(wait)

        try:
            gdf = ox.features_from_bbox(
                bbox=(west, south, east, north),
                tags=tags,
            )
            return gdf.reset_index(drop=True)

        except Exception as exc:
            msg = str(exc).lower()

            # Truly empty result — this is normal, not a transient error
            if any(k in msg for k in ("no elements", "result is empty",
                                      "insufficient", "element count")):
                return gpd.GeoDataFrame()

            # Last attempt — give up
            if attempt == len(delays) - 1:
                print(f"      [WARN] OSM query failed after all retries "
                      f"({label}): {exc}")
                return gpd.GeoDataFrame()

            # Transient error — will retry
            print(f"      [WARN] OSM error (will retry) [{label}]: {exc}")

    return gpd.GeoDataFrame()


# ---------------------------------------------------------------------------
# Land-use extraction and classification
# ---------------------------------------------------------------------------

def extract_landuse_polygons(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only polygon geometries from an OSM features GeoDataFrame and
    attach a normalised 'land_use_type' column using CATEGORY_MAP.
    """
    if raw_gdf.empty or "geometry" not in raw_gdf.columns:
        return gpd.GeoDataFrame()

    polys = raw_gdf[
        raw_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if polys.empty:
        return gpd.GeoDataFrame()

    def normalise(row):
        raw_lu = row.get("landuse", None)
        if pd.notna(raw_lu) and raw_lu:
            return CATEGORY_MAP.get(str(raw_lu).lower(), "unknown")
        return "unknown"

    polys["land_use_type"] = polys.apply(normalise, axis=1)
    return polys[["geometry", "land_use_type"]].reset_index(drop=True)


def classify_hotspot_landuse(point_gdf: gpd.GeoDataFrame,
                              poly_gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Spatial join: assign a land_use_type to each hotspot point.
    When a point falls inside multiple polygons, the highest-priority
    category in LANDUSE_PRIORITY wins. Defaults to 'unknown'.
    """
    if poly_gdf.empty:
        return pd.Series("unknown", index=point_gdf.index)

    joined = gpd.sjoin(
        point_gdf[["geometry"]],
        poly_gdf,
        how="left",
        predicate="within",
    )

    if "land_use_type" not in joined.columns:
        return pd.Series("unknown", index=point_gdf.index)

    def best_category(group):
        labels = group["land_use_type"].dropna().tolist()
        # Pick the highest-priority category
        for priority_cat in LANDUSE_PRIORITY:
            if priority_cat in labels:
                return priority_cat
        # Fall back to any non-unknown value, else 'unknown'
        non_unknown = [l for l in labels if l != "unknown"]
        return non_unknown[0] if non_unknown else "unknown"

    result = joined.groupby(joined.index).apply(best_category)
    return result.reindex(point_gdf.index, fill_value="unknown")


# ---------------------------------------------------------------------------
# Distance to nearest industrial geometry
# ---------------------------------------------------------------------------

def compute_industrial_distance(point_gdf: gpd.GeoDataFrame,
                                 industrial_gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Return the distance in METRES from each hotspot to the nearest
    OSM-tagged industrial geometry, using Web Mercator projection.

    Returns NaN (filled later with MAX_DIST_M sentinel) when no industrial
    geometry exists in the tile.
    """
    if industrial_gdf.empty:
        return pd.Series(np.nan, index=point_gdf.index)

    pts_m = point_gdf.to_crs(METRIC_CRS)
    ind_m = industrial_gdf.to_crs(METRIC_CRS)

    try:
        # geopandas >= 0.10 supports sjoin_nearest with distance_col
        nearest = gpd.sjoin_nearest(
            pts_m[["geometry"]],
            ind_m[["geometry"]],
            how="left",
            distance_col="dist_m",
        )
        # sjoin_nearest can produce duplicate rows on ties — keep first
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        return nearest["dist_m"].reindex(point_gdf.index)

    except Exception as exc:
        print(f"      [WARN] sjoin_nearest failed ({exc}) — using fallback ...")
        # Fallback: union all industrial geometries, compute distance to union
        ind_union = ind_m.geometry.unary_union
        distances = pts_m.geometry.apply(lambda p: p.distance(ind_union))
        return distances.reindex(point_gdf.index)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # ── 1. Load FIRMS hotspot data ─────────────────────────────────────────
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE.name} not found.")
        print("        Run Step 1 first:  python src/fetch_firms_data.py")
        raise SystemExit(1)

    print(f"\n[STEP 2] Loading {INPUT_FILE.name} ...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    print(f"         {len(df):,} hotspots loaded.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 2. Assign each hotspot to a 1°x1° grid tile ───────────────────────
    df["_tile_lat"] = df["latitude"].apply(grid_origin)
    df["_tile_lon"] = df["longitude"].apply(grid_origin)

    tiles = (
        df.groupby(["_tile_lat", "_tile_lon"])
        .size()
        .reset_index(name="n_hotspots")
    )
    print(f"         {len(tiles)} unique 1°x1° tile(s). "
          f"OSMnx queried once per tile (cached).\n")

    # ── 3. Pre-allocate result columns ────────────────────────────────────
    df["land_use_type"]          = "unknown"
    df["distance_to_industrial"] = np.nan

    total_hotspots  = len(df)
    processed_count = 0

    # ── 4. Process tile by tile ───────────────────────────────────────────
    for tile_idx, tile_row in tiles.iterrows():
        lat0   = tile_row["_tile_lat"]
        lon0   = tile_row["_tile_lon"]
        n_pts  = int(tile_row["n_hotspots"])

        west, south, east, north = tile_bbox(lat0, lon0)

        print(f"[TILE {tile_idx + 1}/{len(tiles)}]  "
              f"({lat0:.1f}N, {lon0:.1f}E)  "
              f"{n_pts} hotspot(s)  "
              f"bbox=({west:.2f},{south:.2f},{east:.2f},{north:.2f})")

        # Boolean mask for hotspots in this tile
        mask = (df["_tile_lat"] == lat0) & (df["_tile_lon"] == lon0)
        tile_df = df[mask].copy()

        # Build GeoDataFrame for this tile's hotspot points
        tile_points = gpd.GeoDataFrame(
            tile_df,
            geometry=gpd.points_from_xy(
                tile_df["longitude"], tile_df["latitude"]
            ),
            crs="EPSG:4326",
        )

        # ── 4a. Fetch and classify land-use ───────────────────────────────
        try:
            print(f"  --> Querying land-use polygons from OSM ...")
            raw_lu   = query_osm_features(west, south, east, north,
                                          LANDUSE_TAGS, label="landuse")
            lu_polys = extract_landuse_polygons(raw_lu)
            print(f"      {len(lu_polys)} land-use polygon(s) found.")

            landuse_series = classify_hotspot_landuse(tile_points, lu_polys)
            df.loc[mask, "land_use_type"] = landuse_series.values

        except Exception as exc:
            print(f"      [WARN] Land-use step failed for this tile: {exc}")
            print(f"             Hotspots in this tile will be labelled 'unknown'.")
            df.loc[mask, "land_use_type"] = "unknown"

        # ── 4b. Fetch industrial geometries and compute distances ──────────
        try:
            print(f"  --> Querying industrial geometries from OSM ...")
            raw_ind = query_osm_features(west, south, east, north,
                                         INDUSTRIAL_TAGS, label="industrial")

            if not raw_ind.empty:
                # Accept any geometry type (points, lines, polygons all work)
                ind_geoms = raw_ind[
                    raw_ind.geometry.geom_type.isin([
                        "Point", "Polygon", "MultiPolygon",
                        "LineString", "MultiLineString",
                    ])
                ][["geometry"]].reset_index(drop=True)
                print(f"      {len(ind_geoms)} industrial geometry(ies) found.")
            else:
                ind_geoms = gpd.GeoDataFrame()
                print(f"      No industrial geometries found in this tile.")

            dist_series = compute_industrial_distance(tile_points, ind_geoms)
            df.loc[mask, "distance_to_industrial"] = dist_series.values

        except Exception as exc:
            print(f"      [WARN] Industrial distance step failed for this tile: {exc}")
            print(f"             Hotspots in this tile will use {MAX_DIST_M:,} m sentinel.")
            df.loc[mask, "distance_to_industrial"] = MAX_DIST_M

        # ── 4c. Progress reporting every PROGRESS_EVERY records ───────────
        processed_count += n_pts
        pct = processed_count / total_hotspots * 100

        if processed_count % PROGRESS_EVERY < n_pts or processed_count == total_hotspots:
            print(f"\n  [PROGRESS] {processed_count:,} / {total_hotspots:,} "
                  f"hotspots processed ({pct:.1f}%)\n")
        else:
            print()  # blank line between tiles for readability

    # ── 5. Clean up helper columns ────────────────────────────────────────
    df.drop(columns=["_tile_lat", "_tile_lon"], inplace=True)

    # Fill missing distances with sentinel (no industrial geometry in tile)
    df["distance_to_industrial"] = (
        df["distance_to_industrial"].fillna(MAX_DIST_M)
    )

    # ── 6. Save enriched dataset ──────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # ── 7. Verification summary ───────────────────────────────────────────
    print("=" * 62)
    print("  STEP 2 VERIFICATION -- hotspots_with_landuse.csv")
    print("=" * 62)
    print(f"  Total records      : {len(df):,}")
    print(f"  New columns added  : land_use_type, distance_to_industrial")
    print()

    print("  land_use_type distribution:")
    lu_counts = df["land_use_type"].value_counts()
    for lu, cnt in lu_counts.items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {lu:<15}  {cnt:>6,}  ({pct:5.1f}%)  {bar}")

    print()
    dist_col = df["distance_to_industrial"]
    at_sentinel = (dist_col >= MAX_DIST_M).sum()
    has_real    = len(dist_col) - at_sentinel
    print("  distance_to_industrial (metres):")
    print(f"    Records with real distance : {has_real:,}")
    print(f"    Records at sentinel (50 km): {at_sentinel:,}  "
          f"(no industrial geometry nearby)")
    if has_real > 0:
        real = dist_col[dist_col < MAX_DIST_M]
        print(f"    Min    : {real.min():,.0f} m")
        print(f"    Median : {real.median():,.0f} m")
        print(f"    Max    : {real.max():,.0f} m")

    print("=" * 62)
    print(f"\n[DONE] Saved enriched dataset --> {OUTPUT_FILE.name}")
    print("       Check the distribution above — does it look reasonable?")
    print("       When ready, move to Step 3:  python src/build_features.py\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Partial results may not have been saved.")
        print("              Re-run the script to restart (cached tiles are kept).")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
