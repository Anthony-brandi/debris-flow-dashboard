"""
export_for_qgis.py -- PF-WRP QGIS export
Outputs usgs_deposits.gpkg with deposit points, HUC-12 polygons,
spatial join results, and fire perimeters.
"""

import math
import zipfile
import os
import json

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

GPKG_PATH = "usgs_deposits.gpkg"
CSV_PATH  = "DebrisFlowVolume_Inventory.csv"
ZIP_PATH  = "Master_Fire_Dataset.geojson.zip"
EXTRACT   = "temp_fire_data_v4"

GARTNER_TRAINING = "Gartner et al. (2014)"
INDEPENDENT_FIRES = {"GRAND PRIX", "THOMAS", "OLD"}

HUC12_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services"
    "/wbd/MapServer/6/query"
)


# ==============================================================================
# 1. GARTNER ENGINE
# ==============================================================================

def _gartner(i15, bmh_km2, relief_m):
    if bmh_km2 <= 0.001 or i15 <= 0 or relief_m <= 0:
        return 0.0
    try:
        return math.exp(
            4.22
            + 0.39 * math.sqrt(float(i15))
            + 0.36 * math.log(float(bmh_km2))
            + 0.13 * math.sqrt(float(relief_m))
        )
    except (ValueError, ZeroDivisionError):
        return 0.0


def _error_category(ratio):
    if ratio < 0.5:
        return "under"
    if ratio <= 2.0:
        return "accurate"
    return "over"


def load_and_compute(csv_path):
    df = pd.read_csv(csv_path)
    df["FireName"] = df["FireName"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["Volume_m3", "AreaModHigh_km2", "Relief_m", "i15_mm/h"])
    df = df[
        (df["Volume_m3"] > 0)
        & (df["AreaModHigh_km2"] > 0.001)
        & (df["Relief_m"] > 0)
        & (df["i15_mm/h"] > 0)
    ].copy()

    df["Predicted_m3"] = df.apply(
        lambda r: _gartner(r["i15_mm/h"], r["AreaModHigh_km2"], r["Relief_m"]),
        axis=1,
    )
    df = df[df["Predicted_m3"] > 0].copy()

    agg = (
        df.groupby(["FireName", "WatershedID"], as_index=False)
        .agg(
            Source           =("Source",           "first"),
            DepositLatitude  =("DepositLatitude",  "first"),
            DepositLongitude =("DepositLongitude", "first"),
            Volume_m3        =("Volume_m3",        "mean"),
            Predicted_m3     =("Predicted_m3",     "mean"),
            i15              =("i15_mm/h",          "mean"),
            AreaModHigh_km2  =("AreaModHigh_km2",  "mean"),
            Relief_m         =("Relief_m",          "mean"),
        )
    )

    agg["Ratio"]          = agg["Predicted_m3"] / agg["Volume_m3"]
    agg["pct_error"]      = (agg["Ratio"] - 1.0) * 100.0
    agg["error_category"] = agg["Ratio"].apply(_error_category)
    agg["is_training"]    = agg["Source"].str.contains(GARTNER_TRAINING, na=False)

    return agg


# ==============================================================================
# 2. BUILD GEODATAFRAME FROM DEPOSIT ROWS
# ==============================================================================

def to_points(df):
    valid = df[
        ~df["DepositLatitude"].isin([-9999])
        & df["DepositLatitude"].notna()
        & df["DepositLongitude"].notna()
    ].copy()
    geometry = [
        Point(lon, lat)
        for lat, lon in zip(valid["DepositLatitude"], valid["DepositLongitude"])
    ]
    return gpd.GeoDataFrame(valid, geometry=geometry, crs="EPSG:4326")


# ==============================================================================
# 3. FETCH HUC-12 POLYGONS FROM USGS WBD REST
# ==============================================================================

def fetch_huc12(minx, miny, maxx, maxy, label):
    print(f"  Fetching HUC-12 for {label} ({minx:.3f},{miny:.3f} -> {maxx:.3f},{maxy:.3f}) ...")
    params = {
        "geometry":     f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR":         "4326",
        "spatialRel":   "esriSpatialRelIntersects",
        "outFields":    "huc12,name,states,areasqkm",
        "returnGeometry": "true",
        "outSR":        "4326",
        "f":            "geojson",
    }
    try:
        r = requests.get(HUC12_URL, params=params, timeout=60)
        r.raise_for_status()
        gdf = gpd.read_file(r.text)
        if gdf.empty:
            print(f"    WARNING: no HUC-12 polygons returned for {label}")
            return gpd.GeoDataFrame(columns=["huc12", "name", "geometry"], crs="EPSG:4326")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        print(f"    {len(gdf)} polygons returned")
        return gdf
    except Exception as e:
        print(f"    ERROR fetching HUC-12 for {label}: {e}")
        return gpd.GeoDataFrame(columns=["huc12", "name", "geometry"], crs="EPSG:4326")


def bbox_for_fire(pts_gdf, padding=0.3):
    bounds = pts_gdf.total_bounds  # minx, miny, maxx, maxy
    return (
        bounds[0] - padding,
        bounds[1] - padding,
        bounds[2] + padding,
        bounds[3] + padding,
    )


# ==============================================================================
# 4. SPATIAL JOIN
# ==============================================================================

def join_deposits_huc12(deposits_gdf, huc12_gdfs):
    if huc12_gdfs.empty or deposits_gdf.empty:
        deposits_gdf = deposits_gdf.copy()
        deposits_gdf["huc12_id"]    = None
        deposits_gdf["huc12_name"]  = None
        deposits_gdf["point_in_huc12"] = False
        return deposits_gdf

    joined = gpd.sjoin(
        deposits_gdf,
        huc12_gdfs[["huc12", "name", "geometry"]],
        how="left",
        predicate="within",
    )
    # Drop duplicate rows from one-to-many joins, keep first match
    joined = joined[~joined.index.duplicated(keep="first")].copy()
    joined["huc12_id"]       = joined["huc12"]
    joined["huc12_name"]     = joined["name"]
    joined["point_in_huc12"] = joined["huc12"].notna()
    drop_cols = [c for c in ["index_right", "huc12", "name"] if c in joined.columns]
    joined = joined.drop(columns=drop_cols)
    return joined


# ==============================================================================
# 5. FIRE PERIMETERS FROM ZIP
# ==============================================================================

def load_perimeters(zip_path, extract_dir, target_fires):
    os.makedirs(extract_dir, exist_ok=True)
    gdfs = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            members = [m for m in z.namelist() if m.endswith(".geojson")
                       and not m.startswith("__MACOSX")]
            for member in members:
                z.extract(member, extract_dir)
        for fname in os.listdir(extract_dir):
            if not fname.endswith(".geojson"):
                continue
            gdf = gpd.read_file(os.path.join(extract_dir, fname)).to_crs("EPSG:4326")
            name_col = next(
                (c for c in gdf.columns
                 if c.lower() in ["fire_name", "incident_n", "name", "firename"]),
                None,
            )
            if name_col is None:
                continue
            gdf["fire_key"] = gdf[name_col].astype(str).str.upper()
            matched = gdf[gdf["fire_key"].isin(target_fires)].copy()
            if not matched.empty:
                gdfs.append(matched[["fire_key", "geometry"]])
    except Exception as e:
        print(f"  WARNING: could not load fire perimeters: {e}")
    if gdfs:
        return pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(columns=["fire_key", "geometry"], crs="EPSG:4326")


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("PF-WRP QGIS Export")
    print("=" * 60)

    # -- Load and compute
    print("\n[1] Loading CSV and running Gartner (2014) ...")
    agg = load_and_compute(CSV_PATH)
    print(f"    {len(agg)} aggregated basins across {agg['FireName'].nunique()} fires")

    # -- Point layers
    print("\n[2] Building point GeoDataFrames ...")
    all_pts  = to_points(agg)
    thomas   = to_points(agg[agg["FireName"] == "THOMAS"])
    indep    = to_points(agg[agg["FireName"].isin(INDEPENDENT_FIRES)])
    print(f"    deposits_all:         {len(all_pts)} points")
    print(f"    deposits_thomas:      {len(thomas)} points")
    print(f"    deposits_independent: {len(indep)} points")

    # -- HUC-12 fetch
    print("\n[3] Fetching HUC-12 polygons from USGS WBD ...")
    huc12_layers = {}
    huc12_combined_gdfs = []
    for fire_name, pts in [("THOMAS", thomas), ("GRAND PRIX", all_pts[all_pts["FireName"] == "GRAND PRIX"]),
                            ("OLD", all_pts[all_pts["FireName"] == "OLD"])]:
        layer_key = fire_name.lower().replace(" ", "")
        if pts.empty:
            print(f"  Skipping {fire_name}: no deposit points")
            huc12_layers[layer_key] = gpd.GeoDataFrame(
                columns=["huc12", "name", "geometry"], crs="EPSG:4326"
            )
            continue
        bbox = bbox_for_fire(pts)
        gdf  = fetch_huc12(*bbox, label=fire_name)
        huc12_layers[layer_key] = gdf
        if not gdf.empty:
            gdf_copy = gdf.copy()
            gdf_copy["fire_source"] = fire_name
            huc12_combined_gdfs.append(gdf_copy)

    huc12_all = (
        pd.concat(huc12_combined_gdfs, ignore_index=True)
        if huc12_combined_gdfs
        else gpd.GeoDataFrame(columns=["huc12", "name", "geometry"], crs="EPSG:4326")
    )

    # -- Spatial join
    print("\n[4] Spatial join: deposits -> HUC-12 ...")
    joined = join_deposits_huc12(indep, huc12_all)
    print(f"    {joined['point_in_huc12'].sum()} / {len(joined)} points matched a HUC-12 polygon")

    # -- Fire perimeters
    print("\n[5] Loading fire perimeters ...")
    perimeters = load_perimeters(
        ZIP_PATH, EXTRACT,
        {"THOMAS", "GRAND PRIX", "OLD"}
    )
    print(f"    {len(perimeters)} perimeter features loaded")

    # -- Write GeoPackage
    print(f"\n[6] Writing {GPKG_PATH} ...")
    layers = {
        "deposits_all":         all_pts,
        "deposits_thomas":      thomas,
        "deposits_independent": indep,
        "huc12_thomas":         huc12_layers.get("thomas",   gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")),
        "huc12_grandprix":      huc12_layers.get("grandprix", gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")),
        "huc12_old":            huc12_layers.get("old",       gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")),
        "deposits_huc12_join":  joined,
        "fire_perimeters":      perimeters,
    }
    if os.path.exists(GPKG_PATH):
        os.remove(GPKG_PATH)
    for layer_name, gdf in layers.items():
        if gdf is None or (hasattr(gdf, "__len__") and len(gdf) == 0):
            print(f"    SKIP {layer_name}: empty")
            continue
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, crs="EPSG:4326")
        # Sanitize column types for GPKG (booleans -> int)
        for col in gdf.columns:
            if gdf[col].dtype == bool:
                gdf[col] = gdf[col].astype(int)
        gdf.to_file(GPKG_PATH, layer=layer_name, driver="GPKG")
        print(f"    Wrote {layer_name}: {len(gdf)} features")

    # -- Terminal summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for fire in sorted(indep["FireName"].unique()):
        fire_pts = joined[joined["FireName"] == fire]
        n_total   = len(fire_pts)
        n_matched = int(fire_pts["point_in_huc12"].sum())
        outside   = fire_pts[~fire_pts["point_in_huc12"].astype(bool)]["WatershedID"].tolist()
        print(f"\n{fire}")
        print(f"  Deposit points:        {n_total}")
        print(f"  Inside a HUC-12:       {n_matched}")
        if outside:
            print(f"  Outside any HUC-12:    {outside}")
        else:
            print(f"  Outside any HUC-12:    none")

    print(f"\nOutput: {os.path.abspath(GPKG_PATH)}")
    print("Done.")


if __name__ == "__main__":
    main()
