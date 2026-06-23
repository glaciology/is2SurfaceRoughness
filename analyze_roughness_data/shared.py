"""
shared.py
Author: Derek Pickell
Common configuration base-class, transformer, and utility functions
shared across all roughness analysis scripts.

DATA EXTRACTION: /getDataSlideRule:
    run_Greenland.py + get_icesat_roughness.py: get tiled roughness outputs as CSV files

DATA ANALYSIS:
    plot_roughness_spatial.py   — trend / seasonal maps for single CSV 'title'
    temporal_map.py             — multi-tile trend pipeline imports plot_roughness_spatial 
    median_map.py               — median roughness map 
    error_v_rms.py              — noise model         
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import vectorized, wkb

derek_colors = {"blue": '#3867B1', "black": '#292930', "red": '#BE3445', "purple": '#4a0e82', "light_blue": '#1f8db5', "yellow": '#ffe138', "orange": "#B55E1F"}

###  SHARED TRANSFORMER: Lat/Lon to NSIDC Polar Stereographic ###
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

###  BASE CONFIGURATION ###
class BaseConfig:
    # value of interest
    VALUE_OF_INTEREST = "RMS"       # range, RMS_50...
    SEA_LEVEL_FILTER  = 15          # meters; drop all VALUE_OF_INTEREST below this elevation
    N_PHOTONS         = 200         # number of photons required for 200 m segment. Photon density nominally should be 0.7 m? 

    # empirical noise model:  σ_i = NOISE_SLOPE × RMS + NOISE_INTERCEPT
    # used to estimate uncertainties
    NOISE_SLOPE     = 0.41844
    NOISE_INTERCEPT = -0.00500
    NOISE_FLOOR     = 1e-4
    OUTLIER_MAD_THRESHOLD = 5 # for nodal filtering

    # optional temporal / seasonal filters 
    SUMMER_ONLY   = False
    SUMMER_MONTHS = [6, 7, 8] # SUMMER_ONLY must be TRUE
    YEAR_CROP     = False
    YEARS         = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] # YEAR_CROP must be TRUE

    # seasonal definitions
    SEASONS       = {
        "winter": [12, 1, 2],
        "summer": [6, 7, 8],
    }

    # ice-mask shapefile (PROMICE 2022) 
    GEOPKG_PATH = Path("/Users/f005cb1/Desktop/RoughnessMaps/dataverse_files/06-PROMICE-2022-IceMask-Nunatak-polygon-v3.gpkg") # credit: PROMICE, Luetzenburg 2026
    ICE_MASK_EROSION_M = 200        # meters to erode inward from ice margin: to account for uncertainty in ice margins and misidentification of ice vs land photons
    CACHE_GEOM = CACHE_GEOM = Path("/Users/f005cb1/Documents/Github/is2Roughness/geom_inner_cache.wkb") # eroded ice mask, cached for faster loading

    # coastline basemap 
    COASTLINE_PATH = Path("/Users/f005cb1/Desktop/RoughnessMaps/QGreenland_v3.0.0/Reference/Borders/Greenland coastlines 2017/bas_greenland_coastlines.gpkg") # credit: QGreenland v3

    # lazy-loaded singletons 
    _GDF        = None
    _GEOM_UNION = None
    _GEOM_INNER = None

    @classmethod
    @property
    def GDF(cls):
        """Full ice-mask GeoDataFrame (EPSG:3413)."""
        if cls._GDF is None:
            cls._GDF = gpd.read_file(cls.GEOPKG_PATH).to_crs("EPSG:3413")
        return cls._GDF

    @classmethod
    @property
    def GEOM_UNION(cls):
        """Union of all ice-mask polygons (used for point-in-mask tests)."""
        if cls._GEOM_UNION is None:
            cls._GEOM_UNION = cls.GDF.geometry.union_all()
        return cls._GEOM_UNION

    @classmethod
    def get_geom_inner(cls):
        if cls._GEOM_INNER is None:
            cls._GEOM_INNER = build_eroded_geom(cls.GEOPKG_PATH, cls.ICE_MASK_EROSION_M, cls.CACHE_GEOM)
        return cls._GEOM_INNER

###  NOISE MODEL ###
def obs_sigma(rms_vals: np.ndarray, config=BaseConfig):
    """Per-observation measurement noise: σ = m·RMS + c, clipped at floor."""
    rms = np.asarray(rms_vals, dtype=float)
    return np.maximum(config.NOISE_SLOPE * rms + config.NOISE_INTERCEPT, config.NOISE_FLOOR)

###  ICE-MASK HELPER ###
def build_mask(X: np.ndarray, Y: np.ndarray, geom):
    """Boolean array: True where (X, Y) lies inside geom."""
    return vectorized.contains(geom, X, Y)

def build_eroded_geom(geopkg_path, erosion_m: float, cache_path: Path):
    """
    Return the ice-sheet polygon eroded inward by erosion_m meters.
    The result is cached as WKB so the expensive buffer is only computed once.

    Used by scripts that want to stay away from the ice margin (e.g. median_map).
    """
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return wkb.loads(fh.read())

    print("Building eroded ice-sheet geometry (should only happen once) …")
    gdf      = gpd.read_file(geopkg_path).to_crs("EPSG:3413")
    gdf.geometry = gdf.geometry.make_valid()
    combined = gdf.geometry.union_all().simplify(20, preserve_topology=True)
    geom     = combined.buffer(-erosion_m, resolution=2)
    if geom.is_empty:
        raise ValueError(
            f"Eroded ice geometry is empty: ({erosion_m} m)."
        )
    with open(cache_path, "wb") as fh:
        fh.write(wkb.dumps(geom))
    print("  done.")
    return geom

###  LOAD AND PROJECT ###
def load_and_project(path, config=BaseConfig):
    """
    Load one CSV tile, apply standard filters, project to EPSG:3413,
    snap to nodes, and add derived time/noise columns.

    Columns added / guaranteed present on return:
        x, y            — projected coordinates (m, EPSG:3413)
        x_node, y_node  — coordinates snapped to config.SNAP_M grid
        dec_year        — decimal year (float32)
        month           — calendar month (int8)
        year            — calendar year (int16)
        rms             — alias for VALUE_OF_INTEREST (float32)
        sigma           — per-obs noise estimate (float32)

    Returns an empty DataFrame (with no rows) when nothing survives filtering.
    """
    df = pd.read_csv(path)

    # get value of interest
    val = config.VALUE_OF_INTEREST
    if val == "semivariogram_range":
        df = df[(df[val] > 0) & (df[val] <= 100)]
    df = df.dropna(subset=[val]).copy()
    df["time"] = pd.to_datetime(df["time"])

    # filter based on mean_surface elevation
    if "mean_surface" in df.columns:
        before = len(df)
        df     = df[df["mean_surface"] >= config.SEA_LEVEL_FILTER]
        n_drop = before - len(df)
        if n_drop:
            print(f"  -> Elevation filter: dropped {n_drop:,} pts below {config.SEA_LEVEL_FILTER} m")

    if df.empty:
        return df
    
    # filter based on n_photons_filtered
    if "n_photons_filtered" in df.columns:
        before = len(df)
        df     = df[df["n_photons_filtered"] >= config.N_PHOTONS]
        n_drop = before - len(df)
        if n_drop:
            print(f"  -> Photon filter: dropped {n_drop:,} pts with n_photons_filtered < {config.N_PHOTONS}")
    if df.empty:
        return df

    # project to NSIDC Polar
    df["x"], df["y"] = TRANSFORMER.transform(df["lon_centroid"].values, df["lat_centroid"].values)
    print(f"  -> Raw obs: {len(df):,}")

    # optional temporal filters 
    if getattr(config, "SUMMER_ONLY", False):
        df = df[df["time"].dt.month.isin(config.SUMMER_MONTHS)]
    if getattr(config, "YEAR_CROP", False):
        df = df[df["time"].dt.year.isin(config.YEARS)]

    if df.empty:
        print(f"  [empty] {path} — no data after temporal filter")
        return df

    # ice-mask filter
    geom_for_mask = (
        config.get_geom_inner()
        if hasattr(config, "get_geom_inner")
        else config.GEOM_UNION
    )
    in_mask = build_mask(df["x"].values, df["y"].values, geom_for_mask)
    df      = df[in_mask].reset_index(drop=True)

    if df.empty:
        print(f"  [empty] {path} — no data inside ice mask")
        return df

    # node snapping
    snap = getattr(config, "SNAP_M", None)
    if snap is not None:
        s = int(snap)
        df["x_node"] = (np.floor(df["x"] / s) * s + s // 2).astype(np.int32)
        df["y_node"] = (np.floor(df["y"] / s) * s + s // 2).astype(np.int32)

    # derived time
    df["dec_year"] = (
        df["time"].dt.year + (df["time"].dt.dayofyear - 1) / 365.25
    ).astype(np.float32)
    df["month"] = df["time"].dt.month.astype(np.int8)
    df["year"]  = df["time"].dt.year.astype(np.int16)

    # noise model
    df["rms"]   = df[val].astype(np.float32)
    df["sigma"] = obs_sigma(df["rms"].values, config).astype(np.float32)

    return df.reset_index(drop=True)

###  GeoTIFF OUTPUT ###
def save_as_geotiff(filename, grid: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """Write a single-band float32 GeoTIFF in EPSG:3413."""
    import rasterio
    from rasterio.transform import from_origin

    res_x     = abs(X[0, 1] - X[0, 0])
    res_y     = abs(Y[1, 0] - Y[0, 0])
    grid_out  = np.flipud(grid).astype(np.float32)
    west      = float(X.min()) - res_x / 2
    north     = float(Y.max()) + res_y / 2
    transform = from_origin(west, north, res_x, res_y)

    with rasterio.open(
        filename, "w",
        driver="GTiff",
        height=grid_out.shape[0],
        width=grid_out.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3413",
        transform=transform,
        nodata=np.float32("nan"),
    ) as dst:
        dst.write(grid_out, 1)

    print(f"  Raster saved: {filename}")