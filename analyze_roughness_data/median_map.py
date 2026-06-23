"""
median_map.py
Author: Derek Pickell
median map of RMS (roughness) over the full ICESat-2 mission period.

Pipeline
Pass 1  — per-tile: load_and_project() .csv -> (x_node, y_node, rms)
          Output: summaries_median_map/<tile>_nodes.parquet
          *only need to run once*
Pass 2  — per-tile: MAD outlier filter + median per node -> node-median
          parquet.
          Output: summaries_median_map/<tile>_node_medians.parquet
          *re-run and delete node_medians.parquet if change MIN_PASSES, OUTLIER_MAD_THRESHOLD
Pass 3  — concatenate node-median parquets into rectangular GRID_RES cells
          (a) IDW gap-fill (b) Gaussian smooth (c) GeoTIFF + PNG
          Output: summaries_median_map/global_roughness.tif + .png
"""

import warnings
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LogNorm
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely import vectorized

from shared import BaseConfig, load_and_project, save_as_geotiff

warnings.filterwarnings("ignore")

### CONFIGURATION ###
class Config(BaseConfig):
    """
    Median-map config.  Inherits all BaseConfig fields including
    get_geom_inner(), OUTLIER_MAD_THRESHOLD, COASTLINE_PATH, etc.
    Only median-map-specific parameters are defined below.
    """
    # GRID SETTINGS
    SNAP_M = 100                    # meters, data are 'clumped' into 100 m nodes
    GRID_RES = 1000                 # meters, if changed, no need to delete parquets
    IDW_K = 5                       # how many neighbors for IDW interpolation
    MAX_INTERP_DIST = 10000         # meters; beyond this, IDW weight = 0
    SMOOTH_SIGMA  = 1             # Gaussian sigma in grid cells

    # QUALITY SETTINGS: re-run Pass 2 + 3 if changed
    MIN_PASSES = 8                  # minimum observations per node (currently: using 8 for 500 m)

    # COLOR SCALE for plotting
    COLORMAP_SCALE  = "log"         # "log" or "quantile"
    N_QUANTILE_BINS = 25            # only used when COLORMAP_SCALE = "quantile"

    # DATA SOURCE
    DATA_DIR     = Path("/Users/f005cb1/Documents/Github/is2Roughness/testData/")

    VALUE_OF_INTEREST = "RMS" #"RMS" #rms_sub_median #mean_surface #semivariogram_range

    # OUTPUT
    OUTPUT_DIR = Path("./summaries_median_map") # summaries_median_map # summaries_rms_sub_median
    OUTPUT_TIF = OUTPUT_DIR / f"global_roughness_{GRID_RES}_{VALUE_OF_INTEREST}.tif"

###  PASS 1 — PER-TILE RAW OBSERVATION PARQUETS ###
def process_tile(path, cfg=Config):
    """
    Project one CSV tile and save raw (x_node, y_node, rms) observations
    as parquet.  All filtering, projection, eroded ice-mask, and node
    snapping are handled by load_and_project() in shared.py.

    Output: summaries_median_map/<tile>_nodes.parquet
    Skips tiles whose parquet already exists.
    """
    path      = Path(path)
    tile_name = path.stem
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    out_path  = cfg.OUTPUT_DIR / f"{tile_name}_nodes.parquet"

    if out_path.exists():
        print(f"  [skip] {tile_name}")
        return out_path

    df = load_and_project(path, config=cfg)
    # filter out weak beams
    df = df[df["spot_num"].isin([1, 3, 5])]
    if df.empty:
        print(f"  [empty] {tile_name}")
        return out_path

    out_df = (df[["x_node", "y_node", cfg.VALUE_OF_INTEREST]].astype({"x_node": np.int32, "y_node": np.int32, cfg.VALUE_OF_INTEREST: np.float32}))
    out_df.to_parquet(out_path, index=False)

    n_nodes = out_df.groupby(["x_node", "y_node"]).ngroups
    print(f"  {path.name}: {len(df):,} obs | {n_nodes:,} nodes -> {out_path.name}")

    return out_path

### PASS 2 — PER-TILE NODE MEDIANS ###
def compute_tile_node_medians(obs_parquet, cfg=Config):
    """
    Load one tile's raw-obs parquet, apply MAD outlier filter per node,
    compute the robust median per node, and save a compact node-medians
    parquet (one row per node, two coordinate columns + median_rms).

    *Re-run by deleting *_node_medians.parquet files.*

    Output: summaries_median_map/<tile>_node_medians.parquet
    Skips tiles whose node-medians parquet already exists.
    """
    tile_name = obs_parquet.stem.replace("_nodes", "")
    out_path  = cfg.OUTPUT_DIR / f"{tile_name}_node_medians.parquet"

    if out_path.exists():
        print(f"  [skip] {tile_name}")
        return out_path

    if not obs_parquet.exists():
        print(f"  [missing] {obs_parquet.name} — run Pass 1 first")
        return out_path

    obs_df = pd.read_parquet(obs_parquet)

    node_records = []
    for (xn, yn), grp in obs_df.groupby(["x_node", "y_node"], sort=False):
        pv = grp[cfg.VALUE_OF_INTEREST].values

        # MAD filter
        if len(pv) >= cfg.MIN_PASSES:
            median_y = np.median(pv)
            mad_y    = np.median(np.abs(pv - median_y))
            if mad_y > 0:
                keep = (np.abs(pv - median_y)<= cfg.OUTLIER_MAD_THRESHOLD * mad_y / 0.6745)
                pv = pv[keep]

        if len(pv) < cfg.MIN_PASSES:
            continue

        node_records.append({
            "x_node":     np.int32(xn),
            "y_node":     np.int32(yn),
            "median_rms": np.float32(np.median(pv)),
        })

    if not node_records:
        print(f"  [no valid nodes] {tile_name}")
        return out_path

    node_df = pd.DataFrame(node_records)
    node_df.to_parquet(out_path, index=False)
    print(f"  {tile_name}: {len(obs_df):,} obs -> {len(node_df):,} nodes -> {out_path.name}")

    return out_path

###  PASS 3 — GRID AGGREGATION + IDW + SMOOTH + PLOT ###
def build_map(cfg=Config):
    """
    Concatenate all per-tile node-median parquets into rectangular
    GRID_RES cells via IDW gap-fill > Gaussian smooth > GeoTIFF + PNG.
    """
    median_parquets = sorted(cfg.OUTPUT_DIR.glob("*_node_medians.parquet"))
    if not median_parquets:
        raise FileNotFoundError(f"No *_node_medians.parquet files in {cfg.OUTPUT_DIR} — run Pass 2 first")

    print(f"Loading {len(median_parquets)} node-median parquet(s) ...")
    node_df = pd.concat([pd.read_parquet(p) for p in median_parquets], ignore_index=True)
    print(f"  Total nodes : {len(node_df):,}")

    # aggregate nodes onto rectangular grid
    g = cfg.GRID_RES
    node_df["ix"] = np.floor(node_df["x_node"] / g).astype(int)
    node_df["iy"] = np.floor(node_df["y_node"] / g).astype(int)

    ix_min, ix_max = node_df["ix"].min(), node_df["ix"].max()
    iy_min, iy_max = node_df["iy"].min(), node_df["iy"].max()

    xg = (np.arange(ix_min, ix_max + 1) + 0.5) * g
    yg = (np.arange(iy_min, iy_max + 1) + 0.5) * g
    X, Y = np.meshgrid(xg, yg)
    median_grid = np.full(X.shape, np.nan)

    for (ix, iy), cell in node_df.groupby(["ix", "iy"], sort=False):
        row = int(iy - iy_min)
        col = int(ix - ix_min)
        median_grid[row, col] = float(np.median(cell["median_rms"].values))

    # eroded ice mask: 
    # Nodes are already inside the eroded mask from Pass 1 (shared.py), so this only
    # clips any Gaussian smoothing bleed beyond the boundary.
    geom = cfg.get_geom_inner()
    mask = vectorized.contains(geom, X.ravel(), Y.ravel()).reshape(X.shape)

    # IDW gap-fill
    valid_mask   = ~np.isnan(median_grid) & mask
    missing_mask =  np.isnan(median_grid) & mask

    if np.any(missing_mask) and np.any(valid_mask):
        print(f"Gap-filling with IDW ... {len(missing_mask)}")
        tree    = cKDTree(np.column_stack([X[valid_mask], Y[valid_mask]])) 
        pts_q   = np.column_stack([X[missing_mask], Y[missing_mask]])
        k_query = min(cfg.IDW_K, int(valid_mask.sum()))

        dists, idx = tree.query(pts_q, k=k_query) # query tree for distances to all cells
        dists      = np.where(dists == 0, 1e-10, dists) # handle edge case 
        w          = 1.0 / dists ** 2
        w[dists > cfg.MAX_INTERP_DIST] = 0.0  # clipping
        w_sum      = w.sum(axis=1)

        src_vals = median_grid[valid_mask]
        median_grid[missing_mask] = np.where(w_sum > 0, (w * src_vals[idx]).sum(axis=1) / w_sum, np.nan)

    # NaN-aware Gaussian smoothing 
    nan_mask      = np.isnan(median_grid)
    filled_tmp    = np.where(nan_mask, 0.0, median_grid)
    weights       = (~nan_mask).astype(float)
    smoothed_vals = gaussian_filter(filled_tmp, sigma=cfg.SMOOTH_SIGMA)
    smoothed_w    = gaussian_filter(weights,    sigma=cfg.SMOOTH_SIGMA)
    median_grid   = np.where(smoothed_w > 0, smoothed_vals / smoothed_w, np.nan)

    # clip smoothing bleed outside eroded boundary
    median_grid[~mask] = np.nan

    # save + plot 
    save_as_geotiff(cfg.OUTPUT_TIF, median_grid, X, Y)
    _plot(median_grid, X, Y, cfg)

    return median_grid, X, Y

### PLOT UTILITY ###
def _plot(median_grid, X, Y, cfg=Config):
    fig, ax = plt.subplots(figsize=(7, 9), facecolor="white")

    coast = gpd.read_file(cfg.COASTLINE_PATH).to_crs("EPSG:3413") # credit to QGreenland
    coast.plot(ax=ax, facecolor="#e0e0e0", zorder=1)

    valid = median_grid[~np.isnan(median_grid) & (median_grid > 0)]
    if valid.size:
        if cfg.COLORMAP_SCALE == "quantile":
            boundaries = np.unique(np.nanpercentile(valid, np.linspace(0.5, 99.5, cfg.N_QUANTILE_BINS + 1)))
            norm       = BoundaryNorm(boundaries, ncolors=256)
            cbar_label = f"{cfg.VALUE_OF_INTEREST} [m]"
        else: # log
            norm = LogNorm(vmin=np.nanpercentile(valid, 0.5), vmax=np.nanpercentile(valid, 99.5))
            cbar_label = f"{cfg.VALUE_OF_INTEREST} [m]"

        im = ax.pcolormesh(X, Y, median_grid, cmap="Spectral_r", norm=norm, shading="auto", zorder=2)
        cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02, label=cbar_label)
        cbar.ax.tick_params(labelsize=7)

        if cfg.COLORMAP_SCALE == "quantile":
            cbar.set_ticks(boundaries)
            cbar.set_ticklabels([f"{b:.3f}" for b in boundaries])
            cbar.ax.tick_params(labelsize=6)

    cx_min, cy_min, cx_max, cy_max = coast.total_bounds
    x_pad = (cx_max - cx_min) * 0.03 # buffer so we have some 'ocean'
    y_pad = (cy_max - cy_min) * 0.03
    ax.set_xlim(cx_min - x_pad, cx_max + x_pad)
    ax.set_ylim(cy_min - y_pad, cy_max + y_pad)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.set_xlabel("Easting  [km, EPSG:3413]", fontsize=8)
    ax.set_ylabel("Northing [km, EPSG:3413]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"Median {cfg.VALUE_OF_INTEREST}  |  {cfg.GRID_RES} m grid  "
        f"|  min {cfg.MIN_PASSES} passes  "
        f"|  MAD threshold {cfg.OUTLIER_MAD_THRESHOLD}",
        fontsize=9, pad=6,
    )
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cfg = Config()

    tiles = sorted(cfg.DATA_DIR.glob('*.csv'))
    if not tiles:
        raise FileNotFoundError(f'No files matching "\'*.csv\'" in {cfg.DATA_DIR}')
    print(f"Found {len(tiles)} tile(s)\n")

    # Pass 1: raw-obs parquets — skips existing
    print("=== Pass 1: raw observations ===")
    for tile in tiles:
        process_tile(tile, cfg)

    # Pass 2: node-median parquets — skips existing
    # Delete *_node_medians.parquet to re-run with new MIN_PASSES /
    # OUTLIER_MAD_THRESHOLD without re-reading the CSVs.
    print("\n=== Pass 2: node medians ===")
    obs_parquets = sorted(cfg.OUTPUT_DIR.glob("*_nodes.parquet"))
    for p in obs_parquets:
        compute_tile_node_medians(p, cfg)

    # Pass 3: grid aggregation + plot — always re-runs, very fast
    print("\n=== Pass 3: grid + map ===")
    build_map(cfg)