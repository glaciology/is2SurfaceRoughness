"""
temporal_map.py
Author: Derek Pickell
Multi-tile trend and seasonal pipeline for all of Greenland.

Overview:
Pass 1  — per tile: load CSV (project, ice mask) -> snap to nodes.
    Output: <OUTPUT_DIR>/<tile>_nodes.parquet
    Columns: x_node, y_node, x, y, time, spot_num, dec_year, month, year, VALUE_OF_INTEREST
    Re-run : delete parquet if raw CSVs change.

Pass 2  — per tile: load _nodes.parquet -> assign pass IDs ->
    compute node-level Theil-Sen trends -> aggregate to GRID_RES cells 
    (trend, seasonal contrast, empirical uncertainties) -> save results.
    Output: <OUTPUT_DIR>/<tile>_node_stats_<hash>.parquet
             <OUTPUT_DIR>/<tile>_cell_stats_<hash>.parquet
    Hash: short SHA-256 of all statistical + grid parameters that affect Pass 2 output (TIME_GAP_HOURS, MIN_PASSES,
             MIN_SPAN_YR, MAD_THRESHOLD, MIN_PAIR_DT_YR, BISQUARE_K, MIN_SEASON_TRACKS, SNR_THRESHOLD, MIN_NODES_CELL,
             GRID_RES, EMPIRICAL_SIGMA_SLOPE). Changing any of these produces a new hash and triggers
             regeneration; _nodes.parquet is always reused.
             NOTE: SEAS_AS_PCT_MEDIAN and TREND_AS_PCT_MEDIAN are intentionally excluded from the hash — they are display-only
             flags that operate on already-stored metre values.
    Re-run: delete <tile>_cell_stats_<hash>.parquet (or change a
             hashed parameter, which auto-generates a new filename).

Pass 3  — global: concatenate all <tile>_cell_stats_<hash>.parquet
    files (one row per cell, tiny) and plot trend and seasonal maps. No obs data is loaded; all heavy statistics were done in Pass 2.
    Tiles are non-overlapping, so concatenation requires no merging.

All statistical work is delegated to plot_roughness_spatial.py
(imported as ra):
    Pass 2 -> ra.compute_node_stats()
           -> ra.aggregate_to_grid()
    Pass 3 -> ra.plot_roughness()
"""

import hashlib
import json
import pickle
from pathlib import Path
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm

import temporal_stats as ra
from shared import BaseConfig, derek_colors, load_and_project

###  CONFIGURATION  ###
class Config(BaseConfig):
    """
    Temporal-map configuration.

    SEAS_AS_PCT_MEDIAN and TREND_AS_PCT_MEDIAN are display-only flags.
    They do NOT affect the hash or require re-running Pass 2 — the
    parquets always store raw metre values; the % conversion happens
    at plot time in run_analysis().
    """
    SNAP_M         = 150
    TIME_GAP_HOURS = 1.0

    # per node trends
    MIN_PASSES     = 6
    MIN_SPAN_YR    = 3.0
    MAD_THRESHOLD  = 2.5
    MIN_PAIR_DT_YR = 0.08
    BISQUARE_K     = 4.685

    # seasonal contrast
    MIN_SEASON_TRACKS  = 2

    # "snr"       — Option 1: SNR-based confidence gate
    # "empirical" — Option 2: empirical noise model from crossover pairs
    SIGNIFICANCE_MODE     = "snr"
    EMPIRICAL_SIGMA_SLOPE = 1.77
    SNR_THRESHOLD  = 1.5
    MIN_NODES_CELL = 10

    GRID_RES = 3500   # meters

    VALUE_OF_INTEREST = "RMS"# "semivariogram_range"
    SUMMER_ONLY       = False
    YEAR_CROP         = False
    YEARS             = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    PLOT_SIGNIFICANT  = False

    # display-only flags (not hashed, no parquet regeneration needed)
    SEAS_AS_PCT_MEDIAN  = True   # seasonal diff as % of cell median RMS
    TREND_AS_PCT_MEDIAN = True   # trend as % of cell median RMS per year

    # I/O
    DATA_DIR   = Path("./testData")
    OUTPUT_DIR = Path("./summaries_temporal")


# Elevation bands — used for band stats tables when pct flags are True.
# Colours match spike_contributors and staacked_analysis.
ELEV_BANDS = {
    "Low elevation (<1000 m)":        (0,    1000, derek_colors['red']),
    "Middle elevation (1000-1500 m)": (1000, 1500, "darkorange"),
    "High elevation (1500+ m)":       (1500, 9999, derek_colors['blue']),
}


###  CONFIG HASH  ###
# SEAS_AS_PCT_MEDIAN intentionally absent — it is a display flag only.
# TREND_AS_PCT_MEDIAN same reason.
_HASH_KEYS = [
    "TIME_GAP_HOURS",
    "MIN_PASSES",
    "MIN_SPAN_YR",
    "MAD_THRESHOLD",
    "MIN_PAIR_DT_YR",
    "BISQUARE_K",
    "MIN_SEASON_TRACKS",
    "SNR_THRESHOLD",
    "MIN_NODES_CELL",
    "GRID_RES",
    "EMPIRICAL_SIGMA_SLOPE",
    "VALUE_OF_INTEREST",
    "SUMMER_ONLY",
    "YEAR_CROP",
]

def config_hash(config=Config):
    blob = json.dumps({k: getattr(config, k) for k in _HASH_KEYS}, sort_keys=True).encode()

    return hashlib.sha256(blob).hexdigest()[:8]

###  CELL-STATS SCHEMA  ###
_CELL_COLS = [
    "x_cell", "y_cell",
    "trend_val", "trend_snr", "trend_count",
    "trend_conf", "trend_empirical_conf", "trend_node_median_rms",
    "seas_val", "seas_snr", "seas_count",
    "seas_conf", "seas_empirical_conf", "seas_node_median_rms",
]

def _grids_to_df(X, Y, grids):
    tr = grids["trend"]
    sd = grids["seas_diff"]
    has_data = np.isfinite(tr["val"]) | np.isfinite(sd["val"])
    rows = dict(
        x_cell                  = X[has_data].ravel(),
        y_cell                  = Y[has_data].ravel(),
        trend_val               = tr["val"][has_data].ravel(),
        trend_snr               = tr["snr"][has_data].ravel(),
        trend_count             = tr["count"][has_data].ravel(),
        trend_conf              = tr["confident"][has_data].ravel(),
        trend_empirical_conf    = tr["empirical_conf"][has_data].ravel(),
        trend_node_median_rms   = tr["node_median_rms"][has_data].ravel(),
        seas_val                = sd["val"][has_data].ravel(),
        seas_snr                = sd["snr"][has_data].ravel(),
        seas_count              = sd["count"][has_data].ravel(),
        seas_conf               = sd["confident"][has_data].ravel(),
        seas_empirical_conf     = sd["empirical_conf"][has_data].ravel(),
        seas_node_median_rms    = sd["node_median_rms"][has_data].ravel(),
    )
    return pd.DataFrame(rows)

def _df_to_grids(cell_df, config=Config):
    g    = int(config.GRID_RES)
    mode = getattr(config, "SIGNIFICANCE_MODE", "snr")

    ix = np.round(cell_df["x_cell"].values / g - 0.5).astype(np.int32)
    iy = np.round(cell_df["y_cell"].values / g - 0.5).astype(np.int32)

    ix_min, ix_max = ix.min(), ix.max()
    iy_min, iy_max = iy.min(), iy.max()
    shape = (int(iy_max - iy_min + 1), int(ix_max - ix_min + 1))

    xg = (np.arange(ix_min, ix_max + 1) + 0.5) * g
    yg = (np.arange(iy_min, iy_max + 1) + 0.5) * g
    X, Y = np.meshgrid(xg, yg)

    def _empty():      return np.full(shape, np.nan)
    def _empty_bool(): return np.zeros(shape, dtype=bool)
    def _empty_int():  return np.zeros(shape, dtype=np.int32)

    tr_val   = _empty();      sd_val   = _empty()
    tr_snr   = _empty();      sd_snr   = _empty()
    tr_cnt   = _empty_int();  sd_cnt   = _empty_int()
    tr_conf  = _empty_bool(); sd_conf  = _empty_bool()
    tr_econf = _empty_bool(); sd_econf = _empty_bool()
    tr_nmed  = _empty();      sd_nmed  = _empty()

    r = (iy - iy_min).astype(int)
    c = (ix - ix_min).astype(int)

    tr_val  [r, c] = cell_df["trend_val"].values
    tr_snr  [r, c] = cell_df["trend_snr"].values
    tr_cnt  [r, c] = cell_df["trend_count"].values
    tr_conf [r, c] = cell_df["trend_conf"].values
    tr_econf[r, c] = cell_df["trend_empirical_conf"].values
    tr_nmed [r, c] = cell_df["trend_node_median_rms"].values

    sd_val  [r, c] = cell_df["seas_val"].values
    sd_snr  [r, c] = cell_df["seas_snr"].values
    sd_cnt  [r, c] = cell_df["seas_count"].values
    sd_conf [r, c] = cell_df["seas_conf"].values
    sd_econf[r, c] = cell_df["seas_empirical_conf"].values
    sd_nmed [r, c] = cell_df["seas_node_median_rms"].values

    def _masked(val, conf, econf):
        active = econf if mode == "empirical" else conf
        return np.where(active, val, np.nan)

    grids = {
        "trend": dict(
            val=tr_val, snr=tr_snr, count=tr_cnt,
            confident=tr_conf, empirical_conf=tr_econf,
            node_median_rms=tr_nmed,
            masked=_masked(tr_val, tr_conf, tr_econf),
        ),
        "seas_diff": dict(
            val=sd_val, snr=sd_snr, count=sd_cnt,
            confident=sd_conf, empirical_conf=sd_econf,
            node_median_rms=sd_nmed,
            masked=_masked(sd_val, sd_conf, sd_econf),
        ),
    }
    return X, Y, grids

### PASS 1 — PER-TILE OBSERVATION EXPORT ###
def process_tile(path, config=Config):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    tile_name = Path(path).stem
    voi       = config.VALUE_OF_INTEREST
    out_path  = config.OUTPUT_DIR / f"{tile_name}_nodes_{voi}.parquet"

    if out_path.exists():
        print(f"  [skip] {tile_name}")
        return out_path

    df = load_and_project(path, config=config)
    if df.empty:
        print(f"  [empty] {tile_name}")
        return out_path

    if "dec_year" not in df.columns:
        df["dec_year"] = df["time"].dt.year + (df["time"].dt.dayofyear - 1) / 365.25
    if "month" not in df.columns:
        df["month"] = df["time"].dt.month
    if "year" not in df.columns:
        df["year"] = df["time"].dt.year

    keep = ["x_node", "y_node", "x", "y", "spot_num", "time",
            "dec_year", "month", "year", config.VALUE_OF_INTEREST]
    keep = [col for col in keep if col in df.columns]
    df[keep].to_parquet(out_path, index=False)
    print(f"  {tile_name}: {len(df):,} obs  ->  {out_path.name}")
    return out_path

### PASS 2 — PER-TILE NODE STATS + CELL AGGREGATION ###
def compute_tile_cell_stats(path, config=Config, h=None):
    if h is None:
        h = config_hash(config)

    voi       = config.VALUE_OF_INTEREST
    tile_name = path.stem.replace(f"_nodes_{voi}", "")
    node_out  = config.OUTPUT_DIR / f"{tile_name}_node_stats_{h}.parquet"
    cell_out  = config.OUTPUT_DIR / f"{tile_name}_cell_stats_{h}.parquet"

    if node_out.exists() and cell_out.exists():
        print(f"  [skip] {tile_name}")
        return cell_out

    obs_df = pd.read_parquet(path)

    if getattr(config, "SUMMER_ONLY", False):
        obs_df = obs_df[obs_df["month"].isin(config.SUMMER_MONTHS)]
    if getattr(config, "YEAR_CROP", False):
        obs_df = obs_df[obs_df["year"].isin(config.YEARS)]

    if obs_df.empty:
        print(f"  [empty after filter] {tile_name}")
        return cell_out

    node_df = ra.compute_node_stats(obs_df, config)
    if node_df.empty:
        print(f"  [no valid nodes] {tile_name}")
        return cell_out

    node_df.to_parquet(node_out, index=False)

    X, Y, grids = ra.aggregate_to_grid(node_df, obs_df, config)
    del obs_df

    cell_df = _grids_to_df(X, Y, grids)
    del X, Y, grids

    if cell_df.empty:
        print(f"  [no cells produced] {tile_name}")
        return cell_out

    cell_df.to_parquet(cell_out, index=False)
    print(f"  {tile_name}: {len(node_df):,} nodes  ->  "
          f"{len(cell_df):,} cells  ->  {cell_out.name}")
    return cell_out


### PCT-OF-MEDIAN HELPERS ###
def _as_pct_median(val_grid, node_median_rms_grid):
    """
    Convert val_grid (metres) to % of local median RMS.
    Always computed at display time — parquets store raw metres.
    Cells where node_median_rms <= 0 or NaN are masked to NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(
            np.isfinite(node_median_rms_grid) & (node_median_rms_grid > 0),
            val_grid / node_median_rms_grid * 100.0,
            np.nan,
        )
    return pct.astype(np.float32)

def print_band_stats(pct_grid, elev_grid, label=""):
    """
    Print a per-elevation-band stats table for a % grid.
    Uses the module-level ELEV_BANDS dict.
    """
    w = 32
    print(f"\n  ── {label} ──")
    print(f"  {'Band':<{w}}  {'N':>6}  {'Mean %':>8}  "
          f"{'Median %':>9}  {'Std %':>7}  {'p5 %':>7}  {'p95 %':>7}")
    print(f"  {'-'*w}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}")

    all_vals = []
    for bname, (lo, hi, _) in ELEV_BANDS.items():
        mask = (
            (elev_grid >= lo) & (elev_grid < hi) &
            np.isfinite(pct_grid) & np.isfinite(elev_grid)
        )
        n = int(mask.sum())
        if n == 0:
            print(f"  {bname:<{w}}  {'0':>6}")
            continue
        v        = pct_grid[mask]
        p5, p95  = np.percentile(v, [5, 95])
        all_vals.append(v)
        print(f"  {bname:<{w}}  {n:>6,}  "
              f"{np.mean(v):>+8.2f}  {np.median(v):>+9.2f}  "
              f"{np.std(v):>7.2f}  {p5:>+7.2f}  {p95:>+7.2f}")

    if all_vals:
        v_all    = np.concatenate(all_vals)
        p5a, p95a = np.percentile(v_all, [5, 95])
        print(f"  {'All bands':<{w}}  {len(v_all):>6,}  "
              f"{np.mean(v_all):>+8.2f}  {np.median(v_all):>+9.2f}  "
              f"{np.std(v_all):>7.2f}  {p5a:>+7.2f}  {p95a:>+7.2f}")

def plot_pct_map(pct_grid, X, Y, title, config, filename):
    """
    Map of a % -of-median grid.  Same coast/ice style as plot_roughness().
    """
    fig, ax = plt.subplots(figsize=(7, 9), facecolor="white")

    try:
        coast = gpd.read_file(config.COASTLINE_PATH).to_crs("EPSG:3413")
        coast.plot(ax=ax, facecolor="#e0e0e0", edgecolor="none", zorder=1)
    except Exception:
        pass
    try:
        ice = gpd.read_file(config.ICE_MASK_PATH).to_crs("EPSG:3413")
        ice.boundary.plot(ax=ax, color="#555555", lw=0.3, zorder=5)
    except Exception:
        ice = None

    fin = pct_grid[np.isfinite(pct_grid)]
    if len(fin) == 0:
        ax.set_title("No finite data")
        return fig

    # vmax = float(np.nanpercentile(np.abs(fin), 98))
    # vmax = max(vmax, 0.1)
    # norm = SymLogNorm(linthresh=vmax * 0.2, linscale=1, vmin=-vmax, vmax=vmax, base=10)
    VMAX_PCT = 40.0   
    norm = mcolors.TwoSlopeNorm(vmin=-VMAX_PCT, vcenter=0, vmax=VMAX_PCT)

    im = ax.pcolormesh(X, Y, pct_grid, cmap="RdBu_r", norm=norm,
                       shading="auto", zorder=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, aspect=28)
    cb.set_label("% of cell median RMS", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # extent
    try:
        minx, miny, maxx, maxy = ice.total_bounds
        pad = 5e4
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    except Exception:
        valid = np.isfinite(pct_grid)
        if valid.any():
            rv, cv = np.where(valid)
            ax.set_xlim(X[0, cv.min()] - 1e5, X[0, cv.max()] + 1e5)
            ax.set_ylim(Y[rv.min(), 0] - 1e5, Y[rv.max(), 0] + 1e5)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        f"{title}\n"
        f"n = {int(np.isfinite(pct_grid).sum()):,} cells  "
        f"|  {config.GRID_RES // 1000} km grid",
        fontsize=9, pad=6,
    )
    plt.tight_layout()

    out = config.OUTPUT_DIR / filename
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    return fig

def _load_elev_grid(cell_df, config, shape, iy_min, ix_min):
    """
    Reconstruct an elevation grid from the cell_elevations pickle.
    Returns None if the pickle does not exist.
    """
    elev_pkl = config.OUTPUT_DIR / "cell_elevations.pkl"
    if not elev_pkl.exists():
        # try the anomaly summary dir as a fallback
        alt = Path(f"./summaries_anomaly/{config.GRID_RES}/cell_elevations.pkl")
        if alt.exists():
            elev_pkl = alt
        else:
            print("  WARNING: cell_elevations.pkl not found — "
                  "elevation band stats will be skipped.")
            return None

    with open(elev_pkl, "rb") as f:
        elev_lookup = pickle.load(f)

    g         = int(config.GRID_RES)
    elev_grid = np.full(shape, np.nan, dtype=np.float32)
    ix = np.round(cell_df["x_cell"].values / g - 0.5).astype(np.int32)
    iy = np.round(cell_df["y_cell"].values / g - 0.5).astype(np.int32)

    for xi, yi in zip(ix, iy):
        r   = int(yi - iy_min)
        c   = int(xi - ix_min)
        key = (int(xi), int(yi))
        elev_grid[r, c] = elev_lookup.get(key, np.nan)

    print(f"  Elevation grid: {int(np.isfinite(elev_grid).sum()):,} cells with data")
    return elev_grid


### PASS 3 — GLOBAL CONCATENATION AND PLOTTING ###
def run_analysis(config=Config):
    """
    Concatenate all per-tile cell_stats parquets, reconstruct the global
    grid, and produce:
      - Standard trend and seasonal maps (always)
      - % -of-median seasonal map + elevation stats  (SEAS_AS_PCT_MEDIAN=True)
      - % -of-median trend map + elevation stats     (TREND_AS_PCT_MEDIAN=True)

    The % flags do not affect the hash and do not require re-running Pass 2.
    """
    h = config_hash(config)
    cell_parquets = sorted(config.OUTPUT_DIR.glob(f"*_cell_stats_{h}.parquet"))

    if not cell_parquets:
        raise FileNotFoundError(
            f"No *_cell_stats_{h}.parquet files found in {config.OUTPUT_DIR}.\n"
            f"Run Pass 2 first (or check that config parameters match hash {h})."
        )

    print(f"\nLoading {len(cell_parquets)} cell-stats parquet(s)  [hash: {h}] ...")
    cell_df = pd.concat([pd.read_parquet(p) for p in cell_parquets], ignore_index=True)
    print(f"  Total cells : {len(cell_df):,}")

    X, Y, grids = _df_to_grids(cell_df, config)
    ny, nx  = X.shape
    print(f"  Grid: {ny} x {nx}  "
          f"({nx * config.GRID_RES / 1e3:.0f} x "
          f"{ny * config.GRID_RES / 1e3:.0f} km)")

    mode = getattr(config, "SIGNIFICANCE_MODE", "snr")

    # standard plots
    for metric, name in [("trend", "year"), ("seas_diff", "seasonal")]:
        g       = grids[metric]
        val     = g["val"]
        conf    = g["empirical_conf"] if mode == "empirical" else g["confident"]
        n_total = int(np.isfinite(val).sum())
        n_conf  = int(conf.sum())
        pct_pos = 100 * np.sum(conf & (val > 0)) / max(n_conf, 1)
        pct_neg = 100 * np.sum(conf & (val < 0)) / max(n_conf, 1)

        print(f"\n-- {name} --")
        print(f"  Cells with data   : {n_total}")
        print(f"  Significant cells : {n_conf}  (mode: {mode})")
        print(f"  Significant +/-   : {pct_pos:.2f}% / {pct_neg:.2f}%")

        if config.PLOT_SIGNIFICANT:
            ra.plot_roughness(g["masked"], g["masked"], conf, n_conf, n_total, pct_pos, pct_neg, X, Y, name=name, config=config, significance_mode=mode)
        else:
            ra.plot_roughness(g["masked"], val, conf, n_conf, n_total, pct_pos, pct_neg, X, Y, name=name, config=config, significance_mode=mode)

    # elevation grid (shared by both pct features) 
    seas_as_pct  = getattr(config, "SEAS_AS_PCT_MEDIAN",  False)
    trend_as_pct = getattr(config, "TREND_AS_PCT_MEDIAN", False)

    elev_grid = None
    if seas_as_pct or trend_as_pct:
        # recover ix_min / iy_min from the reconstructed X, Y grids
        g_res  = int(config.GRID_RES)
        ix_min = int(round(X[0, 0]  / g_res - 0.5))
        iy_min = int(round(Y[0, 0]  / g_res - 0.5))
        elev_grid = _load_elev_grid(cell_df, config,
                                    shape=(ny, nx),
                                    iy_min=iy_min, ix_min=ix_min)

    # SEAS_AS_PCT_MEDIAN
    if seas_as_pct:
        sd      = grids["seas_diff"]
        pct_sea = _as_pct_median(sd["val"], sd["node_median_rms"])

        print("\n── Seasonal contrast  (% of cell median RMS) ──")
        if elev_grid is not None:
            print_band_stats(pct_sea, elev_grid,
                             label="Seasonal diff (% median RMS)")
        else:
            fin = pct_sea[np.isfinite(pct_sea)]
            print(f"  n={len(fin):,}  mean={np.mean(fin):+.2f}%  "
                  f"median={np.median(fin):+.2f}%  std={np.std(fin):.2f}%")

        plot_pct_map(pct_sea, X, Y,
            title=(f"Seasonal roughness contrast\n"
                   f"(% of cell median RMS)  |  summer − winter  |  "
                   f"{config.VALUE_OF_INTEREST}"),
            config=config,
            filename=(f"seas_pct_median_{config.VALUE_OF_INTEREST}_{config.GRID_RES}.png"),
        )

    # TREND_AS_PCT_MEDIAN 
    if trend_as_pct:
        tr       = grids["trend"]
        pct_trend = _as_pct_median(tr["val"], tr["node_median_rms"])

        print("\n── Trend  (% of cell median RMS per year) ──")
        if elev_grid is not None:
            print_band_stats(pct_trend, elev_grid, label="Trend (% median RMS / yr)")
        else:
            fin = pct_trend[np.isfinite(pct_trend)]
            print(f"  n={len(fin):,}  mean={np.mean(fin):+.2f}%/yr  median={np.median(fin):+.2f}%/yr  std={np.std(fin):.2f}%/yr")

        plot_pct_map(pct_trend, X, Y,
            title=(f"Roughness trend\n"
                   f"(% of cell median RMS per year)  |  "
                   f"{config.VALUE_OF_INTEREST}"),
            config=config,
            filename=(f"trend_pct_median_{config.VALUE_OF_INTEREST}_{config.GRID_RES}.png"),
        )

    plt.show()
    return cell_df, X, Y, grids


### MAIN ###
if __name__ == "__main__":
    cfg = Config()
    h   = config_hash(cfg)
    print(f"Config hash: {h}")
    print(f"Output dir : {cfg.OUTPUT_DIR}\n")

    raw_tiles = sorted(cfg.DATA_DIR.glob("*.csv"))
    print(f"Found {len(raw_tiles)} tile(s) in {cfg.DATA_DIR}\n")

    print("=== Pass 1: raw observation export ===")
    for tile in raw_tiles:
        process_tile(tile, config=cfg)

    print(f"\n=== Pass 2: node stats + cell aggregation  [hash: {h}] ===")
    nodes_parquets = sorted(
        cfg.OUTPUT_DIR.glob(f"*_nodes_{cfg.VALUE_OF_INTEREST}.parquet"))
    for p in nodes_parquets:
        compute_tile_cell_stats(p, config=cfg, h=h)

    print(f"\n=== Pass 3: global plot  [hash: {h}] ===")
    run_analysis(config=cfg)