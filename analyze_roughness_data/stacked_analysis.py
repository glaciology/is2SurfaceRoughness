"""
roughness_analysis.py:
Unified analysis combining staacked_analysis.py and spike_contributors_2019.py
with a single, consistent z-score baseline throughout.

Z-score methodology (identical everywhere):
For each grid cell and target window (year × months):
  1. MAD outlier removal on the full record.
  2. Seasonally-equalised baseline: exclude the target window, then
     subsample up to SAMPLE_PER_MONTH observations per calendar month
     so every month contributes equally regardless of ICESat-2 orbit density.
  3. z = (median of target window − baseline median) / baseline MAD-σ

This is the spike_contributors baseline applied universally — the monthly
time series in staacked_analysis.py used a raw (unequalized) baseline which
biased z=0 toward winter. Here z=0 is the true all-season average.

Three figures:
  1. 2019 summer (Jul–Sep) roughness anomaly map
  2. Monthly roughness anomaly time series by elevation band
     (one z-score per cell per month, stacked + block-bootstrap CI)
  3. MAR Spearman correlation heatmaps
     (2019-only left panel, multi-year pooled right panel)

Outputs go to OUTPUT_DIR (separate from original script outputs).
"""

from __future__ import annotations

import gc
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr
from shapely import vectorized

from shared import derek_colors, TRANSFORMER

warnings.filterwarnings("ignore")
mpl.rcParams["axes.labelsize"]   = 10
mpl.rcParams["axes.titlesize"]   = 10
mpl.rcParams["axes.labelweight"] = "light"


### CONFIG
GRID_RES = 15_000   # metres

SUMMARY_DIR   = Path(f"./summaries_anomaly/{GRID_RES}")
ICE_MASK_PATH = Path("/Users/f005cb1/Desktop/RoughnessMaps/dataverse_files/"
                     "06-PROMICE-2022-IceMask-Nunatak-polygon-v3.gpkg")
COAST_PATH    = Path("/Users/f005cb1/Desktop/RoughnessMaps/QGreenland_v3.0.0/"
                     "Reference/Borders/Greenland coastlines 2017/"
                     "bas_greenland_coastlines.gpkg")
MAR_DIR       = Path("/Users/f005cb1/Desktop/MAR/")

# Separate output dir — will NOT overwrite original script outputs
OUTPUT_DIR = Path("./unified_analysis_output/")

# spike event
SPIKE_YEAR   = 2019
SPIKE_MONTHS = [7, 8, 9]
ALL_YEARS    = list(range(2019, 2026))
ALL_MONTHS   = list(range(1, 13))

#: z-score quality thresholds:
MIN_PASSES          = 30
MIN_TARGET_OBS      = 5     # per-month target so monthly series is possible
PIXEL_MAD_THRESHOLD = 3.0
SAMPLE_PER_MONTH    = 30    # equalised baseline cap per calendar month
RANDOM_SEED         = 42

#: anomaly map:
CONTRIBUTOR_Z = 0.5
Z_CLIP        = 2.0

#: elevation bands:
ELEV_BANDS = {
    "Low elevation (<1000 m)":        (0,    1000, derek_colors["red"]),
    "Middle elevation (1000-1500 m)": (1000, 1500, "darkorange"),
    "High elevation (1500+ m)":       (1500, 9999, derek_colors["blue"]),
}

#: time series:
CI_N_BOOT     = 500
CI_LEVEL      = 0.95
CI_BLOCK_SIZE = 10          # spatial blocks in grid cells (~150 km at 15 km grid)
TARGET_N_PER_MONTH = 10     # minimum obs in a month for that month-cell to count

# event highlight windows
EVENT_WINDOWS = [
    ("Jun–Aug 2019", pd.Timestamp("2019-06-01"),
     pd.Timestamp("2019-08-31"), "#E24B4A"),
    ("Jun–Aug 2023", pd.Timestamp("2023-06-01"),
     pd.Timestamp("2023-08-31"), "#F4A261"),
]

#: MAR:
MAR_FILL_THRESHOLD = 1e10
MAR_ELEV_VAR       = "SRF"
MAR_EXPLORE_VARS   = [
    ("SMBcorr", "SMB",         ALL_MONTHS),
    ("RUcorr",  "Runoff",      ALL_MONTHS),
    ("AL",      "Albedo",      ALL_MONTHS),
    ("MEcorr",  "Melt",        ALL_MONTHS),
    ("LWD",     "LW down",     ALL_MONTHS),
    ("SWD",     "SW down",     ALL_MONTHS),
    ("LHF",     "Latent HF",   ALL_MONTHS),
    ("SHF",     "Sensible HF", ALL_MONTHS),
    ("SH",      "Snow height", ALL_MONTHS),
]


### DATA LOADING
def load_global_cells():
    pkl_files = sorted(SUMMARY_DIR.glob("*_cell_passes.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No *_cell_passes.pkl in {SUMMARY_DIR}.")
    print(f"Loading {len(pkl_files)} pickle(s) ...")
    merged = defaultdict(lambda: {"time_ns": [], "values": []})

    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            tile = pickle.load(f)
        for (ix, iy), arrays in tile.items():
            merged[(ix, iy)]["time_ns"].append(arrays["time_ns"])
            merged[(ix, iy)]["values"].append(arrays["values"])
    cells = {
        k: {"time_ns": np.concatenate(d["time_ns"]),
            "values":  np.concatenate(d["values"])}
        for k, d in merged.items()
    }

    print(f"  {len(cells):,} cells loaded")

    return cells


def load_elevation_lookup():
    p = SUMMARY_DIR / "cell_elevations.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    with open(p, "rb") as f:
        return pickle.load(f)


### GRID AND ICE MASK
def build_grid(cells):
    all_ix = np.array([k[0] for k in cells])
    all_iy = np.array([k[1] for k in cells])
    xg = (np.arange(all_ix.min(), all_ix.max() + 1) + 0.5) * GRID_RES
    yg = (np.arange(all_iy.min(), all_iy.max() + 1) + 0.5) * GRID_RES
    X, Y = np.meshgrid(xg, yg)
    rc = {(ix, iy): (iy - all_iy.min(), ix - all_ix.min())
          for (ix, iy) in cells}
    
    return X, Y, rc

def build_ice_mask(X, Y):
    try:
        gdf = gpd.read_file(ICE_MASK_PATH).to_crs("EPSG:3413")
        geom = gdf.geometry.union_all()
        mask = vectorized.contains(geom, X, Y)
        print(f"  Ice mask: {int(mask.sum()):,} cells inside")

        return mask
    
    except Exception as e:
        print(f"  Ice mask failed ({e}) — using all cells")
        return np.ones(X.shape, dtype=bool)


### Z SCORE CALC
def _zscore_for_window(arrays, year, months, rng, min_target_obs=MIN_TARGET_OBS):
    """
    Seasonally-equalised z-score for one cell, one (year, months) window.

    Baseline: all observations OUTSIDE the target window, capped at
    SAMPLE_PER_MONTH per calendar month so every month contributes equally.

    Returns (z, True) or (None, False).
    """
    t_ns = arrays["time_ns"]
    vals = arrays["values"].astype(float)

    ok = np.isfinite(vals) & (vals > 0)
    if ok.sum() < MIN_PASSES:
        return None, False

    times = pd.to_datetime(t_ns[ok].astype("datetime64[ns]"))
    vals  = vals[ok]

    # outlier removal on full record
    mu_r  = float(np.median(vals))
    mad_r = float(np.median(np.abs(vals - mu_r)))
    if mad_r > 0:
        keep  = np.abs(vals - mu_r) <= PIXEL_MAD_THRESHOLD * (mad_r / 0.6745)
        times = times[keep]
        vals  = vals[keep]
    if len(vals) < MIN_PASSES:
        return None, False

    # target window
    target_mask = (times.year == year) & times.month.isin(months)
    if target_mask.sum() < min_target_obs:
        return None, False
    v_target = float(np.median(vals[target_mask]))

    # equalised baseline
    df_base = pd.DataFrame(
        {"val": vals[~target_mask]}, index=times[~target_mask])
    sampled = []
    for _, grp in df_base.groupby(df_base.index.month):
        v = grp["val"].values
        if len(v) <= SAMPLE_PER_MONTH:
            sampled.append(v)
        else:
            sampled.append(v[rng.choice(len(v), SAMPLE_PER_MONTH, replace=False)])
    if not sampled:
        return None, False
    v_base = np.concatenate(sampled)
    if len(v_base) < 3:
        return None, False

    mu = float(np.median(v_base))
    mad = float(np.median(np.abs(v_base - mu)))
    sigma = (mad / 0.6745) if mad > 0 else float(np.std(v_base, ddof=1))
    if sigma <= 0 or not np.isfinite(sigma):
        
        return None, False

    return float((v_target - mu) / sigma), True


# Z SCORE GRIDS
def build_spike_grids(cells, elev_lookup, X, Y, rc):
    """
    Build:
      z_2019 — z-score for Jul–Sep 2019  (Fig 1 map + Fig 3 correlations)
      yr_grids — z-score for Jul–Sep of each year  (Fig 3 multi-year)
      elev_grid — median elevation per cell
    """
    ny, nx    = X.shape
    z_2019    = np.full((ny, nx), np.nan, dtype=np.float32)
    elev_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    yr_grids  = {yr: np.full((ny, nx), np.nan, dtype=np.float32)
                 for yr in ALL_YEARS}

    rng = np.random.default_rng(RANDOM_SEED)

    for (ix, iy), arrays in cells.items():
        rc_val = rc.get((ix, iy))
        if rc_val is None:
            continue
        r, c = rc_val

        # 2019 spike z (stricter threshold for the map)
        z, ok = _zscore_for_window(arrays, SPIKE_YEAR, SPIKE_MONTHS, rng, min_target_obs=30)
        if not ok:
            continue
        z_2019[r, c]    = z
        elev_grid[r, c] = elev_lookup.get((ix, iy), np.nan)

        # per-year z-scores (same stricter threshold)
        for yr in ALL_YEARS:
            z_yr, ok_yr = _zscore_for_window(arrays, yr, SPIKE_MONTHS, rng, min_target_obs=30)
            if ok_yr:
                yr_grids[yr][r, c] = z_yr

    n_valid = int(np.isfinite(z_2019).sum())
    print(f"\n  2019 z-grid: {n_valid:,} cells valid")
    print(f"  Cells > +{CONTRIBUTOR_Z}σ: "
          f"{int((z_2019 > CONTRIBUTOR_Z).sum()):,}  "
          f"({100*float((z_2019>CONTRIBUTOR_Z).sum())/max(n_valid,1):.1f}%)")
    
    return z_2019, elev_grid, yr_grids


### HELPERS
def _load_geo():
    try:
        coast = gpd.read_file(COAST_PATH).to_crs("EPSG:3413")
    except Exception:
        coast = None
    try:
        ice_gdf = gpd.read_file(ICE_MASK_PATH).to_crs("EPSG:3413")
    except Exception:
        ice_gdf = None

    return coast, ice_gdf

def _map_base(ax, coast, ice_gdf):
    if coast is not None:
        coast.plot(ax=ax, facecolor="#E8E8E8", edgecolor="none", zorder=1)
    if ice_gdf is not None:
        ice_gdf.boundary.plot(ax=ax, color="#555", lw=0.4, zorder=5)

def _map_extent(grid, X, Y, pad=1.5e5):
    valid = np.isfinite(grid)
    if not valid.any():
        return X.min(), X.max(), Y.min(), Y.max()
    rv, cv = np.where(valid)

    return (float(X[0, cv.min()]) - pad, float(X[0, cv.max()]) + pad,
            float(Y[rv.min(), 0]) - pad, float(Y[rv.max(), 0]) + pad)

def _elev_contours(ax, X, Y, elev_grid):
    levels = [1000, 1500]
    colors = [derek_colors["red"], "darkorange"]
    try:
        ax.contour(X, Y, elev_grid, levels=levels,
                   colors=colors, linewidths=0.9, alpha=0.75, zorder=4)
    except Exception:
        pass

    return levels, colors


### MAR HELPERS
def _year_from_name(name):
    m = re.search(r"(20\d{2}|19\d{2})", name)
    
    return int(m.group(1)) if m else None

def _mar_regrid(arr_2d, y_mar, x_mar, X_t, Y_t):
    y = y_mar.copy(); x = x_mar.copy(); t = arr_2d.copy()
    if y[0] > y[-1]: y = y[::-1]; t = t[::-1, :]
    if x[0] > x[-1]: x = x[::-1]; t = t[:, ::-1]
    interp = RegularGridInterpolator((y, x), t, method="linear", bounds_error=False, fill_value=np.nan)
    pts = np.stack([Y_t.ravel(), X_t.ravel()], axis=1)
    
    return interp(pts).reshape(X_t.shape).astype(np.float32)

def _mask_fill(arr):
    out = arr.astype(np.float32)
    out[out > MAR_FILL_THRESHOLD] = np.nan
    return out

def _load_mar_year(varname, months, X, Y, year):
    nc_files = sorted(MAR_DIR.glob("*.nc"))
    f = next((p for p in nc_files if _year_from_name(p.name) == year), None)
    if f is None:
        return None
    
    try:
        with xr.open_dataset(f, decode_times=False) as ds:
            if varname not in ds:
                return None
            arr = _mask_fill(ds[varname].values)
            x_mar = ds.x.values.copy()
            y_mar = ds.y.values.copy()
    except Exception:
        return None
    
    keep = [i for i in range(arr.shape[0]) if (i % 12) + 1 in months]
    if not keep:
        return None
    
    mean_arr = np.nanmean(arr[keep], axis=0)
    del arr; gc.collect()

    return _mar_regrid(mean_arr, y_mar, x_mar, X, Y)


def _mar_zscore_grids(varname, months, X, Y, all_yrs, ice):
    raw = {}
    for yr in all_yrs:
        g = _load_mar_year(varname, months, X, Y, yr)
        if g is not None:
            g = g.copy(); g[~ice] = np.nan
        raw[yr] = g
    yrs_ok = [yr for yr in all_yrs if raw[yr] is not None]
    if not yrs_ok:
        return None
    
    stack  = np.stack([raw[yr] for yr in yrs_ok], axis=0)
    median = np.nanmedian(stack, axis=0)
    mad    = np.nanmedian(np.abs(stack - median[np.newaxis]), axis=0)
    sigma  = np.where(mad > 0, mad / 0.6745, np.nanstd(stack, axis=0, ddof=1))
    no_var = ~np.isfinite(sigma) | (sigma <= 0)
    z_out  = {}
    for yr in yrs_ok:
        with np.errstate(invalid="ignore", divide="ignore"):
            zg = ((raw[yr] - median) / sigma).astype(np.float32)
        zg[~np.isfinite(zg)] = np.nan
        zg[no_var] = np.nan
        zg[~ice]   = np.nan
        z_out[yr]  = zg

    return z_out

def _load_mar_elev(X, Y):
    nc_files = sorted(MAR_DIR.glob("*.nc"))
    f = next((p for p in nc_files if _year_from_name(p.name) == SPIKE_YEAR), None)
    if f is None:
        return None
    
    try:
        with xr.open_dataset(f, decode_times=False) as ds:
            if MAR_ELEV_VAR not in ds:
                return None
            
            arr   = _mask_fill(ds[MAR_ELEV_VAR].values)
            x_mar = ds.x.values.copy()
            y_mar = ds.y.values.copy()
    except Exception:
        return None
    
    return _mar_regrid(arr, y_mar, x_mar, X, Y)

#  FIGURE 1 — 2019 ANOMALY MAP
def plot_anomaly_map(z_2019, elev_grid, mar_elev, X, Y):
    coast, ice_gdf = _load_geo()
    elev_src = mar_elev if mar_elev is not None else elev_grid
    minx, maxx, miny, maxy = _map_extent(z_2019, X, Y)

    fig, ax = plt.subplots(facecolor="white")
    norm = mcolors.TwoSlopeNorm(vmin=-Z_CLIP, vcenter=0, vmax=Z_CLIP)

    _map_base(ax, coast, ice_gdf)
    im = ax.pcolormesh(X, Y, np.clip(z_2019, -Z_CLIP, Z_CLIP), cmap="RdBu_r", norm=norm, shading="auto", zorder=3)
    plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02, label=f"Jul–Sep {SPIKE_YEAR} roughness z-score\n(seasonally-equalised baseline)")

    elev_levels, elev_colors = _elev_contours(ax, X, Y, elev_src)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect("equal"); ax.set_axis_off()

    handles = [Line2D([0], [0], color=col, lw=0.9, alpha=0.75, label=f"{lvl} m") for lvl, col in zip(elev_levels, elev_colors)]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.85)

    n_above = int((z_2019 > CONTRIBUTOR_Z).sum())
    n_tot = int(np.isfinite(z_2019).sum())
    ax.set_title(
        f"Jul–Sep {SPIKE_YEAR} roughness anomaly  "
        f"[{GRID_RES//1000} km grid]\n"
        f"{n_above:,} / {n_tot:,} cells above +{CONTRIBUTOR_Z}σ",
        fontsize=9)

    return fig

#  FIGURE 2 — MONTHLY TIME SERIES BY ELEVATION BAN
def _compute_monthly_series(arrays, rng):
    """
    For one cell, compute a monthly z-score time series.

    Each month in the record is treated as a separate target window of width = 1 month. The baseline is all OTHER months, equalised via
    SAMPLE_PER_MONTH. This gives exactly the same baseline logic as the 2019 map but applied month-by-month so the full temporal
    record is preserved.
    """
    t_ns = arrays["time_ns"]
    vals = arrays["values"].astype(float)

    ok = np.isfinite(vals) & (vals > 0)
    if ok.sum() < MIN_PASSES:
        return None

    times = pd.to_datetime(t_ns[ok].astype("datetime64[ns]"))
    vals  = vals[ok]

    # outlier removal
    mu_r  = float(np.median(vals))
    mad_r = float(np.median(np.abs(vals - mu_r)))
    if mad_r > 0:
        keep  = np.abs(vals - mu_r) <= PIXEL_MAD_THRESHOLD * (mad_r / 0.6745)
        times = times[keep]; vals = vals[keep]
    if len(vals) < MIN_PASSES:
        return None

    df = pd.DataFrame({"val": vals}, index=times)

    # group by calendar month (MS = month start)
    monthly_z, monthly_idx = [], []
    for month_ts, grp in df.groupby(pd.Grouper(freq="MS")):
        if len(grp) < TARGET_N_PER_MONTH:
            continue

        yr = month_ts.year
        mo = month_ts.month

        # equalised baseline: all obs NOT in this month-year window
        base_mask = ~((times.year == yr) & (times.month == mo))
        df_base   = pd.DataFrame(
            {"val": vals[base_mask]}, index=times[base_mask])

        sampled = []
        for _, bgrp in df_base.groupby(df_base.index.month):
            v = bgrp["val"].values
            if len(v) <= SAMPLE_PER_MONTH:
                sampled.append(v)
            else:
                sampled.append(v[rng.choice(len(v), SAMPLE_PER_MONTH, replace=False)])
        if not sampled:
            continue
        v_base = np.concatenate(sampled)
        if len(v_base) < 3:
            continue

        mu  = float(np.median(v_base))
        mad  = float(np.median(np.abs(v_base - mu)))
        sigma = (mad / 0.6745) if mad > 0 else float(np.std(v_base, ddof=1))
        if sigma <= 0 or not np.isfinite(sigma):
            continue

        # subsample the target month before computing median z
        v_tgt = grp["val"].values
        if len(v_tgt) > TARGET_N_PER_MONTH:
            v_tgt = v_tgt[rng.choice(len(v_tgt), TARGET_N_PER_MONTH, replace=False)]
        z = float((np.median(v_tgt) - mu) / sigma)
        monthly_z.append(z)
        monthly_idx.append(month_ts)

    if not monthly_z:
        return None
    
    return pd.Series(monthly_z, index=pd.DatetimeIndex(monthly_idx)).sort_index()

def _block_bootstrap_ci(pixel_df, keys):
    rng    = np.random.default_rng(RANDOM_SEED)
    ix_arr = np.array([k[0] for k in keys])
    iy_arr = np.array([k[1] for k in keys])
    bix    = (ix_arr // CI_BLOCK_SIZE) * CI_BLOCK_SIZE
    biy    = (iy_arr // CI_BLOCK_SIZE) * CI_BLOCK_SIZE
    block_ids = list(set(zip(bix.tolist(), biy.tolist())))

    block_to_cols = defaultdict(list)
    for i, k in enumerate(keys):
        col = str(k)
        if col in pixel_df.columns:
            block_to_cols[(int(bix[i]), int(biy[i]))].append(col)

    valid_blocks = [b for b in block_ids if block_to_cols[b]]
    n_blocks     = len(valid_blocks)
    if n_blocks < 3:
        return pixel_df.quantile(0.25, axis=1), pixel_df.quantile(0.75, axis=1)

    boot_medians = []
    alpha = (1 - CI_LEVEL) / 2
    for _ in range(CI_N_BOOT):
        idx  = rng.choice(n_blocks, size=n_blocks, replace=True)
        cols = []
        for bi in idx:
            cols.extend(block_to_cols[valid_blocks[bi]])
        if cols:
            boot_medians.append(pixel_df[cols].median(axis=1))

    if not boot_medians:
        return pixel_df.quantile(0.25, axis=1), pixel_df.quantile(0.75, axis=1)

    boot_df = pd.DataFrame(boot_medians).T

    return boot_df.quantile(alpha, axis=1), boot_df.quantile(1 - alpha, axis=1)

def plot_timeseries(cells, elev_lookup, rc, ice_mask):
    n_bands = len(ELEV_BANDS)
    fig = plt.figure(facecolor="white")
    gs  = GridSpec(n_bands, 2, figure=fig, width_ratios=[5, 1], hspace=0.45, wspace=0.25)
    month_labels = list("JFMAMJJASOND")

    # compute all bands first to get shared y limits
    band_results = {}
    for bname, (lo, hi, bcol) in ELEV_BANDS.items():
        print(f"\n  Monthly time series: {bname} ...")
        rng  = np.random.default_rng(RANDOM_SEED)
        keys = [
            (ix, iy) for (ix, iy) in cells
            if (lo <= elev_lookup.get((ix, iy), np.nan) < hi)
            and rc.get((ix, iy)) is not None
            and ice_mask[rc[(ix, iy)]]
        ]
        pix = {}
        for k in keys:
            s = _compute_monthly_series(cells[k], rng)
            if s is not None and len(s) > 0:
                pix[k] = s
        if not pix:
            band_results[bname] = None
            continue

        pixel_df = pd.DataFrame(pix).sort_index()
        med      = pixel_df.median(axis=1)
        p25      = pixel_df.quantile(0.25, axis=1)
        p75      = pixel_df.quantile(0.75, axis=1)
        n_pix    = pixel_df.notna().sum(axis=1)

        print(f"    {len(pix):,} cells  |  mean z = {med.mean():+.3f}")
        print(f"    Block bootstrap CI ({CI_N_BOOT} resamples) ...")
        pixel_df.columns = [str(k) for k in pixel_df.columns]
        ci_lo, ci_hi = _block_bootstrap_ci(pixel_df, keys)

        sea_df = med.to_frame("z")
        sea_df["month"] = sea_df.index.month
        seasonal = sea_df.groupby("month")["z"].mean()

        band_results[bname] = dict(
            med=med, p25=p25, p75=p75,
            ci_lo=ci_lo, ci_hi=ci_hi,
            n_pix=n_pix, n_cells=len(pix),
            seasonal=seasonal, color=bcol,
        )

    # shared y limits
    all_meds = [r["med"] for r in band_results.values() if r is not None]
    all_seas = [r["seasonal"] for r in band_results.values() if r is not None]
    if all_meds:
        ts_min = min(s.min() for s in all_meds)
        ts_max = max(s.max() for s in all_meds)
        ts_pad = (ts_max - ts_min) * 0.15
        ts_ylim = (ts_min - ts_pad, ts_max + ts_pad)
        sea_min = min(s.min() for s in all_seas)
        sea_max = max(s.max() for s in all_seas)
        sea_pad = (sea_max - sea_min) * 0.15
        sea_ylim = (sea_min - sea_pad, sea_max + sea_pad)
    else:
        ts_ylim = (-2, 2); sea_ylim = (-1, 1)

    for row, (bname, res) in enumerate(band_results.items()):
        ax_ts  = fig.add_subplot(gs[row, 0])
        ax_sea = fig.add_subplot(gs[row, 1])

        if res is None:
            ax_ts.text(0.5, 0.5, "No data",
                       transform=ax_ts.transAxes, ha="center")
            ax_sea.set_visible(False)
            continue

        bcol   = res["color"]
        med    = res["med"]
        ci_lo  = res["ci_lo"]
        ci_hi  = res["ci_hi"]
        n_pix  = res["n_pix"]
        sea    = res["seasonal"]

        for _, ev_start, ev_end, ev_col in EVENT_WINDOWS:
            ax_ts.axvspan(ev_start, ev_end, alpha=0.12, color=ev_col, zorder=1)

        ax_ts.fill_between(med.index, ci_lo, ci_hi, alpha=0.25, color=bcol, zorder=2, label=f"{int(CI_LEVEL*100)}% CI")
        ax_ts.plot(med.index, med, color=bcol, lw=2, zorder=3, label="Monthly median")
        ax_ts.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)

        ax2 = ax_ts.twinx()
        ax2.fill_between(n_pix.index, 0, n_pix, color=bcol, alpha=0.06, step="post")
        ax2.set_ylabel("N cells", fontsize=7, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=6)
        ax2.set_ylim(0, n_pix.max() * 5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_ts.xaxis.set_major_locator(mdates.YearLocator())
        ax_ts.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax_ts.spines["top"].set_visible(False)
        ax_ts.spines["right"].set_visible(False)
        ax_ts.set_ylim(ts_ylim)
        ax_ts.set_ylabel("Roughness anomaly (σ)", fontsize=9, labelpad=6)
        ax_ts.set_title(
            f"{bname}  |  {res['n_cells']:,} cells", fontsize=9)
        ax_ts.tick_params(labelsize=8)
        if row == 0:
            ax_ts.legend(fontsize=7, loc="upper left", framealpha=0.85)

        # seasonal climatology bar chart
        bar_cols = [bcol if v >= 0 else "lightgray" for v in sea.values]
        ax_sea.bar(sea.index, sea.values, color=bar_cols, alpha=0.8, width=0.8, edgecolor="white", lw=0.5)
        ax_sea.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
        ax_sea.set_xticks(range(1, 13))
        ax_sea.set_xticklabels(month_labels, fontsize=6)
        ax_sea.set_ylim(sea_ylim)
        ax_sea.set_ylabel("Mean z", fontsize=8, labelpad=6)
        ax_sea.tick_params(labelsize=7)
        ax_sea.spines["top"].set_visible(False)
        ax_sea.spines["right"].set_visible(False)
        if row == 0:
            ax_sea.set_title("Seasonal climatology", fontsize=8)

    fig.suptitle(
        f"ICESat-2 roughness anomaly by elevation band  |  "
        f"{ALL_YEARS[0]}–{ALL_YEARS[-1]}\n"
        f"Seasonally-equalised z-score  |  {GRID_RES//1000} km grid  |  "
        f"{int(CI_LEVEL*100)}% block bootstrap CI",
        fontsize=9)
    return fig

#  FIGURE 3 — MAR SPEARMAN CORRELATION HEATMAP
def _band_spearman(z_rough, z_mar, elev_grid):
    band_items = list(ELEV_BANDS.items())
    r_vals = np.full(len(band_items), np.nan)
    n_vals = np.zeros(len(band_items), dtype=int)
    for bi, (_, (lo, hi, _)) in enumerate(band_items):
        mask = ((elev_grid >= lo) & (elev_grid < hi) &
                np.isfinite(z_rough) & np.isfinite(z_mar))
        if mask.sum() < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(z_mar[mask], z_rough[mask])
        r_vals[bi] = rho
        n_vals[bi] = int(mask.sum())

    return r_vals, n_vals

def _draw_heatmap(ax, r_matrix, col_labels, row_labels, band_colors, norm, cmap, title, show_yticks=True, fig=None, show_cbar=False):
    im = ax.imshow(r_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    if show_cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03, label="Spearman r")
        cb.ax.tick_params(labelsize=7)

    for vi in range(r_matrix.shape[0]):
        for ci in range(r_matrix.shape[1]):
            rv = r_matrix[vi, ci]
            if not np.isfinite(rv):
                ax.text(ci, vi, "—", ha="center", va="center", fontsize=7.5, color="#aaaaaa")
                continue
            col = "white" if abs(rv) > 0.35 else "black"
            ax.text(ci, vi, f"{rv:+.2f}", ha="center", va="center", fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8.5)
    for ci, c in enumerate(band_colors):
        ax.get_xticklabels()[ci].set_color(c)

    if show_yticks:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8.5)
    else:
        ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=6)

    return im

def plot_mar_correlations(z_2019, elev_grid, yr_grids, ice, X, Y):
    all_yrs     = sorted(yr_grids.keys())
    band_items  = list(ELEV_BANDS.items())
    band_names  = [b.split("(")[0].strip() for b, _ in band_items]
    band_colors = [v[2] for _, v in band_items]
    n_bands     = len(band_items)
    var_labels  = [v[1] for v in MAR_EXPLORE_VARS]
    n_vars      = len(MAR_EXPLORE_VARS)

    z_spike = z_2019.copy(); z_spike[~ice] = np.nan

    R_spike = np.full((n_vars, n_bands), np.nan)
    R_multi = np.full((n_vars, n_bands), np.nan)

    print("\n── MAR Spearman correlations ──")
    for vi, (varname, varlabel, months) in enumerate(MAR_EXPLORE_VARS):
        print(f"  {varlabel} ...", end=" ", flush=True)
        z_mar_all = _mar_zscore_grids(varname, months, X, Y, all_yrs, ice)
        if z_mar_all is None:
            print("no data"); continue

        if SPIKE_YEAR in z_mar_all:
            r_s, _ = _band_spearman(z_spike, z_mar_all[SPIKE_YEAR], elev_grid)
            R_spike[vi] = r_s

        pool_r, pool_v = [], []
        for yr in all_yrs:
            if yr not in z_mar_all or yr not in yr_grids:
                continue
            zr = yr_grids[yr].copy(); zr[~ice] = np.nan
            pool_r.append(zr); pool_v.append(z_mar_all[yr])

        if pool_r:
            zr_s = np.stack(pool_r, axis=0)
            zv_s = np.stack(pool_v, axis=0)
            for bi, (_, (lo, hi, _)) in enumerate(band_items):
                bm   = ((elev_grid >= lo) & (elev_grid < hi) & np.isfinite(elev_grid))
                zr_b = zr_s[:, bm].ravel()
                zv_b = zv_s[:, bm].ravel()
                fin  = np.isfinite(zr_b) & np.isfinite(zv_b)
                if fin.sum() < 10:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, _ = spearmanr(zv_b[fin], zr_b[fin])
                R_multi[vi, bi] = rho
        print("done")

    all_r = np.concatenate([R_spike.ravel(), R_multi.ravel()])
    vmax  = max(float(np.nanmax(np.abs(all_r))), 0.1)
    norm  = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap  = "RdBu_r"

    fig, (ax_l, ax_r) = plt.subplots( 1, 2, facecolor="white",
        figsize=(max(10, 2.2 * n_bands * 2), max(5, 0.55 * n_vars + 2)), gridspec_kw={"wspace": 0.08})

    _draw_heatmap(ax_l, R_spike, band_names, var_labels, band_colors, norm, cmap,
                  title=f"{SPIKE_YEAR} only", show_yticks=True, show_cbar=False)
    _draw_heatmap(ax_r, R_multi, band_names, var_labels, band_colors, norm, cmap,
                  title=f"Multi-year  ({all_yrs[0]}–{all_yrs[-1]})", show_yticks=False, show_cbar=True, fig=fig)

    fig.suptitle(
        f"Spearman r: roughness z-score vs MAR variable z-score\n"
        f"Annual means  |  per-cell MAD normalisation  |  "
        f"{GRID_RES//1000} km grid",
        fontsize=10, y=1.01)

    return fig

### MAIN
def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = load_global_cells()
    elev_lookup = load_elevation_lookup()

    print("\nBuilding grid and ice mask ...")
    X, Y, rc = build_grid(cells)
    ice_mask = build_ice_mask(X, Y)

    print("\nLoading MAR elevation ...")
    mar_elev = _load_mar_elev(X, Y)
    if mar_elev is not None:
        mar_elev[~ice_mask] = np.nan

    print("\nComputing spike z-score grids ...")
    z_2019, elev_grid, yr_grids = build_spike_grids(
        cells, elev_lookup, X, Y, rc)

    z_2019[~ice_mask] = np.nan
    elev_grid[~ice_mask] = np.nan
    for yr in yr_grids:
        yr_grids[yr][~ice_mask] = np.nan

    print("\n── Figure 1: 2019 anomaly map ──")
    plot_anomaly_map(z_2019, elev_grid, mar_elev, X, Y)

    print("\n── Figure 2: monthly time series ──")
    plot_timeseries(cells, elev_lookup, rc, ice_mask)

    print("\n── Figure 3: MAR correlations ──")
    plot_mar_correlations(z_2019, elev_grid, yr_grids, ice_mask, X, Y)

    plt.show()
    print(f"\nDone.  All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    run()