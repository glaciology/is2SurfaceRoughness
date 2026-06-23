"""
plot_roughness_spatial.py
Author: Derek Pickell
Node-level trend and cell-level seasonal analysis for a single CSV tile.

Trend and seasonal contrast are computed at different spatial scales, reflecting the underlying sampling geometry of ICESat-2.

  TREND  — [m/yr] estimated at the "node" level (~150 m snap grid), then aggregated into GRID_RES cells via robust weighted median.
           Each node sits on a single ground track that repeats every ~91 days, providing a time series suitable for slope estimation.

  SEASONAL CONTRAST  — [m] estimated by pooling raw pass-median values from all nodes within a GRID_RES cell, fitting a single
           OLS linear trend to the pooled series, and computing SUMMER minus WINTER bisquare-weighted means on the pooled
           residuals. Pooling across nodes combines multiple ground tracks with different orbital repeat phases, collectively
           spanning the full seasonal cycle that no single node can provide. GRID_RES should be large enough to contain
           several ground tracks (~10-12 km cross-track spacing at 70 N).

All beams from the same overpass at the same node are collapsed to a single median before any statistical fitting. The aggregation key is
(x_node, y_node, pass_id): one temporal sample per overpass per node, regardless of how many beams contributed.

Statistical pipeline
--------------------
Node level:
  1.  Snap raw segments to SNAP_M nodes (shared.py: load_and_project).
  2.  Assign globally unique pass IDs via a per-beam time-gap criterion.
  3.  Groupby (x_node, y_node, pass_id) -> one median per overpass per node.
  4.  Per node: MAD outlier removal (n >= 10?), minimum temporal span and minimum-pass checks, pairwise Theil-Sen slope with a minimum-
      pair-separation gate.

Cell level:
  5.  Aggregate node trends into GRID_RES cells via robust weighted median (weight = 1 / trend_sigma^2). 
      Cell sigma is the weighted MAD of node trends divided by sqrt(n_nodes).
  6.  Pool raw pass-median values from all nodes in the cell. 
      Fit a single OLS linear trend to the pooled series and compute summer minus winter bisquare-weighted means on the residuals.
      Require MIN_SEASON_TRACKS distinct passes in each seasonal pool.
  7.  Confidence gate: controlled by SIGNIFICANCE_MODE.
      "snr"        |trend| / cell_sigma >= SNR_THRESHOLD
                   AND n_nodes >= MIN_NODES_CELL  (both trend and seasonal)
      "empirical"  trend: all finite cells are shown (no gate applied);
                   seasonal: |seas_diff| > sigma_empirical, where
                   sigma_empirical = EMPIRICAL_SIGMA_SLOPE * node_median_rms
                   * sqrt(1/n_summer_tracks + 1/n_winter_tracks),
                   propagating the per-measurement empirical noise model
                   (derived from crossover pairs in error_v_rms.py) through
                   the summer-minus-winter difference.

Full pipeline
-------------
  1.  load_and_project    — CSV -> projected, filtered, snapped df (shared.py)
  2.  assign_pass_ids     — globally unique integer pass label per obs
  3.  compute_node_stats  — pass-collapse + Theil-Sen per node
  4.  aggregate_to_grid   — node trends + pooled seasonal -> grid
  5.  plot_roughness      — render trend and seasonal maps
  6.  run_single_analysis — end-to-end wrapper
"""
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from pathlib import Path
from scipy.stats import theilslopes

from shared import (
    BaseConfig,
    TRANSFORMER,
    build_mask,
    load_and_project,
    obs_sigma,
    save_as_geotiff,
)

mpl.rcParams["axes.labelsize"]   = 12
mpl.rcParams["axes.labelweight"] = "light"
mpl.rcParams["axes.titlesize"]   = 10

###  CONFIGURATION  ###
class Config(BaseConfig):
    """
    Full configuration for the two-stage trend/seasonal analysis.
    Inherits all BaseConfig fields; adds node/trend/grid/confidence
    parameters.
    """
    # NODE SNAPPING
    # ICESat-2 SlideRule segments are ~200 m long along track. A snap
    # grid of 150 m ensures at most one or two segments per pass land on
    # each node
    SNAP_M         = 150            # meters

    # PASS ID TIME GAP
    # A new overpass begins when consecutive observations on the same beam
    # are separated by more than TIME_GAP_HOURS. ICESat-2 repeats every
    # ~91 days; a single orbit takes ~90 min, so 1 h should be very very generous
    TIME_GAP_HOURS = 1.0

    # PER NODE (TREND CALCULATION) REQUIREMENTS
    MIN_PASSES     = 6              # minimum independent passes per node (think, 6 passes over 2019-2025 period)
    MIN_SPAN_YR    = 3.0            # minimum time span (years)

    # OUTLIER REMOVAL
    MAD_THRESHOLD  = 2.5            # keep |y - median| <= k * MAD / 0.6745
    MIN_PAIR_DT_YR = 0.08           # ~1 month; minimum pair separation for Theil-Sen to exclude near-simultaneous
                                    # passes from the slope distribution. Thus, data <MIN_PAIR_DT_YR not included in trend calculation. 
    BISQUARE_K     = 4.685          # Tukey bisquare tuning constant

    # GRID RESOLUTION: IMPORTANT!!!
    GRID_RES       = 3500           # cell side length (meters)

    # SEASONAL CONTRAST
    # MIN_SEASON_TRACKS: min number of unique pass_ids contribute to both summer and winter pools before calculating seas_diff. 
    # Prevents single-track cells with poor sampling from contaminating seasonal calculation. 
    MIN_SEASON_TRACKS  = 2          # equivalent to...
    SEAS_AS_PCT_MEDIAN = False      # plot seasonality as a percentage of median roughness 
    PLOT_SIGNIFICANT   = False      # only plot significant (>SNR) data

    # SIGNIFICANCE/UNCERTAINTY
    # "snr"       — Option 1: SNR-based confidence gate (original behavior)
    # "empirical" — Option 2: empirical noise model from crossover pairs
    SIGNIFICANCE_MODE    = "snr"
    EMPIRICAL_SIGMA_SLOPE = 1.77 # derived from error_v_rims.py (used in empirical); noise model sigma(r) = EMPIRICAL_SIGMA_SLOPE * r,
    SNR_THRESHOLD  = 1.5             # SNR threshold: |cell_trend| / cell_sigma >= SNR_THRESHOLD AND AND n_nodes >= MIN_NODES_CELL
                                     # cell_sigma is the weighted MAD of node trends within the cell
                                     # divided by sqrt(n_nodes), combining inter-node spread with
                                     # individual node uncertainties.  SNR 1.5 ~ 87% one-sided normal CI.
    MIN_NODES_CELL = 10              # minimum nodes per cell for aggregation

    # ── output ────────────────────────────────────────────────────────────────
    OUTPUT_DIR     = Path("summaries_temporal")


###PASS-ID ASSIGNMENT###
def assign_pass_ids(times, spot_nums, gap_hours = Config.TIME_GAP_HOURS):
    """
    Label each raw observation with a globally unique integer pass ID.

    Each beam (spot_num) is processed independently.

    Parameters:
    times: datetime64 array, length M
    spot_nums: beam identifier array, length M
    gap_hours: time gap (hours) that separates consecutive overpasses
    """
    gap_ns   = gap_hours * 3_600 * 1e9             # hours -> nanoseconds
    pass_ids = np.empty(len(times), dtype=np.int64)
    t_ns     = np.asarray(times, dtype="datetime64[ns]").astype(np.int64)
    offset   = 0

    for spot in np.unique(spot_nums):
        mask     = spot_nums == spot
        t_spot   = t_ns[mask]
        sort_idx = np.argsort(t_spot)

        gaps = np.diff(t_spot[sort_idx]) > gap_ns
        ids  = np.concatenate([[0], np.cumsum(gaps)])   # 0-based within beam

        ids_unsorted           = np.empty(len(ids), dtype=np.int64)
        ids_unsorted[sort_idx] = ids + offset
        pass_ids[mask]         = ids_unsorted
        offset                += int(ids.max()) + 1

    return pass_ids

###  STATISTICAL HELPERS  ###
def bisquare_weights(residuals, k):
    """Tukey bisquare weights; downweights observations far from the median."""
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad == 0:
        return np.ones_like(residuals, dtype=float)
    scale = k * mad / 0.6745
    u     = residuals / scale
    return np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)

def weighted_mean(vals, k):
    """
    Bisquare-weighted mean and standard error.

    Returns (mean, sigma).  Both NaN when all weights are zero.
    """
    if len(vals) == 0:
        return np.nan, np.nan
    resid = vals - np.median(vals)
    w     = bisquare_weights(resid, k)
    W     = w.sum()
    if W == 0:
        return np.nan, np.nan
    mean  = np.sum(w * vals) / W
    n_eff = W ** 2 / np.sum(w ** 2)
    var   = np.sum(w * (vals - mean) ** 2) / W
    sigma = np.sqrt(var / max(n_eff, 1.0))
    
    return float(mean), float(sigma)

def theil_trend(t, y, min_pair_dt = Config.MIN_PAIR_DT_YR):
    """
    Pairwise Theil-Sen slope with a minimum pair-separation gate.

    Pairs separated by less than min_pair_dt years are excluded to prevent near-simultaneous overpasses from flooding the slope
    distribution with noisy short-baseline estimates.

    slope_sigma is half the 95% CI width from scipy.stats.theilslopes, with a MAD-based fallback when scipy raises an exception.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < 3:
        return np.nan, np.nan, 0

    i_idx, j_idx = np.triu_indices(n, k=1)
    dt   = t[j_idx] - t[i_idx]
    dy   = y[j_idx] - y[i_idx]
    keep = np.abs(dt) >= min_pair_dt

    if keep.sum() < 1:
        return np.nan, np.nan, 0

    slopes = dy[keep] / dt[keep]
    slope  = float(np.median(slopes))

    try:
        result      = theilslopes(y, t, confidence=0.95)
        slope_sigma = float((result.high_slope - result.low_slope) / 2.0)
    except Exception:
        slope_sigma = float(np.median(np.abs(slopes - slope)) / 0.6745 / np.sqrt(max(keep.sum(), 1)))

    return slope, slope_sigma, int(keep.sum())

###  NODE-LEVEL STATISTICS  ###
def compute_node_stats(df, config=Config):
    """
    Compute a Theil-Sen trend for every (x_node, y_node) node.

    Processing steps:
    1. Assign globally unique pass IDs via a per-beam time-gap criterion.
    2. Groupby (x_node, y_node, pass_id) -> one median per overpass per node. Collapses all beams from the same overpass at the same
        node into a single temporal sample.
    3. Per node: MAD outlier removal (when n >= 10), minimum-span and minimum-pass checks, pairwise Theil-Sen trend.
    """

    voi = config.VALUE_OF_INTEREST # value of interest

    # get pass ids
    pass_ids = assign_pass_ids(df["time"].values.astype("datetime64[ns]"), df["spot_num"].values, gap_hours=config.TIME_GAP_HOURS)
    df = df.copy()
    df["pass_id"] = pass_ids
    print(f"  Raw obs: {len(df):,}  |  Unique passes: {df['pass_id'].nunique():,}")

    # collapse to one median per (node, pass) 
    # All beams from the same overpass at the same node collapse to one temporal sample, in the case that there are multiple points in a single node
    node_pass = (df.groupby(["x_node", "y_node", "pass_id"], sort=False).agg(pass_val  = (voi, "median"), pass_time = ("time", "median")).reset_index()
    )

    # decimal year and month derived once for the full aggregated table
    dt_index = pd.to_datetime(node_pass["pass_time"])
    node_pass["t_yr"] = (dt_index.dt.year + (dt_index.dt.dayofyear - 1) / 365.25).astype(np.float32)
    node_pass["month"] = dt_index.dt.month.astype(np.int8)

    # trend for each node
    records = []
    for (xn, yn), grp in node_pass.groupby(["x_node", "y_node"], sort=False):
        y   = grp["pass_val"].values.astype(float)
        t   = grp["t_yr"].values.astype(float)
        m   = grp["month"].values
        pid = grp["pass_id"].values

        valid        = np.isfinite(y)
        y, t, m, pid = y[valid], t[valid], m[valid], pid[valid]
        n            = len(y)
        n_raw        = n
        span_yr      = float(t.max() - t.min()) if n > 1 else 0.0

        # filter nodes without enough data quanity or temporal span here
        if n < config.MIN_PASSES or span_yr < config.MIN_SPAN_YR:
            continue

        # MAD outlier removal
        # Applied only when n >= 10 so the median is itself robust.
        if n >= 10:
            med_y = np.median(y)
            mad_y = np.median(np.abs(y - med_y))
            if mad_y > 0:
                keep         = np.abs(y - med_y) <= config.MAD_THRESHOLD * mad_y / 0.6745
                y, t, m, pid = y[keep], t[keep], m[keep], pid[keep]
                n            = len(y)
                span_yr      = float(t.max() - t.min()) if n > 1 else 0.0

        if n < config.MIN_PASSES or span_yr < config.MIN_SPAN_YR:
            continue

        # Theil-Sen trend
        t0 = t.mean()           # center for numerical stability
        slope, slope_sigma, _ = theil_trend(t - t0, y, config.MIN_PAIR_DT_YR)

        if not np.isfinite(slope):
            continue

        records.append(dict(
            x_node       = int(xn),
            y_node       = int(yn),
            trend        = float(slope),
            trend_sigma  = float(slope_sigma),
            node_median  = float(np.median(y)),   # for SEAS_AS_PCT_MEDIAN and
                                                   # empirical uncertainty
            n_obs        = int(n),
            n_obs_raw    = int(n_raw),
            span_yr      = float(span_yr),
        ))

    print(f"  Nodes with valid trend: {len(records):,}")
    return pd.DataFrame(records) if records else pd.DataFrame()

###  CELL-LEVEL AGGREGATION  ###
def aggregate_to_grid(node_df, obs_df, config=Config):
    """
    Aggregate node-level trend estimates into GRID_RES (cells) 
    AND compute cell-level seasonal contrast from pooled raw observations.

    Trend aggregation (per cell):
    Weighted median of node trends, where weights are 1 / trend_sigma^2.
    Cell sigma is the weighted MAD of node trends divided by sqrt(n_nodes), combining inter-node spread with individual node uncertainties.

    Seasonal contrast (per cell):
    Raw pass-median values from obs_df are pooled across all nodes in the cell. A single OLS linear trend is removed from the pooled
    series before computing summer minus winter bisquare-weighted means. Using raw values and one shared detrend rather than concatenating
    per-node residuals ensures a single consistent baseline for the whole cell. 

    Parameters:
    node_df: output of compute_node_stats — one row per node, trend only
    obs_df: raw observation DataFrame from load_and_project or Pass 1 parquet — must contain x_node, y_node, time, spot_num,
              and config.VALUE_OF_INTEREST columns
    config: Config instance
    """
    g   = int(config.GRID_RES)
    voi = config.VALUE_OF_INTEREST
    df  = node_df.copy()

    # assign nodes to grid cells
    df["ix"] = np.floor(df["x_node"] / g).astype(np.int32)
    df["iy"] = np.floor(df["y_node"] / g).astype(np.int32)

    ix_min, ix_max = int(df["ix"].min()), int(df["ix"].max())
    iy_min, iy_max = int(df["iy"].min()), int(df["iy"].max())
    shape = (iy_max - iy_min + 1, ix_max - ix_min + 1)

    xg = (np.arange(ix_min, ix_max + 1) + 0.5) * g
    yg = (np.arange(iy_min, iy_max + 1) + 0.5) * g
    X, Y = np.meshgrid(xg, yg)

    # pre-allocate output grids
    trend_val           = np.full(shape, np.nan)
    trend_snr           = np.full(shape, np.nan)
    trend_cnt           = np.zeros(shape, dtype=np.int32)
    trend_conf          = np.zeros(shape, dtype=bool)
    trend_empirical_conf = np.zeros(shape, dtype=bool)
    trend_node_med_rms  = np.full(shape, np.nan)

    seas_val            = np.full(shape, np.nan)
    seas_snr            = np.full(shape, np.nan)
    seas_cnt            = np.zeros(shape, dtype=np.int32)
    seas_conf           = np.zeros(shape, dtype=bool)
    seas_empirical_conf = np.zeros(shape, dtype=bool)
    seas_node_med_rms   = np.full(shape, np.nan)

    # Prepare raw observations for seasonal pooling 
    # Assign pass IDs and collapse to one pass-median per (node, pass).
    # This mirrors compute_node_stats but operates on the full obs_df so
    # nodes that failed the trend quality gate still contribute seasonal data.
    obs = obs_df.copy()
    obs["pass_id"] = assign_pass_ids(obs["time"].values.astype("datetime64[ns]"), obs["spot_num"].values, gap_hours=config.TIME_GAP_HOURS)
    obs["ix"] = np.floor(obs["x_node"] / g).astype(np.int32)
    obs["iy"] = np.floor(obs["y_node"] / g).astype(np.int32)

    dt_obs = pd.to_datetime(obs["time"])
    obs["t_yr"] = (dt_obs.dt.year + (dt_obs.dt.dayofyear - 1) / 365.25).astype(np.float32)
    obs["month"]  = dt_obs.dt.month.astype(np.int8)

    # collapse to one pass-median per (cell, pass) across all nodes —
    # one temporal sample per overpass per cell, regardless of how many
    # nodes or beams contributed
    cell_pass = (obs.groupby(["ix", "iy", "pass_id"], sort=False)
        .agg(pass_val   = (voi,"median"), t_yr = ("t_yr",  "median"), month = ("month", "median")).reset_index())
    cell_pass["month"] = cell_pass["month"].round().astype(np.int8)

    # aggragate per cell
    for (ix, iy), cell in df.groupby(["ix", "iy"], sort=False):
        r = int(iy - iy_min)
        c = int(ix - ix_min)

        trends  = cell["trend"].values.astype(float)
        sigmas  = cell["trend_sigma"].values.astype(float)
        n_nodes = len(trends)

        # cell median RMS — median of per-node roughness medians; used by
        # both SEAS_AS_PCT_MEDIAN normalization and the empirical noise model
        cell_node_med_rms = float(np.median(cell["node_median"].values))

        # Trend: robust weighted median across nodes
        valid_s = np.isfinite(sigmas) & (sigmas > 0)
        if valid_s.sum() == 0:
            continue

        w = np.where(valid_s, 1.0 / sigmas ** 2, 0.0)
        sort_idx = np.argsort(trends)
        t_s = trends[sort_idx]
        w_s = w[sort_idx]
        cum_w = np.cumsum(w_s)
        cell_trend = float(t_s[np.searchsorted(cum_w, cum_w[-1] / 2.0)])

        # cell sigma: weighted MAD / sqrt(n_nodes) combines inter-node spread with individual node uncertainties
        abs_dev = np.abs(trends - cell_trend)
        sort2 = np.argsort(abs_dev)
        cum_w2 = np.cumsum(w[sort2])
        wmad = float(abs_dev[sort2][np.searchsorted(cum_w2, cum_w2[-1] / 2.0)])
        cell_sigma = (wmad / 0.6745) / np.sqrt(max(n_nodes, 1))

        trend_val[r, c] = cell_trend
        trend_cnt[r, c] = n_nodes
        trend_node_med_rms[r, c] = cell_node_med_rms

        if cell_sigma > 0:
            snr = abs(cell_trend) / cell_sigma
            trend_snr[r, c] = snr
            trend_conf[r, c] = (snr >= config.SNR_THRESHOLD and n_nodes >= config.MIN_NODES_CELL)
        else:
            trend_conf[r, c] = n_nodes >= config.MIN_NODES_CELL

        # empirical mode: all finite trend cells are shown — no gate applied
        trend_empirical_conf[r, c] = np.isfinite(cell_trend)

        ### SEASONAL CONTRAST: pool raw pass medians, detrend once.
        # Select all pass-level samples for this cell from the pre-collapsed cell_pass table. Fit one OLS trend to the pooled series so that
        # all nodes share a single consistent baseline before the summer / winter means are computed.
        cp = cell_pass[(cell_pass["ix"] == ix) & (cell_pass["iy"] == iy)]
        if len(cp) == 0:
            continue

        all_vals   = cp["pass_val"].values.astype(float)
        all_t_yr   = cp["t_yr"].values.astype(float)
        all_months = cp["month"].values
        all_pids   = cp["pass_id"].values

        # single detrend on the pooled cell time series
        finite = np.isfinite(all_vals)
        all_vals, all_t_yr, all_months, all_pids = (all_vals[finite], all_t_yr[finite], all_months[finite], all_pids[finite])

        if len(all_vals) < 3:
            continue

        # MAD outlier removal on pooled cell series before detrending
        if len(all_vals) >= 10: # HARDCODED VALUE HERE
            med_cell = np.median(all_vals)
            mad_cell = np.median(np.abs(all_vals - med_cell))
            if mad_cell > 0:
                keep_mad = (np.abs(all_vals - med_cell) <= config.MAD_THRESHOLD * mad_cell / 0.6745)
                all_vals, all_t_yr, all_months, all_pids = (all_vals[keep_mad], all_t_yr[keep_mad], all_months[keep_mad], all_pids[keep_mad])

        if len(all_vals) < 3:
            continue

        # single detrend on the pooled cell time series
        t0_cell = all_t_yr.mean()
        ts_result = theilslopes(all_vals, all_t_yr - t0_cell)
        cell_s = ts_result.slope
        cell_i = ts_result.intercept
        all_resid = all_vals - (cell_s * (all_t_yr - t0_cell) + cell_i)

        summer_mask = np.isin(all_months, config.SEASONS["summer"])
        winter_mask = np.isin(all_months, config.SEASONS["winter"])

        n_summer_tracks = len(np.unique(all_pids[summer_mask]))
        n_winter_tracks = len(np.unique(all_pids[winter_mask]))

        if (n_summer_tracks >= config.MIN_SEASON_TRACKS and n_winter_tracks >= config.MIN_SEASON_TRACKS):

            mu_s, sig_s = weighted_mean(all_resid[summer_mask], config.BISQUARE_K)
            mu_w, sig_w = weighted_mean(all_resid[winter_mask], config.BISQUARE_K)

            if np.isfinite(mu_s) and np.isfinite(mu_w):
                sd = float(mu_s - mu_w)
                sd_sigma = float(np.hypot(sig_s, sig_w))

                # optional normalization by cell median RMS
                # if getattr(config, "SEAS_AS_PCT_MEDIAN", False):
                #     if cell_node_med_rms > 0:
                #         sd = sd / cell_node_med_rms * 100.0
                #         sd_sigma = sd_sigma / cell_node_med_rms * 100.0
                #     else:
                #         sd = np.nan

                seas_val[r, c] = sd
                seas_cnt[r, c] = int(n_summer_tracks + n_winter_tracks)
                seas_node_med_rms[r, c] = cell_node_med_rms

                if np.isfinite(sd) and sd_sigma > 0:
                    snr = abs(sd) / sd_sigma
                    seas_snr[r, c] = snr
                    seas_conf[r, c] = snr >= config.SNR_THRESHOLD
                else:
                    seas_conf[r, c] = True

                # empirical uncertainty propagated through the summer-minus-winter difference:
                #   sigma_seas = EMPIRICAL_SIGMA_SLOPE * r * sqrt(1/n_summer + 1/n_winter)
                # where r is the cell's median node_median RMS.
                if cell_node_med_rms > 0 and np.isfinite(sd):
                    sigma_emp = (config.EMPIRICAL_SIGMA_SLOPE* cell_node_med_rms* np.sqrt(1.0 / n_summer_tracks + 1.0 / n_winter_tracks))
                    seas_empirical_conf[r, c] = abs(sd) > sigma_emp

    # outputs
    grids = {
        "trend": dict(
            val              = trend_val,
            snr              = trend_snr,
            count            = trend_cnt,
            confident        = trend_conf,
            empirical_conf   = trend_empirical_conf,
            node_median_rms  = trend_node_med_rms,
            masked           = None,   # filled in run_single_analysis
        ),
        "seas_diff": dict(
            val              = seas_val,
            snr              = seas_snr,
            count            = seas_cnt,
            confident        = seas_conf,
            empirical_conf   = seas_empirical_conf,
            node_median_rms  = seas_node_med_rms,
            masked           = None,   # filled in run_single_analysis
        ),
    }
    return X, Y, grids

###  PLOTTING  ###
def plot_roughness(sig_grid, full_grid, confident_mask, n_confident, n_total, pct_pos, pct_neg, X, Y, name="year", config=Config, significance_mode="snr"):
    
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = np.nanpercentile(np.abs(sig_grid), 98)
    if not np.isfinite(vmax) or vmax == 0:
        print(f"  [plot_roughness] No finite confident data for '{name}' — skipping.")
        plt.close(fig)
        return

    seas_pct = getattr(config, "SEAS_AS_PCT_MEDIAN", False)

    if seas_pct and name == "seasonal":
        # linear symmetric scale for percentage — already normalized
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        # log-linear scale
        norm = SymLogNorm(linthresh=vmax * 0.1, linscale=1, vmin=-vmax, vmax=vmax, base=10)

    ax.pcolormesh(X, Y, full_grid, cmap="RdBu_r", norm=norm, shading="auto", alpha=1, zorder=3)
    trend_plot = ax.pcolormesh(X, Y, sig_grid, cmap="RdBu_r", norm=norm, shading="auto", zorder=4)

    coast = gpd.read_file(config.COASTLINE_PATH).to_crs("EPSG:3413")
    coast.plot(ax=ax, facecolor="#e0e0e0", zorder=1)
    config.GDF.boundary.plot(ax=ax, color="black", linewidth=0.1, zorder=5)

    minx, miny, maxx, maxy = config.GDF.total_bounds
    ax.set_xlim(minx - 15_000, maxx + 15_000)
    ax.set_ylim(miny - 15_000, maxy + 15_000)

    if name == "year":
        cbar_label = "Trend [m yr\u207b\u00b9]"
    elif seas_pct:
        cbar_label = "Seasonal diff (%)  [summer \u2212 winter]"
    else:
        cbar_label = "Seasonal diff (m)  [summer \u2212 winter]"

    plt.colorbar(trend_plot, ax=ax, label=cbar_label)

    # signifiance criteria
    if significance_mode == "empirical":
        if name == "year":
            sig_label = "empirical \u03c3: all finite cells shown"
        else:
            sig_label = (f"empirical \u03c3 = {config.EMPIRICAL_SIGMA_SLOPE}\u00b7r \u00b7\u221a(1/n_s + 1/n_w)")
    else:
        sig_label = (f"SNR \u2265 {config.SNR_THRESHOLD}, n \u2265 {config.MIN_NODES_CELL} nodes")

    ax.set_title(
        f"{name}  |  {config.VALUE_OF_INTEREST}  |  grid {config.GRID_RES} m\n"
        f"Significant: {n_confident}/{n_total}  ({sig_label})\n"
        f"Sig+: {pct_pos:.1f}%   Sig\u2212: {pct_neg:.1f}%"
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_aspect("equal")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    tif = (config.OUTPUT_DIR / f"roughness_map_{name}_{config.VALUE_OF_INTEREST}_{config.GRID_RES}.tif")
    save_as_geotiff(tif, full_grid, X, Y)
    plt.tight_layout()

###  END-TO-END WRAPPER  ###
def run_single_analysis(path, config=Config):
    """
    Full pipeline for one CSV.

    Steps: load -> node stats -> grid aggregation -> plots.

    Parameters:
    path: str, Path
    config: Config instance or subclass

    Returns:
    node_df: per-node statistics DataFrame (from compute_node_stats)
    X, Y: coordinate meshgrids used for the plots
    grids: dict returned by aggregate_to_grid
    """
    if isinstance(path, (list, tuple)):
        parts = [load_and_project(p, config) for p in path]
        df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    else:
        df = load_and_project(path, config)

    if df.empty:
        print("No data after loading — aborting.")
        return None, None, None

    print("Computing node-level statistics ...")
    node_df = compute_node_stats(df, config)

    print(f"  Nodes with trend     : {len(node_df):,}")
    print("Aggregating to grid ...")
    X, Y, grids = aggregate_to_grid(node_df, df, config)

    # SIGNIFICANCE MASKING
    mode = getattr(config, "SIGNIFICANCE_MODE", "snr")
    for g in grids.values():
        if mode == "empirical":
            g["masked"] = np.where(g["empirical_conf"], g["val"], np.nan)
        else:
            g["masked"] = np.where(g["confident"],      g["val"], np.nan)

    for metric, name in [("trend", "year"), ("seas_diff", "seasonal")]:
        g       = grids[metric]
        val     = g["val"]
        conf    = g["empirical_conf"] if mode == "empirical" else g["confident"]
        n_total = int(np.isfinite(val).sum())
        n_conf  = int(conf.sum())
        pct_pos = 100 * np.sum(conf & (val > 0)) / max(n_conf, 1)
        pct_neg = 100 * np.sum(conf & (val < 0)) / max(n_conf, 1)

        print(f"\n-- {name} --")
        print(f"  Cells with data : {n_total}")
        print(f"  Significant cells : {n_conf}  (mode: {mode})")
        print(f"  +/-   : {pct_pos:.2f}% / {pct_neg:.2f}%")

        if config.PLOT_SIGNIFICANT:
            plot_roughness(g["masked"], g["masked"], conf, n_conf, n_total, pct_pos, pct_neg, X, Y, name=name, config=config, significance_mode=mode)
        else:
            plot_roughness(g["masked"], val, conf, n_conf, n_total, pct_pos, pct_neg, X, Y, name=name, config=config, significance_mode=mode)

    plt.show()
    return node_df, X, Y, grids

###  MAIN  ###
if __name__ == "__main__":
    path = ("/Users/f005cb1/Documents/Github/is2Roughness/testData/region_052_segments_combined.csv")
    run_single_analysis(path)