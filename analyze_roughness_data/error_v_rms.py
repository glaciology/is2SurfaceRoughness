"""
script: error_v_rms.py
author: Derek Pickell
purpose: ICESat-2 Roughness Noise Model — two-regime version. Data is heteroskedastic... see manuscript for what this means!

Uses/generates a file called crossover_raw.pkl 
"""
import pickle
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from shared import TRANSFORMER

### CONFIGURATION ###
DATA_DIR             = Path("/Users/f005cb1/Documents/GitHub/is2Roughness/testData/")
OUTPUT_DIR           = Path("./summaries_error_estimate")
PICKLE_RAW           = OUTPUT_DIR / "crossover_raw.pkl"
SPATIAL_THRESH       = 5.0 # how far apart data needs to be to calculated the difference in values
TEMPORAL_THRESH_DAYS = 30. # how far temporally the data needs to be, at a maximum, to calculate differences

### BINNING
RMS_SPLIT_M          = 0.10   # boundary between smooth and rough regimes (in meters, value of roughness)
BIN_WIDTH_SMOOTH_M   = 0.005  # fine bins for the well-sampled smooth region
BIN_WIDTH_FULL_M     = 0.05   # coarser bins for the full range
MIN_PAIRS_BIN        = 30     # minimum pairs to include a bin
WINDOW_DENSE_PCTL    = 95.0   # tail cap

### STYLE + PLOTTING
FG      = "#1a1a2e"      # deep navy — text, spines, markers
GRID_C  = "#ebebeb"      # light grey grid
C_SMOOTH = "#2c6fad"     # steel blue — smooth-regime fit
C_FULL   = "#b94040"     # brick red  — full-range fit
C_REF    = "#aaaaaa"     # light grey — identity / extrapolation reference
COUNT_CMAP = "GnBu"        # cool blue-green sequential; clean on white

### DATA LOADING AND PAIR FINDING
def load_file(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["RMS", "mean_surface"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset=["lon_centroid", "lat_centroid", "time", "RMS"])
    df = df.copy()
    df["x"], df["y"] = TRANSFORMER.transform(df["lon_centroid"].values, df["lat_centroid"].values)

    return df.reset_index(drop=True)

def collect_all_pairs(data_dir, spatial_thresh, temporal_thresh_days):
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    
    all_diffs, all_mean_rms = [], []
    for path in csv_files:
        print(f"  {path.name} …", end=" ", flush=True)
        try:
            df = load_file(path)
        except Exception as exc:
            print(f"SKIPPED ({exc})"); continue
        
        d, m = get_pairs(df, spatial_thresh, temporal_thresh_days)
        print(f"{len(d):,} pairs")
        if len(d):
            all_diffs.append(d)
            all_mean_rms.append(m)

    if not all_diffs:
        raise RuntimeError("No valid pairs found.")
    
    return np.concatenate(all_diffs), np.concatenate(all_mean_rms)

def get_pairs(df, spatial_thresh, temporal_thresh_days):
    coords = np.column_stack((df["x"].values, df["y"].values))
    times  = df["time"].values.astype("datetime64[D]").astype(np.int64)
    rms    = df["RMS"].values.astype(np.float64)
    pairs  = cKDTree(coords).query_pairs(r=spatial_thresh)
    diffs, mean_rms = [], []
    
    for i, j in pairs:
        dt = int(abs(times[i] - times[j]))
        if 0.01 <= dt <= temporal_thresh_days:
            diffs.append(abs(rms[i] - rms[j]))
            mean_rms.append((rms[i] + rms[j]) / 2.0)

    if not diffs:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    
    return np.array(diffs), np.array(mean_rms)

### BINNING DATA FOR DISPLAY AND ANALYSIS PURPOSES
def bin_sigma68(mean_rms, diffs, bin_width, min_pairs, x_max):
    """
    Non-overlapping fixed-width bins over [0, x_max].
    Returns (centers, sigmas, counts): one independent estimate per bin.
    Uses 68% percentile of data
    """
    edges = np.arange(0, x_max + bin_width, bin_width)
    bin_idx = np.digitize(mean_rms, edges) - 1

    centers, sigmas, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = (bin_idx == b)
        n    = int(mask.sum())
        if n < min_pairs:
            continue
        centers.append(float(edges[b] + bin_width / 2))
        sigmas .append(float(np.percentile(diffs[mask], 68.27)))
        counts .append(n)

    return (np.array(centers), np.array(sigmas), np.array(counts))


### FITTING
def linear(x, m, c):
    return m * x + c

def fit_linear(centers, sigmas, counts, label="", force_zero_intercept=False):
    """
    Weighted linear fit  sigma = m·RMS + c.
    If force_zero_intercept=True, fits sigma = m·RMS (c fixed at 0).
    """
    weights = np.sqrt(counts.astype(float))

    if force_zero_intercept:
        # one-parameter fit
        def _model(x, m):
            return m * x
        
        popt, pcov = curve_fit(_model, centers, sigmas, p0=[0.5], sigma=1.0 / weights, bounds=([0], [np.inf]))
        m = float(popt[0])
        c = 0.0
        perr = np.array([float(np.sqrt(pcov[0, 0])), 0.0])
        func = lambda x, m=m: m * x

    else:
        popt, pcov = curve_fit(linear, centers, sigmas, p0=[0.5, 0.0], sigma=1.0 / weights, bounds=([0, 0], [np.inf, np.inf]))
        m, c  = float(popt[0]), float(popt[1])
        perr  = np.sqrt(np.diag(pcov))
        func  = lambda x, m=m, c=c: m * x + c

    pred = func(centers)
    rmse = float(np.sqrt(np.mean((sigmas - pred) ** 2)))

    tag = f" [{label}]" if label else ""
    print(f"\n  Fit{tag}:  σ = {m:.4f}·RMS + {c:.5f}   RMSE = {rmse:.5f} m")
    print(f"    m = {m:.4f} ± {perr[0]:.4f}" + (f"   c = {c:.5f} ± {perr[1]:.5f}" if not force_zero_intercept else "   (c forced = 0)"))

    return {
        "m": m, "c": c, "perr": perr, "rmse": rmse,
        "func": func, "label": label,
        "str": (f"σ = {m:.3f}·RMS" if force_zero_intercept
                else f"σ = {m:.3f}·RMS + {c:.4f}"),
    }

def print_mdc_table(fit, x_min, x_max, label="", n_rows=12):
    func = fit["func"]
    print(f"\n  MDC (95%) = σ × 1.96 × √2   [{label}]")
    print(f"  {'Mean RMS (m)':>14}  {'σ (m)':>10}  {'MDC 95% (m)':>12}")
    print("  " + "─" * 42)
    for xv in np.linspace(max(x_min, 1e-6), x_max, n_rows):
        s   = float(func(xv))
        mdc = 1.96 * s * np.sqrt(2)
        print(f"  {xv:>14.4f}  {s:>10.6f}  {mdc:>12.6f}")


### PLOTTING
def _style_ax(ax, title=""):
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#cccccc")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=FG, labelsize=8.5, length=3, width=0.7)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(color=GRID_C, linewidth=0.6, zorder=0)
    if title:
        ax.set_title(title, fontsize=9.5, color=FG, pad=8, fontweight="normal")

def _scatter_bins(ax, centers, sigmas, counts):
    """Scatter with count-encoded colour and size."""
    norm = mpl.colors.LogNorm(vmin=max(counts.min(), 1), vmax=counts.max())
    cmap   = plt.get_cmap(COUNT_CMAP)
    colors = cmap(norm(counts))
    sizes  = 28 + 100 * (counts - counts.min()) / max(float(counts.max() - counts.min()), 1)
    sc = ax.scatter(centers, sigmas, c=colors, s=sizes, edgecolors=FG, linewidths=0.4, zorder=5)

    return sc, norm, cmap

def _fit_band(ax, fit, x_arr, color, lw=2.0, ls="-", label=None, alpha_band=0.12):
    """Plot fit line + shaded ±1sigma_fit uncertainty band."""
    y     = fit["func"](x_arr)
    m, c  = fit["m"], fit["c"]
    dm, dc = fit["perr"][0], fit["perr"][1]
    y_hi  = (m + dm) * x_arr + (c + dc)
    y_lo  = np.maximum((m - dm) * x_arr + max(c - dc, 0), 0)

    lbl = label if label is not None else fit["str"]
    ax.plot(x_arr, y, color=color, lw=lw, ls=ls, zorder=6, label=lbl)
    ax.fill_between(x_arr, y_lo, y_hi, color=color, alpha=alpha_band, zorder=3)

def plot_two_panel(centers_s, sigmas_s, counts_s, fit_smooth, centers_f, sigmas_f, counts_f, fit_full, dense_cap, output_dir):
    """
    Left panel: smooth region (RMS < RMS_SPLIT_M), fine bins
    Right panel: full dense range, coarser bins
    """
    mpl.rcParams.update({"font.family": "sans-serif", "axes.labelsize": 9, "axes.titlesize": 9.5})

    fig, (ax_l, ax_r) = plt.subplots(1, 2, facecolor="white", gridspec_kw={"wspace": 0.32})

    # LEFT
    _style_ax(ax_l, title=f"Smooth ice  (RMS < {RMS_SPLIT_M} m)")

    sc_l, norm_l, cmap_l = _scatter_bins(ax_l, centers_s, sigmas_s, counts_s)

    x_s = np.linspace(0, RMS_SPLIT_M, 300)
    _fit_band(ax_l, fit_smooth, x_s, C_SMOOTH, label=fit_smooth["str"])

    # identity reference
    ax_l.plot(x_s, x_s, color=C_REF, lw=0.9, ls=":", zorder=2, label="σ = RMS  (ref)")

    ax_l.set_xlim(-0.005, RMS_SPLIT_M * 1.05)
    ax_l.set_ylim(-0.002, sigmas_s.max() * 1.25)
    ax_l.set_xlabel("Mean RMS of pair  (m)")
    ax_l.set_ylabel("σ₆₈ of |ΔRMS|  (m)")
    ax_l.legend(fontsize=8, framealpha=0.9, facecolor="white", edgecolor=GRID_C, labelcolor=FG, loc="upper left")

    cb_l = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_l, cmap=cmap_l), ax=ax_l, shrink=0.72, pad=0.02, aspect=22)
    cb_l.set_label("Pairs per bin", fontsize=8, color=FG)
    cb_l.ax.yaxis.set_tick_params(colors=FG, labelsize=7)
    cb_l.outline.set_edgecolor(GRID_C)

    # stats box
    n_bins_s = len(centers_s)
    n_pairs_s = int(counts_s.sum())
    ax_l.text(0.97, 0.05, f"{n_bins_s} bins  |  {n_pairs_s:,} pairs", transform=ax_l.transAxes, fontsize=7.5, color=FG, ha="right", va="bottom",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID_C, alpha=0.9))

    # RIGHT
    _style_ax(ax_r, title=f"Full range")

    sc_r, norm_r, cmap_r = _scatter_bins(ax_r, centers_f, sigmas_f, counts_f)

    x_f = np.linspace(0, dense_cap, 500)
    _fit_band(ax_r, fit_full, x_f, C_FULL, label=fit_full["str"] + "  [full]")

    # smooth fit extrapolated as dashed reference
    _fit_band(ax_r, fit_smooth, x_f, C_SMOOTH, lw=1.4, ls="--", alpha_band=0.06, label=fit_smooth["str"] + "  [smooth, extrap.]")

    # identity reference
    ax_r.plot(x_f, x_f, color=C_REF, lw=0.9, ls=":", zorder=2, label="σ = RMS  (ref)")

    # vertical split line
    ax_r.axvline(RMS_SPLIT_M, color=C_SMOOTH, lw=0.9, ls="--", alpha=0.5, zorder=4)
    ax_r.text(RMS_SPLIT_M + 0.01, sigmas_f.max() * 0.92, f"split\n{RMS_SPLIT_M} m", fontsize=7, color=C_SMOOTH, va="top")

    ax_r.set_xlim(-0.02, dense_cap * 1.04)
    ax_r.set_ylim(-0.01, sigmas_f.max() * 1.28)
    ax_r.set_xlabel("Mean RMS of pair  (m)")
    ax_r.set_ylabel("σ₆₈ of |ΔRMS|  (m)")
    ax_r.legend(fontsize=8, framealpha=0.9, facecolor="white", edgecolor=GRID_C, labelcolor=FG, loc="upper left")

    cb_r = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_r, cmap=cmap_r), ax=ax_r, shrink=0.72, pad=0.02, aspect=22)
    cb_r.set_label("Pairs per bin", fontsize=8, color=FG)
    cb_r.ax.yaxis.set_tick_params(colors=FG, labelsize=7)
    cb_r.outline.set_edgecolor(GRID_C)

    n_bins_f  = len(centers_f)
    n_pairs_f = int(counts_f.sum())
    ax_r.text(0.97, 0.05, f"{n_bins_f} bins  |  {n_pairs_f:,} pairs",
              transform=ax_r.transAxes, fontsize=7.5, color=FG, ha="right", va="bottom",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID_C, alpha=0.9))

    fig.suptitle(
        f"ICESat-2 roughness noise model  "
        f"|  spatial ≤ {SPATIAL_THRESH} m  "
        f"|  temporal ≤ {TEMPORAL_THRESH_DAYS} d  "
        f"|  σ₆₈ of |ΔRMS|  "
        f"|  shaded band = ±1σ (fit)",
        fontsize=9, color=FG)

    plt.show()


### PICKLES
def _save_pkl(obj, path):
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached → {path}")

def _load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

### MAIN
def run(force_recompute: bool = False):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. pairs 
    if PICKLE_RAW.exists() and not force_recompute:
        print(f"Loading cached pairs from {PICKLE_RAW} …")
        data = _load_pkl(PICKLE_RAW)
        diffs, mean_rms = data["diffs"], data["mean_rms"]
        print(f"  {len(diffs):,} pairs")
    else:
        print(f"Scanning {DATA_DIR} …")
        diffs, mean_rms = collect_all_pairs(DATA_DIR, SPATIAL_THRESH, TEMPORAL_THRESH_DAYS)
        _save_pkl({"diffs": diffs, "mean_rms": mean_rms}, PICKLE_RAW)

    print(f"\n  Total pairs    : {len(diffs):,}")
    print(f"  |ΔRMS| range   : {diffs.min():.4f} – {diffs.max():.4f} m")
    print(f"  Mean RMS range : {mean_rms.min():.4f} – {mean_rms.max():.4f} m")
    print("\n  mean_RMS percentiles:")
    for pct in [50, 75, 90, 95, 99, 100]:
        print(f"    {pct:>3}th: {np.percentile(mean_rms, pct):.4f} m")

    # 2. dense cap 
    dense_cap  = float(np.percentile(mean_rms, WINDOW_DENSE_PCTL))
    dense_mask = mean_rms <= dense_cap
    print(f"\n  Dense cap ({WINDOW_DENSE_PCTL}th pctl) : {dense_cap:.4f} m  ({dense_mask.sum():,} pairs)")

    # 3. smooth-region bins (fine)
    smooth_mask = mean_rms <= RMS_SPLIT_M
    print(f"\n  Smooth region (RMS ≤ {RMS_SPLIT_M} m): {smooth_mask.sum():,} pairs")
    centers_s, sigmas_s, counts_s = bin_sigma68(mean_rms[smooth_mask], diffs[smooth_mask], BIN_WIDTH_SMOOTH_M, MIN_PAIRS_BIN, RMS_SPLIT_M)
    print(f"  Smooth bins  : {len(centers_s)}  (width={BIN_WIDTH_SMOOTH_M} m, min {MIN_PAIRS_BIN} pairs)")
    print(f"  Pairs/bin    : min={counts_s.min()}  med={int(np.median(counts_s))}  max={counts_s.max()}")

    # 4. full-range bins (coarser)
    centers_f, sigmas_f, counts_f = bin_sigma68(mean_rms[dense_mask], diffs[dense_mask], BIN_WIDTH_FULL_M, MIN_PAIRS_BIN, dense_cap)
    print(f"\n  Full bins    : {len(centers_f)}  (width={BIN_WIDTH_FULL_M} m, min {MIN_PAIRS_BIN} pairs)")
    print(f"  Pairs/bin    : min={counts_f.min()}  med={int(np.median(counts_f))}  max={counts_f.max()}")

    # 5. two fits 
    fit_smooth = fit_linear(centers_s, sigmas_s, counts_s, label=f"smooth (RMS<{RMS_SPLIT_M}m)", force_zero_intercept=False)
    fit_full   = fit_linear(centers_f, sigmas_f, counts_f, label="full range", force_zero_intercept=False)

    # 6. MDC tables 
    print_mdc_table(fit_smooth, mean_rms[smooth_mask].min(), RMS_SPLIT_M, label="smooth")
    print_mdc_table(fit_full, mean_rms[dense_mask].min(),dense_cap, label="full")

    # 7. plot
    plot_two_panel(centers_s, sigmas_s, counts_s, fit_smooth, centers_f, sigmas_f, counts_f, fit_full, dense_cap, OUTPUT_DIR)

    return fit_smooth, fit_full


if __name__ == "__main__":
    fit_smooth, fit_full = run(force_recompute=False)