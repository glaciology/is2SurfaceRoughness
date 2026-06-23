"""
script: roughness_pipeline.py
author: Derek Pickell
purpose: get ICESat-2 surface roughness using algorithm described in manuscript. 
run: use this script to derive roughness for a certain single region (defined by geojson), or 
    use run_Greenland for all Greenland (need all geojson files). See main() entrypoint
    for required inputs, such as filepath and parameters. 

Outputs:
    spot_num, time, lat_centroid, lon_centroid,
    RMS (200 m linear detrend),
    rms_sub_median (median of 4 × 50 m linear detrend subsegments),
    n_valid_subsegments,
    semivariogram_range,
    azimuth (integer degrees),
    mean_surface,
    cnf_used,
    n_photons_filtered,
    source_granule

Some aspects of this algorithm were inspired from Van Tiggelen et al. 2021 https://doi.org/10.5194/tc-15-2601-2021, 
including the adaptive photon filter and the upper/lower moving median filter thresholds.
"""

import os
import pandas as pd
from pyproj import Geod
import time
import gstools as gs
from pathlib import Path
import numpy as np
from sliderule import sliderule, icesat2, earthdata
import multiprocessing as mp
import matplotlib.pyplot as plt

geod = Geod(ellps="WGS84")

#########################################
###### GLOBAL SETTINGS ##################
#########################################
ADAPTIVE_CNF       = True # uses photons of varying levels of confidence, if not enough high confidence photons
NOMINAL_SPACING_M  = 0.7 # spacing for photons along-track
CNF_LEVELS         = [4, 3, 2, 1] # see sliderule documentation for confidence level defitions
MIN_SUBSEG_POINTS  = 30   # minimum resampled points per 50 m subsegment

##### ROUGHNESS EXTRACTION FUNCTIONS #######
def moving_filter(x, y, window_size=10):
    """
    Filter data that are outliers using uneven sampling (see threshold_1 and threshold_2)
    x: positions along track
    y: photon heights
    window_size in meters
    IMPORTANT: THRESHOLD_1 and THRESHOLD_2 are FIXED... SEE MANUSCRIPT. 
    """
    filtered_y = []
    filtered_x = []

    ### IMPORTANT: remove more photons below than above
    threshold_1 = 1/0.6745 # low photons
    threshold_2 = 2/0.6745 # high photons
    
    # Convert x and y to numpy arrays for convenience
    x = np.array(x)
    y = np.array(y)
    
    # Start moving window based on x values
    min_x, max_x = np.min(x), np.max(x)
    current_window_start = min_x
    
    while current_window_start < max_x:
        # Define the window range
        window_end = current_window_start + window_size
        
        # Find indices of x values within the window
        in_window = (x >= current_window_start) & (x < window_end)
        
        if np.sum(in_window) > 0:
            # Get y values within the window
            y_window = y[in_window]
            x_window = x[in_window]
            
            # Calculate the median of the window
            median_y = np.median(y_window)
            
            # Calculate the Median Absolute Deviation (MAD) of all points in window
            mad = np.median(np.abs(y_window - median_y))
            
            # Use the MAD to filter out outliers
            if mad == 0:
                non_outliers = y_window == median_y # handle when MAD == 0
            else:
                lower_bound = median_y - threshold_1 * mad
                upper_bound = median_y + threshold_2 * mad
                non_outliers = (y_window >= lower_bound) & (y_window <= upper_bound)

            # Keep only y values that are within the threshold
            filtered_y.extend( y_window[non_outliers])
            filtered_x.extend(x_window[non_outliers])
        
        # Move to the next window
        current_window_start += window_size
    filtered_x, filtered_y = np.array(filtered_x), np.array(filtered_y)
    
    return filtered_x, filtered_y

def gaussian_kernel(x, gaussian_bandwidth=2):
    """ Gaussian kernel function. """
    return np.exp(-0.5 * (x / gaussian_bandwidth)**2) / (gaussian_bandwidth * np.sqrt(2 * np.pi))

def fixed_bandwidth_smoothing(x, y, gaussian_bandwidth):
    """
    Applies kernel smoothing with a fixed bandwidth window on non-evenly spaced data.
    
    Parameters:
    - x: array-like, independent variable (not necessarily evenly spaced)
    - y: array-like, dependent variable
    - guassian_bandwidth: float, the fixed bandwidth window size (in meters)
    
    Returns:
    - smoothed_y: array, the smoothed y-values based on kernel smoothing with fixed bandwidth
    """
    smoothed_y = np.zeros_like(y)
    
    # Loop over each x point to compute a smoothed value
    for i in range(len(x)):
        # Calculate distances from the current point
        distances = np.abs(x - x[i])
        
        # Only consider points within the bandwidth window (~10 meters)
        in_bandwidth = distances <= gaussian_bandwidth
        
        # Compute weights using a Gaussian kernel with fixed bandwidth
        weights = gaussian_kernel(distances[in_bandwidth], gaussian_bandwidth)
        
        # Perform weighted average for smoothing
        smoothed_y[i] = np.sum(weights * y[in_bandwidth]) / np.sum(weights)
    
    return smoothed_y

def fit_semivariogram(x, y, n_lags=20):
    """
    I think using a 1D Guassian model, but this can be changed to Exponential
    Returns (range, valid_flag).
    valid_flag is False if fit is outside physical bounds or fails.
    """
    if len(y) < 10:
        return np.nan, False

    max_dist  = x.max() - x.min()
    lag_edges = np.linspace(0.1, max_dist / 3, n_lags + 1)

    bin_center, gamma = gs.vario_estimate((x,), y, lag_edges)
    if len(bin_center) < 3:
        return np.nan, False

    model = gs.Gaussian(dim=1, var=float(np.nanmax(gamma)), len_scale=max_dist / 4) # or gs.Exponential... 
    try:
        model.fit_variogram(bin_center, gamma)
    except Exception:
        return np.nan, False

    corr_length = model.len_scale

    # physical plausibility check: must be between 1 m and half segment length
    valid = (1.0 <= corr_length <= max_dist / 2)

    return corr_length, valid

def compute_subsegment_rms(x_surface, y_surface, n_subsegments=4, min_points=MIN_SUBSEG_POINTS):
    """
    Split the already-computed 200 m surface into n_subsegments equal
    positional bins. Fit a separate linear detrend to each bin and compute
    RMS of residuals. Returns median RMS across valid subsegments and count of valid subsegments.

    Uses the pre-computed surface.
    """
    x_min, x_max  = x_surface.min(), x_surface.max()
    subseg_length = (x_max - x_min) / n_subsegments

    rms_list = []

    for k in range(n_subsegments):
        lo = x_min + k * subseg_length
        hi = lo + subseg_length
        mask = (x_surface >= lo) & (x_surface < hi)

        xs = x_surface[mask]
        ys = y_surface[mask]

        if len(xs) < min_points:
            continue

        slope_s, intercept_s = np.polyfit(xs, ys, 1)
        detrended_s = ys - (slope_s * xs + intercept_s)
        rms_list.append(float(np.std(detrended_s)))

    n_valid = len(rms_list)
    if n_valid == 0:
        return np.nan, 0

    return float(np.median(rms_list)), n_valid

def roughness_algorithm(x, y, moving_filter_bandwidth=10, gaussian_bandwidth=2, plot=False):
    """
    Apply (1) moving filter, (2) Gaussian smoothing, (3) detrend, (4) compute RMS roughness.
    
    Parameters:
    - x: 1D array of along-track positions
    - y: 1D array of photon heights
    - moving_filter_bandwidth: float, bandwidth for smoothing
    
    Returns: 
    - x_resampled: 1D array, resampled positions (1m spacing)
    - detrended_y: 1D array, detrended heights
    - rms: float, RMS roughness + extras
    """
    # 1. Moving filter
    x_filt, y_filt = moving_filter(x, y, window_size=moving_filter_bandwidth)
    if len(x_filt) < 10: # hard-coded value here
        return None

    # 2. Gaussian surface extraction
    y_smooth  = fixed_bandwidth_smoothing(x_filt, y_filt, gaussian_bandwidth)
    x_surface = np.arange(x_filt.min(), x_filt.max(), 0.5)
    y_surface = np.interp(x_surface, x_filt, y_smooth)

    if len(x_surface) < 10: # hard-coded value here
        return None

    # 3. Full 200 m linear detrend and RMS calculation
    slope, intercept = np.polyfit(x_surface, y_surface, 1)
    detrended = y_surface - (slope * x_surface + intercept)
    rms = float(np.std(detrended))

    # 4. Subsegment RMS (4 × 50 m, reusing x_surface / y_surface)
    rms_sub_median, n_valid_subsegments = compute_subsegment_rms(x_surface, y_surface)

    # 5. Semivariogram on full 200 m detrended surface
    semivariogram_range, semivariogram_valid = fit_semivariogram(x_surface, detrended)
    
    # example plot
    if plot:
        # find filtered-out photons
        x_rejected = x[~np.isin(x, x_filt)]
        y_rejected = y[~np.isin(x, x_filt)]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.scatter(x, y, s=4, color="lightgray", label=f"All photons (n={len(x)})", zorder=1)
        ax.scatter(x_rejected, y_rejected, s=4, color="salmon", label=f"Rejected (n={len(x_rejected)})", zorder=2)
        ax.scatter(x_filt, y_filt, s=4, color="steelblue", label=f"Filtered (n={len(x_filt)})", zorder=3)
        ax.plot (x_surface, y_surface, color="crimson", lw=2, label="Gaussian surface", zorder=4)
        ax.set_ylabel("Height (m)")
        ax.set_title(f"RMS={rms*100:.2f} cm  |  RMS_sub={rms_sub_median*100:.2f} cm  |  Svario range={semivariogram_range:.1f} m")
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(x_surface, detrended, color="darkorange", lw=1.2, label="Detrended surface")
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_ylabel("Residual (m)")
        ax2.set_xlabel("Along-track distance (m)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.show()
    

    return dict(
        x_filt               = x_filt,
        y_filt               = y_filt,
        x_surface            = x_surface,
        y_surface            = y_surface,
        detrended            = detrended,
        slope                = slope,
        rms                  = rms,
        rms_sub_median       = rms_sub_median,
        n_valid_subsegments  = n_valid_subsegments,
        semivariogram_range  = semivariogram_range,
        semivariogram_valid  = semivariogram_valid,
        n_photons_filtered   = len(x_filt),
    )

# ADAPTIVE CONFIDENCE FILTER, from van Tigglen Paper
def _density_ok(x):
    if len(x) < 2:
        return False
    span = x.max() - x.min()
    if span < 200*.8: ### HARDCODED WARNING!!! Checks that photons span at least 80% of segment length
        return False
    if len(x) < span / NOMINAL_SPACING_M: # checks photon density in that span
        return False
    gaps = np.diff(np.sort(x))

    return gaps.max() <= NOMINAL_SPACING_M*10 # checks no gaps larger than 10*nominal spacing

def roughness_algorithm_adaptive(x_all, y_all, cnf_all, moving_filter_bandwidth=10, gaussian_bandwidth=2, plot=False):
    """
    Adaptive cnf fallback: tries cnf≥4 first, adds lower-confidence
    photons until density threshold is met.
    Returns roughness_algorithm result dict plus cnf_used string.
    """
    cnf_all = np.asarray(cnf_all, dtype=int)
    levels_used = []

    density_achieved = False
    for cnf_level in CNF_LEVELS:
        mask  = cnf_all >= cnf_level
        x_use = x_all[mask]
        y_use = y_all[mask]
        levels_used.append(cnf_level)
        if _density_ok(x_use):
            density_achieved = True
            break

    cnf_str = (str(levels_used[0]) if len(levels_used) == 1 else f"{levels_used[0]}+{levels_used[-1]}")
    
    if not density_achieved:
        return None, cnf_str

    result = roughness_algorithm(x_use, y_use, moving_filter_bandwidth=moving_filter_bandwidth, gaussian_bandwidth=gaussian_bandwidth)
    
    if plot and result is not None:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        cnf_colors = {4: "steelblue", 3: "mediumseagreen", 2: "orange", 1: "salmon"}
        cnf_labels = {4: "CNF 4 (high)", 3: "CNF 3", 2: "CNF 2", 1: "CNF 1 (low)"}

        # photons excluded entirely (below the lowest cnf level used)
        used_mask = cnf_all >= levels_used[-1]
        rejected_mask = ~used_mask

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
        ax = axes[0]

        # rejected photons (not used at all)
        if rejected_mask.sum() > 0:
            ax.scatter(x_all[rejected_mask], y_all[rejected_mask], s=4, color="lightgray", label=f"Excluded (n={rejected_mask.sum()})", zorder=1)

        # used photons coloured by cnf level
        for cnf_val in sorted(cnf_colors.keys()):
            m = used_mask & (cnf_all == cnf_val)
            if m.sum() > 0:
                ax.scatter(x_all[m], y_all[m], s=4, color=cnf_colors[cnf_val], label=f"{cnf_labels[cnf_val]} (n={m.sum()})", zorder=2 + cnf_val)

        # Gaussian surface
        ax.plot(result["x_surface"], result["y_surface"], color="crimson", lw=2, label="Gaussian surface", zorder=7)

        rms = result["rms"]
        rms_sub = result["rms_sub_median"]
        svario = result["semivariogram_range"]
        ax.set_title(f"cnf_used={cnf_str}  |  RMS={rms*100:.2f} cm  |  RMS_sub={rms_sub*100:.2f} cm  |  Svario range={svario:.1f} m")
        ax.set_ylabel("Height (m)")
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # detrended residuals
        ax2 = axes[1]
        ax2.plot(result["x_surface"], result["detrended"], color="darkorange", lw=1.2, label="Detrended surface")
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_ylabel("Residual (m)")
        ax2.set_xlabel("Along-track distance (m)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
    return result, cnf_str

### PROCESS A SINGLE SEGMENT. A segment is a along-track swath of data... 
def process_one_segment(seg_tuple, filter_length, moving_filter_bandwidth, gaussian_bandwidth):
    """
    seg_tuple: (segment, spot, x_vals, y_vals, lat_vals, lon_vals,
                index_time, cnf_vals)
    """
    segment, spot, x_vals, y_vals, lat_vals, lon_vals, \
        index_time, cnf_vals = seg_tuple

    geod_local = Geod(ellps="WGS84")

    if len(x_vals) < filter_length:
        return None

    if ADAPTIVE_CNF and cnf_vals is not None:
        result, cnf_used = roughness_algorithm_adaptive(
            x_vals, y_vals, cnf_vals,
            moving_filter_bandwidth=moving_filter_bandwidth,
            gaussian_bandwidth=gaussian_bandwidth,
        )
    else:
        result   = roughness_algorithm(
            x_vals, y_vals,
            moving_filter_bandwidth=moving_filter_bandwidth,
            gaussian_bandwidth=gaussian_bandwidth,
        )
        cnf_used = "4"

    if result is None:
        return None

    azimuth, _, _ = geod_local.inv(
        lon_vals[0], lat_vals[0], lon_vals[-1], lat_vals[-1])

    return {
        "spot_num":             int(spot),
        "time":                 np.datetime64(index_time),
        "lat_centroid":         float(np.mean(lat_vals)),
        "lon_centroid":         float(np.mean(lon_vals)),
        "RMS":                  result["rms"],
        "rms_sub_median":       result["rms_sub_median"],
        "n_valid_subsegments":  result["n_valid_subsegments"],
        "semivariogram_range":  result["semivariogram_range"],
        "azimuth":              int(round(azimuth % 360)),
        "mean_surface":         float(np.mean(result["y_surface"])),
        "cnf_used":             cnf_used,
        "n_photons_filtered":   result["n_photons_filtered"],
    }

# Helpers
def _cnf_column(gdf):
    for col in ["atl03_cnf", "quality_ph", "cnf"]:
        if col in gdf.columns:
            return col
    return None

def build_segment_tuples(gdf, filter_length):
    cnf_col = _cnf_column(gdf)
    tuples  = []

    for (segment, spot), seg_gdf in gdf.groupby(["segment_id", "spot"]):
        if len(seg_gdf) < filter_length:
            continue
        cnf_vals = (seg_gdf[cnf_col].values.astype(int)
                    if cnf_col is not None
                    else np.full(len(seg_gdf), 4, dtype=int))
        tuples.append((
            segment,
            spot,
            seg_gdf["x_atc"].values,
            seg_gdf["height"].values,
            seg_gdf.geometry.y.values,
            seg_gdf.geometry.x.values,
            seg_gdf.index[0],
            cnf_vals,
        ))
    return tuples

### PARALLELIZATION + PROCESSING
def process_granule_chunk(chunk_id, granules_sublist, name, detrend_length, bounding_box, filter_length, moving_filter_bandwidth, gaussian_bandwidth):

    print(f"[Worker {chunk_id}] Initializing SlideRule", flush=True)
    icesat2.init("slideruleearth.io", verbose=False, max_resources=40000)

    base_parms = {
        "srt":          icesat2.SRT_LAND,
        "cnf":          1,
        "quality_ph":   0,
        "pass_invalid": False,
        "len":          detrend_length,
        "res":          detrend_length,
        "poly":         bounding_box["poly"]
    }

    out_file           = Path(f"{name}_chunk{chunk_id}_segments.csv")
    completed_granules = set()

    if out_file.exists():
        try:
            existing_df = pd.read_csv(out_file)
            if "source_granule" in existing_df.columns:
                completed_granules = set(existing_df["source_granule"].unique())
                print(f"[Worker {chunk_id}] Resuming: "
                      f"{len(completed_granules)}/{len(granules_sublist)} "
                      f"already done", flush=True)
            else:
                print(f"[Worker {chunk_id}] WARNING: no source_granule column "
                      f"— starting fresh", flush=True)
        except Exception as e:
            print(f"[Worker {chunk_id}] WARNING: {e} — starting fresh",
                  flush=True)
    else:
        print(f"[Worker {chunk_id}] No existing file — starting fresh",
              flush=True)

    remaining = len(granules_sublist) - len(completed_granules)
    print(f"[Worker {chunk_id}] {remaining} granules to process", flush=True)

    for i, granule in enumerate(granules_sublist, 1):
        if granule in completed_granules:
            print(f"[Worker {chunk_id}] {i}/{len(granules_sublist)}: "
                  f"SKIP {granule}", flush=True)
            continue

        print(f"[Worker {chunk_id}] {i}/{len(granules_sublist)}: "
              f"{granule}", flush=True)

        gdf = None
        for attempt in range(2):
            try:
                gdf = sliderule.run("atl03x", base_parms, resources=[granule])

                break
            except Exception as e:
                err = str(e)
                if ("File transfer already in progress" in err
                        or "'NoneType' object is not iterable" in err):
                    print(f"[Worker {chunk_id}] Arrow lock attempt "
                          f"{attempt + 1}: {e}", flush=True)
                    icesat2.init("slideruleearth.io", verbose=False,
                                 max_resources=40000)
                    if attempt < 1:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"[Worker {chunk_id}] Failed {granule} "
                              f"[ARROW]", flush=True)
                else:
                    print(f"[Worker {chunk_id}] Failed {granule}: "
                          f"{e}", flush=True)
                    break

        if gdf is None or gdf.empty:
            print(f"[Worker {chunk_id}] {i}: empty/failed — skip", flush=True)
            continue

        gdf["segment_id"]  = (gdf["x_atc"] // detrend_length).astype(int)
        segment_tuples     = build_segment_tuples(gdf, filter_length)

        granule_results = []
        for seg in segment_tuples:
            result = process_one_segment(
                seg, filter_length,
                moving_filter_bandwidth, gaussian_bandwidth,
            )
            if result is not None:
                result["source_granule"] = granule
                granule_results.append(result)

        if granule_results:
            df_granule = pd.DataFrame(granule_results)
            tmp_file   = Path(f"{name}_chunk{chunk_id}_segments.tmp")
            df_granule.to_csv(tmp_file, mode="w", header=True, index=False)

            write_header = not out_file.exists()
            with open(out_file, "a") as f_out, \
                 open(tmp_file,  "r") as f_tmp:
                if not write_header:
                    next(f_tmp)
                f_out.write(f_tmp.read())
            tmp_file.unlink()

            print(f"[Worker {chunk_id}] {i}: "
                  f"wrote {len(granule_results)} segments", flush=True)
        else:
            print(f"[Worker {chunk_id}] {i}: no valid segments", flush=True)

    print(f"[Worker {chunk_id}] DONE", flush=True)

def run_parallel_granule_processing(name, detrend_length, time_start, time_end, bounding_box, filter_length, moving_filter_bandwidth, gaussian_bandwidth, num_workers):
    """Top-level script that parallelizes process_granule_chunk() and extraction pipeline"""

    print(f"\nRun started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Adaptive CNF: {ADAPTIVE_CNF}")
    print(f"Output      : {name}")
    print(f"Workers     : {num_workers}")
    print(f"Segment len : {detrend_length} m")
    print(f"Filter bw   : {moving_filter_bandwidth} m  |  Gaussian bw: {gaussian_bandwidth} m\n")

    t_start = time.time()

    print("Querying CMR...")
    icesat2.init("slideruleearth.io", verbose=False, max_resources=40000)

    granules_list = earthdata.cmr(short_name="ATL03", polygon=bounding_box["poly"], time_start=time_start, time_end=time_end, version="007")

    print(f"Total granules: {len(granules_list)}")
    granule_chunks = np.array_split(granules_list, num_workers)

    processes = []
    for i in range(num_workers):
        p = mp.Process(target=process_granule_chunk, args=(i, list(granule_chunks[i]), name, detrend_length, bounding_box, filter_length, moving_filter_bandwidth, gaussian_bandwidth))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    crashed = [p for p in processes if p.exitcode != 0]
    if crashed:
        print(f"\nWARNING: {len(crashed)} worker(s) crashed:")
        for p in crashed:
            print(f"  PID {p.pid}  exit code: {p.exitcode}")

    print("\nMerging CSVs...")
    dfs, failed = [], []
    for i in range(num_workers):
        chunk_file = Path(f"{name}_chunk{i}_segments.csv")
        try:
            df = pd.read_csv(chunk_file)
            dfs.append(df)
            print(f"  Chunk {i}: {len(df):,} segments")
        except FileNotFoundError:
            print(f"  WARNING: chunk {i} not found")
            failed.append(i)
        except Exception as e:
            print(f"  WARNING: chunk {i} error: {e}")
            failed.append(i)

    if failed:
        print(f"  WARNING: missing chunks: {failed}")

    if not dfs:
        print("ERROR: no chunks loaded — CSV not written.")
        return time.time() - t_start, crashed, 0, False

    final_df = pd.concat(dfs, ignore_index=True)
    out_path = f"{name}_segments_combined.csv"
    final_df.to_csv(out_path, index=False)
    print(f"\nFinal CSV: {out_path}  ({len(final_df):,} segments)")

    run_time = time.time() - t_start
    print(f"Total runtime: {run_time:.1f} s  ({run_time / 60:.1f} min)\n")

    return run_time, crashed, len(final_df), True


if __name__ == "__main__":
    # settings
    region_file    = "/Users/username/Desktop/lakes_ji_test.geojson" # must be geojson
    detrend_length            = 200 # meters
    time_start                = "2018-10-10"
    time_end                  = "2026-01-10"
    moving_filter_bandwidth   = 10 # meters
    gaussian_bandwidth        = 2. # meters, for smoothing photons
    num_workers               = 1  # how many CPUs to use

    ###########
    name = f"output/{Path(region_file).stem}"
    os.makedirs("output", exist_ok=True)

    bounding_box = sliderule.toregion(region_file, cellsize=0.02)

    print(f"Region  : {region_file}")
    print(f"Output  : {name}")

    run_time, crashed_workers, segment_count, success = \
        run_parallel_granule_processing(
            name,
            detrend_length,
            time_start,
            time_end,
            bounding_box,
            detrend_length,         # filter_length = detrend_length
            moving_filter_bandwidth,
            gaussian_bandwidth,
            num_workers,
        )

    print(f"\nSuccess: {success}  |  {segment_count:,} segments  |  {run_time / 60:.1f} min")