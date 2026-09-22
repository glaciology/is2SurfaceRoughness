"""
Validate ICESat-2-derived 200 m segment RMS (surface roughness) against
ArcticDEM DEMs:
Porter, Claire, et al., 2023, "ArcticDEM, Version 4.1", https://doi.org/10.7910/DVN/3VDC4W, Harvard Dataverse, V1, [Last accessed: 31 Aug 2026].

METHOD
------
1. Load all 6 ICESat-2 ground-track KMZs.
2. Build a manifest of every CSV file's spatial/temporal extent. DEM only loads the CSV files that overlap it.
3. For each DEM: filter candidate rows by bbox + |dt| <= MAX_DAYS, snap each row's centroid to the nearest ground track, build a 200 m window of 0.5 m
   samples centered on that centroid, sample the DEM along KMZ ground-tracks, apply the same Gaussian smoothing as the ICESat-2 pipeline, linearly detrend, and
   take the RMS of the residuals.
4. Concatenate all DEMs' results into one CSV (with dem_file/csv_file provenance) and produce ONE combined scatter plot of QC-passed matches, split
   above/below the DEM noise floor (11 cm), with Pearson/Spearman/slope stats printed for both the full dataset and the above-floor subset.

**MAX_DAYS is fixed below the ICESat-2 repeat period, so a given ground location is only compared against a given DEM within one repeat cycle.
**Mosaics: made from data from many years (often much earlier than ICESat-2 mission) so assumption is surface is static. Probably use STRIPS instead. 
**DEM_NOISE_FLOOR_M = 0.11 m comes from the DEM cross-strip repeatability analysis (dem_floor_repeatability.py) -- below that, dem_rms is not
  distinguishable from the DEM's own noise, so any ICESat-2 vs. DEM disagreement there is expected and not attributable to either instrument.
"""

import os
import re
import json
import glob
import time
import zipfile
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates, convolve1d
from scipy import stats as scipy_stats
from pyproj import Transformer
import shapely
from shapely.geometry import LineString, box as shapely_box

KMZ_PATHS = [
    "./DEM/arcticallorbits/Arctic_repeat1_GT1L.kmz",
    "./DEM/arcticallorbits/Arctic_repeat1_GT1R.kmz",
    "./DEM/arcticallorbits/Arctic_repeat1_GT2L.kmz",
    "./DEM/arcticallorbits/Arctic_repeat1_GT2R.kmz",
    "./DEM/arcticallorbits/Arctic_repeat1_GT3L.kmz",
    "./DEM/arcticallorbits/Arctic_repeat1_GT3R.kmz",
] # DOWNLOAD from ICESat-2 project webpage https://icesat-2.gsfc.nasa.gov/science/specs

DEM_DIR = "./DEM/Strips/*/"     # directory (may include a glob wildcard) of ArcticDEM mosaic .tif files from PGC Fridge
CSV_DIR = "./Data"  # directory of region-split ICESat-2 CSVs
OUT_DIR = "./dem_validation_out"  # combined CSV + scatter plot go here

HALF_LEN_M = 100.0          # half-window length (-> 200 m window)
STEP_M = 0.5                # along-track sample spacing
MIN_COVERAGE = 0.9          # min fraction of valid (non-nodata) DEM samples required
TIME_COL = "time"
RMS_COL = "RMS"              # ICESat-2 RMS column in the CSVs
APPLY_GAUSSIAN_SMOOTHING = True
GAUSSIAN_BANDWIDTH_M = 2.0   # must match the upstream IS2 pipeline's smoothing bandwidth
MOSAIC_MODE = False
STRIP_DATE_OVERRIDES = {}          # optional per-file override: {"filename.tif": "2015-09-23"} (strip mode only)

### Probably only want to adjust these:
MAX_DAYS = 90               # kept below the ICESat-2 repeat period -- see module docstring
MAX_CROSS_TRACK_M = 10.0    # flag matches where centroid is this far off the KMZ line
MOSAIC_STRIP_DATE = "2023-04-01"   # label only, not used for filtering/QC when MOSAIC_MODE=True

# determined DEM noise floor (see dem_floor_repeatability.py) -- dem_rms
# below this is not distinguishable from DEM measurement noise
DEM_NOISE_FLOOR_M = 0.20

MANIFEST_CACHE_PATH = os.path.join(OUT_DIR, "csv_manifest_cache.json")
ARCTICDEM_DATE_RE = re.compile(r"(\d{8})")
GT_NAME_RE = re.compile(r"(GT[123][LR])", re.IGNORECASE)

### STYLE ###
derek_colors = {
    "blue": "#3867B1", "black": "#292930", "red": "#BE3445", "purple": "#4a0e82",
    "light_blue": "#1f8db5", "yellow": "#ffe138", "orange": "#B55E1F",
}

def read_csv_fast(path, usecols=None, parse_dates=None):
    """pandas.read_csv via the pyarrow engine when available (much faster on
    large files), falling back to the default engine otherwise."""
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)

### KMZ FILE HANDLING ###
def extract_kmz(kmz_path, workdir):
    with zipfile.ZipFile(kmz_path, "r") as zf:
        kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise ValueError(f"No .kml found inside {kmz_path}")
        zf.extract(kml_names[0], workdir)
        return os.path.join(workdir, kml_names[0])

def parse_kml_linestrings(kml_path):
    """Return shapely LineStrings (lon/lat) from plain <LineString>."""
    import xml.etree.ElementTree as ET

    root = ET.parse(kml_path).getroot()
    lines = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "LineString":
            coord_elem = next((c for c in elem.iter() if c.tag.split("}")[-1] == "coordinates"), None)
            if coord_elem is not None and coord_elem.text:
                coords = []
                for triplet in coord_elem.text.strip().split():
                    parts = triplet.split(",")
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
                if len(coords) >= 2:
                    lines.append(LineString(coords))

    if not lines:
        raise ValueError(f"No LineString:Track geometry found in {kml_path}")

    return lines

def gt_name_from_path(path):
    m = GT_NAME_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else os.path.splitext(os.path.basename(path))[0]

def load_raw_tracks(kmz_paths):
    """Load all ground-track lines (lon/lat), tagged with their ground-track name."""
    lines, names = [], []
    for path in kmz_paths:
        gt_name = gt_name_from_path(path)
        try:
            kml_path = path
            if path.lower().endswith(".kmz"):
                kml_path = extract_kmz(path, tempfile.mkdtemp())
            for line in parse_kml_linestrings(kml_path):
                lines.append(line)
                names.append(gt_name)
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")
    return lines, names

def reproject_lines(lines, target_crs):
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    out = []
    for line in lines:
        lon, lat = zip(*line.coords)
        x, y = transformer.transform(lon, lat)
        out.append(LineString(zip(x, y)))
    return out
##########################
### DEM PROCESSING ###
def assign_tracks(xs, ys, track_lines, chunk_size=5000):
    """Nearest ground-track line for every point, and its cross-track distance."""
    n = len(xs)
    idx = np.empty(n, dtype=int)
    cross_track = np.empty(n, dtype=float)
    t0 = time.time()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pts = shapely.points(xs[start:end], ys[start:end])
        dists = np.stack([shapely.distance(pts, line) for line in track_lines], axis=1)
        chunk_idx = np.argmin(dists, axis=1)
        idx[start:end] = chunk_idx
        cross_track[start:end] = dists[np.arange(end - start), chunk_idx]
        print(f"      assign_tracks: {end}/{n} points ({time.time() - t0:.1f}s elapsed)", flush=True)

    return idx, cross_track

def line_bounds_array(lines):
    return np.array([line.bounds for line in lines])

def filter_lines_near_bounds(lines, names, bounds_arr, target_bounds, buffer_m):
    minx, miny, maxx, maxy = target_bounds
    minx, miny, maxx, maxy = minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m

    bbox_keep = ~(
        (bounds_arr[:, 2] < minx) | (bounds_arr[:, 0] > maxx) |
        (bounds_arr[:, 3] < miny) | (bounds_arr[:, 1] > maxy)
    )
    idxs = np.where(bbox_keep)[0]
    if idxs.size == 0:
        return [], []

    candidate_lines = np.array([lines[i] for i in idxs], dtype=object)
    query_box = shapely_box(minx, miny, maxx, maxy)
    exact_keep = shapely.intersects(candidate_lines, query_box)
    final_idxs = idxs[exact_keep]
    return [lines[i] for i in final_idxs], [names[i] for i in final_idxs]

def compute_along_track(xs, ys, idx, track_lines):
    pts = shapely.points(xs, ys)
    s = np.empty(len(xs))
    for i, line in enumerate(track_lines):
        m = idx == i
        if m.any():
            s[m] = shapely.line_locate_point(line, pts[m])
    return s

def sample_grid_coords(s, idx, track_lines, offsets):
    """(n, len(offsets)) grid of DEM-CRS coordinates along each point's track,
    centered on s. Points that fall outside the track's length are NaN.
    """
    n, m = len(s), len(offsets)
    xs_grid = np.full((n, m), np.nan)
    ys_grid = np.full((n, m), np.nan)
    valid_grid = np.zeros((n, m), dtype=bool)
    s_grid = s[:, None] + offsets[None, :]

    for i, line in enumerate(track_lines):
        rows = np.where(idx == i)[0]
        if rows.size == 0:
            continue
        sg = s_grid[rows]
        valid = (sg >= 0) & (sg <= line.length)
        sg_clipped = np.clip(sg, 0, line.length)
        pts = shapely.line_interpolate_point(line, sg_clipped.ravel())
        coords = shapely.get_coordinates(pts).reshape(rows.size, m, 2)
        xs_grid[rows] = np.where(valid, coords[..., 0], np.nan)
        ys_grid[rows] = np.where(valid, coords[..., 1], np.nan)
        valid_grid[rows] = valid

    return xs_grid, ys_grid, valid_grid

def sample_dem_bulk(dem_ds, xs_grid, ys_grid, valid_grid):
    """Bilinear-sample the DEM at every grid point in one windowed read."""
    if not valid_grid.any():
        return np.full(xs_grid.shape, np.nan)

    pad = STEP_M * 2
    minx, maxx = np.min(xs_grid[valid_grid]), np.max(xs_grid[valid_grid])
    miny, maxy = np.min(ys_grid[valid_grid]), np.max(ys_grid[valid_grid])
    win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, transform=dem_ds.transform)

    nodata = dem_ds.nodata
    fill = nodata if nodata is not None else np.nan
    band = dem_ds.read(1, window=win, boundless=True, fill_value=fill).astype(float)
    if nodata is not None:
        band[band == nodata] = np.nan

    t = dem_ds.window_transform(win)
    det = t.a * t.e - t.b * t.d
    dx, dy = xs_grid - t.c, ys_grid - t.f
    cols = (t.e * dx - t.b * dy) / det
    rows = (-t.d * dx + t.a * dy) / det

    safe_rows = np.where(valid_grid, rows, 0.0)
    safe_cols = np.where(valid_grid, cols, 0.0)
    elev = map_coordinates(band, [safe_rows.ravel(), safe_cols.ravel()],
                            order=1, mode="constant", cval=np.nan)
    elev = elev.reshape(xs_grid.shape)
    elev[~valid_grid] = np.nan
    return elev

def gaussian_smooth_bulk(elev, step, bandwidth):
    """NaN-aware Gaussian smoothing along axis=1, matching the IS2 pipeline's
    fixed_bandwidth_smoothing() (truncated at one bandwidth, per-row basis)."""
    half_n = int(np.floor(bandwidth / step))
    k_offsets = np.arange(-half_n, half_n + 1) * step
    weights = np.exp(-0.5 * (k_offsets / bandwidth) ** 2)

    mask = np.isfinite(elev)
    vals = np.where(mask, elev, 0.0)
    num = convolve1d(vals, weights, axis=1, mode="constant", cval=0.0)
    den = convolve1d(mask.astype(float), weights, axis=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)

def detrend_rms_bulk(elev, offsets):
    """Vectorized per-row linear detrend + RMS of residuals."""
    mask = np.isfinite(elev)
    n = mask.sum(axis=1).astype(float)
    y = np.where(mask, elev, 0.0)
    x = offsets[None, :]

    Sx = (x * mask).sum(axis=1)
    Sxx = ((x ** 2) * mask).sum(axis=1)
    Sy = y.sum(axis=1)
    Sxy = (x * y).sum(axis=1)
    denom = n * Sxx - Sx ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * Sxy - Sx * Sy) / denom
        intercept = (Sy - slope * Sx) / n
        trend = slope[:, None] * x + intercept[:, None]
        resid = np.where(mask, elev - trend, np.nan)
        rms = np.sqrt(np.nanmean(resid ** 2, axis=1))

    coverage = n / offsets.shape[0]
    rms[n < 10] = np.nan
    return rms, n, coverage

def _get_col(df, col, default=np.nan):
    if col in df.columns:
        return df[col].to_numpy()
    return np.full(len(df), default)

def process_dem(dem_path, raw_lines, raw_names, track_cache, bounds_cache, csv_manifest):
    try:
        dem_ds = rasterio.open(dem_path)
    except Exception as e:
        print(f"  Skipping {os.path.basename(dem_path)}: cannot open ({e})")
        return None

    strip_date = parse_strip_date(dem_path)
    if strip_date is None:
        print(f"  Skipping {os.path.basename(dem_path)}: cannot parse strip date")
        dem_ds.close()
        return None

    crs_key = dem_ds.crs.to_string()
    if crs_key not in track_cache:
        track_cache[crs_key] = reproject_lines(raw_lines, dem_ds.crs)
        bounds_cache[crs_key] = line_bounds_array(track_cache[crs_key])
    all_track_lines = track_cache[crs_key]
    all_bounds = bounds_cache[crs_key]

    bounds = dem_ds.bounds
    buffer_m = HALF_LEN_M + MAX_CROSS_TRACK_M
    track_lines, track_names = filter_lines_near_bounds(
        all_track_lines, raw_names, all_bounds,
        (bounds.left, bounds.bottom, bounds.right, bounds.top), buffer_m,
    )
    print(f"  {len(track_lines)}/{len(all_track_lines)} ground-track pieces near this DEM tile", flush=True)
    if not track_lines:
        print(f"  Skipping {os.path.basename(dem_path)}: no ground tracks near this DEM")
        dem_ds.close()
        return None

    # Candidate CSVs: bbox (in lon/lat, via DEM corner reprojection), plus a
    # date-overlap check: skipped in MOSAIC_MODE since mosaics have no
    # single meaningful acquisition date (see module docstring).
    to4326 = Transformer.from_crs(dem_ds.crs, "EPSG:4326", always_xy=True)
    lons, lats = to4326.transform([bounds.left, bounds.right], [bounds.bottom, bounds.top])
    buf_deg = (HALF_LEN_M + MAX_CROSS_TRACK_M) / 111000 * 3
    lon_min, lon_max = min(lons) - buf_deg, max(lons) + buf_deg
    lat_min, lat_max = min(lats) - buf_deg, max(lats) + buf_deg

    def _bbox_ok(m):
        return (m["lon_max"] >= lon_min and m["lon_min"] <= lon_max and
                m["lat_max"] >= lat_min and m["lat_min"] <= lat_max)

    if MOSAIC_MODE:
        candidates = [m["path"] for m in csv_manifest if _bbox_ok(m)]
    else:
        t_min, t_max = strip_date - pd.Timedelta(days=MAX_DAYS), strip_date + pd.Timedelta(days=MAX_DAYS)
        candidates = [m["path"] for m in csv_manifest if
                      _bbox_ok(m) and m["t_max"] >= t_min and m["t_min"] <= t_max]

    if not candidates:
        print(f"  {os.path.basename(dem_path)}: no candidate CSVs")
        dem_ds.close()
        return None

    frames = []
    for p in candidates:
        d = read_csv_fast(p, parse_dates=[TIME_COL])
        d["csv_file"] = os.path.basename(p)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    transformer = Transformer.from_crs("EPSG:4326", dem_ds.crs, always_xy=True)
    xs, ys = transformer.transform(df["lon_centroid"].to_numpy(), df["lat_centroid"].to_numpy())
    delta_days = ((df[TIME_COL] - strip_date).abs().dt.total_seconds() / 86400.0).to_numpy()
    spatial_mask = (
        (xs >= bounds.left - buffer_m) & (xs <= bounds.right + buffer_m) &
        (ys >= bounds.bottom - buffer_m) & (ys <= bounds.top + buffer_m)
    )
    mask = spatial_mask if MOSAIC_MODE else (spatial_mask & (delta_days <= MAX_DAYS))
    if not mask.any():
        print(f"  {os.path.basename(dem_path)}: no rows pass spatial/temporal filter")
        dem_ds.close()
        return None

    df = df.loc[mask].reset_index(drop=True)
    xs, ys, delta_days = xs[mask], ys[mask], delta_days[mask]
    print(f"  {os.path.basename(dem_path)}: {len(df)} candidate segments", flush=True)

    t0 = time.time()
    idx, cross_track = assign_tracks(xs, ys, track_lines)
    print(f"    assign_tracks:       {time.time() - t0:6.1f}s", flush=True)

    t0 = time.time()
    s = compute_along_track(xs, ys, idx, track_lines)
    print(f"    compute_along_track: {time.time() - t0:6.1f}s", flush=True)

    t0 = time.time()
    offsets = np.arange(-HALF_LEN_M, HALF_LEN_M + 1e-9, STEP_M)
    xs_grid, ys_grid, valid_grid = sample_grid_coords(s, idx, track_lines, offsets)
    print(f"    sample_grid_coords:  {time.time() - t0:6.1f}s", flush=True)

    t0 = time.time()
    elev = sample_dem_bulk(dem_ds, xs_grid, ys_grid, valid_grid)
    dem_ds.close()
    print(f"    sample_dem_bulk:     {time.time() - t0:6.1f}s", flush=True)

    t0 = time.time()
    elev_for_fit = gaussian_smooth_bulk(elev, STEP_M, GAUSSIAN_BANDWIDTH_M) if APPLY_GAUSSIAN_SMOOTHING else elev
    print(f"    gaussian_smooth:     {time.time() - t0:6.1f}s", flush=True)

    t0 = time.time()
    rms, n_valid, coverage = detrend_rms_bulk(elev_for_fit, offsets)
    print(f"    detrend_rms:         {time.time() - t0:6.1f}s", flush=True)

    passes_qc = (
        np.isfinite(rms) &
        (coverage >= MIN_COVERAGE) &
        (cross_track <= MAX_CROSS_TRACK_M)
    )
    if not MOSAIC_MODE:
        passes_qc &= (delta_days <= MAX_DAYS)

    return pd.DataFrame({
        "dem_file": os.path.basename(dem_path),
        "csv_file": df["csv_file"].to_numpy(),
        "lat_centroid": df["lat_centroid"].to_numpy(),
        "lon_centroid": df["lon_centroid"].to_numpy(),
        "time": df[TIME_COL].to_numpy(),
        "ground_track": np.array(track_names)[idx],
        "is2_rms": df[RMS_COL].to_numpy(),
        "dem_rms": rms,
        "dem_coverage_frac": coverage,
        "cross_track_offset_m": cross_track,
        "delta_t_days": delta_days,
        "strip_date": strip_date.date().isoformat(),
        "passes_qc": passes_qc,
        # IS2-side quality columns, carried through unused from the CSV so
        # they can be filtered on later without re-running DEM sampling
        # see roughness_pipeline.py for what each one means:
        "spot_num": _get_col(df, "spot_num"),
        "cnf_used": _get_col(df, "cnf_used", default=""),
        "n_photons_filtered": _get_col(df, "n_photons_filtered"),
        "n_valid_subsegments": _get_col(df, "n_valid_subsegments"),
        "rms_sub_median": _get_col(df, "rms_sub_median"),
    })

##########################
def build_csv_manifest(csv_dir, cache_path=MANIFEST_CACHE_PATH):
    """Spatial/temporal extent per CSV file, cached to disk keyed on file
    size + mtime so unchanged files aren't re-scanned on repeat runs."""
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = {e["path"]: e for e in json.load(f)}

    manifest, changed = [], False
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        stat = os.stat(path)
        entry = cached.get(path)
        if entry and entry["mtime"] == stat.st_mtime and entry["size"] == stat.st_size:
            manifest.append(entry)
            continue

        try:
            d = read_csv_fast(path, usecols=["lat_centroid", "lon_centroid", TIME_COL],
                               parse_dates=[TIME_COL])
        except ValueError:
            print(f"  Skipping {path}: missing expected columns")
            continue
        if d.empty:
            continue

        manifest.append({
            "path": path, "mtime": stat.st_mtime, "size": stat.st_size,
            "lat_min": float(d["lat_centroid"].min()), "lat_max": float(d["lat_centroid"].max()),
            "lon_min": float(d["lon_centroid"].min()), "lon_max": float(d["lon_centroid"].max()),
            "t_min": d[TIME_COL].min().isoformat(), "t_max": d[TIME_COL].max().isoformat(),
        })
        changed = True

    if changed or not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(manifest, f)

    for m in manifest:
        m["t_min"] = pd.Timestamp(m["t_min"])
        m["t_max"] = pd.Timestamp(m["t_max"])
    return manifest

def parse_strip_date(dem_path):
    if MOSAIC_MODE:
        return pd.Timestamp(MOSAIC_STRIP_DATE)
    override = STRIP_DATE_OVERRIDES.get(os.path.basename(dem_path))
    if override:
        return pd.Timestamp(override)
    m = ARCTICDEM_DATE_RE.search(os.path.basename(dem_path))
    if not m:
        return None
    return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d"))

### FILTERING ###
def _add_spatial_bins(df, bin_km):
    """Approximate bin_km x bin_km grid cell id from lat/lon, using per-row
    cos(lat) correction so longitude bins stay roughly bin_km wide even at
    high latitude."""
    lat_bin_deg = bin_km / 111.0
    lon_bin_deg = bin_km / (111.0 * np.cos(np.radians(df["lat_centroid"])))
    lat_idx = np.floor(df["lat_centroid"] / lat_bin_deg).astype(int)
    lon_idx = np.floor(df["lon_centroid"] / lon_bin_deg).astype(int)
    return lat_idx.astype(str) + "_" + lon_idx.astype(str)

def _filtered_segments(csv_path, require_strong_beam=True, require_full_confidence=True,
 min_photons_filtered=200, min_valid_subsegments=None, max_subseg_diff=None,
                        mad_thresh=False, outlier_on="is2_rms", outlier_spatial_bin_km=30):
    """Load segment_comparison.csv and apply QC/zero/IS2-quality/outlier
    filtering. Factored out so the single plot below and any future
    tabular analysis use exactly the same filtered rows."""
    df = pd.read_csv(csv_path)
    passed = df[df["passes_qc"]].dropna(subset=["is2_rms", "dem_rms"]).copy()
    passed = passed[(passed["is2_rms"] > 0) & (passed["dem_rms"] > 0)]
    passed = passed[(passed["delta_t_days"] < MAX_DAYS)]

    for label, keep_mask in [
        ("strong beam only", passed["spot_num"].isin([1, 3, 5]) if require_strong_beam else None),
        ("full confidence only", passed["cnf_used"].astype(str) == "4" if require_full_confidence else None),
        (f"n_photons_filtered >= {min_photons_filtered}",
         passed["n_photons_filtered"] >= min_photons_filtered if min_photons_filtered is not None else None),
        (f"n_valid_subsegments >= {min_valid_subsegments}",
         passed["n_valid_subsegments"] >= min_valid_subsegments if min_valid_subsegments is not None else None),
        (f"|is2_rms - rms_sub_median| <= {max_subseg_diff}",
         (passed["is2_rms"] - passed["rms_sub_median"]).abs() <= max_subseg_diff if max_subseg_diff is not None else None),
    ]:
        if keep_mask is None:
            continue
        n_before = len(passed)
        passed = passed[keep_mask.reindex(passed.index, fill_value=False)]
        print(f"IS2 quality filter '{label}': removed {n_before - len(passed)} of {n_before} rows "
              f"({100 * (n_before - len(passed)) / n_before:.1f}%)")

    n_before_mad = len(passed)
    if mad_thresh:
        if outlier_on == "residual":
            series = passed["is2_rms"] - passed["dem_rms"]
        elif outlier_on in ("is2_rms", "dem_rms"):
            series = passed[outlier_on]
        else:
            raise ValueError("outlier_on must be 'residual', 'is2_rms', or 'dem_rms'")

        if outlier_spatial_bin_km:
            grp = _add_spatial_bins(passed, outlier_spatial_bin_km)
            med = series.groupby(grp).transform("median")
            abs_dev = (series - med).abs()
            mad = abs_dev.groupby(grp).transform("median")
        else:
            med = series.median()
            abs_dev = (series - med).abs()
            mad = abs_dev.median()

        robust_std = 1.4826 * mad
        keep = (abs_dev <= mad_thresh * robust_std) | (robust_std == 0)
        passed = passed[keep]

        n_removed = n_before_mad - len(passed)
        scope = f"within {outlier_spatial_bin_km} km spatial bins" if outlier_spatial_bin_km else "globally"
        print(f"MAD filter on '{outlier_on}' ({scope}): removed {n_removed} of "
              f"{n_before_mad} rows ({100 * n_removed / n_before_mad:.1f}%)")

    return passed

def _by_bin_table(passed, bin_col, other_col, bin_edges, min_n_for_corr=30):
    """Bin by bin_col, report median/spread of other_col in each bin"""
    df = passed.copy()
    df["_bin"] = pd.cut(df[bin_col], bins=bin_edges, include_lowest=True)

    def _stat(g):
        n = len(g)
        if n == 0:
            return pd.Series({"n": 0, "median_bin_col": np.nan, "median_other": np.nan, "sigma68_other": np.nan})
        return pd.Series({
            "n": n,
            "median_bin_col": g[bin_col].median(),
            "median_other": g[other_col].median(),
            "sigma68_other": float(np.percentile((g[other_col] - g[other_col].median()).abs(), 68.27)),
        })

    return df.groupby("_bin", observed=False).apply(_stat)

def _regression_stats(x, y, label=""):
    """Pearson r, Spearman rho, and a weighted-least-squares slope/intercept
    (via scipy.stats.linregress) for one (x, y) subset. Printed so the
    slope can be checked against 1:1 directly, alongside its standard error."""
    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
    spearman_rho, spearman_p = scipy_stats.spearmanr(x, y)
    lr = scipy_stats.linregress(x, y)
    bias = float(np.mean(x - y))
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))

    print(f"\n  [{label}]  n={len(x)}")
    print(f"    bias (is2 - dem)  = {bias:+.4f} m")
    print(f"    RMSE              = {rmse:.4f} m")
    print(f"    Pearson  r        = {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"    Spearman rho      = {spearman_rho:.4f}  (p={spearman_p:.2e})")
    print(f"    slope             = {lr.slope:.4f} ± {lr.stderr:.4f}   "
          f"intercept = {lr.intercept:+.4f} ± {lr.intercept_stderr:.4f}")
    print(f"    deviation from 1:1 slope: {lr.slope - 1.0:+.4f}")

    return {"label": label, "n": len(x), "bias": bias, "rmse": rmse,
            "pearson_r": pearson_r, "spearman_rho": spearman_rho,
            "slope": lr.slope, "slope_se": lr.stderr,
            "intercept": lr.intercept, "intercept_se": lr.intercept_stderr}

### OUTPUT PLOT ###
def _draw_scatter_layer(ax, passed, below_mask, by_bin, noise_floor_m, lim):
    ax.plot([0, lim], [0, lim], "--", color=derek_colors["black"], lw=1.2, zorder=1)
    ax.axhline(noise_floor_m, color=derek_colors["red"], lw=1.3, ls=":", zorder=1)

    ax.scatter(passed.loc[below_mask, "is2_rms"], passed.loc[below_mask, "dem_rms"],
               s=16, color=derek_colors["light_blue"], alpha=0.12, edgecolors="none",
               rasterized=True, zorder=2)
    ax.scatter(passed.loc[~below_mask, "is2_rms"], passed.loc[~below_mask, "dem_rms"],
               s=16, color=derek_colors["blue"], alpha=0.12, edgecolors="none",
               rasterized=True, zorder=2)

    ax.errorbar(by_bin["median_bin_col"], by_bin["median_other"], yerr=by_bin["sigma68_other"],
                fmt="o", ms=6, color=derek_colors["orange"], ecolor=derek_colors["black"],
                elinewidth=1, capsize=2, zorder=5)

    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(derek_colors["black"])
    ax.tick_params(colors=derek_colors["black"])

def plot_and_score(csv_path, out_path, noise_floor_m=DEM_NOISE_FLOOR_M, inset_max_m=0.50,
                    mad_thresh=False, outlier_on="is2_rms", outlier_spatial_bin_km=30,
                    require_strong_beam=True, require_full_confidence=True,
                    min_photons_filtered=500, min_valid_subsegments=None, max_subseg_diff=None,
                    rms_bin_edges=(0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12,
                                   0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60, 0.85, 1.2, 1.5, 2, 2.5, 3, 3.5, 5, 10, np.inf)):
    """
    Load segment_comparison.csv, filter to QC-passed rows, and produce ONE
    scatter plot of ICESat-2 vs. DEM RMS: uniform color, split into
    below-noise-floor and above-noise-floor groups by dem_rms vs.
    noise_floor_m, with median-binned points (± sigma68) overlaid, plus a
    zoomed inset (0 to inset_max_m on both axes) so the low-roughness
    region -- where the noise floor and the correlation breakdown live --
    isn't squashed into a corner of the full-range plot.

    Prints, to the console:
      - which dem_file(s) and csv_file(s) went into this analysis
      - Pearson r / Spearman rho / slope-vs-1:1 for ALL data
      - the same stats for only rows with dem_rms > noise_floor_m

    See _filtered_segments() docstring (below the argument list here) for
    what each IS2-side quality filter does; mad_thresh/outlier_on/
    outlier_spatial_bin_km control optional MAD-based outlier removal.
    """
    passed = _filtered_segments(
        csv_path, require_strong_beam, require_full_confidence, min_photons_filtered,
        min_valid_subsegments, max_subseg_diff, mad_thresh, outlier_on, outlier_spatial_bin_km,
    )
    if passed.empty:
        print("No valid rows remain after filtering.")
        return None

    print(f"\nFiles analyzed:")
    print(f"  DEM files ({passed['dem_file'].nunique()}): {sorted(passed['dem_file'].unique())}")
    print(f"  CSV files ({passed['csv_file'].nunique()}): {sorted(passed['csv_file'].unique())}")

    x_all, y_all = passed["is2_rms"].to_numpy(), passed["dem_rms"].to_numpy()
    stats_all = _regression_stats(x_all, y_all, label="ALL data")

    above = passed[passed["dem_rms"] > noise_floor_m]
    if len(above) >= 2:
        stats_above = _regression_stats(above["is2_rms"].to_numpy(), above["dem_rms"].to_numpy(),
                                          label=f"DEM roughness > {noise_floor_m*100:.0f} cm noise floor")
    else:
        stats_above = None
        print(f"\n  [dem_rms > {noise_floor_m*100:.0f} cm]: too few rows ({len(above)}) to compute stats")

    by_bin = _by_bin_table(passed, "is2_rms", "dem_rms", rms_bin_edges)
    by_bin = by_bin.dropna(subset=["median_bin_col", "median_other"])

    below_mask = passed["dem_rms"] <= noise_floor_m
    lim = max(x_all.max(), y_all.max()) * 1.05

    fig, ax = plt.subplots(figsize=(8, 8.5))
    fig.subplots_adjust(top=0.84, bottom=0.08)

    _draw_scatter_layer(ax, passed, below_mask, by_bin, noise_floor_m, lim)

    ax.xaxis.label.set_color(derek_colors["black"])
    ax.yaxis.label.set_color(derek_colors["black"])
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("ICESat-2 ATL03 Roughness (RMS) (m)")
    ax.set_ylabel("DEM Roughness (RMS) (m)")

    fig.suptitle("ICESat-2 vs. DEM-derived roughness (QC-passed)", fontsize=13,
                 color=derek_colors["black"], y=0.97)

    legend_handles = [
        plt.Line2D([], [], ls="--", color=derek_colors["black"], lw=1.2, alpha=1, label="1:1"),
        plt.Line2D([], [], ls=":", color=derek_colors["red"], lw=1.3, alpha=1,
                   label=f"DEM noise floor = {noise_floor_m*100:.0f} cm"),
        plt.Line2D([], [], marker="o", ls="", color=derek_colors["light_blue"], alpha=1, markersize=9,
                   label=f"DEM roughness \u2264 {noise_floor_m*100:.0f} cm (n={below_mask.sum():,})"),
        plt.Line2D([], [], marker="o", ls="", color=derek_colors["blue"], alpha=1, markersize=9,
                   label=f"DEM roughness > {noise_floor_m*100:.0f} cm (n={(~below_mask).sum():,})"),
        plt.Line2D([], [], marker="o", ls="", color=derek_colors["orange"], alpha=1, markersize=9,
                   label="binned median \u00b1 \u03c368"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.03),
              ncol=2, fontsize=11, facecolor="white", edgecolor="#dddddd",
              labelcolor=derek_colors["black"])

    axins = ax.inset_axes([0.06, 0.55, 0.40, 0.40])
    _draw_scatter_layer(axins, passed, below_mask, by_bin, noise_floor_m, inset_max_m)
    axins.set_xlim(0, inset_max_m)
    axins.set_ylim(0, inset_max_m)
    axins.set_aspect("equal")
    axins.tick_params(labelsize=7)
    axins.set_title(f"0\u2013{inset_max_m*100:.0f} cm", fontsize=8, color=derek_colors["black"])

    rect, connectors = ax.indicate_inset_zoom(axins, edgecolor=derek_colors["black"])
    rect.set_linewidth(0.8)
    rect.set_alpha(0.5)
    for c in connectors:
        if c is None:
            continue
        c.set_linewidth(0.6)
        c.set_linestyle((0, (3, 3)))
        c.set_alpha(0.35)

    stats_txt = (
        f"ALL: n={stats_all['n']}  r={stats_all['pearson_r']:.3f}  "
        f"\u03c1={stats_all['spearman_rho']:.3f}  slope={stats_all['slope']:.3f}\n"
    )
    if stats_above is not None:
        stats_txt += (
            f">{noise_floor_m*100:.0f} cm: n={stats_above['n']}  r={stats_above['pearson_r']:.3f}  "
            f"\u03c1={stats_above['spearman_rho']:.3f}  slope={stats_above['slope']:.3f}"
        )
    ax.text(0.98, 0.02, stats_txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=derek_colors["black"],
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#dddddd", alpha=0.9))

    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nSaved scatter plot to {out_path}")

##########
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading ground-track KMZ files...")
    raw_lines, raw_names = load_raw_tracks(KMZ_PATHS)
    print(f"  {len(raw_lines)} track segment(s) across {len(set(raw_names))} ground tracks")
    if len(raw_lines) > 50:
        print(f"  Note: {len(raw_lines)} individual line pieces -- each DEM filters this "
              f"down to only the pieces near its own tile before any distance search.")

    print("Scanning CSV directory for spatial/temporal extents...")
    csv_manifest = build_csv_manifest(CSV_DIR)
    print(f"  {len(csv_manifest)} CSV file(s) found")

    dem_paths = sorted(glob.glob(os.path.join(DEM_DIR, "*dem.tif")))
    print(f"Found {len(dem_paths)} DEM strip(s)")

    track_cache, bounds_cache, all_results = {}, {}, []
    for dem_path in dem_paths:
        print(f"Processing {os.path.basename(dem_path)}", flush=True)
        t_dem = time.time()
        result = process_dem(dem_path, raw_lines, raw_names, track_cache, bounds_cache, csv_manifest)
        print(f"  -> done in {time.time() - t_dem:.1f}s", flush=True)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No segments matched any DEM. Nothing to save.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "segment_comparison.csv")
    combined.to_csv(out_csv, index=False)
    print(f"Saved {len(combined)} segment comparisons to {out_csv}")
    print(f"  {int(combined['passes_qc'].sum())} passed QC")

if __name__ == "__main__":
    # main() # uncomment to generate csv comparison
    plot_and_score(os.path.join(OUT_DIR, "segment_comparison.csv"), os.path.join(OUT_DIR, "rms_scatter.png"))