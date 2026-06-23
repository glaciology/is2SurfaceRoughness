"""
I use this script on server-side to process multiple files (.geojson files that define a bounding box). Each file is parallelized and processed. 
"""

from get_icesat_roughness import run_parallel_granule_processing
from sliderule import sliderule
import glob
from pathlib import Path
import os

# Science Settings
detrend_length           = 200 # meters
time_start               = '2018-10-10'
time_end                 = '2026-01-10'
filter_length            = detrend_length      # min photons; adaptive cnf handles density
moving_filter_bandwidth  = 10 # meters
gaussian_kernal_bandwidth = 2 # meters

# Parallelization Settings
num_workers = 12 # number of cpus to use

if __name__ == "__main__":
    geoson_folder = 'divided_greenland/*.geojson'
    files         = sorted(glob.glob(geoson_folder))

    print(f"******************************************************")
    print(f"Processing ALL of GREENLAND: {len(files)} files.")
    print(f"******************************************************")

    # if any region files have already been completed, enter them here
    completed_regions = {}#{"region_000", "region_001", "region_002", "region_003", "region_004", "region_005", "region_006", "region_007"}

    for geoSON in files:
        region_name = Path(geoSON).stem

        if region_name in completed_regions:
            print(f"Skipping {region_name} (already completed)")
            continue

        bounding_box = sliderule.toregion(geoSON, cellsize=0.02)
        name         = 'output/' + region_name

        print(f"\n****** PROCESSING >>>>>>>> {region_name}  workers: {num_workers}")

        run_time, crashed_workers, segment_count, success = run_parallel_granule_processing(name, detrend_length, time_start, time_end, bounding_box, filter_length, moving_filter_bandwidth, gaussian_kernal_bandwidth, num_workers)

        print(f"Success: {success}  |  {segment_count:,} segments  |  {run_time/60:.1f} min  |  {region_name}")

        # Clean up stale temp files -- issue on Linux + sliderule?
        cleaned = 0
        for f in glob.glob("/tmp/tmp*"):
            try:
                if os.path.isfile(f) and os.stat(f).st_uid == os.getuid():
                    os.unlink(f)
                    cleaned += 1
            except Exception:
                pass
        if cleaned:
            print(f"Cleaned {cleaned} stale temp files from /tmp")