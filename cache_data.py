"""Pre-cache the NWB → numpy/pandas conversion so training and visualisation
start in <1 s instead of waiting for nlb_tools to re-parse the HDF5 file.

Usage
-----
    python cache_data.py              # uses Config defaults (bin_ms=10)
    python cache_data.py --bin_ms 5   # cache a different bin width
"""

import argparse
import os
import pickle
import time

from config import Config
from data import load_mcmaze

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def cache_path_for(bin_ms: int) -> str:
    return os.path.join(CACHE_DIR, f"mcmaze_bin{bin_ms}ms.pkl")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin_ms", type=int, default=None,
                        help="Resampling bin width in ms (default: Config.bin_ms)")
    args = parser.parse_args()

    cfg = Config()
    bin_ms = args.bin_ms if args.bin_ms is not None else cfg.bin_ms
    nwb_path = cfg.nwb_path

    print(f"Loading NWB: {nwb_path}")
    print(f"  bin_ms = {bin_ms}")
    t0 = time.time()
    spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = load_mcmaze(
        nwb_path, bin_ms
    )
    t_load = time.time() - t0
    print(f"  NWB load took {t_load:.1f} s")
    print(f"  spikes_raw: {spikes_raw.shape}  trial_info: {len(trial_info)} rows  "
          f"hand_pos: {'yes' if hand_pos_raw is not None else 'no'}")

    cache = {
        "spikes_raw": spikes_raw,
        "bin_width_s": bin_width_s,
        "trial_info": trial_info,
        "time_index_s": time_index_s,
        "hand_pos_raw": hand_pos_raw,
        "nwb_path": nwb_path,
        "bin_ms": bin_ms,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = cache_path_for(bin_ms)
    t0 = time.time()
    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_save = time.time() - t0

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved cache: {out_path}")
    print(f"  Size: {size_mb:.1f} MB  |  Write time: {t_save:.2f} s")


if __name__ == "__main__":
    main()
