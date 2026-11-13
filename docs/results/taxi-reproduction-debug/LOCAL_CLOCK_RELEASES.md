# Reusable five-class datasets with native departure times

Use these server directories for new full-five-class experiments:

* `/workspace/cnr-experiments/difftraj-five-10k-localclock-v1` (50,000 training trips, five classes).
* `/workspace/cnr-experiments/difftraj-five-20k-localclock-v1` (100,000 training trips, five classes).

Both contain `train`, `validation` and `test` partitions with `traj.npy` (N, 200, 2), `head.npy` (N, 8), and `index.npz` (IDs and separate class labels). Pass coordinates and headers from the same corrected partition to the loader. For the author's input convention, transpose the coordinates to N × 2 × 200. Coordinates are already standardized using that release's training population. Head column 0 is a five-minute simulation-time category. Columns 1–5 are the original training-standardized numerical conditions. Columns 6 and 7 are the original training-grid endpoint IDs. No extra normalization is needed to use the five-class arrays as released.

## Verification and storage

For every selected ID, the converter recovered the first timestamp from its original native trajectory, checked it against its SUMO departure receipt, checked the last timestamp against arrival minus one second, and set column 0 to floor(first timestamp / 300) modulo 288. The shift is synthetic-data-specific and must not be applied to calendar timestamps in real datasets. All new departure bins (0–3 across these five-class packs) in development and test occur in the corresponding training population. A small number of buses or other vehicles may depart late in the launch window; the actual departure timestamp determines the bin.

The 10k-per-class conversion checked 50,000/5,000/5,000 rows. All 60,000 changed their departure bin. The 20k-per-class conversion checked 100,000/5,000/5,000 rows; 109,950 changed their departure bin, while 50 already mapped to the native value. All seven remaining header columns were byte-for-byte identical. ID and episode splits remain disjoint. Original files were not edited.

The new releases have their own small `head.npy` files and `manifest.json`. Large coordinate, ID, normalization and count files are links to their source releases. The original episode archives, including **historical** per-episode `h` arrays, can be reached as `raw_shards/<episode>/trips.npz`; `arrivals.xml` is in the same episode directory. Read `raw` in these archives for native simulation times. Read the release-level `head.npy` for corrected model inputs. Retain the source release directories or dereference links when transferring these corrected packs off-server. A future single-mode fit should still refit numerical normalization and OD bounds on that mode's training subset.

## Compatibility of scripts

The current versions of `make_balanced_trips.py`, `expand_balanced_trips.py` and `taxi10k_pack.py` recover or preserve the native simulation departure time. Old server copies or pre-existing archives can still use the original offset scheme. The forensic source `taxi_preprocessing_forensics.py` and frozen-checkpoint interventions intentionally audit the historical 10k taxi fit. The sealed fresh-taxi exporter/validator pair `fresh_taxi_holdout.py` and `verify_fresh_taxi.py` stores and verifies the legacy offset in its *model pack*, also writing native observations. For those specific historical packs `taxi_debug_data.py:actual_time` subtracts the known offset exactly once. It must not be applied to the corrected native-clock inputs. Avoid inferring the clock convention from the coordinate file alone.

SHA-256 hashes and the exact clock distributions of every partition are in the [10k manifest](../difftraj-five-10k/localclock-v1-manifest.json) and [20k manifest](../difftraj-five-20k/localclock-v1-manifest.json). The release tool is `experiments/difftraj/release_local_clock.py`; it uses a lock and an atomic directory rename.
