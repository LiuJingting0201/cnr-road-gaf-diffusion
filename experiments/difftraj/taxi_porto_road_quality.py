"""Paired 10k Porto taxi road audit on the archived native-clock holdout requests."""
import argparse
import csv
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from road_match import init, match, summarize
from utils.utils import resample_trajectory

BOUNDS = (-8.82, 40.90, -8.20, 41.42)
SEEDS = [42, 43]


def evaluate(track):
    row = match((track, 15., 'taxi'), with_interior=True)
    x0, y0, x1, y1 = BOUNDS
    inside = (track[:, 0] >= x0) & (track[:, 0] <= x1) & (track[:, 1] >= y0) & (track[:, 1] <= y1)
    row['entirely_in_box'] = bool(inside.all())
    row['points_outside_box'] = int((~inside).sum())
    return row


def add_metrics(summary, rows):
    for key in ['interior_intervals', 'interior_linked', 'interior_walk', 'entirely_in_box', 'points_outside_box']:
        summary[key] = sum(r[key] for r in rows)
    summary['fully_in_box_walk'] = sum(r['walk'] for r in rows if r['entirely_in_box'])
    summary['fully_in_box_passed'] = sum(r['passed'] for r in rows if r['entirely_in_box'])
    return summary


def main(args):
    args.output.mkdir(exist_ok=True, parents=True)
    with np.load(args.pack / 'observations.npz') as raw:
        sizes = raw['length']
        ends = np.r_[0, sizes.cumsum()]
        delta = raw['delta']
    reference = None
    cohorts = {}
    for seed in SEEDS:
        folder = args.pack / str(seed)
        with np.load(folder / 'draws.npz') as draws:
            ids = draws['ids']
            if reference is None:
                reference = ids.copy()
                assert len(reference) == len(set(reference)) == 1000
                cohorts['observed'] = [delta[ends[i]:ends[i+1]].cumsum(0)*1e-6 for i in reference]
            np.testing.assert_array_equal(ids, reference)
            digest = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
            prior = json.loads((folder / 'porto-final.json').read_text())
            assert prior['request_index_sha256'] == digest
            cohorts[str(seed)] = [resample_trajectory(x,int(sizes[i])) for x,i in zip(draws['xy'],ids,strict=True)]
    summary = dict(request_index_sha256=digest, requests=1000, dt_s=15,
                   bounds_lonlat=BOUNDS, interior='interval midpoint in [10%,90%] of elapsed time',
                   threshold_m=50, gps_accuracy_m=10, interval_speed_ceiling_mps=40,
                   graph_sha256=hashlib.sha256(args.pbf.read_bytes()).hexdigest(), results={})
    fields = ['cohort','request_order','dataset_row','intervals','linked_intervals','walk','reasonable','passed',
              'interior_intervals','interior_linked','interior_walk','entirely_in_box','points_outside_box',
              'valid_projections','points_within_50m','matched_error']
    final_path = args.output / 'porto-road-requests.csv.gz'
    tmp_path = final_path.with_name(final_path.name + '.tmp')
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.config.resolve(),)) as pool, gzip.open(tmp_path, 'wt', newline='') as file:
        table = csv.DictWriter(file, fieldnames=fields)
        table.writeheader()
        for group, tracks in cohorts.items():
            rows = list(pool.map(evaluate, tracks, chunksize=4))
            summary['results'][group] = add_metrics(summarize(rows), rows)
            for k, r in enumerate(rows):
                offsets = np.asarray(r['offsets'])
                record = dict(cohort=group, request_order=k, dataset_row=int(reference[k]),
                              intervals=r['n']-1, linked_intervals=r['linked'], valid_projections=len(offsets),
                              points_within_50m=int((offsets <= 50).sum()), matched_error=r.get('error',''))
                record.update({field:r[field] for field in fields if field not in record})
                table.writerow(record)
            print(group, summary['results'][group], flush=True)
    tmp_path.replace(final_path)
    (args.output / 'porto-road-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pbf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    main(parser.parse_args())
