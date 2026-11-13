"""Replay four distribution metrics and measure requested endpoint errors for saved taxi draws."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from porto import grid
from taxi_od_diagnostic import audit, native_xy

GROUPS = (
    ('SUMO', '10k', 'development.json', 8),
    ('Porto', 'porto', 'porto-final.json', 9),
)


def evaluate(run, original_score, split, city, seed, writer):
    """Validate saved identifiers and measure unaltered endpoints, before any map matching."""
    replay = audit(run, run, run / original_score)
    full = replay['native']['full']
    with np.load(run / 'draws.npz') as draws, np.load(run / 'prepared.npz') as prep, np.load(run / 'observations.npz') as native:
        ids = draws['ids']
        eligible = np.flatnonzero(prep['split'] == split)
        expected = np.random.default_rng(42).choice(eligible, len(eligible), replace=False)
        np.testing.assert_array_equal(expected, ids)
        np.testing.assert_equal(len(ids), 1000)
        real, length = native_xy(native, ids)
        observed = np.asarray([r[[0, -1]] for r in real])
        generated = draws['xy'][:, [0, -1]]
        requested = prep['h'][ids, 6:8].astype(int)
        lo, hi = prep['lo'], prep['hi']
        observed_cells = grid(observed, lo, hi)
        generated_cells = grid(generated, lo, hi)
        errors = np.asarray([[length(np.stack([obs[j], gen[j]])) for j in (0, 1)]
                             for obs, gen in zip(observed, generated, strict=True)])
        hit = generated_cells == requested
        inside = ((generated >= lo) & (generated <= hi)).all(axis=2)
        joint = hit.all(axis=1)
        header_mismatches = (observed_cells != requested).sum(axis=0).tolist()
        reference = replay['reference_sha256']
    np.testing.assert_allclose(np.quantile(errors, [.1, .5, .9], axis=0).T,
                               [replay['q10_q50_q90_origin_error_m'], replay['q10_q50_q90_destination_error_m']])
    if not any(header_mismatches):
        np.testing.assert_allclose([*hit.mean(axis=0), joint.mean()],
                                   [full['origin_hit'], full['destination_hit'], full['joint_hit']])
    digest = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
    for i in range(len(ids)):
        writer.writerow((city, seed, i, int(ids[i]), f'{errors[i,0]:.6f}', f'{errors[i,1]:.6f}',
                         int(hit[i,0]), int(hit[i,1]), int(joint[i]), int(inside[i,0]), int(inside[i,1])))
    return dict(city=city, seed=seed, split='development' if split == 8 else 'reserve', n=len(ids),
                draw_id_sha256=digest, reference_sha256=reference,
                grid_full_minmax_coordinates=dict(low=lo.tolist(), high=hi.tolist()),
                requested_vs_observed_cell_mismatches=header_mismatches,
                distribution={k: full[k] for k in ('density_js', 'od_js', 'top20_cell_f1')} | {'length_js': replay['native']['length_js']},
                origin_error_m={str(q):float(np.percentile(errors[:,0],q)) for q in (50,90,95)},
                destination_error_m={str(q):float(np.percentile(errors[:,1],q)) for q in (50,90,95)},
                requested_cell_hits=dict(origin=int(hit[:,0].sum()), destination=int(hit[:,1].sum()), both=int(joint.sum())),
                requested_cell_hits_strict_inside=dict(origin=int((hit[:,0]&inside[:,0]).sum()), destination=int((hit[:,1]&inside[:,1]).sum()),
                                                      both=int((joint&inside.all(axis=1)).sum())),
                both_endpoints_within_m={str(m):int((errors <= m).all(axis=1).sum()) for m in (100,250,500,1000)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records = []
    trip_file = args.output/'per-trip.csv'
    with trip_file.with_suffix('.tmp').open('w',newline='') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow(('city','seed','draw_index','pack_index','origin_error_m','destination_error_m',
                         'origin_requested_cell_hit','destination_requested_cell_hit','both_requested_cells_hit',
                         'origin_inside_training_bounds','destination_inside_training_bounds'))
        for city, folder, orig, split in GROUPS:
            items = [evaluate(args.experiment/folder/str(seed), orig, split, city, seed, writer) for seed in (42, 43)]
            assert len({item['draw_id_sha256'] for item in items}) == 1
            assert len({item['reference_sha256'] for item in items}) == 1
            records.extend(items)
    trip_file.with_suffix('.tmp').replace(trip_file)
    (args.output/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
