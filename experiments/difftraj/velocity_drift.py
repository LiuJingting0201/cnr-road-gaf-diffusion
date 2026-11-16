"""Measure net motion direction and distance error across trip durations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main(data, draws, output):
    with np.load(draws) as sim, np.load(data/'validation/index.npz') as index:
        rows = np.flatnonzero(index['classes'] == 'taxi')
        assert np.array_equal(index['ids'][rows], sim['ids'])
        generated = sim['xy_m'].astype('d')
    with np.load(data/'normalization.npz') as scale:
        observed = np.load(data/'validation/traj.npy', mmap_mode='r')[rows] * scale['std'] + scale['mean']
        head = np.load(data/'validation/head.npy', mmap_mode='r')[rows]
        duration = np.rint(head[:,2] * scale['header_std'][1] + scale['header_mean'][1])
    obs_delta = observed[:,-1] - observed[:,0]
    gen_delta = generated[:,-1] - generated[:,0]
    vector_error = np.linalg.norm(gen_delta - obs_delta, axis=1)
    observed_norm = np.linalg.norm(obs_delta, axis=1)
    generated_norm = np.linalg.norm(gen_delta, axis=1)
    valid = (observed_norm > 1) & (generated_norm > 1)
    cosine = (obs_delta[valid] * gen_delta[valid]).sum(axis=1) / (observed_norm[valid] * generated_norm[valid])
    bearing_error = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    order = np.array_split(np.argsort(duration, kind='stable'), 3)
    summary = dict(valid_vectors=int(valid.sum()), median_net_bearing_error_degrees=float(np.median(bearing_error)),
                   p90_net_bearing_error_degrees=float(np.quantile(bearing_error, .9)),
                   mean_velocity_vector_error_mps=float(np.mean(vector_error/duration)),
                   duration_vs_vector_error_rank_correlation=float(spearmanr(duration, vector_error).statistic),
                   duration_tertiles=[dict(n=len(i), min_s=int(duration[i].min()), max_s=int(duration[i].max()),
                        median_displacement_error_m=float(np.median(vector_error[i]))) for i in order])
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('--data','--draws','--output'):
        p.add_argument(name, type=Path, required=True)
    a=p.parse_args()
    main(a.data, a.draws, a.output)
