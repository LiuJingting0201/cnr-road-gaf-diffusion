"""Paired, all-candidate physical evaluation of frozen G28 generators."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from cnr.episode_w1 import episode_w1
from g16_data import NAMES
from g22_diffusion import decode, sample, load
from g22_eval import empirical
from g23_eval import phys
from g23_modes import speed_series
from g28_train import ARMS, Denoiser

ROOT = Path('g28-data')
EVALUATED = (*ARMS, 'E')
COLUMNS = ['longest_steady_W1_s', 'duration_W1_s', 'lag4_W1_mps',
           'mean_speed_W1_mps', 'mean_abs_acceleration_W1_mps2',
           'stopped_fraction_W1', 'speed_p95_W1_mps']


def physical(t, xy):
    if t[-1] - t[0] < 5:
        raise ValueError("Generated trip shorter than physical metric support; keep pool and audit failure")
    base = phys(t, xy)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy.astype(float), axis=0), axis=1))]
    speed = np.diff(np.interp(np.arange(int(np.floor(t[-1]-t[0]))+1), t-t[0], distance))
    return base + [float(np.abs(np.diff(speed)).mean()), float((speed < .3).mean()),
                   float(np.quantile(speed, .95))]


def draw(arm, seed, request):
    root = ROOT / 'models' / f'{arm}-{seed}'
    if arm == 'E':
        root.mkdir(parents=True, exist_ok=True)
        times, profiles = empirical(load('train'), request, seed)
        return dict(ids=request['ids'],duration=times,progress=profiles,
                    bounded=np.zeros_like(times,dtype=bool),flat=np.zeros_like(times,dtype=bool))
    if not (root / 'complete.json').exists():
        raise RuntimeError(f'Incomplete training {root}')
    out = root / 'samples.npz'
    if not out.exists():
        model = Denoiser().cuda().eval()
        model.load_state_dict(torch.load(root/'model.pt', weights_only=True, map_location='cuda'))
        begin = time.monotonic()
        generated = sample(model, request, json.loads((root/'stats.json').read_text()), seed)
        np.savez_compressed(root/'samples.tmp.npz', **generated)
        (root/'samples.tmp.npz').replace(out)
        (root/'sampling.json').write_text(json.dumps({'seconds': time.monotonic()-begin}))
    with np.load(out) as data:
        pool = {k:data[k] for k in data.files}
    assert np.array_equal(pool['ids'], request['ids'])
    return pool


def run():
    torch.set_num_threads(2)
    with np.load(ROOT/'requests.npz') as d:
        request = {k:d[k] for k in d.files}
    # Save all pools before reading observed outcomes.
    for arm in EVALUATED:
        for seed in (42,43):
            draw(arm, seed, request)
    with np.load(ROOT/'references.npz') as ref:
        assert np.array_equal(ref['ids'], request['ids'])
        observations, offsets = ref['txy'], ref['offset']
        reference = np.array([physical(observations[a:b,0], observations[a:b,1:])
                              for a,b in zip(offsets[:-1], offsets[1:])])
        reference[:,1] = ref['duration']
    labels = request['labels']
    episode = np.unique([i.split('_')[0] for i in request['ids']], return_inverse=True)[1]
    assert len(np.unique(episode)) == 8
    weights = np.vstack([np.ones(8), np.random.default_rng(280133).multinomial(8, np.ones(8)/8, 2000)])
    curves, tables = {}, {}
    for arm in EVALUATED:
        seed_curves, seed_tables, diversity = [], [], []
        caps = flats = 0
        endpoint = 0.
        for seed in (42,43):
            generated = draw(arm, seed, request)
            caps += int(generated['bounded'].sum())
            flats += int(generated['flat'].sum())
            metrics, profiles = [], []
            for path, times, progress in zip(request['paths'],generated['duration'],generated['progress']):
                row, speedrow = [], []
                for duration, profile in zip(times, progress):
                    t, xy = decode(path, duration, profile)
                    endpoint = max(endpoint, float(np.linalg.norm(xy[-1]-path[-1])))
                    row.append(physical(t,xy))
                    speedrow.append(speed_series(t,xy)[0])
                metrics.append(row)
                profiles.append(speedrow)
            metrics, profiles = np.array(metrics), np.array(profiles)
            np.savez_compressed(ROOT/'models'/f'{arm}-{seed}'/'physical.npz',ids=request['ids'],metrics=metrics,velocity=profiles)
            per_class = []
            for c in range(8):
                use = labels == c
                per_class.append(np.stack([episode_w1(metrics[use,:,j],reference[use,j,None],episode[use],False,weights)
                                           for j in range(len(COLUMNS))],axis=1))
            seed_curves.append(per_class)
            seed_tables.append(np.array(per_class)[:,0])
            pair_dist = np.mean([np.sqrt(np.mean((profiles[:,i]-profiles[:,j])**2,axis=1))
                                 for i in range(4) for j in range(i+1,4)],axis=0)
            diversity.append(np.array([[pair_dist[labels==c].mean(),generated['duration'][labels==c].std(1).mean()]
                                       for c in range(8)]))
        arrays = np.mean(seed_curves,axis=0)
        curves[arm] = {'all': arrays.mean(0), 'motor': arrays[[1,2,3,5,6,7]].mean(0)}
        tables[arm] = {'macro':curves[arm]['all'][0].tolist(),'motor_macro':curves[arm]['motor'][0].tolist(),
                       'byclass':dict(zip(NAMES,np.mean(seed_tables,axis=0).tolist())),
                       'diversity_columns':['within_request_speed_RMS_mps','within_request_duration_std_s'],
                       'diversity_byclass':dict(zip(NAMES,np.mean(diversity,axis=0).tolist())),
                       'diversity_macro':np.mean(diversity,axis=(0,1)).tolist(),
                       'duration_clips':caps,'flat_fallbacks':flats,'endpoint_max_m':endpoint}
    differences = {}
    for a,b in [('SV','B'),('SV','E'),('B','E'),('S','B'),('V','B'),('P','B'),('SV','S'),('SV','V'),('SV','P'),('SV','SH')]:
        differences[a+'-'+b] = {}
        for population in ('all','motor'):
            delta = curves[a][population] - curves[b][population]
            differences[a+'-'+b][population] = {'delta':delta[0].tolist(),
                'ci':np.quantile(delta[1:],[.025,.975],axis=0).T.tolist()}
    result = {'requests':len(labels),'episodes':8,'candidates_per_arm':len(labels)*8,
              'columns':COLUMNS,'arms':tables,'differences':differences,
              'scope':'same-map independent traffic, fixed two fitting seeds, all generated candidates'}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (ROOT/'evaluation-complete.txt').write_text('complete')


if __name__ == '__main__':
    run()
