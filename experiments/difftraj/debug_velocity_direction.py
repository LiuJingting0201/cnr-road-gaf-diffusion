"""Decompose integrated displacement errors in saved paired validation draws."""
import argparse
import json
from pathlib import Path

import numpy as np


def norm(x):
    return np.linalg.norm(x, axis=-1)


def summarize(obs, gen, head, scale, duration):
    u, v = (p[:, -1] - p[:, 0] for p in (obs, gen))
    a, b = norm(u), norm(v)
    cos = np.clip((u*v).sum(1) / np.maximum(a*b, 1e-20), -1, 1)
    radial = (a-b)**2
    angular = 2*a*b*(1-cos)
    np.testing.assert_allclose(radial + angular, norm(v-u)**2, atol=1e-6)
    def hit(end):
        ij = np.floor((end-scale['lo'])/(scale['hi']-scale['lo'])*16).astype(int)
        return float(np.mean(((ij >= 0) & (ij < 16)).all(1) & (ij[:,0]*16+ij[:,1] == head[:,7])))
    valid = (a>1) & (b>1)
    g0 = gen[:,0]
    local = []
    for stride in (1, 10, 50):
        dx, dy = [p[:,stride:]-p[:,:-stride] for p in (obs,gen)]
        mask = (np.minimum(norm(dx),norm(dy)) > duration[:,None]*stride/199)
        angle = np.degrees(np.arccos(np.clip((dx*dy).sum(2)[mask]/(norm(dx)[mask]*norm(dy)[mask]),-1,1)))
        local.append(dict(stride=stride, moving_fraction=float(mask.mean()), median_deg=float(np.median(angle))))
    start_err = g0 - obs[:,0]
    delta_err = v-u
    cancellation = -2*(start_err*delta_err).sum()/(np.square(start_err).sum()+np.square(delta_err).sum())
    return dict(n=len(a), start_displacement_cancellation_fraction=float(cancellation),
                median_end_error_m=float(np.median(norm(gen[:,-1]-obs[:,-1]))), median_start_error_m=float(np.median(norm(gen[:,0]-obs[:,0]))),
                median_displacement_error_m=float(np.median(norm(v-u))),
                median_signed_net_length_error_m=float(np.median(b-a)),
                median_abs_net_length_error_m=float(np.median(abs(b-a))),
                median_net_angle_deg=float(np.median(np.degrees(np.arccos(cos[valid])))),
                rms_net_length_term_m=float(np.sqrt(radial.mean())), rms_net_angle_term_m=float(np.sqrt(angular.mean())),
                angle_share_of_squared_displacement_error=float(angular.sum()/(radial+angular).sum()),
                original_hit=hit(gen[:,-1]), observed_start_only_hit=hit(obs[:,0]+v),
                observed_net_bearing_only_hit=hit(g0+u*(b/np.maximum(a,1e-20))[:,None]),
                observed_net_length_only_hit=hit(g0+v*(a/np.maximum(b,1e-20))[:,None]),
                angular_valid=int(valid.sum()), local_heading=local)


def main(run, data):
    scale = dict(np.load(data/'normalization.npz'))
    idx = np.load(data/'validation/index.npz')
    out = {}
    for rep in ('coordinate','velocity'):
        for s in (42,43):
            z = np.load(run/f'{rep}-{s}/eval/draw-cache.npz')
            rows = z['validation_rows']
            np.testing.assert_array_equal(z['ids'], idx['ids'][rows])
            if out:
                np.testing.assert_array_equal(rows, original_rows)
                assert z['noise_sha256'].item()==noise
            original_rows, noise = rows, z['noise_sha256'].item()
            n = len(rows)
            np.testing.assert_array_equal(z['draw_rows'][:n], np.arange(n))
            np.testing.assert_array_equal(z['draw_labels'][:n], z['true_labels'])
            obs = np.load(data/'validation/traj.npy',mmap_mode='r')[rows].astype('d')*scale['std']+scale['mean']
            gen = z['xy_m'][:n].astype('d')
            h = np.load(data/'validation/head.npy')[rows]
            duration = np.rint(h[:,2]*scale['header_std'][1]+scale['header_mean'][1])
            labs = idx['classes'][rows]
            out[f'{rep}-{s}'] = {key:summarize(obs[m],gen[m],h[m],scale,duration[m]) for key,m in
                                 [('all',np.arange(n))]+[(key,labs==key) for key in np.unique(labs)]}
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','data'): p.add_argument('--'+k,required=True,type=Path)
    args=p.parse_args()
    main(args.run,args.data)
