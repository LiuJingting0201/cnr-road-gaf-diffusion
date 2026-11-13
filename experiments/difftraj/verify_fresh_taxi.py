"""Reopen a sealed SUMO pack and compare native metre/time arrays with scorer inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

receipt=json.loads(Path('manifest.json').read_text())
for path,key in [('native.npz','raw_sha256'),('observations.npz','observations_sha256')]:
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==receipt[key]
with np.load('native.npz') as native, np.load('observations.npz') as obs:
    ids=native['taxi'];counts=native['length'];raw=native['raw'];dxy=obs['delta']
    assert len(ids)==receipt['trips'] and np.array_equal(ids,obs['taxi'])
    assert np.array_equal(counts,obs['length'])
    offset=np.array([int(k.split('_')[0])%1440*60 for k in ids])
    ends=np.r_[0,counts.cumsum()]; t0=raw[ends[:-1],0].astype(int)
    assert np.array_equal(t0,obs['start']-offset)
    assert np.all(obs['episode_split']==9)
    max_error=0.
    for a,b in zip(ends[:-1],ends[1:]):
        v=raw[a:b]; assert np.isfinite(v).all() and len(v)>=120 and np.all(np.diff(v[:,0])==1)
        max_error=max(max_error,float(np.max(np.abs(dxy[a:b].cumsum(0)*.01-v[:,1:3]))))
    assert max_error<=0.0051
    report=dict(trips=len(ids),points=int(ends[-1]),episodes=len(receipt['episodes_used']),
                observed_start_minus_offset_equals_sim_start=True,
                maximum_coordinate_decode_error_m=max_error,coordinates_unit='metres',
                native_columns=['simulation_seconds','x_m','y_m','speed_mps'],
                model_score_evaluated=False)
Path('integrity.json').write_text(json.dumps(report,indent=2)+'\n')
