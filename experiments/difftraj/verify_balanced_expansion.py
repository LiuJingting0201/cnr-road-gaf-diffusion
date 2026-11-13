"""Verify nested membership, split isolation, native sizes and normalized files."""
import json
from pathlib import Path
import numpy as np
old=Path('/workspace/cnr-experiments/difftraj-five-10k')
seen=set();seen_episodes=set();report={}
for split in ['train','validation','test']:
    folder=Path(split); prev=np.load(old/split/'index.npz'); idx=np.load(folder/'index.npz')
    ids=idx['ids'];labels=idx['classes'];cur=set(ids);ep={k.split('_')[0] for k in ids}
    assert len(cur)==len(ids) and not seen&cur and not seen_episodes&ep
    seen.update(cur);seen_episodes.update(ep)
    assert np.array_equal(ids[:len(prev['ids'])],prev['ids']) and np.array_equal(labels[:len(prev['ids'])],prev['classes'])
    if split!='train':assert len(ids)==len(prev['ids'])
    counts=dict(zip(*np.unique(labels,return_counts=True)))
    assert len(counts)==5 and all(v==(20000 if split=='train' else 1000) for v in counts.values())
    x=np.load(folder/'traj.npy',mmap_mode='r');h=np.load(folder/'head.npy',mmap_mode='r')
    assert x.shape==(len(ids),200,2) and h.shape==(len(ids),8) and np.isfinite(x).all() and np.isfinite(h).all()
    report[split]=dict(n=len(ids),episodes=len(ep),classes={k:int(v) for k,v in counts.items()},source_episode_roots=[str(old),str(Path.cwd())])
Path('verified.json').write_text(json.dumps(report,indent=2))
