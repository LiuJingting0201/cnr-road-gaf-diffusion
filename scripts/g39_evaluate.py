"""Use inherited independent native physical/road metrics for direct G39 draws."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import g38_evaluate as metrics
from g33_evaluate import extract_record,digest,save
from g33_model import CLASSES


def diagnostics(generated,raw_cache,output):
    receipt=json.loads((raw_cache/'complete.json').read_text())
    assert receipt['complete'] and digest(raw_cache/'raw.npz')==receipt['raw_sha256']
    with np.load(raw_cache/'raw.npz',allow_pickle=False) as f:
        ids=f['ids'];labels=f['labels'];profiles=f['profiles'];train=f['partition']=='train'
    library={}
    for c in range(5):
        ii=sorted(np.flatnonzero(train & (labels==c)),key=lambda i:hashlib.sha256(str(ids[i]).encode()).digest())[:512]
        library[c]=profiles[ii]
    rows=json.loads((generated/'candidates.json').read_text())['rows'];byclass={}
    with np.load(generated/'samples.npz',allow_pickle=False) as f:
        txy,offsets=f['txy'],f['offsets']
    pp=np.stack([extract_record(txy[a:b])['profile'] for a,b in zip(offsets[:-1],offsets[1:])])
    for c,name in enumerate(CLASSES):
        ix=[i for i,r in enumerate(rows) if r['label']==c];groups={}
        for i in ix:groups.setdefault(rows[i]['request_id'],[]).append(i)
        diversity=[float(np.mean([np.mean(np.abs(pp[a]-pp[b])) for a,b in itertools.combinations(g,2)])) for g in groups.values()]
        nn=cdist(pp[ix],library[c],metric='cityblock').min(1)/257 if ix else np.array([])
        byclass[name]={'requested_outputs':len(ix),'requests':len(groups),'training_library_n':len(library[c]),
            'mean_pairwise_progress_distance':float(np.mean(diversity)) if diversity else None,
            'nearest_training_shape_distance_mean':float(nn.mean()) if len(nn) else None,
            'nearest_training_shape_distance_quantiles':np.quantile(nn,[0,.1,.5,.9,1]).tolist() if len(nn) else [],
            'near_shape_rate_threshold_1e4':float(np.mean(nn<1e-4)) if len(nn) else None,
            'negative_decode_fraction':sum(rows[i]['negative_bins'] for i in ix)/max(sum(rows[i]['motion_bins'] for i in ix),1),
            'upper_decode_fraction':sum(rows[i]['upper_clipped_bins'] for i in ix)/max(sum(rows[i]['motion_bins'] for i in ix),1),
            'constant_speed_fallbacks':sum(rows[i]['constant_speed_fallback'] for i in ix)}
    save(output/'diagnostics.json',{'complete':True,'by_class':byclass,'generated_sha256':digest(generated/'complete.json'),
        'training_cache_sha256':digest(raw_cache/'complete.json'),'library':'fixed SHA-ranked up to512 train legs per class',
        'limit':'Normalized progress shape only; approximate finite-library near-copy diagnostic, not full spatial privacy/memorization proof.'})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('generated','catalog','reference','simulation','network','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--raw-cache',type=Path,required=True)
    a=p.parse_args();metrics.METHODS=['R']
    metrics.evaluate(a.generated,a.catalog,a.reference,a.simulation,a.network,a.output)
    diagnostics(a.generated,a.raw_cache,a.output)
