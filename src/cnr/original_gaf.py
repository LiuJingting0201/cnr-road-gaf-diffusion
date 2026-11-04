"""Adapter around unmodified original Hilbert/sequence source.

CNN implementation is new; image semantics follow the original RGB encoder.
Explicit deployment changes: fit coordinate/channel bounds on training only,
freeze them for validation and generation, clip out-of-range input and record it.
No derivative through this discrete encoder is claimed.
"""
import sys
from pathlib import Path
import numpy as np
from pyts.image import GramianAngularField,MarkovTransitionField
vendor=Path(__file__).resolve().parents[2]/'vendor/original_cnr/src'
if str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
from cnr_trajectory.encoding.hilbert import hilbert_index
from cnr_trajectory.encoding.sequence import squeeze,duplicate
from cnr_trajectory.encoding.gaf import encode_hilbert_sequences


def trajectory_sequence(xy,lo,hi,eps=.0037,length=224):
    xy=np.asarray(xy,dtype=np.float64)
    if len(xy)==0 or not np.isfinite(xy).all():raise ValueError('Empty or nonfinite trajectory')
    unit=(xy-lo)/(hi-lo)
    clipped=int(np.any((unit<0)|(unit>1),axis=1).sum());unit=unit.clip(0,1)
    h=[hilbert_index(float(x),float(y),eps) for x,y in unit]
    result=duplicate(squeeze(h,length),length)
    if len(result)!=length:raise ValueError('Original squeeze cannot compress this sequence to target length')
    return np.array(result,dtype=np.float64),clipped


def raw_channels(sequences):
    x=np.asarray(sequences,dtype=np.float64)
    return np.stack([GramianAngularField(method='summation',sample_range=(0,1)).fit_transform(x),
      GramianAngularField(method='difference',sample_range=(0,1)).fit_transform(x),
      MarkovTransitionField(strategy='uniform',n_bins=8).fit_transform(x)],axis=-1)


def fit_channels(sequences,batch=64):
    lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
    for i in range(0,len(sequences),batch):
        x=raw_channels(sequences[i:i+batch]);lo=np.minimum(lo,x.min((0,1,2)));hi=np.maximum(hi,x.max((0,1,2)))
    if np.any(hi<=lo):raise ValueError('Constant encoded channel, as rejected by original encoder')
    return lo,hi


def encode_frozen(sequences,lo,hi,batch=64):
    images=[];outside=0
    for i in range(0,len(sequences),batch):
        raw=raw_channels(sequences[i:i+batch]);outside+=int(((raw<lo)|(raw>hi)).sum())
        images.append(np.round(((raw-lo)/(hi-lo)).clip(0,1)*255).astype(np.uint8))
    return np.concatenate(images),outside
