"""Compact canonical time/id/XY/class data for bandwidth-limited SSH transfer."""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
import numpy as np
import pandas as pd
from cnr.data import CLASSES

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
columns=['timestep_time','vehicle_id','vehicle_x','vehicle_y','class2']
df=pd.read_csv(a.source,usecols=columns,dtype={'vehicle_id':str})
df=df.sort_values(['vehicle_id','timestep_time'],kind='stable')
if df[columns].isna().any().any():raise ValueError('Null canonical field')
xy=df[['vehicle_x','vehicle_y']].to_numpy(dtype=np.float64);t=df.timestep_time.to_numpy(dtype=np.float64)
qxy=np.rint(xy*100).astype(np.int32);qt=np.rint(t*4).astype(np.int32)
if not np.allclose(xy,qxy.astype(float)/100,atol=1e-9,rtol=0):raise ValueError('Coordinates exceed 0.01 precision')
if not np.array_equal(t,qt.astype(float)/4):raise ValueError('Times not on 0.25 grid')
if not np.array_equal(xy.astype(np.float32),(qxy.astype(float)/100).astype(np.float32)):
    raise ValueError('Model float32 coordinates would change')
groups=df.groupby('vehicle_id',sort=True)
if groups.class2.nunique().max()!=1:raise ValueError('Mixed class ID')
counts=groups.size().to_numpy(dtype=np.int32);ids=np.array(groups.size().index.tolist())
labels=np.array([CLASSES.index(x) for x in groups.class2.first()],dtype=np.int8)
h=hashlib.sha256()
with Path(a.source).open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
meta={'raw_source_sha256':h.hexdigest(),'source_name':Path(a.source).name,'rows':len(df),
      'classes':CLASSES,'coordinate_scale':100,'time_scale':4,
      'note':'Canonical columns only; speed stays in original CSV. Float32 XY and times verified unchanged.',
      'max_coordinate_decimal_roundoff':float(np.abs(xy-qxy.astype(float)/100).max())}
buffer=io.BytesIO()
np.savez(buffer,delta_xy=np.diff(qxy,axis=0,prepend=np.zeros((1,2),dtype=np.int32)),
         delta_t=np.diff(qt,prepend=np.array([0],dtype=np.int32)),counts=counts,ids=ids,labels=labels,
         metadata=np.array(json.dumps(meta)))
out.parent.mkdir(parents=True,exist_ok=True)
out.write_bytes(lzma.compress(buffer.getvalue(),preset=6))
print(json.dumps(meta|{'packed_bytes':out.stat().st_size},indent=2))
