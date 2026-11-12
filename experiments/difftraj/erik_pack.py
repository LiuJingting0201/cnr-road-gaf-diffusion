"""Pack Erik's released CSV segments, keeping original lengths and elapsed durations."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
paths=sorted(Path(sys.argv[1]).glob('*.csv'))
delta=[]; sizes=[]; users=[]; starts=[]; durations=[]
for p in paths:
    with p.open() as f:
        rows=list(csv.DictReader(f))
    times=[datetime.fromisoformat(row['timestamp']).replace(tzinfo=timezone.utc).timestamp() for row in rows]
    if len(rows)<2 or times[-1]<=times[0]: raise ValueError(p)
    coords=np.array([[float(row['lon']),float(row['lat'])] for row in rows])
    xy=np.rint(coords*1e6).astype('i4')
    delta.append(np.diff(xy,axis=0,prepend=np.zeros((1,2),dtype='i4')))
    sizes.append(len(xy));users.append(int(rows[0]['uid']));starts.append(times[0]);durations.append(times[-1]-times[0])
np.savez_compressed(sys.argv[2],delta=np.concatenate(delta),length=sizes,taxi=users,start=starts,duration=durations)
print(len(paths),len(set(users)),sum(sizes))
