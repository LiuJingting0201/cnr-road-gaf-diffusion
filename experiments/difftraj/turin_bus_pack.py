"""Pack class-checked GTT whole-trip bus candidates in observed timestamp order."""
import csv,gzip,sys
from collections import defaultdict
import numpy as np
groups=defaultdict(list)
with gzip.open(sys.argv[1],'rt') as f:
    for r in csv.DictReader(f):
        if (r['mode'],r['class_check'],r['terminal_covered'])==('bus','consistent','1'):
            key=tuple(r[k] for k in ('vehicle','service_date','trip','segment'))
            groups[key].append((int(r['unix_seconds']),float(r['lon']),float(r['lat'])))
length=[]; taxi=[]; start=[]; duration=[]; delta=[]
for key,rows in sorted(groups.items()):
    t,lon,lat=np.array(sorted(rows)).T
    assert np.all(np.diff(t)>0)
    xy=np.rint(np.column_stack((lon,lat))*1e6).astype('int32')
    delta.append(np.diff(xy,axis=0,prepend=np.zeros((1,2),dtype='int32')))
    length.append(len(rows));taxi.append(key[0]);start.append(int(t[0]));duration.append(t[-1]-t[0])
np.savez_compressed(sys.argv[2],length=length,taxi=taxi,start=start,duration=duration,delta=np.concatenate(delta),timezone='Europe/Rome')
