"""Count labelled, continuous Beijing road-mode segments before model fitting."""
import json
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
import numpy as np

epoch = datetime(1899, 12, 30)
def serial(s):
    return (datetime.strptime(s, '%Y/%m/%d %H:%M:%S') - epoch).total_seconds()
classes = {'bus', 'bike', 'car', 'taxi', 'motorcycle'}
counts = defaultdict(int)
users = defaultdict(set)
with zipfile.ZipFile(sys.argv[1]) as z:
    names = z.namelist()
    for label in (n for n in names if n.endswith('/labels.txt')):
        parent = label.rsplit('/', 1)[0]
        intervals = sorted((serial(a), serial(b), m) for a,b,m in
                           (line.split('\t') for line in z.read(label).decode().splitlines()[1:] if line.strip()))
        starts = np.array([a for a,b,m in intervals])
        ends = np.array([b for a,b,m in intervals])
        for n in (name for name in names if name.startswith(parent+'/Trajectory/') and name.endswith('.plt')):
            with z.open(n) as f:
                data = np.loadtxt(f, delimiter=',', skiprows=6, usecols=(0,1,4),ndmin=2)
            if len(data)<120:
                continue
            lat,lon,t=data.T
            t=np.rint(t*86400)
            idx=np.searchsorted(starts,t,side='right')-1
            valid=(idx>=0)&(t<=ends[idx])&(lon>=116)&(lon<=116.8)&(lat>=39.6)&(lat<=40.3)
            key=np.where(valid,idx,-1)
            cuts=np.flatnonzero((np.diff(key)!=0)|(np.diff(t)>120)|(np.diff(t)<=0))+1
            for a,b in zip(np.r_[0,cuts],np.r_[cuts,len(t)]):
                if b-a>=120 and key[a]>=0 and (mode:=intervals[key[a]][2]) in classes:
                    counts[mode]+=1
                    users[mode].add(parent)
print(json.dumps({k:{'segments':counts[k],'people':len(users[k])} for k in sorted(classes)},indent=2))
