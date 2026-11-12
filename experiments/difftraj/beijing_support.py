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
minimum=int(sys.argv[3]) if len(sys.argv)>3 else 120
records=[]
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
            if len(data)<minimum:
                continue
            lat,lon,t=data.T
            t=np.rint(t*86400)
            idx=np.searchsorted(starts,t,side='right')-1
            valid=(idx>=0)&(t<=ends[idx])&(lon>=116)&(lon<=116.8)&(lat>=39.6)&(lat<=40.3)
            key=np.where(valid,idx,-1)
            cuts=np.flatnonzero((np.diff(key)!=0)|(np.diff(t)>120)|(np.diff(t)<=0))+1
            for a,b in zip(np.r_[0,cuts],np.r_[cuts,len(t)]):
                if b-a>=minimum and key[a]>=0 and (mode:=intervals[key[a]][2]) in classes:
                    if len(sys.argv)>2:
                        # Local east/north in metres, origin 116 E, 40 N.
                        points=np.column_stack([(lon[a:b]-116)*111195*np.cos(np.deg2rad(40)),(lat[a:b]-40)*111195])
                        times=t[a:b]-t[a]
                        grid=np.linspace(0,times[-1],200)
                        xy=np.stack([np.interp(grid,times,coord) for coord in points.T])
                        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
                        if arc[-1]>0:
                            dt=times[-1]/199
                            v=np.diff(np.interp(grid,times,arc))/dt
                            h=[int((t[a]+8*3600)//300)%288,arc[-1],times[-1],len(times),arc[-1]/(len(times)-1),arc[-1]/times[-1]]
                            records.append((xy,v,h,dt,mode,parent,n,int(a),int(b)))
                    counts[mode]+=1
                    users[mode].add(parent)
print(json.dumps({k:{'segments':counts[k],'people':len(users[k])} for k in sorted(classes)},indent=2))

if records:
    import hashlib
    xy,v,h,dt,mode,person,source,start_row,end_row=map(np.asarray,zip(*records))
    train=np.array([int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%5!=0 for u in person])
    classes,y=np.unique(mode,return_inverse=True)
    lo=xy[train].min((0,2)); hi=xy[train].max((0,2))
    def cell(pos):
        q=np.clip(((pos-lo)/(hi-lo)*16).astype(int),0,15)
        return q[:,0]*16+q[:,1]
    h[:,1:6]=(h[:,1:6]-h[train,1:6].mean(0))/h[train,1:6].std(0).clip(1e-8)
    h=np.column_stack([h,cell(xy[:,:,0]),cell(xy[:,:,-1])])
    for mask in [train,~train]:
        assert len(np.unique(y[mask]))==len(classes)
        print(dict(zip(classes.tolist(),np.bincount(y[mask]).tolist())),len(np.unique(person[mask])))
    assert not set(person[train]) & set(person[~train])
    np.savez_compressed(sys.argv[2],xy=xy.astype('f'),v=v.astype('f'),h=h.astype('f'),dt=dt.astype('f'),y=y,classes=classes,train=train,person=person,source=source,start_row=start_row,end_row=end_row)
