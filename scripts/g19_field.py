"""Train-only calibration and monotone coordinate for road-dependent speed."""
import csv
import io
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import shapely
from g16_data import CORPUS,NAMES,csv_tracks
from cnr.roads import RoadNetwork,CLASS_TO_VCLASSES,allows

ROOT=Path("g19-data")
DATA=Path("../g16-job/g16-data")
NET=Path("data/maps/most.net.xml")
ANGLES=[3.,10.,25.,50.]
LIMITS=[3.,7.,11.,16.,23.]
BINS=(len(ANGLES)+1)*(len(LIMITS)+1)


def spatial_xy(arc,xy,positions):
    return np.stack([np.interp(positions,arc,v) for v in xy.T],axis=-1)


def cells(arc,xy,pos,limit):
    mid=spatial_xy(arc,xy,pos)
    before=mid-spatial_xy(arc,xy,np.maximum(pos-10,0))
    after=spatial_xy(arc,xy,np.minimum(pos+10,arc[-1]))-mid
    denom=np.linalg.norm(before,axis=-1)*np.linalg.norm(after,axis=-1)
    angle=np.rad2deg(np.arccos(np.clip(np.sum(before*after,axis=-1)/np.maximum(denom,1e-10),-1,1)))
    angle[denom<1]=0
    return np.searchsorted(ANGLES,angle)*(len(LIMITS)+1)+np.searchsorted(LIMITS,limit)


def native_limits():
    xml=ET.parse(NET).getroot()
    lookup={}
    for edge in xml.findall("edge"):
        lanes=edge.findall("lane")
        if lanes:
            lookup["edge",edge.get("id")]=np.median([float(l.get("speed")) for l in lanes])
        for l in lanes:
            lookup["lane",l.get("id")]=float(l.get("speed"))
    by_id=defaultdict(list)
    with ZipFile(CORPUS/"road-annotations.zip") as z, z.open("train.csv") as raw:
        for r in csv.DictReader(io.TextIOWrapper(raw)):
            by_id[r["id"]].append((int(r["start_index"]),int(r["stop_index"]),lookup[r["reference_kind"],r["reference_id"]]))
    return by_id


def prepare():
    native=native_limits()
    counts=np.zeros((8,BINS))
    logsum=np.zeros_like(counts)
    entry=[]
    with np.load(DATA/"expanded.npz") as d:
        starts,lab,target=d["start"],d["label"],d["target"]
    for ident,c,x in csv_tracks("train"):
        xy=x.astype(float)
        inc=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        arc=np.r_[0.,np.cumsum(inc)]
        if arc[-1]<64:
            continue
        i=len(entry)
        assert c==lab[i] and np.allclose(starts[i],xy[0]) and np.allclose(target[i],np.log1p(inc),atol=1e-6)
        limit=np.full(512,np.nan)
        for a,b,s in native[ident]:
            limit[a:b]=s
        assert np.isfinite(limit).all()
        keep=np.r_[True,np.diff(arc)>0]
        dist=arc[keep]
        pos=np.arange(10,dist[-1]-10,5.)
        times=np.arange(512)[keep]*.25
        t1=np.interp(pos+2.5,dist,times)
        t0=np.interp(pos-2.5,dist,times)
        idx=np.searchsorted(arc,pos).clip(0,511)
        bins=cells(dist,xy[keep],pos,limit[idx])
        counts[c]+=np.bincount(bins,minlength=BINS)
        logsum[c]+=np.bincount(bins,weights=np.log(np.clip(5/np.maximum(t1-t0,1e-6),.3,35)),minlength=BINS)
        entry.append(cells(dist,xy[keep],(arc[1:]+arc[:-1])/2,limit[1:]).astype(np.uint8))
    assert len(entry)==len(lab)
    base=logsum.sum(1)/counts.sum(1)
    expected=(logsum+64*base[:,None])/(counts+64)
    factor=np.exp(np.clip(base[:,None]-expected,-np.log(2),np.log(2)))
    np.save(ROOT/"factor.npy",factor)
    adjusted=np.log1p(np.expm1(target)*factor[lab[:,None],np.stack(entry)])
    np.save(ROOT/"road-target.npy",adjusted.astype(np.float32))
    (ROOT/"field.json").write_text(json.dumps(dict(tracks=len(lab),by_class=np.bincount(lab,minlength=8).tolist(),
        calibration_positions=int(counts.sum()),bin_counts=counts.tolist(),factor=factor.tolist()),indent=2))


class MapField:
    def __init__(self,factor,network=NET):
        self.factor=factor
        net=RoadNetwork(network)
        self.by_class={}
        for c,name in enumerate(NAMES):
            keep=[i for i,lane in enumerate(net.lanes) if any(allows(lane,v) for v in CLASS_TO_VCLASSES[name])]
            self.by_class[c]=(shapely.STRtree(net.shapes[keep]),np.array([float(net.lanes[i]["speed"]) for i in keep]))

    def coordinate(self,path,length,c):
        # Paths from the planner are already uniformly sampled in arc distance.
        dist=np.linspace(0,length,len(path))
        grid=np.linspace(0,length,int(np.ceil(length/2))+1)
        mid=(grid[1:]+grid[:-1])/2
        points=spatial_xy(dist,path,mid)
        tree,limits=self.by_class[int(c)]
        speed=limits[tree.nearest(shapely.points(points))]
        bins=cells(dist,path,mid,speed)
        transformed=np.r_[0.,np.cumsum(np.diff(grid)*self.factor[c,bins])]
        return grid,transformed


if __name__=="__main__":
    ROOT.mkdir(exist_ok=True)
    prepare()
