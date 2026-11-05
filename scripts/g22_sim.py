"""Map-first mixed SUMO episodes with explicit departures and verified arrivals."""
from collections import Counter,defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
import numpy as np
import shapely
from shapely.ops import substring
from cnr.route_paths import RoutePaths
from g16_data import NAMES
from cnr.roads import allows

ROOT=Path("g22-data")
NET=Path("data/maps/most.net.xml")
VC=["pedestrian","bus","motorcycle","passenger","bicycle","taxi","truck","delivery"]
SPLITS={"train":range(272001,272017),"monitor":range(273001,273004),"evaluation":range(274001,274005)}
COUNT=48
NODES=2049


def plan(m,c,rng,normal):
    label=NAMES[c]
    permitted=m.permitted[label]
    for attempt in range(800):
        u=int(rng.choice(normal[c]))
        start=u
        edge=m.road.lanes[u]["edge"]
        if c==0:
            length=m.lengths[u]
            left,right=sorted(rng.uniform(.02,.98,2)*length)
            if right-left<60:
                continue
            pts=substring(m.road.shapes[u],left,right)
            pieces=[pts]
            s,e=left,right
            edges=[edge]
        else:
            target=rng.uniform(750,4500)
            s=rng.uniform(0.16,0.65)*m.lengths[u]
            if s<16:
                continue
            edges=[edge]
            seen={u}
            pieces=[substring(m.road.shapes[u],s,m.lengths[u])]
            distance=pieces[0].length
            for _ in range(250):
                choices=[i for i in sorted(m.successors[u]) if i not in seen and i in permitted
                         and allows(m.road.lanes[i],VC[c])
                         and not (m.road.lanes[i]["function"]=="normal" and m.road.lanes[i]["edge"] in edges)
                         and np.linalg.norm(np.subtract(m.road.shapes[u].coords[-1],m.road.shapes[i].coords[0]))<=2.]
                if not choices:
                    break
                u=int(rng.choice(choices))
                seen.add(u)
                ln=m.road.lanes[u]
                if ln["function"]=="normal":
                    edges.append(ln["edge"])
                if ln["function"]=="normal" and distance+m.lengths[u]*.8>=target and m.lengths[u]>30:
                    e=float(np.clip(target-distance,15,m.lengths[u]*.85))
                    pieces.append(substring(m.road.shapes[u],0,e))
                    break
                pieces.append(m.road.shapes[u])
                distance+=m.lengths[u]
            else:
                continue
            if not choices or not (ln["function"]=="normal" and len(edges)>1 and distance+m.lengths[u]*.8>=target and m.lengths[u]>30):
                continue
        line=shapely.LineString([p for seg in pieces for p in seg.coords])
        coords=shapely.get_coordinates(shapely.line_interpolate_point(line,np.linspace(0,line.length,NODES)))
        startpos=s/m.lengths[start]*float(m.road.lanes[start]["length"])
        endpos=e/m.lengths[u]*float(m.road.lanes[u]["length"])
        return coords,line.length,edges,m.road.lanes[start]["index"],m.road.lanes[u]["index"],startpos,endpos
    raise ValueError(f"No connected route for {label}")


def episode(task):
    split,seed,n=task
    out=ROOT/split/str(seed)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        return
    rng=np.random.default_rng(seed)
    m=RoutePaths(str(NET))
    normal=[[i for i in sorted(m.permitted[label]) if m.road.lanes[i]["function"]=="normal"
             and allows(m.road.lanes[i],VC[c]) and m.lengths[i]>(110 if c==0 else 40)] for c,label in enumerate(NAMES)]
    root=ET.Element("routes")
    for c in range(1,8): ET.SubElement(root,"vType",id=NAMES[c],vClass=VC[c])
    jobs=[]
    plans={}
    for c,label in enumerate(NAMES):
        for slot in range(n):
            ident=f"{seed}_{c}_{slot}"
            path,L,edges,first,last,s,e=plan(m,c,rng,normal)
            depart=rng.uniform(0,90)
            if c==0:
                obj=ET.Element("person",id=ident,depart=f"{depart:.2f}",departPos=f"{s:.3f}")
                ET.SubElement(obj,"walk",edges=edges[0],arrivalPos=f"{e:.3f}")
            else:
                obj=ET.Element("vehicle",id=ident,type=label,depart=f"{depart:.2f}",departLane=first,
                               arrivalLane=last,departPos=f"{s:.3f}",arrivalPos=f"{e:.3f}")
                ET.SubElement(obj,"route",edges=" ".join(edges))
            jobs.append((depart,obj));plans[ident]=(c,path.astype("float32"),L)
    for _,obj in sorted(jobs,key=lambda j:j[0]): root.append(obj)
    demand=out/"demand.xml";ET.ElementTree(root).write(demand)
    with TemporaryDirectory(dir=out) as scratch:
        fcd=Path(scratch)/"fcd.xml"
        cmd=["g10-env/sumo/bin/sumo","-n",str(NET),"-r",str(demand),"--seed",str(seed),
             "--step-length",".25","--end","2400","--time-to-teleport","-1","--fcd-output",str(fcd),
             "--device.fcd.period","1","--fcd-output.attributes","x,y","--tripinfo-output",str(out/"arrival.xml"),
             "--tripinfo-output.write-unfinished","true","--no-step-log","true"]
        with (out/"sumo.log").open("w") as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        tracks=defaultdict(list)
        for _,step in ET.iterparse(fcd,events=["end"]):
            if step.tag=="timestep":
                t=float(step.get("time"))
                for item in step:
                    tracks[item.get("id")].append((t,float(item.get("x")),float(item.get("y"))))
                step.clear()
    info={}
    for item in ET.parse(out/"arrival.xml").getroot():
        if item.tag=="tripinfo":
            info[item.get("id")]=(float(item.get("depart")),float(item.get("arrival")))
        elif item.tag=="personinfo":
            walk=item.find("walk")
            if walk is not None:
                info[item.get("id")]=(float(item.get("depart")),float(walk.get("arrival","-1")))
    keep=[];reject=Counter()
    for ident,(c,path,L) in plans.items():
        dep,arr=info.get(ident,(-1,-1))
        raw=np.array(tracks[ident])
        if dep<0 or arr<=dep: reject["not_arrived"]+=1;continue
        if len(raw)<4 or not np.allclose(np.diff(raw[:,0]),1.): reject["short_or_gap"]+=1;continue
        if max(np.linalg.norm(raw[0,1:]-path[0]),np.linalg.norm(raw[-1,1:]-path[-1]))>25:
            reject["endpoint_mismatch_25m"]+=1;continue
        if np.max(np.linalg.norm(np.diff(raw[:,1:],axis=0),axis=1))>(5 if c==0 else 60):
            reject["position_jump"]+=1;continue
        times=np.r_[dep,raw[:,0],arr]
        xy=np.vstack([path[0],raw[:,1:],path[-1]])
        if np.any(np.diff(times)<0):reject["time_bounds"]+=1;continue
        dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        u=np.interp(np.linspace(dep,arr,257),times,dist)/dist[-1]
        keep.append((ident,c,path,L,arr-dep,u,raw[:,1:].astype("float32")))
    ids,c,paths,length,duration,progress,xy=zip(*keep)
    np.savez_compressed(out/"trips.npz",ids=ids,labels=c,paths=paths,length=length,duration=duration,
                        progress=np.array(progress,np.float32),offset=np.r_[0,np.cumsum([len(x) for x in xy])],xy=np.concatenate(xy))
    (out/"complete.json").write_text(json.dumps(dict(seed=seed,attempted=8*n,retained=len(keep),
                                reasons=reject,classes=dict(Counter(NAMES[y] for y in c))),indent=2))


if __name__=="__main__":
    if len(sys.argv)>1:
        episode(("probe",271999,3))
    else:
        with ProcessPoolExecutor(max_workers=8) as pool:
            list(pool.map(episode,[(s,i,COUNT) for s,seeds in SPLITS.items() for i in seeds]))
