import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from cnr.roads import RoadNetwork,CLASS_TO_VCLASSES,allows
from cnr.data import CLASSES

p=argparse.ArgumentParser()
p.add_argument('--map',default='data/maps/most.net.xml')
p.add_argument('--output',default='data/maps/road_fields_4m.npz')
p.add_argument('--resolution',type=float,default=4.)
a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
roads=RoadNetwork(a.map)
x0,y0,x1,y1=map(float,roads.location['convBoundary'].split(','))
origin=np.array([x0-256,y0-256]);end=np.array([x1+256,y1+256])
w,h=np.ceil((end-origin)/a.resolution).astype(int)+1
fields=[];counts={}
for name in CLASSES:
    mask=Image.new('L',(int(w),int(h)),0);draw=ImageDraw.Draw(mask);count=0
    for lane,shape in zip(roads.lanes,roads.shapes):
        if any(allows(lane,c) for c in CLASS_TO_VCLASSES[name]):
            coords=(np.asarray(shape.coords)-origin)/a.resolution
            draw.line([tuple(p) for p in coords],fill=255,width=1);count+=1
    if not count:raise ValueError(name)
    fields.append((distance_transform_edt(np.array(mask)==0)*a.resolution).astype(np.float32))
    counts[name]=count
out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out,distance=np.stack(fields),origin=origin,resolution=np.array(a.resolution))
record={'map_sha256':hashlib.sha256(Path(a.map).read_bytes()).hexdigest(),'resolution_m':a.resolution,
        'origin':origin.tolist(),'shape':[6,int(h),int(w)],'lane_counts':counts,
        'scope':'raster distance to permitted centerlines; approximate; not a drivable-area or topology guarantee'}
out.with_suffix('.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
