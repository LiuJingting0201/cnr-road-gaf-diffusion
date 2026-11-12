"""Convert timestamped public OSM trackpoint pages to continuous, unlabelled segments."""
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

src, out = map(Path,sys.argv[1:3])
out.mkdir(parents=True,exist_ok=True)
seen=set()
nsegments=npoints=0
def emit(rows, writer):
    global nsegments,npoints
    if len(rows)<120 or len({(lat,lon) for _,lat,lon in rows})<2:
        return
    key=hashlib.sha256(repr(rows).encode()).hexdigest()[:20]
    for ts,lat,lon in rows:
        writer.writerow((ts,key,lon,lat,'unknown'))
    nsegments+=1
    npoints+=len(rows)

with (out/'turin-gps.csv').open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(('unix_seconds','segment_id','longitude','latitude','mode'))
    for page in sorted(src.glob('*.gpx')):
        for segment in ET.parse(page).findall('.//{*}trkseg'):
            rows=[]
            for p in segment.findall('{*}trkpt'):
                stamp=p.findtext('{*}time')
                if not stamp:
                    emit(rows,writer);rows=[];continue
                ts=int(datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp())
                pt=(ts,float(p.attrib['lat']),float(p.attrib['lon']))
                if pt in seen:
                    continue
                seen.add(pt)
                if rows and not 0<ts-rows[-1][0]<=120:
                    emit(rows,writer);rows=[]
                rows.append(pt)
            emit(rows,writer)
print(json.dumps(dict(source='OpenStreetMap public GPS trackpoints, contributors',bbox=[7.665,45.057,7.695,45.080],pages=5,unique_timestamped_points=len(seen),segments_min120=nsegments,retained_points=npoints,labels='unknown; geographically clipped fragments'),indent=2))
