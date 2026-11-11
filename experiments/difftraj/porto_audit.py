"""Audit the UCI Porto source and the paper's minimum-observation rule."""
import collections
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import shutil
import zipfile
import numpy as np

def audit(stream, pack=None):
    tracks=[]
    vehicles=[]
    starts=[]
    counts=collections.Counter()
    eligible_lengths=[]
    taxis=set()
    for row in csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline='')):
        counts['records']+=1
        if row['MISSING_DATA'].lower()=='true':
            counts['missing_flag']+=1
            continue
        # Count coordinate pairs before parsing longer lines.
        poly=row['POLYLINE']
        if poly.count('[')-1<120:
            counts['fewer_than_120']+=1
            continue
        coords=np.array(json.loads(poly))
        if coords.ndim!=2 or coords.shape[1]!=2 or not np.isfinite(coords).all():
            counts['invalid_coordinates']+=1
            continue
        if pack:
            packed=np.rint(coords*1e6).astype(np.int32)
            if np.abs(packed*1e-6-coords).max()>1e-8:
                raise ValueError('Unexpected coordinate precision')
            tracks.append(np.diff(packed,axis=0,prepend=np.zeros((1,2),dtype=np.int32)))
            vehicles.append(int(row['TAXI_ID']))
            starts.append(int(row['TIMESTAMP']))
        eligible_lengths.append(len(coords))
        taxis.add(row['TAXI_ID'])
    if pack:
        np.savez_compressed(pack,delta=np.concatenate(tracks),length=eligible_lengths,taxi=vehicles,start=starts)
    return dict(source='UCI 339, train.csv',counts=counts,eligible=len(eligible_lengths),taxis=len(taxis),median_original_points=float(np.median(eligible_lengths)))
with zipfile.ZipFile(sys.argv[1]) as outer:
    inner=next(n for n in outer.namelist() if n.endswith('train.csv.zip'))
    with tempfile.TemporaryFile() as tmp, outer.open(inner) as src:
        shutil.copyfileobj(src,tmp)
        tmp.seek(0)
        with zipfile.ZipFile(tmp) as archive:
            name=next(n for n in archive.namelist() if n.endswith('train.csv'))
            with archive.open(name) as stream:
                print(json.dumps(audit(stream,sys.argv[2] if len(sys.argv)>2 else None),indent=2))
