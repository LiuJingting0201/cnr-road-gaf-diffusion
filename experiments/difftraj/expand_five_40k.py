"""Extend the verified five-class collection with fresh, episode-disjoint training journeys."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from xml.etree import ElementTree as ET

import numpy as np

from make_balanced_trips import binary, episode, modes, src

PARENT = Path('/workspace/cnr-experiments/difftraj-five-20k-localclock-v1')
START = 5100000
MAX_EPISODES = 3400
TARGET = 40000
PARTS = ('train', 'validation', 'test')
GIB = 1024**3
ROOT = Path.cwd()
STORE = ROOT/'raw_shards'
TMP = ROOT/'work'


def write_json(path, value):
    temp = path.with_name(path.name + '.pending')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def disk_check():
    free = shutil.disk_usage(ROOT).free
    if free < 9*GIB:
        raise RuntimeError(f'Free space {free/GIB:.2f} GiB below 9 GiB launch threshold')
    return round(free/GIB, 2)


def parents():
    if not (PARENT/'COMPLETE').is_file():
        raise ValueError('Missing verified parent')
    membership = {}
    known = set()
    episodes = set()
    for part in PARTS:
        with np.load(PARENT/part/'index.npz') as z:
            ids, labels = z['ids'].tolist(), z['classes'].tolist()
        groups = group_rows(ids)
        assert not episodes.intersection(groups)
        episodes.update(groups)
        assert len(ids)==len(set(ids)) and set(Counter(labels).values())=={20000 if part=='train' else 1000}
        membership[part] = (ids, labels)
        for seed, rows in groups.items():
            folder = PARENT/'raw_shards'/seed
            link = STORE/seed
            if not link.exists():
                link.symlink_to(folder.resolve(),target_is_directory=True)
            with np.load(folder/'trips.npz') as d:
                xy = d['xy']
                for i, key in enumerate(d['ids']):
                    if key in rows:
                        digest = hashlib.sha256(xy[i].tobytes()).digest()
                        if digest in known:
                            raise ValueError('Parent release contains repeated geometries')
                        known.add(digest)
    assert len(known)==sum(len(ids) for ids,_ in membership.values())
    return membership, known


def group_rows(ids):
    by_seed=defaultdict(dict)
    for i,k in enumerate(ids):
        by_seed[k.split('_')[0]][k]=i
    return by_seed


def simulate(args):
    seed, demand = args
    os.chdir(TMP)
    episode(seed, per_class=demand)
    return seed


def select(seed, counts, known):
    dest=STORE/str(seed)
    stage=ROOT/('saving-'+str(seed))
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    work=TMP/str(seed)
    chosen=[]
    with np.load(work/'trips.npz') as d:
        ids,labels,xy,h,raw=(d[k] for k in ('ids','classes','xy','h','raw'))
        ends=np.r_[0,h[:,3].astype(int).cumsum()]
        for i,label in enumerate(labels):
            if counts[str(label)] >= TARGET:
                continue
            key=hashlib.sha256(xy[i].tobytes()).digest()
            if key in known:
                continue
            known.add(key)
            counts[str(label)]+=1
            chosen.append(i)
        ix=np.array(chosen,dtype=int)
        native = np.concatenate([raw[ends[i]:ends[i+1]] for i in ix]) if len(ix) else np.empty((0,4),'f')
        np.savez_compressed(stage/'trips.npz',ids=ids[ix],classes=labels[ix],xy=xy[ix],h=h[ix],raw=native)
        chosen_ids=set(ids[ix])
    tree=ET.parse(work/'arrivals.xml')
    tree_root=tree.getroot()
    for item in list(tree_root):
        if item.get('id') not in chosen_ids:
            tree_root.remove(item)
    assert {v.get('id') for v in tree_root} == chosen_ids
    tree.write(stage/'arrivals.xml')
    write_json(stage/'retained.json',dict(seed=seed,eligible=len(ids),retained=len(ix),counts=dict(Counter(labels[ix].tolist()))))
    stage.rename(dest)
    shutil.rmtree(work)


def register_saved(counts, known):
    seeds=sorted(int(d.name) for d in STORE.iterdir() if d.name.isdigit() and not d.is_symlink())
    for offset,seed in enumerate(seeds):
        assert seed==START+offset, 'non-contiguous completed episode archive'
        with np.load(STORE/str(seed)/'trips.npz') as d:
            counts.update(d['classes'].tolist())
            for x in d['xy']:
                digest=hashlib.sha256(x.tobytes()).digest()
                assert digest not in known
                known.add(digest)
    return seeds


def publish(membership, new_seeds):
    disk_check()
    base_ids, base_labels=membership['train']
    train_ids,train_labels=base_ids.copy(),base_labels.copy()
    for seed in new_seeds:
        with np.load(STORE/str(seed)/'trips.npz') as d:
            train_ids.extend(d['ids'].tolist())
            train_labels.extend(d['classes'].tolist())
    assert len(train_ids)==len(set(train_ids))==5*TARGET and set(Counter(train_labels).values())=={TARGET}
    assert train_ids[:len(base_ids)]==base_ids
    membership['train']=(train_ids,train_labels)
    split_seeds=set()
    receipt={}
    for part in PARTS:
        ids,labels=membership[part]
        groups=group_rows(ids)
        assert not split_seeds.intersection(groups)
        split_seeds.update(groups)
        x=np.empty((len(ids),200,2),'f'); h=np.empty((len(ids),6),'d')
        filled=np.zeros(len(ids),bool)
        for seed, positions in groups.items():
            ep=STORE/seed
            with np.load(ep/'trips.npz') as d:
                xs,hs,ks,ms=[d[k] for k in ('xy','h','ids','classes')]
                arrivals={v.get('id'):(float(v.get('depart')),float(v.get('arrival'))) for v in ET.parse(ep/'arrivals.xml').getroot().iterfind('tripinfo')}
                for j,key in enumerate(ks):
                    if key not in positions:
                        continue
                    i=positions[key]
                    dep,arr=arrivals[key]
                    assert not filled[i] and labels[i]==ms[j] and hs[j,2]+1==hs[j,3]==arr-dep and hs[j,3]>=120
                    x[i]=xs[j]; h[i]=hs[j]; h[i,0]=int(dep//300)%288;filled[i]=True
        assert filled.all() and np.isfinite(x).all() and np.isfinite(h).all()
        if part=='train':
            mu=x.mean((0,1),dtype='d');sd=x.std((0,1),dtype='d');lo=x.min((0,1));hi=x.max((0,1))
            hm=h[:,1:].mean(0);hd=h[:,1:].std(0).clip(1e-7)
            np.savez('normalization.npz',mean=mu,std=sd,lo=lo,hi=hi,header_mean=hm,header_std=hd)
        folder=ROOT/part;folder.mkdir(exist_ok=True)
        q=np.clip(np.floor((x[:,[0,-1]]-lo)/(hi-lo)*16),0,15).astype(int)
        h[:,1:]=(h[:,1:]-hm)/hd
        np.save(folder/'head.npy',np.column_stack((h,q[:,:,0]*16+q[:,:,1])).astype('f'))
        output=np.lib.format.open_memmap(folder/'traj.npy',mode='w+',dtype='f',shape=x.shape)
        for start in range(0,len(x),8192):
            end=start+8192
            output[start:end]=(x[start:end]-mu)/sd
        output.flush(); del output,x,h
        np.savez_compressed(folder/'index.npz',ids=ids,classes=labels)
        write_json(folder/'counts.json',dict(Counter(labels)))
        receipt[part]={'n':len(ids),'episodes':len(groups),'classes':dict(Counter(labels)),
                       'sha256':{f:hash_file(folder/f) for f in ('traj.npy','head.npy','index.npz')}}
    write_json(ROOT/'manifest.json',dict(parent=str(PARENT),new_episodes=len(new_seeds),all_native_seconds=True,
               source_network_sha256=hash_file(src/'most.net.xml'),sumo_binary_sha256=hash_file(binary),
               splits=receipt,normalization_sha256=hash_file(ROOT/'normalization.npz')))
    (ROOT/'COMPLETE').touch()


def hash_file(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run(workers):
    STORE.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    start=time.time()
    counts=Counter({m:20000 for m in modes})
    write_json(ROOT/'progress.json',dict(phase='reading parent ID and geometry fingerprints',free_gib=disk_check(),started_unix=start))
    membership,known=parents()
    seeds=register_saved(counts,known)
    initial_count=sum(counts.values())
    def write_progress(phase, scheduled):
        write_json(ROOT/'progress.json',dict(phase=phase,counts=dict(counts),new_counts={k:v-20000 for k,v in counts.items()},
             target_per_class=TARGET,workers=workers,done_episodes=len(seeds),scheduled=scheduled,elapsed_s=round(time.time()-start,1),
             generated_this_run=sum(counts.values())-initial_count,free_gib=disk_check(),updated_unix=time.time(),
             start_seed=START,maximum_episodes=MAX_EPISODES))
    # Higher bicycle demand offsets its lower rate of eligible completed journeys.
    base={'taxi':100,'motorcycle':100,'passenger':100,'bus':65,'bicycle':210}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while min(counts.values()) < TARGET:
            disk_check()
            deficit={k:TARGET-counts[k] for k in modes}
            demand={k:max(18,round(base[k]*deficit[k]/max(deficit.values()))) for k in modes}
            wave=list(range(START+len(seeds),min(START+len(seeds)+workers,START+MAX_EPISODES)))
            if not wave:
                raise RuntimeError(f'Episode limit reached with {counts}')
            write_progress('simulating',wave)
            for seed in pool.map(simulate,[(s,demand) for s in wave]):
                select(seed,counts,known)
                seeds.append(seed)
                write_progress('selecting',wave)
    write_progress('publishing',[])
    publish(membership,seeds)
    write_progress('complete',[])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if not 1 <= args.workers <= 20:
        raise ValueError('workers outside supported range')
    run(args.workers)
