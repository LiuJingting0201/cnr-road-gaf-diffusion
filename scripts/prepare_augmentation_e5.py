"""Build class-count-matched augmentation groups from TRAIN conditions only."""
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from cnr.augmentation import time_warp,validate_training_parents
from cnr.candidate_selection import select_candidates
from cnr.data import CLASSES
from cnr.original_gaf import trajectory_sequence,encode_frozen
from cnr.representations import RepresentationClassifier,training_stats
from cnr.topology import LaneConsistency

def init_graph():
    global matcher
    matcher=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64)

def graph_one(item):
    x,label=item
    return matcher.match(x,CLASSES[int(label)])

def main():
    torch.set_num_threads(4)
    root=Path('data/processed/augmentation-e5');root.mkdir(parents=True,exist_ok=False)
    source=Path('data/processed/gaf-duration-e2/length512')
    real=np.load(source/'train.npz');val=np.load(source/'val.npz');meta=json.loads((source/'dataset.json').read_text())
    s=np.load('runs/augmentation-e5/pool/candidates.npz');xy=s['xy'];labels=s['labels']
    validate_training_parents(s['parent_ids'],real['ids'],val['ids'])
    assert np.array_equal(s['parent_ids'],real['ids'][s['source_index']])
    assert np.array_equal(labels,real['labels'][s['source_index']])
    groups=[np.flatnonzero(s['condition_id']==i) for i in np.unique(s['condition_id'])]
    assert len(groups)==425 and all(len(g)==8 for g in groups)
    parent=np.array([s['source_index'][g[0]] for g in groups])
    extra_labels=real['labels'][parent]
    seq=[];enc_valid=[];coordinate_clipped=0
    for x in xy:
        try:
            h,n=trajectory_sequence(x,np.array(meta['xy_min']),np.array(meta['xy_max']),eps=meta['hilbert_eps'])
            enc_valid.append(True);coordinate_clipped+=n
        except ValueError:
            h=np.zeros(224);enc_valid.append(False)
        seq.append(h)
    images,channel_clipped=encode_frozen(seq,np.array(meta['channel_min']),np.array(meta['channel_max']))
    net=resnet18(weights=None);net.fc=torch.nn.Linear(net.fc.in_features,6)
    gafpath=Path('runs/gaf-duration-e2/length512-seed42/final.pt')
    net.load_state_dict(torch.load(gafpath,weights_only=True,map_location='cpu'));net=net.cuda().eval()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.no_grad():
        probs=[]
        for i in range(0,len(xy),32):
            x=torch.from_numpy(images[i:i+32]).cuda().permute(0,3,1,2).float()/255
            probs.append(net((x-mean)/std).softmax(-1).cpu().numpy())
    gaf=np.concatenate(probs);del net,images
    tcnroot=Path('runs/long-selection-e4/tcn-seed42');tcfg=json.loads((tcnroot/'config.json').read_text())
    net=RepresentationClassifier('xy_motion_tcn',tcfg['stats']).cuda().eval()
    net.load_state_dict(torch.load(tcnroot/'latest.pt',map_location='cuda',weights_only=True)['model'])
    with torch.no_grad():
        tcn=np.concatenate([net(torch.from_numpy(xy[i:i+64]).cuda()).softmax(-1).cpu().numpy() for i in range(0,len(xy),64)])
    del net
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init_graph) as pool:
        graphs=[]
        for i,g in enumerate(pool.map(graph_one,zip(xy,labels),chunksize=8)):
            graphs.append(g)
            if i%256==0:print(json.dumps({'audited':i,'total':len(xy)}),flush=True)
    valid=np.array([g['graph_consistent'] and not g['candidate_truncation_warning'] for g in graphs]) & np.array(enc_valid) & (s['progress'][:,-1]<s['route_length'])
    rng=np.random.default_rng(20261105);priority=rng.random(len(xy))
    extras={'repeat':real['xy'][parent], 'timewarp':np.stack([time_warp(x,float(rng.uniform(-.02,.02))) for x in real['xy'][parent]])}
    selections={}
    for name,score in [('synthetic_random',None),('synthetic_gaf',gaf[np.arange(len(xy)),labels]),('synthetic_tcn',tcn[np.arange(len(xy)),labels])]:
        ix,failed=select_candidates(groups,valid,score,priority)
        assert np.array_equal(labels[ix],extra_labels)
        extras[name]=xy[ix];selections[name]={'indices':ix.tolist(),'fallback_conditions':int(failed.sum())}
    for name in ['real_only',*extras]:
        d=root/name;d.mkdir()
        x=real['xy'] if name=='real_only' else np.concatenate([real['xy'],extras[name]])
        y=real['labels'] if name=='real_only' else np.r_[real['labels'],extra_labels]
        np.savez_compressed(d/'train.npz',xy=x.astype(np.float32),labels=y)
    # The real validation set is shared and never enters any training pool.
    np.savez_compressed(root/'val.npz',xy=val['xy'],labels=val['labels'],ids=val['ids'])
    report={'source':str(source),'source_hashes':{split:hashlib.sha256((source/f'{split}.npz').read_bytes()).hexdigest() for split in ['train','val']},
        'candidate_sha256':hashlib.sha256(Path('runs/augmentation-e5/pool/candidates.npz').read_bytes()).hexdigest(),
        'selectors':{'gaf_sha256':hashlib.sha256(gafpath.read_bytes()).hexdigest(),'tcn_sha256':hashlib.sha256((tcnroot/'latest.pt').read_bytes()).hexdigest()},
        'groups':['real_only',*extras],'real_train':len(real['xy']),'real_val':len(val['xy']),'extra_per_group':425,
        'real_class_counts':np.bincount(real['labels'],minlength=6).tolist(),'validation_class_counts':np.bincount(val['labels'],minlength=6).tolist(),
        'extra_class_counts':np.bincount(extra_labels,minlength=6).tolist(),'stats':training_stats(real['xy']),
        'valid_candidate_fraction':float(valid.mean()),'encoding_failures':int((~np.array(enc_valid)).sum()),'coordinate_clipped_points':coordinate_clipped,'channel_clipped_values':channel_clipped,
        'selections':selections,'scope':'Training-only known-path motion augmentation, five synthetic classes, six real classes. Fixed 425 extras and same extra labels/conditions. Real-only training normalization and class weights. No test access; not low-label or new-route generation.',
        'hashes':{name:hashlib.sha256((root/name/'train.npz').read_bytes()).hexdigest() for name in ['real_only',*extras]}}
    (root/'dataset.json').write_text(json.dumps(report,indent=2))
    Path('runs/augmentation-e5/selection_graphs.json').write_text(json.dumps(graphs))
    print(json.dumps({k:v for k,v in report.items() if k not in ['stats','selections']},indent=2),flush=True)

if __name__=='__main__':main()
