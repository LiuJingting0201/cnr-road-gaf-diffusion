"""Replay original CNR positive control and verify raw-to-image pixel parity."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from cnr.original_gaf import trajectory_sequence, encode_frozen
from g29_train import metrics

root = Path('g29-data')
source = Path('../data/processed/original-hilbert64grid224')
meta = json.loads((source/'dataset.json').read_text())
with np.load(source/'val.npz') as z:
    images, labels, ids = z['images'], z['labels'], z['ids']
with np.load(root/'raw/cnr-available-val.npz') as z:
    raw = {k:z[k] for k in ('ids','labels','full_txy','full_offset')}
lookup = {v:i for i,v in enumerate(raw['ids'])}
chosen = np.concatenate([np.flatnonzero(labels==c)[:3] for c in range(6)])
sequences=[]
for j in chosen:
    i=lookup[ids[j]]
    a,b=raw['full_offset'][i:i+2]
    seq,_=trajectory_sequence(raw['full_txy'][a:b,1:],np.array(meta['xy_min']),np.array(meta['xy_max']),eps=meta['hilbert_eps'])
    sequences.append(seq)
encoded,_=encode_frozen(np.array(sequences),np.array(meta['channel_min']),np.array(meta['channel_max']))
assert np.array_equal(encoded,images[chosen]), 'Original CNR RGB pixel parity failed'
torch.set_num_threads(2)
results={}
for seed in (42,43,44):
    folder=Path('../runs/gaf-resnet-e1')/f'grid64-imagenet-seed{seed}'
    net=resnet18(weights=None);net.fc=torch.nn.Linear(512,6)
    net.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True))
    net.cuda().eval();probs=[]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            x=torch.from_numpy(images[start:start+64]).cuda().permute(0,3,1,2).float()/255
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            probs.append(net(x).softmax(1).cpu().numpy())
    score=metrics(labels,np.concatenate(probs))
    saved=json.loads((folder/'complete.json').read_text())['final_validation']
    delta={k:score[k]-saved[k] for k in ('accuracy','macro_f1')}
    assert max(abs(v) for v in delta.values())<1e-6
    results[str(seed)]={'replayed':score,'difference_from_saved':delta}
result={'scope':'original full-observation CNR validation; replayed existing fixed E1 weights, no fitting or test access',
        'pixel_parity_ids':len(chosen),'pixel_exact':True,'validation_ids':len(ids),'seeds':results,
        'mean_macro_f1':float(np.mean([v['replayed']['macro_f1'] for v in results.values()])),
        'mean_accuracy':float(np.mean([v['replayed']['accuracy'] for v in results.values()]))}
(root/'positive.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ('pixel_exact','mean_macro_f1','mean_accuracy')}))
