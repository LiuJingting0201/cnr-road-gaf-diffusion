"""Classify real full trajectories and real short windows with the same frozen encoder.

The window check is a domain-transfer diagnostic, not generation or guidance.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cnr.original_gaf import trajectory_sequence,encode_frozen
from cnr.representations import ImageClassifier
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
torch.set_num_threads(4);cfg=json.loads((root/'config.json').read_text());meta=cfg['dataset']
model=ImageClassifier(3).cuda().eval();model.load_state_dict(torch.load(root/'final.pt',map_location='cuda',weights_only=True))
windows=np.load('data/processed/most64/val.npz');h=[];labels=[];failures=0
for xy,label in zip(windows['xy'],windows['labels']):
 try:seq,_=trajectory_sequence(xy,np.array(meta['xy_min']),np.array(meta['xy_max']))
 except ValueError:failures+=1;continue
 h.append(seq);labels.append(label)
images,clipped=encode_frozen(h,np.array(meta['channel_min']),np.array(meta['channel_max']))
with torch.no_grad():
 pred=torch.cat([model(torch.from_numpy(images[j:j+32]).cuda().permute(0,3,1,2).float()/127.5-1).argmax(-1).cpu() for j in range(0,len(images),32)]).numpy()
labels=np.array(labels);cm=np.bincount(labels*6+pred,minlength=36).reshape(6,6);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
summary={'full_vehicle_validation':json.loads((root/'complete.json').read_text())['final_validation'],
 'short_window_transfer':{'samples':len(labels),'macro_f1':float(f1.mean()),'accuracy':float(np.mean(pred==labels)),'confusion':cm.tolist(),'per_class_f1':f1.tolist(),'failed_windows':failures,'channel_clipped_values':clipped},
 'scope':'Same frozen original Hilbert/RGB encoding and same CNN; full-vehicle training vs 64-point real-window diagnostic. Not original-group CNN reproduction; not evidence about generated trajectories. Test untouched.'}
out=Path('docs/results/original-gaf-alignment')/root.name;out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
