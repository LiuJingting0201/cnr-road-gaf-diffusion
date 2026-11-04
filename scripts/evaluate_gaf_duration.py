"""Compare full-vehicle transfer and duration-trained models, with common-ID views."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18

torch.set_num_threads(4)
out=Path('docs/results/gaf-duration-e2')
out.mkdir(parents=True,exist_ok=True)
data={n:np.load(f'data/processed/gaf-duration-e2/length{n}/val.npz') for n in [64,224,512]}
common=set.intersection(*[set(d['ids']) for d in data.values()])
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

def metrics(y,p):
    cm=np.bincount(y*6+p.argmax(1),minlength=36).reshape(6,6)
    f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    return {'vehicles':len(y),'class_counts':np.bincount(y,minlength=6).tolist(),'macro_f1':float(f1.mean()),'accuracy':float(np.mean(p.argmax(1)==y)),
        'per_class_f1':f1.tolist(),'confusion':cm.tolist()}

records=[]
for length,d in data.items():
    mask=np.array([i in common for i in d['ids']])
    for source in ['full_vehicle','duration_matched']:
        for seed in [42,43,44]:
            run=Path(f'runs/gaf-resnet-e1/grid64-imagenet-seed{seed}') if source=='full_vehicle' else Path(f'runs/gaf-duration-e2/length{length}-seed{seed}')
            assert json.loads((run/'complete.json').read_text())['complete']
            model=resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,6)
            model.load_state_dict(torch.load(run/'final.pt',weights_only=True,map_location='cpu'));model=model.cuda().eval()
            prob=[]
            with torch.no_grad():
                for batch in range(0,len(d['ids']),64):
                    x=torch.from_numpy(d['images'][batch:batch+64]).cuda().permute(0,3,1,2).float()/255
                    prob.append(model((x-mean)/std).softmax(-1).cpu().numpy())
            prob=np.concatenate(prob)
            records.append({'length':length,'source':source,'seed':seed,'all_eligible':metrics(d['labels'],prob),'common_ids':metrics(d['labels'][mask],prob[mask])})
            del model
            if source=='duration_matched':
                dest=out/run.name;dest.mkdir(exist_ok=True)
                for name in ['complete.json','best_validation.json','config.json','metrics.jsonl']:
                    (dest/name).write_text((run/name).read_text())
diagnostic=json.loads(Path('data/processed/gaf-duration-e2/diagnostic.json').read_text())
(out/'results.json').write_text(json.dumps({'common_validation_vehicles':len(common),'records':records,'representation_diagnostic':diagnostic},indent=2))
lines=['# E2 duration-matched inherited Hilbert GAF diagnostic','',
       '64x64 Hilbert grid, unchanged squeeze/mirror to 224 RGB. One centered window of longest continuous segment per ID; all three lengths share a common center, up to half a sample. No test access. New ResNet18 models, fixed epoch 50. Not generated-data evaluation.','',
       f'Common validation cohort: {len(common)} vehicles. Common-cohort evaluation controls validation identity, but training eligibility still differs with duration; duration and training population effects cannot be fully separated.','',
       '| Points | Model training domain | All eligible accuracy % | All eligible macro-F1 % | Common-ID macro-F1 % |','|---|---|---:|---:|---:|']
for n in data:
    for source in ['full_vehicle','duration_matched']:
        rows=[r for r in records if r['length']==n and r['source']==source]
        def fmt(pop,key):
            x=[r[pop][key]*100 for r in rows]
            return f'{np.mean(x):.2f} +/- {np.std(x):.2f}'
        lines.append(f'| {n} | {source} | {fmt("all_eligible","accuracy")} | {fmt("all_eligible","macro_f1")} | {fmt("common_ids","macro_f1")} |')
lines+=['','Next: select a generation horizon considering road reconstruction coverage, class coverage, diversity and feedback information. Do not pick by CNN accuracy alone. Keep the existing diffusion baseline; evaluate GAF selection on identical candidate pools against random and trajectory-classifier selection.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
